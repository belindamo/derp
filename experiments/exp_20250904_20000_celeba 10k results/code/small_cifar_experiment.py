"""
Small-scale DERP-VAE Experiment on CIFAR-10 (1000 samples)
Quick validation test with the existing derp_vae.py implementation
"""
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import json
import logging
from pathlib import Path
import time
import sys

# Add current directory to Python path
sys.path.append('.')

# Import our modules
from derp_vae import DERP_VAE, EnhancedStandardVAE as StandardVAE, compute_enhanced_metrics, enhanced_statistical_test

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_small_cifar_dataset(n_samples: int = 1000, batch_size: int = 64):
    """Get a small subset of CIFAR-10 for quick testing"""
    logger.info(f"Loading CIFAR-10 subset with {n_samples} samples")
    
    # Transform to [0,1] range for BCE loss
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1] automatically
    ])
    
    # Load CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root='../../../data/vision',
        train=True,
        download=False,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='../../../data/vision',
        train=False,
        download=False,
        transform=transform
    )
    
    # Create small subsets
    train_indices = torch.randperm(len(train_dataset))[:n_samples]
    test_indices = torch.randperm(len(test_dataset))[:n_samples//5]  # 20% for test
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    return train_loader, test_loader


def run_small_experiment(model, train_loader, test_loader, epochs: int = 5, 
                        beta: float = 1.0, model_name: str = "Model"):
    """Run a small-scale training experiment"""
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_losses = []
    test_metrics = []
    
    start_time = time.time()
    
    logger.info(f"Starting training for {model_name}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Flatten images
            batch_data = batch_data.view(batch_data.size(0), -1)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model.compute_loss(batch_data, batch_labels, beta=beta)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        
        # Evaluation
        if epoch == epochs - 1:  # Only evaluate at the end
            model.eval()
            test_loss = 0.0
            test_batches = 0
            
            all_mus = []
            all_logvars = []
            all_class_logits = []
            all_true_labels = []
            
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data = batch_data.view(batch_data.size(0), -1)
                    
                    # Compute test loss
                    loss_dict = model.compute_loss(batch_data, batch_labels, beta=beta)
                    test_loss += loss_dict['total_loss'].item()
                    test_batches += 1
                    
                    # Collect samples
                    _, class_logits, mu, logvar, z = model.forward(batch_data)
                    all_mus.append(mu)
                    all_logvars.append(logvar)
                    all_class_logits.append(class_logits)
                    all_true_labels.append(batch_labels)
            
            # Concatenate
            all_mus = torch.cat(all_mus, dim=0)
            all_logvars = torch.cat(all_logvars, dim=0)
            all_class_logits = torch.cat(all_class_logits, dim=0)
            all_true_labels = torch.cat(all_true_labels, dim=0)
            
            # Compute metrics
            metrics = compute_enhanced_metrics(all_mus, all_logvars, all_class_logits, all_true_labels)
            
            test_loss /= test_batches
            
            test_metric = {
                'epoch': epoch + 1,
                'test_loss': test_loss,
                **metrics
            }
            
            test_metrics.append(test_metric)
            
            logger.info(
                f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, "
                f"Test Loss={test_loss:.4f}, "
                f"KL Div={metrics['kl_divergence']:.4f}, "
                f"Accuracy={metrics['classification_accuracy']:.4f}"
            )
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'final_metrics': test_metrics[-1] if test_metrics else {}
    }


def main():
    """Run small-scale DERP-VAE experiment on CIFAR-10"""
    logger.info("Starting Small-Scale DERP-VAE Experiment on CIFAR-10")
    
    # Set seeds
    set_seeds(42)
    
    # Get small CIFAR dataset
    train_loader, test_loader = get_small_cifar_dataset(n_samples=1000, batch_size=64)
    
    # Model configurations for CIFAR-10
    base_config = {
        'input_dim': 3072,  # 32*32*3 for CIFAR-10
        'hidden_dim': 256,
        'latent_dim': 4,
        'n_classes': 10
    }
    
    # Define experiments
    experiments = [
        ('Standard_VAE', StandardVAE(**base_config), 1.0),
        ('Beta_VAE_0.5', StandardVAE(**base_config), 0.5),
        ('DERP_VAE_3probes', DERP_VAE(**base_config, n_probes=3, enforcement_weight=1.0, device='cpu'), 1.0),
    ]
    
    results = {}
    
    # Run experiments
    for name, model, beta in experiments:
        logger.info(f"\\n{'='*60}")
        logger.info(f"Running: {name}")
        logger.info(f"Beta: {beta}")
        logger.info(f"{'='*60}")
        
        result = run_small_experiment(
            model, train_loader, test_loader, 
            epochs=5, beta=beta, model_name=name
        )
        result['model_name'] = name
        result['beta'] = beta
        results[name] = result
        
        # Print summary
        final = result['final_metrics']
        logger.info(f"\\n{name} Final Results:")
        logger.info(f"  KL Divergence: {final.get('kl_divergence', 0):.4f}")
        logger.info(f"  Classification Accuracy: {final.get('classification_accuracy', 0):.4f}")
        logger.info(f"  Activation Rate: {final.get('activation_rate', 0):.4f}")
        logger.info(f"  Training Time: {result['training_time']:.2f}s")
    
    # Compare results
    logger.info("\\n" + "="*70)
    logger.info("COMPARISON RESULTS")
    logger.info("="*70)
    
    if 'Standard_VAE' in results and 'DERP_VAE_3probes' in results:
        baseline = results['Standard_VAE']['final_metrics']
        derp = results['DERP_VAE_3probes']['final_metrics']
        
        kl_improvement = (baseline['kl_divergence'] - derp['kl_divergence']) / baseline['kl_divergence'] * 100
        acc_diff = (derp['classification_accuracy'] - baseline['classification_accuracy']) * 100
        
        logger.info(f"\\nKL Divergence:")
        logger.info(f"  Standard VAE: {baseline['kl_divergence']:.4f}")
        logger.info(f"  DERP-VAE: {derp['kl_divergence']:.4f}")
        logger.info(f"  Improvement: {kl_improvement:.1f}%")
        
        logger.info(f"\\nClassification Accuracy:")
        logger.info(f"  Standard VAE: {baseline['classification_accuracy']:.4f}")
        logger.info(f"  DERP-VAE: {derp['classification_accuracy']:.4f}")
        logger.info(f"  Difference: {acc_diff:+.1f}%")
    
    # Save results
    results_path = Path("../results")
    results_path.mkdir(exist_ok=True)
    
    with open(results_path / "small_cifar_experiment_results.json", 'w') as f:
        json_results = {
            name: {
                'model_name': r['model_name'],
                'beta': r['beta'],
                'final_metrics': r['final_metrics'],
                'training_time': r['training_time']
            }
            for name, r in results.items()
        }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\\nResults saved to {results_path / 'small_cifar_experiment_results.json'}")
    
    return results


if __name__ == "__main__":
    results = main()