"""
DERP-VAE Experiment on CelebA Dataset
Testing H1 & H2 with larger, more complex image data

CelebA contains 202,599 face images with 40 binary attributes.
We use the 'Smiling' attribute for classification task.
Images are resized to 64x64 for computational efficiency.
"""
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import logging
from pathlib import Path
import time
import sys
import matplotlib.pyplot as plt

# Add current directory to Python path
sys.path.append('.')

# Import our modules
from derp_vae import DERP_VAE, EnhancedStandardVAE as StandardVAE, compute_enhanced_metrics, enhanced_statistical_test
from data_loader import get_celeba_dataloaders

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_ks_distance_for_latents(latent_samples: torch.Tensor, n_random_projections: int = 10) -> float:
    """
    Compute average KS-distance for latent representations using random projections.
    This is a standalone function that can be applied to any VAE model's latents.
    
    Args:
        latent_samples: [batch_size, latent_dim] tensor of latent representations
        n_random_projections: number of random projections to average over
    
    Returns:
        average KS-distance across all projections
    """
    batch_size, latent_dim = latent_samples.shape
    device = latent_samples.device
    
    total_ks_distance = 0.0
    
    for _ in range(n_random_projections):
        # Generate random projection vector
        projection_vector = torch.randn(latent_dim, device=device)
        projection_vector = projection_vector / torch.norm(projection_vector)
        
        # Project latents to 1D
        projections = torch.matmul(latent_samples, projection_vector)  # [batch_size]
        
        # Sort projections for KS test
        projections_sorted, _ = torch.sort(projections)
        n = len(projections_sorted)
        
        # Empirical CDF
        empirical_cdf = torch.arange(1, n+1, dtype=torch.float32, device=device) / n
        
        # Theoretical CDF (standard normal)
        theoretical_cdf = 0.5 * (1 + torch.erf(projections_sorted / np.sqrt(2)))
        
        # KS distance (maximum deviation)
        ks_distance = torch.max(torch.abs(empirical_cdf - theoretical_cdf))
        total_ks_distance += ks_distance.item()
    
    return total_ks_distance / n_random_projections


def get_celeba_dataset(batch_size: int = 32, image_size: int = 64, num_samples: int = None):
    """Get CelebA dataset loaders"""
    logger.info(f"Loading CelebA dataset (image_size={image_size}x{image_size})")
    
    # Get CelebA dataloaders
    train_loader, test_loader = get_celeba_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        download=True,
        data_dir='../../../data/vision',
        num_samples=num_samples  # Limit samples for testing
    )
    
    logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    return train_loader, test_loader


def run_celeba_experiment(model, train_loader, test_loader, epochs: int = 10, 
                         beta: float = 1.0, model_name: str = "Model", device: str = 'cpu'):
    """Run training and evaluation on CelebA"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Lower LR for larger images
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    train_losses = []
    test_metrics = []
    
    start_time = time.time()
    
    logger.info(f"Starting training for {model_name} on {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_ks = 0.0
        n_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_data, batch_labels in progress_bar:
            # Move to device
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Flatten images
            batch_data_flat = batch_data.view(batch_data.size(0), -1)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model.compute_loss(batch_data_flat, batch_labels, beta=beta)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            if 'ks_distance' in loss_dict:
                epoch_ks += loss_dict['ks_distance'].item()
            n_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': epoch_loss / n_batches,
                'ks': epoch_ks / n_batches if epoch_ks > 0 else 0
            })
        
        avg_loss = epoch_loss / n_batches
        avg_ks = epoch_ks / n_batches if epoch_ks > 0 else 0
        train_losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Evaluation every 2 epochs or at the end
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            model.eval()
            test_loss = 0.0
            test_ks = 0.0
            test_batches = 0
            
            all_mus = []
            all_logvars = []
            all_class_logits = []
            all_true_labels = []
            all_latents = []  # Collect latent samples for KS-distance calculation
            
            with torch.no_grad():
                for batch_data, batch_labels in tqdm(test_loader, desc="Evaluating", leave=False):
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    batch_data_flat = batch_data.view(batch_data.size(0), -1)
                    
                    # Compute test loss
                    loss_dict = model.compute_loss(batch_data_flat, batch_labels, beta=beta)
                    test_loss += loss_dict['total_loss'].item()
                    if 'ks_distance' in loss_dict:
                        test_ks += loss_dict['ks_distance'].item()
                    test_batches += 1
                    
                    # Collect samples (limit to prevent memory issues)
                    if len(all_mus) < 50:  # Collect first 50 batches
                        _, class_logits, mu, logvar, z = model.forward(batch_data_flat)
                        all_mus.append(mu.cpu())
                        all_logvars.append(logvar.cpu())
                        all_class_logits.append(class_logits.cpu())
                        all_true_labels.append(batch_labels.cpu())
                        all_latents.append(z.cpu())  # Collect latent samples
            
            # Concatenate
            all_mus = torch.cat(all_mus, dim=0)
            all_logvars = torch.cat(all_logvars, dim=0)
            all_class_logits = torch.cat(all_class_logits, dim=0)
            all_true_labels = torch.cat(all_true_labels, dim=0)
            all_latents = torch.cat(all_latents, dim=0)
            
            # Compute metrics
            metrics = compute_enhanced_metrics(all_mus, all_logvars, all_class_logits, all_true_labels)
            
            # Compute KS-distance for all models (not just DERP-VAE)
            # This is done during evaluation only, so it doesn't affect training
            eval_ks_distance = compute_ks_distance_for_latents(all_latents, n_random_projections=10)
            
            test_loss /= test_batches
            test_ks /= test_batches if test_ks > 0 else 1
            
            test_metric = {
                'epoch': epoch + 1,
                'test_loss': test_loss,
                'test_ks': test_ks,
                'eval_ks_distance': eval_ks_distance,  # KS-distance computed during evaluation
                **metrics
            }
            
            test_metrics.append(test_metric)
            
            logger.info(
                f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, "
                f"Test Loss={test_loss:.4f}, "
                f"KL Div={metrics['kl_divergence']:.4f}, "
                f"Accuracy={metrics['classification_accuracy']:.4f}, "
                f"KS Dist={avg_ks:.4f}, "
                f"Eval KS={eval_ks_distance:.4f}"
            )
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'final_metrics': test_metrics[-1] if test_metrics else {}
    }


def main():
    """Run DERP-VAE experiment on CelebA"""
    logger.info("Starting DERP-VAE Experiment on CelebA")
    
    # Set seeds
    set_seeds(42)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Configuration
    IMAGE_SIZE = 64  # Resize CelebA images to 64x64
    BATCH_SIZE = 32  # Smaller batch size for larger images
    NUM_SAMPLES = None  # Use full dataset (202,599 images)
    EPOCHS = 10  # Number of epochs
    
    # Get CelebA dataset
    train_loader, test_loader = get_celeba_dataset(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_samples=NUM_SAMPLES
    )
    
    # Model configurations for CelebA
    base_config = {
        'input_dim': IMAGE_SIZE * IMAGE_SIZE * 3,  # 64*64*3 = 12,288
        'hidden_dim': 512,  # Larger hidden dim for complex images
        'latent_dim': 64,   # Larger latent space for face features
        'n_classes': 2      # Binary classification (Smiling/Not Smiling)
    }
    
    logger.info(f"Model config: input_dim={base_config['input_dim']}, "
                f"hidden_dim={base_config['hidden_dim']}, "
                f"latent_dim={base_config['latent_dim']}")
    
    # Define experiments
    experiments = [
        ('Standard_VAE', StandardVAE(**base_config), 1.0),
        ('Beta_VAE_0.1', StandardVAE(**base_config), 0.1),  # Lower beta for complex images
        ('DERP_VAE_5probes', DERP_VAE(**base_config, n_probes=5, enforcement_weight=0.5, device=device), 1.0),
    ]
    
    results = {}
    
    # Run experiments
    for name, model, beta in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {name}")
        logger.info(f"Beta: {beta}")
        logger.info(f"{'='*60}")
        
        result = run_celeba_experiment(
            model, train_loader, test_loader, 
            epochs=EPOCHS, beta=beta, model_name=name, device=device
        )
        result['model_name'] = name
        result['beta'] = beta
        result['config'] = base_config.copy()
        results[name] = result
        
        # Print summary
        final = result['final_metrics']
        logger.info(f"\n{name} Final Results:")
        logger.info(f"  Test Loss: {final.get('test_loss', 0):.4f}")
        logger.info(f"  KL Divergence: {final.get('kl_divergence', 0):.4f}")
        logger.info(f"  Classification Accuracy: {final.get('classification_accuracy', 0):.4f}")
        logger.info(f"  Activation Rate: {final.get('activation_rate', 0):.4f}")
        if 'test_ks' in final:
            logger.info(f"  Model KS Distance: {final.get('test_ks', 0):.4f}")
        logger.info(f"  Evaluation KS Distance: {final.get('eval_ks_distance', 0):.4f}")
        logger.info(f"  Training Time: {result['training_time']:.2f}s")
    
    # Compare results
    logger.info("\n" + "="*70)
    logger.info("COMPARISON RESULTS")
    logger.info("="*70)
    
    if 'Standard_VAE' in results and 'DERP_VAE_5probes' in results:
        baseline = results['Standard_VAE']['final_metrics']
        derp = results['DERP_VAE_5probes']['final_metrics']
        
        kl_improvement = (baseline['kl_divergence'] - derp['kl_divergence']) / baseline['kl_divergence'] * 100
        acc_diff = (derp['classification_accuracy'] - baseline['classification_accuracy']) * 100
        
        logger.info(f"\nKL Divergence:")
        logger.info(f"  Standard VAE: {baseline['kl_divergence']:.4f}")
        logger.info(f"  DERP-VAE: {derp['kl_divergence']:.4f}")
        logger.info(f"  Improvement: {kl_improvement:.1f}%")
        
        logger.info(f"\nClassification Accuracy (Smiling Detection):")
        logger.info(f"  Standard VAE: {baseline['classification_accuracy']:.4f}")
        logger.info(f"  DERP-VAE: {derp['classification_accuracy']:.4f}")
        logger.info(f"  Difference: {acc_diff:+.1f}%")
        
        logger.info(f"\nKS Distance (Latent Normality):")
        logger.info(f"  Standard VAE: {baseline.get('eval_ks_distance', 0):.4f}")
        logger.info(f"  DERP-VAE: {derp.get('eval_ks_distance', 0):.4f}")
        if baseline.get('eval_ks_distance', 0) > 0:
            ks_improvement = (baseline['eval_ks_distance'] - derp['eval_ks_distance']) / baseline['eval_ks_distance'] * 100
            logger.info(f"  Improvement: {ks_improvement:.1f}%")
    
    # Save results
    results_path = Path("../results")
    results_path.mkdir(exist_ok=True)
    
    # Save to JSON
    json_results = {
        name: {
            'model_name': r['model_name'],
            'beta': r['beta'],
            'config': r['config'],
            'final_metrics': r['final_metrics'],
            'training_time': r['training_time'],
            'train_losses': r['train_losses'][-5:] if len(r['train_losses']) > 5 else r['train_losses']
        }
        for name, r in results.items()
    }
    
    json_results['experiment_config'] = {
        'dataset': 'CelebA',
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'num_samples': NUM_SAMPLES,
        'epochs': EPOCHS,
        'device': device
    }
    
    with open(results_path / "celeba_experiment_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path / 'celeba_experiment_results.json'}")
    
    # Simple visualization
    plt.figure(figsize=(15, 4))
    
    # Plot training losses
    plt.subplot(1, 4, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot KL divergence comparison
    plt.subplot(1, 4, 2)
    models = list(results.keys())
    kl_values = [results[m]['final_metrics']['kl_divergence'] for m in models]
    plt.bar(range(len(models)), kl_values)
    plt.xlabel('Model')
    plt.ylabel('KL Divergence')
    plt.title('Final KL Divergence')
    plt.xticks(range(len(models)), [m.replace('_', ' ') for m in models], rotation=45)
    plt.grid(True, axis='y')
    
    # Plot accuracy comparison
    plt.subplot(1, 4, 3)
    acc_values = [results[m]['final_metrics']['classification_accuracy'] for m in models]
    plt.bar(range(len(models)), acc_values)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Smiling Detection Accuracy')
    plt.xticks(range(len(models)), [m.replace('_', ' ') for m in models], rotation=45)
    plt.grid(True, axis='y')
    
    # Plot KS-distance comparison
    plt.subplot(1, 4, 4)
    ks_values = [results[m]['final_metrics'].get('eval_ks_distance', 0) for m in models]
    plt.bar(range(len(models)), ks_values)
    plt.xlabel('Model')
    plt.ylabel('KS Distance')
    plt.title('Latent Normality (KS Distance)')
    plt.xticks(range(len(models)), [m.replace('_', ' ') for m in models], rotation=45)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_path / 'celeba_results.png', dpi=150, bbox_inches='tight')
    logger.info(f"Plots saved to {results_path / 'celeba_results.png'}")
    
    return results


if __name__ == "__main__":
    results = main()