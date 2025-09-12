"""
Small-scale DERP-VAE experiment for faster execution and validation
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

# Add current directory to Python path
sys.path.append('.')

from derp_vae import DERP_VAE, StandardVAE, compute_posterior_collapse_metrics, statistical_normality_test
from data_loader import create_synthetic_dataset
from torch.utils.data import TensorDataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)

def create_small_dataset(n_samples: int = 1000, input_dim: int = 784):
    """Create normalized dataset for VAE training"""
    # Create synthetic high-dimensional data with mixture of Gaussians
    data, labels = create_synthetic_dataset(n_samples=n_samples, n_dims=input_dim, n_components=5)
    
    # Normalize to [0,1] range for BCE loss
    data = torch.sigmoid(data)
    
    # Create train/test split
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_data, torch.zeros(len(train_data)))
    test_dataset = TensorDataset(test_data, torch.zeros(len(test_data)))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def run_experiment(model, train_loader, test_loader, epochs: int = 20, beta: float = 1.0):
    """Run training and evaluation for a single model"""
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_losses = []
    test_metrics = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            loss_dict = model.compute_loss(batch, beta=beta)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        train_losses.append(epoch_loss / n_batches)
        
        # Evaluation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            test_loss = 0.0
            test_batches = 0
            all_latents = []
            all_mus = []
            all_logvars = []
            
            with torch.no_grad():
                for batch, _ in test_loader:
                    loss_dict = model.compute_loss(batch, beta=beta)
                    test_loss += loss_dict['total_loss'].item()
                    test_batches += 1
                    
                    # Collect latent samples for analysis
                    _, mu, logvar, z = model.forward(batch)
                    all_latents.append(z)
                    all_mus.append(mu)
                    all_logvars.append(logvar)
            
            # Compute comprehensive metrics
            all_latents = torch.cat(all_latents, dim=0)
            all_mus = torch.cat(all_mus, dim=0)
            all_logvars = torch.cat(all_logvars, dim=0)
            
            # Posterior collapse metrics
            collapse_metrics = compute_posterior_collapse_metrics(all_mus, all_logvars)
            
            # Statistical normality testing
            normality_metrics = statistical_normality_test(all_latents)
            
            test_metrics.append({
                'epoch': epoch + 1,
                'test_loss': test_loss / test_batches,
                **collapse_metrics,
                **normality_metrics
            })
            
            logger.info(f"Epoch {epoch+1}: "
                       f"Train Loss={train_losses[-1]:.4f}, "
                       f"Test Loss={test_loss/test_batches:.4f}, "
                       f"KL Div={collapse_metrics['kl_divergence']:.4f}, "
                       f"Activation Rate={collapse_metrics['activation_rate']:.4f}")
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'final_metrics': test_metrics[-1] if test_metrics else {}
    }

def main():
    """Run small-scale comparative experiment"""
    logger.info("Starting small-scale DERP-VAE experiment")
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Create dataset
    train_loader, test_loader = create_small_dataset(n_samples=2000, input_dim=256)
    logger.info(f"Created dataset with {len(train_loader)*64} training samples")
    
    # Model configurations
    base_config = {
        'input_dim': 256,
        'hidden_dim': 128,
        'latent_dim': 32
    }
    
    experiments = [
        ('Standard_VAE', StandardVAE(**base_config), 1.0),
        ('Beta_VAE_0.5', StandardVAE(**base_config), 0.5),
        ('Beta_VAE_2.0', StandardVAE(**base_config), 2.0),
        ('DERP_VAE_3probes', DERP_VAE(**base_config, n_probes=3, enforcement_weight=1.0, device='cpu'), 1.0),
        ('DERP_VAE_5probes', DERP_VAE(**base_config, n_probes=5, enforcement_weight=1.0, device='cpu'), 1.0),
    ]
    
    results = {}
    
    for name, model, beta in experiments:
        logger.info(f"\\n{'='*50}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*50}")
        
        result = run_experiment(model, train_loader, test_loader, epochs=15, beta=beta)
        result['model_name'] = name
        result['beta'] = beta
        results[name] = result
        
        # Print summary
        final = result['final_metrics']
        logger.info(f"{name} Results:")
        logger.info(f"  Final KL Divergence: {final.get('kl_divergence', 0):.4f}")
        logger.info(f"  Activation Rate: {final.get('activation_rate', 0):.4f}")
        logger.info(f"  Normality Compliance: {final.get('normality_compliance', 0):.4f}")
        logger.info(f"  Training Time: {result['training_time']:.2f}s")
    
    # Statistical comparison
    logger.info("\\n" + "="*60)
    logger.info("STATISTICAL COMPARISON")
    logger.info("="*60)
    
    # Compare DERP-VAE vs baselines
    baseline_kl = results['Standard_VAE']['final_metrics']['kl_divergence']
    derp_kl = results['DERP_VAE_5probes']['final_metrics']['kl_divergence']
    kl_improvement = (baseline_kl - derp_kl) / baseline_kl * 100
    
    baseline_activation = results['Standard_VAE']['final_metrics']['activation_rate']
    derp_activation = results['DERP_VAE_5probes']['final_metrics']['activation_rate']
    activation_improvement = (derp_activation - baseline_activation) / baseline_activation * 100
    
    logger.info(f"KL Divergence Reduction: {kl_improvement:.1f}%")
    logger.info(f"Activation Rate Improvement: {activation_improvement:.1f}%")
    
    # Hypothesis assessment
    h1_supported = kl_improvement > 10  # Looking for meaningful improvement
    h2_supported = activation_improvement > 5
    
    logger.info(f"\\nH1 (Active Enforcement): {'SUPPORTED' if h1_supported else 'MIXED EVIDENCE'}")
    logger.info(f"H2 (Identifiability): {'SUPPORTED' if h2_supported else 'MIXED EVIDENCE'}")
    
    # Save results
    results_path = Path("../results")
    results_path.mkdir(exist_ok=True)
    
    with open(results_path / "small_scale_results.json", 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for k, v in results.items():
            json_results[k] = {
                'model_name': v['model_name'],
                'beta': v['beta'],
                'training_time': v['training_time'],
                'final_metrics': v['final_metrics'],
                'train_losses': v['train_losses'][-5:],  # Last 5 epochs only
                'test_metrics': v['test_metrics']
            }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path / 'small_scale_results.json'}")
    
    return results

if __name__ == "__main__":
    results = main()