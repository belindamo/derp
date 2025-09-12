"""
Training script for DERP-VAE experiment
Tests hypothesis H1 & H2: Active distributional enforcement prevents posterior collapse
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import time
import pandas as pd

from derp_vae import DERP_VAE, StandardVAE, compute_posterior_collapse_metrics, statistical_normality_test
from data_loader import get_cifar10_dataloaders, get_data_statistics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class VAETrainer:
    """Trainer class for VAE experiments with comprehensive metrics tracking"""
    
    def __init__(self, model, device: str = 'cpu', learning_rate: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Metrics tracking
        self.train_metrics = []
        self.test_metrics = []
        self.training_time = 0.0
        
    def train_epoch(self, dataloader, beta: float = 1.0) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0, 
            'kl_loss': 0.0,
            'distributional_loss': 0.0,
            'kl_divergence': 0.0,
            'mutual_information': 0.0,
            'activation_rate': 0.0,
            'avg_posterior_variance': 0.0
        }
        
        n_batches = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Training")):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Compute loss
            loss_dict = self.model.compute_loss(data, beta=beta)
            loss = loss_dict['total_loss']
            
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            for key in ['total_loss', 'recon_loss', 'kl_loss', 'distributional_loss']:
                epoch_metrics[key] += loss_dict[key].item()
            
            # Compute posterior collapse metrics (every 10 batches to save computation)
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    _, mu, logvar, _ = self.model.forward(data)
                    collapse_metrics = compute_posterior_collapse_metrics(mu, logvar)
                    
                    for key in ['kl_divergence', 'mutual_information', 'activation_rate', 'avg_posterior_variance']:
                        epoch_metrics[key] += collapse_metrics[key]
            
            n_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        return epoch_metrics
    
    def evaluate(self, dataloader, beta: float = 1.0) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        eval_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'distributional_loss': 0.0,
            'kl_divergence': 0.0,
            'mutual_information': 0.0,
            'activation_rate': 0.0,
            'avg_posterior_variance': 0.0,
            'normality_compliance': 0.0,
            'ks_test_mean_pval': 0.0
        }
        
        n_batches = 0
        all_latents = []
        
        with torch.no_grad():
            for data, _ in tqdm(dataloader, desc="Evaluating"):
                data = data.to(self.device)
                
                # Compute loss
                loss_dict = self.model.compute_loss(data, beta=beta)
                
                # Forward pass for latent analysis
                _, mu, logvar, z = self.model.forward(data)
                
                # Accumulate loss metrics
                for key in ['total_loss', 'recon_loss', 'kl_loss', 'distributional_loss']:
                    eval_metrics[key] += loss_dict[key].item()
                
                # Compute posterior collapse metrics
                collapse_metrics = compute_posterior_collapse_metrics(mu, logvar)
                for key in ['kl_divergence', 'mutual_information', 'activation_rate', 'avg_posterior_variance']:
                    eval_metrics[key] += collapse_metrics[key]
                
                # Collect latent samples for statistical testing
                all_latents.append(z.cpu())
                n_batches += 1
        
        # Average loss and collapse metrics
        for key in ['total_loss', 'recon_loss', 'kl_loss', 'distributional_loss', 
                   'kl_divergence', 'mutual_information', 'activation_rate', 'avg_posterior_variance']:
            eval_metrics[key] /= n_batches
        
        # Statistical normality testing on collected latent samples
        if all_latents:
            all_latents = torch.cat(all_latents, dim=0)
            normality_metrics = statistical_normality_test(all_latents)
            eval_metrics.update(normality_metrics)
        
        return eval_metrics
    
    def train(self, train_loader, test_loader, epochs: int = 50, beta: float = 1.0, 
             eval_interval: int = 5) -> Dict[str, List]:
        """Full training loop with evaluation"""
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, beta=beta)
            self.train_metrics.append(train_metrics)
            
            # Evaluate periodically
            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                test_metrics = self.evaluate(test_loader, beta=beta)
                self.test_metrics.append(test_metrics)
                
                logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                          f"Test Loss: {test_metrics['total_loss']:.4f}, "
                          f"KL Div: {test_metrics['kl_divergence']:.4f}, "
                          f"Activation Rate: {test_metrics['activation_rate']:.4f}")
        
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        return {
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'training_time': self.training_time
        }

def run_single_experiment(model_type: str, model_config: Dict, 
                         train_loader, test_loader, 
                         epochs: int = 30, beta: float = 1.0, 
                         device: str = 'cpu') -> Dict:
    """Run a single experiment configuration"""
    logger.info(f"Running experiment: {model_type} with config {model_config}")
    
    # Create model
    if model_type == 'standard':
        model = StandardVAE(**model_config)
    elif model_type == 'derp':
        model = DERP_VAE(**model_config, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create trainer and run experiment
    trainer = VAETrainer(model, device=device)
    results = trainer.train(train_loader, test_loader, epochs=epochs, beta=beta)
    
    # Add configuration to results
    results['model_type'] = model_type
    results['model_config'] = model_config
    results['beta'] = beta
    results['epochs'] = epochs
    
    return results

def comparative_experiment(data_path: str, results_path: str, device: str = 'cpu'):
    """Run comparative experiment between Standard VAE, β-VAE, and DERP-VAE"""
    logger.info("Starting comparative VAE experiment")
    
    # Load data
    train_loader, test_loader = get_cifar10_dataloaders(
        data_path, 
        batch_size=128, 
        subset_size=5000  # Use subset for faster experimentation
    )
    
    # Log dataset statistics
    train_stats = get_data_statistics(train_loader)
    logger.info(f"Dataset statistics: {train_stats}")
    
    # Experimental configurations
    base_config = {
        'input_dim': 3072,  # 32*32*3 for CIFAR-10
        'hidden_dim': 512,
        'latent_dim': 64
    }
    
    experiments = [
        # Standard VAE
        {
            'name': 'Standard_VAE',
            'model_type': 'standard',
            'config': base_config.copy(),
            'beta': 1.0
        },
        # β-VAE variants
        {
            'name': 'Beta_VAE_0.5',
            'model_type': 'standard',
            'config': base_config.copy(),
            'beta': 0.5
        },
        {
            'name': 'Beta_VAE_2.0',
            'model_type': 'standard',
            'config': base_config.copy(),
            'beta': 2.0
        },
        # DERP-VAE variants
        {
            'name': 'DERP_VAE_5probes',
            'model_type': 'derp',
            'config': {**base_config, 'n_probes': 5, 'enforcement_weight': 1.0},
            'beta': 1.0
        },
        {
            'name': 'DERP_VAE_10probes',
            'model_type': 'derp', 
            'config': {**base_config, 'n_probes': 10, 'enforcement_weight': 1.0},
            'beta': 1.0
        },
        {
            'name': 'DERP_VAE_heavy_enforcement',
            'model_type': 'derp',
            'config': {**base_config, 'n_probes': 5, 'enforcement_weight': 2.0},
            'beta': 1.0
        }
    ]
    
    results = []
    
    for exp in experiments:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {exp['name']}")
        logger.info(f"{'='*50}")
        
        result = run_single_experiment(
            model_type=exp['model_type'],
            model_config=exp['config'],
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=25,  # Reduced for faster experimentation
            beta=exp['beta'],
            device=device
        )
        
        result['experiment_name'] = exp['name']
        results.append(result)
        
        # Save intermediate results
        with open(Path(results_path) / f"{exp['name']}_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_safe_result = {}
            for k, v in result.items():
                if isinstance(v, (list, dict, str, int, float)):
                    json_safe_result[k] = v
                else:
                    json_safe_result[k] = str(v)
            json.dump(json_safe_result, f, indent=2)
    
    # Save combined results
    with open(Path(results_path) / 'comparative_results.json', 'w') as f:
        json_safe_results = []
        for result in results:
            json_safe_result = {}
            for k, v in result.items():
                if isinstance(v, (list, dict, str, int, float)):
                    json_safe_result[k] = v
                else:
                    json_safe_result[k] = str(v)
            json_safe_results.append(json_safe_result)
        json.dump(json_safe_results, f, indent=2)
    
    logger.info("Comparative experiment completed!")
    return results

if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Paths
    data_path = "/home/runner/work/derp/derp/data/vision/cifar-10-batches-py"
    results_path = "/home/runner/work/derp/derp/experiments/exp_20250904_180037/results"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    # Run comparative experiment
    results = comparative_experiment(data_path, results_path, device=device)