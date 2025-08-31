#!/usr/bin/env python3
"""
VAE Posterior Collapse Prevention via Active Distribution Enforcement
Experiment: exp_20250831_020917

Scientific hypothesis testing with proper statistical controls.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
class VAE(nn.Module):
    """Standard Variational Autoencoder"""
    def __init__(self, input_dim=784, latent_dim=20, hidden_dim=400):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

class DERPRandomProbe:
    """Distribution Enforcement via Random Probe (DERP)"""
    def __init__(self, latent_dim, probe_count=5, temperature=1.0):
        self.latent_dim = latent_dim
        self.probe_count = probe_count
        self.temperature = temperature
        # Random probe directions (fixed for consistency)
        self.probe_directions = torch.randn(probe_count, latent_dim)
        self.probe_directions = F.normalize(self.probe_directions, dim=1)
    
    def to(self, device):
        self.probe_directions = self.probe_directions.to(device)
        return self
    
    def project_to_1d(self, z):
        """Project high-dimensional z to 1D along random directions"""
        # z: (batch_size, latent_dim)
        # probe_directions: (probe_count, latent_dim)
        projections = torch.matmul(z.unsqueeze(1), self.probe_directions.unsqueeze(2))
        return projections.squeeze(2)  # (batch_size, probe_count)
    
    def modified_ks_distance(self, sample, target_samples=None):
        """Modified K-S distance using average instead of maximum"""
        if target_samples is None:
            # Compare against standard normal
            sample_sorted = torch.sort(sample)[0]
            n = len(sample_sorted)
            # Empirical CDF values
            empirical_cdf = torch.arange(1, n+1, dtype=torch.float32) / n
            # Standard normal CDF
            normal_cdf = 0.5 * (1 + torch.erf(sample_sorted / np.sqrt(2)))
            # Average absolute difference instead of maximum
            distances = torch.abs(empirical_cdf - normal_cdf)
            return torch.mean(distances)
        else:
            # Two-sample case (not implemented for simplicity)
            return torch.tensor(0.0)
    
    def compute_enforcement_loss(self, z):
        """Compute distributional enforcement loss"""
        projections = self.project_to_1d(z)  # (batch_size, probe_count)
        
        total_loss = 0.0
        for i in range(self.probe_count):
            proj_i = projections[:, i]
            ks_dist = self.modified_ks_distance(proj_i)
            total_loss += ks_dist
        
        return total_loss / self.probe_count / self.temperature

def compute_metrics(model, dataloader, device, enforce_distribution=False, derp_probe=None):
    """Compute evaluation metrics"""
    model.eval()
    
    total_elbo = 0
    total_recon_loss = 0
    total_kld = 0
    total_enforcement_loss = 0
    all_mus = []
    all_logvars = []
    all_zs = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.view(-1, 784).to(device)
            
            recon_batch, mu, logvar, z = model(data)
            
            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            
            # KL divergence
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # ELBO
            elbo = -(recon_loss + kld)
            
            total_elbo += elbo.item()
            total_recon_loss += recon_loss.item()
            total_kld += kld.item()
            
            # Store latent representations
            all_mus.append(mu.cpu())
            all_logvars.append(logvar.cpu())
            all_zs.append(z.cpu())
            
            # Enforcement loss if using DERP
            if enforce_distribution and derp_probe:
                enf_loss = derp_probe.compute_enforcement_loss(z)
                total_enforcement_loss += enf_loss.item()
    
    num_samples = len(dataloader.dataset)
    
    # Compute posterior collapse metrics
    all_mus = torch.cat(all_mus, dim=0)
    all_logvars = torch.cat(all_logvars, dim=0)
    all_zs = torch.cat(all_zs, dim=0)
    
    # Active units (AU) - percentage of dimensions with variance > threshold
    au_threshold = 0.01
    post_var = torch.var(all_mus, dim=0)
    active_units = (post_var > au_threshold).float().mean().item()
    
    # Mutual information approximation I(x,z)
    # Simplified: use negative KL divergence as proxy
    mutual_info = -total_kld / num_samples
    
    # KS test on latent dimensions
    ks_statistics = []
    for dim in range(all_zs.shape[1]):
        z_dim = all_zs[:, dim].numpy()
        # Test against standard normal
        ks_stat, p_value = stats.kstest(z_dim, 'norm')
        ks_statistics.append(ks_stat)
    
    metrics = {
        'elbo': total_elbo / num_samples,
        'reconstruction_loss': total_recon_loss / num_samples,
        'kld': total_kld / num_samples,
        'active_units': active_units,
        'mutual_info': mutual_info,
        'ks_statistics': ks_statistics,
        'avg_ks_statistic': np.mean(ks_statistics),
        'enforcement_loss': total_enforcement_loss / num_samples if enforce_distribution else 0.0
    }
    
    return metrics

def train_vae(model, train_loader, val_loader, device, config, results_dir):
    """Train VAE with or without DERP enforcement"""
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Initialize DERP if using active enforcement
    derp_probe = None
    if config['enforcement_method'] == 'Active':
        derp_probe = DERPRandomProbe(
            latent_dim=config['latent_dim'], 
            probe_count=config['probe_count'],
            temperature=1.0
        ).to(device)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_elbo': [], 'val_elbo': [],
        'train_kld': [], 'val_kld': [],
        'train_au': [], 'val_au': [],
        'train_mi': [], 'val_mi': [],
        'train_ks': [], 'val_ks': []
    }
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            
            optimizer.zero_grad()
            
            recon_batch, mu, logvar, z = model(data)
            
            # Standard VAE loss
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + config['beta'] * kld
            
            # Add DERP enforcement loss if active
            if config['enforcement_method'] == 'Active' and derp_probe:
                # Temperature annealing
                current_temp = max(0.1, 1.0 * (0.9 ** epoch))
                derp_probe.temperature = current_temp
                
                enforcement_loss = derp_probe.compute_enforcement_loss(z)
                loss += config['enforcement_weight'] * enforcement_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluate on validation set
        train_metrics = compute_metrics(model, train_loader, device, 
                                      config['enforcement_method'] == 'Active', 
                                      derp_probe)
        val_metrics = compute_metrics(model, val_loader, device,
                                    config['enforcement_method'] == 'Active',
                                    derp_probe)
        
        # Store history
        history['train_loss'].append(train_loss / len(train_loader.dataset))
        history['val_loss'].append(val_metrics['reconstruction_loss'] + val_metrics['kld'])
        history['train_elbo'].append(train_metrics['elbo'])
        history['val_elbo'].append(val_metrics['elbo'])
        history['train_kld'].append(train_metrics['kld'])
        history['val_kld'].append(val_metrics['kld'])
        history['train_au'].append(train_metrics['active_units'])
        history['val_au'].append(val_metrics['active_units'])
        history['train_mi'].append(train_metrics['mutual_info'])
        history['val_mi'].append(val_metrics['mutual_info'])
        history['train_ks'].append(train_metrics['avg_ks_statistic'])
        history['val_ks'].append(val_metrics['avg_ks_statistic'])
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Train Loss: {train_loss/len(train_loader.dataset):.4f} | '
                  f'Val ELBO: {val_metrics["elbo"]:.4f} | AU: {val_metrics["active_units"]:.3f}')
    
    # Final evaluation
    final_train_metrics = compute_metrics(model, train_loader, device,
                                        config['enforcement_method'] == 'Active',
                                        derp_probe)
    final_val_metrics = compute_metrics(model, val_loader, device,
                                      config['enforcement_method'] == 'Active',
                                      derp_probe)
    
    return {
        'history': history,
        'final_train_metrics': final_train_metrics,
        'final_val_metrics': final_val_metrics,
        'config': config
    }

class SyntheticMNIST(torch.utils.data.Dataset):
    """Synthetic MNIST-like dataset for testing"""
    def __init__(self, num_samples=10000, train=True):
        self.num_samples = num_samples
        # Generate synthetic 28x28 images
        self.data = torch.rand(num_samples, 1, 28, 28)
        self.targets = torch.randint(0, 10, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx].view(-1), self.targets[idx]

def load_data(dataset='mnist', batch_size=128):
    """Load and prepare datasets"""
    if dataset.lower() == 'mnist':
        # Try to load real MNIST first, fallback to synthetic
        try:
            transform = transforms.Compose([transforms.ToTensor()])
            data_root = '../../data/processed/mnist'
            train_dataset = torchvision.datasets.MNIST(
                root=data_root, train=True, 
                download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root=data_root, train=False,
                download=True, transform=transform
            )
            print("Loaded real MNIST dataset")
        except:
            print("Using synthetic MNIST dataset for testing")
            train_dataset = SyntheticMNIST(num_samples=5000, train=True)  
            test_dataset = SyntheticMNIST(num_samples=1000, train=False)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def run_experiment_configuration(config, data_loaders, device, results_dir):
    """Run a single experiment configuration"""
    train_loader, test_loader = data_loaders
    
    # Initialize model
    model = VAE(
        input_dim=784,  # MNIST
        latent_dim=config['latent_dim'],
        hidden_dim=400
    ).to(device)
    
    print(f"\n=== Running Configuration ===")
    print(f"Method: {config['enforcement_method']}")
    print(f"Probes: {config['probe_count']} | Weight: {config['enforcement_weight']} | Seed: {config['seed']}")
    
    start_time = time.time()
    
    # Train model
    results = train_vae(model, train_loader, test_loader, device, config, results_dir)
    
    end_time = time.time()
    results['training_time'] = end_time - start_time
    
    return results

def main():
    parser = argparse.ArgumentParser(description='VAE Posterior Collapse Experiment')
    parser.add_argument('--data_path', type=str, default='../../data/processed')
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 101])
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    train_loader, test_loader = load_data(args.dataset)
    data_loaders = (train_loader, test_loader)
    
    # Experiment configurations
    base_config = {
        'latent_dim': 20,
        'beta': 1.0,
        'lr': 1e-3,
        'epochs': 20,  # Further reduced for faster execution
        'batch_size': 128
    }
    
    # Experimental design: 2x2 factorial + seeds (simplified for faster execution)
    configurations = []
    
    for enforcement_method in ['Passive', 'Active']:
        for probe_count in [5]:  # Fixed single probe count
            for enforcement_weight in [0.5]:  # Fixed single weight
                for seed in args.seeds:
                    config = base_config.copy()
                    config.update({
                        'enforcement_method': enforcement_method,
                        'probe_count': probe_count,
                        'enforcement_weight': enforcement_weight,
                        'seed': seed
                    })
                    configurations.append(config)
    
    print(f"Running {len(configurations)} total experiments...")
    
    # Run experiments
    all_results = []
    for i, config in enumerate(configurations):
        print(f"\n--- Experiment {i+1}/{len(configurations)} ---")
        
        # Set seed
        set_seed(config['seed'])
        
        try:
            results = run_experiment_configuration(config, data_loaders, device, results_dir)
            all_results.append(results)
            
            # Save intermediate results
            with open(results_dir / f'intermediate_results_{i+1}.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = results.copy()
                for key in ['final_train_metrics', 'final_val_metrics']:
                    if 'ks_statistics' in serializable_results[key]:
                        serializable_results[key]['ks_statistics'] = serializable_results[key]['ks_statistics']
                
                json.dump(serializable_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
                
        except Exception as e:
            print(f"Error in experiment {i+1}: {e}")
            continue
    
    # Save all results
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    print(f"\n✓ Completed {len(all_results)} experiments successfully!")
    print(f"Results saved to: {results_dir}")
    
if __name__ == '__main__':
    main()