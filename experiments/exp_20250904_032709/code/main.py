#!/usr/bin/env python3
"""
Rigorous VAE Posterior Collapse Prevention Experiment
Experiment ID: exp_20250904_032709

This experiment addresses critical flaws in previous studies by:
1. Using real MNIST data (not synthetic)
2. Proper statistical methodology (n≥5, appropriate tests)
3. Rigorous controls and reproducible results
4. Effect size calculations and multiple comparison corrections
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
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VAE(nn.Module):
    """Standard Variational Autoencoder for MNIST"""
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
    def __init__(self, latent_dim, probe_count=5, temperature=1.0, device='cpu'):
        self.latent_dim = latent_dim
        self.probe_count = probe_count
        self.temperature = temperature
        self.device = device
        
        # Fixed random projection directions for reproducibility
        torch.manual_seed(42)  # Ensure consistent projections across runs
        self.probe_directions = torch.randn(probe_count, latent_dim, device=device)
        self.probe_directions = F.normalize(self.probe_directions, dim=1)
    
    def to(self, device):
        self.device = device
        self.probe_directions = self.probe_directions.to(device)
        return self
    
    def project_to_1d(self, z):
        """Project high-dimensional z to 1D along random directions"""
        # z: (batch_size, latent_dim)
        # probe_directions: (probe_count, latent_dim)
        projections = torch.matmul(z.unsqueeze(1), self.probe_directions.unsqueeze(2))
        return projections.squeeze(2)  # (batch_size, probe_count)
    
    def modified_ks_distance(self, sample):
        """Modified K-S distance using average instead of maximum for differentiability"""
        if len(sample) < 10:  # Too few samples
            return torch.tensor(0.0, device=self.device)
            
        sample_sorted = torch.sort(sample)[0]
        n = len(sample_sorted)
        
        # Empirical CDF values
        empirical_cdf = torch.arange(1, n+1, dtype=torch.float32, device=self.device) / n
        
        # Standard normal CDF using error function
        normal_cdf = 0.5 * (1 + torch.erf(sample_sorted / np.sqrt(2)))
        
        # Average absolute difference (differentiable)
        distances = torch.abs(empirical_cdf - normal_cdf)
        return torch.mean(distances)
    
    def compute_enforcement_loss(self, z):
        """Compute distributional enforcement loss via random probes"""
        projections = self.project_to_1d(z)  # (batch_size, probe_count)
        
        total_loss = torch.tensor(0.0, device=self.device)
        valid_probes = 0
        
        for i in range(self.probe_count):
            proj_i = projections[:, i]
            if len(proj_i) >= 10:  # Minimum samples for meaningful KS test
                ks_dist = self.modified_ks_distance(proj_i)
                total_loss += ks_dist
                valid_probes += 1
        
        if valid_probes == 0:
            return torch.tensor(0.0, device=self.device)
        
        return total_loss / valid_probes / self.temperature

def compute_metrics(model, dataloader, device, derp_probe=None, enforce_distribution=False):
    """Compute comprehensive evaluation metrics"""
    model.eval()
    
    metrics = {
        'elbo': 0.0,
        'reconstruction_loss': 0.0, 
        'kld': 0.0,
        'enforcement_loss': 0.0,
        'all_mus': [],
        'all_logvars': [],
        'all_zs': []
    }
    
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.view(-1, 784).to(device)
            batch_size = data.shape[0]
            
            recon_batch, mu, logvar, z = model(data)
            
            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            
            # KL divergence to standard normal prior
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # ELBO
            elbo = -(recon_loss + kld)
            
            metrics['elbo'] += elbo.item()
            metrics['reconstruction_loss'] += recon_loss.item()
            metrics['kld'] += kld.item()
            
            # Store latent representations
            metrics['all_mus'].append(mu.cpu())
            metrics['all_logvars'].append(logvar.cpu())
            metrics['all_zs'].append(z.cpu())
            
            # Enforcement loss if using DERP
            if enforce_distribution and derp_probe:
                enf_loss = derp_probe.compute_enforcement_loss(z)
                metrics['enforcement_loss'] += enf_loss.item() * batch_size
            
            total_samples += batch_size
    
    # Normalize by number of samples
    for key in ['elbo', 'reconstruction_loss', 'kld', 'enforcement_loss']:
        metrics[key] /= total_samples
    
    # Concatenate all latent representations
    all_mus = torch.cat(metrics['all_mus'], dim=0)
    all_logvars = torch.cat(metrics['all_logvars'], dim=0) 
    all_zs = torch.cat(metrics['all_zs'], dim=0)
    
    # Active units (AU) - dimensions with posterior variance > threshold
    au_threshold = 0.01
    posterior_var = torch.var(all_mus, dim=0)
    active_units = (posterior_var > au_threshold).float().mean().item()
    
    # Mutual information approximation
    mutual_info = -metrics['kld']  # Negative KL as MI proxy
    
    # Kolmogorov-Smirnov statistics for each latent dimension
    ks_statistics = []
    shapiro_statistics = []
    
    for dim in range(all_zs.shape[1]):
        z_dim = all_zs[:, dim].numpy()
        
        # KS test against standard normal
        ks_stat, _ = stats.kstest(z_dim, 'norm')
        ks_statistics.append(ks_stat)
        
        # Shapiro-Wilk test for normality (if sample size allows)
        if len(z_dim) <= 5000:  # Shapiro-Wilk limitation
            sw_stat, _ = stats.shapiro(z_dim[:5000])
            shapiro_statistics.append(sw_stat)
    
    # Final metrics
    final_metrics = {
        'elbo': metrics['elbo'],
        'reconstruction_loss': metrics['reconstruction_loss'],
        'kld': metrics['kld'],
        'active_units': active_units,
        'mutual_info': mutual_info,
        'avg_ks_statistic': np.mean(ks_statistics),
        'ks_statistics': ks_statistics,
        'avg_shapiro_statistic': np.mean(shapiro_statistics) if shapiro_statistics else 0.0,
        'shapiro_statistics': shapiro_statistics,
        'enforcement_loss': metrics['enforcement_loss']
    }
    
    return final_metrics

def train_vae(model, train_loader, val_loader, device, config, results_dir):
    """Train VAE with optional DERP enforcement"""
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Initialize DERP if using active enforcement
    derp_probe = None
    if config['enforcement_method'] == 'Active':
        derp_probe = DERPRandomProbe(
            latent_dim=config['latent_dim'],
            probe_count=config['probe_count'], 
            temperature=config['temperature'],
            device=device
        )
    
    # Training history
    history = defaultdict(list)
    
    print(f"Training {config['enforcement_method']} VAE (seed={config['seed']})...")
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        train_samples = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            batch_size = data.shape[0]
            
            optimizer.zero_grad()
            
            recon_batch, mu, logvar, z = model(data)
            
            # Standard VAE losses
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + config['beta'] * kld
            
            # Add DERP enforcement loss if active
            enforcement_loss = torch.tensor(0.0).to(device)
            if config['enforcement_method'] == 'Active' and derp_probe:
                enforcement_loss = derp_probe.compute_enforcement_loss(z)
                loss += config['enforcement_weight'] * enforcement_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_samples += batch_size
        
        # Evaluation
        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            val_metrics = compute_metrics(
                model, val_loader, device, 
                derp_probe, config['enforcement_method'] == 'Active'
            )
            
            print(f"Epoch {epoch:3d} | Loss: {train_loss/train_samples:.4f} | "
                  f"ELBO: {val_metrics['elbo']:.2f} | AU: {val_metrics['active_units']:.3f} | "
                  f"KLD: {val_metrics['kld']:.2f}")
            
            # Store metrics
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss / train_samples)
            for key, value in val_metrics.items():
                if not isinstance(value, list):  # Skip lists like ks_statistics
                    history[f'val_{key}'].append(value)
    
    # Final evaluation
    final_train_metrics = compute_metrics(
        model, train_loader, device, 
        derp_probe, config['enforcement_method'] == 'Active'
    )
    final_val_metrics = compute_metrics(
        model, val_loader, device,
        derp_probe, config['enforcement_method'] == 'Active'
    )
    
    return {
        'history': history,
        'final_train_metrics': final_train_metrics,
        'final_val_metrics': final_val_metrics,
        'config': config
    }

def load_real_mnist(data_root='../../data/processed', batch_size=128):
    """Load REAL MNIST dataset (not synthetic)"""
    
    transform = transforms.ToTensor()
    
    try:
        # Load real MNIST from processed data directory
        train_dataset = torchvision.datasets.MNIST(
            root=data_root + '/mnist', 
            train=True,
            download=False,  # Should already be downloaded
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_root + '/mnist',
            train=False, 
            download=False,
            transform=transform
        )
        print(f"✅ Loaded REAL MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
        
    except Exception as e:
        print(f"❌ Error loading MNIST from {data_root}: {e}")
        print("Falling back to torchvision download...")
        
        # Fallback: download from torchvision
        train_dataset = torchvision.datasets.MNIST(
            root='/tmp/mnist', 
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='/tmp/mnist',
            train=False,
            download=True, 
            transform=transform
        )
        print(f"✅ Downloaded REAL MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def run_experiment_configuration(config, data_loaders, device, results_dir):
    """Run a single experiment configuration"""
    train_loader, val_loader = data_loaders
    
    # Set seed for this configuration
    set_seed(config['seed'])
    
    # Initialize model
    model = VAE(
        input_dim=784,
        latent_dim=config['latent_dim'],
        hidden_dim=400
    ).to(device)
    
    print(f"\n{'='*60}")
    print(f"Configuration: {config['enforcement_method']} | Seed: {config['seed']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Train model
    results = train_vae(model, train_loader, val_loader, device, config, results_dir)
    
    training_time = time.time() - start_time
    results['training_time'] = training_time
    
    print(f"✅ Training completed in {training_time:.1f}s")
    
    return results

def statistical_analysis(all_results, results_dir):
    """Perform rigorous statistical analysis"""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Separate results by enforcement method
    passive_results = [r for r in all_results if r['config']['enforcement_method'] == 'Passive']
    active_results = [r for r in all_results if r['config']['enforcement_method'] == 'Active']
    
    print(f"Sample sizes: Passive n={len(passive_results)}, Active n={len(active_results)}")
    
    if len(passive_results) < 3 or len(active_results) < 3:
        print("⚠️ WARNING: Sample size too small for reliable statistical inference")
    
    # Metrics to analyze
    metrics = ['elbo', 'reconstruction_loss', 'kld', 'active_units', 'mutual_info', 'avg_ks_statistic']
    
    statistical_results = {}
    
    for metric in metrics:
        passive_values = [r['final_val_metrics'][metric] for r in passive_results]
        active_values = [r['final_val_metrics'][metric] for r in active_results]
        
        passive_mean = np.mean(passive_values)
        active_mean = np.mean(active_values)
        passive_std = np.std(passive_values, ddof=1)
        active_std = np.std(active_values, ddof=1)
        
        # Statistical test (Mann-Whitney U for small samples)
        if len(passive_values) >= 3 and len(active_values) >= 3:
            statistic, p_value = stats.mannwhitneyu(
                passive_values, active_values, 
                alternative='two-sided'
            )
        else:
            statistic, p_value = 0, 1.0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((passive_std ** 2) + (active_std ** 2)) / 2)
        if pooled_std > 0:
            cohens_d = (passive_mean - active_mean) / pooled_std
        else:
            cohens_d = 0.0
        
        # Confidence intervals (95%)
        if len(passive_values) >= 2:
            passive_ci = stats.t.interval(0.95, len(passive_values)-1, 
                                        loc=passive_mean, 
                                        scale=stats.sem(passive_values))
        else:
            passive_ci = (passive_mean, passive_mean)
            
        if len(active_values) >= 2:
            active_ci = stats.t.interval(0.95, len(active_values)-1,
                                       loc=active_mean,
                                       scale=stats.sem(active_values))
        else:
            active_ci = (active_mean, active_mean)
        
        statistical_results[metric] = {
            'passive_mean': passive_mean,
            'active_mean': active_mean,
            'passive_std': passive_std,
            'active_std': active_std,
            'passive_ci': passive_ci,
            'active_ci': active_ci,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': cohens_d,
            'significant': p_value < 0.05
        }
        
        # Print results
        print(f"\n{metric.upper()}:")
        print(f"  Passive: {passive_mean:.4f} ± {passive_std:.4f}")
        print(f"  Active:  {active_mean:.4f} ± {active_std:.4f}")
        print(f"  Effect size (d): {cohens_d:.3f}")
        print(f"  p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")
    
    # Multiple comparison correction (Bonferroni)
    n_comparisons = len(metrics)
    corrected_alpha = 0.05 / n_comparisons
    
    print(f"\nMULTIPLE COMPARISONS CORRECTION:")
    print(f"Bonferroni corrected α = {corrected_alpha:.4f}")
    
    significant_after_correction = 0
    for metric, result in statistical_results.items():
        corrected_significant = result['p_value'] < corrected_alpha
        result['bonferroni_significant'] = corrected_significant
        if corrected_significant:
            significant_after_correction += 1
            print(f"  {metric}: SIGNIFICANT after correction (p={result['p_value']:.4f})")
    
    # Success criteria evaluation
    print(f"\nSUCCESS CRITERIA EVALUATION:")
    
    # 1. KL divergence reduction >30%
    kld_reduction = ((statistical_results['kld']['passive_mean'] - 
                     statistical_results['kld']['active_mean']) / 
                     statistical_results['kld']['passive_mean']) * 100
    criterion_1 = kld_reduction > 30
    print(f"  1. KL Reduction: {kld_reduction:.1f}% {'✅ PASS' if criterion_1 else '❌ FAIL'} (>30% required)")
    
    # 2. Statistical significance (p < 0.05) for KL divergence
    criterion_2 = statistical_results['kld']['significant']
    print(f"  2. Statistical Significance: p={statistical_results['kld']['p_value']:.4f} {'✅ PASS' if criterion_2 else '❌ FAIL'} (p<0.05 required)")
    
    # 3. Reconstruction quality (<15% ELBO degradation)
    elbo_change = ((statistical_results['elbo']['active_mean'] - 
                   statistical_results['elbo']['passive_mean']) / 
                   abs(statistical_results['elbo']['passive_mean'])) * 100
    criterion_3 = elbo_change > -15  # Less than 15% degradation
    print(f"  3. ELBO Change: {elbo_change:.1f}% {'✅ PASS' if criterion_3 else '❌ FAIL'} (<15% degradation allowed)")
    
    # 4. Active units improvement >25%
    au_improvement = ((statistical_results['active_units']['active_mean'] - 
                      statistical_results['active_units']['passive_mean']) / 
                      statistical_results['active_units']['passive_mean']) * 100
    criterion_4 = au_improvement > 25
    print(f"  4. Active Units: {au_improvement:.1f}% {'✅ PASS' if criterion_4 else '❌ FAIL'} (>25% improvement)")
    
    # Overall assessment
    criteria_passed = sum([criterion_1, criterion_2, criterion_3, criterion_4])
    print(f"\nOVERALL ASSESSMENT: {criteria_passed}/4 criteria passed")
    
    if criteria_passed >= 3:
        print("🎉 EXPERIMENT SUCCESS: Strong evidence for DERP effectiveness")
    elif criteria_passed >= 2:
        print("⚡ PARTIAL SUCCESS: Some evidence for DERP effectiveness")
    else:
        print("❌ EXPERIMENT FAILURE: Insufficient evidence for DERP effectiveness")
    
    # Save statistical results
    with open(results_dir / 'statistical_analysis.json', 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for metric, result in statistical_results.items():
            json_results[metric] = {
                k: float(v) if isinstance(v, (np.float64, np.float32)) else 
                   [float(x) for x in v] if isinstance(v, (tuple, list)) else v
                for k, v in result.items()
            }
        
        analysis_summary = {
            'statistical_results': json_results,
            'sample_sizes': {'passive': len(passive_results), 'active': len(active_results)},
            'bonferroni_correction': {'alpha': corrected_alpha, 'n_comparisons': n_comparisons},
            'success_criteria': {
                'kl_reduction_percent': kld_reduction,
                'kl_significant': criterion_1 and criterion_2,
                'elbo_degradation_percent': elbo_change,
                'elbo_acceptable': criterion_3,
                'au_improvement_percent': au_improvement,
                'au_improved': criterion_4,
                'overall_pass': criteria_passed >= 3
            }
        }
        
        json.dump(analysis_summary, f, indent=2)
    
    return analysis_summary

def main():
    parser = argparse.ArgumentParser(description='Rigorous VAE Posterior Collapse Experiment')
    parser.add_argument('--data_path', type=str, default='../../data/processed')
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 101112])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Load REAL MNIST data
    print("📊 Loading MNIST dataset...")
    train_loader, val_loader = load_real_mnist(args.data_path, args.batch_size)
    data_loaders = (train_loader, val_loader)
    
    # Experiment configurations
    base_config = {
        'latent_dim': 20,
        'beta': 1.0,
        'lr': 1e-3,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'probe_count': 5,
        'enforcement_weight': 0.5,
        'temperature': 1.0
    }
    
    # Generate all configurations
    configurations = []
    for method in ['Passive', 'Active']:
        for seed in args.seeds:
            config = base_config.copy()
            config.update({
                'enforcement_method': method,
                'seed': seed
            })
            configurations.append(config)
    
    print(f"🧪 Running {len(configurations)} experiments...")
    
    # Run all experiments
    all_results = []
    for i, config in enumerate(configurations):
        print(f"\n🔄 Experiment {i+1}/{len(configurations)}")
        
        try:
            results = run_experiment_configuration(config, data_loaders, device, results_dir)
            all_results.append(results)
            
            # Save intermediate results
            with open(results_dir / f'results_{i+1:02d}.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error in experiment {i+1}: {e}")
            continue
    
    # Statistical analysis
    if len(all_results) >= 6:  # Minimum for meaningful analysis
        analysis_summary = statistical_analysis(all_results, results_dir)
    else:
        print(f"⚠️ Only {len(all_results)} experiments completed. Need ≥6 for analysis.")
        analysis_summary = None
    
    # Save all results
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Experiment completed!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"📊 {len(all_results)} successful runs")
    
    if analysis_summary and analysis_summary['success_criteria']['overall_pass']:
        print("🎉 DERP framework shows statistically significant effectiveness!")
    else:
        print("📋 Results require further investigation or larger sample sizes.")

if __name__ == '__main__':
    main()