"""
Convolutional Distribution Enforcement via Random Probe (DERP) VAE Implementation
For CIFAR-10 image experiments
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import Tuple, List, Dict
import logging

class RandomProbeTest:
    """
    Random Probe Testing for distributional enforcement using modified K-S distance
    """
    def __init__(self, n_probes: int = 5, device: str = 'cpu'):
        self.n_probes = n_probes
        self.device = device
        # Generate random projection vectors
        self.probe_vectors = None
    
    def initialize_probes(self, latent_dim: int):
        """Initialize random probe vectors for testing"""
        self.probe_vectors = torch.randn(self.n_probes, latent_dim, device=self.device)
        # Normalize probe vectors
        self.probe_vectors = F.normalize(self.probe_vectors, dim=1)
    
    def project_to_1d(self, latent_samples: torch.Tensor) -> torch.Tensor:
        """
        Project high-dimensional latent samples to 1D using random probes
        Args:
            latent_samples: [batch_size, latent_dim]
        Returns:
            projections: [n_probes, batch_size]
        """
        if self.probe_vectors is None:
            self.initialize_probes(latent_samples.shape[1])
        
        # Matrix multiplication for all probes at once
        projections = torch.mm(self.probe_vectors, latent_samples.T)  # [n_probes, batch_size]
        return projections
    
    def modified_ks_distance(self, x: torch.Tensor, target_dist: str = 'normal') -> torch.Tensor:
        """
        Compute modified K-S distance using average rather than maximum deviation
        for differentiable optimization
        
        Args:
            x: 1D projected samples [batch_size]
            target_dist: target distribution ('normal' for standard Gaussian)
        Returns:
            average K-S distance (differentiable)
        """
        x_sorted, _ = torch.sort(x)
        n = len(x_sorted)
        
        # Empirical CDF values
        empirical_cdf = torch.arange(1, n+1, dtype=torch.float32, device=x.device) / n
        
        # Theoretical CDF (standard normal)
        if target_dist == 'normal':
            # Standard normal CDF approximation (differentiable)
            theoretical_cdf = 0.5 * (1 + torch.erf(x_sorted / np.sqrt(2)))
        else:
            raise ValueError(f"Unsupported target distribution: {target_dist}")
        
        # Modified K-S: average absolute deviation instead of maximum
        deviations = torch.abs(empirical_cdf - theoretical_cdf)
        avg_ks_distance = torch.mean(deviations)
        
        return avg_ks_distance
    
    def compute_distributional_loss(self, latent_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute distributional enforcement loss using random probes
        
        Args:
            latent_samples: [batch_size, latent_dim]
        Returns:
            distributional_loss: scalar tensor
        """
        projections = self.project_to_1d(latent_samples)  # [n_probes, batch_size]
        
        total_loss = 0.0
        for i in range(self.n_probes):
            probe_projection = projections[i]  # [batch_size]
            ks_distance = self.modified_ks_distance(probe_projection, target_dist='normal')
            total_loss += ks_distance
        
        # Average across probes
        return total_loss / self.n_probes


class ConvVAEEncoder(nn.Module):
    """Convolutional VAE Encoder for CIFAR-10 images"""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # CIFAR-10 is 32x32x3
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)  # 16x16x32
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # 8x8x64
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)  # 4x4x128
        
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
    def forward(self, x):
        # x: [batch_size, 3, 32, 32]
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(h.size(0), -1)  # Flatten: [batch_size, 128*4*4]
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConvVAEDecoder(nn.Module):
    """Convolutional VAE Decoder for CIFAR-10 images"""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(32, 3, 4, 2, 1)    # 16x16 -> 32x32
        
    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(h.size(0), 128, 4, 4)  # Reshape to feature maps
        
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = torch.sigmoid(self.deconv3(h))  # [batch_size, 3, 32, 32]
        return h


class DERP_VAE(nn.Module):
    """
    Distribution Enforcement via Random Probe VAE
    Implements active distributional enforcement to prevent posterior collapse
    """
    def __init__(self, latent_dim: int = 64, n_probes: int = 5, 
                 enforcement_weight: float = 1.0, device: str = 'cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.enforcement_weight = enforcement_weight
        
        # Convolutional architecture for images
        self.encoder = ConvVAEEncoder(latent_dim)
        self.decoder = ConvVAEDecoder(latent_dim)
        
        # DERP component for distributional enforcement
        self.random_probe = RandomProbeTest(n_probes=n_probes, device=device)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through DERP-VAE
        Returns: reconstruction, mu, logvar, z_sample
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar, z
    
    def compute_loss(self, x: torch.Tensor, beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute DERP-VAE loss with distributional enforcement
        
        Args:
            x: input data [batch_size, 3, 32, 32]
            beta: beta-VAE weight for KL divergence
        Returns:
            Dictionary of loss components
        """
        x_recon, mu, logvar, z = self.forward(x)
        
        # Flatten for loss computation
        x_flat = x.view(x.size(0), -1)
        x_recon_flat = x_recon.view(x_recon.size(0), -1)
        
        # Reconstruction loss (BCE)
        recon_loss = F.binary_cross_entropy(x_recon_flat, x_flat, reduction='sum') / x.size(0)
        
        # KL divergence loss (standard VAE regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # DERP distributional enforcement loss
        distributional_loss = self.random_probe.compute_distributional_loss(z)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss + self.enforcement_weight * distributional_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'distributional_loss': distributional_loss
        }


class StandardVAE(nn.Module):
    """Standard VAE for baseline comparison"""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvVAEEncoder(latent_dim)
        self.decoder = ConvVAEDecoder(latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar, z
    
    def compute_loss(self, x: torch.Tensor, beta: float = 1.0) -> Dict[str, torch.Tensor]:
        x_recon, mu, logvar, z = self.forward(x)
        
        x_flat = x.view(x.size(0), -1)
        x_recon_flat = x_recon.view(x_recon.size(0), -1)
        
        recon_loss = F.binary_cross_entropy(x_recon_flat, x_flat, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'distributional_loss': torch.tensor(0.0)  # No distributional loss for standard VAE
        }


def compute_posterior_collapse_metrics(mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics to assess posterior collapse
    
    Args:
        mu: posterior means [batch_size, latent_dim]
        logvar: posterior log variances [batch_size, latent_dim]
    Returns:
        Dictionary of collapse metrics
    """
    with torch.no_grad():
        # Convert to numpy for statistical analysis
        mu_np = mu.cpu().numpy()
        logvar_np = logvar.cpu().numpy()
        var_np = np.exp(logvar_np)
        
        # 1. KL divergence to prior (standard normal)
        kl_div = -0.5 * np.sum(1 + logvar_np - mu_np**2 - var_np) / mu.size(0)
        
        # 2. Mutual information approximation I(x,z) ≈ -KL(q(z|x)||p(z))
        mutual_info = -kl_div
        
        # 3. Activation rate (percentage of dimensions with meaningful variance)
        activation_threshold = 0.01
        active_units = np.mean(var_np > activation_threshold, axis=0)
        activation_rate = np.mean(active_units)
        
        # 4. Average posterior variance
        avg_posterior_var = np.mean(var_np)
        
        return {
            'kl_divergence': float(kl_div),
            'mutual_information': float(mutual_info),
            'activation_rate': float(activation_rate),
            'avg_posterior_variance': float(avg_posterior_var)
        }


def statistical_normality_test(samples: torch.Tensor) -> Dict[str, float]:
    """
    Perform statistical tests to verify distributional assumptions
    
    Args:
        samples: latent samples [batch_size, latent_dim]
    Returns:
        Dictionary of statistical test results
    """
    samples_np = samples.detach().cpu().numpy()
    
    results = {}
    
    # Test each dimension independently
    n_dims = samples_np.shape[1]
    shapiro_pvals = []
    ks_pvals = []
    
    for dim in range(min(n_dims, 10)):  # Test first 10 dimensions to avoid computational burden
        dim_samples = samples_np[:, dim]
        
        # Shapiro-Wilk test for normality
        if len(dim_samples) <= 5000:  # Shapiro-Wilk limitation
            _, shapiro_pval = stats.shapiro(dim_samples)
            shapiro_pvals.append(shapiro_pval)
        
        # Kolmogorov-Smirnov test against standard normal
        _, ks_pval = stats.kstest(dim_samples, 'norm', args=(0, 1))
        ks_pvals.append(ks_pval)
    
    results['shapiro_wilk_mean_pval'] = float(np.mean(shapiro_pvals)) if shapiro_pvals else 0.0
    results['ks_test_mean_pval'] = float(np.mean(ks_pvals))
    results['normality_compliance'] = float(np.mean([p > 0.05 for p in ks_pvals]))  # Fraction passing normality
    
    return results