"""
CIFAR-10 Enhanced DERP-VAE Implementation
Adapted for real image data with improved architecture and statistical rigor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import Tuple, List, Dict, Optional
import logging


class RandomProbeTest:
    """
    Random Probe Testing for distributional enforcement using modified K-S distance
    Enhanced for real image data
    """
    def __init__(self, n_probes: int = 5, device: str = 'cpu'):
        self.n_probes = n_probes
        self.device = device
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
        """
        projections = self.project_to_1d(latent_samples)  # [n_probes, batch_size]
        
        total_loss = 0.0
        for i in range(self.n_probes):
            probe_projection = projections[i]  # [batch_size]
            ks_distance = self.modified_ks_distance(probe_projection, target_dist='normal')
            total_loss += ks_distance
        
        # Average across probes
        return total_loss / self.n_probes


class CIFARVAEEncoder(nn.Module):
    """CIFAR-10 VAE Encoder optimized for 32x32x3 images"""
    def __init__(self, latent_dim: int = 4):
        super().__init__()
        # Convolutional layers for spatial feature extraction
        self.conv_layers = nn.Sequential(
            # Input: 3x32x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 32x16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128x4x4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),  # 256x1x1
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Fully connected layers for latent parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Convolutional feature extraction
        h = self.conv_layers(x)  # [batch_size, 256, 1, 1]
        h = h.view(h.size(0), -1)  # Flatten to [batch_size, 256]
        h = self.dropout(h)
        
        # Latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class CIFARVAEDecoder(nn.Module):
    """CIFAR-10 VAE Decoder for reconstructing 32x32x3 images"""
    def __init__(self, latent_dim: int = 4):
        super().__init__()
        # Fully connected layer to expand from latent
        self.fc = nn.Linear(latent_dim, 256)
        
        # Transposed convolutional layers for spatial reconstruction
        self.deconv_layers = nn.Sequential(
            # Start from 256x1x1
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0),  # 128x4x4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 3x32x32
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
    def forward(self, z):
        h = F.relu(self.fc(z))  # [batch_size, 256]
        h = h.view(h.size(0), 256, 1, 1)  # Reshape for convolution
        x_recon = self.deconv_layers(h)  # [batch_size, 3, 32, 32]
        return x_recon


class ClassificationHead(nn.Module):
    """Classification head for multi-task learning on CIFAR-10 classes"""
    def __init__(self, latent_dim: int = 4, n_classes: int = 5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim * 4, latent_dim * 2), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, n_classes)
        )
    
    def forward(self, z):
        return self.classifier(z)


class PerceptualLoss(nn.Module):
    """Perceptual loss using simple feature extraction for CIFAR images"""
    def __init__(self):
        super().__init__()
        # Simple feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # 64x8x8
            nn.AdaptiveAvgPool2d(4)  # 64x4x4
        )
    
    def forward(self, x_real: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between real and reconstructed images"""
        features_real = self.feature_extractor(x_real).view(x_real.size(0), -1)
        features_recon = self.feature_extractor(x_recon).view(x_recon.size(0), -1)
        return F.mse_loss(features_real, features_recon)


class CIFARDERPVAE(nn.Module):
    """
    CIFAR-10 Enhanced Distribution Enforcement via Random Probe VAE
    Multi-loss framework with real image data
    """
    def __init__(self, latent_dim: int = 4, n_classes: int = 5, n_probes: int = 5, 
                 enforcement_weight: float = 1.0, classification_weight: float = 0.5, 
                 perceptual_weight: float = 0.3, device: str = 'cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.enforcement_weight = enforcement_weight
        self.classification_weight = classification_weight
        self.perceptual_weight = perceptual_weight
        
        # CIFAR-optimized VAE components
        self.encoder = CIFARVAEEncoder(latent_dim)
        self.decoder = CIFARVAEDecoder(latent_dim)
        
        # Classification head for multi-task learning
        self.classifier = ClassificationHead(latent_dim, n_classes)
        
        # Perceptual loss module
        self.perceptual_loss = PerceptualLoss()
        
        # DERP component for distributional enforcement
        self.random_probe = RandomProbeTest(n_probes=n_probes, device=device)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through CIFAR DERP-VAE
        Returns: reconstruction, class_logits, mu, logvar, z_sample
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        # Classify
        class_logits = self.classifier(z)
        
        return x_recon, class_logits, mu, logvar, z
    
    def compute_loss(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                    beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss for CIFAR data
        """
        x_recon, class_logits, mu, logvar, z = self.forward(x)
        
        # 1. Reconstruction loss (MSE for images)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # 2. KL divergence loss (standard VAE regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # 3. DERP distributional enforcement loss
        distributional_loss = self.random_probe.compute_distributional_loss(z)
        
        # 4. Classification loss (if labels provided)
        classification_loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            classification_loss = F.cross_entropy(class_logits, labels)
        
        # 5. Perceptual loss
        perceptual_loss = self.perceptual_loss(x, x_recon)
        
        # Total loss with multi-objective weighting
        total_loss = (recon_loss + 
                     beta * kl_loss + 
                     self.enforcement_weight * distributional_loss +
                     self.classification_weight * classification_loss +
                     self.perceptual_weight * perceptual_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'distributional_loss': distributional_loss,
            'classification_loss': classification_loss,
            'perceptual_loss': perceptual_loss,
            'class_logits': class_logits
        }


class CIFARStandardVAE(nn.Module):
    """Standard VAE baseline for CIFAR-10 comparison"""
    def __init__(self, latent_dim: int = 4, n_classes: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        # Same architecture as DERP-VAE for fair comparison
        self.encoder = CIFARVAEEncoder(latent_dim)
        self.decoder = CIFARVAEDecoder(latent_dim)
        self.classifier = ClassificationHead(latent_dim, n_classes)
        self.perceptual_loss = PerceptualLoss()
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        class_logits = self.classifier(z)
        
        return x_recon, class_logits, mu, logvar, z
    
    def compute_loss(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                    beta: float = 1.0, classification_weight: float = 0.5, 
                    perceptual_weight: float = 0.3) -> Dict[str, torch.Tensor]:
        x_recon, class_logits, mu, logvar, z = self.forward(x)
        
        # Standard VAE losses
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Multi-task losses
        classification_loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            classification_loss = F.cross_entropy(class_logits, labels)
        
        perceptual_loss = self.perceptual_loss(x, x_recon)
        
        total_loss = (recon_loss + 
                     beta * kl_loss + 
                     classification_weight * classification_loss +
                     perceptual_weight * perceptual_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'distributional_loss': torch.tensor(0.0),  # No distributional loss for standard VAE
            'classification_loss': classification_loss,
            'perceptual_loss': perceptual_loss,
            'class_logits': class_logits
        }


def compute_cifar_metrics(mu: torch.Tensor, logvar: torch.Tensor, 
                         class_logits: torch.Tensor, true_labels: torch.Tensor,
                         x_real: torch.Tensor, x_recon: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive metrics for CIFAR experiments
    """
    with torch.no_grad():
        # Convert to numpy for analysis
        mu_np = mu.cpu().numpy()
        logvar_np = logvar.cpu().numpy()
        var_np = np.exp(logvar_np)
        
        # Standard posterior collapse metrics
        kl_div = -0.5 * np.sum(1 + logvar_np - mu_np**2 - var_np) / mu.size(0)
        
        # Activation rate (adapted for 4D latent space)
        activation_threshold = 0.01  # Threshold for active units
        active_units = np.mean(var_np > activation_threshold, axis=0)
        activation_rate = np.mean(active_units)
        
        avg_posterior_var = np.mean(var_np)
        
        # Classification accuracy
        class_predictions = torch.argmax(class_logits, dim=1)
        classification_accuracy = (class_predictions == true_labels).float().mean().item()
        
        # Image reconstruction quality (PSNR)
        mse = F.mse_loss(x_recon, x_real, reduction='mean')
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        return {
            'kl_divergence': float(kl_div),
            'activation_rate': float(activation_rate),
            'avg_posterior_variance': float(avg_posterior_var),
            'classification_accuracy': float(classification_accuracy),
            'psnr': float(psnr.item()),
            'mse': float(mse.item())
        }


def cifar_statistical_test(samples: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Statistical tests adapted for CIFAR latent representations
    """
    samples_np = samples.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    results = {}
    
    # Normality testing for all 4 dimensions
    n_dims = samples_np.shape[1]
    shapiro_pvals = []
    ks_pvals = []
    
    for dim in range(n_dims):
        dim_samples = samples_np[:, dim]
        
        # Shapiro-Wilk test for normality
        if len(dim_samples) <= 5000:
            _, shapiro_pval = stats.shapiro(dim_samples)
            shapiro_pvals.append(shapiro_pval)
        
        # Kolmogorov-Smirnov test against standard normal
        _, ks_pval = stats.kstest(dim_samples, 'norm', args=(0, 1))
        ks_pvals.append(ks_pval)
    
    results['shapiro_wilk_mean_pval'] = float(np.mean(shapiro_pvals)) if shapiro_pvals else 0.0
    results['ks_test_mean_pval'] = float(np.mean(ks_pvals))
    results['normality_compliance'] = float(np.mean([p > 0.05 for p in ks_pvals]))
    
    # Class separation analysis
    unique_labels = np.unique(labels_np)
    if len(unique_labels) > 1:
        # Compute between-class vs within-class variance ratio
        between_class_var = 0.0
        within_class_var = 0.0
        
        overall_mean = np.mean(samples_np, axis=0)
        
        for label in unique_labels:
            mask = labels_np == label
            class_samples = samples_np[mask]
            class_mean = np.mean(class_samples, axis=0)
            
            # Between-class variance
            between_class_var += np.sum((class_mean - overall_mean)**2) * np.sum(mask)
            
            # Within-class variance
            within_class_var += np.sum((class_samples - class_mean)**2)
        
        between_class_var /= len(samples_np)
        within_class_var /= len(samples_np)
        
        # Fisher's ratio
        fisher_ratio = np.mean(between_class_var / (within_class_var + 1e-8))
        results['class_separation_ratio'] = float(fisher_ratio)
    else:
        results['class_separation_ratio'] = 0.0
    
    return results