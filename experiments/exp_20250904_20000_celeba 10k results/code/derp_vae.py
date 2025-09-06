"""
Enhanced Distribution Enforcement via Random Probe (DERP) VAE Implementation
Testing Hypothesis H1 & H2 with Multi-Loss Framework and Reduced Hidden Dimensions

Key Enhancements:
- Reduced latent dimensions (32 → 4) for more challenging collapse conditions
- Multi-loss framework: Classification + Reconstruction + Perceptual + Modified KS
- Label-aware training with synthetic Gaussian mixture labels
- Enhanced statistical evaluation framework
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
    
    def compute_distributional_loss(self, latent_samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distributional enforcement loss using random probes
        
        Args:
            latent_samples: [batch_size, latent_dim]
        Returns:
            distributional_loss: scalar tensor
            avg_ks_distance: scalar tensor (for tracking)
        """
        projections = self.project_to_1d(latent_samples)  # [n_probes, batch_size]
        
        total_loss = 0.0
        ks_distances = []
        for i in range(self.n_probes):
            probe_projection = projections[i]  # [batch_size]
            ks_distance = self.modified_ks_distance(probe_projection, target_dist='normal')
            total_loss += ks_distance
            ks_distances.append(ks_distance.item())
        
        # Average across probes
        avg_loss = total_loss / self.n_probes
        avg_ks = sum(ks_distances) / len(ks_distances)
        return avg_loss, torch.tensor(avg_ks)


class VAEEncoder(nn.Module):
    """Enhanced VAE Encoder with reduced hidden dimensions"""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, latent_dim: int = 4):
        super().__init__()
        # Reduced capacity architecture for more challenging conditions
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Enhanced VAE Decoder with reduced hidden dimensions"""
    def __init__(self, latent_dim: int = 4, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        # Symmetric architecture to encoder
        self.fc1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class ClassificationHead(nn.Module):
    """Classification head for multi-task learning"""
    def __init__(self, latent_dim: int = 4, n_classes: int = 5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim * 2, n_classes)
        )
    
    def forward(self, z):
        return self.classifier(z)


class PerceptualLoss(nn.Module):
    """Simple perceptual loss using feature matching"""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        # Simple feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32)  # Feature dimension
        )
    
    def forward(self, x_real: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between real and reconstructed data"""
        features_real = self.feature_extractor(x_real)
        features_recon = self.feature_extractor(x_recon)
        return F.mse_loss(features_real, features_recon)


class DERP_VAE(nn.Module):
    """
    Enhanced Distribution Enforcement via Random Probe VAE
    Implements multi-loss framework with classification, reconstruction, perceptual, and KS losses
    """
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, latent_dim: int = 4, 
                 n_classes: int = 5, n_probes: int = 5, enforcement_weight: float = 1.0, 
                 classification_weight: float = 0.5, perceptual_weight: float = 0.3, 
                 device: str = 'cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.enforcement_weight = enforcement_weight
        self.classification_weight = classification_weight
        self.perceptual_weight = perceptual_weight
        
        # Enhanced VAE components with reduced capacity
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        
        # Classification head for multi-task learning
        self.classifier = ClassificationHead(latent_dim, n_classes)
        
        # Perceptual loss module
        self.perceptual_loss = PerceptualLoss(input_dim, hidden_dim)
        
        # DERP component for distributional enforcement
        self.random_probe = RandomProbeTest(n_probes=n_probes, device=device)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Enhanced DERP-VAE
        Returns: reconstruction, class_logits, mu, logvar, z_sample
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
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
        Compute multi-objective loss with all components
        
        Args:
            x: input data [batch_size, ...]
            labels: class labels [batch_size] (optional)
            beta: beta-VAE weight for KL divergence
        Returns:
            Dictionary of loss components
        """
        x_recon, class_logits, mu, logvar, z = self.forward(x)
        
        # Flatten for loss computation
        x_flat = x.view(x.size(0), -1)
        x_recon_flat = x_recon.view(x_recon.size(0), -1)
        
        # 1. Reconstruction loss (BCE)
        recon_loss = F.binary_cross_entropy(x_recon_flat, x_flat, reduction='sum') / x.size(0)
        
        # 2. KL divergence loss (standard VAE regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # 3. DERP distributional enforcement loss (Modified KS)
        distributional_loss, ks_distance = self.random_probe.compute_distributional_loss(z)
        
        # 4. Classification loss (if labels provided)
        classification_loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            classification_loss = F.cross_entropy(class_logits, labels)
        
        # 5. Perceptual loss
        perceptual_loss = self.perceptual_loss(x_flat, x_recon_flat)
        
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
            'ks_distance': ks_distance,  # Add KS distance for tracking
            'classification_loss': classification_loss,
            'perceptual_loss': perceptual_loss,
            'class_logits': class_logits
        }


class EnhancedStandardVAE(nn.Module):
    """Enhanced Standard VAE for baseline comparison with same architecture"""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, latent_dim: int = 4, n_classes: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        # Same architecture as Enhanced DERP-VAE for fair comparison
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.classifier = ClassificationHead(latent_dim, n_classes)
        self.perceptual_loss = PerceptualLoss(input_dim, hidden_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        class_logits = self.classifier(z)
        
        return x_recon, class_logits, mu, logvar, z
    
    def compute_loss(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                    beta: float = 1.0, classification_weight: float = 0.5, 
                    perceptual_weight: float = 0.3) -> Dict[str, torch.Tensor]:
        x_recon, class_logits, mu, logvar, z = self.forward(x)
        
        x_flat = x.view(x.size(0), -1)
        x_recon_flat = x_recon.view(x_recon.size(0), -1)
        
        # Standard VAE losses
        recon_loss = F.binary_cross_entropy(x_recon_flat, x_flat, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Multi-task losses (same as Enhanced DERP-VAE except no distributional loss)
        classification_loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            classification_loss = F.cross_entropy(class_logits, labels)
        
        perceptual_loss = self.perceptual_loss(x_flat, x_recon_flat)
        
        total_loss = (recon_loss + 
                     beta * kl_loss + 
                     classification_weight * classification_loss +
                     perceptual_weight * perceptual_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'distributional_loss': torch.tensor(0.0),  # No distributional loss for standard VAE
            'ks_distance': torch.tensor(0.0),  # No KS distance for standard VAE
            'classification_loss': classification_loss,
            'perceptual_loss': perceptual_loss,
            'class_logits': class_logits
        }


def compute_enhanced_metrics(mu: torch.Tensor, logvar: torch.Tensor, 
                           class_logits: torch.Tensor, true_labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute enhanced metrics including classification accuracy
    
    Args:
        mu: posterior means [batch_size, latent_dim]
        logvar: posterior log variances [batch_size, latent_dim]
        class_logits: classification logits [batch_size, n_classes]
        true_labels: ground truth labels [batch_size]
    Returns:
        Dictionary of enhanced metrics
    """
    with torch.no_grad():
        # Convert to numpy for statistical analysis
        mu_np = mu.cpu().numpy()
        logvar_np = logvar.cpu().numpy()
        var_np = np.exp(logvar_np)
        
        # Standard posterior collapse metrics
        kl_div = -0.5 * np.sum(1 + logvar_np - mu_np**2 - var_np) / mu.size(0)
        mutual_info = -kl_div
        
        # Activation rate (with lower threshold for 4D latent space)
        activation_threshold = 0.005  # Reduced for 4D space
        active_units = np.mean(var_np > activation_threshold, axis=0)
        activation_rate = np.mean(active_units)
        
        avg_posterior_var = np.mean(var_np)
        
        # Classification accuracy
        class_predictions = torch.argmax(class_logits, dim=1)
        classification_accuracy = (class_predictions == true_labels).float().mean().item()
        
        return {
            'kl_divergence': float(kl_div),
            'mutual_information': float(mutual_info),
            'activation_rate': float(activation_rate),
            'avg_posterior_variance': float(avg_posterior_var),
            'classification_accuracy': float(classification_accuracy)
        }


def enhanced_statistical_test(samples: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Enhanced statistical tests including label-aware analysis
    
    Args:
        samples: latent samples [batch_size, latent_dim]
        labels: ground truth labels [batch_size]
    Returns:
        Dictionary of statistical test results
    """
    samples_np = samples.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    results = {}
    
    # Standard normality testing
    n_dims = min(samples_np.shape[1], 4)  # Test all dimensions for 4D space
    shapiro_pvals = []
    ks_pvals = []
    
    for dim in range(n_dims):
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
    results['normality_compliance'] = float(np.mean([p > 0.05 for p in ks_pvals]))
    
    # Label-aware analysis: Check if latent space preserves class separation
    unique_labels = np.unique(labels_np)
    if len(unique_labels) > 1:
        # Compute between-class vs within-class variance ratio
        total_var = np.var(samples_np, axis=0)
        between_class_var = 0.0
        within_class_var = 0.0
        
        for label in unique_labels:
            mask = labels_np == label
            class_samples = samples_np[mask]
            class_mean = np.mean(class_samples, axis=0)
            
            # Between-class variance (how far each class mean is from overall mean)
            overall_mean = np.mean(samples_np, axis=0)
            between_class_var += np.sum((class_mean - overall_mean)**2) * np.sum(mask)
            
            # Within-class variance
            within_class_var += np.sum((class_samples - class_mean)**2)
        
        between_class_var /= len(samples_np)
        within_class_var /= len(samples_np)
        
        # Fisher's ratio: between-class variance / within-class variance
        fisher_ratio = np.mean(between_class_var / (within_class_var + 1e-8))
        results['class_separation_ratio'] = float(fisher_ratio)
    else:
        results['class_separation_ratio'] = 0.0
    
    return results