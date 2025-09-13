"""
Enhanced data loading utilities for DERP-VAE experiments with labels
Now supports CIFAR-10 dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GaussianMixtureDataset(Dataset):
    """Dataset for synthetic Gaussian mixture data with labels"""
    
    def __init__(self, data_path: str, normalize: bool = True):
        self.data_path = Path(data_path)
        self.normalize = normalize
        
        # Load pre-generated synthetic data
        self.data, self.labels = self._load_synthetic_data()
        
        if self.normalize:
            self.data = self._normalize_data(self.data)
    
    def _load_synthetic_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load synthetic Gaussian mixture data and labels"""
        try:
            # Path to data files relative to experiment directory
            data_file = self.data_path / 'synthetic' / 'gaussian_mixture_data.npy'
            labels_file = self.data_path / 'synthetic' / 'gaussian_mixture_labels.npy'
            
            data = np.load(data_file)
            labels = np.load(labels_file)
            
            logger.info(f"Loaded synthetic data: {data.shape} samples, {labels.shape} labels")
            logger.info(f"Label distribution: {np.bincount(labels)}")
            
            return torch.from_numpy(data).float(), torch.from_numpy(labels).long()
            
        except FileNotFoundError as e:
            logger.warning(f"Pre-generated data not found: {e}")
            logger.info("Generating new synthetic dataset...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples: int = 10000, n_dims: int = 256, 
                                n_components: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic Gaussian mixture data if pre-generated data is not available"""
        np.random.seed(42)
        
        data = []
        labels = []
        
        samples_per_component = n_samples // n_components
        
        for i in range(n_components):
            # Generate random mean and covariance for each component
            mean = np.random.randn(n_dims) * 2
            
            # Generate random covariance matrix
            A = np.random.randn(n_dims, n_dims // 4)  # Reduced rank for stability
            cov = np.dot(A, A.T) + 0.1 * np.eye(n_dims)  # Ensure positive definite
            
            # Sample from multivariate Gaussian
            component_data = np.random.multivariate_normal(mean, cov, samples_per_component)
            component_labels = np.full(samples_per_component, i)
            
            data.append(component_data)
            labels.extend(component_labels)
        
        data = np.vstack(data)
        labels = np.array(labels)
        
        # Shuffle the data
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        logger.info(f"Generated synthetic data: {data.shape} samples, {labels.shape} labels")
        
        return torch.from_numpy(data).float(), torch.from_numpy(labels).long()
    
    def _normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data to [0,1] range for BCE loss"""
        # Apply sigmoid to map to [0,1] range
        return torch.sigmoid(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dataloaders(data_path: str, batch_size: int = 64, 
                           train_split: float = 0.8, subset_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create enhanced data loaders with labels for multi-task VAE training
    
    Args:
        data_path: Path to data directory (should contain synthetic/ subfolder)
        batch_size: Batch size for training
        train_split: Fraction of data to use for training
        subset_size: Optional subset size for faster experimentation
    
    Returns:
        train_loader, test_loader
    """
    
    # Create dataset
    dataset = GaussianMixtureDataset(data_path, normalize=True)
    
    # Create subset if specified
    if subset_size is not None:
        indices = torch.randperm(len(dataset))[:subset_size]
        subset_data = dataset.data[indices]
        subset_labels = dataset.labels[indices]
        dataset = TensorDataset(subset_data, subset_labels)
    
    # Train/test split
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    return train_loader, test_loader


def analyze_dataset_properties(dataloader: DataLoader) -> Dict[str, float]:
    """
    Analyze dataset properties for experimental validation
    
    Args:
        dataloader: PyTorch DataLoader
    Returns:
        Dictionary of dataset statistics
    """
    all_data = []
    all_labels = []
    
    for batch_data, batch_labels in dataloader:
        batch_flat = batch_data.view(batch_data.size(0), -1)
        all_data.append(batch_flat)
        all_labels.append(batch_labels)
    
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Basic statistics
    stats = {
        'num_samples': len(all_data),
        'num_features': all_data.shape[1],
        'data_mean': float(torch.mean(all_data)),
        'data_std': float(torch.std(all_data)),
        'data_min': float(torch.min(all_data)),
        'data_max': float(torch.max(all_data)),
        'num_classes': int(torch.max(all_labels) + 1),
    }
    
    # Class distribution
    for i in range(stats['num_classes']):
        class_count = torch.sum(all_labels == i).item()
        stats[f'class_{i}_count'] = class_count
        stats[f'class_{i}_fraction'] = class_count / len(all_labels)
    
    return stats


def verify_data_quality(data: torch.Tensor, labels: torch.Tensor) -> Dict[str, bool]:
    """
    Verify data quality for experimental validity
    
    Args:
        data: Input data tensor
        labels: Label tensor
    Returns:
        Dictionary of quality checks
    """
    checks = {}
    
    # Check for NaN or inf values
    checks['no_nan_data'] = not torch.isnan(data).any().item()
    checks['no_inf_data'] = not torch.isinf(data).any().item()
    checks['no_nan_labels'] = not torch.isnan(labels.float()).any().item()
    
    # Check data range
    checks['data_in_unit_range'] = (data.min() >= 0.0) and (data.max() <= 1.0)
    
    # Check label validity
    checks['labels_non_negative'] = (labels >= 0).all().item()
    max_label = labels.max().item()
    min_label = labels.min().item()
    checks['labels_contiguous'] = (max_label - min_label + 1) == len(torch.unique(labels))
    
    # Check class balance (no class should have less than 5% of data)
    unique_labels, counts = torch.unique(labels, return_counts=True)
    min_class_fraction = counts.min().float() / len(labels)
    checks['reasonable_class_balance'] = min_class_fraction >= 0.05
    
    return checks


def get_celeba_dataloaders(batch_size: int = 64, image_size: int = 64,
                          download: bool = True, data_dir: str = './data',
                          num_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CelebA data loaders for DERP-VAE experiments with proper train/val/test splits
    
    Args:
        batch_size: Batch size for training
        image_size: Size to resize images to (will be image_size x image_size)
        download: Whether to download if not present
        data_dir: Directory to store/load CelebA data
        num_samples: Optional limit on number of samples (for testing)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transforms for CelebA
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # Converts to [0,1] range
    ])
    
    # Process labels to use 'Smiling' attribute as binary classification
    class CelebABinaryWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset, attr_idx=31):
            self.dataset = dataset
            self.attr_idx = attr_idx  # 31 is 'Smiling'
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, attrs = self.dataset[idx]
            # Convert multi-attribute to single binary label
            label = attrs[self.attr_idx].item()
            return img, label
    
    # Load official CelebA splits
    train_dataset = torchvision.datasets.CelebA(
        root=data_dir,
        split='train',  # Official train split
        target_type='attr',
        transform=transform,
        download=download
    )
    train_dataset = CelebABinaryWrapper(train_dataset)
    
    valid_dataset = torchvision.datasets.CelebA(
        root=data_dir,
        split='valid',  # Official validation split
        target_type='attr',
        transform=transform,
        download=False  # Already downloaded
    )
    valid_dataset = CelebABinaryWrapper(valid_dataset)
    
    test_dataset = torchvision.datasets.CelebA(
        root=data_dir,
        split='test',  # Official test split
        target_type='attr',
        transform=transform,
        download=False  # Already downloaded
    )
    test_dataset = CelebABinaryWrapper(test_dataset)
    
    # Limit samples if specified (for debugging)
    if num_samples is not None:
        train_samples = min(num_samples, len(train_dataset))
        val_samples = min(num_samples // 5, len(valid_dataset))  # Smaller val set
        test_samples = min(num_samples // 4, len(test_dataset))  # Smaller test set
        
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_samples))
        valid_dataset = torch.utils.data.Subset(valid_dataset, range(val_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_samples))
    
    # Create data loaders with appropriate settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for performance
        drop_last=True,   # Drop last incomplete batch for stable batch norm
        pin_memory=True
    )
    
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )
    
    logger.info(f"CelebA loaded with official splits:")
    logger.info(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    logger.info(f"  Valid: {len(valid_dataset)} samples ({len(val_loader)} batches)")
    logger.info(f"  Test: {len(test_dataset)} samples ({len(test_loader)} batches)")
    logger.info(f"  Image size: {image_size}x{image_size}, Total dims: {image_size*image_size*3}")
    
    return train_loader, val_loader, test_loader


def get_cifar10_dataloaders(batch_size: int = 64, normalize: bool = True, 
                          download: bool = True, data_dir: str = './data') -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 data loaders for DERP-VAE experiments
    
    Args:
        batch_size: Batch size for training
        normalize: Whether to normalize to [0,1] range
        download: Whether to download if not present
        data_dir: Directory to store/load CIFAR-10 data
    
    Returns:
        train_loader, test_loader
    """
    # Define transforms
    transform_list = []
    transform_list.append(transforms.ToTensor())  # Converts to [0,1] range automatically
    
    if normalize:
        # CIFAR-10 normalization values
        transform_list.append(transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ))
    
    transform = transforms.Compose(transform_list)
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )
    
    # Create data loaders (num_workers=0 to avoid multiprocessing issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        drop_last=True  # Ensure consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        drop_last=False
    )
    
    logger.info(f"CIFAR-10 loaded: {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    return train_loader, test_loader


def create_controlled_experiment_data(n_samples: int = 2000, n_dims: int = 256, 
                                    latent_dim: int = 4, n_components: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create controlled synthetic dataset for specific experimental conditions
    
    Args:
        n_samples: Number of samples
        n_dims: Input dimensionality
        latent_dim: True latent dimensionality (for controlled experiments)
        n_components: Number of mixture components
    
    Returns:
        data: [n_samples, n_dims]
        labels: [n_samples]
    """
    np.random.seed(42)  # Reproducible results
    
    # Generate true latent factors
    samples_per_component = n_samples // n_components
    latent_factors = []
    labels = []
    
    for i in range(n_components):
        # Each component has different mean in latent space
        latent_mean = np.zeros(latent_dim)
        latent_mean[i % latent_dim] = 3.0  # Separate components in latent space
        
        component_latents = np.random.multivariate_normal(
            latent_mean, 0.5 * np.eye(latent_dim), samples_per_component
        )
        component_labels = np.full(samples_per_component, i)
        
        latent_factors.append(component_latents)
        labels.extend(component_labels)
    
    latent_factors = np.vstack(latent_factors)
    labels = np.array(labels)
    
    # Generate random linear mapping from latent to observed space
    W = np.random.randn(n_dims, latent_dim) * 0.5
    b = np.random.randn(n_dims) * 0.1
    
    # Generate observed data
    observed_data = np.dot(latent_factors, W.T) + b
    
    # Add noise
    noise = np.random.randn(*observed_data.shape) * 0.1
    observed_data += noise
    
    # Shuffle
    indices = np.random.permutation(len(observed_data))
    observed_data = observed_data[indices]
    labels = labels[indices]
    
    # Convert to tensors and normalize
    data = torch.from_numpy(observed_data).float()
    data = torch.sigmoid(data)  # Normalize to [0,1]
    labels = torch.from_numpy(labels).long()
    
    logger.info(f"Created controlled experiment data: {data.shape}, latent_dim={latent_dim}")
    
    return data, labels