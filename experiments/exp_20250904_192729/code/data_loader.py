"""
Data loading utilities for DERP-VAE experiments
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

class CIFAR10Dataset(Dataset):
    """Custom CIFAR-10 dataset loader for VAE experiments"""
    
    def __init__(self, data_path: str, train: bool = True, transform=None, normalize: bool = True):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize = normalize
        
        if train:
            self.data, self.targets = self._load_train_data()
        else:
            self.data, self.targets = self._load_test_data()
    
    def _load_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load CIFAR-10 training data"""
        data_list = []
        targets_list = []
        
        for i in range(1, 6):
            batch_path = self.data_path / f'data_batch_{i}'
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data_list.append(batch[b'data'])
                targets_list.extend(batch[b'labels'])
        
        data = np.concatenate(data_list, axis=0)
        targets = np.array(targets_list)
        
        # Reshape to (N, H, W, C) and normalize
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        return data, targets
    
    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load CIFAR-10 test data"""
        test_path = self.data_path / 'test_batch'
        with open(test_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']
            targets = np.array(batch[b'labels'])
        
        # Reshape to (N, H, W, C)
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        return data, targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return image, target


def get_cifar10_dataloaders(data_path: str, batch_size: int = 64, 
                           num_workers: int = 2, subset_size: int = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 data loaders for VAE training
    
    Args:
        data_path: Path to CIFAR-10 data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        subset_size: Optional subset size for faster experimentation
    
    Returns:
        train_loader, test_loader
    """
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor() if isinstance(transforms.ToTensor(), type) else lambda x: x,
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Create datasets
    train_dataset = CIFAR10Dataset(data_path, train=True, transform=None)
    test_dataset = CIFAR10Dataset(data_path, train=False, transform=None)
    
    # Create subset if specified (for faster experimentation)
    if subset_size is not None:
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        test_indices = torch.randperm(len(test_dataset))[:subset_size//5]  # 20% for test
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def load_synthetic_gaussian_mixtures(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load synthetic Gaussian mixture data for testing
    
    Args:
        data_path: Path to synthetic data directory
    Returns:
        data, labels
    """
    data_path = Path(data_path)
    
    data = np.load(data_path / 'gaussian_mixture_data.npy')
    labels = np.load(data_path / 'gaussian_mixture_labels.npy')
    
    return data, labels


def create_synthetic_dataset(n_samples: int = 1000, n_dims: int = 64, n_components: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic high-dimensional dataset for testing distributional assumptions
    
    Args:
        n_samples: Number of samples to generate
        n_dims: Dimensionality of the data
        n_components: Number of Gaussian mixture components
    
    Returns:
        data: [n_samples, n_dims]
        labels: [n_samples]
    """
    np.random.seed(42)
    
    data = []
    labels = []
    
    samples_per_component = n_samples // n_components
    
    for i in range(n_components):
        # Generate random mean and covariance for each component
        mean = np.random.randn(n_dims) * 2
        
        # Generate random covariance matrix
        A = np.random.randn(n_dims, n_dims)
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
    
    return torch.from_numpy(data).float(), torch.from_numpy(labels).long()


def get_data_statistics(dataloader: DataLoader) -> Dict[str, float]:
    """
    Compute basic statistics of the dataset
    
    Args:
        dataloader: PyTorch DataLoader
    Returns:
        Dictionary of dataset statistics
    """
    all_data = []
    
    for batch, _ in dataloader:
        batch_flat = batch.view(batch.size(0), -1)
        all_data.append(batch_flat)
    
    all_data = torch.cat(all_data, dim=0)
    
    return {
        'mean': float(torch.mean(all_data)),
        'std': float(torch.std(all_data)),
        'min': float(torch.min(all_data)),
        'max': float(torch.max(all_data)),
        'num_samples': len(all_data),
        'num_features': all_data.shape[1]
    }