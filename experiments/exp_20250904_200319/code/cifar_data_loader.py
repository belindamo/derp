"""
CIFAR-10 Data Loader for DERP-VAE Experiments
Loads 2000 samples from CIFAR-10 with 5 class subset for real data validation
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import os
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset

class CIFAR10Subset:
    """
    CIFAR-10 dataset loader with subset functionality for efficient experimentation
    """
    
    def __init__(self, data_dir: str = '../../../data/vision', n_samples: int = 2000, 
                 n_classes: int = 5, seed: int = 42):
        self.data_dir = data_dir
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.seed = seed
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load raw CIFAR-10 data
        self.train_data, self.train_labels, self.test_data, self.test_labels = self._load_cifar10_raw()
        
        # Create subset with balanced classes
        self.subset_data, self.subset_labels = self._create_balanced_subset()
        
        print(f"Loaded CIFAR-10 subset: {len(self.subset_data)} samples, {n_classes} classes")
    
    def _load_cifar10_raw(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load raw CIFAR-10 data from downloaded files"""
        cifar_dir = os.path.join(self.data_dir, 'cifar-10-batches-py')
        
        # Load training batches
        train_data = []
        train_labels = []
        
        for i in range(1, 6):
            batch_file = os.path.join(cifar_dir, f'data_batch_{i}')
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                train_data.append(batch[b'data'])
                train_labels.extend(batch[b'labels'])
        
        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.array(train_labels)
        
        # Load test batch
        test_file = os.path.join(cifar_dir, 'test_batch')
        with open(test_file, 'rb') as f:
            test_batch = pickle.load(f, encoding='bytes')
            test_data = test_batch[b'data']
            test_labels = np.array(test_batch[b'labels'])
        
        # Reshape data from (N, 3072) to (N, 32, 32, 3)
        train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        return train_data, train_labels, test_data, test_labels
    
    def _create_balanced_subset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create balanced subset with n_classes and n_samples total"""
        
        # Select first n_classes from CIFAR-10 (0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer)
        selected_classes = list(range(self.n_classes))
        samples_per_class = self.n_samples // self.n_classes
        
        subset_data = []
        subset_labels = []
        
        # Combine train and test data for larger pool
        all_data = np.concatenate([self.train_data, self.test_data], axis=0)
        all_labels = np.concatenate([self.train_labels, self.test_labels], axis=0)
        
        for class_idx in selected_classes:
            class_mask = all_labels == class_idx
            class_data = all_data[class_mask]
            
            # Randomly sample from this class
            if len(class_data) >= samples_per_class:
                selected_indices = np.random.choice(len(class_data), samples_per_class, replace=False)
                selected_data = class_data[selected_indices]
            else:
                # If not enough samples, use all and warn
                selected_data = class_data
                print(f"Warning: Only {len(class_data)} samples available for class {class_idx}")
            
            subset_data.append(selected_data)
            subset_labels.extend([class_idx] * len(selected_data))
        
        subset_data = np.concatenate(subset_data, axis=0)
        subset_labels = np.array(subset_labels)
        
        # Shuffle the final dataset
        indices = np.random.permutation(len(subset_data))
        subset_data = subset_data[indices]
        subset_labels = subset_labels[indices]
        
        return subset_data, subset_labels
    
    def get_data_tensors(self, normalize: bool = True, flatten: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to PyTorch tensors with optional normalization and flattening
        
        Args:
            normalize: Normalize pixel values to [0, 1]
            flatten: Flatten images to vectors (for non-convolutional VAE)
        Returns:
            data tensor, labels tensor
        """
        data = self.subset_data.astype(np.float32)
        labels = self.subset_labels.astype(np.int64)
        
        if normalize:
            data = data / 255.0
        
        # Convert to torch tensors
        if flatten:
            # Flatten to (N, 3072) for fully connected VAE
            data_tensor = torch.from_numpy(data.reshape(len(data), -1))
        else:
            # Keep spatial dimensions (N, H, W, C) -> (N, C, H, W)
            data_tensor = torch.from_numpy(data.transpose(0, 3, 1, 2))
        
        labels_tensor = torch.from_numpy(labels)
        
        return data_tensor, labels_tensor
    
    def get_train_val_split(self, val_split: float = 0.2, flatten: bool = False) -> Tuple[DataLoader, DataLoader]:
        """
        Create train/validation DataLoaders
        
        Args:
            val_split: Fraction for validation set
            flatten: Flatten images for fully connected networks
        Returns:
            train_loader, val_loader
        """
        data_tensor, labels_tensor = self.get_data_tensors(normalize=True, flatten=flatten)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
        
        # Train/val split
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        # Create data loaders
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def get_class_info(self) -> dict:
        """Get information about classes in the subset"""
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        unique_labels, counts = np.unique(self.subset_labels, return_counts=True)
        class_distribution = {
            class_names[label]: count 
            for label, count in zip(unique_labels, counts)
        }
        
        return {
            'n_classes': self.n_classes,
            'n_samples': len(self.subset_data),
            'class_names': [class_names[i] for i in range(self.n_classes)],
            'class_distribution': class_distribution,
            'image_shape': self.subset_data.shape[1:],
            'data_range': [self.subset_data.min(), self.subset_data.max()]
        }


def test_cifar_loader():
    """Test the CIFAR-10 loader functionality"""
    print("Testing CIFAR-10 Subset Loader...")
    
    # Initialize loader
    loader = CIFAR10Subset(n_samples=100, n_classes=3, seed=42)  # Small test
    
    # Get class info
    info = loader.get_class_info()
    print(f"Dataset info: {info}")
    
    # Test tensor conversion
    data_tensor, labels_tensor = loader.get_data_tensors(normalize=True, flatten=True)
    print(f"Flattened data shape: {data_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Data range: [{data_tensor.min():.3f}, {data_tensor.max():.3f}]")
    
    # Test data loaders
    train_loader, val_loader = loader.get_train_val_split(val_split=0.2, flatten=True)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch_data, batch_labels in train_loader:
        print(f"Batch shape: {batch_data.shape}, Labels: {batch_labels.shape}")
        print(f"Label distribution in batch: {torch.bincount(batch_labels)}")
        break
    
    print("CIFAR-10 loader test completed successfully! ✅")


if __name__ == "__main__":
    test_cifar_loader()