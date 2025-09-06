"""Test CIFAR loading"""
import sys
sys.path.append('.')

from data_loader import get_cifar10_dataloaders
import torch

print("Testing CIFAR-10 loading...")

try:
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=128,
        normalize=False,
        download=False,  # Don't download, use existing
        data_dir='../../../data/vision'
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Get one batch
    for batch_data, batch_labels in train_loader:
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Data range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
        print(f"Labels: {batch_labels[:10]}")
        break
        
    print("\nCIFAR-10 loading successful!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()