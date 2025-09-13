#!/usr/bin/env python3
"""Test script to verify the CelebA experiment runs with fixes"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from derp_vae import DERP_VAE, EnhancedStandardVAE as StandardVAE
from data_loader import get_celeba_dataloaders

def test_quick_run():
    """Test that the models initialize and run without errors"""
    print("Testing CelebA experiment setup with fixes...")
    
    # Test data loading with official splits
    print("\n1. Testing data loading with official splits...")
    train_loader, val_loader, test_loader = get_celeba_dataloaders(
        batch_size=32,
        image_size=64,
        download=True,
        data_dir='../../../data/vision',
        num_samples=100  # Small sample for testing
    )
    print("✅ Data loaders created successfully with train/val/test splits")
    
    # Test model initialization
    print("\n2. Testing model initialization...")
    base_config = {
        'input_dim': 64 * 64 * 3,
        'hidden_dim': 512,
        'latent_dim': 64,
        'n_classes': 2
    }
    
    loss_weights = {
        'classification_weight': 0.5,
        'perceptual_weight': 0.3,
        'enforcement_weight': 0.5
    }
    
    # Test StandardVAE
    model1 = StandardVAE(**base_config)
    print("✅ StandardVAE initialized successfully")
    
    # Test DERP_VAE
    model2 = DERP_VAE(**base_config, n_probes=5, **loss_weights, device='cpu')
    print("✅ DERP_VAE initialized successfully")
    
    # Test forward pass and loss computation
    print("\n3. Testing forward pass and loss computation...")
    batch = next(iter(train_loader))
    images, labels = batch
    images_flat = images.view(images.size(0), -1)
    
    # Test StandardVAE
    loss_dict1 = model1.compute_loss(images_flat, labels, beta=1.0,
                                    classification_weight=0.5,
                                    perceptual_weight=0.3)
    print(f"✅ StandardVAE loss computed: {loss_dict1['total_loss'].item():.4f}")
    
    # Test DERP_VAE
    loss_dict2 = model2.compute_loss(images_flat, labels, beta=1.0)
    print(f"✅ DERP_VAE loss computed: {loss_dict2['total_loss'].item():.4f}")
    print(f"   - KS distance: {loss_dict2['ks_distance'].item():.4f}")
    
    print("\n✅ All tests passed! The experiment should run correctly.")

if __name__ == "__main__":
    test_quick_run()
