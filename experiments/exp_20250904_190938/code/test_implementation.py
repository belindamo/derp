"""
Test implementation with minimal experiment to verify functionality
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to Python path
sys.path.append('.')

from derp_vae import DERP_VAE, StandardVAE, RandomProbeTest
from data_loader import create_synthetic_dataset

def test_random_probe():
    """Test random probe functionality"""
    print("Testing Random Probe Implementation...")
    
    # Create synthetic high-dimensional data
    latent_samples = torch.randn(100, 64)  # 100 samples, 64 dimensions
    
    # Test random probe
    probe = RandomProbeTest(n_probes=5, device='cpu')
    distributional_loss = probe.compute_distributional_loss(latent_samples)
    
    print(f"Random probe distributional loss: {distributional_loss.item():.4f}")
    assert distributional_loss.item() >= 0, "Distributional loss should be non-negative"
    print("✓ Random Probe test passed")

def test_models():
    """Test DERP-VAE and Standard VAE models"""
    print("\nTesting Model Implementations...")
    
    batch_size = 32
    input_dim = 3072  # 32x32x3 CIFAR-10
    latent_dim = 64
    
    # Create random data
    x = torch.rand(batch_size, 32, 32, 3)
    
    # Test Standard VAE
    standard_vae = StandardVAE(input_dim=input_dim, latent_dim=latent_dim)
    standard_loss = standard_vae.compute_loss(x, beta=1.0)
    print(f"Standard VAE loss components: {standard_loss}")
    assert 'total_loss' in standard_loss, "Standard VAE should return total_loss"
    print("✓ Standard VAE test passed")
    
    # Test DERP-VAE
    derp_vae = DERP_VAE(input_dim=input_dim, latent_dim=latent_dim, n_probes=3, device='cpu')
    derp_loss = derp_vae.compute_loss(x, beta=1.0)
    print(f"DERP-VAE loss components: {derp_loss}")
    assert 'distributional_loss' in derp_loss, "DERP-VAE should return distributional_loss"
    print("✓ DERP-VAE test passed")

def test_data_loader():
    """Test data loading functionality"""
    print("\nTesting Data Loader...")
    
    # Test synthetic data creation
    data, labels = create_synthetic_dataset(n_samples=100, n_dims=64)
    print(f"Synthetic data shape: {data.shape}, labels shape: {labels.shape}")
    assert data.shape[1] == 64, f"Unexpected data shape: {data.shape}"
    print("✓ Synthetic data loader test passed")

def mini_experiment():
    """Run a minimal experiment to verify end-to-end functionality"""
    print("\nRunning Mini Experiment...")
    
    # Create synthetic data (normalize to [0,1] for BCE loss)
    data, _ = create_synthetic_dataset(n_samples=200, n_dims=128)
    data = torch.sigmoid(data)  # Normalize to [0,1]
    
    # Create simple dataset
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(data.view(data.size(0), -1), torch.zeros(data.size(0)))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test models
    configs = [
        ('standard', {'input_dim': 128, 'latent_dim': 16}),
        ('derp', {'input_dim': 128, 'latent_dim': 16, 'n_probes': 3, 'enforcement_weight': 1.0, 'device': 'cpu'})
    ]
    
    results = {}
    
    for model_type, config in configs:
        print(f"Testing {model_type} model...")
        
        if model_type == 'standard':
            model = StandardVAE(**config)
        else:
            model = DERP_VAE(**config)
        
        # Forward pass on one batch
        for batch, _ in dataloader:
            loss_dict = model.compute_loss(batch)
            results[model_type] = loss_dict
            print(f"{model_type} loss: {loss_dict['total_loss'].item():.4f}")
            break
    
    print("✓ Mini experiment completed successfully")
    return results

if __name__ == "__main__":
    print("Running Implementation Tests...")
    
    try:
        test_random_probe()
        test_models()
        test_data_loader()
        results = mini_experiment()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✓")
        print("Implementation is ready for full experiment")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)