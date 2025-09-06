"""Quick test for CelebA data loading"""
import sys
sys.path.append('.')

from data_loader import get_celeba_dataloaders
import torch

print("Testing CelebA loading with small subset...")

try:
    # Test with only 100 samples for quick validation
    train_loader, test_loader = get_celeba_dataloaders(
        batch_size=16,
        image_size=64,
        download=True,  # Will download if not present
        data_dir='../../../data/vision',
        num_samples=100  # Small subset for testing
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Get one batch
    for batch_data, batch_labels in train_loader:
        print(f"\nBatch info:")
        print(f"  Data shape: {batch_data.shape}")  # Should be [16, 3, 64, 64]
        print(f"  Labels shape: {batch_labels.shape}")  # Should be [16]
        print(f"  Data range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
        print(f"  Labels (first 10): {batch_labels[:10].tolist()}")
        print(f"  Unique labels: {torch.unique(batch_labels).tolist()}")
        
        # Test flattening
        flat_data = batch_data.view(batch_data.size(0), -1)
        print(f"  Flattened shape: {flat_data.shape}")  # Should be [16, 12288]
        break
        
    print("\n✅ CelebA loading successful!")
    print(f"Ready for experiments with {64*64*3} dimensional input")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()