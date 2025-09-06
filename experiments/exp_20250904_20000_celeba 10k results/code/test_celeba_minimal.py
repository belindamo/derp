"""Minimal test for CelebA - just check if it can load"""
import sys
sys.path.append('.')

print("Testing CelebA dataset availability...")

try:
    import torch
    import torchvision
    
    # Try to load CelebA info
    print("\nAttempting to load CelebA metadata...")
    
    # This will download if needed
    dataset = torchvision.datasets.CelebA(
        root='../../../data/vision',
        split='train',
        target_type='attr',
        download=False  # Set to True to download
    )
    
    print(f"✅ CelebA dataset found!")
    print(f"   Number of samples: {len(dataset)}")
    print(f"   Number of attributes: 40")
    
except FileNotFoundError:
    print("❌ CelebA not found locally")
    print("   Set download=True in celeba_experiment.py to download (~1.4GB)")
    print("   Or download manually from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    
except Exception as e:
    print(f"Error: {e}")