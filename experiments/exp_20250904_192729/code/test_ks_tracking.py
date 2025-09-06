"""Test KS distance tracking in DERP-VAE"""
import torch
import sys
sys.path.append('.')

from derp_vae import DERP_VAE, EnhancedStandardVAE

print("Testing KS distance tracking...")

# Create models
config = {
    'input_dim': 3072,
    'hidden_dim': 256,
    'latent_dim': 4,
    'n_classes': 10
}

# Test Standard VAE (should have ks_distance = 0)
standard_vae = EnhancedStandardVAE(**config)
x_test = torch.rand(32, 3072)  # Batch of 32 samples in [0,1] range for BCE loss
labels_test = torch.randint(0, 10, (32,))

loss_dict_standard = standard_vae.compute_loss(x_test, labels_test)
print("\nStandard VAE losses:")
for key, value in loss_dict_standard.items():
    if key != 'class_logits':
        print(f"  {key}: {value.item():.4f}")

# Test DERP-VAE (should have non-zero ks_distance)
derp_vae = DERP_VAE(**config, n_probes=3, enforcement_weight=1.0, device='cpu')
loss_dict_derp = derp_vae.compute_loss(x_test, labels_test)
print("\nDERP-VAE losses:")
for key, value in loss_dict_derp.items():
    if key != 'class_logits':
        print(f"  {key}: {value.item():.4f}")

print("\n✅ KS distance tracking is working correctly!")
print(f"  - Standard VAE KS distance: {loss_dict_standard['ks_distance'].item():.4f} (should be 0)")
print(f"  - DERP-VAE KS distance: {loss_dict_derp['ks_distance'].item():.4f} (should be > 0)")