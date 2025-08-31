

# Datasets for Distribution Enforcement via Random Probe (DERP) Research

## Dataset Collection Complete ✅

Successfully identified, downloaded, and organized **6 primary datasets** and **3 synthetic distributions** following scientific research methodology. Total storage: ~6.2GB managed via Git LFS.

## Primary Research Datasets

### Standard Computer Vision Benchmarks

#### **MNIST** (Classic VAE Baseline)
- **Path**: `data/processed/mnist/`
- **Size**: 70,000 samples (60k train + 10k test)
- **Format**: 28×28 grayscale, 10 classes (digits)
- **Relevance**: Fundamental VAE benchmark for distributional modeling baseline
- **DERP Application**: Test basic posterior collapse prevention, K-S distance modifications
- **Literature**: Kingma & Welling (2013) original VAE paper

#### **Fashion-MNIST** (Complex Textures)
- **Path**: `data/processed/fashion_mnist/`
- **Size**: 70,000 samples (60k train + 10k test)
- **Format**: 28×28 grayscale, 10 classes (clothing)
- **Relevance**: More complex textures than MNIST for VAE evaluation
- **DERP Application**: Test distributional enforcement on complex texture patterns
- **Cross-Domain**: Transfer learning validation (MNIST→Fashion-MNIST)

#### **CIFAR-10** (Standard Complexity)
- **Path**: `data/processed/cifar10/`
- **Size**: 60,000 samples (50k train + 10k test)
- **Format**: 32×32×3 color, 10 classes (objects/animals)
- **Relevance**: Standard VAE/VQ-VAE benchmark for distributional complexity analysis
- **DERP Application**: Multi-channel distributional enforcement, color space projections
- **Literature**: Razavi et al. (2019) VQ-VAE-2, Chadebec et al. (2022) Pythae benchmark

#### **CIFAR-100** (High Complexity)
- **Path**: `data/processed/cifar100/`
- **Size**: 60,000 samples (50k train + 10k test)
- **Format**: 32×32×3 color, 100 fine-grained classes
- **Relevance**: Higher complexity distributional modeling challenges
- **DERP Application**: Test scaling of distributional enforcement to complex class structures
- **Hypothesis**: More classes require stronger distributional constraints

#### **SVHN** (Distributional Shift)
- **Path**: `data/processed/tfds/svhn_cropped/`
- **Size**: ~99,000 samples (73k train + 26k test)
- **Format**: 32×32×3 color, 10 classes (real-world digits)
- **Relevance**: Real-world digit recognition, distributional shift studies vs MNIST
- **DERP Application**: Cross-domain robustness testing (MNIST→SVHN)
- **Research Question**: How does DERP handle domain gap in digit distributions?

### Specialized Generative Benchmarks

#### **dSprites** (Disentanglement)
- **Path**: `data/processed/dsprites/`
- **Size**: 737,280 images
- **Format**: 64×64 binary images
- **Latent Factors**: 6 (shape, scale, orientation, x-pos, y-pos, color)
- **Relevance**: Disentangled representation learning benchmark, VAE evaluation
- **DERP Application**: Test distributional enforcement on known latent structure
- **Hypothesis**: DERP should improve disentanglement via explicit distributional constraints

## Synthetic Distributions for Statistical Testing

### **2D Gaussian Mixture** (Known Multimodal)
- **Path**: `data/synthetic/gaussian_mixture_2d.npy`
- **Size**: 10,000 samples, 5 Gaussian components
- **Purpose**: Test distributional enforcement on known multimodal distributions
- **DERP Application**: Validate random projection methods on controlled distributions
- **Ground Truth**: Known component means, covariances for validation

### **High-Dimensional Gaussian** (Random Projection Testing)
- **Path**: `data/synthetic/gaussian_100d.npy`
- **Size**: 10,000 samples, 100 dimensions
- **Distribution**: Multivariate Gaussian (μ=0, Σ=I)
- **Purpose**: Test random projection methods, validate Cramer-Wold theorem
- **DERP Application**: Verify 1D projections capture multivariate Gaussianity
- **Theoretical**: Direct validation of core mathematical foundation

### **Beta Distribution** (Non-Gaussian Testing)
- **Path**: `data/synthetic/beta_distribution.npy`
- **Size**: 10,000 samples, 10 dimensions
- **Distribution**: Beta(α=2, β=5) - asymmetric, bounded
- **Purpose**: Test non-Gaussian distributional enforcement capabilities
- **DERP Application**: Verify method works beyond Gaussian assumptions
- **Challenge**: Non-symmetric, bounded support distribution

## Research Applications & Experimental Design

### Primary Experimental Frameworks

1. **VAE Posterior Collapse Analysis**
   - **Datasets**: MNIST, Fashion-MNIST, CIFAR-10/100
   - **Baseline**: Standard VAE training
   - **DERP Enhancement**: Active distributional enforcement via random probes
   - **Metrics**: Posterior collapse rate, reconstruction quality, distributional distance

2. **VQ-VAE Codebook Utilization**
   - **Datasets**: dSprites, CIFAR datasets
   - **Problem**: Codebook underutilization ("collapse")
   - **DERP Solution**: Distributional constraints on discrete representations
   - **Metrics**: Codebook usage entropy, representation quality

3. **Random Projection Validation**
   - **Datasets**: High-dimensional synthetic data
   - **Theory**: Cramer-Wold theorem validation
   - **Implementation**: 1D vs higher-dimensional statistical tests
   - **Metrics**: Statistical power, computational efficiency

4. **Cross-Domain Robustness**
   - **Transfer**: MNIST→Fashion-MNIST, MNIST→SVHN
   - **Hypothesis**: DERP improves distributional robustness
   - **Metrics**: Cross-domain accuracy, distributional similarity measures

### Statistical Testing Framework

#### Modified Kolmogorov-Smirnov Distance
- **Innovation**: Average-based vs maximum-based deviation
- **Benefit**: Smoother gradients for backpropagation
- **Testing**: All datasets with ground truth distributions
- **Validation**: Compare statistical power with classical K-S test

#### Random Probe Implementation
- **Method**: Random 1D projections of high-dimensional representations
- **Theoretical**: Cramer-Wold theorem foundation
- **Efficiency**: O(d) vs O(d²) for full multivariate tests
- **Datasets**: All image datasets + synthetic distributions

## Literature-Level Hypothesis Validation

### Core Research Questions

1. **Active vs Passive Distributional Modeling**
   - **Literature Gap**: Implicit distributional assumptions in VAE/VQ-VAE
   - **DERP Hypothesis**: Active enforcement superior to emergent properties
   - **Validation**: Comparative studies across all datasets

2. **Computational Efficiency of Statistical Testing**
   - **Challenge**: High-dimensional distributional verification
   - **DERP Solution**: Random projection-based testing
   - **Validation**: Efficiency metrics on synthetic high-dimensional data

3. **Identifiability in Generative Models**
   - **Problem**: Posterior collapse as identifiability failure
   - **DERP Framework**: Statistical constraints improve identifiability
   - **Testing**: Identifiability measures on VAE experiments

### Expected Outcomes

- **Quantitative**: 30%+ improvement in posterior utilization (VAE), codebook usage (VQ-VAE)
- **Qualitative**: Sharper, more coherent generated samples
- **Theoretical**: Validation of temperature-driven distributional enforcement
- **Practical**: Faster inference via efficient statistical testing

## Dataset Access and Reproducibility

### Loading Examples
```python
# Standard PyTorch loading
import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()
mnist = datasets.MNIST('data/processed/mnist', train=True, transform=transform)
cifar10 = datasets.CIFAR10('data/processed/cifar10', train=True, transform=transform)

# dSprites
import numpy as np
dsprites_imgs = np.load('data/processed/dsprites/images.npy')
dsprites_factors = np.load('data/processed/dsprites/labels.npy')

# Synthetic distributions
gaussian_2d = np.load('data/synthetic/gaussian_mixture_2d.npy')
gaussian_100d = np.load('data/synthetic/gaussian_100d.npy')
beta_dist = np.load('data/synthetic/beta_distribution.npy')
```

### Version Control
- **Git LFS**: All datasets managed for reproducibility
- **Metadata**: Complete dataset provenance and statistics
- **Documentation**: Comprehensive loading instructions and research applications

---

**Dataset Collection Status**: ✅ **COMPLETE**  
**Total Storage**: 6.2GB via Git LFS  
**Research-Ready**: All datasets validated and documented for DERP experiments

*Following Stanford scientific research methodology for literature-level hypothesis testing*