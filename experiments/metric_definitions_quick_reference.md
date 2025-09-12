# VAE Metrics Quick Reference

## Core Metrics

### 1. **Test Loss** 
- **What**: Combined objective function value on test data
- **Lower is better**: Yes
- **Example**: `6838.8493` (scale depends on data dimensionality)

### 2. **KL Divergence**
- **What**: Distance between learned latent distribution and prior N(0,I)
- **Healthy range**: 10-50 for complex data
- **Warning**: < 1 indicates posterior collapse
- **Example**: `35.8710` (good - indicates active latent space)

### 3. **Classification Accuracy**
- **What**: Fraction of correctly classified samples
- **Range**: 0.0 to 1.0 
- **Example**: `0.6263` = 62.63% accuracy
- **Baseline**: Random = 0.1 (CIFAR-10), 0.5 (binary)

### 4. **Activation Rate**
- **What**: Fraction of latent dimensions actively used
- **Range**: 0.0 to 1.0
- **Healthy**: > 0.8
- **Example**: `0.9923` = 99.23% dimensions active (excellent)

### 5. **KS Distance**
- **What**: Maximum deviation from normal distribution
- **Range**: 0.0 to 1.0
- **Good**: < 0.1 
- **Example**: `0.0690` (very good normality)

## Quick Interpretation Guide

| Metric | 🟢 Good | 🟡 Caution | 🔴 Problem |
|--------|---------|------------|------------|
| KL Divergence | 10-50 | 5-10 or 50-100 | <5 or >100 |
| Activation Rate | >0.8 | 0.5-0.8 | <0.5 |
| KS Distance | <0.1 | 0.1-0.2 | >0.2 |

## Model Comparison
- **Standard VAE**: Baseline for all metrics
- **β-VAE**: Lower KL, potentially worse reconstruction
- **DERP-VAE**: Better KS distance while maintaining other metrics
