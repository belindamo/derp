# Mathematical Formulas for VAE Metrics

## 1. Test Loss

For a VAE with classification, the total loss is:

```
L_total = L_recon + β·L_KL + α·L_class + λ·L_other
```

Where:
- `L_recon`: Reconstruction loss (BCE or MSE)
- `L_KL`: KL divergence loss
- `L_class`: Classification loss (CrossEntropy)
- `L_other`: Additional losses (e.g., perceptual, KS distance)
- `β, α, λ`: Weighting factors

### Reconstruction Loss (Binary Cross-Entropy):
```
L_recon = -1/N Σᵢ [xᵢ·log(x̂ᵢ) + (1-xᵢ)·log(1-x̂ᵢ)]
```

## 2. KL Divergence

For a Gaussian VAE with latent dimension D:

```
L_KL = -0.5 · Σⱼ₌₁ᴰ [1 + log(σⱼ²) - μⱼ² - σⱼ²]
```

Where:
- `μⱼ`: Mean of latent dimension j
- `σⱼ²`: Variance of latent dimension j
- Sum is over all D latent dimensions

## 3. Classification Accuracy

```
Accuracy = Σᵢ 1[ŷᵢ = yᵢ] / N
```

Where:
- `ŷᵢ`: Predicted class for sample i
- `yᵢ`: True class for sample i
- `1[·]`: Indicator function (1 if true, 0 if false)
- `N`: Total number of samples

## 4. Activation Rate

For each latent dimension j:

```
Var(zⱼ) = 1/N Σᵢ (zᵢⱼ - z̄ⱼ)²
```

A dimension is active if `Var(zⱼ) > τ` (typically τ = 0.01)

```
Activation Rate = |{j : Var(zⱼ) > τ}| / D
```

Where:
- `D`: Total number of latent dimensions
- `|·|`: Cardinality (count) of the set

## 5. KS Distance

For random projection vector `v` and latent samples `Z`:

1. Project: `pᵢ = zᵢᵀ·v` for each sample i
2. Sort projections: `p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₙ₎`
3. Compute empirical CDF: `F̂(p₍ᵢ₎) = i/n`
4. Compute theoretical CDF: `F(p) = Φ(p)` where Φ is standard normal CDF

```
KS = max |F̂(p₍ᵢ₎) - F(p₍ᵢ₎)|
     i
```

### Standard Normal CDF:
```
Φ(x) = 0.5 · [1 + erf(x/√2)]
```

Where `erf` is the error function.

## DERP-VAE Specific

### Modified KS Loss (used during training):
```
L_KS = 1/K Σₖ₌₁ᴷ [1/N Σᵢ |F̂ₖ(pᵢₖ) - Φ(pᵢₖ)|]
```

Where:
- `K`: Number of random projections
- Average deviation used instead of maximum for differentiability

### Total DERP-VAE Loss:
```
L_DERP = L_recon + β·L_KL + α·L_class + λ_p·L_perceptual + λ_ks·L_KS
```

## Interpretation Guidelines

### KL Divergence Values:
- **Posterior Collapse**: KL < 1 (encoder ignores input)
- **Healthy Range**: 10 < KL < 50
- **Under-regularized**: KL > 100

### Activation Rate Thresholds:
- **Good**: AR > 0.8 (80%+ dimensions active)
- **Concerning**: 0.5 < AR < 0.8
- **Collapse**: AR < 0.5

### KS Distance Interpretation:
- **Excellent**: KS < 0.05
- **Good**: 0.05 < KS < 0.10
- **Acceptable**: 0.10 < KS < 0.15
- **Poor**: KS > 0.15
