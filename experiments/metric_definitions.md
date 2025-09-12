# VAE Experiment Metrics Definitions

This document provides detailed explanations of the key metrics used in DERP-VAE experiments to evaluate model performance and behavior.

## 1. Test Loss

**Definition**: The total loss computed on the test dataset, representing the overall objective function that the model is trying to minimize.

**Components**: In VAE experiments, this typically includes:
- Reconstruction loss (how well the model reconstructs input data)
- KL divergence loss (regularization term)
- Classification loss (if applicable)
- Additional losses (e.g., perceptual loss, distributional enforcement loss in DERP-VAE)

**Interpretation**: 
- Lower values indicate better overall model performance
- The absolute value depends on data dimensionality and loss scale
- Example: `Test Loss: 6838.8493` indicates the combined loss value

**Formula**: 
```
Total Loss = Reconstruction Loss + β * KL Loss + α * Classification Loss + Other Losses
```

## 2. KL Divergence (Kullback-Leibler Divergence)

**Definition**: Measures how much the learned latent distribution q(z|x) diverges from the prior distribution p(z), typically a standard normal N(0,I).

**Purpose**: 
- Regularizes the latent space to follow the prior distribution
- Prevents posterior collapse (when the encoder ignores input and produces constant outputs)
- Ensures smooth, interpretable latent representations

**Interpretation**:
- Lower values indicate the latent distribution is closer to the prior
- Very low values (< 1) might indicate posterior collapse
- Example: `KL Divergence: 35.8710` shows substantial divergence from the prior
- Healthy range typically: 10-50 for complex datasets

**Formula**:
```
KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```
where μ and σ are the mean and standard deviation of the encoder output.

## 3. Classification Accuracy

**Definition**: The proportion of correctly classified samples in a multi-class or binary classification task.

**Calculation**: 
```
Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
```

**Interpretation**:
- Ranges from 0.0 to 1.0 (or 0% to 100%)
- Example: `Classification Accuracy: 0.6263` means 62.63% of samples were correctly classified
- For CIFAR-10 (10 classes), random guessing would yield ~0.10 accuracy
- For binary tasks (like CelebA's "Smiling" attribute), random guessing would yield ~0.50

**Context**: In VAE experiments, this measures how well the latent representations preserve class-relevant information.

## 4. Activation Rate

**Definition**: The proportion of latent dimensions that are "active" (meaningfully used) versus those that have collapsed to the prior.

**Calculation**:
1. For each latent dimension, compute the variance across the dataset
2. A dimension is considered "active" if its variance is above a threshold (e.g., 0.01)
3. Activation Rate = (Number of Active Dimensions) / (Total Latent Dimensions)

**Interpretation**:
- Ranges from 0.0 to 1.0
- Example: `Activation Rate: 0.9923` means 99.23% of latent dimensions are actively used
- Low activation rates indicate posterior collapse or dimension pruning
- High rates suggest the model effectively uses its latent capacity

**Importance**: 
- Indicates latent space utilization efficiency
- Low rates suggest the model might benefit from fewer latent dimensions
- DERP-VAE aims to maintain high activation rates while preserving normality

## 5. KS Distance (Kolmogorov-Smirnov Distance)

**Definition**: Measures the maximum difference between the empirical cumulative distribution function (CDF) of latent projections and the theoretical CDF of a standard normal distribution.

**Purpose**:
- Quantifies how "normal" the latent distribution is
- Tests whether latent codes follow the assumed prior distribution
- Used in DERP-VAE to enforce distributional properties

**Calculation**:
1. Project high-dimensional latents to 1D using random projections
2. Compare empirical CDF with standard normal CDF
3. KS distance = max|F_empirical(x) - F_theoretical(x)|

**Interpretation**:
- Ranges from 0.0 to 1.0
- Lower values indicate better normality
- Example: `KS Distance: 0.0690` indicates good adherence to normal distribution
- Values < 0.1 generally indicate good distributional match
- Values > 0.2 suggest significant deviation from normality

**DERP-VAE Context**:
- DERP-VAE explicitly minimizes KS distance during training
- Helps ensure latent codes are properly distributed
- Supports better generation and interpolation properties

## Metric Relationships

These metrics are interconnected:
- **Trade-offs**: Lower KL divergence often comes at the cost of reconstruction quality
- **Balance**: DERP-VAE aims to maintain low KS distance while preserving high activation rates
- **Holistic View**: No single metric tells the complete story; all should be considered together

## Typical Healthy Ranges

| Metric | Good Range | Warning Signs |
|--------|------------|---------------|
| Test Loss | Dataset-dependent | Sudden increases, divergence |
| KL Divergence | 10-50 | < 1 (collapse), > 100 (poor regularization) |
| Classification Accuracy | > random baseline | Near random performance |
| Activation Rate | > 0.8 | < 0.5 indicates severe collapse |
| KS Distance | < 0.1 | > 0.2 indicates poor normality |

## Usage in Model Selection

When comparing models:
1. **Standard VAE**: Baseline for all metrics
2. **β-VAE**: May have lower KL divergence but potentially worse reconstruction
3. **DERP-VAE**: Should maintain low KS distance with minimal impact on other metrics

The goal is to find models that balance all metrics effectively rather than optimizing for any single one.
