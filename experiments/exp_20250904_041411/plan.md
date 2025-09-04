# Experiment Plan: Computational Efficiency of Random Probe Testing (exp3)

**Experiment ID**: exp_20250904_041411  
**Date**: September 4, 2025  
**Selected from**: proposal.jsonl exp3

## Hypothesis to Test

**Primary Hypothesis (H3, H11)**: Random projection-based distributional testing provides comparable statistical power to full multivariate methods at significantly lower computational cost

**Specific Claims**:
1. Random projections preserve essential statistical properties beyond distance preservation
2. 1D statistical tests on random projections can effectively detect distributional deviations in high-dimensional spaces
3. Computational complexity scales linearly or sub-linearly with dimensionality

## Scientific Methodology

### Independent Variables
- **Testing Method**: Full multivariate, Random projections (1D), PCA projections (1D)
- **Projection Count**: [1, 5, 10, 20, 50] projections per test
- **Dimensionality**: [10, 50, 100, 500, 1000]
- **Distribution Type**: Gaussian, Mixture of Gaussians, Uniform, Heavy-tailed (t-distribution)

### Dependent Variables
- **Statistical Power**: True positive rate (ability to detect distribution mismatch)
- **Type I Error**: False positive rate (incorrectly rejecting correct distribution)
- **Computational Cost**: Wall clock time, FLOPS count
- **Memory Usage**: Peak memory consumption
- **Scalability**: Time complexity as function of dimensionality

### Success Criteria
- **Efficiency**: >90% computational reduction vs full multivariate methods
- **Statistical Validity**: >80% statistical power, <5% Type I error
- **Scalability**: Linear or sub-linear scaling with dimensionality

## Experimental Design

### Controls and Baselines
1. **Full Multivariate Tests**: 
   - Anderson-Darling multivariate test
   - Energy statistics test
   - Maximum Mean Discrepancy (MMD)

2. **Random Projection Variants**:
   - Standard Gaussian projections
   - Rademacher (±1) projections
   - PCA-based projections (for comparison)

3. **Statistical Tests on Projections**:
   - Kolmogorov-Smirnov test
   - Anderson-Darling test
   - Wasserstein distance

### Dataset Strategy

#### Synthetic Distributions (Ground Truth Known)
1. **Null Hypothesis Data** (H0: same distribution):
   - Generate pairs from identical distributions
   - Test Type I error rates

2. **Alternative Hypothesis Data** (H1: different distributions):
   - Generate pairs from different distributions
   - Test statistical power

#### Real Datasets from data/ folder
- Use available synthetic datasets: gaussian_100d.npy, beta_distribution.npy
- Generate additional controlled synthetic data for comprehensive testing

## Implementation Plan

### Phase 1: Infrastructure (30 min)
1. Load synthetic datasets from data/
2. Implement distribution generators for controlled testing
3. Set up random projection generators
4. Implement multivariate baseline tests

### Phase 2: Statistical Testing Framework (45 min)
1. Implement statistical power calculation
2. Create Type I/II error measurement
3. Set up computational profiling (time, memory)
4. Implement multiple hypothesis correction (Bonferroni/Benjamini-Hochberg)

### Phase 3: Experimental Execution (60 min)
1. Run null hypothesis tests (Type I error)
2. Run alternative hypothesis tests (statistical power)
3. Profile computational performance
4. Test scalability across dimensions

### Phase 4: Statistical Analysis (30 min)
1. Calculate effect sizes and confidence intervals
2. Perform statistical significance testing
3. Generate performance vs accuracy trade-off curves
4. Create scalability analysis plots

## Expected Outcomes

**If Hypothesis Supported**:
- Random projections maintain >80% statistical power
- >90% reduction in computational cost
- Linear/sub-linear scaling
- Validates theoretical foundation (Cramer-Wold theorem application)

**If Hypothesis Rejected**:
- Statistical power significantly degraded
- Computational savings insufficient
- Poor scaling properties
- Suggests need for more sophisticated projection strategies

## Experimental Controls

### Reproducibility
- Fixed random seeds for all experiments
- Multiple independent runs (n≥30) for statistical validity
- Version control all code and results

### Confounding Variables
- Control for implementation efficiency differences
- Use identical statistical test implementations where possible
- Account for warm-up effects in timing measurements

### Statistical Rigor
- Pre-register success criteria (avoid p-hacking)
- Use appropriate multiple comparison corrections
- Report both statistical and practical significance
- Include confidence intervals and effect sizes

## Risk Assessment

**High Risk**: Implementation complexity of multivariate baseline tests
**Medium Risk**: Computational profiling accuracy across different hardware
**Low Risk**: Statistical test implementation (well-established methods)

**Mitigation**: Start with simpler baselines, use existing statistical libraries where possible, validate against known results

---

This experiment directly addresses the fundamental theoretical question of whether random projections preserve sufficient distributional information for practical statistical testing, with rigorous controls and adequate statistical power.