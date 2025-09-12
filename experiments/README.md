# Experiments

This folder contains all research experiments for the DERP-VAE project.

## Experiment Folders

- `exp_20250904_192729 cifar/` - CIFAR-10 experiments
- `exp_20250904_20000 celeba all/` - CelebA experiments

## Documentation

- [`metric_definitions.md`](metric_definitions.md) - Comprehensive explanations of all evaluation metrics
- [`metric_definitions_quick_reference.md`](metric_definitions_quick_reference.md) - Quick reference guide for metrics
- [`metric_formulas.md`](metric_formulas.md) - Mathematical formulas and detailed calculations
- [`example_metrics_interpretation.md`](example_metrics_interpretation.md) - Example interpretation of actual experiment output

## Key Metrics Tracked

All experiments track the following metrics:
1. **Test Loss** - Overall objective function value
2. **KL Divergence** - Latent space regularization measure
3. **Classification Accuracy** - Task performance
4. **Activation Rate** - Latent dimension utilization
5. **KS Distance** - Distributional normality measure

See the metric definition files for detailed explanations and interpretation guidelines.