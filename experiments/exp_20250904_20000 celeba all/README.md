# Experiment: exp_20250904_20000 - CelebA Face Attributes

**Domain**: Deep Learning - Variational Autoencoders  
**Hypothesis**: Testing DERP-VAE on large-scale face image dataset  
**Dataset**: CelebA - 202,599 face images with 40 binary attributes  
**Method**: DERP-VAE with multi-objective optimization on high-dimensional image data

## ⚠️ Important: Code Analysis Available

**Before running this experiment**, please review these critical analysis documents:

- 📋 [`EXPERIMENT_ANALYSIS.md`](EXPERIMENT_ANALYSIS.md) - Comprehensive code analysis with bugs and issues
- 🔍 [`ISSUES_SUMMARY.md`](ISSUES_SUMMARY.md) - Quick visual summary of all problems
- ✅ [`FIX_CHECKLIST.md`](FIX_CHECKLIST.md) - Step-by-step guide to fix issues

### Critical Issues Identified:
1. **🔴 Wrong Data Split**: Uses only 'train' split then randomly splits it (should use official train/val/test)
2. **🔴 Limited Evaluation**: Only evaluates on 50 batches (~1,600 samples) instead of full test set
3. **🟡 No Validation Set**: Cannot properly tune hyperparameters
4. **🟡 Single Seed**: No statistical significance testing

## Experiment Overview

### Dataset: CelebA
- **Total Images**: 202,599 celebrity face images
- **Image Size**: Resized to 64×64 (originally 178×218)
- **Attributes**: 40 binary attributes
- **Task**: Binary classification on 'Smiling' attribute
- **Input Dimensions**: 64×64×3 = 12,288

### Models Tested
1. **Standard VAE** (baseline)
   - β = 1.0
   - Standard architecture

2. **β-VAE** 
   - β = 0.1 (lower for complex images)
   - Same architecture as baseline

3. **DERP-VAE**
   - β = 1.0
   - 5 random probes
   - Enforcement weight = 0.5

### Architecture Details
- **Encoder**: 12,288 → 512 → 256 → 64
- **Decoder**: 64 → 256 → 512 → 12,288  
- **Latent Space**: 64 dimensions
- **Classifier**: 64 → 32 → 2
- **Activation**: ReLU, Sigmoid (output)
- **Dropout**: 0.2

## Research Questions

1. **Posterior Collapse Prevention**: Can DERP-VAE prevent collapse on high-dimensional image data?
2. **Scalability**: Does distributional enforcement scale to complex visual features?
3. **Computational Cost**: What is the overhead on large-scale datasets?

## Expected Outcomes

- ✅ Higher activation rate (>90% vs baseline)
- ✅ Lower KS distance (<0.1)
- ✅ Maintained classification accuracy
- ✅ Reasonable computational overhead (<50%)

## How to Run

### Basic Execution
```bash
cd code
python celeba_experiment.py
```

### Configuration
- Batch size: 32 (for memory efficiency)
- Epochs: 10
- Learning rate: 1e-4
- Device: Auto-detects GPU

### Output Files
- `results/celeba_experiment_results.json` - All metrics
- `results/celeba_results.png` - Visualizations
- `results/hyperparameter_summary.md` - Full config details

## Metrics Tracked

1. **Test Loss** - Overall objective value
2. **KL Divergence** - Regularization measure
3. **Classification Accuracy** - Smiling detection
4. **Activation Rate** - Latent utilization
5. **KS Distance** - Distributional normality

See [`../../metric_definitions.md`](../../metric_definitions.md) for detailed explanations.

## Known Issues & Fixes

Please review the analysis documents before running. Quick fixes:

1. **Data Split Fix** (in `data_loader.py`):
```python
# Change line 256 from:
split='train'  # Wrong!
# To use all three splits:
train_data = CelebA(split='train')
val_data = CelebA(split='valid')
test_data = CelebA(split='test')
```

2. **Evaluation Fix** (in `celeba_experiment.py`):
```python
# Remove line 187:
if len(all_mus) < 50:  # DELETE THIS
```

## Future Improvements

- Multiple seed runs for statistical significance
- More β-VAE baselines (β = 0.5, 2.0, 4.0)
- Reconstruction quality visualizations
- FID score computation
- Latent space interpolations