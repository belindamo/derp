# CelebA Experiment Fix Checklist

## 🚨 Priority 1: Critical Fixes (Do First)

- [ ] **Fix Train/Test Split**
  - [ ] Use official CelebA splits (train/valid/test)
  - [ ] Update `get_celeba_dataloaders()` function
  - [ ] Remove random split code

- [ ] **Fix Evaluation Data Collection**
  - [ ] Remove 50-batch limit in evaluation loop
  - [ ] Collect metrics from entire test set
  - [ ] Or increase to at least 200 batches if memory constrained

## ⚠️ Priority 2: Scientific Validity

- [ ] **Add Validation Set**
  - [ ] Create 3-way data split
  - [ ] Use validation for early stopping
  - [ ] Use validation for hyperparameter selection

- [ ] **Multiple Seeds**
  - [ ] Run with seeds: [42, 123, 456, 789, 2024]
  - [ ] Report mean ± std for all metrics
  - [ ] Add seed as experiment parameter

- [ ] **Fix Hyperparameters**
  - [ ] Make loss weights configurable
  - [ ] Add more β-VAE baselines (β = 0.5, 2.0, 4.0)
  - [ ] Ensure fair comparison between models

## 💡 Priority 3: Improvements

- [ ] **Reproducibility**
  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```

- [ ] **Add Missing Evaluations**
  - [ ] Reconstruction visualizations
  - [ ] Latent space interpolations
  - [ ] Statistical hypothesis testing
  - [ ] Confidence intervals

- [ ] **Model Checkpointing**
  - [ ] Save best model based on validation loss
  - [ ] Save final model
  - [ ] Save training curves

## 📝 Quick Code Fixes

### 1. Data Loader Fix
```python
# In data_loader.py
def get_celeba_dataloaders_fixed(...):
    # Load all three splits
    train_data = torchvision.datasets.CelebA(root=data_dir, split='train', ...)
    val_data = torchvision.datasets.CelebA(root=data_dir, split='valid', ...)
    test_data = torchvision.datasets.CelebA(root=data_dir, split='test', ...)
```

### 2. Evaluation Fix
```python
# In celeba_experiment.py, line 187
# Remove this condition:
if len(all_mus) < 50:  # DELETE THIS LINE
    # Keep the indented code
```

### 3. Add Seeds Loop
```python
# In main()
SEEDS = [42, 123, 456, 789, 2024]
all_results = {}

for seed in SEEDS:
    set_seeds(seed)
    results = run_experiments()
    all_results[seed] = results

# Compute statistics across seeds
```

## 🧪 Testing After Fixes

1. Verify data splits are correct:
   ```python
   print(f"Train: {len(train_dataset)}")  # Should be ~162,770
   print(f"Val: {len(val_dataset)}")      # Should be ~19,867  
   print(f"Test: {len(test_dataset)}")    # Should be ~19,962
   ```

2. Check evaluation uses full test set:
   ```python
   print(f"Evaluated on {len(all_mus)} samples")  # Should match test set size
   ```

3. Verify reproducibility:
   - Run twice with same seed
   - Results should be identical

## 📊 Expected Improvements

After fixes, you should see:
- More reliable metrics (evaluated on full test set)
- Reproducible results across runs
- Valid comparison with other papers using CelebA
- Statistically significant conclusions

## 🚀 Next Steps

1. Implement Priority 1 fixes
2. Run quick test to verify fixes work
3. Implement Priority 2 fixes
4. Run full experiment with multiple seeds
5. Add visualizations and additional metrics
