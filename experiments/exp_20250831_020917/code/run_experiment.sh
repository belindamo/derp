#!/bin/bash

# VAE Posterior Collapse Prevention Experiment
# Scientific methodology with proper statistical controls

echo "=== VAE Posterior Collapse Prevention Experiment ==="
echo "Experiment ID: exp_20250831_020917"
echo "Date: $(date)"
echo

# Check Python and dependencies
echo "Checking Python environment..."
python --version
echo

# Install required packages if needed
pip install torch torchvision matplotlib seaborn scipy pandas --quiet

# Run main experiment
echo "Starting experiment execution..."
cd /home/runner/work/derp/derp/experiments/exp_20250831_020917/code

python main.py \
    --data_path ../../../data/processed \
    --output_dir ../results \
    --dataset mnist \
    --seeds 42 123 456

echo
echo "Running statistical analysis..."
python analysis.py \
    --results_path ../results/all_results.json \
    --output_dir ../results

echo
echo "=== Experiment Complete ==="
echo "Results available in: ../results/"
echo "Analysis available in: ../results/statistical_analysis.json"