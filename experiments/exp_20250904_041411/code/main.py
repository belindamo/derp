#!/usr/bin/env python3
"""
Computational Efficiency of Random Probe Testing
Experiment: exp_20250904_041411

Test whether random projection-based distributional testing provides 
comparable statistical power to full multivariate methods at significantly 
lower computational cost.

Hypothesis: Random projections preserve essential statistical properties 
for high-dimensional distributional testing with >90% computational reduction.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
import psutil
import tracemalloc
from typing import Tuple, List, Dict, Any

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    np.random.seed(seed)

class DistributionGenerator:
    """Generate synthetic distributions for statistical testing"""
    
    @staticmethod
    def multivariate_gaussian(n_samples: int, dim: int, mean=None, cov=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if mean is None:
            mean = np.zeros(dim)
        if cov is None:
            cov = np.eye(dim)
        return np.random.multivariate_normal(mean, cov, n_samples)
    
    @staticmethod
    def gaussian_mixture(n_samples: int, dim: int, n_components: int = 3, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Create mixture components
        components = []
        weights = np.random.dirichlet(np.ones(n_components))
        
        for i in range(n_components):
            mean = np.random.randn(dim) * 2
            cov = np.eye(dim) + np.random.randn(dim, dim) * 0.1
            cov = cov @ cov.T  # Ensure positive definite
            components.append((mean, cov))
        
        # Generate samples
        samples = []
        component_counts = np.random.multinomial(n_samples, weights)
        
        for i, (mean, cov) in enumerate(components):
            if component_counts[i] > 0:
                comp_samples = np.random.multivariate_normal(mean, cov, component_counts[i])
                samples.append(comp_samples)
        
        return np.vstack(samples)
    
    @staticmethod
    def uniform_hypercube(n_samples: int, dim: int, low=0, high=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low, high, (n_samples, dim))
    
    @staticmethod
    def multivariate_t(n_samples: int, dim: int, df: int = 3, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Generate multivariate t-distribution using Gaussian + Chi-squared
        gaussian = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n_samples)
        chi2 = np.random.chisquare(df, n_samples)
        return gaussian / np.sqrt(chi2 / df)[:, np.newaxis]

class RandomProjectionTester:
    """Statistical testing using random projections"""
    
    def __init__(self, n_projections: int = 10):
        self.n_projections = n_projections
        
    def generate_projections(self, dim: int, projection_type='gaussian'):
        """Generate random projection vectors"""
        projections = []
        
        for _ in range(self.n_projections):
            if projection_type == 'gaussian':
                proj = np.random.randn(dim)
            elif projection_type == 'rademacher':
                proj = np.random.choice([-1, 1], dim)
            else:
                raise ValueError(f"Unknown projection type: {projection_type}")
            
            # Normalize
            proj = proj / np.linalg.norm(proj)
            projections.append(proj)
        
        return np.array(projections)
    
    def project_samples(self, X: np.ndarray, projections: np.ndarray):
        """Project high-dimensional samples to 1D"""
        return X @ projections.T
    
    def ks_test_projections(self, X1: np.ndarray, X2: np.ndarray, projections: np.ndarray):
        """Apply K-S test to each projection"""
        proj1 = self.project_samples(X1, projections)
        proj2 = self.project_samples(X2, projections)
        
        p_values = []
        statistics = []
        
        for i in range(len(projections)):
            stat, p_val = stats.ks_2samp(proj1[:, i], proj2[:, i])
            statistics.append(stat)
            p_values.append(p_val)
        
        return np.array(statistics), np.array(p_values)
    
    def combined_test(self, X1: np.ndarray, X2: np.ndarray, projection_type='gaussian', 
                     method='bonferroni'):
        """Perform combined statistical test across projections"""
        dim = X1.shape[1]
        projections = self.generate_projections(dim, projection_type)
        
        statistics, p_values = self.ks_test_projections(X1, X2, projections)
        
        # Multiple comparison correction
        if method == 'bonferroni':
            corrected_alpha = 0.05 / len(p_values)
            significant = np.any(p_values < corrected_alpha)
            min_p = np.min(p_values) * len(p_values)  # Bonferroni correction
        elif method == 'benjamini_hochberg':
            from scipy.stats import false_discovery_control
            significant_mask = false_discovery_control(p_values)
            significant = np.any(significant_mask)
            min_p = np.min(p_values[significant_mask]) if significant else 1.0
        else:
            significant = np.any(p_values < 0.05)
            min_p = np.min(p_values)
        
        return {
            'significant': significant,
            'p_value': min_p,
            'individual_p_values': p_values,
            'statistics': statistics,
            'n_projections': len(projections)
        }

class MultivariateBaselines:
    """Baseline multivariate statistical tests"""
    
    @staticmethod
    def energy_statistic(X1: np.ndarray, X2: np.ndarray):
        """Energy statistics test for equal distributions"""
        from scipy.spatial.distance import pdist, cdist
        
        n1, n2 = len(X1), len(X2)
        
        # Within-group distances
        d11 = np.mean(pdist(X1))
        d22 = np.mean(pdist(X2))
        
        # Between-group distances  
        d12 = np.mean(cdist(X1, X2))
        
        # Energy statistic
        E = 2 * d12 - d11 - d22
        
        # Permutation test for p-value (simplified)
        combined = np.vstack([X1, X2])
        n_perm = 1000
        perm_stats = []
        
        for _ in range(n_perm):
            idx = np.random.permutation(len(combined))
            perm_X1 = combined[idx[:n1]]
            perm_X2 = combined[idx[n1:]]
            
            perm_d11 = np.mean(pdist(perm_X1)) if len(perm_X1) > 1 else 0
            perm_d22 = np.mean(pdist(perm_X2)) if len(perm_X2) > 1 else 0
            perm_d12 = np.mean(cdist(perm_X1, perm_X2))
            
            perm_E = 2 * perm_d12 - perm_d11 - perm_d22
            perm_stats.append(perm_E)
        
        p_value = np.mean(np.array(perm_stats) >= E)
        
        return {'significant': p_value < 0.05, 'p_value': p_value, 'statistic': E}
    
    @staticmethod
    def mmd_test(X1: np.ndarray, X2: np.ndarray, gamma=1.0):
        """Maximum Mean Discrepancy test with RBF kernel"""
        from sklearn.metrics.pairwise import rbf_kernel
        
        n1, n2 = len(X1), len(X2)
        
        # Compute kernel matrices
        K11 = rbf_kernel(X1, X1, gamma=gamma)
        K22 = rbf_kernel(X2, X2, gamma=gamma)
        K12 = rbf_kernel(X1, X2, gamma=gamma)
        
        # MMD statistic
        mmd = (np.sum(K11) / (n1 * n1) + 
               np.sum(K22) / (n2 * n2) - 
               2 * np.sum(K12) / (n1 * n2))
        
        # Permutation test
        combined = np.vstack([X1, X2])
        n_perm = 500  # Reduced for computational efficiency
        perm_mmds = []
        
        for _ in range(n_perm):
            idx = np.random.permutation(len(combined))
            perm_X1 = combined[idx[:n1]]
            perm_X2 = combined[idx[n1:]]
            
            K11_perm = rbf_kernel(perm_X1, perm_X1, gamma=gamma)
            K22_perm = rbf_kernel(perm_X2, perm_X2, gamma=gamma)
            K12_perm = rbf_kernel(perm_X1, perm_X2, gamma=gamma)
            
            mmd_perm = (np.sum(K11_perm) / (n1 * n1) + 
                       np.sum(K22_perm) / (n2 * n2) - 
                       2 * np.sum(K12_perm) / (n1 * n2))
            
            perm_mmds.append(mmd_perm)
        
        p_value = np.mean(np.array(perm_mmds) >= mmd)
        
        return {'significant': p_value < 0.05, 'p_value': p_value, 'statistic': mmd}

class ExperimentRunner:
    """Main experiment execution class"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = defaultdict(list)
        
    def measure_computational_cost(self, func, *args, **kwargs):
        """Measure time and memory usage of a function"""
        # Start memory tracking
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Time the function
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Measure memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'wall_time': end_time - start_time,
            'peak_memory_mb': peak / 1024 / 1024,
            'memory_delta_mb': end_memory - start_memory
        }
    
    def run_null_hypothesis_test(self, n_trials: int = 30):
        """Test Type I error rates (H0: same distribution)"""
        print("Running null hypothesis tests (Type I error measurement)...")
        
        dimensions = [10, 50, 100, 500]  # Reduced for computational efficiency
        projection_counts = [1, 5, 10, 20]
        n_samples = 200
        
        for dim in dimensions:
            print(f"  Testing dimension {dim}...")
            
            for trial in range(n_trials):
                # Generate two samples from the same distribution
                X1 = DistributionGenerator.multivariate_gaussian(n_samples, dim, seed=trial*2)
                X2 = DistributionGenerator.multivariate_gaussian(n_samples, dim, seed=trial*2+1)
                
                # Test with different numbers of projections
                for n_proj in projection_counts:
                    tester = RandomProjectionTester(n_proj)
                    
                    # Random projection test
                    rp_result = self.measure_computational_cost(
                        tester.combined_test, X1, X2, 'gaussian', 'bonferroni'
                    )
                    
                    self.results['null_tests'].append({
                        'trial': trial,
                        'dimension': dim,
                        'n_projections': n_proj,
                        'method': 'random_projection',
                        'significant': rp_result['result']['significant'],
                        'p_value': rp_result['result']['p_value'],
                        'wall_time': rp_result['wall_time'],
                        'peak_memory_mb': rp_result['peak_memory_mb'],
                        'sample_size': n_samples
                    })
                
                # Baseline multivariate tests (only for smaller dimensions)
                if dim <= 100:  # Computational constraint
                    # Energy statistic
                    energy_result = self.measure_computational_cost(
                        MultivariateBaselines.energy_statistic, X1, X2
                    )
                    
                    self.results['null_tests'].append({
                        'trial': trial,
                        'dimension': dim,
                        'n_projections': np.nan,
                        'method': 'energy_statistic',
                        'significant': energy_result['result']['significant'],
                        'p_value': energy_result['result']['p_value'],
                        'wall_time': energy_result['wall_time'],
                        'peak_memory_mb': energy_result['peak_memory_mb'],
                        'sample_size': n_samples
                    })
    
    def run_power_analysis_test(self, n_trials: int = 30):
        """Test statistical power (H1: different distributions)"""
        print("Running power analysis tests (statistical power measurement)...")
        
        dimensions = [10, 50, 100, 500]
        projection_counts = [1, 5, 10, 20]
        n_samples = 200
        
        # Different distribution scenarios
        scenarios = [
            ('gaussian_vs_mixture', 'Different distribution families'),
            ('gaussian_shift', 'Same family, different parameters'),
            ('gaussian_vs_uniform', 'Gaussian vs Uniform'),
        ]
        
        for scenario_name, scenario_desc in scenarios:
            print(f"  Testing scenario: {scenario_desc}")
            
            for dim in dimensions:
                for trial in range(n_trials):
                    # Generate different distributions based on scenario
                    if scenario_name == 'gaussian_vs_mixture':
                        X1 = DistributionGenerator.multivariate_gaussian(n_samples, dim, seed=trial*2)
                        X2 = DistributionGenerator.gaussian_mixture(n_samples, dim, 3, seed=trial*2+1)
                    elif scenario_name == 'gaussian_shift':
                        X1 = DistributionGenerator.multivariate_gaussian(n_samples, dim, seed=trial*2)
                        mean_shift = np.ones(dim) * 0.5
                        X2 = DistributionGenerator.multivariate_gaussian(n_samples, dim, mean=mean_shift, seed=trial*2+1)
                    elif scenario_name == 'gaussian_vs_uniform':
                        X1 = DistributionGenerator.multivariate_gaussian(n_samples, dim, seed=trial*2)
                        X2 = DistributionGenerator.uniform_hypercube(n_samples, dim, -2, 2, seed=trial*2+1)
                    
                    # Test with different numbers of projections
                    for n_proj in projection_counts:
                        tester = RandomProjectionTester(n_proj)
                        
                        # Random projection test
                        rp_result = self.measure_computational_cost(
                            tester.combined_test, X1, X2, 'gaussian', 'bonferroni'
                        )
                        
                        self.results['power_tests'].append({
                            'trial': trial,
                            'scenario': scenario_name,
                            'dimension': dim,
                            'n_projections': n_proj,
                            'method': 'random_projection',
                            'significant': rp_result['result']['significant'],
                            'p_value': rp_result['result']['p_value'],
                            'wall_time': rp_result['wall_time'],
                            'peak_memory_mb': rp_result['peak_memory_mb'],
                            'sample_size': n_samples
                        })
                    
                    # Baseline tests (smaller dimensions only)
                    if dim <= 100:
                        energy_result = self.measure_computational_cost(
                            MultivariateBaselines.energy_statistic, X1, X2
                        )
                        
                        self.results['power_tests'].append({
                            'trial': trial,
                            'scenario': scenario_name,
                            'dimension': dim,
                            'n_projections': np.nan,
                            'method': 'energy_statistic',
                            'significant': energy_result['result']['significant'],
                            'p_value': energy_result['result']['p_value'],
                            'wall_time': energy_result['wall_time'],
                            'peak_memory_mb': energy_result['peak_memory_mb'],
                            'sample_size': n_samples
                        })
    
    def analyze_results(self):
        """Analyze experimental results and compute summary statistics"""
        print("Analyzing results...")
        
        # Convert results to DataFrames
        null_df = pd.DataFrame(self.results['null_tests'])
        power_df = pd.DataFrame(self.results['power_tests'])
        
        analysis = {}
        
        # Type I Error Analysis (should be ≤ 0.05)
        if len(null_df) > 0:
            type_i_error = null_df.groupby(['method', 'dimension', 'n_projections'])['significant'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            type_i_error.columns = ['method', 'dimension', 'n_projections', 'type_i_error', 'std', 'count']
            analysis['type_i_error'] = type_i_error.to_dict(orient='records')
        
        # Statistical Power Analysis (should be high, ideally >0.8)
        if len(power_df) > 0:
            power_analysis = power_df.groupby(['method', 'scenario', 'dimension', 'n_projections'])['significant'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            power_analysis.columns = ['method', 'scenario', 'dimension', 'n_projections', 'power', 'std', 'count']
            analysis['power'] = power_analysis.to_dict(orient='records')
        
        # Computational Efficiency Analysis
        combined_df = pd.concat([
            null_df.assign(test_type='null'),
            power_df.assign(test_type='power')
        ], ignore_index=True)
        
        if len(combined_df) > 0:
            efficiency = combined_df.groupby(['method', 'dimension', 'n_projections']).agg({
                'wall_time': ['mean', 'std'],
                'peak_memory_mb': ['mean', 'std']
            }).reset_index()
            efficiency.columns = ['method', 'dimension', 'n_projections', 'mean_time', 'std_time', 'mean_memory', 'std_memory']
            analysis['efficiency'] = efficiency.to_dict(orient='records')
        
        # Success criteria evaluation
        analysis['success_criteria'] = self.evaluate_success_criteria(analysis)
        
        # Save detailed results
        self.save_results({
            'analysis': analysis,
            'raw_null_tests': self.results['null_tests'],
            'raw_power_tests': self.results['power_tests']
        })
        
        return analysis
    
    def evaluate_success_criteria(self, analysis: Dict) -> Dict:
        """Evaluate against predefined success criteria"""
        criteria = {
            'efficiency_reduction': False,
            'statistical_validity': False,
            'scalability': False
        }
        
        try:
            # Efficiency: >90% computational reduction
            efficiency_data = analysis.get('efficiency', [])
            if efficiency_data:
                df = pd.DataFrame(efficiency_data)
                
                # Compare random projection (n_proj=10) vs energy statistic
                rp_data = df[(df['method'] == 'random_projection') & (df['n_projections'] == 10)]
                energy_data = df[df['method'] == 'energy_statistic']
                
                if len(rp_data) > 0 and len(energy_data) > 0:
                    # Find common dimensions
                    common_dims = set(rp_data['dimension']) & set(energy_data['dimension'])
                    if common_dims:
                        rp_times = rp_data[rp_data['dimension'].isin(common_dims)]['mean_time']
                        energy_times = energy_data[energy_data['dimension'].isin(common_dims)]['mean_time']
                        
                        avg_reduction = 1 - (rp_times.mean() / energy_times.mean())
                        criteria['efficiency_reduction'] = avg_reduction > 0.9
            
            # Statistical validity: >80% power, <5% Type I error
            power_data = analysis.get('power', [])
            type_i_data = analysis.get('type_i_error', [])
            
            if power_data:
                df = pd.DataFrame(power_data)
                rp_power = df[df['method'] == 'random_projection']['power']
                criteria['statistical_validity'] = rp_power.mean() > 0.8 if len(rp_power) > 0 else False
            
            if type_i_data:
                df = pd.DataFrame(type_i_data)
                rp_type_i = df[df['method'] == 'random_projection']['type_i_error']
                criteria['statistical_validity'] = (criteria['statistical_validity'] and 
                                                  rp_type_i.mean() < 0.05 if len(rp_type_i) > 0 else False)
            
            # Scalability: Check if time grows linearly or sub-linearly
            if efficiency_data:
                df = pd.DataFrame(efficiency_data)
                rp_data = df[(df['method'] == 'random_projection') & (df['n_projections'] == 10)]
                
                if len(rp_data) >= 3:  # Need multiple data points
                    dims = rp_data['dimension'].values
                    times = rp_data['mean_time'].values
                    
                    # Fit linear and quadratic models
                    linear_coeff = np.polyfit(dims, times, 1)
                    quad_coeff = np.polyfit(dims, times, 2)
                    
                    # Check if quadratic term is small (indicates linear scaling)
                    criteria['scalability'] = abs(quad_coeff[0]) < 1e-6
        
        except Exception as e:
            print(f"Error evaluating success criteria: {e}")
        
        return criteria
    
    def save_results(self, results: Dict):
        """Save experimental results"""
        with open(self.output_dir / 'experimental_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {self.output_dir / 'experimental_results.json'}")
    
    def generate_plots(self):
        """Generate visualization plots"""
        print("Generating plots...")
        
        try:
            # Load results
            with open(self.output_dir / 'experimental_results.json', 'r') as f:
                results = json.load(f)
            
            # Create plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Type I Error by Method and Dimension
            if 'type_i_error' in results['analysis']:
                type_i_df = pd.DataFrame(results['analysis']['type_i_error'])
                
                # Filter for reasonable projections
                type_i_df = type_i_df[type_i_df['n_projections'].isin([10, np.nan])]
                
                sns.barplot(data=type_i_df, x='dimension', y='type_i_error', 
                           hue='method', ax=ax1)
                ax1.axhline(y=0.05, color='red', linestyle='--', label='Target (≤0.05)')
                ax1.set_title('Type I Error Rate by Dimension')
                ax1.set_ylabel('Type I Error Rate')
                ax1.legend()
            
            # Plot 2: Statistical Power by Scenario
            if 'power' in results['analysis']:
                power_df = pd.DataFrame(results['analysis']['power'])
                power_df = power_df[power_df['n_projections'].isin([10, np.nan])]
                
                sns.barplot(data=power_df, x='scenario', y='power', 
                           hue='method', ax=ax2)
                ax2.axhline(y=0.8, color='red', linestyle='--', label='Target (≥0.8)')
                ax2.set_title('Statistical Power by Scenario')
                ax2.set_ylabel('Statistical Power')
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
                ax2.legend()
            
            # Plot 3: Computational Time vs Dimension
            if 'efficiency' in results['analysis']:
                eff_df = pd.DataFrame(results['analysis']['efficiency'])
                
                for method in eff_df['method'].unique():
                    method_data = eff_df[eff_df['method'] == method]
                    if method == 'random_projection':
                        method_data = method_data[method_data['n_projections'] == 10]
                    ax3.plot(method_data['dimension'], method_data['mean_time'], 
                            'o-', label=method, linewidth=2, markersize=6)
                
                ax3.set_xlabel('Dimension')
                ax3.set_ylabel('Wall Time (seconds)')
                ax3.set_title('Computational Efficiency')
                ax3.set_yscale('log')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Memory Usage vs Dimension
            if 'efficiency' in results['analysis']:
                for method in eff_df['method'].unique():
                    method_data = eff_df[eff_df['method'] == method]
                    if method == 'random_projection':
                        method_data = method_data[method_data['n_projections'] == 10]
                    ax4.plot(method_data['dimension'], method_data['mean_memory'], 
                            'o-', label=method, linewidth=2, markersize=6)
                
                ax4.set_xlabel('Dimension')
                ax4.set_ylabel('Peak Memory (MB)')
                ax4.set_title('Memory Usage')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'experimental_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plots saved to {self.output_dir / 'experimental_analysis.png'}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")

def main():
    parser = argparse.ArgumentParser(description='Random Projection Testing Experiment')
    parser.add_argument('--output_dir', type=str, default='../results',
                       help='Output directory for results')
    parser.add_argument('--n_trials', type=int, default=30,
                       help='Number of trials per condition')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    print("=" * 60)
    print("Random Projection Testing Experiment")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Initialize experiment runner
    runner = ExperimentRunner(args.output_dir)
    
    try:
        # Run experiments
        runner.run_null_hypothesis_test(args.n_trials)
        runner.run_power_analysis_test(args.n_trials)
        
        # Analyze results
        analysis = runner.analyze_results()
        
        # Generate plots
        runner.generate_plots()
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED")
        print("=" * 60)
        
        success_criteria = analysis.get('success_criteria', {})
        print("Success Criteria Evaluation:")
        print(f"  ✓ Efficiency (>90% reduction): {success_criteria.get('efficiency_reduction', False)}")
        print(f"  ✓ Statistical Validity: {success_criteria.get('statistical_validity', False)}")
        print(f"  ✓ Scalability: {success_criteria.get('scalability', False)}")
        
        overall_success = all(success_criteria.values())
        print(f"\n🔬 Overall Hypothesis: {'SUPPORTED' if overall_success else 'NOT FULLY SUPPORTED'}")
        
        print(f"\n📊 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during experiment execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())