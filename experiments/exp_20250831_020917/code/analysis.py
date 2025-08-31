#!/usr/bin/env python3
"""
Statistical Analysis for VAE Posterior Collapse Prevention Experiment
Rigorous hypothesis testing with proper statistical controls.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway
import warnings
warnings.filterwarnings('ignore')

def load_results(results_path):
    """Load experimental results from JSON"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def extract_metrics_dataframe(all_results):
    """Convert results to pandas DataFrame for analysis"""
    data = []
    
    for result in all_results:
        config = result['config']
        final_metrics = result['final_val_metrics']
        
        row = {
            'enforcement_method': config['enforcement_method'],
            'probe_count': config['probe_count'], 
            'enforcement_weight': config['enforcement_weight'],
            'seed': config['seed'],
            'training_time': result['training_time'],
            
            # Key metrics
            'elbo': final_metrics['elbo'],
            'reconstruction_loss': final_metrics['reconstruction_loss'],
            'kld': final_metrics['kld'],
            'active_units': final_metrics['active_units'],
            'mutual_info': final_metrics['mutual_info'],
            'avg_ks_statistic': final_metrics['avg_ks_statistic'],
            'enforcement_loss': final_metrics.get('enforcement_loss', 0.0)
        }
        data.append(row)
    
    return pd.DataFrame(data)

def compute_effect_size(group1, group2):
    """Compute Cohen's d effect size"""
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                          (len(group2) - 1) * np.var(group2, ddof=1)) / 
                         (len(group1) + len(group2) - 2))
    
    if pooled_std == 0:
        return 0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def hypothesis_test_primary(df):
    """Test primary hypothesis: Active vs Passive enforcement"""
    print("=" * 60)
    print("PRIMARY HYPOTHESIS TEST: Active vs Passive Enforcement")
    print("=" * 60)
    
    passive = df[df['enforcement_method'] == 'Passive']
    active = df[df['enforcement_method'] == 'Active']
    
    print(f"Sample sizes: Passive (n={len(passive)}), Active (n={len(active)})")
    
    # Test each key metric
    metrics = ['elbo', 'reconstruction_loss', 'kld', 'active_units', 'mutual_info', 'avg_ks_statistic']
    results = {}
    
    for metric in metrics:
        passive_vals = passive[metric].values
        active_vals = active[metric].values
        
        # Normality tests
        _, p_passive = stats.shapiro(passive_vals)
        _, p_active = stats.shapiro(active_vals)
        normal = (p_passive > 0.05) and (p_active > 0.05)
        
        # Choose appropriate test
        if normal and len(passive_vals) > 10 and len(active_vals) > 10:
            # Welch's t-test (unequal variances)
            t_stat, p_value = ttest_ind(passive_vals, active_vals, equal_var=False)
            test_type = "Welch's t-test"
        else:
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = mannwhitneyu(passive_vals, active_vals, alternative='two-sided')
            test_type = "Mann-Whitney U"
            t_stat = u_stat
        
        # Effect size
        effect_size = compute_effect_size(passive_vals, active_vals)
        
        # Confidence intervals
        passive_ci = stats.t.interval(0.95, len(passive_vals)-1, 
                                    loc=np.mean(passive_vals), 
                                    scale=stats.sem(passive_vals))
        active_ci = stats.t.interval(0.95, len(active_vals)-1,
                                   loc=np.mean(active_vals), 
                                   scale=stats.sem(active_vals))
        
        results[metric] = {
            'passive_mean': np.mean(passive_vals),
            'active_mean': np.mean(active_vals),
            'passive_std': np.std(passive_vals, ddof=1),
            'active_std': np.std(active_vals, ddof=1),
            'passive_ci': passive_ci,
            'active_ci': active_ci,
            'test_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'test_type': test_type,
            'significant': p_value < 0.05
        }
        
        print(f"\n{metric.upper()}:")
        print(f"  Passive: {np.mean(passive_vals):.4f} ± {np.std(passive_vals, ddof=1):.4f}")
        print(f"  Active:  {np.mean(active_vals):.4f} ± {np.std(active_vals, ddof=1):.4f}")
        print(f"  {test_type}: stat={t_stat:.4f}, p={p_value:.6f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            effect_interp = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interp = "small"
        elif abs(effect_size) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        print(f"  Effect size interpretation: {effect_interp}")
    
    return results

def check_success_criteria(df, primary_results):
    """Check if success criteria from proposal are met"""
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 60)
    
    criteria_met = {}
    
    # 1. Collapse prevention: >50% reduction in KL collapse
    passive_kld = df[df['enforcement_method'] == 'Passive']['kld'].mean()
    active_kld = df[df['enforcement_method'] == 'Active']['kld'].mean()
    kld_reduction = (passive_kld - active_kld) / passive_kld * 100
    
    collapse_criterion_met = kld_reduction > 50
    criteria_met['collapse_prevention'] = collapse_criterion_met
    
    print(f"1. COLLAPSE PREVENTION (>50% KLD reduction):")
    print(f"   Passive KLD: {passive_kld:.4f}")
    print(f"   Active KLD:  {active_kld:.4f}")
    print(f"   Reduction:   {kld_reduction:.1f}%")
    print(f"   Criterion:   {'✓ MET' if collapse_criterion_met else '✗ NOT MET'}")
    
    # 2. Reconstruction maintenance: <10% degradation
    passive_recon = df[df['enforcement_method'] == 'Passive']['reconstruction_loss'].mean()
    active_recon = df[df['enforcement_method'] == 'Active']['reconstruction_loss'].mean()
    recon_degradation = (active_recon - passive_recon) / passive_recon * 100
    
    recon_criterion_met = recon_degradation < 10
    criteria_met['reconstruction_maintenance'] = recon_criterion_met
    
    print(f"\n2. RECONSTRUCTION MAINTENANCE (<10% degradation):")
    print(f"   Passive Recon: {passive_recon:.4f}")
    print(f"   Active Recon:  {active_recon:.4f}")
    print(f"   Degradation:   {recon_degradation:.1f}%")
    print(f"   Criterion:     {'✓ MET' if recon_criterion_met else '✗ NOT MET'}")
    
    # 3. Efficiency: <20% training time increase
    passive_time = df[df['enforcement_method'] == 'Passive']['training_time'].mean()
    active_time = df[df['enforcement_method'] == 'Active']['training_time'].mean()
    time_increase = (active_time - passive_time) / passive_time * 100
    
    efficiency_criterion_met = time_increase < 20
    criteria_met['efficiency'] = efficiency_criterion_met
    
    print(f"\n3. EFFICIENCY (<20% time increase):")
    print(f"   Passive Time: {passive_time:.1f}s")
    print(f"   Active Time:  {active_time:.1f}s") 
    print(f"   Increase:     {time_increase:.1f}%")
    print(f"   Criterion:    {'✓ MET' if efficiency_criterion_met else '✗ NOT MET'}")
    
    # Overall success
    overall_success = all(criteria_met.values())
    print(f"\n{'='*20}")
    print(f"OVERALL SUCCESS: {'✓ ALL CRITERIA MET' if overall_success else '✗ SOME CRITERIA NOT MET'}")
    print(f"{'='*20}")
    
    return criteria_met

def multiple_comparisons_correction(results, alpha=0.05):
    """Apply Bonferroni correction for multiple comparisons"""
    p_values = [results[metric]['p_value'] for metric in results.keys()]
    n_comparisons = len(p_values)
    corrected_alpha = alpha / n_comparisons
    
    print(f"\nMULTIPLE COMPARISONS CORRECTION (Bonferroni):")
    print(f"Original α: {alpha}")
    print(f"Number of comparisons: {n_comparisons}")
    print(f"Corrected α: {corrected_alpha:.6f}")
    
    significant_after_correction = []
    for metric, result in results.items():
        significant = result['p_value'] < corrected_alpha
        significant_after_correction.append(significant)
        print(f"{metric}: p={result['p_value']:.6f}, significant={significant}")
    
    return significant_after_correction, corrected_alpha

def generate_visualizations(df, results_dir):
    """Generate statistical visualizations"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    metrics = ['elbo', 'reconstruction_loss', 'kld', 'active_units', 'mutual_info', 'avg_ks_statistic']
    metric_names = ['ELBO', 'Reconstruction Loss', 'KL Divergence', 'Active Units', 'Mutual Info', 'KS Statistic']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # Box plots comparing Active vs Passive
        df_plot = df.copy()
        sns.boxplot(data=df_plot, x='enforcement_method', y=metric, ax=ax)
        ax.set_title(name)
        ax.set_xlabel('Enforcement Method')
        
        # Add statistical significance annotation
        passive_vals = df[df['enforcement_method'] == 'Passive'][metric]
        active_vals = df[df['enforcement_method'] == 'Active'][metric]
        _, p_val = ttest_ind(passive_vals, active_vals)
        
        if p_val < 0.001:
            sig_text = "p < 0.001***"
        elif p_val < 0.01:
            sig_text = f"p = {p_val:.3f}**"
        elif p_val < 0.05:
            sig_text = f"p = {p_val:.3f}*"
        else:
            sig_text = f"p = {p_val:.3f}ns"
        
        ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(results_dir / 'statistical_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Training time comparison
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='enforcement_method', y='training_time')
    plt.title('Training Time Comparison')
    plt.ylabel('Training Time (seconds)')
    plt.savefig(results_dir / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {results_dir}/")

def main():
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Statistical Analysis')
    parser.add_argument('--results_path', type=str, default='../results/all_results.json')
    parser.add_argument('--output_dir', type=str, default='../results')
    args = parser.parse_args()
    
    results_dir = Path(args.output_dir)
    
    # Load results
    print("Loading experimental results...")
    all_results = load_results(args.results_path)
    
    if len(all_results) == 0:
        print("No results found! Make sure experiments completed successfully.")
        return
    
    # Convert to DataFrame
    df = extract_metrics_dataframe(all_results)
    print(f"Loaded {len(df)} experimental runs")
    print(f"Methods: {df['enforcement_method'].value_counts().to_dict()}")
    
    # Primary hypothesis testing
    primary_results = hypothesis_test_primary(df)
    
    # Check success criteria
    criteria_results = check_success_criteria(df, primary_results)
    
    # Multiple comparisons correction
    significant_corrected, corrected_alpha = multiple_comparisons_correction(primary_results)
    
    # Generate visualizations
    generate_visualizations(df, results_dir)
    
    # Save statistical analysis
    analysis_report = {
        'primary_results': primary_results,
        'success_criteria': criteria_results,
        'multiple_comparisons': {
            'corrected_alpha': corrected_alpha,
            'significant_after_correction': dict(zip(primary_results.keys(), significant_corrected))
        },
        'sample_sizes': {
            'passive': len(df[df['enforcement_method'] == 'Passive']),
            'active': len(df[df['enforcement_method'] == 'Active'])
        }
    }
    
    with open(results_dir / 'statistical_analysis.json', 'w') as f:
        json.dump(analysis_report, f, indent=2, default=str)
    
    print(f"\nStatistical analysis complete!")
    print(f"Detailed results saved to: {results_dir}/statistical_analysis.json")

if __name__ == '__main__':
    main()