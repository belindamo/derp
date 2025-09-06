"""
Enhanced DERP-VAE Experiment with Multi-Loss Framework on CIFAR-10
Testing H1 & H2 with reduced latent dimensions and comprehensive evaluation

Key Features:
- Hidden dims reduced from 32 to 4 for challenging posterior collapse conditions
- Multi-loss framework: Classification + Reconstruction + Perceptual + Modified KS
- CIFAR-10 dataset with 10 classes
- Statistical hypothesis testing with proper controls
- Enhanced metrics including classification accuracy and class separation
"""
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import logging
from pathlib import Path
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats
from typing import Dict, Any

# Add current directory to Python path
sys.path.append('.')

# Import our enhanced modules
from derp_vae import (
    DERP_VAE, EnhancedStandardVAE as StandardVAE, 
    compute_enhanced_metrics, enhanced_statistical_test
)
from data_loader import (
    get_cifar10_dataloaders, analyze_dataset_properties, 
    verify_data_quality
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_cifar_dataset(batch_size: int = 64):
    """Get CIFAR-10 dataset loaders"""
    logger.info("Loading CIFAR-10 dataset")
    
    # Get CIFAR-10 dataloaders
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=batch_size,
        normalize=False,  # Don't normalize for BCE loss compatibility
        download=True,
        data_dir='../../../data/vision'  # Path to data/vision directory
    )
    
    logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    return train_loader, test_loader


def run_enhanced_experiment(model, train_loader, test_loader, epochs: int = 30, 
                          beta: float = 1.0, model_name: str = "Model"):
    """Run enhanced training and evaluation with comprehensive metrics"""
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    train_losses = []
    test_metrics = []
    loss_components = {'recon': [], 'kl': [], 'distributional': [], 'classification': [], 'perceptual': []}
    
    start_time = time.time()
    
    logger.info(f"Starting training for {model_name}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_losses = {k: 0.0 for k in ['total', 'recon', 'kl', 'distributional', 'classification', 'perceptual']}
        n_batches = 0
        
        for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            
            # Forward pass with multi-loss computation
            loss_dict = model.compute_loss(batch_data, batch_labels, beta=beta)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses.keys():
                if key == 'total':
                    epoch_losses[key] += loss.item()
                else:
                    loss_key = key + '_loss'
                    if loss_key in loss_dict:
                        epoch_losses[key] += loss_dict[loss_key].item()
            
            n_batches += 1
        
        # Average epoch losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= n_batches
        
        train_losses.append(epoch_losses['total'])
        
        # Store loss components
        for key in loss_components.keys():
            loss_components[key].append(epoch_losses[key])
        
        # Learning rate scheduling
        scheduler.step(epoch_losses['total'])
        
        # Evaluation every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            test_loss = 0.0
            test_loss_components = {k: 0.0 for k in loss_components.keys()}
            test_batches = 0
            
            all_latents = []
            all_mus = []
            all_logvars = []
            all_class_logits = []
            all_true_labels = []
            
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    # Compute test loss (CIFAR data comes as (data, label) tuples)
                    loss_dict = model.compute_loss(batch_data, batch_labels, beta=beta)
                    test_loss += loss_dict['total_loss'].item()
                    test_batches += 1
                    
                    # Accumulate loss components
                    test_loss_components['recon'] += loss_dict['recon_loss'].item()
                    test_loss_components['kl'] += loss_dict['kl_loss'].item()
                    test_loss_components['distributional'] += loss_dict['distributional_loss'].item()
                    test_loss_components['classification'] += loss_dict['classification_loss'].item()
                    test_loss_components['perceptual'] += loss_dict['perceptual_loss'].item()
                    
                    # Collect samples for comprehensive analysis
                    _, class_logits, mu, logvar, z = model.forward(batch_data)
                    all_latents.append(z)
                    all_mus.append(mu)
                    all_logvars.append(logvar)
                    all_class_logits.append(class_logits)
                    all_true_labels.append(batch_labels)
            
            # Concatenate all test data
            all_latents = torch.cat(all_latents, dim=0)
            all_mus = torch.cat(all_mus, dim=0)
            all_logvars = torch.cat(all_logvars, dim=0)
            all_class_logits = torch.cat(all_class_logits, dim=0)
            all_true_labels = torch.cat(all_true_labels, dim=0)
            
            # Compute comprehensive metrics
            enhanced_metrics = compute_enhanced_metrics(all_mus, all_logvars, all_class_logits, all_true_labels)
            
            # Statistical normality testing with label awareness
            statistical_metrics = enhanced_statistical_test(all_latents, all_true_labels)
            
            # Average test losses
            test_loss /= test_batches
            for key in test_loss_components.keys():
                test_loss_components[key] /= test_batches
            
            # Compile test metrics
            test_metric = {
                'epoch': epoch + 1,
                'test_loss': test_loss,
                **{f'test_{k}_loss': v for k, v in test_loss_components.items()},
                **enhanced_metrics,
                **statistical_metrics
            }
            
            test_metrics.append(test_metric)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}: Train Loss={epoch_losses['total']:.4f}, "
                f"Test Loss={test_loss:.4f}, "
                f"KL Div={enhanced_metrics['kl_divergence']:.4f}, "
                f"Classification Acc={enhanced_metrics['classification_accuracy']:.4f}, "
                f"Activation Rate={enhanced_metrics['activation_rate']:.4f}"
            )
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'loss_components': loss_components,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'final_metrics': test_metrics[-1] if test_metrics else {}
    }


def statistical_hypothesis_testing(results: Dict) -> Dict[str, Any]:
    """Conduct rigorous statistical hypothesis testing"""
    
    logger.info("Conducting statistical hypothesis testing...")
    
    # Extract final metrics for comparison
    baseline_name = 'Enhanced_Standard_VAE'
    derp_name = 'DERP_VAE_5probes'
    
    if baseline_name not in results or derp_name not in results:
        logger.warning("Required models not found for statistical testing")
        return {}
    
    baseline = results[baseline_name]['final_metrics']
    derp = results[derp_name]['final_metrics']
    
    # Hypothesis H1: DERP-VAE reduces posterior collapse (lower KL divergence)
    kl_improvement = (baseline['kl_divergence'] - derp['kl_divergence']) / baseline['kl_divergence'] * 100
    h1_threshold = 10.0  # Looking for >10% improvement
    h1_supported = kl_improvement > h1_threshold
    
    # Hypothesis H2: DERP-VAE maintains/improves classification performance
    classification_improvement = (derp['classification_accuracy'] - baseline['classification_accuracy']) * 100
    h2_threshold = -5.0  # Allow up to 5% degradation
    h2_supported = classification_improvement > h2_threshold
    
    # Additional hypothesis: DERP-VAE improves class separation in latent space
    class_sep_improvement = (derp.get('class_separation_ratio', 0) - baseline.get('class_separation_ratio', 0))
    h3_supported = class_sep_improvement > 0.1
    
    # Effect sizes
    cohen_d_kl = (baseline['kl_divergence'] - derp['kl_divergence']) / np.sqrt(
        (baseline.get('kl_variance', 0.01) + derp.get('kl_variance', 0.01)) / 2
    )
    
    statistical_results = {
        'h1_posterior_collapse': {
            'supported': h1_supported,
            'improvement_percent': kl_improvement,
            'threshold': h1_threshold,
            'effect_size_cohens_d': cohen_d_kl,
            'significance': 'large' if abs(cohen_d_kl) > 0.8 else 'medium' if abs(cohen_d_kl) > 0.5 else 'small'
        },
        'h2_classification_performance': {
            'supported': h2_supported,
            'improvement_percent': classification_improvement,
            'threshold': h2_threshold,
            'baseline_accuracy': baseline['classification_accuracy'],
            'derp_accuracy': derp['classification_accuracy']
        },
        'h3_class_separation': {
            'supported': h3_supported,
            'improvement': class_sep_improvement,
            'baseline_ratio': baseline.get('class_separation_ratio', 0),
            'derp_ratio': derp.get('class_separation_ratio', 0)
        },
        'overall_assessment': {
            'hypotheses_supported': sum([h1_supported, h2_supported, h3_supported]),
            'total_hypotheses': 3,
            'success_rate': (sum([h1_supported, h2_supported, h3_supported]) / 3) * 100
        }
    }
    
    return statistical_results


def create_visualizations(results: Dict, output_dir: Path):
    """Create comprehensive visualizations of experimental results"""
    
    logger.info("Creating experimental visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Training curves comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Enhanced DERP-VAE Experiment Results', fontsize=16)
    
    # Training loss curves
    ax1 = axes[0, 0]
    for model_name, result in results.items():
        ax1.plot(result['train_losses'], label=model_name, linewidth=2)
    ax1.set_title('Training Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # KL divergence comparison
    ax2 = axes[0, 1]
    models = list(results.keys())
    kl_values = [results[model]['final_metrics']['kl_divergence'] for model in models]
    ax2.bar(range(len(models)), kl_values, alpha=0.7)
    ax2.set_title('Posterior Collapse (KL Divergence)')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('KL Divergence')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.replace('_', ' ') for m in models], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Classification accuracy comparison
    ax3 = axes[1, 0]
    acc_values = [results[model]['final_metrics']['classification_accuracy'] for model in models]
    ax3.bar(range(len(models)), acc_values, alpha=0.7, color='orange')
    ax3.set_title('Classification Accuracy')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Accuracy')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels([m.replace('_', ' ') for m in models], rotation=45)
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3)
    
    # Loss components for DERP-VAE
    ax4 = axes[1, 1]
    derp_model = 'DERP_VAE_5probes'
    if derp_model in results:
        components = results[derp_model]['loss_components']
        epochs = range(1, len(components['recon']) + 1)
        
        ax4.plot(epochs, components['recon'], label='Reconstruction', linewidth=2)
        ax4.plot(epochs, components['kl'], label='KL Divergence', linewidth=2)
        ax4.plot(epochs, components['distributional'], label='Distributional (KS)', linewidth=2)
        ax4.plot(epochs, components['classification'], label='Classification', linewidth=2)
        ax4.plot(epochs, components['perceptual'], label='Perceptual', linewidth=2)
        
        ax4.set_title('DERP-VAE Loss Components')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Run enhanced DERP-VAE experiment"""
    logger.info("Starting Enhanced DERP-VAE Experiment")
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader = get_cifar_dataset(batch_size=128)
    
    # Analyze dataset properties
    dataset_stats = analyze_dataset_properties(train_loader)
    logger.info(f"Dataset statistics: {dataset_stats}")
    
    # Enhanced model configurations for CIFAR-10
    base_config = {
        'input_dim': 3072,  # 32*32*3 for CIFAR-10
        'hidden_dim': 256,  # Increased for image data
        'latent_dim': 4,    # KEY CHANGE: Reduced from 32 to 4
        'n_classes': 10     # CIFAR-10 has 10 classes
    }
    
    # Define experiments with enhanced multi-loss models
    experiments = [
        ('Enhanced_Standard_VAE', StandardVAE(**base_config), 1.0),
        ('Enhanced_Beta_VAE_0.5', StandardVAE(**base_config), 0.5),
        ('Enhanced_Beta_VAE_2.0', StandardVAE(**base_config), 2.0),
        ('DERP_VAE_3probes', DERP_VAE(**base_config, n_probes=3, enforcement_weight=1.0, device='cpu'), 1.0),
        ('DERP_VAE_5probes', DERP_VAE(**base_config, n_probes=5, enforcement_weight=1.0, device='cpu'), 1.0),
    ]
    
    results = {}
    
    # Run all experiments
    for name, model, beta in experiments:
        logger.info(f"\\n{'='*60}")
        logger.info(f"Running: {name}")
        logger.info(f"Beta: {beta}, Latent Dim: {base_config['latent_dim']}")
        logger.info(f"{'='*60}")
        
        result = run_enhanced_experiment(
            model, train_loader, test_loader, 
            epochs=30, beta=beta, model_name=name
        )
        result['model_name'] = name
        result['beta'] = beta
        result['config'] = base_config.copy()
        results[name] = result
        
        # Print summary
        final = result['final_metrics']
        logger.info(f"\\n{name} Final Results:")
        logger.info(f"  KL Divergence: {final.get('kl_divergence', 0):.4f}")
        logger.info(f"  Classification Accuracy: {final.get('classification_accuracy', 0):.4f}")
        logger.info(f"  Activation Rate: {final.get('activation_rate', 0):.4f}")
        logger.info(f"  Class Separation Ratio: {final.get('class_separation_ratio', 0):.4f}")
        logger.info(f"  Normality Compliance: {final.get('normality_compliance', 0):.4f}")
        if 'test_ks_distance_loss' in final:
            logger.info(f"  KS Distance: {final.get('test_ks_distance_loss', 0):.4f}")
        logger.info(f"  Training Time: {result['training_time']:.2f}s")
    
    # Statistical hypothesis testing
    statistical_results = statistical_hypothesis_testing(results)
    
    logger.info("\\n" + "="*70)
    logger.info("STATISTICAL HYPOTHESIS TESTING RESULTS")
    logger.info("="*70)
    
    if statistical_results:
        h1 = statistical_results['h1_posterior_collapse']
        h2 = statistical_results['h2_classification_performance']
        h3 = statistical_results['h3_class_separation']
        overall = statistical_results['overall_assessment']
        
        logger.info(f"\\nH1 (Posterior Collapse Prevention): {'✅ SUPPORTED' if h1['supported'] else '❌ NOT SUPPORTED'}")
        logger.info(f"  KL Divergence Improvement: {h1['improvement_percent']:.1f}% (threshold: {h1['threshold']:.1f}%)")
        logger.info(f"  Effect Size (Cohen's d): {h1['effect_size_cohens_d']:.3f} ({h1['significance']})")
        
        logger.info(f"\\nH2 (Classification Performance): {'✅ MAINTAINED' if h2['supported'] else '❌ DEGRADED'}")
        logger.info(f"  Accuracy Change: {h2['improvement_percent']:.1f}% (threshold: >{h2['threshold']:.1f}%)")
        logger.info(f"  Baseline: {h2['baseline_accuracy']:.3f}, DERP: {h2['derp_accuracy']:.3f}")
        
        logger.info(f"\\nH3 (Class Separation): {'✅ IMPROVED' if h3['supported'] else '❌ NO IMPROVEMENT'}")
        logger.info(f"  Separation Ratio Improvement: {h3['improvement']:.3f}")
        logger.info(f"  Baseline: {h3['baseline_ratio']:.3f}, DERP: {h3['derp_ratio']:.3f}")
        
        logger.info(f"\\nOVERALL ASSESSMENT:")
        logger.info(f"  Hypotheses Supported: {overall['hypotheses_supported']}/{overall['total_hypotheses']}")
        logger.info(f"  Success Rate: {overall['success_rate']:.1f}%")
    
    # Save results
    results_path = Path("../results")
    results_path.mkdir(exist_ok=True)
    
    # Create visualizations
    create_visualizations(results, results_path)
    
    # Save detailed results
    with open(results_path / "enhanced_experiment_results.json", 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for k, v in results.items():
            json_results[k] = {
                'model_name': v['model_name'],
                'beta': v['beta'],
                'config': v['config'],
                'training_time': v['training_time'],
                'final_metrics': v['final_metrics'],
                'train_losses': v['train_losses'],
                'test_metrics': v['test_metrics'][-5:],  # Last 5 evaluations
            }
        
        # Add statistical results
        json_results['statistical_analysis'] = statistical_results
        json_results['dataset_statistics'] = dataset_stats
        
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\\nDetailed results saved to {results_path / 'enhanced_experiment_results.json'}")
    
    return results, statistical_results


if __name__ == "__main__":
    results, statistical_results = main()