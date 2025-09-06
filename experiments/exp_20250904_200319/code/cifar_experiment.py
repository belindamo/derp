"""
CIFAR-10 DERP-VAE Experiment Runner
Execute comprehensive experiment with statistical rigor and proper controls
"""

import torch
import torch.optim as optim
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

from cifar_data_loader import CIFAR10Subset
from cifar_derp_vae import CIFARDERPVAE, CIFARStandardVAE, compute_cifar_metrics, cifar_statistical_test


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CIFARExperimentRunner:
    """
    Comprehensive experiment runner for CIFAR-10 DERP-VAE validation
    """
    
    def __init__(self, seed: int = 42, device: str = None):
        self.seed = seed
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set reproducibility seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        logger.info(f"Experiment initialized with device: {self.device}, seed: {seed}")
        
        # Experiment configuration
        self.config = {
            'n_samples': 2000,
            'n_classes': 5,
            'latent_dim': 4,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'n_epochs': 20,
            'val_split': 0.2,
            'beta_values': [0.5, 2.0],  # β-VAE variants
            'derp_probes': [3, 5],     # DERP probe counts
            'enforcement_weight': 1.0,
            'classification_weight': 0.5,
            'perceptual_weight': 0.3
        }
        
        # Results storage
        self.results = {}
        
    def load_data(self):
        """Load and prepare CIFAR-10 data"""
        logger.info("Loading CIFAR-10 dataset...")
        
        self.data_loader = CIFAR10Subset(
            n_samples=self.config['n_samples'],
            n_classes=self.config['n_classes'],
            seed=self.seed
        )
        
        # Get train/val splits
        self.train_loader, self.val_loader = self.data_loader.get_train_val_split(
            val_split=self.config['val_split'],
            flatten=False  # Keep spatial dimensions for CNN
        )
        
        # Log dataset info
        info = self.data_loader.get_class_info()
        logger.info(f"Dataset loaded: {info['n_samples']} samples, {info['n_classes']} classes")
        logger.info(f"Class distribution: {info['class_distribution']}")
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
    def create_models(self) -> Dict[str, torch.nn.Module]:
        """Create all model variants for comparison"""
        models = {}
        
        # Standard VAE (baseline)
        models['Standard_VAE'] = CIFARStandardVAE(
            latent_dim=self.config['latent_dim'],
            n_classes=self.config['n_classes']
        ).to(self.device)
        
        # β-VAE variants
        for beta in self.config['beta_values']:
            models[f'Beta_VAE_{beta}'] = CIFARStandardVAE(
                latent_dim=self.config['latent_dim'],
                n_classes=self.config['n_classes']
            ).to(self.device)
        
        # DERP-VAE variants
        for n_probes in self.config['derp_probes']:
            models[f'DERP_VAE_{n_probes}'] = CIFARDERPVAE(
                latent_dim=self.config['latent_dim'],
                n_classes=self.config['n_classes'],
                n_probes=n_probes,
                enforcement_weight=self.config['enforcement_weight'],
                classification_weight=self.config['classification_weight'],
                perceptual_weight=self.config['perceptual_weight'],
                device=self.device
            ).to(self.device)
        
        logger.info(f"Created {len(models)} model variants: {list(models.keys())}")
        return models
    
    def train_model(self, model: torch.nn.Module, model_name: str, 
                   beta: float = 1.0) -> Dict[str, List[float]]:
        """Train a single model and return training history"""
        logger.info(f"Training {model_name} with β={beta}...")
        
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        # Training history
        history = {
            'train_total_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'train_distributional_loss': [],
            'train_classification_loss': [],
            'train_perceptual_loss': [],
            'val_total_loss': [],
            'val_classification_accuracy': [],
            'val_psnr': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.config['n_epochs']):
            # Training phase
            model.train()
            train_losses = {
                'total': 0, 'recon': 0, 'kl': 0, 'distributional': 0, 
                'classification': 0, 'perceptual': 0
            }
            
            for batch_data, batch_labels in self.train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Compute loss (different for different model types)
                if 'DERP' in model_name:
                    loss_dict = model.compute_loss(batch_data, batch_labels, beta=beta)
                else:
                    loss_dict = model.compute_loss(
                        batch_data, batch_labels, beta=beta,
                        classification_weight=self.config['classification_weight'],
                        perceptual_weight=self.config['perceptual_weight']
                    )
                
                loss_dict['total_loss'].backward()
                optimizer.step()
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    if key != 'class_logits' and 'total' not in key:
                        loss_key = key.replace('_loss', '')
                        if loss_key in train_losses:
                            train_losses[loss_key] += value.item()
                    elif key == 'total_loss':
                        train_losses['total'] += value.item()
            
            # Average training losses
            n_batches = len(self.train_loader)
            for key in train_losses:
                train_losses[key] /= n_batches
                if key == 'total':
                    history['train_total_loss'].append(train_losses[key])
                else:
                    history[f'train_{key}_loss'].append(train_losses[key])
            
            # Validation phase
            if (epoch + 1) % 5 == 0 or epoch == self.config['n_epochs'] - 1:
                val_metrics = self.evaluate_model(model, self.val_loader, beta=beta)
                history['val_total_loss'].append(val_metrics['total_loss'])
                history['val_classification_accuracy'].append(val_metrics['classification_accuracy'])
                history['val_psnr'].append(val_metrics['psnr'])
                
                logger.info(f"Epoch {epoch+1}/{self.config['n_epochs']}: "
                           f"Train Loss={train_losses['total']:.4f}, "
                           f"Val Acc={val_metrics['classification_accuracy']:.3f}, "
                           f"Val PSNR={val_metrics['psnr']:.2f}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed for {model_name}: {training_time:.2f}s")
        
        return history, training_time
    
    def evaluate_model(self, model: torch.nn.Module, data_loader, beta: float = 1.0) -> Dict[str, float]:
        """Evaluate model on given data loader"""
        model.eval()
        
        total_losses = {
            'total': 0, 'recon': 0, 'kl': 0, 'distributional': 0,
            'classification': 0, 'perceptual': 0
        }
        all_metrics = []
        
        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                x_recon, class_logits, mu, logvar, z = model(batch_data)
                
                # Compute losses
                if hasattr(model, 'random_probe'):  # DERP model
                    loss_dict = model.compute_loss(batch_data, batch_labels, beta=beta)
                else:  # Standard/β-VAE
                    loss_dict = model.compute_loss(
                        batch_data, batch_labels, beta=beta,
                        classification_weight=self.config['classification_weight'],
                        perceptual_weight=self.config['perceptual_weight']
                    )
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    if key != 'class_logits' and 'total' not in key:
                        loss_key = key.replace('_loss', '')
                        if loss_key in total_losses:
                            total_losses[loss_key] += value.item()
                    elif key == 'total_loss':
                        total_losses['total'] += value.item()
                
                # Compute batch metrics
                batch_metrics = compute_cifar_metrics(
                    mu, logvar, class_logits, batch_labels, batch_data, x_recon
                )
                all_metrics.append(batch_metrics)
        
        # Average all metrics
        n_batches = len(data_loader)
        for key in total_losses:
            total_losses[key] /= n_batches
        
        # Aggregate metrics across batches
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            aggregated_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Combine losses and metrics
        result = {f'{key}_loss': value for key, value in total_losses.items()}
        result.update(aggregated_metrics)
        
        return result
    
    def run_experiment(self) -> Dict[str, Dict]:
        """Run complete experiment with all model variants"""
        logger.info("Starting CIFAR-10 DERP-VAE comprehensive experiment...")
        
        # Load data
        self.load_data()
        
        # Create models
        models = self.create_models()
        
        # Run experiments for each model
        for model_name, model in models.items():
            logger.info(f"\n=== Experiment: {model_name} ===")
            
            # Determine β value for this model
            if 'Beta_VAE_0.5' in model_name:
                beta = 0.5
            elif 'Beta_VAE_2.0' in model_name:
                beta = 2.0
            else:
                beta = 1.0
            
            # Train model
            history, training_time = self.train_model(model, model_name, beta=beta)
            
            # Final evaluation on validation set
            val_metrics = self.evaluate_model(model, self.val_loader, beta=beta)
            
            # Statistical tests on latent representations
            model.eval()
            with torch.no_grad():
                all_z = []
                all_labels = []
                for batch_data, batch_labels in self.val_loader:
                    batch_data = batch_data.to(self.device)
                    _, _, mu, logvar, z = model(batch_data)
                    all_z.append(z)
                    all_labels.append(batch_labels)
                
                all_z = torch.cat(all_z, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                
                statistical_results = cifar_statistical_test(all_z, all_labels)
            
            # Store comprehensive results
            self.results[model_name] = {
                'config': {
                    'beta': beta,
                    'n_probes': self.config['derp_probes'] if 'DERP' in model_name else None,
                    'latent_dim': self.config['latent_dim']
                },
                'training_time': training_time,
                'training_history': history,
                'final_metrics': val_metrics,
                'statistical_tests': statistical_results
            }
            
            # Log key results
            logger.info(f"Results for {model_name}:")
            logger.info(f"  Final Val Accuracy: {val_metrics['classification_accuracy']:.3f}")
            logger.info(f"  Final PSNR: {val_metrics['psnr']:.2f} dB")
            logger.info(f"  KL Divergence: {val_metrics['kl_divergence']:.4f}")
            logger.info(f"  Training Time: {training_time:.2f}s")
            logger.info(f"  Class Separation Ratio: {statistical_results['class_separation_ratio']:.3f}")
        
        logger.info("\nExperiment completed! Running statistical analysis...")
        self.analyze_results()
        
        return self.results
    
    def analyze_results(self):
        """Perform statistical analysis and hypothesis testing"""
        logger.info("\n=== Statistical Analysis ===")
        
        # Extract key metrics for comparison
        model_names = list(self.results.keys())
        baseline_name = 'Standard_VAE'
        
        if baseline_name not in self.results:
            logger.warning(f"Baseline model {baseline_name} not found!")
            return
        
        baseline_metrics = self.results[baseline_name]['final_metrics']
        
        # Hypothesis testing results
        hypothesis_results = {
            'H1_posterior_collapse_prevention': {},
            'H2_performance_maintenance': {},
            'H3_real_data_validation': {}
        }
        
        # H1: Posterior Collapse Prevention
        logger.info("\nH1: Testing Posterior Collapse Prevention")
        baseline_kl = baseline_metrics['kl_divergence']
        
        for model_name in model_names:
            if model_name == baseline_name:
                continue
                
            model_kl = self.results[model_name]['final_metrics']['kl_divergence']
            kl_improvement = (baseline_kl - model_kl) / baseline_kl * 100
            
            # Test if improvement > 10%
            h1_supported = kl_improvement > 10.0
            
            hypothesis_results['H1_posterior_collapse_prevention'][model_name] = {
                'kl_divergence': model_kl,
                'kl_improvement_percent': kl_improvement,
                'supported': h1_supported,
                'threshold': 10.0
            }
            
            logger.info(f"  {model_name}: KL={model_kl:.4f} ({kl_improvement:+.1f}%) - {'✅ SUPPORTED' if h1_supported else '❌ NOT SUPPORTED'}")
        
        # H2: Performance Maintenance
        logger.info("\nH2: Testing Performance Maintenance")
        baseline_acc = baseline_metrics['classification_accuracy']
        
        for model_name in model_names:
            if model_name == baseline_name:
                continue
                
            model_acc = self.results[model_name]['final_metrics']['classification_accuracy']
            acc_change = (model_acc - baseline_acc) * 100
            
            # Test if within ±5% of baseline
            h2_supported = abs(acc_change) <= 5.0
            
            hypothesis_results['H2_performance_maintenance'][model_name] = {
                'classification_accuracy': model_acc,
                'accuracy_change_percent': acc_change,
                'supported': h2_supported,
                'threshold': 5.0
            }
            
            logger.info(f"  {model_name}: Acc={model_acc:.3f} ({acc_change:+.1f}%) - {'✅ MAINTAINED' if h2_supported else '❌ DEGRADED'}")
        
        # H3: Real Data Validation
        logger.info("\nH3: Testing Real Data Validation")
        
        # Look for DERP models with significant benefits
        derp_models = [name for name in model_names if 'DERP' in name]
        h3_evidence = []
        
        for model_name in derp_models:
            metrics = self.results[model_name]['final_metrics']
            stats = self.results[model_name]['statistical_tests']
            
            # Check multiple indicators of success
            kl_better = metrics['kl_divergence'] < baseline_metrics['kl_divergence']
            acc_maintained = abs(metrics['classification_accuracy'] - baseline_acc) <= 0.05
            separation_good = stats['class_separation_ratio'] > 1.0
            
            success_indicators = sum([kl_better, acc_maintained, separation_good])
            h3_supported = success_indicators >= 2  # At least 2 out of 3 indicators
            
            h3_evidence.append(h3_supported)
            
            hypothesis_results['H3_real_data_validation'][model_name] = {
                'kl_better': kl_better,
                'accuracy_maintained': acc_maintained, 
                'separation_good': separation_good,
                'success_indicators': success_indicators,
                'supported': h3_supported
            }
            
            logger.info(f"  {model_name}: {success_indicators}/3 indicators - {'✅ VALIDATED' if h3_supported else '❌ NOT VALIDATED'}")
        
        # Overall H3 result
        overall_h3 = any(h3_evidence)
        hypothesis_results['H3_real_data_validation']['overall_supported'] = overall_h3
        
        logger.info(f"\nH3 Overall: {'✅ DERP benefits transfer to real data' if overall_h3 else '❌ DERP benefits do not transfer'}")
        
        # Store hypothesis results
        self.results['hypothesis_testing'] = hypothesis_results
        
        # Summary statistics
        self.generate_summary_table()
    
    def generate_summary_table(self):
        """Generate summary comparison table"""
        logger.info("\n=== EXPERIMENT SUMMARY TABLE ===")
        
        # Create comparison DataFrame
        rows = []
        for model_name, results in self.results.items():
            if model_name == 'hypothesis_testing':
                continue
                
            metrics = results['final_metrics']
            config = results['config']
            stats = results['statistical_tests']
            
            rows.append({
                'Model': model_name,
                'KL Divergence': f"{metrics['kl_divergence']:.4f}",
                'Classification Acc': f"{metrics['classification_accuracy']:.3f}",
                'PSNR (dB)': f"{metrics['psnr']:.1f}",
                'Class Separation': f"{stats['class_separation_ratio']:.3f}",
                'Training Time (s)': f"{results['training_time']:.1f}",
                'β': config['beta'] if 'beta' in config else '1.0',
                'Probes': config['n_probes'] if config['n_probes'] else '-'
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"\n{df.to_string(index=False)}")
        
        # Store for later use
        self.results['summary_table'] = df.to_dict('records')
    
    def save_results(self, output_dir: str = '../results'):
        """Save comprehensive results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main results
        with open(output_path / 'cifar_experiment_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_for_json(self.results)
            json.dump(json_results, f, indent=2)
        
        # Save summary table
        if 'summary_table' in self.results:
            df = pd.DataFrame(self.results['summary_table'])
            df.to_csv(output_path / 'summary_table.csv', index=False)
        
        logger.info(f"Results saved to {output_path}/")
    
    def _convert_for_json(self, obj):
        """Recursively convert numpy/torch types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj


def main():
    """Run the complete CIFAR-10 DERP-VAE experiment"""
    logger.info("🚀 Starting CIFAR-10 Enhanced DERP-VAE Experiment")
    
    # Initialize experiment
    experiment = CIFARExperimentRunner(seed=42)
    
    # Run comprehensive experiment
    results = experiment.run_experiment()
    
    # Save results
    experiment.save_results()
    
    logger.info("\n🎉 Experiment completed successfully!")
    logger.info("📊 Results saved to ../results/cifar_experiment_results.json")
    
    return results


if __name__ == "__main__":
    results = main()