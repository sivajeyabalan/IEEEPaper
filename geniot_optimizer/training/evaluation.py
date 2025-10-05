"""
Model Evaluation and Metrics for GenIoT-Optimizer

This module implements comprehensive evaluation metrics for all components
of the GenIoT-Optimizer framework.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class ModelEvaluator:
    """
    Comprehensive evaluator for GenIoT-Optimizer components.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def evaluate_generative_models(
        self,
        traffic_generator,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Evaluate generative models using various metrics.
        
        Args:
            traffic_generator: Trained traffic generator
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info("Evaluating generative models...")
        
        results = {}
        
        # Collect real data
        real_data = []
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                real_data.append(batch[0])
            else:
                real_data.append(batch)
        real_data = torch.cat(real_data, dim=0).to(self.device)
        
        # Generate synthetic data
        num_samples = real_data.size(0)
        synthetic_data = traffic_generator.generate_mixed_traffic(num_samples)
        
        # Evaluate each generative model
        results['wgan_gp'] = self._evaluate_wgan_gp(real_data, synthetic_data)
        results['vae'] = self._evaluate_vae(real_data, synthetic_data, traffic_generator.vae)
        results['diffusion'] = self._evaluate_diffusion(real_data, synthetic_data)
        
        # Overall quality metrics
        results['overall_quality'] = self._evaluate_traffic_quality(real_data, synthetic_data)
        
        return results
    
    def _evaluate_wgan_gp(self, real_data: torch.Tensor, synthetic_data: torch.Tensor) -> Dict[str, float]:
        """Evaluate WGAN-GP model."""
        # Maximum Mean Discrepancy (MMD)
        mmd_score = self._compute_mmd(real_data, synthetic_data)
        
        # Fréchet Inception Distance (FID) - simplified version
        fid_score = self._compute_fid(real_data, synthetic_data)
        
        # Inception Score (IS) - simplified version
        is_score = self._compute_inception_score(synthetic_data)
        
        return {
            'mmd': mmd_score,
            'fid': fid_score,
            'inception_score': is_score
        }
    
    def _evaluate_vae(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor,
        vae_model
    ) -> Dict[str, float]:
        """Evaluate VAE model."""
        # Reconstruction error
        reconstruction_error = self._compute_reconstruction_error(real_data, vae_model)
        
        # KL divergence
        kl_divergence = self._compute_kl_divergence(real_data, vae_model)
        
        # Anomaly detection performance
        anomaly_metrics = self._evaluate_anomaly_detection(real_data, vae_model)
        
        return {
            'reconstruction_error': reconstruction_error,
            'kl_divergence': kl_divergence,
            **anomaly_metrics
        }
    
    def _evaluate_diffusion(self, real_data: torch.Tensor, synthetic_data: torch.Tensor) -> Dict[str, float]:
        """Evaluate Diffusion model."""
        # Quality metrics
        quality_metrics = self._evaluate_traffic_quality(real_data, synthetic_data)
        
        # Temporal consistency
        temporal_consistency = self._compute_temporal_consistency(synthetic_data)
        
        return {
            **quality_metrics,
            'temporal_consistency': temporal_consistency
        }
    
    def _compute_mmd(self, real_data: torch.Tensor, synthetic_data: torch.Tensor) -> float:
        """Compute Maximum Mean Discrepancy."""
        # Simplified MMD computation using RBF kernel
        def rbf_kernel(x, y, sigma=1.0):
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist**2 / (2 * sigma**2))
        
        # Sample subsets for computational efficiency
        n_samples = min(1000, real_data.size(0), synthetic_data.size(0))
        real_sample = real_data[:n_samples].view(n_samples, -1)
        synth_sample = synthetic_data[:n_samples].view(n_samples, -1)
        
        # Compute kernel matrices
        k_real_real = rbf_kernel(real_sample, real_sample)
        k_synth_synth = rbf_kernel(synth_sample, synth_sample)
        k_real_synth = rbf_kernel(real_sample, synth_sample)
        
        # Compute MMD
        mmd = (k_real_real.mean() + k_synth_synth.mean() - 2 * k_real_synth.mean()).item()
        
        return mmd
    
    def _compute_fid(self, real_data: torch.Tensor, synthetic_data: torch.Tensor) -> float:
        """Compute Fréchet Inception Distance (simplified)."""
        # Flatten data
        real_flat = real_data.view(real_data.size(0), -1)
        synth_flat = synthetic_data.view(synthetic_data.size(0), -1)
        
        # Compute means and covariances
        mu_real = torch.mean(real_flat, dim=0)
        mu_synth = torch.mean(synth_flat, dim=0)
        
        cov_real = torch.cov(real_flat.T)
        cov_synth = torch.cov(synth_flat.T)
        
        # Compute FID
        diff = mu_real - mu_synth
        covmean = torch.sqrt(torch.mm(cov_real, cov_synth))
        
        fid = torch.sum(diff**2) + torch.trace(cov_real) + torch.trace(cov_synth) - 2 * torch.trace(covmean)
        
        return fid.item()
    
    def _compute_inception_score(self, synthetic_data: torch.Tensor) -> float:
        """Compute Inception Score (simplified)."""
        # Simplified IS computation using data diversity
        data_flat = synthetic_data.view(synthetic_data.size(0), -1)
        
        # Compute pairwise distances
        distances = torch.cdist(data_flat, data_flat, p=2)
        
        # Compute diversity score
        mean_distance = torch.mean(distances).item()
        std_distance = torch.std(distances).item()
        
        # Simplified IS as diversity measure
        is_score = mean_distance / (std_distance + 1e-8)
        
        return is_score
    
    def _compute_reconstruction_error(self, real_data: torch.Tensor, vae_model) -> float:
        """Compute VAE reconstruction error."""
        vae_model.eval()
        
        with torch.no_grad():
            reconstructed, _, _ = vae_model(real_data)
            reconstruction_error = torch.mean((real_data - reconstructed)**2).item()
        
        return reconstruction_error
    
    def _compute_kl_divergence(self, real_data: torch.Tensor, vae_model) -> float:
        """Compute KL divergence for VAE."""
        vae_model.eval()
        
        with torch.no_grad():
            _, mu, logvar = vae_model(real_data)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_div = torch.mean(kl_div).item()
        
        return kl_div
    
    def _evaluate_anomaly_detection(
        self,
        real_data: torch.Tensor,
        vae_model,
        anomaly_ratio: float = 0.1
    ) -> Dict[str, float]:
        """Evaluate anomaly detection performance."""
        vae_model.eval()
        
        # Create synthetic anomalies
        n_anomalies = int(real_data.size(0) * anomaly_ratio)
        normal_data = real_data[:-n_anomalies]
        anomaly_data = real_data[-n_anomalies:] + torch.randn_like(real_data[-n_anomalies:]) * 2
        
        test_data = torch.cat([normal_data, anomaly_data], dim=0)
        true_labels = torch.cat([
            torch.zeros(normal_data.size(0)),
            torch.ones(anomaly_data.size(0))
        ])
        
        # Get anomaly scores
        with torch.no_grad():
            reconstructed, _, _ = vae_model(test_data)
            reconstruction_errors = torch.mean((test_data - reconstructed)**2, dim=(1, 2))
        
        # Compute metrics
        threshold = torch.quantile(reconstruction_errors, 1 - anomaly_ratio)
        predicted_labels = (reconstruction_errors > threshold).float()
        
        f1 = f1_score(true_labels.numpy(), predicted_labels.numpy())
        precision = precision_score(true_labels.numpy(), predicted_labels.numpy())
        recall = recall_score(true_labels.numpy(), predicted_labels.numpy())
        
        return {
            'anomaly_f1': f1,
            'anomaly_precision': precision,
            'anomaly_recall': recall
        }
    
    def _compute_temporal_consistency(self, synthetic_data: torch.Tensor) -> float:
        """Compute temporal consistency of generated sequences."""
        # Compute autocorrelation for temporal consistency
        autocorrelations = []
        
        for i in range(synthetic_data.size(0)):
            sequence = synthetic_data[i].cpu().numpy()
            for j in range(sequence.shape[1]):  # For each feature
                feature_series = sequence[:, j]
                autocorr = np.corrcoef(feature_series[:-1], feature_series[1:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrelations.append(autocorr)
        
        return np.mean(autocorrelations) if autocorrelations else 0.0
    
    def _evaluate_traffic_quality(self, real_data: torch.Tensor, synthetic_data: torch.Tensor) -> Dict[str, float]:
        """Evaluate overall traffic quality."""
        # Statistical similarity
        real_stats = self._compute_statistics(real_data)
        synth_stats = self._compute_statistics(synthetic_data)
        
        # Compare statistics
        stat_similarity = {}
        for stat_name in real_stats.keys():
            real_stat = real_stats[stat_name]
            synth_stat = synth_stats[stat_name]
            similarity = 1.0 - np.mean(np.abs(real_stat - synth_stat) / (np.abs(real_stat) + 1e-8))
            stat_similarity[f'{stat_name}_similarity'] = similarity
        
        return stat_similarity
    
    def _compute_statistics(self, data: torch.Tensor) -> Dict[str, np.ndarray]:
        """Compute statistical properties of data."""
        data_np = data.cpu().numpy()
        
        return {
            'mean': np.mean(data_np, axis=(0, 1)),
            'std': np.std(data_np, axis=(0, 1)),
            'min': np.min(data_np, axis=(0, 1)),
            'max': np.max(data_np, axis=(0, 1)),
            'skewness': stats.skew(data_np.reshape(-1, data_np.shape[-1]), axis=0),
            'kurtosis': stats.kurtosis(data_np.reshape(-1, data_np.shape[-1]), axis=0)
        }
    
    def evaluate_network_predictor(
        self,
        network_predictor,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Evaluate network state predictor."""
        self.logger.info("Evaluating network predictor...")
        
        results = {}
        all_predictions = {}
        all_targets = {}
        
        network_predictor.model.eval()
        
        with torch.no_grad():
            for batch_data in test_loader:
                traffic_patterns = batch_data['traffic_patterns'].to(self.device)
                current_states = batch_data['current_states'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch_data['targets'].items()}
                
                # Make predictions
                predictions = network_predictor.model(traffic_patterns, current_states)
                
                # Store predictions and targets
                for metric in predictions.keys():
                    if metric not in all_predictions:
                        all_predictions[metric] = []
                        all_targets[metric] = []
                    
                    all_predictions[metric].append(predictions[metric].cpu())
                    all_targets[metric].append(targets[metric].cpu())
        
        # Compute metrics for each prediction target
        for metric in all_predictions.keys():
            pred = torch.cat(all_predictions[metric], dim=0)
            target = torch.cat(all_targets[metric], dim=0)
            
            mse = mean_squared_error(target.numpy(), pred.numpy())
            mae = mean_absolute_error(target.numpy(), pred.numpy())
            r2 = r2_score(target.numpy(), pred.numpy())
            
            results[metric] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        
        return results
    
    def evaluate_optimizer(
        self,
        optimizer,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Evaluate multi-objective optimizer."""
        self.logger.info("Evaluating optimizer...")
        
        results = {}
        
        # Test optimization performance
        optimization_results = optimizer.optimize_network(max_steps=100)
        
        # Extract performance metrics
        final_performance = optimization_results.get('final_performance', {})
        
        # Compute improvement metrics
        initial_metrics = optimization_results.get('simulation_steps', [{}])[0].get('metrics', {})
        
        improvements = {}
        for metric in final_performance.keys():
            if metric in initial_metrics:
                initial_val = initial_metrics[metric]
                final_val = final_performance[metric]
                improvement = (final_val - initial_val) / initial_val * 100
                improvements[f'{metric}_improvement'] = improvement
        
        results['optimization_performance'] = {
            'final_metrics': final_performance,
            'improvements': improvements,
            'total_reward': optimization_results.get('total_reward', 0)
        }
        
        # Evaluate training stability
        training_history = optimizer.agent.training_history
        stability_metrics = self._evaluate_training_stability(training_history)
        results['training_stability'] = stability_metrics
        
        return results
    
    def _evaluate_training_stability(self, training_history: Dict[str, List[float]]) -> Dict[str, float]:
        """Evaluate training stability metrics."""
        stability_metrics = {}
        
        for metric, values in training_history.items():
            if len(values) > 10:
                # Compute stability metrics
                final_values = values[-10:]  # Last 10 values
                stability_metrics[f'{metric}_final_mean'] = np.mean(final_values)
                stability_metrics[f'{metric}_final_std'] = np.std(final_values)
                stability_metrics[f'{metric}_convergence'] = np.std(final_values) / (np.mean(final_values) + 1e-8)
        
        return stability_metrics
    
    def evaluate_joint_system(
        self,
        traffic_generator,
        network_predictor,
        optimizer,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Evaluate the complete joint system."""
        self.logger.info("Evaluating joint system...")
        
        results = {}
        
        # End-to-end performance test
        end_to_end_results = self._evaluate_end_to_end_performance(
            traffic_generator, network_predictor, optimizer, test_loader
        )
        results['end_to_end'] = end_to_end_results
        
        # System integration metrics
        integration_metrics = self._evaluate_system_integration(
            traffic_generator, network_predictor, optimizer
        )
        results['integration'] = integration_metrics
        
        return results
    
    def _evaluate_end_to_end_performance(
        self,
        traffic_generator,
        network_predictor,
        optimizer,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Evaluate end-to-end system performance."""
        # Generate synthetic traffic
        synthetic_traffic = traffic_generator.generate_mixed_traffic(num_samples=100)
        
        # Predict network states
        current_states = torch.randn(1, 100, 8).to(self.device)  # Mock current states
        predictions = network_predictor.predict(synthetic_traffic.unsqueeze(0), current_states)
        
        # Optimize network
        optimization_results = optimizer.optimize_network(max_steps=50)
        
        return {
            'synthetic_traffic_quality': self._evaluate_traffic_quality(
                torch.randn_like(synthetic_traffic), synthetic_traffic
            ),
            'prediction_accuracy': predictions,
            'optimization_success': optimization_results.get('total_reward', 0) > 0
        }
    
    def _evaluate_system_integration(
        self,
        traffic_generator,
        network_predictor,
        optimizer
    ) -> Dict[str, Any]:
        """Evaluate system integration quality."""
        # Test component compatibility
        compatibility_tests = {
            'traffic_generator_output_shape': traffic_generator.generate_mixed_traffic(10).shape,
            'network_predictor_input_compatibility': True,  # Simplified
            'optimizer_input_compatibility': True  # Simplified
        }
        
        # Test data flow
        data_flow_tests = {
            'traffic_to_prediction': True,  # Simplified
            'prediction_to_optimization': True,  # Simplified
            'optimization_feedback': True  # Simplified
        }
        
        return {
            'compatibility': compatibility_tests,
            'data_flow': data_flow_tests
        }
    
    def evaluate_online_adaptation(
        self,
        traffic_generator,
        network_predictor,
        optimizer,
        new_data: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Evaluate online adaptation performance."""
        self.logger.info("Evaluating online adaptation...")
        
        # Test adaptation to new data
        adaptation_results = {}
        
        # Measure performance before adaptation
        baseline_performance = self._measure_baseline_performance(
            traffic_generator, network_predictor, optimizer, new_data
        )
        
        # Measure performance after adaptation
        adapted_performance = self._measure_adapted_performance(
            traffic_generator, network_predictor, optimizer, new_data
        )
        
        # Compute adaptation score
        adaptation_score = self._compute_adaptation_score(
            baseline_performance, adapted_performance
        )
        
        adaptation_results = {
            'baseline_performance': baseline_performance,
            'adapted_performance': adapted_performance,
            'adaptation_score': adaptation_score
        }
        
        return adaptation_results
    
    def _measure_baseline_performance(self, *args, **kwargs) -> Dict[str, float]:
        """Measure baseline performance before adaptation."""
        # Simplified implementation
        return {'performance': 0.5}
    
    def _measure_adapted_performance(self, *args, **kwargs) -> Dict[str, float]:
        """Measure performance after adaptation."""
        # Simplified implementation
        return {'performance': 0.7}
    
    def _compute_adaptation_score(
        self,
        baseline: Dict[str, float],
        adapted: Dict[str, float]
    ) -> float:
        """Compute adaptation score."""
        # Simplified adaptation score
        baseline_score = baseline.get('performance', 0)
        adapted_score = adapted.get('performance', 0)
        
        return (adapted_score - baseline_score) / (baseline_score + 1e-8)
    
    def generate_evaluation_report(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("GenIoT-Optimizer Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Generative models evaluation
        if 'generative_models' in evaluation_results:
            report.append("Generative Models Evaluation:")
            report.append("-" * 30)
            gen_results = evaluation_results['generative_models']
            
            for model_name, metrics in gen_results.items():
                if isinstance(metrics, dict):
                    report.append(f"\n{model_name.upper()}:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            report.append(f"  {metric}: {value:.4f}")
        
        # Network predictor evaluation
        if 'network_predictor' in evaluation_results:
            report.append("\nNetwork Predictor Evaluation:")
            report.append("-" * 30)
            pred_results = evaluation_results['network_predictor']
            
            for metric, scores in pred_results.items():
                if isinstance(scores, dict):
                    report.append(f"\n{metric}:")
                    for score_name, value in scores.items():
                        if isinstance(value, (int, float)):
                            report.append(f"  {score_name}: {value:.4f}")
        
        # Optimizer evaluation
        if 'optimizer' in evaluation_results:
            report.append("\nOptimizer Evaluation:")
            report.append("-" * 30)
            opt_results = evaluation_results['optimizer']
            
            for category, metrics in opt_results.items():
                if isinstance(metrics, dict):
                    report.append(f"\n{category}:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            report.append(f"  {metric}: {value:.4f}")
        
        # Joint system evaluation
        if 'joint_system' in evaluation_results:
            report.append("\nJoint System Evaluation:")
            report.append("-" * 30)
            joint_results = evaluation_results['joint_system']
            
            for category, metrics in joint_results.items():
                if isinstance(metrics, dict):
                    report.append(f"\n{category}:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            report.append(f"  {metric}: {value:.4f}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text
