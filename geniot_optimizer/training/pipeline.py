"""
Three-Phase Training Pipeline for GenIoT-Optimizer

This module implements the complete training pipeline as described in the paper:
1. Phase 1: Generative Model Pre-training
2. Phase 2: Joint Fine-tuning
3. Phase 3: Online Learning
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from datetime import datetime
import json
import os
from pathlib import Path

from ..core.traffic_generator import TrafficGenerator
from ..core.network_predictor import NetworkStatePredictor
from ..core.optimizer import MultiObjectiveOptimizer
from ..core.digital_twin import DigitalTwin
from .data_loader import IoTDataLoader
from .evaluation import ModelEvaluator


class TrainingPipeline:
    """
    Complete training pipeline for GenIoT-Optimizer framework.
    
    This class orchestrates the three-phase training process as described in the paper.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./training_outputs"
    ):
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.traffic_generator = TrafficGenerator(device=device)
        self.network_predictor = NetworkStatePredictor(device=device)
        self.optimizer = MultiObjectiveOptimizer(device=device)
        self.digital_twin = DigitalTwin(device=device)
        
        # Initialize data loader and evaluator
        self.data_loader = IoTDataLoader(config.get('data', {}))
        self.evaluator = ModelEvaluator(device=device)
        
        # Training history
        self.training_history = {
            'phase1': {},
            'phase2': {},
            'phase3': {}
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.save_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_complete_training(
        self,
        data_path: str,
        validation_split: float = 0.2,
        test_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run the complete three-phase training pipeline.
        
        Args:
            data_path: Path to IoT dataset
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            
        Returns:
            Complete training results
        """
        self.logger.info("Starting complete training pipeline...")
        
        # Load and prepare data
        self.logger.info("Loading and preparing data...")
        train_data, val_data, test_data = self.data_loader.load_and_split_data(
            data_path, validation_split, test_split
        )
        
        # Phase 1: Generative Model Pre-training
        self.logger.info("Phase 1: Generative Model Pre-training")
        phase1_results = self._phase1_pretraining(train_data, val_data)
        
        # Phase 2: Joint Fine-tuning
        self.logger.info("Phase 2: Joint Fine-tuning")
        phase2_results = self._phase2_joint_finetuning(train_data, val_data)
        
        # Phase 3: Online Learning
        self.logger.info("Phase 3: Online Learning")
        phase3_results = self._phase3_online_learning(test_data)
        
        # Final evaluation
        self.logger.info("Final evaluation...")
        final_results = self._final_evaluation(test_data)
        
        # Compile results
        complete_results = {
            'phase1': phase1_results,
            'phase2': phase2_results,
            'phase3': phase3_results,
            'final_evaluation': final_results,
            'training_config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_training_results(complete_results)
        
        self.logger.info("Complete training pipeline finished!")
        return complete_results
    
    def _phase1_pretraining(
        self,
        train_data: data.DataLoader,
        val_data: data.DataLoader
    ) -> Dict[str, Any]:
        """
        Phase 1: Generative Model Pre-training
        
        All generative models are independently pre-trained on historical IoT traffic data
        using unsupervised learning to establish baseline capabilities.
        """
        self.logger.info("Starting Phase 1: Generative Model Pre-training")
        
        phase1_config = self.config.get('phase1', {})
        results = {}
        
        # Train WGAN-GP
        self.logger.info("Training WGAN-GP...")
        wgan_epochs = phase1_config.get('wgan_epochs', 100)
        wgan_results = self.traffic_generator.wgan_gp.train(
            train_data, epochs=wgan_epochs
        )
        results['wgan_gp'] = wgan_results
        
        # Train VAE
        self.logger.info("Training VAE...")
        vae_epochs = phase1_config.get('vae_epochs', 100)
        vae_results = self.traffic_generator.vae.train_model(
            train_data, epochs=vae_epochs
        )
        results['vae'] = vae_results
        
        # Train DDPM
        self.logger.info("Training DDPM...")
        diffusion_epochs = phase1_config.get('diffusion_epochs', 100)
        diffusion_results = self.traffic_generator.diffusion.train(
            train_data, epochs=diffusion_epochs
        )
        results['diffusion'] = diffusion_results
        
        # Evaluate generative models
        self.logger.info("Evaluating generative models...")
        gen_eval_results = self.evaluator.evaluate_generative_models(
            self.traffic_generator, val_data
        )
        results['evaluation'] = gen_eval_results
        
        # Save Phase 1 models
        self._save_phase_models('phase1', {
            'wgan_gp': self.traffic_generator.wgan_gp,
            'vae': self.traffic_generator.vae,
            'diffusion': self.traffic_generator.diffusion
        })
        
        self.training_history['phase1'] = results
        return results
    
    def _phase2_joint_finetuning(
        self,
        train_data: data.DataLoader,
        val_data: data.DataLoader
    ) -> Dict[str, Any]:
        """
        Phase 2: Joint Fine-tuning
        
        The generative models and reinforcement learning agent are jointly fine-tuned
        through adversarial training, where the agent learns to optimize networks using
        synthetic traffic while the generators improve based on agent feedback.
        """
        self.logger.info("Starting Phase 2: Joint Fine-tuning")
        
        phase2_config = self.config.get('phase2', {})
        results = {}
        
        # Train Network Predictor
        self.logger.info("Training Network Predictor...")
        predictor_epochs = phase2_config.get('predictor_epochs', 50)
        predictor_results = self.network_predictor.train(
            train_data, epochs=predictor_epochs
        )
        results['network_predictor'] = predictor_results
        
        # Train Multi-Objective Optimizer
        self.logger.info("Training Multi-Objective Optimizer...")
        optimizer_timesteps = phase2_config.get('optimizer_timesteps', 50000)
        optimizer_results = self.optimizer.train(
            total_timesteps=optimizer_timesteps
        )
        results['optimizer'] = optimizer_results
        
        # Joint adversarial training
        self.logger.info("Starting joint adversarial training...")
        adversarial_epochs = phase2_config.get('adversarial_epochs', 20)
        adversarial_results = self._adversarial_training(
            train_data, adversarial_epochs
        )
        results['adversarial_training'] = adversarial_results
        
        # Evaluate joint system
        self.logger.info("Evaluating joint system...")
        joint_eval_results = self.evaluator.evaluate_joint_system(
            self.traffic_generator,
            self.network_predictor,
            self.optimizer,
            val_data
        )
        results['evaluation'] = joint_eval_results
        
        # Save Phase 2 models
        self._save_phase_models('phase2', {
            'network_predictor': self.network_predictor,
            'optimizer': self.optimizer
        })
        
        self.training_history['phase2'] = results
        return results
    
    def _phase3_online_learning(self, test_data: data.DataLoader) -> Dict[str, Any]:
        """
        Phase 3: Online Learning
        
        Deployed systems continuously adapt through online learning, incorporating
        new traffic patterns and optimization experiences to maintain performance
        in dynamic environments.
        """
        self.logger.info("Starting Phase 3: Online Learning")
        
        phase3_config = self.config.get('phase3', {})
        results = {}
        
        # Initialize digital twin
        self.logger.info("Initializing digital twin...")
        self.digital_twin.start_simulation()
        
        # Online learning loop
        online_epochs = phase3_config.get('online_epochs', 10)
        adaptation_results = []
        
        for epoch in range(online_epochs):
            self.logger.info(f"Online learning epoch {epoch + 1}/{online_epochs}")
            
            # Collect new data from digital twin
            new_data = self._collect_online_data()
            
            # Adapt models to new data
            adaptation_result = self._adapt_models(new_data)
            adaptation_results.append(adaptation_result)
            
            # Evaluate adaptation
            eval_result = self.evaluator.evaluate_online_adaptation(
                self.traffic_generator,
                self.network_predictor,
                self.optimizer,
                new_data
            )
            
            self.logger.info(f"Epoch {epoch + 1} adaptation score: {eval_result.get('adaptation_score', 0):.4f}")
        
        # Stop digital twin
        self.digital_twin.stop_simulation()
        
        results['adaptation_results'] = adaptation_results
        results['final_adaptation_score'] = adaptation_results[-1] if adaptation_results else {}
        
        # Save Phase 3 models
        self._save_phase_models('phase3', {
            'traffic_generator': self.traffic_generator,
            'network_predictor': self.network_predictor,
            'optimizer': self.optimizer
        })
        
        self.training_history['phase3'] = results
        return results
    
    def _adversarial_training(
        self,
        train_data: data.DataLoader,
        epochs: int
    ) -> Dict[str, Any]:
        """
        Perform adversarial training between generative models and optimizer.
        """
        self.logger.info("Starting adversarial training...")
        
        results = {
            'generator_losses': [],
            'discriminator_losses': [],
            'optimizer_losses': []
        }
        
        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            epoch_opt_loss = 0
            num_batches = 0
            
            for batch_data in train_data:
                # Generate synthetic traffic
                synthetic_traffic = self.traffic_generator.generate_mixed_traffic(
                    num_samples=batch_data[0].size(0)
                )
                
                # Train optimizer on synthetic data
                opt_loss = self._train_optimizer_on_synthetic(synthetic_traffic)
                epoch_opt_loss += opt_loss
                
                # Update generators based on optimizer feedback
                gen_loss, disc_loss = self._update_generators_with_feedback(
                    synthetic_traffic, batch_data[0]
                )
                epoch_gen_loss += gen_loss
                epoch_disc_loss += disc_loss
                
                num_batches += 1
            
            # Average losses
            avg_gen_loss = epoch_gen_loss / num_batches
            avg_disc_loss = epoch_disc_loss / num_batches
            avg_opt_loss = epoch_opt_loss / num_batches
            
            results['generator_losses'].append(avg_gen_loss)
            results['discriminator_losses'].append(avg_disc_loss)
            results['optimizer_losses'].append(avg_opt_loss)
            
            if epoch % 5 == 0:
                self.logger.info(
                    f"Adversarial epoch {epoch}: Gen={avg_gen_loss:.4f}, "
                    f"Disc={avg_disc_loss:.4f}, Opt={avg_opt_loss:.4f}"
                )
        
        return results
    
    def _train_optimizer_on_synthetic(self, synthetic_traffic: torch.Tensor) -> float:
        """Train optimizer on synthetic traffic data."""
        # Convert synthetic traffic to environment format
        # This is a simplified implementation
        return 0.1  # Placeholder loss
    
    def _update_generators_with_feedback(
        self,
        synthetic_traffic: torch.Tensor,
        real_traffic: torch.Tensor
    ) -> Tuple[float, float]:
        """Update generators based on optimizer feedback."""
        # This is a simplified implementation
        # In practice, you would use the optimizer's feedback to improve generators
        return 0.1, 0.1  # Placeholder losses
    
    def _collect_online_data(self) -> data.DataLoader:
        """Collect new data from digital twin simulation."""
        # Get recent data from digital twin
        recent_data = self.digital_twin.get_historical_data(
            start_time=datetime.now() - timedelta(minutes=10)
        )
        
        # Convert to DataLoader format
        # This is a simplified implementation
        return data.DataLoader([], batch_size=32, shuffle=True)
    
    def _adapt_models(self, new_data: data.DataLoader) -> Dict[str, Any]:
        """Adapt models to new data."""
        adaptation_results = {}
        
        # Fine-tune traffic generator
        if len(new_data.dataset) > 0:
            gen_adaptation = self.traffic_generator.train_models(
                new_data, epochs=5, save_models=False
            )
            adaptation_results['traffic_generator'] = gen_adaptation
        
        # Fine-tune network predictor
        pred_adaptation = self.network_predictor.train(
            new_data, epochs=3, save_interval=10
        )
        adaptation_results['network_predictor'] = pred_adaptation
        
        # Fine-tune optimizer
        opt_adaptation = self.optimizer.train(
            total_timesteps=1000, save_interval=1000
        )
        adaptation_results['optimizer'] = opt_adaptation
        
        return adaptation_results
    
    def _final_evaluation(self, test_data: data.DataLoader) -> Dict[str, Any]:
        """Perform final evaluation on test data."""
        self.logger.info("Performing final evaluation...")
        
        # Evaluate all components
        results = {}
        
        # Generative models evaluation
        gen_results = self.evaluator.evaluate_generative_models(
            self.traffic_generator, test_data
        )
        results['generative_models'] = gen_results
        
        # Network predictor evaluation
        pred_results = self.evaluator.evaluate_network_predictor(
            self.network_predictor, test_data
        )
        results['network_predictor'] = pred_results
        
        # Optimizer evaluation
        opt_results = self.evaluator.evaluate_optimizer(
            self.optimizer, test_data
        )
        results['optimizer'] = opt_results
        
        # Joint system evaluation
        joint_results = self.evaluator.evaluate_joint_system(
            self.traffic_generator,
            self.network_predictor,
            self.optimizer,
            test_data
        )
        results['joint_system'] = joint_results
        
        return results
    
    def _save_phase_models(self, phase: str, models: Dict[str, Any]) -> None:
        """Save models for a specific phase."""
        phase_dir = self.save_dir / phase
        phase_dir.mkdir(exist_ok=True)
        
        for model_name, model in models.items():
            if hasattr(model, 'save_model'):
                model_path = phase_dir / f"{model_name}.pth"
                model.save_model(str(model_path))
                self.logger.info(f"Saved {model_name} to {model_path}")
    
    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Save complete training results."""
        results_file = self.save_dir / "training_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Training results saved to {results_file}")
    
    def load_trained_models(self, phase: str) -> None:
        """Load trained models for a specific phase."""
        phase_dir = self.save_dir / phase
        
        if not phase_dir.exists():
            self.logger.error(f"Phase {phase} models not found at {phase_dir}")
            return
        
        # Load models based on phase
        if phase == 'phase1':
            wgan_path = phase_dir / "wgan_gp.pth"
            vae_path = phase_dir / "vae.pth"
            diffusion_path = phase_dir / "diffusion.pth"
            
            if wgan_path.exists():
                self.traffic_generator.wgan_gp.load_model(str(wgan_path))
            if vae_path.exists():
                self.traffic_generator.vae.load_model(str(vae_path))
            if diffusion_path.exists():
                self.traffic_generator.diffusion.load_model(str(diffusion_path))
        
        elif phase == 'phase2':
            predictor_path = phase_dir / "network_predictor.pth"
            optimizer_path = phase_dir / "optimizer.pth"
            
            if predictor_path.exists():
                self.network_predictor.load_model(str(predictor_path))
            if optimizer_path.exists():
                self.optimizer.load_model(str(optimizer_path))
        
        elif phase == 'phase3':
            # Load all models
            self.load_trained_models('phase1')
            self.load_trained_models('phase2')
        
        self.logger.info(f"Loaded {phase} models from {phase_dir}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        summary = {
            'phases_completed': len([p for p in self.training_history.values() if p]),
            'total_phases': 3,
            'current_phase': None,
            'training_history': self.training_history
        }
        
        # Determine current phase
        if self.training_history['phase3']:
            summary['current_phase'] = 'completed'
        elif self.training_history['phase2']:
            summary['current_phase'] = 'phase3'
        elif self.training_history['phase1']:
            summary['current_phase'] = 'phase2'
        else:
            summary['current_phase'] = 'phase1'
        
        return summary
