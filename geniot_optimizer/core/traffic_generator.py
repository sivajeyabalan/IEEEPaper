"""
Synthetic Traffic Generation Engine

This module implements the integrated traffic generation system that combines
WGAN-GP, VAE, and DDPM models to generate realistic IoT traffic patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

from .generative_models import WGAN_GP, VAE, DDPM


class TrafficPattern:
    """
    Represents different types of IoT traffic patterns.
    """
    
    BURST = "burst"           # Short-term high-intensity traffic
    CYCLICAL = "cyclical"     # Periodic traffic patterns
    EVENT_TRIGGERED = "event" # Event-driven traffic surges
    STEADY = "steady"         # Constant baseline traffic
    ANOMALOUS = "anomalous"   # Unusual traffic patterns


class IoTDeviceType:
    """
    Represents different types of IoT devices and their characteristics.
    """
    
    SENSOR = "sensor"         # Environmental sensors
    ACTUATOR = "actuator"     # Control devices
    CAMERA = "camera"         # Video/image devices
    GATEWAY = "gateway"       # Network gateways
    MOBILE = "mobile"         # Mobile devices


class TrafficGenerator:
    """
    Integrated synthetic traffic generation engine that combines multiple generative models
    to create realistic IoT traffic patterns with various characteristics.
    """
    
    def __init__(
        self,
        sequence_length: int = 100,
        num_features: int = 8,
        latent_dim: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[Dict[str, Any]] = None
    ):
        self.device = device
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.latent_dim = latent_dim
        
        # Default configuration
        self.config = config or self._get_default_config()
        
        # Initialize generative models
        self.wgan_gp = WGAN_GP(
            noise_dim=self.config['wgan_gp']['noise_dim'],
            hidden_dim=self.config['wgan_gp']['hidden_dim'],
            sequence_length=sequence_length,
            num_features=num_features,
            device=device
        )
        
        self.vae = VAE(
            input_dim=num_features,
            sequence_length=sequence_length,
            hidden_dim=self.config['vae']['hidden_dim'],
            latent_dim=latent_dim,
            beta=self.config['vae']['beta'],
            device=device
        )
        
        self.diffusion = DDPM(
            in_channels=num_features,
            out_channels=num_features,
            sequence_length=sequence_length,
            num_timesteps=self.config['diffusion']['num_timesteps'],
            device=device
        )
        
        # Traffic pattern templates
        self.pattern_templates = self._initialize_pattern_templates()
        
        # Device type characteristics
        self.device_characteristics = self._initialize_device_characteristics()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for generative models."""
        return {
            'wgan_gp': {
                'noise_dim': 100,
                'hidden_dim': 256,
                'lambda_gp': 10.0,
                'n_critic': 5
            },
            'vae': {
                'hidden_dim': 256,
                'beta': 1.0
            },
            'diffusion': {
                'num_timesteps': 1000,
                'beta_start': 0.0001,
                'beta_end': 0.02
            }
        }
    
    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize traffic pattern templates."""
        return {
            TrafficPattern.BURST: {
                'duration_range': (5, 20),
                'intensity_multiplier': (2.0, 5.0),
                'frequency': 0.1,
                'features': [0, 1, 2, 3]  # Packet size, rate, latency, jitter
            },
            TrafficPattern.CYCLICAL: {
                'period_range': (50, 200),
                'amplitude_range': (0.5, 2.0),
                'phase_offset': (0, 2 * np.pi),
                'features': [0, 1, 4, 5]  # Packet size, rate, energy, reliability
            },
            TrafficPattern.EVENT_TRIGGERED: {
                'trigger_probability': 0.05,
                'surge_duration': (10, 30),
                'surge_intensity': (1.5, 3.0),
                'features': [0, 1, 2, 6]  # Packet size, rate, latency, priority
            },
            TrafficPattern.STEADY: {
                'baseline_range': (0.8, 1.2),
                'variance': 0.1,
                'features': [0, 1, 4, 7]  # Packet size, rate, energy, timestamp
            },
            TrafficPattern.ANOMALOUS: {
                'anomaly_probability': 0.02,
                'anomaly_intensity': (3.0, 10.0),
                'anomaly_duration': (3, 15),
                'features': [0, 1, 2, 3, 4, 5, 6, 7]  # All features
            }
        }
    
    def _initialize_device_characteristics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize device type characteristics."""
        return {
            IoTDeviceType.SENSOR: {
                'packet_size_range': (32, 128),
                'transmission_rate': (0.1, 1.0),
                'energy_consumption': (0.5, 2.0),
                'reliability': (0.9, 0.99),
                'latency_sensitivity': 'low'
            },
            IoTDeviceType.ACTUATOR: {
                'packet_size_range': (64, 256),
                'transmission_rate': (0.5, 5.0),
                'energy_consumption': (1.0, 5.0),
                'reliability': (0.95, 0.999),
                'latency_sensitivity': 'high'
            },
            IoTDeviceType.CAMERA: {
                'packet_size_range': (1024, 8192),
                'transmission_rate': (10.0, 100.0),
                'energy_consumption': (5.0, 20.0),
                'reliability': (0.8, 0.95),
                'latency_sensitivity': 'medium'
            },
            IoTDeviceType.GATEWAY: {
                'packet_size_range': (256, 1024),
                'transmission_rate': (50.0, 500.0),
                'energy_consumption': (10.0, 50.0),
                'reliability': (0.99, 0.999),
                'latency_sensitivity': 'medium'
            },
            IoTDeviceType.MOBILE: {
                'packet_size_range': (128, 512),
                'transmission_rate': (1.0, 10.0),
                'energy_consumption': (2.0, 8.0),
                'reliability': (0.85, 0.95),
                'latency_sensitivity': 'high'
            }
        }
    
    def generate_burst_traffic(
        self,
        num_samples: int,
        device_type: str = IoTDeviceType.SENSOR,
        intensity: float = 1.0
    ) -> torch.Tensor:
        """
        Generate burst traffic patterns.
        
        Args:
            num_samples: Number of samples to generate
            device_type: Type of IoT device
            intensity: Burst intensity multiplier
            
        Returns:
            Generated burst traffic patterns
        """
        # Generate base traffic using WGAN-GP
        base_traffic = self.wgan_gp.generate_samples(num_samples)
        
        # Apply burst pattern
        burst_traffic = self._apply_burst_pattern(base_traffic, intensity)
        
        # Apply device-specific characteristics
        device_traffic = self._apply_device_characteristics(burst_traffic, device_type)
        
        return device_traffic
    
    def generate_cyclical_traffic(
        self,
        num_samples: int,
        device_type: str = IoTDeviceType.SENSOR,
        period: int = 100,
        amplitude: float = 1.0
    ) -> torch.Tensor:
        """
        Generate cyclical traffic patterns.
        
        Args:
            num_samples: Number of samples to generate
            device_type: Type of IoT device
            period: Cycle period
            amplitude: Cycle amplitude
            
        Returns:
            Generated cyclical traffic patterns
        """
        # Generate base traffic using VAE
        base_traffic = self.vae.generate_samples(num_samples)
        
        # Apply cyclical pattern
        cyclical_traffic = self._apply_cyclical_pattern(base_traffic, period, amplitude)
        
        # Apply device-specific characteristics
        device_traffic = self._apply_device_characteristics(cyclical_traffic, device_type)
        
        return device_traffic
    
    def generate_event_triggered_traffic(
        self,
        num_samples: int,
        device_type: str = IoTDeviceType.ACTUATOR,
        event_probability: float = 0.1
    ) -> torch.Tensor:
        """
        Generate event-triggered traffic patterns.
        
        Args:
            num_samples: Number of samples to generate
            device_type: Type of IoT device
            event_probability: Probability of event occurrence
            
        Returns:
            Generated event-triggered traffic patterns
        """
        # Generate base traffic using diffusion model
        base_traffic = self.diffusion.sample(num_samples)
        
        # Apply event-triggered pattern
        event_traffic = self._apply_event_pattern(base_traffic, event_probability)
        
        # Apply device-specific characteristics
        device_traffic = self._apply_device_characteristics(event_traffic, device_type)
        
        return device_traffic
    
    def generate_steady_traffic(
        self,
        num_samples: int,
        device_type: str = IoTDeviceType.SENSOR,
        baseline: float = 1.0
    ) -> torch.Tensor:
        """
        Generate steady baseline traffic patterns.
        
        Args:
            num_samples: Number of samples to generate
            device_type: Type of IoT device
            baseline: Baseline traffic level
            
        Returns:
            Generated steady traffic patterns
        """
        # Generate base traffic using VAE (most stable)
        base_traffic = self.vae.generate_samples(num_samples)
        
        # Apply steady pattern
        steady_traffic = self._apply_steady_pattern(base_traffic, baseline)
        
        # Apply device-specific characteristics
        device_traffic = self._apply_device_characteristics(steady_traffic, device_type)
        
        return device_traffic
    
    def generate_anomalous_traffic(
        self,
        num_samples: int,
        device_type: str = IoTDeviceType.SENSOR,
        anomaly_intensity: float = 3.0
    ) -> torch.Tensor:
        """
        Generate anomalous traffic patterns.
        
        Args:
            num_samples: Number of samples to generate
            device_type: Type of IoT device
            anomaly_intensity: Intensity of anomalies
            
        Returns:
            Generated anomalous traffic patterns
        """
        # Generate base traffic using WGAN-GP (can produce diverse patterns)
        base_traffic = self.wgan_gp.generate_samples(num_samples)
        
        # Apply anomalous pattern
        anomalous_traffic = self._apply_anomalous_pattern(base_traffic, anomaly_intensity)
        
        # Apply device-specific characteristics
        device_traffic = self._apply_device_characteristics(anomalous_traffic, device_type)
        
        return device_traffic
    
    def generate_mixed_traffic(
        self,
        num_samples: int,
        pattern_weights: Optional[Dict[str, float]] = None,
        device_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Generate mixed traffic patterns combining multiple types.
        
        Args:
            num_samples: Number of samples to generate
            pattern_weights: Weights for different traffic patterns
            device_types: List of device types to include
            
        Returns:
            Generated mixed traffic patterns
        """
        if pattern_weights is None:
            pattern_weights = {
                TrafficPattern.STEADY: 0.5,
                TrafficPattern.BURST: 0.2,
                TrafficPattern.CYCLICAL: 0.15,
                TrafficPattern.EVENT_TRIGGERED: 0.1,
                TrafficPattern.ANOMALOUS: 0.05
            }
        
        if device_types is None:
            device_types = [IoTDeviceType.SENSOR, IoTDeviceType.ACTUATOR, IoTDeviceType.GATEWAY]
        
        # Calculate number of samples for each pattern
        pattern_samples = {}
        remaining_samples = num_samples
        
        for pattern, weight in pattern_weights.items():
            if pattern == list(pattern_weights.keys())[-1]:  # Last pattern gets remaining samples
                pattern_samples[pattern] = remaining_samples
            else:
                samples = int(num_samples * weight)
                pattern_samples[pattern] = samples
                remaining_samples -= samples
        
        # Generate traffic for each pattern
        all_traffic = []
        
        for pattern, samples in pattern_samples.items():
            if samples > 0:
                # Randomly select device type
                device_type = np.random.choice(device_types)
                
                if pattern == TrafficPattern.BURST:
                    traffic = self.generate_burst_traffic(samples, device_type)
                elif pattern == TrafficPattern.CYCLICAL:
                    traffic = self.generate_cyclical_traffic(samples, device_type)
                elif pattern == TrafficPattern.EVENT_TRIGGERED:
                    traffic = self.generate_event_triggered_traffic(samples, device_type)
                elif pattern == TrafficPattern.STEADY:
                    traffic = self.generate_steady_traffic(samples, device_type)
                elif pattern == TrafficPattern.ANOMALOUS:
                    traffic = self.generate_anomalous_traffic(samples, device_type)
                else:
                    continue
                
                all_traffic.append(traffic)
        
        # Concatenate all traffic patterns
        if all_traffic:
            mixed_traffic = torch.cat(all_traffic, dim=0)
            # Shuffle the samples
            indices = torch.randperm(mixed_traffic.size(0))
            mixed_traffic = mixed_traffic[indices]
        else:
            # Fallback to steady traffic
            mixed_traffic = self.generate_steady_traffic(num_samples)
        
        return mixed_traffic
    
    def _apply_burst_pattern(self, traffic: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply burst pattern to traffic."""
        burst_traffic = traffic.clone()
        batch_size, seq_len, features = traffic.shape
        
        # Randomly select burst positions
        burst_positions = torch.randint(0, seq_len, (batch_size,))
        burst_durations = torch.randint(5, 20, (batch_size,))
        
        for i in range(batch_size):
            start_pos = burst_positions[i]
            duration = min(burst_durations[i], seq_len - start_pos)
            
            # Apply burst intensity
            burst_traffic[i, start_pos:start_pos + duration, :] *= intensity
        
        return burst_traffic
    
    def _apply_cyclical_pattern(self, traffic: torch.Tensor, period: int, amplitude: float) -> torch.Tensor:
        """Apply cyclical pattern to traffic."""
        cyclical_traffic = traffic.clone()
        batch_size, seq_len, features = traffic.shape
        
        # Create sinusoidal pattern
        t = torch.arange(seq_len, dtype=torch.float32, device=self.device)
        cycle = amplitude * torch.sin(2 * np.pi * t / period)
        
        # Apply to relevant features
        for i in range(batch_size):
            for feature_idx in [0, 1, 4, 5]:  # Packet size, rate, energy, reliability
                if feature_idx < features:
                    cyclical_traffic[i, :, feature_idx] *= (1 + cycle)
        
        return cyclical_traffic
    
    def _apply_event_pattern(self, traffic: torch.Tensor, event_probability: float) -> torch.Tensor:
        """Apply event-triggered pattern to traffic."""
        event_traffic = traffic.clone()
        batch_size, seq_len, features = traffic.shape
        
        # Randomly trigger events
        event_mask = torch.rand(batch_size, device=self.device) < event_probability
        
        for i in range(batch_size):
            if event_mask[i]:
                # Random event position and duration
                event_pos = torch.randint(0, seq_len - 10, (1,)).item()
                event_duration = torch.randint(10, 30, (1,)).item()
                event_intensity = torch.rand(1).item() * 2.0 + 1.0  # 1.0 to 3.0
                
                # Apply event surge
                end_pos = min(event_pos + event_duration, seq_len)
                event_traffic[i, event_pos:end_pos, :] *= event_intensity
        
        return event_traffic
    
    def _apply_steady_pattern(self, traffic: torch.Tensor, baseline: float) -> torch.Tensor:
        """Apply steady pattern to traffic."""
        steady_traffic = traffic.clone()
        
        # Apply baseline scaling with small random variations
        noise = torch.randn_like(traffic) * 0.1
        steady_traffic = steady_traffic * baseline + noise
        
        return steady_traffic
    
    def _apply_anomalous_pattern(self, traffic: torch.Tensor, anomaly_intensity: float) -> torch.Tensor:
        """Apply anomalous pattern to traffic."""
        anomalous_traffic = traffic.clone()
        batch_size, seq_len, features = traffic.shape
        
        # Randomly introduce anomalies
        anomaly_mask = torch.rand(batch_size, device=self.device) < 0.02  # 2% anomaly rate
        
        for i in range(batch_size):
            if anomaly_mask[i]:
                # Random anomaly characteristics
                anomaly_pos = torch.randint(0, seq_len - 5, (1,)).item()
                anomaly_duration = torch.randint(3, 15, (1,)).item()
                anomaly_strength = torch.rand(1).item() * (anomaly_intensity - 1.0) + 1.0
                
                # Apply anomaly
                end_pos = min(anomaly_pos + anomaly_duration, seq_len)
                anomalous_traffic[i, anomaly_pos:end_pos, :] *= anomaly_strength
        
        return anomalous_traffic
    
    def _apply_device_characteristics(self, traffic: torch.Tensor, device_type: str) -> torch.Tensor:
        """Apply device-specific characteristics to traffic."""
        if device_type not in self.device_characteristics:
            return traffic
        
        device_traffic = traffic.clone()
        characteristics = self.device_characteristics[device_type]
        
        # Apply device-specific scaling factors
        # This is a simplified implementation - in practice, you'd want more sophisticated modeling
        
        # Packet size scaling (feature 0)
        packet_size_range = characteristics['packet_size_range']
        packet_size_scale = (packet_size_range[0] + packet_size_range[1]) / 2 / 1000  # Normalize
        device_traffic[:, :, 0] *= packet_size_scale
        
        # Transmission rate scaling (feature 1)
        rate_range = characteristics['transmission_rate']
        rate_scale = (rate_range[0] + rate_range[1]) / 2 / 10  # Normalize
        device_traffic[:, :, 1] *= rate_scale
        
        # Energy consumption scaling (feature 4)
        energy_range = characteristics['energy_consumption']
        energy_scale = (energy_range[0] + energy_range[1]) / 2 / 10  # Normalize
        device_traffic[:, :, 4] *= energy_scale
        
        return device_traffic
    
    def train_models(
        self,
        real_traffic_data: torch.utils.data.DataLoader,
        epochs: int = 100,
        save_models: bool = True,
        model_save_path: str = "./models"
    ) -> Dict[str, Any]:
        """
        Train all generative models on real traffic data.
        
        Args:
            real_traffic_data: DataLoader containing real IoT traffic data
            epochs: Number of training epochs
            save_models: Whether to save trained models
            model_save_path: Path to save models
            
        Returns:
            Training history for all models
        """
        self.logger.info("Starting training of generative models...")
        
        training_history = {}
        
        # Train WGAN-GP
        self.logger.info("Training WGAN-GP...")
        wgan_history = self.wgan_gp.train(real_traffic_data, epochs=epochs)
        training_history['wgan_gp'] = wgan_history
        
        # Train VAE
        self.logger.info("Training VAE...")
        vae_history = self.vae.train_model(real_traffic_data, epochs=epochs)
        training_history['vae'] = vae_history
        
        # Train DDPM
        self.logger.info("Training DDPM...")
        diffusion_history = self.diffusion.train(real_traffic_data, epochs=epochs)
        training_history['diffusion'] = diffusion_history
        
        # Save models if requested
        if save_models:
            import os
            os.makedirs(model_save_path, exist_ok=True)
            
            self.wgan_gp.save_model(f"{model_save_path}/wgan_gp.pth")
            self.vae.save_model(f"{model_save_path}/vae.pth")
            self.diffusion.save_model(f"{model_save_path}/diffusion.pth")
            
            self.logger.info(f"Models saved to {model_save_path}")
        
        self.logger.info("Training completed!")
        return training_history
    
    def load_models(self, model_save_path: str) -> None:
        """
        Load pre-trained generative models.
        
        Args:
            model_save_path: Path to saved models
        """
        self.wgan_gp.load_model(f"{model_save_path}/wgan_gp.pth")
        self.vae.load_model(f"{model_save_path}/vae.pth")
        self.diffusion.load_model(f"{model_save_path}/diffusion.pth")
        
        self.logger.info(f"Models loaded from {model_save_path}")
    
    def get_traffic_statistics(self, traffic: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics for generated traffic.
        
        Args:
            traffic: Generated traffic tensor
            
        Returns:
            Dictionary of traffic statistics
        """
        stats = {}
        
        # Basic statistics
        stats['mean'] = torch.mean(traffic).item()
        stats['std'] = torch.std(traffic).item()
        stats['min'] = torch.min(traffic).item()
        stats['max'] = torch.max(traffic).item()
        
        # Per-feature statistics
        for i in range(traffic.shape[-1]):
            feature_data = traffic[:, :, i]
            stats[f'feature_{i}_mean'] = torch.mean(feature_data).item()
            stats[f'feature_{i}_std'] = torch.std(feature_data).item()
        
        return stats

