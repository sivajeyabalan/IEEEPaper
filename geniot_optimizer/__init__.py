"""
GenIoT-Optimizer: Generative AI for IoT Network Performance Simulation and Optimization

This package implements the GenIoT-Optimizer framework that combines Generative Adversarial Networks,
Variational Autoencoders, and Diffusion Models with Deep Reinforcement Learning to optimize IoT network performance.

Key Components:
- Synthetic Traffic Generation Engine
- Network State Predictor
- Multi-Objective Optimizer
- Digital Twin Integration

Author: Based on IEEE paper by R. Pavithra, Sivajeyabalan S, Vishwanath P, Vinoth Kumar K B
"""

__version__ = "1.0.0"
__author__ = "GenIoT-Optimizer Team"

from .core.traffic_generator import TrafficGenerator, TrafficPattern, IoTDeviceType
from .core.network_predictor import NetworkStatePredictor
from .core.optimizer import MultiObjectiveOptimizer
from .core.digital_twin import DigitalTwin

__all__ = [
    "TrafficGenerator",
    "TrafficPattern",
    "IoTDeviceType",
    "NetworkStatePredictor", 
    "MultiObjectiveOptimizer",
    "DigitalTwin"
]

