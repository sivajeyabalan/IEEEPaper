# GenIoT-Optimizer: Generative AI for IoT Network Performance Simulation and Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

GenIoT-Optimizer is a comprehensive framework that combines Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Diffusion Models with Deep Reinforcement Learning to optimize IoT network performance. This implementation is based on the IEEE paper "Generative AI for Simulating and Optimizing IoT Network Performance" and provides a complete solution for IoT network simulation, prediction, and optimization.

## Key Features

- **Synthetic Traffic Generation**: Generate realistic IoT traffic patterns using WGAN-GP, VAE, and DDPM
- **Network State Prediction**: Transformer-based prediction of future network conditions
- **Multi-Objective Optimization**: PPO-based optimization balancing latency, throughput, energy, and QoS
- **Digital Twin Integration**: Real-time network simulation and what-if analysis
- **Three-Phase Training**: Comprehensive training pipeline for all components
- **Multiple Use Cases**: Support for smart cities, manufacturing, and smart homes

## Performance Improvements

Based on the research paper, GenIoT-Optimizer achieves:
- **31.4%** reduction in communication latency
- **46.3%** increase in data throughput
- **29.9%** improvement in energy efficiency
- **91.8%** F1-score for anomaly detection (23.7% improvement)
- **Linear scalability** with <150ms processing time for 10,000-node networks

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd geniot-optimizer

# Install the package
pip install -e .

# Run demonstration
python -m geniot_optimizer.examples.demo
```

## Architecture

### Core Components

1. **Synthetic Traffic Generation Engine**
   - WGAN-GP for packet-level communication traces
   - VAE for compressed latent representations and anomaly detection
   - DDPM for sequential traffic modeling with temporal dependencies

2. **Network State Predictor**
   - Transformer architecture with multi-head self-attention
   - Recurrent components for temporal pattern recognition
   - Attention mechanisms for focusing on relevant network states

3. **Multi-Objective Optimizer**
   - PPO (Proximal Policy Optimization) for deep reinforcement learning
   - Multi-criteria reward function balancing competing objectives
   - Hierarchical processing for local and global optimization

4. **Digital Twin Integration**
   - Real-time synchronization with virtual network representation
   - What-if analysis for testing configuration changes
   - Predictive maintenance through anomaly detection

## Usage Examples

### Basic Traffic Generation

```python
from geniot_optimizer import TrafficGenerator, TrafficPattern, IoTDeviceType

# Initialize traffic generator
generator = TrafficGenerator()

# Generate burst traffic from sensors
burst_traffic = generator.generate_burst_traffic(
    num_samples=100,
    device_type=IoTDeviceType.SENSOR,
    intensity=2.0
)

# Generate mixed traffic patterns
mixed_traffic = generator.generate_mixed_traffic(
    num_samples=500,
    pattern_weights={
        'steady': 0.5,
        'burst': 0.2,
        'cyclical': 0.15,
        'event_triggered': 0.1,
        'anomalous': 0.05
    }
)
```

### Network State Prediction

```python
from geniot_optimizer import NetworkStatePredictor
import torch

# Initialize predictor
predictor = NetworkStatePredictor()

# Prepare input data
traffic_patterns = torch.randn(10, 100, 8)  # (batch, sequence, features)
current_states = torch.randn(10, 100, 8)

# Make predictions
predictions = predictor.predict(traffic_patterns, current_states)

# Access specific predictions
latency_pred = predictions['latency']
throughput_pred = predictions['throughput']
```

### Multi-Objective Optimization

```python
from geniot_optimizer import MultiObjectiveOptimizer

# Initialize optimizer
optimizer = MultiObjectiveOptimizer(
    num_devices=100,
    reward_weights={
        'latency': 0.3,
        'throughput': 0.3,
        'energy': 0.2,
        'qos': 0.2
    }
)

# Run optimization
results = optimizer.optimize_network(max_steps=100)

# Access results
final_performance = results['final_performance']
improvements = results['improvements']
```

### Digital Twin Simulation

```python
from geniot_optimizer import DigitalTwin

# Initialize digital twin
digital_twin = DigitalTwin(num_devices=100)

# Start simulation
digital_twin.start_simulation()

# Perform what-if analysis
scenario = {
    'device_failures': [0, 5, 10],
    'traffic_surge': {
        'affected_devices': [1, 2, 3, 4],
        'intensity': 2.0
    }
}

results = digital_twin.what_if_analysis(scenario, duration=100)

# Stop simulation
digital_twin.stop_simulation()
```

### Complete Training Pipeline

```python
from geniot_optimizer.training import TrainingPipeline

# Configuration
config = {
    'phase1': {
        'wgan_epochs': 100,
        'vae_epochs': 100,
        'diffusion_epochs': 100
    },
    'phase2': {
        'predictor_epochs': 50,
        'optimizer_timesteps': 50000,
        'adversarial_epochs': 20
    },
    'phase3': {
        'online_epochs': 10
    }
}

# Initialize training pipeline
pipeline = TrainingPipeline(config)

# Run complete training
results = pipeline.run_complete_training(
    data_path="path/to/iot_dataset.csv",
    validation_split=0.2,
    test_split=0.1
)
```

## Use Cases

### Smart City Infrastructure

```python
# Traffic management optimization
traffic_light_traffic = generator.generate_cyclical_traffic(
    num_samples=50, 
    device_type=IoTDeviceType.ACTUATOR, 
    period=30
)

# Environmental monitoring
sensor_traffic = generator.generate_steady_traffic(
    num_samples=100, 
    device_type=IoTDeviceType.SENSOR
)

# Security surveillance
camera_traffic = generator.generate_event_triggered_traffic(
    num_samples=30, 
    device_type=IoTDeviceType.CAMERA, 
    event_probability=0.3
)
```

### Industrial Manufacturing

```python
# Production line optimization
production_traffic = generator.generate_cyclical_traffic(
    num_samples=80, 
    device_type=IoTDeviceType.ACTUATOR, 
    period=60
)

# Quality control
quality_sensors = generator.generate_steady_traffic(
    num_samples=120, 
    device_type=IoTDeviceType.SENSOR
)

# Anomaly detection
anomalous_traffic = generator.generate_anomalous_traffic(
    num_samples=20, 
    device_type=IoTDeviceType.SENSOR, 
    anomaly_intensity=2.5
)
```

### Smart Home Networks

```python
# Energy management
thermostat_traffic = generator.generate_cyclical_traffic(
    num_samples=40, 
    device_type=IoTDeviceType.ACTUATOR, 
    period=120
)

# Security monitoring
security_traffic = generator.generate_steady_traffic(
    num_samples=60, 
    device_type=IoTDeviceType.SENSOR
)

# Smart assistants
speaker_traffic = generator.generate_event_triggered_traffic(
    num_samples=25, 
    device_type=IoTDeviceType.GATEWAY, 
    event_probability=0.1
)
```

## Evaluation Metrics

### Traffic Generation Quality
- **Maximum Mean Discrepancy (MMD)**: Lower is better
- **Fréchet Inception Distance (FID)**: Lower is better
- **Inception Score (IS)**: Higher is better

### Network Performance
- **Latency**: Average end-to-end packet delay (ms)
- **Throughput**: Network data transmission rate (Mbps)
- **Energy Efficiency**: Energy per transmitted bit (nJ/bit)
- **QoS Satisfaction**: Percentage of SLA requirements met

### Anomaly Detection
- **F1-Score**: Precision and recall balance
- **Detection Rate**: Percentage of anomalies identified
- **False Positive Rate**: Minimize false alarms

## File Structure

```
geniot_optimizer/
├── core/
│   ├── generative_models/
│   │   ├── wgan_gp.py          # Wasserstein GAN with gradient penalty
│   │   ├── vae.py              # Variational Autoencoder
│   │   └── diffusion.py        # Denoising Diffusion Probabilistic Model
│   ├── traffic_generator.py    # Synthetic traffic generation engine
│   ├── network_predictor.py    # Transformer-based state prediction
│   ├── optimizer.py            # PPO multi-objective optimizer
│   └── digital_twin.py         # Digital twin implementation
├── training/
│   ├── pipeline.py             # Three-phase training pipeline
│   ├── data_loader.py          # IoT dataset handling
│   └── evaluation.py           # Performance metrics and evaluation
├── simulation/
│   ├── network_simulator.py    # IoT network simulation
│   ├── traffic_models.py       # Traffic pattern modeling
│   └── metrics.py              # Network performance metrics
├── utils/
│   ├── config.py               # Configuration management
│   ├── visualization.py        # Plotting and visualization
│   └── io_utils.py             # Data I/O utilities
└── examples/
    ├── demo.py                 # Demonstration script
    ├── urban_infrastructure.py # Smart city use case
    ├── manufacturing.py        # Industrial IoT use case
    └── smart_home.py           # Residential IoT use case
```

## Configuration

The framework can be configured through YAML files or Python dictionaries:

```yaml
# config.yaml
data:
  sequence_length: 100
  num_features: 8
  batch_size: 32
  normalization: "standard"

phase1:
  wgan_epochs: 100
  vae_epochs: 100
  diffusion_epochs: 100

phase2:
  predictor_epochs: 50
  optimizer_timesteps: 50000
  adversarial_epochs: 20

phase3:
  online_epochs: 10
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd geniot-optimizer

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
black geniot_optimizer/
flake8 geniot_optimizer/
```

## Citation

If you use GenIoT-Optimizer in your research, please cite the original paper:

```bibtex
@article{pavithra2024generative,
  title={Generative AI for Simulating and Optimizing IoT Network Performance},
  author={Pavithra, R and Sivajeyabalan, S and Vishwanath, P and Vinoth Kumar, K B},
  journal={IEEE Transactions on Network and Service Management},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on research by R. Pavithra, Sivajeyabalan S, Vishwanath P, and Vinoth Kumar K B
- Velammal College of Engineering and Technology for computational resources
- The open-source community for the foundational libraries and frameworks

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation and examples

## Roadmap

- [ ] Federated learning support for privacy-preserving training
- [ ] 5G/6G network optimization integration
- [ ] Explainable AI techniques for model interpretability
- [ ] Adversarial robustness against malicious attacks
- [ ] Web-based dashboard for real-time monitoring
- [ ] Cloud deployment and scaling capabilities
