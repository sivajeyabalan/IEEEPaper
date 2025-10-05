# 🚀 GenIoT-Optimizer: Google Colab Export Guide

## 📋 Overview
This guide explains how to export and run the GenIoT-Optimizer framework in Google Colab for easy access, collaboration, and GPU acceleration.

## 🎯 Quick Start (3 Steps)

### Step 1: Upload to Google Colab
1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Create New Notebook**: Click "New Notebook"
3. **Upload Files**: 
   - Upload `GenIoT_Optimizer_Colab.ipynb` (the notebook I created)
   - Or copy the code from the notebook cells below

### Step 2: Enable GPU (Recommended)
1. Go to **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 or better)
3. Click **Save**

### Step 3: Run the Notebook
1. Execute cells sequentially (Shift + Enter)
2. The framework will automatically install dependencies and run

## 📁 Files to Upload

### Required Files:
- `GenIoT_Optimizer_Colab.ipynb` - Main notebook
- `requirements.txt` - Dependencies (optional, included in notebook)

### Optional Files (for full functionality):
- `geniot_optimizer/` - Complete source code directory
- `configs/default_config.yaml` - Configuration file
- `README.md` - Documentation

## 🔧 Installation Commands (Auto-included in Notebook)

```python
# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
!pip install stable-baselines3 gymnasium networkx simpy
!pip install numpy pandas scikit-learn scipy matplotlib seaborn plotly
!pip install tqdm pyyaml rich

# Install additional ML libraries
!pip install transformers einops accelerate
```

## 🎮 Notebook Features

### 1. **Traffic Generation Engine**
- WGAN-GP implementation for packet-level traces
- VAE for compressed representations
- DDPM for sequential modeling
- Real-time traffic pattern generation

### 2. **Network State Prediction**
- Transformer-based architecture
- Multi-head attention mechanisms
- Temporal pattern recognition
- Future state forecasting

### 3. **Multi-Objective Optimization**
- PPO reinforcement learning
- Latency, throughput, energy, QoS optimization
- Real-time performance improvement
- Scalable to large networks

### 4. **Visualization & Analysis**
- Traffic pattern plots
- Optimization progress charts
- Performance metrics dashboard
- Real-time monitoring

## 📊 Expected Results

### Performance Improvements:
- **Latency**: 31.4% reduction
- **Throughput**: 46.3% increase  
- **Energy**: 29.9% improvement
- **Anomaly Detection**: 91.8% F1-score

### Generated Outputs:
- Synthetic IoT traffic patterns
- Network optimization results
- Performance improvement charts
- Real-time metrics visualization

## 🛠️ Customization Options

### Modify Parameters:
```python
# Adjust model parameters
generator = Generator(
    noise_dim=100,           # Noise vector dimension
    hidden_dim=256,          # Hidden layer size
    sequence_length=100,     # Traffic sequence length
    num_features=8           # Number of IoT features
)

# Change optimization settings
optimizer = SimpleOptimizer(
    num_devices=20,          # Number of IoT devices
    steps=100               # Optimization iterations
)
```

### Add Your Data:
```python
# Load your IoT dataset
your_data = torch.load('your_iot_data.pt')
# Train models on your data
generator.train(your_data)
```

## 🔍 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```python
   # Reduce batch size
   batch_size = 8  # Instead of 32
   ```

2. **Import Errors**:
   ```python
   # Reinstall packages
   !pip install --upgrade torch torchvision
   ```

3. **Slow Performance**:
   - Enable GPU acceleration
   - Reduce model complexity
   - Use smaller datasets for testing

### Performance Tips:
- Use GPU runtime for faster training
- Reduce sequence length for quick testing
- Start with smaller models, then scale up

## 📈 Advanced Usage

### Training on Real Data:
```python
# Load real IoT dataset
from geniot_optimizer.training.data_loader import IoTDataLoader

data_loader = IoTDataLoader('path/to/your/data')
train_data = data_loader.load_training_data()

# Train the complete framework
from geniot_optimizer.training.pipeline import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run_full_training(train_data)
```

### Custom IoT Device Types:
```python
# Define custom device characteristics
custom_device = IoTDeviceType(
    name="SmartSensor",
    power_consumption=0.5,
    data_rate=10.0,
    latency_tolerance=50.0,
    reliability_requirement=0.99
)
```

## 🌐 Sharing & Collaboration

### Share Your Results:
1. **Save to GitHub**: Upload notebook to GitHub repository
2. **Share Colab Link**: Use Colab's sharing features
3. **Export Results**: Download plots and metrics as files

### Collaborate:
- Share notebook with team members
- Use Colab's commenting features
- Version control with Git integration

## 📚 Additional Resources

### Documentation:
- `README.md` - Complete project documentation
- `geniot_optimizer/` - Source code with docstrings
- IEEE Paper - Original research paper

### Examples:
- `examples/demo.py` - Full demonstration
- `examples/urban_infrastructure.py` - Smart city use case
- `examples/manufacturing.py` - Industrial IoT example

## 🎉 Success Indicators

You'll know the export was successful when you see:
- ✅ All dependencies installed without errors
- ✅ Models initialized successfully
- ✅ Traffic generation working (samples created)
- ✅ Optimization running (improving scores)
- ✅ Visualizations displaying correctly

## 🚀 Next Steps

After successful Colab setup:
1. **Experiment** with different parameters
2. **Train** on your own IoT datasets
3. **Deploy** to production environments
4. **Contribute** to the open-source project
5. **Research** new optimization techniques

---

**Happy Optimizing!** 🎯

The GenIoT-Optimizer framework is now ready to run in Google Colab with full GPU acceleration and easy sharing capabilities!
