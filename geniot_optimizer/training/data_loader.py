"""
IoT Data Loader for Training Pipeline

This module handles loading and preprocessing of IoT datasets for training
the GenIoT-Optimizer framework.
"""

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class IoTDataset(data.Dataset):
    """
    Custom dataset class for IoT traffic data.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        transform: Optional[callable] = None
    ):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            return torch.FloatTensor(sample), torch.FloatTensor(self.labels[idx])
        else:
            return torch.FloatTensor(sample)


class IoTDataLoader:
    """
    Data loader for IoT datasets with preprocessing and augmentation capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = None
        self.feature_names = None
        self.logger = logging.getLogger(__name__)
        
        # Data preprocessing parameters
        self.sequence_length = config.get('sequence_length', 100)
        self.num_features = config.get('num_features', 8)
        self.normalization_method = config.get('normalization', 'standard')  # 'standard' or 'minmax'
        self.augmentation_enabled = config.get('augmentation', True)
        
    def load_and_split_data(
        self,
        data_path: str,
        validation_split: float = 0.2,
        test_split: float = 0.1
    ) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
        """
        Load IoT dataset and split into train/validation/test sets.
        
        Args:
            data_path: Path to dataset file
            validation_split: Fraction for validation set
            test_split: Fraction for test set
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.logger.info(f"Loading data from {data_path}")
        
        # Load raw data
        raw_data = self._load_raw_data(data_path)
        
        # Preprocess data
        processed_data = self._preprocess_data(raw_data)
        
        # Split data
        train_data, val_data, test_data = self._split_data(
            processed_data, validation_split, test_split
        )
        
        # Create data loaders
        train_loader = self._create_data_loader(train_data, shuffle=True)
        val_loader = self._create_data_loader(val_data, shuffle=False)
        test_loader = self._create_data_loader(test_data, shuffle=False)
        
        self.logger.info(f"Data loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_loader, val_loader, test_loader
    
    def _load_raw_data(self, data_path: str) -> Dict[str, Any]:
        """Load raw data from various formats."""
        data_path = Path(data_path)
        
        if data_path.suffix == '.csv':
            return self._load_csv_data(data_path)
        elif data_path.suffix == '.json':
            return self._load_json_data(data_path)
        elif data_path.suffix == '.pkl':
            return self._load_pickle_data(data_path)
        elif data_path.suffix == '.npy':
            return self._load_numpy_data(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    def _load_csv_data(self, file_path: Path) -> Dict[str, Any]:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        
        # Extract features and labels
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'device_id', 'label']]
        self.feature_names = feature_columns
        
        data_dict = {
            'features': df[feature_columns].values,
            'timestamps': df['timestamp'].values if 'timestamp' in df.columns else None,
            'device_ids': df['device_id'].values if 'device_id' in df.columns else None,
            'labels': df['label'].values if 'label' in df.columns else None
        }
        
        return data_dict
    
    def _load_json_data(self, file_path: Path) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        # Convert JSON to numpy arrays
        features = np.array(json_data.get('features', []))
        labels = np.array(json_data.get('labels', [])) if 'labels' in json_data else None
        
        data_dict = {
            'features': features,
            'labels': labels,
            'timestamps': json_data.get('timestamps'),
            'device_ids': json_data.get('device_ids')
        }
        
        return data_dict
    
    def _load_pickle_data(self, file_path: Path) -> Dict[str, Any]:
        """Load data from pickle file."""
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        return data_dict
    
    def _load_numpy_data(self, file_path: Path) -> Dict[str, Any]:
        """Load data from numpy file."""
        data_array = np.load(file_path)
        
        data_dict = {
            'features': data_array,
            'labels': None,
            'timestamps': None,
            'device_ids': None
        }
        
        return data_dict
    
    def _preprocess_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess raw data."""
        features = raw_data['features']
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        # Normalize features
        features = self._normalize_features(features)
        
        # Create sequences
        sequences = self._create_sequences(features)
        
        # Data augmentation
        if self.augmentation_enabled:
            sequences = self._augment_data(sequences)
        
        return sequences
    
    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in the data."""
        # Replace NaN values with median
        for i in range(features.shape[1]):
            column = features[:, i]
            median_val = np.nanmedian(column)
            features[:, i] = np.where(np.isnan(column), median_val, column)
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using specified method."""
        if self.normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        normalized_features = self.scaler.fit_transform(features)
        return normalized_features
    
    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """Create sequences from time series data."""
        sequences = []
        
        for i in range(len(features) - self.sequence_length + 1):
            sequence = features[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def _augment_data(self, sequences: np.ndarray) -> np.ndarray:
        """Apply data augmentation techniques."""
        augmented_sequences = [sequences]
        
        # Add noise
        noise_sequences = sequences + np.random.normal(0, 0.01, sequences.shape)
        augmented_sequences.append(noise_sequences)
        
        # Time warping
        warped_sequences = self._time_warp(sequences)
        augmented_sequences.append(warped_sequences)
        
        # Magnitude scaling
        scaled_sequences = sequences * np.random.uniform(0.9, 1.1, (sequences.shape[0], 1, 1))
        augmented_sequences.append(scaled_sequences)
        
        return np.concatenate(augmented_sequences, axis=0)
    
    def _time_warp(self, sequences: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply time warping augmentation."""
        warped_sequences = sequences.copy()
        
        for i in range(sequences.shape[0]):
            sequence = sequences[i]
            seq_len = sequence.shape[0]
            
            # Generate warping curve
            warping_curve = np.cumsum(np.random.normal(1, sigma, seq_len))
            warping_curve = warping_curve / warping_curve[-1] * (seq_len - 1)
            
            # Apply warping
            for j in range(sequence.shape[1]):
                warped_sequences[i, :, j] = np.interp(
                    np.arange(seq_len), warping_curve, sequence[:, j]
                )
        
        return warped_sequences
    
    def _split_data(
        self,
        data: np.ndarray,
        validation_split: float,
        test_split: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets."""
        # First split: train+val vs test
        train_val_data, test_data = train_test_split(
            data, test_size=test_split, random_state=42
        )
        
        # Second split: train vs val
        train_data, val_data = train_test_split(
            train_val_data, test_size=validation_split/(1-test_split), random_state=42
        )
        
        return train_data, val_data, test_data
    
    def _create_data_loader(
        self,
        data: np.ndarray,
        shuffle: bool = True,
        batch_size: Optional[int] = None
    ) -> data.DataLoader:
        """Create PyTorch DataLoader."""
        if batch_size is None:
            batch_size = self.config.get('batch_size', 32)
        
        dataset = IoTDataset(data)
        
        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
    
    def generate_synthetic_dataset(
        self,
        num_samples: int = 10000,
        num_devices: int = 100,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate synthetic IoT dataset for testing purposes.
        
        Args:
            num_samples: Number of samples to generate
            num_devices: Number of devices
            save_path: Optional path to save generated data
            
        Returns:
            Generated synthetic data
        """
        self.logger.info(f"Generating synthetic dataset: {num_samples} samples, {num_devices} devices")
        
        # Generate synthetic features
        features = []
        
        for _ in range(num_samples):
            # Device-specific features
            device_id = np.random.randint(0, num_devices)
            device_type = np.random.choice(['sensor', 'actuator', 'gateway', 'camera'])
            
            # Generate realistic IoT features
            sample_features = self._generate_device_features(device_type)
            features.append(sample_features)
        
        features = np.array(features)
        
        # Create sequences
        sequences = self._create_sequences(features)
        
        # Save if path provided
        if save_path:
            self._save_synthetic_data(sequences, save_path)
        
        return sequences
    
    def _generate_device_features(self, device_type: str) -> np.ndarray:
        """Generate realistic features for a specific device type."""
        if device_type == 'sensor':
            # Low power, periodic data
            features = [
                np.random.uniform(0.1, 0.5),  # CPU usage
                np.random.uniform(0.1, 0.3),  # Memory usage
                np.random.uniform(80, 100),   # Battery level
                np.random.uniform(20, 30),    # Temperature
                np.random.uniform(-70, -50),  # Signal strength
                np.random.uniform(0, 0.01),   # Packet loss rate
                np.random.uniform(10, 50),    # Latency
                np.random.uniform(1, 10)      # Throughput
            ]
        elif device_type == 'actuator':
            # Medium power, event-driven
            features = [
                np.random.uniform(0.3, 0.7),  # CPU usage
                np.random.uniform(0.2, 0.5),  # Memory usage
                np.random.uniform(60, 90),    # Battery level
                np.random.uniform(25, 40),    # Temperature
                np.random.uniform(-60, -40),  # Signal strength
                np.random.uniform(0, 0.02),   # Packet loss rate
                np.random.uniform(20, 80),    # Latency
                np.random.uniform(5, 20)      # Throughput
            ]
        elif device_type == 'gateway':
            # High power, always on
            features = [
                np.random.uniform(0.5, 0.9),  # CPU usage
                np.random.uniform(0.4, 0.8),  # Memory usage
                np.random.uniform(40, 80),    # Battery level
                np.random.uniform(30, 50),    # Temperature
                np.random.uniform(-50, -30),  # Signal strength
                np.random.uniform(0, 0.01),   # Packet loss rate
                np.random.uniform(5, 30),     # Latency
                np.random.uniform(20, 100)    # Throughput
            ]
        else:  # camera
            # Very high power, high bandwidth
            features = [
                np.random.uniform(0.7, 0.95), # CPU usage
                np.random.uniform(0.6, 0.9),  # Memory usage
                np.random.uniform(20, 60),    # Battery level
                np.random.uniform(35, 60),    # Temperature
                np.random.uniform(-40, -20),  # Signal strength
                np.random.uniform(0, 0.03),   # Packet loss rate
                np.random.uniform(30, 100),   # Latency
                np.random.uniform(50, 200)    # Throughput
            ]
        
        return np.array(features)
    
    def _save_synthetic_data(self, data: np.ndarray, save_path: str) -> None:
        """Save synthetic data to file."""
        save_path = Path(save_path)
        
        if save_path.suffix == '.npy':
            np.save(save_path, data)
        elif save_path.suffix == '.csv':
            # Flatten sequences for CSV
            flattened_data = data.reshape(data.shape[0], -1)
            df = pd.DataFrame(flattened_data)
            df.to_csv(save_path, index=False)
        else:
            raise ValueError(f"Unsupported save format: {save_path.suffix}")
        
        self.logger.info(f"Synthetic data saved to {save_path}")
    
    def get_data_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        stats = {
            'num_samples': len(data),
            'sequence_length': data.shape[1],
            'num_features': data.shape[2],
            'mean': np.mean(data, axis=(0, 1)),
            'std': np.std(data, axis=(0, 1)),
            'min': np.min(data, axis=(0, 1)),
            'max': np.max(data, axis=(0, 1))
        }
        
        if self.feature_names:
            stats['feature_names'] = self.feature_names
        
        return stats
