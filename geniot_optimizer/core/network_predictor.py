"""
Network State Predictor with Transformer Architecture

This module implements a Transformer-based network state prediction system that forecasts
future network conditions based on current observations and synthetic traffic patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import logging


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer to handle sequential data.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Input with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class NetworkStateEncoder(nn.Module):
    """
    Encoder for network state features.
    
    This module processes various network state features including:
    - Traffic patterns
    - Device states
    - Network topology information
    - Performance metrics
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        super(NetworkStateEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Feature processing layers
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Normalization and activation
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode network state features.
        
        Args:
            x: Input features of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Encoded features of shape (batch_size, seq_len, output_dim)
        """
        # Project to hidden dimension
        x = self.feature_projection(x)
        
        # Self-attention
        attn_output, _ = self.feature_attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        x = self.dropout(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class NetworkStatePredictorModel(nn.Module):
    """
    Transformer-based network state predictor.
    
    This model predicts future network states based on current observations
    and synthetic traffic patterns using multi-head self-attention mechanisms.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        prediction_horizon: int = 10
    ):
        super(NetworkStatePredictorModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Network state encoder (input_dim * 2 because we concatenate traffic_patterns and current_states)
        self.state_encoder = NetworkStateEncoder(
            input_dim=input_dim * 2,  # Fixed: account for concatenation
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Prediction heads for different network metrics
        self.latency_predictor = nn.Linear(hidden_dim, prediction_horizon)
        self.throughput_predictor = nn.Linear(hidden_dim, prediction_horizon)
        self.energy_predictor = nn.Linear(hidden_dim, prediction_horizon)
        self.qos_predictor = nn.Linear(hidden_dim, prediction_horizon)
        self.congestion_predictor = nn.Linear(hidden_dim, prediction_horizon)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        traffic_patterns: torch.Tensor,
        current_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future network states.
        
        Args:
            traffic_patterns: Traffic patterns of shape (batch_size, seq_len, input_dim)
            current_states: Current network states of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Dictionary of predictions for different network metrics
        """
        batch_size, seq_len, _ = traffic_patterns.shape
        
        # Combine traffic patterns and current states
        combined_input = torch.cat([traffic_patterns, current_states], dim=-1)
        
        # Encode network states
        encoded_states = self.state_encoder(combined_input)
        
        # Add positional encoding
        encoded_states = encoded_states.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        encoded_states = self.pos_encoding(encoded_states)
        encoded_states = encoded_states.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Apply dropout
        encoded_states = self.dropout(encoded_states)
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(encoded_states, src_key_padding_mask=mask)
        
        # Global average pooling for sequence-level prediction
        pooled_output = torch.mean(transformer_output, dim=1)  # (batch_size, hidden_dim)
        
        # Generate predictions for different metrics
        predictions = {
            'latency': self.latency_predictor(pooled_output),
            'throughput': self.throughput_predictor(pooled_output),
            'energy': self.energy_predictor(pooled_output),
            'qos': self.qos_predictor(pooled_output),
            'congestion': self.congestion_predictor(pooled_output)
        }
        
        return predictions
    
    def predict_single_step(
        self,
        traffic_patterns: torch.Tensor,
        current_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next single step network states.
        
        Args:
            traffic_patterns: Current traffic patterns
            current_states: Current network states
            
        Returns:
            Single-step predictions
        """
        predictions = self.forward(traffic_patterns, current_states)
        
        # Extract first timestep predictions
        single_step_predictions = {}
        for metric, pred in predictions.items():
            single_step_predictions[metric] = pred[:, 0:1]  # First timestep only
        
        return single_step_predictions


class NetworkStatePredictor:
    """
    High-level interface for network state prediction.
    
    This class provides a complete interface for training and using the
    Transformer-based network state predictor.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        prediction_horizon: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.prediction_horizon = prediction_horizon
        
        # Initialize model
        self.model = NetworkStatePredictorModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            prediction_horizon=prediction_horizon
        ).to(device)
        
        # Loss functions for different metrics
        self.loss_functions = {
            'latency': nn.MSELoss(),
            'throughput': nn.MSELoss(),
            'energy': nn.MSELoss(),
            'qos': nn.MSELoss(),
            'congestion': nn.MSELoss()
        }
        
        # Training history
        self.training_history = {
            'total_loss': [],
            'latency_loss': [],
            'throughput_loss': [],
            'energy_loss': [],
            'qos_loss': [],
            'congestion_loss': []
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute prediction losses.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss_weights: Optional weights for different loss components
            
        Returns:
            Total loss and individual losses
        """
        if loss_weights is None:
            loss_weights = {
                'latency': 1.0,
                'throughput': 1.0,
                'energy': 1.0,
                'qos': 1.0,
                'congestion': 1.0
            }
        
        individual_losses = {}
        total_loss = 0.0
        
        for metric in predictions.keys():
            if metric in targets and metric in self.loss_functions:
                loss = self.loss_functions[metric](predictions[metric], targets[metric])
                weighted_loss = loss_weights[metric] * loss
                individual_losses[metric] = loss
                total_loss += weighted_loss
        
        return total_loss, individual_losses
    
    def train_step(
        self,
        traffic_patterns: torch.Tensor,
        current_states: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            traffic_patterns: Input traffic patterns
            current_states: Current network states
            targets: Target values for different metrics
            optimizer: Optimizer
            loss_weights: Optional loss weights
            
        Returns:
            Dictionary of loss values
        """
        optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(traffic_patterns, current_states)
        
        # Compute loss
        total_loss, individual_losses = self.compute_loss(predictions, targets, loss_weights)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Return loss values
        loss_dict = {'total_loss': total_loss.item()}
        for metric, loss in individual_losses.items():
            loss_dict[f'{metric}_loss'] = loss.item()
        
        return loss_dict
    
    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 0.001,
        save_interval: int = 10,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the network state predictor.
        
        Args:
            dataloader: DataLoader for training data
            epochs: Number of training epochs
            lr: Learning rate
            save_interval: Interval for printing progress
            loss_weights: Optional loss weights
            
        Returns:
            Training history
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        for epoch in range(epochs):
            epoch_losses = {key: 0.0 for key in self.training_history.keys()}
            num_batches = 0
            
            for batch_idx, batch_data in enumerate(dataloader):
                # Unpack batch data
                traffic_patterns = batch_data['traffic_patterns'].to(self.device)
                current_states = batch_data['current_states'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch_data['targets'].items()}
                
                # Training step
                loss_dict = self.train_step(
                    traffic_patterns, current_states, targets, optimizer, loss_weights
                )
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    epoch_losses[key] += value
                num_batches += 1
            
            # Average losses
            for key in epoch_losses.keys():
                avg_loss = epoch_losses[key] / num_batches
                self.training_history[key].append(avg_loss)
            
            # Update learning rate
            scheduler.step(epoch_losses['total_loss'])
            
            if epoch % save_interval == 0:
                self.logger.info(
                    f"Epoch [{epoch}/{epochs}] - "
                    f"Total Loss: {epoch_losses['total_loss']/num_batches:.4f}, "
                    f"Latency: {epoch_losses['latency_loss']/num_batches:.4f}, "
                    f"Throughput: {epoch_losses['throughput_loss']/num_batches:.4f}"
                )
        
        return self.training_history
    
    def predict(
        self,
        traffic_patterns: torch.Tensor,
        current_states: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for network states.
        
        Args:
            traffic_patterns: Input traffic patterns
            current_states: Current network states
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions for different network metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(traffic_patterns, current_states)
            
            if return_uncertainty:
                # Monte Carlo dropout for uncertainty estimation
                self.model.train()  # Enable dropout
                predictions_list = []
                
                for _ in range(10):  # 10 Monte Carlo samples
                    pred = self.model(traffic_patterns, current_states)
                    predictions_list.append(pred)
                
                self.model.eval()  # Disable dropout
                
                # Compute mean and variance
                pred_stack = torch.stack(predictions_list, dim=0)
                pred_mean = torch.mean(pred_stack, dim=0)
                pred_var = torch.var(pred_stack, dim=0)
                
                predictions = {
                    f'{metric}_mean': pred_mean[metric] for metric in pred_mean.keys()
                }
                predictions.update({
                    f'{metric}_var': pred_var[metric] for metric in pred_var.keys()
                })
        
        return predictions
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            dataloader: DataLoader for test data
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation metrics
        """
        if metrics is None:
            metrics = ['latency', 'throughput', 'energy', 'qos', 'congestion']
        
        self.model.eval()
        evaluation_results = {}
        
        with torch.no_grad():
            total_losses = {metric: 0.0 for metric in metrics}
            num_batches = 0
            
            for batch_data in dataloader:
                traffic_patterns = batch_data['traffic_patterns'].to(self.device)
                current_states = batch_data['current_states'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch_data['targets'].items()}
                
                # Make predictions
                predictions = self.model(traffic_patterns, current_states)
                
                # Compute losses
                for metric in metrics:
                    if metric in predictions and metric in targets:
                        loss = self.loss_functions[metric](predictions[metric], targets[metric])
                        total_losses[metric] += loss.item()
                
                num_batches += 1
            
            # Average losses
            for metric in metrics:
                evaluation_results[f'{metric}_mse'] = total_losses[metric] / num_batches
        
        return evaluation_results
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'prediction_horizon': self.prediction_horizon
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint['training_history']
        self.prediction_horizon = checkpoint['prediction_horizon']

