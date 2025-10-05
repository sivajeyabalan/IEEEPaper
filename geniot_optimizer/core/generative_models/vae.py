"""
Variational Autoencoder (VAE) for IoT Traffic Pattern Recognition and Anomaly Detection

This module implements a VAE for learning compressed latent representations of IoT traffic patterns
and detecting anomalies in network behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class Encoder(nn.Module):
    """
    Encoder network for VAE that maps traffic patterns to latent space.
    
    Architecture:
    - Input: Traffic sequence (batch_size, sequence_length, features)
    - Output: Mean and log variance of latent distribution
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        sequence_length: int = 100,
        hidden_dim: int = 256,
        latent_dim: int = 32
    ):
        super(Encoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Latent space projection
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode traffic patterns to latent space.
        
        Args:
            x: Traffic tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        # Transpose for conv1d: (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Transpose back for LSTM: (batch_size, sequence_length, hidden_dim)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling
        x = torch.mean(lstm_out, dim=1)
        
        # Project to latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network for VAE that reconstructs traffic patterns from latent space.
    
    Architecture:
    - Input: Latent vector (batch_size, latent_dim)
    - Output: Reconstructed traffic sequence (batch_size, sequence_length, features)
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        sequence_length: int = 100,
        output_dim: int = 8
    ):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # Initial projection from latent to hidden space
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # LSTM for sequential generation
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output projection layers
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to traffic sequence.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed traffic tensor of shape (batch_size, sequence_length, output_dim)
        """
        batch_size = z.size(0)
        
        # Project to hidden space
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        
        # Expand to sequence length
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Project to output features
        output = self.output_proj(lstm_out)
        
        return self.activation(output)


class VAE(nn.Module):
    """
    Variational Autoencoder for IoT traffic pattern learning and anomaly detection.
    
    This VAE learns compressed representations of normal traffic patterns and can
    detect anomalies by measuring reconstruction error and KL divergence.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        sequence_length: int = 100,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        beta: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(VAE, self).__init__()
        
        self.device = device
        self.latent_dim = latent_dim
        self.beta = beta  # Beta-VAE parameter for KL divergence weighting
        
        # Initialize encoder and decoder
        self.encoder = Encoder(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        ).to(device)
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            sequence_length=sequence_length,
            output_dim=input_dim
        ).to(device)
        
        # Training history
        self.reconstruction_losses = []
        self.kl_losses = []
        self.total_losses = []
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input traffic tensor
            
        Returns:
            reconstructed: Reconstructed traffic
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encode to latent space
        mu, logvar = self.encoder(x)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode from latent space
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss components.
        
        Args:
            x: Original input
            reconstructed: Reconstructed output
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            total_loss: Combined VAE loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_step(
        self,
        x: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            x: Input batch
            optimizer: Optimizer
            
        Returns:
            Dictionary of loss values
        """
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, mu, logvar = self.forward(x)
        
        # Compute losses
        total_loss, recon_loss, kl_loss = self.compute_loss(x, reconstructed, mu, logvar)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def train_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 0.001,
        save_interval: int = 10
    ) -> Dict[str, list]:
        """
        Train the VAE model.
        
        Args:
            dataloader: DataLoader for training data
            epochs: Number of training epochs
            lr: Learning rate
            save_interval: Interval for printing progress
            
        Returns:
            Training history dictionary
        """
        super().train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            epoch_total_loss = 0
            num_batches = 0
            
            for batch_idx, data in enumerate(dataloader):
                if isinstance(data, (list, tuple)):
                    data = data[0]
                
                data = data.to(self.device)
                
                # Training step
                losses = self.train_step(data, optimizer)
                
                epoch_recon_loss += losses['reconstruction_loss']
                epoch_kl_loss += losses['kl_loss']
                epoch_total_loss += losses['total_loss']
                num_batches += 1
            
            # Average losses
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches
            
            self.reconstruction_losses.append(avg_recon_loss)
            self.kl_losses.append(avg_kl_loss)
            self.total_losses.append(avg_total_loss)
            
            if epoch % save_interval == 0:
                print(f"Epoch [{epoch}/{epochs}] - Total: {avg_total_loss:.4f}, "
                      f"Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}")
        
        return {
            "reconstruction_losses": self.reconstruction_losses,
            "kl_losses": self.kl_losses,
            "total_losses": self.total_losses
        }
    
    def detect_anomalies(
        self,
        dataloader: torch.utils.data.DataLoader,
        threshold_percentile: float = 95.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in traffic patterns.
        
        Args:
            dataloader: DataLoader for test data
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            anomaly_scores: Reconstruction error scores
            is_anomaly: Boolean array indicating anomalies
        """
        self.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for data in dataloader:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                
                data = data.to(self.device)
                
                # Forward pass
                reconstructed, _, _ = self.forward(data)
                
                # Compute reconstruction error (MSE per sample)
                recon_error = F.mse_loss(reconstructed, data, reduction='none')
                recon_error = torch.mean(recon_error, dim=(1, 2))  # Average over sequence and features
                
                reconstruction_errors.extend(recon_error.cpu().numpy())
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Determine threshold
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        
        # Identify anomalies
        is_anomaly = reconstruction_errors > threshold
        
        return reconstruction_errors, is_anomaly
    
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generate new traffic samples from the learned latent distribution.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated traffic samples
        """
        self.eval()
        
        with torch.no_grad():
            # Sample from prior distribution
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            
            # Decode to traffic patterns
            generated_samples = self.decoder(z)
        
        return generated_samples
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two traffic patterns in latent space.
        
        Args:
            x1: First traffic pattern
            x2: Second traffic pattern
            steps: Number of interpolation steps
            
        Returns:
            Interpolated traffic patterns
        """
        self.eval()
        
        with torch.no_grad():
            # Encode both patterns
            mu1, _ = self.encoder(x1)
            mu2, _ = self.encoder(x2)
            
            # Interpolate in latent space
            interpolations = []
            for i in range(steps):
                alpha = i / (steps - 1)
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                
                # Decode interpolated latent vector
                interp_sample = self.decoder(z_interp)
                interpolations.append(interp_sample)
        
        return torch.cat(interpolations, dim=0)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'reconstruction_losses': self.reconstruction_losses,
            'kl_losses': self.kl_losses,
            'total_losses': self.total_losses,
            'latent_dim': self.latent_dim,
            'beta': self.beta
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.reconstruction_losses = checkpoint['reconstruction_losses']
        self.kl_losses = checkpoint['kl_losses']
        self.total_losses = checkpoint['total_losses']
        self.latent_dim = checkpoint['latent_dim']
        self.beta = checkpoint['beta']

