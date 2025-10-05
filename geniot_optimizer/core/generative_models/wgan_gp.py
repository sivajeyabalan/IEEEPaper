"""
Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)

This module implements WGAN-GP for generating synthetic IoT traffic patterns.
Based on the paper: "Improved Training of Wasserstein GANs" by Gulrajani et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class Generator(nn.Module):
    """
    Generator network for WGAN-GP that transforms random noise into synthetic IoT traffic.
    
    Architecture:
    - Input: Random noise z ~ N(0, I)
    - Output: Synthetic traffic sequence
    """
    
    def __init__(
        self,
        noise_dim: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 64,
        sequence_length: int = 100,
        num_features: int = 8
    ):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.sequence_length = sequence_length
        self.num_features = num_features
        
        # Fully connected layers for initial processing
        self.fc1 = nn.Linear(noise_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # LSTM layers for sequential generation
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output projection layers
        self.output_proj = nn.Linear(hidden_dim, num_features)
        self.activation = nn.Tanh()
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic traffic from random noise.
        
        Args:
            z: Random noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            Generated traffic tensor of shape (batch_size, sequence_length, num_features)
        """
        batch_size = z.size(0)
        
        # Initial processing through FC layers
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Expand to sequence length
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Project to output features
        output = self.output_proj(lstm_out)
        
        return self.activation(output)


class Discriminator(nn.Module):
    """
    Discriminator network for WGAN-GP that evaluates authenticity of traffic patterns.
    
    Architecture:
    - Input: Traffic sequence (real or generated)
    - Output: Wasserstein distance estimate
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        sequence_length: int = 100,
        hidden_dim: int = 256
    ):
        super(Discriminator, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        
        # Convolutional layers for spatial feature extraction
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
        
        # Final classification layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate authenticity of traffic patterns.
        
        Args:
            x: Traffic tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Wasserstein distance estimate of shape (batch_size, 1)
        """
        # Transpose for conv1d: (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # Convolutional feature extraction
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        
        # Transpose back for LSTM: (batch_size, sequence_length, hidden_dim)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling
        x = torch.mean(lstm_out, dim=1)
        
        # Final classification
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class WGAN_GP:
    """
    Wasserstein Generative Adversarial Network with Gradient Penalty.
    
    This class implements the complete WGAN-GP training procedure for generating
    synthetic IoT traffic patterns.
    """
    
    def __init__(
        self,
        noise_dim: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 64,
        sequence_length: int = 100,
        num_features: int = 8,
        lr_g: float = 0.0001,
        lr_d: float = 0.0001,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        # Initialize networks
        self.generator = Generator(
            noise_dim=noise_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            num_features=num_features
        ).to(device)
        
        self.discriminator = Discriminator(
            input_dim=num_features,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Initialize optimizers
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        
    def compute_gradient_penalty(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            real_samples: Real traffic samples
            fake_samples: Generated traffic samples
            
        Returns:
            Gradient penalty loss
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        
        # Interpolate between real and fake samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Compute discriminator output for interpolated samples
        d_interpolated = self.discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_discriminator(
        self,
        real_samples: torch.Tensor,
        batch_size: int
    ) -> float:
        """
        Train discriminator for one step.
        
        Args:
            real_samples: Real traffic samples
            batch_size: Batch size
            
        Returns:
            Discriminator loss
        """
        self.optimizer_d.zero_grad()
        
        # Generate fake samples
        noise = torch.randn(batch_size, self.generator.noise_dim).to(self.device)
        fake_samples = self.generator(noise).detach()
        
        # Compute discriminator outputs
        d_real = self.discriminator(real_samples)
        d_fake = self.discriminator(fake_samples)
        
        # Compute gradient penalty
        gradient_penalty = self.compute_gradient_penalty(real_samples, fake_samples)
        
        # Compute Wasserstein loss with gradient penalty
        d_loss = -torch.mean(d_real) + torch.mean(d_fake) + self.lambda_gp * gradient_penalty
        
        d_loss.backward()
        self.optimizer_d.step()
        
        return d_loss.item()
    
    def train_generator(self, batch_size: int) -> float:
        """
        Train generator for one step.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Generator loss
        """
        self.optimizer_g.zero_grad()
        
        # Generate fake samples
        noise = torch.randn(batch_size, self.generator.noise_dim).to(self.device)
        fake_samples = self.generator(noise)
        
        # Compute discriminator output
        d_fake = self.discriminator(fake_samples)
        
        # Compute generator loss (negative of discriminator output)
        g_loss = -torch.mean(d_fake)
        
        g_loss.backward()
        self.optimizer_g.step()
        
        return g_loss.item()
    
    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        save_interval: int = 10
    ) -> Dict[str, list]:
        """
        Train the WGAN-GP model.
        
        Args:
            dataloader: DataLoader for real traffic data
            epochs: Number of training epochs
            save_interval: Interval for saving model checkpoints
            
        Returns:
            Training history dictionary
        """
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            num_batches = 0
            
            for batch_idx, real_data in enumerate(dataloader):
                if isinstance(real_data, (list, tuple)):
                    real_data = real_data[0]
                
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Train discriminator multiple times
                for _ in range(self.n_critic):
                    d_loss = self.train_discriminator(real_data, batch_size)
                    epoch_d_loss += d_loss
                
                # Train generator once
                g_loss = self.train_generator(batch_size)
                epoch_g_loss += g_loss
                num_batches += 1
            
            # Average losses
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / (num_batches * self.n_critic)
            
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            if epoch % save_interval == 0:
                print(f"Epoch [{epoch}/{epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        
        return {
            "generator_losses": self.g_losses,
            "discriminator_losses": self.d_losses
        }
    
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generate synthetic traffic samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated traffic samples
        """
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.generator.noise_dim).to(self.device)
            generated_samples = self.generator(noise)
        
        return generated_samples
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']

