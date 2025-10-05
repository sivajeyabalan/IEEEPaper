"""
Denoising Diffusion Probabilistic Model (DDPM) for Sequential IoT Traffic Generation

This module implements DDPM for generating high-quality sequential IoT traffic patterns
with temporal dependencies and complex statistical properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import math


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding for DDPM."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None]
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        
        return F.silu(h + residual)


class UNet1D(nn.Module):
    """
    1D U-Net architecture for DDPM noise prediction.
    
    This network predicts the noise that was added to the input at each timestep.
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        time_emb_dim: int = 128,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # Time embedding
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial projection
        self.input_proj = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            ResidualBlock(model_channels, model_channels, time_emb_dim, dropout)
            for _ in range(num_res_blocks)
        ])
        
        self.down_sample = nn.Conv1d(model_channels, model_channels, kernel_size=3, stride=2, padding=1)
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(model_channels, model_channels, time_emb_dim, dropout),
            ResidualBlock(model_channels, model_channels, time_emb_dim, dropout)
        ])
        
        # Upsampling blocks
        self.up_sample = nn.ConvTranspose1d(model_channels, model_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.up_blocks = nn.ModuleList([
            ResidualBlock(model_channels * 2, model_channels, time_emb_dim, dropout)
            for _ in range(num_res_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Noisy input tensor (batch_size, channels, sequence_length)
            timesteps: Timestep tensor (batch_size,)
            
        Returns:
            Predicted noise tensor
        """
        # Time embedding
        time_emb = self.time_mlp(timesteps)
        
        # Initial projection
        h = self.input_proj(x)
        
        # Store skip connections
        skip_connections = []
        
        # Downsampling
        for block in self.down_blocks:
            h = block(h, time_emb)
            skip_connections.append(h)
        
        h = self.down_sample(h)
        
        # Middle blocks
        for block in self.middle_blocks:
            h = block(h, time_emb)
        
        # Upsampling
        h = self.up_sample(h)
        
        for block in self.up_blocks:
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, time_emb)
        
        # Output projection
        return self.output_proj(h)


class DDPM:
    """
    Denoising Diffusion Probabilistic Model for IoT traffic generation.
    
    This implementation follows the DDPM paper by Ho et al. and is adapted for
    sequential IoT traffic pattern generation.
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        sequence_length: int = 100,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.num_timesteps = num_timesteps
        self.sequence_length = sequence_length
        
        # Initialize U-Net model
        self.model = UNet1D(
            in_channels=in_channels,
            out_channels=out_channels,
            time_emb_dim=128,
            model_channels=128,
            num_res_blocks=2,
            dropout=0.1
        ).to(device)
        
        # Define noise schedule
        self.betas = self._linear_beta_schedule(num_timesteps, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Training history
        self.losses = []
        
    def _linear_beta_schedule(
        self,
        timesteps: int,
        start: float = 0.0001,
        end: float = 0.02
    ) -> torch.Tensor:
        """Linear beta schedule for noise variance."""
        return torch.linspace(start, end, timesteps)
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """Extract values from a 1-D tensor for a batch of indices."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0).
        
        Args:
            x_start: Original clean data
            t: Timestep tensor
            noise: Optional noise tensor
            
        Returns:
            x_t: Noisy data at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int
    ) -> torch.Tensor:
        """
        Sample from p_θ(x_{t-1} | x_t).
        
        Args:
            x: Noisy data at timestep t
            t: Timestep tensor
            t_index: Current timestep index
            
        Returns:
            x_{t-1}: Denoised data at timestep t-1
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
        
        # Use our model to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def p_sample_loop(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate samples by iteratively denoising.
        
        Args:
            shape: Shape of the samples to generate
            
        Returns:
            Generated samples
        """
        device = next(self.model.parameters()).device
        
        # Start from pure noise
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        
        return img
    
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Generate new samples.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Generated samples
        """
        shape = (batch_size, self.model.in_channels, self.sequence_length)
        return self.p_sample_loop(shape)
    
    def train_step(
        self,
        x_start: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Single training step.
        
        Args:
            x_start: Clean input data
            optimizer: Optimizer
            
        Returns:
            Training loss
        """
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Add noise to data
        x_noisy, _ = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t)
        
        # Compute loss
        loss = F.mse_loss(noise, predicted_noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 0.0001,
        save_interval: int = 10
    ) -> List[float]:
        """
        Train the DDPM model.
        
        Args:
            dataloader: DataLoader for training data
            epochs: Number of training epochs
            lr: Learning rate
            save_interval: Interval for printing progress
            
        Returns:
            Training loss history
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, data in enumerate(dataloader):
                if isinstance(data, (list, tuple)):
                    data = data[0]
                
                data = data.to(self.device)
                
                # Transpose to (batch_size, channels, sequence_length) for conv1d
                if data.dim() == 3:
                    data = data.transpose(1, 2)
                
                # Training step
                loss = self.train_step(data, optimizer)
                
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.losses.append(avg_loss)
            
            if epoch % save_interval == 0:
                print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")
        
        return self.losses
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
            'num_timesteps': self.num_timesteps,
            'losses': self.losses
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.betas = checkpoint['betas']
        self.alphas = checkpoint['alphas']
        self.alphas_cumprod = checkpoint['alphas_cumprod']
        self.num_timesteps = checkpoint['num_timesteps']
        self.losses = checkpoint['losses']
        
        # Recompute derived values
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

