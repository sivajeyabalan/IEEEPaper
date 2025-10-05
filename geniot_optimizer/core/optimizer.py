"""
Multi-Objective Optimizer with Proximal Policy Optimization (PPO)

This module implements a PPO-based multi-objective optimizer that balances competing
objectives including latency, throughput, energy efficiency, and QoS satisfaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque
import gymnasium as gym
from gymnasium import spaces


class PolicyNetwork(nn.Module):
    """
    Policy network for PPO that outputs action probabilities.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action logits
        """
        return self.network(state)


class ValueNetwork(nn.Module):
    """
    Value network for PPO that estimates state values.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.
        
        Args:
            state: Input state tensor
            
        Returns:
            State value estimate
        """
        return self.network(state)


class IoTNetworkEnvironment(gym.Env):
    """
    Custom Gym environment for IoT network optimization.
    
    This environment simulates an IoT network where the agent can make
    optimization decisions to improve network performance.
    """
    
    def __init__(
        self,
        num_devices: int = 100,
        num_features: int = 8,
        max_steps: int = 1000,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        super(IoTNetworkEnvironment, self).__init__()
        
        self.num_devices = num_devices
        self.num_features = num_features
        self.max_steps = max_steps
        self.current_step = 0
        
        # Default reward weights
        self.reward_weights = reward_weights or {
            'latency': 0.3,
            'throughput': 0.3,
            'energy': 0.2,
            'qos': 0.2
        }
        
        # Action space: configuration parameters for each device
        # Actions include: transmission power, routing decisions, resource allocation
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_devices * 3,),  # 3 config parameters per device
            dtype=np.float32
        )
        
        # State space: network state + traffic patterns
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(num_devices * num_features + num_devices * 3,),  # State + previous actions
            dtype=np.float32
        )
        
        # Initialize network state
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize network state
        self.network_state = np.random.uniform(0, 1, (self.num_devices, self.num_features))
        self.previous_actions = np.zeros(self.num_devices * 3)
        
        # Initialize performance metrics
        self.performance_metrics = {
            'latency': np.random.uniform(10, 100, self.num_devices),
            'throughput': np.random.uniform(1, 100, self.num_devices),
            'energy': np.random.uniform(0.1, 10, self.num_devices),
            'qos': np.random.uniform(0.5, 1.0, self.num_devices)
        }
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Apply action to network configuration
        self._apply_action(action)
        
        # Update network state based on action
        self._update_network_state(action)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action to network configuration."""
        # Reshape action to per-device configuration
        device_actions = action.reshape(self.num_devices, 3)
        
        # Update device configurations
        for i in range(self.num_devices):
            # Transmission power adjustment
            power_factor = device_actions[i, 0]
            self.network_state[i, 0] *= (0.5 + power_factor)  # Packet size affected by power
            
            # Routing decision
            routing_factor = device_actions[i, 1]
            self.network_state[i, 1] *= (0.8 + 0.4 * routing_factor)  # Transmission rate
            
            # Resource allocation
            resource_factor = device_actions[i, 2]
            self.network_state[i, 4] *= (0.7 + 0.6 * resource_factor)  # Energy consumption
        
        # Store previous actions
        self.previous_actions = action.copy()
    
    def _update_network_state(self, action: np.ndarray) -> None:
        """Update network state based on current configuration."""
        # Simulate network dynamics
        noise = np.random.normal(0, 0.1, self.network_state.shape)
        self.network_state += noise
        
        # Ensure state stays within bounds
        self.network_state = np.clip(self.network_state, 0, 10)
        
        # Update performance metrics based on network state
        self._update_performance_metrics()
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on current network state."""
        # Latency (inversely related to transmission rate)
        self.performance_metrics['latency'] = 50 / (self.network_state[:, 1] + 0.1)
        
        # Throughput (related to packet size and transmission rate)
        self.performance_metrics['throughput'] = (
            self.network_state[:, 0] * self.network_state[:, 1] * 10
        )
        
        # Energy consumption (related to transmission power)
        self.performance_metrics['energy'] = (
            self.network_state[:, 0] * self.network_state[:, 4] * 2
        )
        
        # QoS (related to overall network health)
        self.performance_metrics['qos'] = np.clip(
            1.0 - (self.performance_metrics['latency'] / 100) - 
            (self.performance_metrics['energy'] / 20),
            0.0, 1.0
        )
    
    def _calculate_reward(self) -> float:
        """Calculate multi-objective reward."""
        # Normalize metrics to [0, 1] range
        normalized_metrics = {
            'latency': 1.0 - np.clip(self.performance_metrics['latency'] / 100, 0, 1),
            'throughput': np.clip(self.performance_metrics['throughput'] / 100, 0, 1),
            'energy': 1.0 - np.clip(self.performance_metrics['energy'] / 20, 0, 1),
            'qos': self.performance_metrics['qos']
        }
        
        # Calculate weighted reward
        reward = 0.0
        for metric, weight in self.reward_weights.items():
            if metric in normalized_metrics:
                reward += weight * np.mean(normalized_metrics[metric])
        
        return float(reward)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Flatten network state and previous actions
        state_flat = self.network_state.flatten()
        observation = np.concatenate([state_flat, self.previous_actions])
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'step': self.current_step
        }


class PPOAgent:
    """
    Proximal Policy Optimization agent for multi-objective IoT network optimization.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(device)
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Training history
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'rewards': []
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        with torch.no_grad():
            # Get action logits
            action_logits = self.policy_net(state)
            
            # Create action distribution
            action_dist = torch.distributions.Normal(
                torch.tanh(action_logits),  # Mean
                torch.ones_like(action_logits) * 0.1  # Standard deviation
            )
            
            if deterministic:
                action = torch.tanh(action_logits)
            else:
                action = action_dist.sample()
            
            # Clip action to valid range
            action = torch.clamp(action, 0.0, 1.0)
            
            # Calculate log probability
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            
            # Get state value
            value = self.value_net(state)
            
        return action, log_prob, value
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            next_values: Next state value estimates
            dones: Done flags
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Compute advantages in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] if not dones[t] else 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        epochs: int = 4
    ) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            states: State sequence
            actions: Action sequence
            old_log_probs: Old log probabilities
            rewards: Reward sequence
            values: Value estimates
            dones: Done flags
            epochs: Number of update epochs
            
        Returns:
            Dictionary of loss values
        """
        # Compute advantages and returns
        with torch.no_grad():
            next_values = torch.cat([values[1:], torch.zeros(1, 1).to(self.device)])
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update networks for multiple epochs
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(epochs):
            # Get current policy outputs
            action_logits = self.policy_net(states)
            current_values = self.value_net(states)
            
            # Create action distribution
            action_dist = torch.distributions.Normal(
                torch.tanh(action_logits),
                torch.ones_like(action_logits) * 0.1
            )
            
            # Calculate current log probabilities
            current_log_probs = action_dist.log_prob(actions).sum(dim=-1)
            
            # Calculate entropy
            entropy = action_dist.entropy().sum(dim=-1).mean()
            
            # Calculate probability ratio
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # Calculate clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(current_values.squeeze(), returns)
            
            # Calculate total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            
            # Store losses
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())
        
        # Update training history
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy_loss = np.mean(entropy_losses)
        avg_total_loss = avg_policy_loss + self.value_coef * avg_value_loss - self.entropy_coef * avg_entropy_loss
        
        self.training_history['policy_loss'].append(avg_policy_loss)
        self.training_history['value_loss'].append(avg_value_loss)
        self.training_history['entropy_loss'].append(avg_entropy_loss)
        self.training_history['total_loss'].append(avg_total_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_total_loss
        }


class MultiObjectiveOptimizer:
    """
    High-level interface for multi-objective IoT network optimization using PPO.
    """
    
    def __init__(
        self,
        num_devices: int = 100,
        num_features: int = 8,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        reward_weights: Optional[Dict[str, float]] = None
    ):
        self.device = device
        self.num_devices = num_devices
        
        # Initialize environment
        self.env = IoTNetworkEnvironment(
            num_devices=num_devices,
            num_features=num_features,
            reward_weights=reward_weights
        )
        
        # Get state and action dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            device=device
        )
        
        # Training parameters
        self.buffer_size = 2048
        self.batch_size = 64
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def collect_rollouts(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """
        Collect rollouts from the environment.
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary containing rollout data
        """
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        state, _ = self.env.reset()
        state = torch.FloatTensor(state).to(self.device)
        
        for step in range(num_steps):
            # Get action from policy
            action, log_prob, value = self.agent.get_action(state)
            
            # Execute action in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            # Update state
            if done:
                state, _ = self.env.reset()
                state = torch.FloatTensor(state).to(self.device)
            else:
                state = torch.FloatTensor(next_state).to(self.device)
        
        # Convert to tensors
        rollout_data = {
            'states': torch.stack(states),
            'actions': torch.stack(actions),
            'rewards': torch.FloatTensor(rewards).to(self.device),
            'values': torch.stack(values).squeeze(),
            'log_probs': torch.stack(log_probs),
            'dones': torch.BoolTensor(dones).to(self.device)
        }
        
        return rollout_data
    
    def train(
        self,
        total_timesteps: int = 100000,
        rollout_length: int = 2048,
        update_epochs: int = 4,
        save_interval: int = 10000,
        save_path: str = "./models"
    ) -> Dict[str, List[float]]:
        """
        Train the multi-objective optimizer.
        
        Args:
            total_timesteps: Total number of training timesteps
            rollout_length: Length of each rollout
            update_epochs: Number of update epochs per rollout
            save_interval: Interval for saving model
            save_path: Path to save model
            
        Returns:
            Training history
        """
        self.logger.info("Starting multi-objective optimization training...")
        
        timesteps = 0
        episode_rewards = []
        
        while timesteps < total_timesteps:
            # Collect rollouts
            rollout_data = self.collect_rollouts(rollout_length)
            timesteps += rollout_length
            
            # Update agent
            loss_dict = self.agent.update(
                states=rollout_data['states'],
                actions=rollout_data['actions'],
                old_log_probs=rollout_data['log_probs'],
                rewards=rollout_data['rewards'],
                values=rollout_data['values'],
                dones=rollout_data['dones'],
                epochs=update_epochs
            )
            
            # Calculate episode reward
            episode_reward = rollout_data['rewards'].sum().item()
            episode_rewards.append(episode_reward)
            self.agent.training_history['rewards'].append(episode_reward)
            
            # Log progress
            if timesteps % save_interval == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                self.logger.info(
                    f"Timesteps: {timesteps}/{total_timesteps}, "
                    f"Avg Reward: {avg_reward:.4f}, "
                    f"Policy Loss: {loss_dict['policy_loss']:.4f}, "
                    f"Value Loss: {loss_dict['value_loss']:.4f}"
                )
                
                # Save model
                self.save_model(f"{save_path}/optimizer_{timesteps}.pth")
        
        self.logger.info("Training completed!")
        return self.agent.training_history
    
    def optimize_network(
        self,
        initial_state: Optional[np.ndarray] = None,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Optimize network configuration for given initial state.
        
        Args:
            initial_state: Initial network state
            max_steps: Maximum optimization steps
            
        Returns:
            Optimization results
        """
        self.agent.policy_net.eval()
        
        if initial_state is not None:
            state = torch.FloatTensor(initial_state).to(self.device)
        else:
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
        
        optimization_history = {
            'states': [],
            'actions': [],
            'rewards': [],
            'performance_metrics': []
        }
        
        total_reward = 0
        
        with torch.no_grad():
            for step in range(max_steps):
                # Get optimized action
                action, _, _ = self.agent.get_action(state, deterministic=True)
                
                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
                done = terminated or truncated
                
                # Store results
                optimization_history['states'].append(state.cpu().numpy())
                optimization_history['actions'].append(action.cpu().numpy())
                optimization_history['rewards'].append(reward)
                optimization_history['performance_metrics'].append(info['performance_metrics'])
                
                total_reward += reward
                
                if done:
                    break
                
                state = torch.FloatTensor(next_state).to(self.device)
        
        optimization_history['total_reward'] = total_reward
        optimization_history['final_performance'] = optimization_history['performance_metrics'][-1]
        
        return optimization_history
    
    def save_model(self, filepath: str) -> None:
        """Save the trained optimizer."""
        torch.save({
            'policy_state_dict': self.agent.policy_net.state_dict(),
            'value_state_dict': self.agent.value_net.state_dict(),
            'training_history': self.agent.training_history,
            'num_devices': self.num_devices
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained optimizer."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.agent.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.agent.training_history = checkpoint['training_history']
        self.num_devices = checkpoint['num_devices']
