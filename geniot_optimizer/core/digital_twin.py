"""
Digital Twin Integration for Real-time IoT Network Simulation

This module implements a generative AI-powered digital twin that maintains a synchronized
virtual representation of the physical IoT network for simulation and optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import threading
import time
from datetime import datetime, timedelta
import json
from collections import deque
import queue

from .traffic_generator import TrafficGenerator
from .network_predictor import NetworkStatePredictor
from .optimizer import MultiObjectiveOptimizer


class NetworkTopology:
    """
    Represents the network topology of the IoT system.
    """
    
    def __init__(
        self,
        num_devices: int = 100,
        connection_probability: float = 0.1,
        max_connections_per_device: int = 10
    ):
        self.num_devices = num_devices
        self.connection_probability = connection_probability
        self.max_connections_per_device = max_connections_per_device
        
        # Generate network topology
        self.adjacency_matrix = self._generate_topology()
        self.device_positions = self._generate_positions()
        
    def _generate_topology(self) -> np.ndarray:
        """Generate random network topology."""
        adjacency_matrix = np.zeros((self.num_devices, self.num_devices), dtype=bool)
        
        for i in range(self.num_devices):
            # Randomly connect to other devices
            num_connections = np.random.poisson(self.connection_probability * self.num_devices)
            num_connections = min(num_connections, self.max_connections_per_device)
            
            if num_connections > 0:
                # Select random devices to connect to
                available_devices = list(range(self.num_devices))
                available_devices.remove(i)
                
                if available_devices:
                    connections = np.random.choice(
                        available_devices,
                        size=min(num_connections, len(available_devices)),
                        replace=False
                    )
                    
                    for device in connections:
                        adjacency_matrix[i, device] = True
                        adjacency_matrix[device, i] = True  # Bidirectional
        
        return adjacency_matrix
    
    def _generate_positions(self) -> np.ndarray:
        """Generate 2D positions for devices."""
        positions = np.random.uniform(0, 100, (self.num_devices, 2))
        return positions
    
    def get_neighbors(self, device_id: int) -> List[int]:
        """Get list of neighboring devices."""
        return [i for i in range(self.num_devices) if self.adjacency_matrix[device_id, i]]
    
    def get_distance(self, device1: int, device2: int) -> float:
        """Calculate distance between two devices."""
        pos1 = self.device_positions[device1]
        pos2 = self.device_positions[device2]
        return np.sqrt(np.sum((pos1 - pos2) ** 2))


class DeviceState:
    """
    Represents the state of an individual IoT device.
    """
    
    def __init__(self, device_id: int, device_type: str = "sensor"):
        self.device_id = device_id
        self.device_type = device_type
        self.timestamp = datetime.now()
        
        # Device metrics
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.battery_level = 100.0
        self.temperature = 25.0
        self.signal_strength = -50.0
        
        # Network metrics
        self.packet_loss_rate = 0.0
        self.latency = 0.0
        self.throughput = 0.0
        self.energy_consumption = 0.0
        
        # Status flags
        self.is_online = True
        self.is_healthy = True
        self.last_heartbeat = datetime.now()
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update device metrics."""
        for key, value in metrics.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.timestamp = datetime.now()
        self.last_heartbeat = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert device state to dictionary."""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'battery_level': self.battery_level,
            'temperature': self.temperature,
            'signal_strength': self.signal_strength,
            'packet_loss_rate': self.packet_loss_rate,
            'latency': self.latency,
            'throughput': self.throughput,
            'energy_consumption': self.energy_consumption,
            'is_online': self.is_online,
            'is_healthy': self.is_healthy,
            'last_heartbeat': self.last_heartbeat.isoformat()
        }


class DigitalTwin:
    """
    Digital twin implementation for IoT network simulation and optimization.
    
    This class maintains a virtual representation of the physical IoT network
    and provides simulation capabilities for what-if analysis and optimization.
    """
    
    def __init__(
        self,
        num_devices: int = 100,
        update_interval: float = 1.0,
        simulation_speed: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.num_devices = num_devices
        self.update_interval = update_interval
        self.simulation_speed = simulation_speed
        self.device = device
        
        # Initialize network topology
        self.topology = NetworkTopology(num_devices)
        
        # Initialize device states
        self.device_states = {}
        for i in range(num_devices):
            device_type = np.random.choice(["sensor", "actuator", "gateway", "camera"])
            self.device_states[i] = DeviceState(i, device_type)
        
        # Initialize generative models
        self.traffic_generator = TrafficGenerator(device=device)
        self.network_predictor = NetworkStatePredictor(device=device)
        self.optimizer = MultiObjectiveOptimizer(
            num_devices=num_devices,
            device=device
        )
        
        # Simulation state
        self.is_running = False
        self.simulation_thread = None
        self.current_time = datetime.now()
        self.simulation_time = datetime.now()
        
        # Data storage
        self.historical_data = deque(maxlen=10000)
        self.prediction_cache = {}
        self.optimization_results = {}
        
        # Event handling
        self.event_queue = queue.Queue()
        self.event_handlers = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def start_simulation(self) -> None:
        """Start the digital twin simulation."""
        if self.is_running:
            self.logger.warning("Simulation is already running")
            return
        
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.logger.info("Digital twin simulation started")
    
    def stop_simulation(self) -> None:
        """Stop the digital twin simulation."""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        
        self.logger.info("Digital twin simulation stopped")
    
    def _simulation_loop(self) -> None:
        """Main simulation loop."""
        while self.is_running:
            start_time = time.time()
            
            # Update simulation time
            self.simulation_time += timedelta(seconds=self.update_interval * self.simulation_speed)
            
            # Update device states
            self._update_device_states()
            
            # Generate synthetic traffic
            self._generate_synthetic_traffic()
            
            # Update network predictions
            self._update_network_predictions()
            
            # Process events
            self._process_events()
            
            # Store historical data
            self._store_historical_data()
            
            # Calculate sleep time to maintain update interval
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.update_interval - elapsed_time)
            time.sleep(sleep_time)
    
    def _update_device_states(self) -> None:
        """Update device states based on current conditions."""
        for device_id, device_state in self.device_states.items():
            # Simulate device behavior
            metrics = self._simulate_device_behavior(device_id)
            device_state.update_metrics(metrics)
            
            # Check for anomalies
            if self._detect_anomaly(device_state):
                self._handle_anomaly(device_id, device_state)
    
    def _simulate_device_behavior(self, device_id: int) -> Dict[str, float]:
        """Simulate realistic device behavior."""
        device_state = self.device_states[device_id]
        
        # Simulate realistic variations
        metrics = {
            'cpu_usage': max(0, min(100, device_state.cpu_usage + np.random.normal(0, 5))),
            'memory_usage': max(0, min(100, device_state.memory_usage + np.random.normal(0, 3))),
            'battery_level': max(0, min(100, device_state.battery_level - np.random.exponential(0.1))),
            'temperature': max(-10, min(80, device_state.temperature + np.random.normal(0, 1))),
            'signal_strength': max(-100, min(0, device_state.signal_strength + np.random.normal(0, 2))),
            'packet_loss_rate': max(0, min(1, device_state.packet_loss_rate + np.random.normal(0, 0.01))),
            'latency': max(0, device_state.latency + np.random.normal(0, 2)),
            'throughput': max(0, device_state.throughput + np.random.normal(0, 5)),
            'energy_consumption': max(0, device_state.energy_consumption + np.random.normal(0, 0.1))
        }
        
        # Device type specific behavior
        if device_state.device_type == "sensor":
            metrics['throughput'] *= 0.5  # Lower throughput for sensors
            metrics['energy_consumption'] *= 0.3  # Lower energy consumption
        elif device_state.device_type == "camera":
            metrics['throughput'] *= 2.0  # Higher throughput for cameras
            metrics['energy_consumption'] *= 3.0  # Higher energy consumption
        elif device_state.device_type == "gateway":
            metrics['cpu_usage'] *= 1.5  # Higher CPU usage for gateways
            metrics['throughput'] *= 1.5  # Higher throughput
        
        return metrics
    
    def _detect_anomaly(self, device_state: DeviceState) -> bool:
        """Detect anomalies in device behavior."""
        # Simple anomaly detection based on thresholds
        anomalies = []
        
        if device_state.cpu_usage > 90:
            anomalies.append("high_cpu")
        if device_state.memory_usage > 90:
            anomalies.append("high_memory")
        if device_state.temperature > 70:
            anomalies.append("high_temperature")
        if device_state.packet_loss_rate > 0.1:
            anomalies.append("high_packet_loss")
        if device_state.latency > 1000:
            anomalies.append("high_latency")
        if device_state.battery_level < 10:
            anomalies.append("low_battery")
        
        return len(anomalies) > 0
    
    def _handle_anomaly(self, device_id: int, device_state: DeviceState) -> None:
        """Handle detected anomalies."""
        self.logger.warning(f"Anomaly detected in device {device_id}")
        
        # Mark device as unhealthy
        device_state.is_healthy = False
        
        # Trigger optimization
        self._trigger_optimization()
        
        # Send alert
        self._send_alert(device_id, "anomaly_detected")
    
    def _generate_synthetic_traffic(self) -> None:
        """Generate synthetic traffic patterns."""
        # Generate mixed traffic patterns
        traffic_patterns = self.traffic_generator.generate_mixed_traffic(
            num_samples=self.num_devices,
            pattern_weights={
                'steady': 0.6,
                'burst': 0.2,
                'cyclical': 0.15,
                'event_triggered': 0.05
            }
        )
        
        # Update device states with traffic information
        for device_id, device_state in self.device_states.items():
            if device_id < traffic_patterns.shape[0]:
                traffic_data = traffic_patterns[device_id]
                
                # Update network metrics based on traffic
                device_state.throughput = float(torch.mean(traffic_data[:, 1])) * 10
                device_state.latency = float(torch.mean(traffic_data[:, 2])) * 50
                device_state.energy_consumption = float(torch.mean(traffic_data[:, 4])) * 2
    
    def _update_network_predictions(self) -> None:
        """Update network state predictions."""
        # Prepare input data for prediction
        current_states = self._get_current_network_state()
        traffic_patterns = self._get_current_traffic_patterns()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.network_predictor.predict(
                traffic_patterns=traffic_patterns,
                current_states=current_states
            )
        
        # Cache predictions
        self.prediction_cache[self.simulation_time] = predictions
    
    def _get_current_network_state(self) -> torch.Tensor:
        """Get current network state as tensor."""
        state_data = []
        
        for device_id in range(self.num_devices):
            device_state = self.device_states[device_id]
            device_features = [
                device_state.cpu_usage / 100.0,
                device_state.memory_usage / 100.0,
                device_state.battery_level / 100.0,
                device_state.temperature / 100.0,
                device_state.signal_strength / 100.0,
                device_state.packet_loss_rate,
                device_state.latency / 1000.0,
                device_state.throughput / 100.0
            ]
            state_data.append(device_features)
        
        return torch.FloatTensor(state_data).unsqueeze(0).to(self.device)
    
    def _get_current_traffic_patterns(self) -> torch.Tensor:
        """Get current traffic patterns as tensor."""
        # Generate current traffic patterns
        traffic_patterns = self.traffic_generator.generate_mixed_traffic(
            num_samples=self.num_devices
        )
        
        return traffic_patterns.unsqueeze(0).to(self.device)
    
    def _process_events(self) -> None:
        """Process queued events."""
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
            except queue.Empty:
                break
    
    def _handle_event(self, event: Dict[str, Any]) -> None:
        """Handle a specific event."""
        event_type = event.get('type')
        
        if event_type in self.event_handlers:
            self.event_handlers[event_type](event)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def _store_historical_data(self) -> None:
        """Store current state in historical data."""
        timestamp = self.simulation_time
        
        # Collect device states
        device_data = {}
        for device_id, device_state in self.device_states.items():
            device_data[device_id] = device_state.to_dict()
        
        # Store data
        historical_entry = {
            'timestamp': timestamp.isoformat(),
            'device_states': device_data,
            'network_metrics': self._calculate_network_metrics(),
            'predictions': self.prediction_cache.get(timestamp, {})
        }
        
        self.historical_data.append(historical_entry)
    
    def _calculate_network_metrics(self) -> Dict[str, float]:
        """Calculate overall network metrics."""
        metrics = {
            'avg_cpu_usage': 0.0,
            'avg_memory_usage': 0.0,
            'avg_latency': 0.0,
            'avg_throughput': 0.0,
            'avg_energy_consumption': 0.0,
            'packet_loss_rate': 0.0,
            'device_health_ratio': 0.0
        }
        
        if not self.device_states:
            return metrics
        
        total_devices = len(self.device_states)
        healthy_devices = 0
        
        for device_state in self.device_states.values():
            metrics['avg_cpu_usage'] += device_state.cpu_usage
            metrics['avg_memory_usage'] += device_state.memory_usage
            metrics['avg_latency'] += device_state.latency
            metrics['avg_throughput'] += device_state.throughput
            metrics['avg_energy_consumption'] += device_state.energy_consumption
            metrics['packet_loss_rate'] += device_state.packet_loss_rate
            
            if device_state.is_healthy:
                healthy_devices += 1
        
        # Calculate averages
        for key in ['avg_cpu_usage', 'avg_memory_usage', 'avg_latency', 
                   'avg_throughput', 'avg_energy_consumption', 'packet_loss_rate']:
            metrics[key] /= total_devices
        
        metrics['device_health_ratio'] = healthy_devices / total_devices
        
        return metrics
    
    def _trigger_optimization(self) -> None:
        """Trigger network optimization."""
        self.logger.info("Triggering network optimization...")
        
        # Get current network state
        current_state = self._get_current_network_state()
        
        # Run optimization
        optimization_results = self.optimizer.optimize_network(
            initial_state=current_state.cpu().numpy().flatten(),
            max_steps=50
        )
        
        # Store results
        self.optimization_results[self.simulation_time] = optimization_results
        
        # Apply optimization recommendations
        self._apply_optimization_recommendations(optimization_results)
    
    def _apply_optimization_recommendations(self, results: Dict[str, Any]) -> None:
        """Apply optimization recommendations to the network."""
        # This is a simplified implementation
        # In practice, you would apply specific configuration changes
        
        self.logger.info("Applying optimization recommendations...")
        
        # Example: Adjust device configurations based on optimization results
        if 'actions' in results and len(results['actions']) > 0:
            final_action = results['actions'][-1]
            
            # Apply action to device configurations
            for device_id in range(min(len(final_action) // 3, self.num_devices)):
                action_start = device_id * 3
                if action_start + 2 < len(final_action):
                    # Apply power, routing, and resource allocation adjustments
                    power_factor = final_action[action_start]
                    routing_factor = final_action[action_start + 1]
                    resource_factor = final_action[action_start + 2]
                    
                    # Update device state based on optimization
                    device_state = self.device_states[device_id]
                    device_state.throughput *= (0.8 + 0.4 * routing_factor)
                    device_state.energy_consumption *= (0.7 + 0.6 * resource_factor)
    
    def _send_alert(self, device_id: int, alert_type: str) -> None:
        """Send alert for device issues."""
        alert = {
            'timestamp': self.simulation_time.isoformat(),
            'device_id': device_id,
            'alert_type': alert_type,
            'severity': 'warning'
        }
        
        self.logger.warning(f"Alert: {alert}")
    
    def what_if_analysis(
        self,
        scenario: Dict[str, Any],
        duration: int = 100
    ) -> Dict[str, Any]:
        """
        Perform what-if analysis for a given scenario.
        
        Args:
            scenario: Scenario configuration
            duration: Simulation duration in steps
            
        Returns:
            Analysis results
        """
        self.logger.info(f"Running what-if analysis: {scenario}")
        
        # Save current state
        original_states = {k: v.to_dict() for k, v in self.device_states.items()}
        
        # Apply scenario changes
        self._apply_scenario(scenario)
        
        # Run simulation
        results = {
            'scenario': scenario,
            'initial_metrics': self._calculate_network_metrics(),
            'simulation_steps': [],
            'final_metrics': {}
        }
        
        for step in range(duration):
            # Update device states
            self._update_device_states()
            
            # Generate traffic
            self._generate_synthetic_traffic()
            
            # Calculate metrics
            step_metrics = self._calculate_network_metrics()
            results['simulation_steps'].append({
                'step': step,
                'metrics': step_metrics
            })
        
        # Get final metrics
        results['final_metrics'] = self._calculate_network_metrics()
        
        # Restore original state
        for device_id, state_dict in original_states.items():
            device_state = self.device_states[device_id]
            for key, value in state_dict.items():
                if key not in ['device_id', 'device_type', 'timestamp']:
                    if hasattr(device_state, key):
                        setattr(device_state, key, value)
        
        return results
    
    def _apply_scenario(self, scenario: Dict[str, Any]) -> None:
        """Apply scenario changes to the network."""
        if 'device_failures' in scenario:
            for device_id in scenario['device_failures']:
                if device_id in self.device_states:
                    self.device_states[device_id].is_online = False
                    self.device_states[device_id].is_healthy = False
        
        if 'traffic_surge' in scenario:
            surge_config = scenario['traffic_surge']
            for device_id in surge_config.get('affected_devices', []):
                if device_id in self.device_states:
                    multiplier = surge_config.get('intensity', 2.0)
                    self.device_states[device_id].throughput *= multiplier
        
        if 'network_changes' in scenario:
            changes = scenario['network_changes']
            # Apply network topology changes, etc.
            pass
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status."""
        return {
            'timestamp': self.simulation_time.isoformat(),
            'num_devices': self.num_devices,
            'online_devices': sum(1 for ds in self.device_states.values() if ds.is_online),
            'healthy_devices': sum(1 for ds in self.device_states.values() if ds.is_healthy),
            'network_metrics': self._calculate_network_metrics(),
            'recent_predictions': dict(list(self.prediction_cache.items())[-5:]),
            'recent_optimizations': dict(list(self.optimization_results.items())[-5:])
        }
    
    def add_event_handler(self, event_type: str, handler: callable) -> None:
        """Add event handler for specific event type."""
        self.event_handlers[event_type] = handler
    
    def queue_event(self, event: Dict[str, Any]) -> None:
        """Queue an event for processing."""
        self.event_queue.put(event)
    
    def get_historical_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get historical data for specified time range."""
        if start_time is None:
            start_time = datetime.min
        if end_time is None:
            end_time = datetime.max
        
        filtered_data = []
        for entry in self.historical_data:
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if start_time <= entry_time <= end_time:
                filtered_data.append(entry)
        
        return filtered_data
    
    def save_state(self, filepath: str) -> None:
        """Save current digital twin state."""
        state_data = {
            'simulation_time': self.simulation_time.isoformat(),
            'device_states': {k: v.to_dict() for k, v in self.device_states.items()},
            'network_metrics': self._calculate_network_metrics(),
            'topology': {
                'adjacency_matrix': self.topology.adjacency_matrix.tolist(),
                'device_positions': self.topology.device_positions.tolist()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """Load digital twin state from file."""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        # Restore simulation time
        self.simulation_time = datetime.fromisoformat(state_data['simulation_time'])
        
        # Restore device states
        for device_id, state_dict in state_data['device_states'].items():
            device_id = int(device_id)
            device_state = DeviceState(device_id, state_dict['device_type'])
            
            for key, value in state_dict.items():
                if key not in ['device_id', 'device_type'] and hasattr(device_state, key):
                    if key in ['timestamp', 'last_heartbeat']:
                        setattr(device_state, key, datetime.fromisoformat(value))
                    else:
                        setattr(device_state, key, value)
            
            self.device_states[device_id] = device_state
