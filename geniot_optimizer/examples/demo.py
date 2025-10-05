"""
GenIoT-Optimizer Demonstration Script

This script demonstrates the complete GenIoT-Optimizer framework with examples
for different IoT use cases including smart cities, manufacturing, and smart homes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import logging
import time
from pathlib import Path
import json

# Import GenIoT-Optimizer components
from ..core.traffic_generator import TrafficGenerator, TrafficPattern, IoTDeviceType
from ..core.network_predictor import NetworkStatePredictor
from ..core.optimizer import MultiObjectiveOptimizer
from ..core.digital_twin import DigitalTwin
from ..training.pipeline import TrainingPipeline
from ..training.data_loader import IoTDataLoader
from ..training.evaluation import ModelEvaluator


class GenIoTDemo:
    """
    Demonstration class for GenIoT-Optimizer framework.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize components
        self.traffic_generator = TrafficGenerator(device=device)
        self.network_predictor = NetworkStatePredictor(device=device)
        self.optimizer = MultiObjectiveOptimizer(device=device)
        self.digital_twin = DigitalTwin(device=device)
        self.evaluator = ModelEvaluator(device=device)
        
        # Results storage
        self.demo_results = {}
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def run_complete_demo(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete demonstration of GenIoT-Optimizer framework.
        
        Args:
            save_results: Whether to save demonstration results
            
        Returns:
            Dictionary containing all demonstration results
        """
        self.logger.info("Starting GenIoT-Optimizer Complete Demonstration")
        
        # 1. Traffic Generation Demo
        self.logger.info("1. Demonstrating Traffic Generation...")
        traffic_results = self._demo_traffic_generation()
        self.demo_results['traffic_generation'] = traffic_results
        
        # 2. Network Prediction Demo
        self.logger.info("2. Demonstrating Network State Prediction...")
        prediction_results = self._demo_network_prediction()
        self.demo_results['network_prediction'] = prediction_results
        
        # 3. Multi-Objective Optimization Demo
        self.logger.info("3. Demonstrating Multi-Objective Optimization...")
        optimization_results = self._demo_optimization()
        self.demo_results['optimization'] = optimization_results
        
        # 4. Digital Twin Demo
        self.logger.info("4. Demonstrating Digital Twin...")
        digital_twin_results = self._demo_digital_twin()
        self.demo_results['digital_twin'] = digital_twin_results
        
        # 5. Use Case Demonstrations
        self.logger.info("5. Demonstrating Use Cases...")
        use_case_results = self._demo_use_cases()
        self.demo_results['use_cases'] = use_case_results
        
        # 6. Performance Evaluation
        self.logger.info("6. Demonstrating Performance Evaluation...")
        evaluation_results = self._demo_evaluation()
        self.demo_results['evaluation'] = evaluation_results
        
        # Save results if requested
        if save_results:
            self._save_demo_results()
        
        self.logger.info("GenIoT-Optimizer Demonstration Completed!")
        return self.demo_results
    
    def _demo_traffic_generation(self) -> Dict[str, Any]:
        """Demonstrate traffic generation capabilities."""
        results = {}
        
        # Generate different types of traffic patterns
        traffic_types = [
            (TrafficPattern.BURST, IoTDeviceType.SENSOR, "Burst traffic from sensors"),
            (TrafficPattern.CYCLICAL, IoTDeviceType.ACTUATOR, "Cyclical traffic from actuators"),
            (TrafficPattern.EVENT_TRIGGERED, IoTDeviceType.CAMERA, "Event-triggered traffic from cameras"),
            (TrafficPattern.STEADY, IoTDeviceType.GATEWAY, "Steady traffic from gateways"),
            (TrafficPattern.ANOMALOUS, IoTDeviceType.MOBILE, "Anomalous traffic from mobile devices")
        ]
        
        generated_traffic = {}
        
        for pattern, device_type, description in traffic_types:
            self.logger.info(f"Generating {description}...")
            
            if pattern == TrafficPattern.BURST:
                traffic = self.traffic_generator.generate_burst_traffic(
                    num_samples=100, device_type=device_type, intensity=2.0
                )
            elif pattern == TrafficPattern.CYCLICAL:
                traffic = self.traffic_generator.generate_cyclical_traffic(
                    num_samples=100, device_type=device_type, period=50, amplitude=1.5
                )
            elif pattern == TrafficPattern.EVENT_TRIGGERED:
                traffic = self.traffic_generator.generate_event_triggered_traffic(
                    num_samples=100, device_type=device_type, event_probability=0.2
                )
            elif pattern == TrafficPattern.STEADY:
                traffic = self.traffic_generator.generate_steady_traffic(
                    num_samples=100, device_type=device_type, baseline=1.0
                )
            elif pattern == TrafficPattern.ANOMALOUS:
                traffic = self.traffic_generator.generate_anomalous_traffic(
                    num_samples=100, device_type=device_type, anomaly_intensity=3.0
                )
            
            generated_traffic[description] = {
                'traffic': traffic.cpu().numpy(),
                'statistics': self.traffic_generator.get_traffic_statistics(traffic)
            }
        
        # Generate mixed traffic
        self.logger.info("Generating mixed traffic patterns...")
        mixed_traffic = self.traffic_generator.generate_mixed_traffic(
            num_samples=500,
            pattern_weights={
                'steady': 0.5,
                'burst': 0.2,
                'cyclical': 0.15,
                'event_triggered': 0.1,
                'anomalous': 0.05
            }
        )
        
        generated_traffic['Mixed Traffic'] = {
            'traffic': mixed_traffic.cpu().numpy(),
            'statistics': self.traffic_generator.get_traffic_statistics(mixed_traffic)
        }
        
        results['generated_traffic'] = generated_traffic
        results['summary'] = f"Generated {len(generated_traffic)} different traffic patterns"
        
        return results
    
    def _demo_network_prediction(self) -> Dict[str, Any]:
        """Demonstrate network state prediction capabilities."""
        results = {}
        
        # Generate sample traffic and network states
        traffic_patterns = torch.randn(10, 100, 8).to(self.device)
        current_states = torch.randn(10, 100, 8).to(self.device)
        
        # Make predictions
        self.logger.info("Making network state predictions...")
        predictions = self.network_predictor.predict(
            traffic_patterns=traffic_patterns,
            current_states=current_states,
            return_uncertainty=True
        )
        
        # Analyze predictions
        prediction_analysis = {}
        for metric, pred in predictions.items():
            if 'mean' in metric:
                metric_name = metric.replace('_mean', '')
                prediction_analysis[metric_name] = {
                    'mean_prediction': pred.cpu().numpy().mean(),
                    'std_prediction': pred.cpu().numpy().std(),
                    'min_prediction': pred.cpu().numpy().min(),
                    'max_prediction': pred.cpu().numpy().max()
                }
        
        results['predictions'] = prediction_analysis
        results['summary'] = f"Generated predictions for {len(prediction_analysis)} network metrics"
        
        return results
    
    def _demo_optimization(self) -> Dict[str, Any]:
        """Demonstrate multi-objective optimization capabilities."""
        results = {}
        
        # Run optimization
        self.logger.info("Running multi-objective optimization...")
        optimization_results = self.optimizer.optimize_network(max_steps=100)
        
        # Analyze results
        final_performance = optimization_results.get('final_performance', {})
        initial_metrics = optimization_results.get('simulation_steps', [{}])[0].get('metrics', {})
        
        improvements = {}
        for metric in final_performance.keys():
            if metric in initial_metrics:
                initial_val = initial_metrics[metric]
                final_val = final_performance[metric]
                improvement = (final_val - initial_val) / initial_val * 100
                improvements[f'{metric}_improvement'] = improvement
        
        results['optimization_results'] = optimization_results
        results['improvements'] = improvements
        results['total_reward'] = optimization_results.get('total_reward', 0)
        results['summary'] = f"Optimization completed with {len(improvements)} metric improvements"
        
        return results
    
    def _demo_digital_twin(self) -> Dict[str, Any]:
        """Demonstrate digital twin capabilities."""
        results = {}
        
        # Start digital twin simulation
        self.logger.info("Starting digital twin simulation...")
        self.digital_twin.start_simulation()
        
        # Let it run for a short time
        time.sleep(5)
        
        # Get network status
        network_status = self.digital_twin.get_network_status()
        
        # Perform what-if analysis
        self.logger.info("Performing what-if analysis...")
        scenario = {
            'device_failures': [0, 5, 10],  # Simulate device failures
            'traffic_surge': {
                'affected_devices': [1, 2, 3, 4],
                'intensity': 2.0
            }
        }
        
        what_if_results = self.digital_twin.what_if_analysis(scenario, duration=50)
        
        # Stop simulation
        self.digital_twin.stop_simulation()
        
        results['network_status'] = network_status
        results['what_if_analysis'] = what_if_results
        results['summary'] = "Digital twin simulation and what-if analysis completed"
        
        return results
    
    def _demo_use_cases(self) -> Dict[str, Any]:
        """Demonstrate different IoT use cases."""
        results = {}
        
        # Smart City Use Case
        self.logger.info("Demonstrating Smart City use case...")
        smart_city_results = self._demo_smart_city()
        results['smart_city'] = smart_city_results
        
        # Manufacturing Use Case
        self.logger.info("Demonstrating Manufacturing use case...")
        manufacturing_results = self._demo_manufacturing()
        results['manufacturing'] = manufacturing_results
        
        # Smart Home Use Case
        self.logger.info("Demonstrating Smart Home use case...")
        smart_home_results = self._demo_smart_home()
        results['smart_home'] = smart_home_results
        
        return results
    
    def _demo_smart_city(self) -> Dict[str, Any]:
        """Demonstrate smart city IoT use case."""
        # Generate traffic for smart city devices
        traffic_light_traffic = self.traffic_generator.generate_cyclical_traffic(
            num_samples=50, device_type=IoTDeviceType.ACTUATOR, period=30
        )
        
        sensor_traffic = self.traffic_generator.generate_steady_traffic(
            num_samples=100, device_type=IoTDeviceType.SENSOR
        )
        
        camera_traffic = self.traffic_generator.generate_event_triggered_traffic(
            num_samples=30, device_type=IoTDeviceType.CAMERA, event_probability=0.3
        )
        
        # Simulate optimization for traffic management
        optimization_results = self.optimizer.optimize_network(max_steps=50)
        
        return {
            'traffic_light_traffic': self.traffic_generator.get_traffic_statistics(traffic_light_traffic),
            'sensor_traffic': self.traffic_generator.get_traffic_statistics(sensor_traffic),
            'camera_traffic': self.traffic_generator.get_traffic_statistics(camera_traffic),
            'optimization_results': optimization_results,
            'description': 'Smart city traffic management optimization'
        }
    
    def _demo_manufacturing(self) -> Dict[str, Any]:
        """Demonstrate manufacturing IoT use case."""
        # Generate traffic for manufacturing devices
        production_line_traffic = self.traffic_generator.generate_cyclical_traffic(
            num_samples=80, device_type=IoTDeviceType.ACTUATOR, period=60
        )
        
        quality_sensor_traffic = self.traffic_generator.generate_steady_traffic(
            num_samples=120, device_type=IoTDeviceType.SENSOR
        )
        
        # Simulate anomaly detection
        anomalous_traffic = self.traffic_generator.generate_anomalous_traffic(
            num_samples=20, device_type=IoTDeviceType.SENSOR, anomaly_intensity=2.5
        )
        
        # Simulate optimization for production efficiency
        optimization_results = self.optimizer.optimize_network(max_steps=50)
        
        return {
            'production_line_traffic': self.traffic_generator.get_traffic_statistics(production_line_traffic),
            'quality_sensor_traffic': self.traffic_generator.get_traffic_statistics(quality_sensor_traffic),
            'anomalous_traffic': self.traffic_generator.get_traffic_statistics(anomalous_traffic),
            'optimization_results': optimization_results,
            'description': 'Manufacturing production line optimization'
        }
    
    def _demo_smart_home(self) -> Dict[str, Any]:
        """Demonstrate smart home IoT use case."""
        # Generate traffic for smart home devices
        thermostat_traffic = self.traffic_generator.generate_cyclical_traffic(
            num_samples=40, device_type=IoTDeviceType.ACTUATOR, period=120
        )
        
        security_sensor_traffic = self.traffic_generator.generate_steady_traffic(
            num_samples=60, device_type=IoTDeviceType.SENSOR
        )
        
        smart_speaker_traffic = self.traffic_generator.generate_event_triggered_traffic(
            num_samples=25, device_type=IoTDeviceType.GATEWAY, event_probability=0.1
        )
        
        # Simulate optimization for energy efficiency
        optimization_results = self.optimizer.optimize_network(max_steps=50)
        
        return {
            'thermostat_traffic': self.traffic_generator.get_traffic_statistics(thermostat_traffic),
            'security_sensor_traffic': self.traffic_generator.get_traffic_statistics(security_sensor_traffic),
            'smart_speaker_traffic': self.traffic_generator.get_traffic_statistics(smart_speaker_traffic),
            'optimization_results': optimization_results,
            'description': 'Smart home energy efficiency optimization'
        }
    
    def _demo_evaluation(self) -> Dict[str, Any]:
        """Demonstrate performance evaluation capabilities."""
        results = {}
        
        # Generate test data
        test_traffic = self.traffic_generator.generate_mixed_traffic(num_samples=200)
        
        # Evaluate traffic generation quality
        self.logger.info("Evaluating traffic generation quality...")
        traffic_eval = self.evaluator.evaluate_generative_models(
            self.traffic_generator, 
            torch.utils.data.DataLoader([test_traffic], batch_size=32)
        )
        results['traffic_evaluation'] = traffic_eval
        
        # Evaluate network predictor
        self.logger.info("Evaluating network predictor...")
        # Create mock test data for predictor
        mock_test_data = {
            'traffic_patterns': torch.randn(32, 100, 8),
            'current_states': torch.randn(32, 100, 8),
            'targets': {
                'latency': torch.randn(32, 10),
                'throughput': torch.randn(32, 10),
                'energy': torch.randn(32, 10),
                'qos': torch.randn(32, 10),
                'congestion': torch.randn(32, 10)
            }
        }
        
        predictor_eval = self.evaluator.evaluate_network_predictor(
            self.network_predictor,
            torch.utils.data.DataLoader([mock_test_data], batch_size=32)
        )
        results['predictor_evaluation'] = predictor_eval
        
        # Evaluate optimizer
        self.logger.info("Evaluating optimizer...")
        optimizer_eval = self.evaluator.evaluate_optimizer(
            self.optimizer,
            torch.utils.data.DataLoader([], batch_size=32)
        )
        results['optimizer_evaluation'] = optimizer_eval
        
        return results
    
    def _save_demo_results(self) -> None:
        """Save demonstration results to file."""
        results_file = Path("demo_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(self.demo_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Demo results saved to {results_file}")
    
    def generate_demo_report(self) -> str:
        """Generate a comprehensive demonstration report."""
        report = []
        report.append("GenIoT-Optimizer Demonstration Report")
        report.append("=" * 50)
        report.append("")
        
        # Traffic Generation Results
        if 'traffic_generation' in self.demo_results:
            report.append("1. Traffic Generation Results:")
            report.append("-" * 30)
            traffic_results = self.demo_results['traffic_generation']
            report.append(f"   {traffic_results.get('summary', 'No summary available')}")
            report.append("")
        
        # Network Prediction Results
        if 'network_prediction' in self.demo_results:
            report.append("2. Network Prediction Results:")
            report.append("-" * 30)
            pred_results = self.demo_results['network_prediction']
            report.append(f"   {pred_results.get('summary', 'No summary available')}")
            report.append("")
        
        # Optimization Results
        if 'optimization' in self.demo_results:
            report.append("3. Optimization Results:")
            report.append("-" * 30)
            opt_results = self.demo_results['optimization']
            report.append(f"   Total Reward: {opt_results.get('total_reward', 0):.4f}")
            report.append(f"   {opt_results.get('summary', 'No summary available')}")
            report.append("")
        
        # Digital Twin Results
        if 'digital_twin' in self.demo_results:
            report.append("4. Digital Twin Results:")
            report.append("-" * 30)
            dt_results = self.demo_results['digital_twin']
            report.append(f"   {dt_results.get('summary', 'No summary available')}")
            report.append("")
        
        # Use Case Results
        if 'use_cases' in self.demo_results:
            report.append("5. Use Case Results:")
            report.append("-" * 30)
            use_case_results = self.demo_results['use_cases']
            for use_case, results in use_case_results.items():
                report.append(f"   {use_case.replace('_', ' ').title()}: {results.get('description', 'No description')}")
            report.append("")
        
        # Evaluation Results
        if 'evaluation' in self.demo_results:
            report.append("6. Evaluation Results:")
            report.append("-" * 30)
            eval_results = self.demo_results['evaluation']
            report.append("   Performance evaluation completed for all components")
            report.append("")
        
        report.append("Demonstration completed successfully!")
        
        return "\n".join(report)


def main():
    """Main function to run the demonstration."""
    print("GenIoT-Optimizer Framework Demonstration")
    print("=" * 50)
    
    # Initialize demo
    demo = GenIoTDemo()
    
    # Run complete demonstration
    results = demo.run_complete_demo(save_results=True)
    
    # Generate and print report
    report = demo.generate_demo_report()
    print("\n" + report)
    
    print("\nDemonstration completed! Check 'demo_results.json' for detailed results.")


if __name__ == "__main__":
    main()
