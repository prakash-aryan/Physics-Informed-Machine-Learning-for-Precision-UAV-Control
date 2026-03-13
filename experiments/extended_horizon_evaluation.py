"""
Extended Horizon Evaluation Script
Tests trained PIATSG models on 30s, 60s, and 120s episodes to analyze long-term stability
Addresses Reviewer 2, Comment 3
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Tuple
import time
from datetime import datetime
from simulation.environment import (
    initialize_simulation, reset_simulation, get_observation,
    step_simulation, apply_action, compute_reward, check_done
)
from core.agent import PIATSGAgent
from utils.config import configure_device, TrainingConfig


class ExtendedHorizonEvaluator:
    """Evaluator for long-horizon episode testing"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "extended_eval_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        self.dt = 0.01  # 100Hz control frequency
        self.current_episode_length = 0
        
        # Test horizons in seconds
        self.test_horizons = [5, 30, 60, 120]
        
        # Results storage
        self.results = {horizon: [] for horizon in self.test_horizons}
        
        print(f"Extended Horizon Evaluator initialized")
        print(f"Device: {self.device}")
        print(f"Model: {model_path}")
        print(f"Test horizons: {self.test_horizons}s")
    
    def load_model(self) -> PIATSGAgent:
        """Load trained model"""
        print(f"\nLoading model from {self.model_path}...")
        
        # Initialize config using the configure_device function
        device, batch_size, buffer_size, _ = configure_device()
        config = TrainingConfig(device, batch_size, buffer_size)
        
        # Initialize agent with config
        agent = PIATSGAgent(config)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Check if it's just the actor state dict or full checkpoint
        if 'actor' in checkpoint:
            # Full agent checkpoint - load just the actor
            agent.actor.load_state_dict(checkpoint['actor'])
        else:
            # Just actor state dict
            agent.actor.load_state_dict(checkpoint)
        
        agent.actor.eval()
        agent.actor.to(self.device)
        
        print("Model loaded successfully")
        return agent
    
    def create_env(self, episode_length_seconds: float):
        """Initialize MuJoCo environment (functional API)"""
        # Initialize simulation if not already done
        try:
            initialize_simulation()
        except:
            pass  # Already initialized
        
        # Store episode length for tracking
        self.current_episode_length = episode_length_seconds
        self.dt = 0.01  # 100Hz control frequency
        return episode_length_seconds  # Return duration for reference
    
    def evaluate_episode(
        self,
        agent: PIATSGAgent,
        episode_length_seconds: float
    ) -> Dict:
        """Run single evaluation episode and collect detailed metrics"""
        
        # Reset simulation
        reset_simulation(randomize=False)
        state = get_observation()
        
        # Metrics storage
        episode_data = {
            'positions': [],
            'velocities': [],
            'references': [],
            'actions': [],
            'rewards': [],
            'timestamps': [],
            'constraint_violations': [],
            'physics_errors': []
        }
        
        step = 0
        cumulative_reward = 0
        max_steps = int(episode_length_seconds / self.dt)
        early_termination = False
        termination_reason = None
        
        for step in range(max_steps):
            # Get action from agent
            with torch.no_grad():
                action = agent.select_action(state, deterministic=True)
            
            # Apply action and step simulation
            apply_action(action, state)
            step_simulation()
            
            # Get next state
            next_state = get_observation()
            reward = compute_reward(next_state)
            done = check_done(next_state)
            
            # Store data (state has 18 dimensions based on get_observation)
            episode_data['positions'].append(state[:3].copy())
            episode_data['velocities'].append(state[7:10].copy())
            episode_data['references'].append(np.array([0.0, 0.0, state[17]]))  # target position
            episode_data['actions'].append(action.copy())
            episode_data['rewards'].append(reward)
            episode_data['timestamps'].append(step * self.dt)
            
            # Constraint violations (check altitude and position bounds)
            pos = next_state[:3]
            constraint_violated = (pos[2] < 0.2 or pos[2] > 2.5 or 
                                 abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5)
            episode_data['constraint_violations'].append(1 if constraint_violated else 0)
            
            # Physics error placeholder (can be computed from dynamics if needed)
            episode_data['physics_errors'].append(0.0)
            
            cumulative_reward += reward
            state = next_state
            
            # Early termination on done
            if done:
                early_termination = True
                # Determine termination reason
                if pos[2] < 0.2:
                    termination_reason = f"Low altitude: {pos[2]:.3f}m"
                elif pos[2] > 2.5:
                    termination_reason = f"High altitude: {pos[2]:.3f}m"
                elif abs(pos[0]) > 1.5:
                    termination_reason = f"X position out of bounds: {pos[0]:.3f}m"
                elif abs(pos[1]) > 1.5:
                    termination_reason = f"Y position out of bounds: {pos[1]:.3f}m"
                else:
                    termination_reason = "Unknown"
                break
        
        # Convert to numpy arrays
        for key in ['positions', 'velocities', 'references', 'actions', 
                    'rewards', 'timestamps', 'constraint_violations', 'physics_errors']:
            episode_data[key] = np.array(episode_data[key])
        
        # Compute metrics
        metrics = self.compute_metrics(episode_data, episode_length_seconds)
        metrics['cumulative_reward'] = cumulative_reward
        metrics['episode_length_achieved'] = (step + 1) * self.dt
        metrics['early_termination'] = early_termination
        metrics['termination_reason'] = termination_reason
        metrics['completion_percentage'] = (metrics['episode_length_achieved'] / episode_length_seconds) * 100
        
        # Store trajectory for visualization if early termination
        if early_termination:
            metrics['trajectory_data'] = episode_data
        
        return metrics
    
    def compute_metrics(self, episode_data: Dict, episode_length: float) -> Dict:
        """Compute comprehensive metrics from episode data"""
        
        positions = episode_data['positions']
        references = episode_data['references']
        velocities = episode_data['velocities']
        actions = episode_data['actions']
        timestamps = episode_data['timestamps']
        
        # Position tracking metrics
        position_errors = np.linalg.norm(positions - references, axis=1)
        
        metrics = {
            # Overall performance
            'mean_position_rmse': np.sqrt(np.mean(position_errors ** 2)),
            'max_position_error': np.max(position_errors),
            'final_position_error': position_errors[-1],
            
            # Time-windowed RMSE (analyze drift over time)
            'rmse_0_to_5s': self._rmse_in_window(position_errors, timestamps, 0, 5),
            'rmse_5_to_15s': self._rmse_in_window(position_errors, timestamps, 5, 15) if episode_length >= 15 else None,
            'rmse_15_to_30s': self._rmse_in_window(position_errors, timestamps, 15, 30) if episode_length >= 30 else None,
            'rmse_30_to_60s': self._rmse_in_window(position_errors, timestamps, 30, 60) if episode_length >= 60 else None,
            'rmse_60_to_120s': self._rmse_in_window(position_errors, timestamps, 60, 120) if episode_length >= 120 else None,
            
            # Cumulative drift analysis
            'error_growth_rate': self._compute_error_growth_rate(position_errors, timestamps),
            
            # Control effort
            'mean_control_effort': np.mean(np.abs(actions)),
            'max_control_effort': np.max(np.abs(actions)),
            'control_variance': np.var(actions, axis=0).mean(),
            
            # Velocity tracking
            'mean_velocity': np.mean(np.linalg.norm(velocities, axis=1)),
            'max_velocity': np.max(np.linalg.norm(velocities, axis=1)),
            
            # Safety metrics
            'constraint_violations': np.sum(episode_data['constraint_violations'] > 0),
            'violation_rate': np.mean(episode_data['constraint_violations'] > 0),
            
            # Physics consistency
            'mean_physics_error': np.mean(episode_data['physics_errors']),
            'max_physics_error': np.max(episode_data['physics_errors']),
        }
        
        return metrics
    
    def _rmse_in_window(
        self,
        errors: np.ndarray,
        timestamps: np.ndarray,
        start_time: float,
        end_time: float
    ) -> float:
        """Compute RMSE in a specific time window"""
        mask = (timestamps >= start_time) & (timestamps < end_time)
        if np.sum(mask) == 0:
            return None
        windowed_errors = errors[mask]
        return np.sqrt(np.mean(windowed_errors ** 2))
    
    def _compute_error_growth_rate(
        self,
        errors: np.ndarray,
        timestamps: np.ndarray
    ) -> float:
        """Compute error growth rate using linear fit"""
        if len(errors) < 10:
            return 0.0
        
        # Fit linear model: error = a * time + b
        coeffs = np.polyfit(timestamps, errors, 1)
        growth_rate = coeffs[0]  # slope
        return growth_rate
    
    def run_evaluation(
        self,
        num_episodes_per_horizon: int = 10
    ) -> Dict:
        """Run complete evaluation across all horizons"""
        
        print("\n" + "="*80)
        print("EXTENDED HORIZON EVALUATION")
        print("="*80)
        
        # Initialize simulation once
        initialize_simulation()
        
        # Load model once
        agent = self.load_model()
        
        # Evaluate each horizon
        for horizon in self.test_horizons:
            print(f"\n{'='*80}")
            print(f"Testing {horizon}s episodes ({num_episodes_per_horizon} episodes)")
            print(f"{'='*80}")
            
            _ = self.create_env(horizon)
            
            for episode_idx in range(num_episodes_per_horizon):
                print(f"\nEpisode {episode_idx + 1}/{num_episodes_per_horizon}...", end=" ")
                
                start_time = time.time()
                metrics = self.evaluate_episode(agent, horizon)
                eval_time = time.time() - start_time
                
                self.results[horizon].append(metrics)
                
                # Print completion info
                completion_pct = metrics.get('completion_percentage', 100)
                print(f"Done! (RMSE: {metrics['mean_position_rmse']:.4f}m, "
                      f"Time: {eval_time:.2f}s, Completion: {completion_pct:.1f}%)")
                
                # Warn about early termination
                if metrics.get('early_termination', False):
                    print(f"    ⚠️  EARLY TERMINATION at {metrics['episode_length_achieved']:.2f}s: {metrics['termination_reason']}")
                    
                    # Save trajectory visualization for first failed episode of each horizon
                    if episode_idx == 0 and 'trajectory_data' in metrics:
                        self._plot_failed_trajectory(
                            metrics['trajectory_data'], 
                            horizon, 
                            metrics['termination_reason']
                        )
        
        # Aggregate results
        aggregated_results = self.aggregate_results()
        
        # Save results
        self.save_results(aggregated_results)
        
        # Generate plots
        self.generate_plots(aggregated_results)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*80 + "\n")
        
        return aggregated_results
    
    def aggregate_results(self) -> Dict:
        """Aggregate results across episodes for each horizon"""
        
        aggregated = {}
        
        for horizon, episodes in self.results.items():
            if not episodes:
                continue
            
            horizon_stats = {}
            
            # Get all metric keys from first episode
            metric_keys = episodes[0].keys()
            
            for metric in metric_keys:
                # Skip non-numeric fields
                if metric in ['trajectory_data', 'termination_reason', 'early_termination']:
                    continue
                    
                values = [ep[metric] for ep in episodes if ep[metric] is not None and isinstance(ep[metric], (int, float, np.number))]
                
                if not values:
                    continue
                
                horizon_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'values': values  # Store individual values for plotting
                }
            
            aggregated[f'{horizon}s'] = horizon_stats
        
        return aggregated
    
    def save_results(self, aggregated_results: Dict):
        """Save results to JSON file"""
        
        # Create serializable version (remove raw values)
        serializable_results = {}
        for horizon, stats in aggregated_results.items():
            serializable_results[horizon] = {}
            for metric, values in stats.items():
                serializable_results[horizon][metric] = {
                    'mean': float(values['mean']),
                    'std': float(values['std']),
                    'min': float(values['min']),
                    'max': float(values['max']),
                    'median': float(values['median'])
                }
        
        # Save to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Also save summary table
        self.save_summary_table(aggregated_results)
    
    def save_summary_table(self, aggregated_results: Dict):
        """Save summary table in readable format"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_file = self.output_dir / f"summary_table_{timestamp}.txt"
        
        with open(table_file, 'w') as f:
            f.write("Extended Horizon Evaluation Results\n")
            f.write("=" * 100 + "\n\n")
            
            # Main metrics table
            f.write("Position Tracking Performance (RMSE in meters)\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Horizon':<12} {'Mean±Std':<20} {'Min':<12} {'Max':<12} {'Median':<12}\n")
            f.write("-" * 100 + "\n")
            
            for horizon in ['5s', '30s', '60s', '120s']:
                if horizon not in aggregated_results:
                    continue
                stats = aggregated_results[horizon]['mean_position_rmse']
                f.write(f"{horizon:<12} {stats['mean']:.4f}±{stats['std']:.4f}      "
                       f"{stats['min']:.4f}      {stats['max']:.4f}      "
                       f"{stats['median']:.4f}\n")
            
            f.write("\n\n")
            
            # Error growth rate
            f.write("Error Growth Rate (m/s)\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Horizon':<12} {'Mean±Std':<20} {'Min':<12} {'Max':<12}\n")
            f.write("-" * 100 + "\n")
            
            for horizon in ['5s', '30s', '60s', '120s']:
                if horizon not in aggregated_results:
                    continue
                stats = aggregated_results[horizon]['error_growth_rate']
                f.write(f"{horizon:<12} {stats['mean']:.6f}±{stats['std']:.6f}  "
                       f"{stats['min']:.6f}  {stats['max']:.6f}\n")
            
            f.write("\n\n")
            
            # Control effort
            f.write("Control Effort\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Horizon':<12} {'Mean Effort':<20} {'Max Effort':<20} {'Variance':<20}\n")
            f.write("-" * 100 + "\n")
            
            for horizon in ['5s', '30s', '60s', '120s']:
                if horizon not in aggregated_results:
                    continue
                mean_effort = aggregated_results[horizon]['mean_control_effort']['mean']
                max_effort = aggregated_results[horizon]['max_control_effort']['mean']
                variance = aggregated_results[horizon]['control_variance']['mean']
                f.write(f"{horizon:<12} {mean_effort:.4f}             "
                       f"{max_effort:.4f}             {variance:.6f}\n")
            
            f.write("\n\n")
            
            # Safety metrics
            f.write("Safety Metrics\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Horizon':<12} {'Violation Rate (%)':<25} {'Total Violations':<20}\n")
            f.write("-" * 100 + "\n")
            
            for horizon in ['5s', '30s', '60s', '120s']:
                if horizon not in aggregated_results:
                    continue
                rate = aggregated_results[horizon]['violation_rate']['mean'] * 100
                violations = aggregated_results[horizon]['constraint_violations']['mean']
                f.write(f"{horizon:<12} {rate:.2f}                    {violations:.2f}\n")
        
        print(f"Summary table saved to: {table_file}")
    
    def generate_plots(self, aggregated_results: Dict):
        """Generate visualization plots"""
        
        print("\nGenerating plots...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 150
        })
        
        # 1. RMSE vs Horizon
        self._plot_rmse_vs_horizon(aggregated_results, timestamp)
        
        # 2. Time-windowed RMSE analysis
        self._plot_windowed_rmse(aggregated_results, timestamp)
        
        # 3. Error growth rate
        self._plot_error_growth(aggregated_results, timestamp)
        
        # 4. Control effort comparison
        self._plot_control_effort(aggregated_results, timestamp)
        
        # 5. Box plots for key metrics
        self._plot_metric_distributions(aggregated_results, timestamp)
        
        print("Plots generated successfully")
    
    def _plot_rmse_vs_horizon(self, results: Dict, timestamp: str):
        """Plot RMSE vs episode horizon"""
        
        horizons = []
        means = []
        stds = []
        
        for horizon in ['5s', '30s', '60s', '120s']:
            if horizon in results:
                horizons.append(int(horizon[:-1]))
                means.append(results[horizon]['mean_position_rmse']['mean'])
                stds.append(results[horizon]['mean_position_rmse']['std'])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.errorbar(horizons, means, yerr=stds, marker='o', capsize=5,
                   linewidth=2, markersize=8, label='PIATSG')
        
        ax.set_xlabel('Episode Horizon (seconds)')
        ax.set_ylabel('Position RMSE (m)')
        ax.set_title('Long-Term Trajectory Tracking Performance')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'rmse_vs_horizon_{timestamp}.png', dpi=300)
        plt.close()
    
    def _plot_windowed_rmse(self, results: Dict, timestamp: str):
        """Plot RMSE in different time windows"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        windows = {
            '0-5s': 'rmse_0_to_5s',
            '5-15s': 'rmse_5_to_15s',
            '15-30s': 'rmse_15_to_30s',
            '30-60s': 'rmse_30_to_60s',
            '60-120s': 'rmse_60_to_120s'
        }
        
        horizons_to_plot = ['30s', '60s', '120s']
        x_positions = np.arange(len(windows))
        bar_width = 0.25
        
        for idx, horizon in enumerate(horizons_to_plot):
            if horizon not in results:
                continue
            
            rmse_values = []
            for window_key in windows.values():
                if window_key in results[horizon] and results[horizon][window_key]['mean'] is not None:
                    rmse_values.append(results[horizon][window_key]['mean'])
                else:
                    rmse_values.append(0)
            
            # Only plot non-zero values
            valid_x = [x for x, v in zip(x_positions, rmse_values) if v > 0]
            valid_values = [v for v in rmse_values if v > 0]
            
            if valid_values:
                offset = (idx - 1) * bar_width
                ax.bar([x + offset for x in valid_x], valid_values, bar_width,
                      label=f'{horizon} episodes')
        
        ax.set_xlabel('Time Window')
        ax.set_ylabel('RMSE (m)')
        ax.set_title('Time-Windowed RMSE Analysis (Drift Detection)')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(windows.keys(), rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'windowed_rmse_{timestamp}.png', dpi=300)
        plt.close()
    
    def _plot_error_growth(self, results: Dict, timestamp: str):
        """Plot error growth rate"""
        
        horizons = []
        growth_rates = []
        errors = []
        
        for horizon in ['5s', '30s', '60s', '120s']:
            if horizon in results:
                horizons.append(horizon)
                growth_rates.append(results[horizon]['error_growth_rate']['mean'])
                errors.append(results[horizon]['error_growth_rate']['std'])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x_pos = np.arange(len(horizons))
        bars = ax.bar(x_pos, growth_rates, yerr=errors, capsize=5,
                     color='steelblue', alpha=0.8, edgecolor='black')
        
        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Episode Horizon')
        ax.set_ylabel('Error Growth Rate (m/s)')
        ax.set_title('Cumulative Error Growth Rate Analysis')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(horizons)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Color bars based on positive/negative growth
        for bar, rate in zip(bars, growth_rates):
            if rate > 0:
                bar.set_color('salmon')
            else:
                bar.set_color('lightgreen')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'error_growth_{timestamp}.png', dpi=300)
        plt.close()
    
    def _plot_control_effort(self, results: Dict, timestamp: str):
        """Plot control effort comparison"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        horizons = []
        mean_efforts = []
        max_efforts = []
        
        for horizon in ['5s', '30s', '60s', '120s']:
            if horizon in results:
                horizons.append(horizon)
                mean_efforts.append(results[horizon]['mean_control_effort']['mean'])
                max_efforts.append(results[horizon]['max_control_effort']['mean'])
        
        x_pos = np.arange(len(horizons))
        
        # Mean control effort
        ax1.bar(x_pos, mean_efforts, color='steelblue', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Episode Horizon')
        ax1.set_ylabel('Mean Control Effort')
        ax1.set_title('Average Control Effort')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(horizons)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Max control effort
        ax2.bar(x_pos, max_efforts, color='coral', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Episode Horizon')
        ax2.set_ylabel('Max Control Effort')
        ax2.set_title('Peak Control Effort')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(horizons)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'control_effort_{timestamp}.png', dpi=300)
        plt.close()
    
    def _plot_metric_distributions(self, results: Dict, timestamp: str):
        """Plot box plots for metric distributions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics_to_plot = [
            ('mean_position_rmse', 'Position RMSE (m)', axes[0, 0]),
            ('constraint_violations', 'Constraint Violations', axes[0, 1]),
            ('control_variance', 'Control Variance', axes[1, 0]),
            ('violation_rate', 'Violation Rate', axes[1, 1])
        ]
        
        for metric_key, label, ax in metrics_to_plot:
            data_to_plot = []
            labels = []
            
            for horizon in ['5s', '30s', '60s', '120s']:
                if horizon in results and metric_key in results[horizon]:
                    values = results[horizon][metric_key]['values']
                    data_to_plot.append(values)
                    labels.append(horizon)
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # Color boxes
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                ax.set_xlabel('Episode Horizon')
                ax.set_ylabel(label)
                ax.set_title(f'{label} Distribution')
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'metric_distributions_{timestamp}.png', dpi=300)
        plt.close()
    
    def _plot_failed_trajectory(self, trajectory_data: Dict, horizon: int, reason: str):
        """Plot detailed trajectory for failed episodes"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 150
        })
        
        positions = trajectory_data['positions']
        velocities = trajectory_data['velocities']
        references = trajectory_data['references']
        actions = trajectory_data['actions']
        timestamps = trajectory_data['timestamps']
        rewards = trajectory_data['rewards']
        
        fig = plt.figure(figsize=(16, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Actual')
        ax1.plot(references[:, 0], references[:, 1], references[:, 2], 'r--', linewidth=2, label='Reference')
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                   color='red', s=100, marker='X', label='Crash point')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'{horizon}s Episode - 3D Trajectory\nReason: {reason}')
        ax1.legend()
        
        # Position over time (X, Y, Z)
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(timestamps, positions[:, 0], label='X', linewidth=2)
        ax2.plot(timestamps, positions[:, 1], label='Y', linewidth=2)
        ax2.plot(timestamps, positions[:, 2], label='Z', linewidth=2)
        ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Min Z')
        ax2.axhline(y=2.5, color='red', linestyle='--', alpha=0.5, label='Max Z')
        ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='XY bounds')
        ax2.axhline(y=-1.5, color='orange', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Velocity over time
        ax3 = fig.add_subplot(2, 3, 3)
        vel_magnitude = np.linalg.norm(velocities, axis=1)
        ax3.plot(timestamps, vel_magnitude, 'g-', linewidth=2, label='Speed')
        ax3.plot(timestamps, velocities[:, 0], alpha=0.5, label='Vx')
        ax3.plot(timestamps, velocities[:, 1], alpha=0.5, label='Vy')
        ax3.plot(timestamps, velocities[:, 2], alpha=0.5, label='Vz')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Velocity vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Control inputs
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(timestamps, actions[:, 0], label='Thrust', linewidth=2)
        ax4.plot(timestamps, actions[:, 1], label='Roll', linewidth=1.5)
        ax4.plot(timestamps, actions[:, 2], label='Pitch', linewidth=1.5)
        ax4.plot(timestamps, actions[:, 3], label='Yaw', linewidth=1.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Control Input')
        ax4.set_title('Control Actions vs Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Position error magnitude
        ax5 = fig.add_subplot(2, 3, 5)
        position_error = np.linalg.norm(positions - references, axis=1)
        ax5.plot(timestamps, position_error, 'r-', linewidth=2)
        ax5.fill_between(timestamps, 0, position_error, alpha=0.3)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Position Error (m)')
        ax5.set_title('Tracking Error vs Time')
        ax5.grid(True, alpha=0.3)
        
        # Reward over time
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(timestamps, rewards, 'purple', linewidth=2)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Reward')
        ax6.set_title('Reward vs Time')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'failed_trajectory_{horizon}s_{timestamp}.png', dpi=300)
        plt.close()
        
        print(f"    📊 Trajectory visualization saved: failed_trajectory_{horizon}s_{timestamp}.png")


def main():
    """Main execution function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Extended Horizon Evaluation")
    parser.add_argument('--model', type=str, default='models/best_precision_2cm.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='extended_eval_results',
                       help='Output directory')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Episodes per horizon')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ExtendedHorizonEvaluator(
        model_path=args.model,
        output_dir=args.output
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(
        num_episodes_per_horizon=args.episodes
    )
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    # Print key insights
    print("\n1. Position Tracking Performance:")
    for horizon in ['5s', '30s', '60s', '120s']:
        if horizon in results:
            rmse = results[horizon]['mean_position_rmse']['mean']
            print(f"   {horizon}: {rmse:.4f}m RMSE")
    
    print("\n2. Error Growth Rates:")
    for horizon in ['5s', '30s', '60s', '120s']:
        if horizon in results:
            growth = results[horizon]['error_growth_rate']['mean']
            print(f"   {horizon}: {growth:.6f}m/s")
    
    print("\n3. Safety Performance:")
    for horizon in ['5s', '30s', '60s', '120s']:
        if horizon in results:
            rate = results[horizon]['violation_rate']['mean'] * 100
            print(f"   {horizon}: {rate:.2f}% violation rate")
    
    print("\n4. Episode Completion Analysis:")
    for horizon_sec in [5, 30, 60, 120]:
        horizon_key = f"{horizon_sec}s"
        if horizon_key in results:
            # Check completion from raw episode data
            episodes = evaluator.results[horizon_sec]
            early_terms = sum(1 for ep in episodes if ep.get('early_termination', False))
            avg_completion = np.mean([ep.get('completion_percentage', 100) for ep in episodes])
            print(f"   {horizon_key}: {avg_completion:.1f}% average completion, {early_terms}/{len(episodes)} early terminations")
            
            # Show common failure reasons
            if early_terms > 0:
                reasons = [ep.get('termination_reason', 'Unknown') for ep in episodes if ep.get('early_termination', False)]
                from collections import Counter
                reason_counts = Counter(reasons)
                print(f"        Common failures: {dict(reason_counts)}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()