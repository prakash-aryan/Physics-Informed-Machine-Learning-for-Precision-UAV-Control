"""
Spatiotemporal Episode Profiler
Generates episode visualizations with maneuver phase identification
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import json
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.environment import initialize_simulation, reset_simulation, get_observation, apply_action, step_simulation, compute_reward, check_done
from core.agent import PIATSGAgent
from utils.config import configure_device, TrainingConfig

mpl.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
})


class EpisodeProfiler:
    def __init__(self):
        self.states = []
        self.actions = []
        self.timestamps = []
        self.rewards = []
        self.target_pos = np.array([0.0, 0.0, 1.0])
        
    def reset(self):
        self.states = []
        self.actions = []
        self.timestamps = []
        self.rewards = []
        
    def log_step(self, state, action, reward, timestamp):
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
    
    def get_trajectory_components(self):
        """Extract trajectory components from logged states"""
        states = np.array(self.states)
        return {
            'position': states[:, 0:3],
            'quaternion': states[:, 3:7],
            'velocity': states[:, 7:10],
            'angular_velocity': states[:, 10:13],
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'timestamps': np.array(self.timestamps)
        }
    
    def extract_maneuvers(self, velocity_threshold=0.1):
        """Segment trajectory into maneuver phases (hover vs motion)"""
        traj = self.get_trajectory_components()
        velocities = np.linalg.norm(traj['velocity'], axis=1)
        
        phases = []
        if len(velocities) == 0:
            return phases
            
        current_phase = 'hover' if velocities[0] < velocity_threshold else 'motion'
        phase_start = 0
        
        for i, v in enumerate(velocities):
            new_phase = 'hover' if v < velocity_threshold else 'motion'
            if new_phase != current_phase:
                phases.append({
                    'type': current_phase,
                    'start': phase_start,
                    'end': i,
                    'duration': self.timestamps[i] - self.timestamps[phase_start]
                })
                current_phase = new_phase
                phase_start = i
        
        # Add final phase
        if len(self.timestamps) > 0:
            phases.append({
                'type': current_phase,
                'start': phase_start,
                'end': len(velocities),
                'duration': self.timestamps[-1] - self.timestamps[phase_start]
            })
        
        return phases
    
    def compute_error_distribution(self):
        """Compute spatial error distribution"""
        traj = self.get_trajectory_components()
        positions = traj['position']
        errors = np.linalg.norm(positions - self.target_pos, axis=1)
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'errors': errors,
            'position_x_error': positions[:, 0] - self.target_pos[0],
            'position_y_error': positions[:, 1] - self.target_pos[1],
            'position_z_error': positions[:, 2] - self.target_pos[2]
        }
    
    def generate_profile_plot(self, save_path, episode_num):
        """Generate complete episode profile visualization"""
        traj = self.get_trajectory_components()
        maneuvers = self.extract_maneuvers()
        errors = self.compute_error_distribution()
        
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)
        
        # 1. 3D trajectory
        ax1 = fig.add_subplot(gs[0:2, 0], projection='3d')
        positions = traj['position']
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1, alpha=0.7, label='Trajectory')
        ax1.scatter(*self.target_pos, color='red', s=100, marker='*', label='Target')
        ax1.scatter(*positions[0], color='green', s=50, marker='o', label='Start')
        ax1.scatter(*positions[-1], color='orange', s=50, marker='s', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'Episode {episode_num}: 3D Trajectory')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Position over time
        ax2 = fig.add_subplot(gs[0, 1])
        for i, label in enumerate(['X', 'Y', 'Z']):
            ax2.plot(traj['timestamps'], positions[:, i], label=label, linewidth=1.5)
        ax2.axhline(y=self.target_pos[2], color='r', linestyle='--', alpha=0.5, linewidth=1, label='Target Z')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position vs Time')
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        self._add_maneuver_phases(ax2, maneuvers)
        
        # 3. Velocity magnitude
        ax3 = fig.add_subplot(gs[1, 1])
        vel_mag = np.linalg.norm(traj['velocity'], axis=1)
        ax3.plot(traj['timestamps'], vel_mag, 'b-', linewidth=1.5)
        ax3.axhline(y=0.1, color='g', linestyle='--', alpha=0.5, linewidth=1, label='Hover threshold')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Velocity Magnitude vs Time')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        self._add_maneuver_phases(ax3, maneuvers)
        
        # 4. Control inputs
        ax4 = fig.add_subplot(gs[2, :])
        actions = traj['actions']
        action_labels = ['Thrust', 'τ_x', 'τ_y', 'τ_z']
        for i in range(min(4, actions.shape[1])):
            ax4.plot(traj['timestamps'], actions[:, i], label=action_labels[i], linewidth=1, alpha=0.8)
        ax4.set_ylabel('Control Input')
        ax4.set_title('Control Actions vs Time')
        ax4.legend(fontsize=8, ncol=4)
        ax4.grid(True, alpha=0.3)
        self._add_maneuver_phases(ax4, maneuvers)
        
        # 5. Rewards
        ax5 = fig.add_subplot(gs[3, :])
        ax5.plot(traj['timestamps'], traj['rewards'], 'g-', linewidth=1.5, alpha=0.7)
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax5.set_ylabel('Reward')
        ax5.set_title('Reward Signal vs Time')
        ax5.grid(True, alpha=0.3)
        self._add_maneuver_phases(ax5, maneuvers)
        
        # 6. Error distribution
        ax6 = fig.add_subplot(gs[4, :])
        ax6.plot(traj['timestamps'], errors['errors'], 'r-', linewidth=1.5, label='Position Error')
        ax6.axhline(y=0.10, color='b', linestyle='--', alpha=0.5, linewidth=1, label='10cm')
        ax6.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='5cm')
        ax6.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, linewidth=1, label='2cm')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Error (m)')
        ax6.set_title('Position Error vs Time (Precision Levels)')
        ax6.legend(fontsize=8, ncol=4)
        ax6.grid(True, alpha=0.3)
        self._add_maneuver_phases(ax6, maneuvers)
        
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved episode profile to: {save_path}")
    
    def _add_maneuver_phases(self, ax, maneuvers):
        """Add colored background for maneuver phases"""
        colors = {'hover': 'green', 'motion': 'blue'}
        for phase in maneuvers:
            start_time = self.timestamps[phase['start']]
            end_idx = min(phase['end'], len(self.timestamps) - 1)
            end_time = self.timestamps[end_idx]
            ax.axvspan(start_time, end_time, alpha=0.1, color=colors[phase['type']])
    
    def save_statistics(self, save_path, episode_num):
        """Save episode statistics to JSON"""
        traj = self.get_trajectory_components()
        maneuvers = self.extract_maneuvers()
        errors = self.compute_error_distribution()
        
        stats = {
            'episode': episode_num,
            'duration': float(traj['timestamps'][-1]) if len(traj['timestamps']) > 0 else 0,
            'num_steps': len(self.states),
            'total_reward': float(np.sum(traj['rewards'])),
            'mean_reward': float(np.mean(traj['rewards'])),
            'error_statistics': {
                'mean': float(errors['mean_error']),
                'std': float(errors['std_error']),
                'max': float(errors['max_error']),
                'min': float(errors['min_error'])
            },
            'precision_rates': {
                '10cm': float(np.mean(errors['errors'] < 0.10) * 100),
                '5cm': float(np.mean(errors['errors'] < 0.05) * 100),
                '2cm': float(np.mean(errors['errors'] < 0.02) * 100)
            },
            'maneuver_phases': [
                {
                    'type': p['type'],
                    'duration': float(p['duration']),
                    'percentage': float(p['duration'] / traj['timestamps'][-1] * 100) if len(traj['timestamps']) > 0 else 0
                }
                for p in maneuvers
            ],
            'control_statistics': {
                'mean_thrust': float(np.mean(traj['actions'][:, 0])),
                'std_thrust': float(np.std(traj['actions'][:, 0])),
                'mean_torque': [float(x) for x in np.mean(traj['actions'][:, 1:4], axis=0)],
                'std_torque': [float(x) for x in np.std(traj['actions'][:, 1:4], axis=0)]
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved statistics to: {save_path}")
        return stats


def profile_episodes(model_path, num_episodes=5, output_dir='spatiotemporal_results'):
    """Profile multiple episodes and generate visualizations"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Initialize simulation
    print("Initializing simulation...")
    initialize_simulation()
    
    # Load agent
    print(f"Loading model from {model_path}...")
    device, batch_size, buffer_size, _ = configure_device()
    config = TrainingConfig(device, batch_size, buffer_size)
    agent = PIATSGAgent(config)
    
    # Load model with weights_only=False for PyTorch 2.6 compatibility
    import torch
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic1.load_state_dict(checkpoint['critic1'])
    agent.critic2.load_state_dict(checkpoint['critic2'])
    if 'pinn' in checkpoint:
        agent.adaptive_pinn.load_state_dict(checkpoint['pinn'])
    if 'operator' in checkpoint:
        agent.neural_operator.load_state_dict(checkpoint['operator'])
    if 'safety' in checkpoint:
        agent.safety_constraint.load_state_dict(checkpoint['safety'])
    
    agent.actor.eval()
    
    print(f"\nProfiling {num_episodes} episodes...\n")
    
    all_stats = []
    profiler = EpisodeProfiler()
    
    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}")
        
        profiler.reset()
        reset_simulation(randomize=True)
        obs = get_observation()
        agent.actor.reset_dt_history()
        
        time_step = 0
        dt = 0.01  # 10ms timestep
        max_steps = 1000
        
        for step in range(max_steps):
            action = agent.select_action(obs, deterministic=True)
            apply_action(action, obs)
            step_simulation()
            next_obs = get_observation()
            reward = compute_reward(next_obs)
            
            profiler.log_step(obs, action, reward, time_step * dt)
            
            obs = next_obs
            time_step += 1
            
            if check_done(obs):
                print(f"  Episode terminated at step {step}")
                break
        
        # Generate visualizations
        plot_filename = output_path / f'episode_{ep}_profile_{timestamp}.pdf'
        stats_filename = output_path / f'episode_{ep}_stats_{timestamp}.json'
        
        profiler.generate_profile_plot(plot_filename, ep)
        stats = profiler.save_statistics(stats_filename, ep)
        all_stats.append(stats)
        
        print(f"  Duration: {stats['duration']:.2f}s")
        print(f"  Total reward: {stats['total_reward']:.1f}")
        print(f"  Mean error: {stats['error_statistics']['mean']:.4f}m")
        print(f"  Precision rates: 10cm={stats['precision_rates']['10cm']:.1f}%, "
              f"5cm={stats['precision_rates']['5cm']:.1f}%, 2cm={stats['precision_rates']['2cm']:.1f}%")
        print(f"  Maneuver phases: {len(stats['maneuver_phases'])}")
        print()
    
    # Save aggregate statistics
    aggregate_stats = {
        'timestamp': timestamp,
        'num_episodes': num_episodes,
        'model_path': model_path,
        'aggregate_metrics': {
            'mean_duration': float(np.mean([s['duration'] for s in all_stats])),
            'mean_total_reward': float(np.mean([s['total_reward'] for s in all_stats])),
            'mean_error': float(np.mean([s['error_statistics']['mean'] for s in all_stats])),
            'mean_precision_10cm': float(np.mean([s['precision_rates']['10cm'] for s in all_stats])),
            'mean_precision_5cm': float(np.mean([s['precision_rates']['5cm'] for s in all_stats])),
            'mean_precision_2cm': float(np.mean([s['precision_rates']['2cm'] for s in all_stats]))
        },
        'individual_episodes': all_stats
    }
    
    aggregate_filename = output_path / f'aggregate_stats_{timestamp}.json'
    with open(aggregate_filename, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*60}")
    print(f"Episodes profiled: {num_episodes}")
    print(f"Mean duration: {aggregate_stats['aggregate_metrics']['mean_duration']:.2f}s")
    print(f"Mean total reward: {aggregate_stats['aggregate_metrics']['mean_total_reward']:.1f}")
    print(f"Mean error: {aggregate_stats['aggregate_metrics']['mean_error']:.4f}m")
    print(f"Mean precision rates:")
    print(f"  10cm: {aggregate_stats['aggregate_metrics']['mean_precision_10cm']:.1f}%")
    print(f"  5cm: {aggregate_stats['aggregate_metrics']['mean_precision_5cm']:.1f}%")
    print(f"  2cm: {aggregate_stats['aggregate_metrics']['mean_precision_2cm']:.1f}%")
    print(f"\nAll results saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate spatiotemporal episode profiles')
    parser.add_argument('--model', type=str, default='models/best_physics.pth',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to profile')
    parser.add_argument('--output', type=str, default='spatiotemporal_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    profile_episodes(args.model, args.episodes, args.output)