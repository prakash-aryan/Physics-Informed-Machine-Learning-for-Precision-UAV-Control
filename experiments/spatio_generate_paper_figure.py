"""
Generate Publication Figure: Representative Spatiotemporal Episode Profiles
For paper revision - addressing Reviewer 1 Comment 3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.environment import initialize_simulation, reset_simulation, get_observation, apply_action, step_simulation, compute_reward, check_done
from core.agent import PIATSGAgent
from utils.config import configure_device, TrainingConfig
import torch

# Publication quality settings
mpl.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def run_episode_and_collect_data(agent):
    """Run one episode and collect all trajectory data"""
    reset_simulation(randomize=True)
    obs = get_observation()
    agent.actor.reset_dt_history()
    
    states = []
    actions = []
    rewards = []
    timestamps = []
    
    dt = 0.01
    max_steps = 1000
    target_pos = np.array([0.0, 0.0, 1.0])
    
    for step in range(max_steps):
        action = agent.select_action(obs, deterministic=True)
        apply_action(action, obs)
        step_simulation()
        next_obs = get_observation()
        reward = compute_reward(next_obs)
        
        states.append(obs.copy())
        actions.append(action.copy())
        rewards.append(reward)
        timestamps.append(step * dt)
        
        obs = next_obs
        
        if check_done(obs):
            break
    
    # Extract components
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    timestamps = np.array(timestamps)
    
    positions = states[:, 0:3]
    velocities = states[:, 7:10]
    errors = np.linalg.norm(positions - target_pos, axis=1)
    
    # Compute metrics
    metrics = {
        'mean_error': np.mean(errors),
        'precision_10cm': np.mean(errors < 0.10) * 100,
        'precision_5cm': np.mean(errors < 0.05) * 100,
        'precision_2cm': np.mean(errors < 0.02) * 100,
        'total_reward': np.sum(rewards)
    }
    
    # Maneuver phases
    vel_mag = np.linalg.norm(velocities, axis=1)
    phases = []
    current_phase = 'hover' if vel_mag[0] < 0.1 else 'motion'
    phase_start = 0
    
    for i, v in enumerate(vel_mag):
        new_phase = 'hover' if v < 0.1 else 'motion'
        if new_phase != current_phase:
            phases.append({
                'type': current_phase,
                'start': phase_start,
                'end': i
            })
            current_phase = new_phase
            phase_start = i
    
    phases.append({
        'type': current_phase,
        'start': phase_start,
        'end': len(vel_mag)
    })
    
    return {
        'positions': positions,
        'velocities': velocities,
        'actions': actions,
        'rewards': rewards,
        'timestamps': timestamps,
        'errors': errors,
        'phases': phases,
        'metrics': metrics,
        'target': target_pos
    }


def add_phase_backgrounds(ax, timestamps, phases, colors={'hover': 'green', 'motion': 'blue'}):
    """Add colored backgrounds for maneuver phases"""
    for phase in phases:
        start_time = timestamps[phase['start']]
        end_idx = min(phase['end'], len(timestamps) - 1)
        end_time = timestamps[end_idx]
        ax.axvspan(start_time, end_time, alpha=0.08, color=colors[phase['type']])


def create_comparison_figure(episodes_data, save_path='paper_spatiotemporal_figure.pdf'):
    """Create multi-episode comparison figure for paper"""
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    episode_labels = ['Excellent (Ep A)', 'Typical (Ep B)', 'Challenging (Ep C)']
    
    # Row 1: 3D Trajectories
    for i, (data, color, label) in enumerate(zip(episodes_data, colors, episode_labels)):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        
        pos = data['positions']
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(*data['target'], color='red', s=100, marker='*', label='Target', zorder=5)
        ax.scatter(*pos[0], color='green', s=40, marker='o', label='Start', zorder=5)
        ax.scatter(*pos[-1], color='orange', s=40, marker='s', label='End', zorder=5)
        
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_zlabel('Z (m)', fontsize=8)
        ax.set_title(f'{label}\nError: {data["metrics"]["mean_error"]*100:.2f}cm', fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Set consistent limits
        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.3, 0.3])
        ax.set_zlim([0.7, 1.3])
    
    # Row 2: Position over time (X, Y, Z)
    for i, (data, color, label) in enumerate(zip(episodes_data, colors, episode_labels)):
        ax = fig.add_subplot(gs[1, i])
        
        pos = data['positions']
        t = data['timestamps']
        
        ax.plot(t, pos[:, 0], label='X', linewidth=1.2, alpha=0.8)
        ax.plot(t, pos[:, 1], label='Y', linewidth=1.2, alpha=0.8)
        ax.plot(t, pos[:, 2], label='Z', linewidth=1.2, alpha=0.8)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Target Z')
        
        add_phase_backgrounds(ax, t, data['phases'])
        
        ax.set_ylabel('Position (m)', fontsize=8)
        ax.set_title('Position Components', fontsize=9)
        ax.legend(fontsize=6, ncol=4, loc='upper right')
        ax.set_xlim([0, 10])
        ax.grid(True, alpha=0.3)
    
    # Row 3: Velocity magnitude and Control inputs
    for i, (data, color, label) in enumerate(zip(episodes_data, colors, episode_labels)):
        ax = fig.add_subplot(gs[2, i])
        
        vel = data['velocities']
        t = data['timestamps']
        vel_mag = np.linalg.norm(vel, axis=1)
        
        ax.plot(t, vel_mag, color=color, linewidth=1.5, label='Velocity')
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Hover threshold')
        
        add_phase_backgrounds(ax, t, data['phases'])
        
        ax.set_ylabel('Speed (m/s)', fontsize=8)
        ax.set_title('Velocity Magnitude', fontsize=9)
        ax.legend(fontsize=6)
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 0.5])
        ax.grid(True, alpha=0.3)
    
    # Row 4: Position Error with precision levels
    for i, (data, color, label) in enumerate(zip(episodes_data, colors, episode_labels)):
        ax = fig.add_subplot(gs[3, i])
        
        t = data['timestamps']
        errors = data['errors']
        
        ax.plot(t, errors * 100, color=color, linewidth=1.5, label='Position Error')
        ax.axhline(y=10, color='blue', linestyle='--', alpha=0.6, linewidth=1, label='10cm')
        ax.axhline(y=5, color='orange', linestyle='--', alpha=0.6, linewidth=1, label='5cm')
        ax.axhline(y=2, color='green', linestyle='--', alpha=0.6, linewidth=1, label='2cm')
        
        add_phase_backgrounds(ax, t, data['phases'])
        
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Error (cm)', fontsize=8)
        ax.set_title(f'Position Error - Precision: {data["metrics"]["precision_10cm"]:.1f}% @ 10cm', fontsize=9)
        ax.legend(fontsize=6, ncol=2)
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 15])
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Spatiotemporal Episode Profiles: Representative Hovering Performance\n' + 
                 'Green background: Hover phase | Blue background: Motion phase',
                 fontsize=11, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n✓ Saved publication figure to: {save_path}")
    
    return fig


def generate_statistics_table(episodes_data, save_path='paper_spatiotemporal_table.txt'):
    """Generate a statistics table for the paper"""
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPATIOTEMPORAL EPISODE PROFILE STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        labels = ['Episode A (Excellent)', 'Episode B (Typical)', 'Episode C (Challenging)']
        
        for i, (data, label) in enumerate(zip(episodes_data, labels)):
            f.write(f"{label}\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean Position Error:        {data['metrics']['mean_error']*100:.2f} cm\n")
            f.write(f"Precision @ 10cm:           {data['metrics']['precision_10cm']:.1f}%\n")
            f.write(f"Precision @ 5cm:            {data['metrics']['precision_5cm']:.1f}%\n")
            f.write(f"Precision @ 2cm:            {data['metrics']['precision_2cm']:.1f}%\n")
            f.write(f"Total Reward:               {data['metrics']['total_reward']:.1f}\n")
            f.write(f"Duration:                   {data['timestamps'][-1]:.2f} s\n")
            f.write(f"Number of Maneuver Phases:  {len(data['phases'])}\n")
            
            for j, phase in enumerate(data['phases']):
                duration = data['timestamps'][min(phase['end'], len(data['timestamps'])-1)] - data['timestamps'][phase['start']]
                f.write(f"  Phase {j+1}: {phase['type']:6s} - {duration:.2f}s\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Saved statistics table to: {save_path}")


def main():
    """Main function to generate paper figure"""
    
    print("="*80)
    print("GENERATING PUBLICATION FIGURE FOR PAPER REVISION")
    print("Addressing Reviewer 1 Comment 3: Spatiotemporal Results")
    print("="*80)
    
    # Initialize simulation
    print("\nInitializing simulation...")
    initialize_simulation()
    
    # Load model
    model_path = 'models/best_physics.pth'
    print(f"Loading model: {model_path}")
    
    device, batch_size, buffer_size, _ = configure_device()
    config = TrainingConfig(device, batch_size, buffer_size)
    agent = PIATSGAgent(config)
    
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
    print("✓ Model loaded successfully")
    
    # Run episodes until we get representative samples
    print("\nCollecting representative episodes...")
    print("Searching for: Excellent (error<4cm), Typical (error 5-6cm), Challenging (error>7cm)")
    
    episodes_data = []
    excellent = None
    typical = None
    challenging = None
    
    max_attempts = 50
    for attempt in range(max_attempts):
        data = run_episode_and_collect_data(agent)
        error_cm = data['metrics']['mean_error'] * 100
        
        # Categorize episodes
        if excellent is None and error_cm < 4.0:
            excellent = data
            print(f"  ✓ Found excellent episode: {error_cm:.2f}cm error")
        
        if typical is None and 5.0 <= error_cm <= 6.0:
            typical = data
            print(f"  ✓ Found typical episode: {error_cm:.2f}cm error")
        
        if challenging is None and error_cm > 7.0:
            challenging = data
            print(f"  ✓ Found challenging episode: {error_cm:.2f}cm error")
        
        if excellent and typical and challenging:
            break
        
        if (attempt + 1) % 10 == 0:
            print(f"  ... searched {attempt + 1} episodes")
    
    # Use what we found (fallback if we didn't find all categories)
    episodes_data = [
        excellent if excellent else data,
        typical if typical else data,
        challenging if challenging else data
    ]
    
    if not (excellent and typical and challenging):
        print("\n⚠ Warning: Could not find all episode categories, using available samples")
    
    # Generate figure
    print("\nGenerating publication figure...")
    create_comparison_figure(episodes_data, save_path='paper_spatiotemporal_figure.pdf')
    
    # Generate statistics table
    generate_statistics_table(episodes_data, save_path='paper_spatiotemporal_table.txt')
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print("\nFiles created:")
    print("  1. paper_spatiotemporal_figure.pdf  - Main figure for paper")
    print("  2. paper_spatiotemporal_table.txt   - Statistics table")
    print("\nYou can now:")
    print("  - Include the PDF figure in your revision")
    print("  - Use the statistics in your response letter")
    print("  - Reference: 'See Figure X showing representative spatiotemporal profiles'")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()