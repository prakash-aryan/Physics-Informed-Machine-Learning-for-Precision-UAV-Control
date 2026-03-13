# experiments/gradient_analysis_balanced.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
from simulation.environment import (
    initialize_simulation, reset_simulation, get_observation,
    step_simulation, apply_action, compute_reward, check_done
)
from core.balanced_agent import BalancedPIATSGAgent
from utils.config import TrainingConfig
from utils.gradient_logger import GradientLogger
import matplotlib.pyplot as plt

def run_gradient_analysis(config_name='balanced_piatsg', num_steps=1000, log_interval=10):
    """Run training with gradient logging"""
    
    # Setup
    initialize_simulation()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TrainingConfig(device, batch_size=256, buffer_size=100000)
    
    # Create balanced agent
    agent = BalancedPIATSGAgent(config)
    
    # Create logger
    logger = GradientLogger(log_dir=f'gradient_logs/{config_name}')
    
    print(f"Starting gradient analysis for {config_name}...")
    print(f"Logging every {log_interval} steps for {num_steps} total steps")
    
    # Reset environment
    reset_simulation(randomize=True)
    state = get_observation()
    
    for step in range(num_steps):
        # Select action
        with torch.no_grad():
            action = agent.select_action(state, deterministic=False)
        
        # Apply action and step
        apply_action(action, state)
        step_simulation()
        next_state = get_observation()
        reward = compute_reward(next_state)
        done = check_done(next_state)
        
        # Store transition
        agent.memory.push(state, action, reward, next_state, done)
        
        # Update and log gradients
        if len(agent.memory) >= config.batch_size and step % log_interval == 0:
            # Sample batch
            batch = agent.memory.sample(config.batch_size)
            states, actions, rewards, next_states, dones = batch
            
            # Ensure they're on the right device
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            
            # Log gradients AFTER balancing
            with torch.enable_grad():
                grads, alignment = logger.log_component_gradients(agent, states, actions, next_states)
            
            # Perform actual update with balancing
            losses = agent.update(config.batch_size)
            
            if step % 100 == 0:
                print(f"Step {step}/{num_steps}")
                mag_str = ', '.join([f"{k.split('_')[0]}: {v['magnitude']:.4f}" 
                                    for k, v in grads.items()])
                print(f"  Gradient magnitudes: {mag_str}")
                if alignment:
                    align_str = ', '.join([f"{k}: {v:.4f}" for k, v in alignment.items()])
                    print(f"  Alignment scores: {align_str}")
                if losses and 'actor_weight' in losses:
                    print(f"  Balance weights: actor={losses['actor_weight']:.3f}, "
                          f"pinn={losses['pinn_weight']:.3f}, "
                          f"op={losses['operator_weight']:.3f}, "
                          f"safe={losses['safety_weight']:.3f}")
        
        # Reset if done
        if done:
            reset_simulation(randomize=True)
            state = get_observation()
        else:
            state = next_state
    
    # Save logs
    logger.save_logs(f'{config_name}_gradients.json')
    
    return logger

def plot_gradient_analysis(logger, save_path='gradient_analysis.png'):
    """Plot gradient magnitudes and alignment over time"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot gradient magnitudes
    ax1 = axes[0]
    for component in ['actor', 'pinn', 'operator', 'safety']:
        key = f'{component}_magnitude'
        if key in logger.gradient_history:
            data = logger.gradient_history[key]
            ax1.plot(data, label=component, alpha=0.7)
    
    ax1.set_xlabel('Update Step')
    ax1.set_ylabel('Gradient Magnitude')
    ax1.set_title('Gradient Magnitudes by Component (WITH ReLoBRaLo)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot alignment scores
    ax2 = axes[1]
    if logger.alignment_history:
        # Extract alignment scores
        all_keys = set()
        for align_dict in logger.alignment_history:
            all_keys.update(align_dict.keys())
        
        for key in all_keys:
            values = [d.get(key, np.nan) for d in logger.alignment_history]
            ax2.plot(values, label=key, alpha=0.7)
    
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Gradient Alignment Scores (WITH ReLoBRaLo)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    # Run gradient analysis with balancing
    logger = run_gradient_analysis(
        config_name='balanced_piatsg',
        num_steps=1000,
        log_interval=10
    )
    
    # Plot results
    plot_gradient_analysis(logger, save_path='gradient_logs/balanced_piatsg/analysis.png')