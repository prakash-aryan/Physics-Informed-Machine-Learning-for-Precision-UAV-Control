#!/usr/bin/env python3
"""
PIATSG Framework - Comprehensive Ablation Study
Physics-Informed Adaptive Transformers with Safety Guarantees

Systematic evaluation of individual component contributions.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import time
import json
from pathlib import Path
import copy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.config import configure_device, set_reproducible_seed, TrainingConfig
from simulation.environment import initialize_simulation
from core.agent import PIATSGAgent
from training.evaluation import evaluate_agent


class AblationAgent(PIATSGAgent):
    """Modified agent for ablation studies with component control"""
    
    def __init__(self, config, ablation_config):
        # Store ablation config before initialization
        self.ablation_config = ablation_config
        
        # Initialize with modifications based on ablation
        super().__init__(config)
        
        # Apply ablation-specific modifications
        self.apply_ablations()
        
    def apply_ablations(self):
        """Apply ablation settings to disable specific components"""
        # Print active ablations
        disabled_components = []
        
        if self.ablation_config.get('disable_pinn', False):
            disabled_components.append("AdaptivePINN")
            
        if self.ablation_config.get('disable_operator', False):
            disabled_components.append("NeuralOperator")
            
        if self.ablation_config.get('disable_safety', False):
            disabled_components.append("SafetyConstraint")
            
        if self.ablation_config.get('disable_dt', False):
            disabled_components.append("DecisionTransformer")
            # For DT ablation, we need to modify the actor's forward method
            self._modify_actor_for_no_dt()
            
        if disabled_components:
            print(f"Ablation: Disabled components - {', '.join(disabled_components)}")
        else:
            print("Ablation: Full PIATSG model (all components enabled)")
    
    def _modify_actor_for_no_dt(self):
        """Modify actor to skip Decision Transformer processing"""
        # Save the original forward method
        original_forward = self.actor.forward
        
        def forward_without_dt(state, deterministic=False):
            batch_size = state.shape[0] if len(state.shape) > 1 else 1
            
            # Skip DT processing, directly use state
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            # Use only the high-level policy without DT context
            high_level_action = self.actor.high_level_policy(
                torch.cat([state, torch.zeros(batch_size, self.actor.embedding_dim).to(state.device)], dim=-1)
            )
            
            # Skip precision refinement
            mean = high_level_action
            mean = torch.clamp(mean, -10, 10)
            
            log_std = self.actor.log_std.expand_as(mean)
            log_std = torch.clamp(log_std, -10, 2)
            std = torch.exp(log_std)
            std = torch.clamp(std, 1e-4, 10)
            
            if torch.isnan(mean).any() or torch.isinf(mean).any():
                mean = torch.zeros_like(mean)
            if torch.isnan(std).any() or torch.isinf(std).any():
                std = torch.ones_like(std) * 0.1
            
            if deterministic:
                return torch.clamp(mean.squeeze(0) if batch_size == 1 else mean, -1, 1)
            else:
                normal = torch.distributions.Normal(mean, std)
                action = normal.rsample()
                action = torch.clamp(action, -1, 1)
                return action.squeeze(0) if batch_size == 1 else action
        
        # Replace the forward method
        self.actor.forward = forward_without_dt
    
    def select_action(self, state, deterministic=False):
        """Select action with ablation-aware safety filtering"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            raw_action = self.actor(state_tensor, deterministic=deterministic)
            
            # Apply safety constraint filtering only if not disabled
            if not self.ablation_config.get('disable_safety', False) and len(self.memory) > 8000:
                # Enable gradients temporarily for CBF computation
                state_for_cbf = state_tensor.clone().requires_grad_(True)
                action_for_cbf = raw_action.clone().requires_grad_(True)
                
                # Compute safety constraint with gradients enabled
                with torch.enable_grad():
                    safety_logits = self.safety_constraint(state_for_cbf.unsqueeze(0), action_for_cbf.unsqueeze(0))
                    safety_mask = (safety_logits > 0).float()
                    
                    if safety_mask.item() < 0.5:
                        raw_action = raw_action * 0.8
            
            # Conservative action clipping
            action = torch.clamp(raw_action, -0.6, 0.6)
            
            # Training phase action scaling
            if len(self.memory) < 3000:
                action = action * 0.5
            elif len(self.memory) < 8000:
                action = action * 0.75
        
        return action.cpu().numpy()
    
    def update(self, batch_size=None, update_physics_components=True):
        """Update all components with ablation-aware physics losses"""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if len(self.memory) < batch_size:
            return None
        
        self.update_count += 1
        self.total_training_cycles += 1
        
        # Update physics weights based on curriculum
        self._update_physics_weights()
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Check for NaN in input data
        if (torch.isnan(states).any() or torch.isnan(actions).any() or 
            torch.isnan(rewards).any() or torch.isnan(next_states).any()):
            print("Warning: NaN detected in training data, skipping update")
            return None
        
        losses = {}
        
        try:
            # Update Critics (always active)
            critic_losses = self._update_critics(states, actions, rewards, next_states, dones)
            losses.update(critic_losses)
            
            # Update Actor (with or without safety)
            actor_loss = self._update_actor_with_ablation(states)
            losses['actor_loss'] = actor_loss
            
            # Update Alpha (always active)
            alpha_loss = self._update_alpha(states)
            losses['alpha_loss'] = alpha_loss
            
            # Update Physics Components with ablation control
            should_update_physics = (
                update_physics_components and 
                not self.recovery_mode and
                len(self.memory) > self.min_buffer_for_physics and
                (self.total_training_cycles % self.current_physics_update_frequency == 0)
            )
            
            if should_update_physics:
                self.physics_update_count += 1
                physics_losses = self._update_physics_components_with_ablation(states, actions, next_states)
                losses.update(physics_losses)
            else:
                # Return zero losses for physics components when not updated
                losses.update({'pinn_loss': 0.0, 'operator_loss': 0.0, 'safety_loss': 0.0})
            
            # Update target networks
            self._update_target_networks()
            
        except Exception as e:
            print(f"Warning: Error in update: {e}")
            return None
        
        return losses
    
    def _update_actor_with_ablation(self, states):
        """Update actor network with ablation-aware safety constraints"""
        with torch.amp.autocast('cuda', enabled=True):
            new_actions = self.actor(states, deterministic=False)
            new_actions = torch.clamp(new_actions, -1, 1)
            
            # Standard actor loss
            q1_new = self.critic1(torch.cat([states, new_actions], dim=1))
            q2_new = self.critic2(torch.cat([states, new_actions], dim=1))
            q_new = torch.min(q1_new, q2_new)
            
            alpha = torch.clamp(self.log_alpha.exp(), 0.01, 5.0)
            log_probs = -0.5 * torch.sum((new_actions ** 2), dim=1, keepdim=True)
            actor_loss = (alpha.detach() * log_probs - q_new).mean()
            
            # Add safety constraint penalty only if not disabled
            if (not self.ablation_config.get('disable_safety', False) and 
                len(self.memory) > self.min_buffer_for_physics and 
                not self.recovery_mode):
                try:
                    states_for_safety = states.clone()
                    actions_for_safety = new_actions.clone()
                    safety_violation = self.safety_constraint.compute_safety_violation_loss(
                        states_for_safety, actions_for_safety
                    )
                    
                    safety_weight = self.safety_loss_weight
                    if self.episode_progress > 0.8 and self.instability_counter > 10:
                        safety_weight *= 0.5
                    
                    actor_loss = actor_loss + safety_weight * safety_violation
                except Exception as safety_error:
                    print(f"Warning: Safety constraint error in actor update: {safety_error}")
                    pass
            
            if torch.isnan(actor_loss):
                print("Warning: NaN in actor loss")
                return 0.0
        
        # Update actor
        self.actor_optimizer.zero_grad()
        if torch.isfinite(actor_loss):
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        
        return actor_loss.item() if torch.isfinite(actor_loss) else 0.0
    
    def _update_physics_components_with_ablation(self, states, actions, next_states):
        """Update physics-informed components with ablation control"""
        losses = {}
        
        try:
            # Apply gradient scaling in late training
            grad_scale = 1.0
            if self.episode_progress > 0.8:
                grad_scale = 0.5
            
            # Physics-Informed Neural Network update (only if not disabled)
            if not self.ablation_config.get('disable_pinn', False):
                self.pinn_optimizer.zero_grad()
                
                states_safe = states.clone()
                actions_safe = actions.clone()
                next_states_safe = next_states.clone()
                
                pinn_loss = self.adaptive_pinn.physics_loss(states_safe, next_states_safe, actions_safe)
                
                # L2 regularization
                l2_weight = 1e-5 if self.episode_progress < 0.8 else 5e-5
                l2_reg = torch.tensor(0.0, device=states.device)
                for param in self.adaptive_pinn.parameters():
                    l2_reg += torch.norm(param)
                
                pinn_loss_total = pinn_loss + l2_weight * l2_reg
                
                if torch.isfinite(pinn_loss_total) and pinn_loss_total < 20.0:
                    (pinn_loss_total * grad_scale).backward()
                    torch.nn.utils.clip_grad_norm_(self.adaptive_pinn.parameters(), 
                                                   self.physics_grad_norm * grad_scale)
                    self.pinn_optimizer.step()
                    self.pinn_scheduler.step(pinn_loss.item())
                    losses['pinn_loss'] = pinn_loss.item()
                else:
                    losses['pinn_loss'] = 0.0
            else:
                losses['pinn_loss'] = 0.0
            
            # Neural Operator update (only if not disabled)
            if not self.ablation_config.get('disable_operator', False):
                self.operator_optimizer.zero_grad()
                
                states_safe = states.clone()
                actions_safe = actions.clone()
                next_states_safe = next_states.clone()
                
                operator_pred = self.neural_operator(states_safe, actions_safe)
                operator_loss = nn.functional.smooth_l1_loss(operator_pred, next_states_safe)
                
                # L2 regularization
                l2_weight = 1e-5 if self.episode_progress < 0.8 else 5e-5
                l2_reg_op = torch.tensor(0.0, device=states.device)
                for param in self.neural_operator.parameters():
                    l2_reg_op += torch.norm(param)
                
                operator_loss_total = operator_loss + l2_weight * l2_reg_op
                
                if torch.isfinite(operator_loss_total) and operator_loss_total < 20.0:
                    (operator_loss_total * grad_scale).backward()
                    torch.nn.utils.clip_grad_norm_(self.neural_operator.parameters(), 
                                                   self.physics_grad_norm * grad_scale)
                    self.operator_optimizer.step()
                    self.operator_scheduler.step(operator_loss.item())
                    losses['operator_loss'] = operator_loss.item()
                else:
                    losses['operator_loss'] = 0.0
            else:
                losses['operator_loss'] = 0.0
            
            # Safety Constraint update (only if not disabled)
            if not self.ablation_config.get('disable_safety', False):
                self.safety_optimizer.zero_grad()
                
                batch_size = states.shape[0]
                
                # Adaptive sampling based on stability
                if self.instability_counter > 20:
                    safe_ratio, unsafe_ratio, boundary_ratio = 0.6, 0.2, 0.2
                elif self.episode_progress > 0.8:
                    safe_ratio, unsafe_ratio, boundary_ratio = 0.4, 0.3, 0.3
                else:
                    safe_ratio, unsafe_ratio, boundary_ratio = 0.3, 0.3, 0.4
                
                safe_size = int(batch_size * safe_ratio)
                unsafe_size = int(batch_size * unsafe_ratio)
                boundary_size = batch_size - safe_size - unsafe_size
                
                indices = torch.randperm(batch_size)
                safe_indices = indices[:safe_size]
                unsafe_indices = indices[safe_size:safe_size + unsafe_size]
                boundary_indices = indices[safe_size + unsafe_size:]
                
                safe_loss = torch.tensor(0.0, device=states.device)
                unsafe_loss = torch.tensor(0.0, device=states.device)
                boundary_loss = torch.tensor(0.0, device=states.device)
                
                # Safe samples with adaptive margin
                safety_margin = 0.1 + min(self.episode_progress * 0.2, 0.2)
                if len(safe_indices) > 0:
                    safe_logits = self.safety_constraint(states[safe_indices], actions[safe_indices])
                    safe_loss = nn.functional.relu(-safe_logits + safety_margin).mean()
                
                # Unsafe samples
                aggression_factor = min(1.2 + self.episode_progress * 0.3, 1.5)
                if len(unsafe_indices) > 0:
                    unsafe_actions = actions[unsafe_indices] * aggression_factor
                    unsafe_actions_clamped = torch.clamp(unsafe_actions, -1, 1)
                    unsafe_logits = self.safety_constraint(states[unsafe_indices], unsafe_actions_clamped)
                    unsafe_loss = nn.functional.relu(unsafe_logits + safety_margin).mean()
                
                # Boundary samples
                boundary_noise = min(0.05 + self.episode_progress * 0.05, 0.1)
                if len(boundary_indices) > 0:
                    boundary_actions = actions[boundary_indices] + boundary_noise * torch.randn_like(actions[boundary_indices])
                    boundary_actions_clamped = torch.clamp(boundary_actions, -1, 1)
                    boundary_logits = self.safety_constraint(states[boundary_indices], boundary_actions_clamped)
                    boundary_loss = torch.abs(boundary_logits).mean()
                
                safety_loss = safe_loss + unsafe_loss + 0.5 * boundary_loss
                
                # L2 regularization
                l2_weight = 1e-5 if self.episode_progress < 0.8 else 5e-5
                l2_reg_safety = torch.tensor(0.0, device=states.device)
                for param in self.safety_constraint.parameters():
                    l2_reg_safety += torch.norm(param)
                
                safety_loss_total = safety_loss + l2_weight * l2_reg_safety
                
                if torch.isfinite(safety_loss_total) and safety_loss_total < 20.0:
                    (safety_loss_total * grad_scale).backward()
                    torch.nn.utils.clip_grad_norm_(self.safety_constraint.parameters(), 
                                                   self.physics_grad_norm * grad_scale)
                    self.safety_optimizer.step()
                    self.safety_scheduler.step(safety_loss.item())
                    losses['safety_loss'] = safety_loss.item()
                else:
                    losses['safety_loss'] = 0.0
            else:
                losses['safety_loss'] = 0.0
                
        except Exception as e:
            print(f"Warning: Error in physics components: {e}")
            losses = {'pinn_loss': 0.0, 'operator_loss': 0.0, 'safety_loss': 0.0}
        
        return losses


def run_ablation_evaluation(model_path, ablation_config, episodes=20, verbose=True):
    """Run evaluation for a specific ablation configuration"""
    # Initialize environment without viewer
    if not hasattr(run_ablation_evaluation, 'initialized'):
        initialize_simulation()
        run_ablation_evaluation.initialized = True
    
    # Configure device
    device, batch_size, buffer_size, _ = configure_device()
    
    # Create proper config using TrainingConfig
    config = TrainingConfig(device, batch_size, buffer_size)
    
    # Create agent with ablation
    agent = AblationAgent(config, ablation_config)
    
    # Load trained weights
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Load all components (they'll be disabled in forward pass if needed)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic1.load_state_dict(checkpoint['critic1'])
        agent.critic2.load_state_dict(checkpoint['critic2'])
        agent.adaptive_pinn.load_state_dict(checkpoint['pinn'])
        agent.neural_operator.load_state_dict(checkpoint['operator'])
        agent.safety_constraint.load_state_dict(checkpoint['safety'])
        agent.log_alpha = checkpoint['log_alpha']
        
        # Copy target networks
        agent.target_critic1.load_state_dict(agent.critic1.state_dict())
        agent.target_critic2.load_state_dict(agent.critic2.state_dict())
        
        if verbose:
            print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found!")
    
    # Run actual evaluation using your existing evaluate_agent function
    print(f"Running {episodes} evaluation episodes...")
    results = evaluate_agent(agent, episodes=episodes)
    
    # Add configuration info
    results['config'] = ablation_config
    results['config_name'] = get_config_name(ablation_config)
    
    return results


def get_config_name(ablation_config):
    """Generate descriptive name for ablation configuration"""
    if not any(ablation_config.values()):
        return "Full_PIATSG"
    
    disabled = []
    if ablation_config.get('disable_pinn', False):
        disabled.append("PINN")
    if ablation_config.get('disable_operator', False):
        disabled.append("NeuralOp")
    if ablation_config.get('disable_safety', False):
        disabled.append("Safety")
    if ablation_config.get('disable_dt', False):
        disabled.append("DT")
        
    return f"Without_{'+'.join(disabled)}"


def run_complete_ablation_study(model_path, output_dir="ablation_results"):
    """Run complete ablation study with all configurations"""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Define ablation configurations
    configurations = [
        {'disable_pinn': False, 'disable_operator': False, 'disable_safety': False, 'disable_dt': False},  # Full model
        {'disable_pinn': True, 'disable_operator': False, 'disable_safety': False, 'disable_dt': False},   # Without PINN
        {'disable_pinn': False, 'disable_operator': True, 'disable_safety': False, 'disable_dt': False},   # Without NeuralOperator
        {'disable_pinn': False, 'disable_operator': False, 'disable_safety': True, 'disable_dt': False},   # Without Safety
        {'disable_pinn': False, 'disable_operator': False, 'disable_safety': False, 'disable_dt': True},   # Without DT
        {'disable_pinn': True, 'disable_operator': True, 'disable_safety': False, 'disable_dt': False},    # Without PINN+NeuralOp
        {'disable_pinn': True, 'disable_operator': True, 'disable_safety': True, 'disable_dt': False},     # Without all physics
        {'disable_pinn': True, 'disable_operator': True, 'disable_safety': True, 'disable_dt': True},      # Baseline (no enhancements)
    ]
    
    # Run evaluations
    all_results = []
    
    for i, config in enumerate(configurations):
        config_name = get_config_name(config)
        print(f"\n{'='*60}")
        print(f"Evaluating Configuration {i+1}/{len(configurations)}: {config_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        results = run_ablation_evaluation(model_path, config, episodes=20)
        eval_time = time.time() - start_time
        
        results['eval_time'] = eval_time
        all_results.append(results)
        
        # Print summary
        print(f"\n{config_name} Results:")
        print(f"  Precision (10cm): {results['mean_precision_10cm']:.1f}%")
        print(f"  Super-precision (5cm): {results['mean_precision_5cm']:.1f}%")
        print(f"  Ultra-precision (2cm): {results['mean_precision_2cm']:.1f}%")
        print(f"  Physics score: {results['mean_physics_score']:.1f}%")
        print(f"  Safety score: {results['mean_safety_score']:.1f}%")
        print(f"  Evaluation time: {eval_time:.1f}s")
    
    # Save raw results
    results_file = output_path / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison DataFrame
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Configuration': result['config_name'],
            'Precision_10cm': result['mean_precision_10cm'],
            'Precision_5cm': result['mean_precision_5cm'],
            'Precision_2cm': result['mean_precision_2cm'],
            'Physics_Score': result['mean_physics_score'],
            'Safety_Score': result['mean_safety_score']
        })
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_path / "ablation_comparison.csv", index=False)
    
    # Generate plots
    generate_ablation_plots(df, output_path)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    print("\nPerformance Summary:")
    print(df.to_string(index=False))
    
    # Calculate relative improvements
    full_idx = 0      # First configuration is the full model
    baseline_idx = len(df) - 1  # Last configuration is the most stripped down
    
    print("\nRelative Improvements (Full PIATSG vs configurations):")
    for idx in range(1, len(df)):
        print(f"\n{df.loc[idx, 'Configuration']}:")
        for metric in ['Precision_10cm', 'Precision_5cm', 'Precision_2cm', 'Physics_Score', 'Safety_Score']:
            full_val = df.loc[full_idx, metric]
            other_val = df.loc[idx, metric]
            diff = full_val - other_val
            if diff > 0:
                print(f"  {metric}: {diff:+.1f}% drop (Full PIATSG better)")
            elif diff < 0:
                print(f"  {metric}: {abs(diff):.1f}% worse (Component removal improved performance)")
            else:
                print(f"  {metric}: No change")
    
    # Compare with baseline
    print(f"\nFull PIATSG vs Baseline (no enhancements):")
    for metric in ['Precision_10cm', 'Precision_5cm', 'Precision_2cm', 'Physics_Score', 'Safety_Score']:
        full_val = df.loc[full_idx, metric]
        baseline_val = df.loc[baseline_idx, metric]
        improvement = ((full_val - baseline_val) / baseline_val) * 100 if baseline_val > 0 else 0
        print(f"  {metric}: {improvement:+.1f}% improvement")
    
    return all_results, df


def generate_ablation_plots(df, output_path):
    """Generate visualization plots for ablation study"""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
    
    # 1. Bar plot for all metrics
    fig, ax = plt.subplots(figsize=(16, 8))
    
    metrics = ['Precision_10cm', 'Precision_5cm', 'Precision_2cm', 'Physics_Score', 'Safety_Score']
    x = np.arange(len(df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2) * width + width/2
        bars = ax.bar(x + offset, df[metric], width, label=metric.replace('_', ' '), color=colors[i])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('PIATSG Ablation Study: Component Contributions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Configuration'], rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'ablation_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Radar plot for full model vs baseline
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Get data for full model and most basic configuration
    full_data = df.iloc[0]  # Full PIATSG
    baseline_data = df.iloc[len(df)-1]  # Most stripped down version
    
    # Prepare data
    categories = ['Precision\n10cm', 'Precision\n5cm', 'Precision\n2cm', 'Physics\nScore', 'Safety\nScore']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    full_values = [full_data[m] for m in metrics]
    full_values += full_values[:1]
    
    baseline_values = [baseline_data[m] for m in metrics]
    baseline_values += baseline_values[:1]
    
    # Plot
    ax.plot(angles, full_values, 'o-', linewidth=2, label='Full PIATSG', color='blue')
    ax.fill(angles, full_values, alpha=0.25, color='blue')
    
    ax.plot(angles, baseline_values, 'o-', linewidth=2, label=baseline_data['Configuration'], color='red')
    ax.fill(angles, baseline_values, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(f'Full PIATSG vs {baseline_data["Configuration"]}', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / 'ablation_radar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Component importance heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate relative performance drop when each component is removed
    importance_data = []
    full_performance = df.iloc[0]
    
    # Single component removals
    component_configs = {
        'PINN': 1,
        'NeuralOp': 2,
        'Safety': 3,
        'DT': 4
    }
    
    for component, idx in component_configs.items():
        reduced_performance = df.iloc[idx]
        
        importance_row = []
        for metric in metrics:
            drop = full_performance[metric] - reduced_performance[metric]
            importance_row.append(drop)
        
        importance_data.append(importance_row)
    
    importance_df = pd.DataFrame(importance_data, 
                                columns=[m.replace('_', ' ') for m in metrics],
                                index=list(component_configs.keys()))
    
    sns.heatmap(importance_df, annot=True, fmt='.1f', cmap='RdYlBu_r', 
               center=0, vmin=-5, vmax=30,
               cbar_kws={'label': 'Performance Drop (%)'})
    
    plt.title('Component Importance: Performance Drop When Removed', fontsize=14, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Components', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path / 'ablation_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_path}/")


def main():
    """Main execution function for ablation study"""
    parser = argparse.ArgumentParser(description='PIATSG Framework - Ablation Study')
    parser.add_argument('--model', type=str, default='models/best_physics.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='ablation_results',
                       help='Output directory for results')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of evaluation episodes per configuration')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--single', type=str, default=None,
                       help='Run single ablation: pinn, operator, safety, dt')
    
    args = parser.parse_args()
    
    # Set seed
    set_reproducible_seed(args.seed)
    
    if args.single:
        # Run single ablation
        config = {
            'disable_pinn': args.single == 'pinn',
            'disable_operator': args.single == 'operator',
            'disable_safety': args.single == 'safety',
            'disable_dt': args.single == 'dt'
        }
        
        print(f"Running single ablation: {get_config_name(config)}")
        results = run_ablation_evaluation(args.model, config, episodes=args.episodes)
        
        print(f"\nResults:")
        print(f"  Precision (10cm): {results['mean_precision_10cm']:.1f}%")
        print(f"  Super-precision (5cm): {results['mean_precision_5cm']:.1f}%")
        print(f"  Ultra-precision (2cm): {results['mean_precision_2cm']:.1f}%")
        print(f"  Physics score: {results['mean_physics_score']:.1f}%")
        print(f"  Safety score: {results['mean_safety_score']:.1f}%")
    else:
        # Run complete ablation study
        run_complete_ablation_study(args.model, args.output)


if __name__ == "__main__":
    main()