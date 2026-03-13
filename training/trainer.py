"""
PIATSG Framework - Main Training Loop
Physics-Informed Adaptive Transformers with Safety Guarantees

Training loop with logging and evaluation.
"""

import time
import torch
import numpy as np
from collections import deque
from tensorboardX import SummaryWriter

from core.agent import PIATSGAgent
from simulation.environment import (
    reset_simulation, get_observation, apply_action, 
    step_simulation, compute_reward, check_done
)
from training.evaluation import evaluate_and_save_best, create_training_summary

class PIATSGTrainer:
    """Main trainer for PIATSG agent with physics-informed components"""
    
    def __init__(self, config):
        self.config = config
        self.agent = PIATSGAgent(config)
        
        # Training tracking
        self.best_scores = {
            'precision_10cm': 0.0,
            'precision_5cm': 0.0,
            'precision_2cm': 0.0,
            'physics': 0.0,
            'safety': 0.0
        }
        
        # Physics and safety loss tracking
        self.physics_loss_history = deque(maxlen=1000)
        self.safety_loss_history = deque(maxlen=1000)
        
        # Physics update frequency control
        self.training_cycle_count = 0
        self.physics_update_frequency = 1  # Update physics every training cycle
        
        # Logging
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'logs/piatsg_training_{timestamp}')
        self.reward_history = deque(maxlen=200)
        
        print("PIATSG Trainer initialized!")
        print("Physics-Informed Components:")
        print(f"- AdaptivePINN: UAV dynamics with normalized constraints")
        print(f"- NeuralOperator: Function-to-function mapping with residual connections")
        print(f"- SafetyConstraint: Control Barrier Functions for flight envelope")
        print("Training Configuration:")
        print(f"- Batch size: {config.batch_size:,}")
        print(f"- Training frequency: Every {config.training_frequency} steps")
        print(f"- Updates per session: {config.updates_per_training}")
        print(f"- Buffer size: {config.buffer_size:,}")
        print(f"- Physics update frequency: Every {self.physics_update_frequency} training cycles")
    
    def train(self):
        """Main training loop with physics-informed monitoring"""
        episode = 0
        training_start = time.time()
        
        print("PIATSG Training Started!")
        
        while episode < self.config.num_episodes:
            episode_reward, episode_steps = self._run_episode(episode)
            
            # Update tracking
            self.reward_history.append(episode_reward)
            avg_reward = np.mean(self.reward_history)
            
            # Logging
            self._log_episode(episode, episode_reward, episode_steps, avg_reward)
            
            # Progress reporting with physics metrics (less frequent for speed)
            if episode % self.config.log_frequency == 0:
                self._report_progress(episode, episode_reward, episode_steps, 
                                    avg_reward, training_start)
            
            # Evaluation with reduced frequency for speed
            if episode > 0 and episode % self.config.evaluation_frequency == 0:
                evaluate_and_save_best(self.agent, episode, self.best_scores, 'models')
            
            # More aggressive cleanup for speed
            if episode % 200 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            episode += 1
        
        # Training complete
        training_end = time.time()
        total_time = training_end - training_start
        
        create_training_summary(episode, total_time, self.best_scores)
        self._log_final_physics_metrics()
        self.writer.close()
        
        return self.agent, self.best_scores
    
    def _run_episode(self, episode):
        """Run a single training episode with physics monitoring"""
        reset_simulation()
        obs = get_observation()
        episode_reward = 0.0
        episode_steps = 0
        
        # Reset actor history for new episode
        self.agent.actor.reset_dt_history()
        
        for step in range(self.config.max_steps_per_episode):
            # Select and apply action
            action = self.agent.select_action(obs, deterministic=False)
            apply_action(action, obs)
            step_simulation()
            
            # Get next observation and reward
            next_obs = get_observation()
            reward = compute_reward(next_obs)
            done = check_done(next_obs)
            
            # Store transition
            self.agent.store_transition(obs, action, reward, next_obs, done)
            self.agent.actor.update_dt_history(obs, action, reward)
            
            # Training updates with controlled physics frequency and SPEED OPTIMIZATIONS
            if (len(self.agent.memory) > max(self.config.batch_size, 1000) and 
                step % self.config.training_frequency == 0):
                
                self.training_cycle_count += 1
                
                # Speed optimization: Skip some training cycles when buffer is full and performance is good
                skip_training = False
                if (len(self.agent.memory) >= self.config.buffer_size * 0.9 and  # Buffer nearly full
                    episode_reward > 15000000 and  # Good performance (use episode_reward instead of avg_reward)
                    self.training_cycle_count % 3 == 0):  # Skip every 3rd training cycle
                    skip_training = True
                
                if not skip_training:
                    # Multiple updates per training session for efficiency
                    for update_idx in range(self.config.updates_per_training):
                        # Determine if this is a physics update cycle
                        update_physics = (self.training_cycle_count % self.physics_update_frequency == 0)
                        
                        losses = self.agent.update(
                            batch_size=self.config.batch_size,
                            update_physics_components=update_physics
                        )
                        
                        # Track physics and safety losses only when they're updated
                        if losses and update_physics:
                            self._track_physics_losses(losses)
                        
                        # Log losses less frequently to avoid overhead
                        if (losses and episode % 500 == 0 and 
                            step % (self.config.training_frequency * 4) == 0):
                            self._log_losses_with_physics(losses, episode * self.config.max_steps_per_episode + step)
            
            # Update episode tracking
            episode_reward += reward
            episode_steps += 1
            obs = next_obs
            
            # Early episode termination for speed when performance is excellent
            if (episode_steps > 200 and  # Give episode time to develop
                episode_reward > 15000000 and  # High reward achieved
                len(self.agent.memory) >= self.config.buffer_size * 0.8):  # Sufficient experience
                # Early terminate successful episodes to speed up training
                if episode_steps % 100 == 0:  # Check every 100 steps
                    recent_reward_rate = episode_reward / episode_steps
                    if recent_reward_rate > 50000:  # Very high reward rate
                        break
            
            if done:
                break
        
        return episode_reward, episode_steps
    
    def _track_physics_losses(self, losses):
        """Track physics-informed component losses"""
        if 'pinn_loss' in losses:
            self.physics_loss_history.append(losses['pinn_loss'])
        if 'safety_loss' in losses:
            self.safety_loss_history.append(losses['safety_loss'])
    
    def _log_episode(self, episode, episode_reward, episode_steps, avg_reward):
        """Log episode metrics to tensorboard including physics metrics"""
        if episode % self.config.log_frequency == 0:
            self.writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
            self.writer.add_scalar('Training/Average_Reward', avg_reward, episode)
            self.writer.add_scalar('Training/Episode_Length', episode_steps, episode)
            
            # Physics-informed metrics
            if len(self.physics_loss_history) > 0:
                avg_physics_loss = np.mean(list(self.physics_loss_history)[-100:])
                self.writer.add_scalar('Physics/PINN_Loss', avg_physics_loss, episode)
            
            if len(self.safety_loss_history) > 0:
                avg_safety_loss = np.mean(list(self.safety_loss_history)[-100:])
                self.writer.add_scalar('Safety/CBF_Loss', avg_safety_loss, episode)
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                self.writer.add_scalar('Training/GPU_Memory', gpu_memory, episode)
    
    def _log_losses_with_physics(self, losses, step):
        """Log training losses including physics components to tensorboard"""
        for key, value in losses.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                if key == 'pinn_loss':
                    self.writer.add_scalar('Physics/PINN_Training_Loss', value, step)
                elif key == 'operator_loss':
                    self.writer.add_scalar('Physics/Neural_Operator_Loss', value, step)
                elif key == 'safety_loss':
                    self.writer.add_scalar('Safety/CBF_Training_Loss', value, step)
                else:
                    self.writer.add_scalar(f'Loss/{key}', value, step)
    
    def _report_progress(self, episode, episode_reward, episode_steps, 
                        avg_reward, training_start):
        """Report training progress including physics metrics and timing"""
        elapsed = time.time() - training_start
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        
        # Target tracking - FIXED calculation for 5K episodes
        target_time_hours = 4.0  # Realistic target for 5K episodes
        target_eps_per_sec = self.config.num_episodes / (target_time_hours * 3600)
        progress_percent = (episode + 1) / self.config.num_episodes * 100
        
        if eps_per_sec > 0:
            estimated_total_time = self.config.num_episodes / eps_per_sec / 3600
            eta_hours = (self.config.num_episodes - episode) / (eps_per_sec * 3600)
        else:
            estimated_total_time = 0
            eta_hours = 0
        
        # Memory and performance stats
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_percent = (gpu_memory / total_vram) * 100
        else:
            gpu_memory = 0.0
            gpu_percent = 0.0
        
        # Buffer and training status
        buffer_size = len(self.agent.memory)
        buffer_percent = (buffer_size / self.config.buffer_size) * 100
        
        # Training phase indicators
        if buffer_size < 10000:
            training_phase = "Exploration"
        elif buffer_size < 30000:
            training_phase = "Early Learning"
        elif buffer_size < 100000:
            training_phase = "Active Learning"
        else:
            training_phase = "Advanced Learning"
        
        # Recent reward trend
        recent_rewards = list(self.reward_history)[-10:] if len(self.reward_history) >= 10 else list(self.reward_history)
        recent_avg = np.mean(recent_rewards) if recent_rewards else avg_reward
        
        # Physics component status with phase information
        physics_status = "Inactive"
        safety_status = "Inactive"
        current_physics_weight = getattr(self.agent, 'physics_loss_weight', 0)
        current_safety_weight = getattr(self.agent, 'safety_loss_weight', 0)
        current_frequency = getattr(self.agent, 'current_physics_update_frequency', 0)
        current_phase = getattr(self.agent, 'current_phase', "Unknown")
        episode_progress = getattr(self.agent, 'episode_progress', 0) * 100
        
        if len(self.physics_loss_history) > 5:
            recent_physics_loss = np.mean(list(self.physics_loss_history)[-5:])
            physics_status = f"Learning (Loss: {recent_physics_loss:.4f})"
        elif buffer_size > self.agent.min_buffer_for_physics:
            physics_status = "Starting..."
        
        if len(self.safety_loss_history) > 5:
            recent_safety_loss = np.mean(list(self.safety_loss_history)[-5:])
            safety_status = f"Learning (Loss: {recent_safety_loss:.4f})"
        elif buffer_size > self.agent.min_buffer_for_physics:
            safety_status = "Starting..."
        
        # Time tracking indicators
        speed_ratio = eps_per_sec / target_eps_per_sec if target_eps_per_sec > 0 else 0
        if speed_ratio >= 1.0:
            time_status = "✅ ON TARGET"
        elif speed_ratio >= 0.8:
            time_status = "⚠️ SLIGHTLY SLOW"
        else:
            time_status = "❌ TOO SLOW"
        
        print(f"Episode {episode}, Reward: {episode_reward:.0f}, Steps: {episode_steps}, "
            f"Avg: {avg_reward:.0f}, Recent: {recent_avg:.0f}, "
            f"Speed: {eps_per_sec:.3f} eps/sec (Target: {target_eps_per_sec:.3f}, Ratio: {speed_ratio:.2f}x), "
            f"Progress: {progress_percent:.1f}%, ETA: {eta_hours:.1f}h, {time_status}")
        
        # Secondary line with system info
        print(f"         GPU: {gpu_memory:.1f}GB/{total_vram:.1f}GB ({gpu_percent:.1f}%), "
            f"Buffer: {buffer_size:,}/{self.config.buffer_size:,} ({buffer_percent:.1f}%), "
            f"Phase: {training_phase}")
        
        # Physics component status (less frequent to avoid clutter)
        if episode % (self.config.log_frequency * 4) == 0:
            # Corrected physics update ratio calculation
            total_cycles = self.agent.total_training_cycles
            physics_updates = self.agent.physics_update_count
            physics_ratio = (physics_updates / total_cycles * 100) if total_cycles > 0 else 0
            
            print(f"  {current_phase} (Progress: {episode_progress:.1f}%)")
            print(f"  Physics: {physics_status} | Safety: {safety_status}")  
            print(f"  Weights: P={current_physics_weight:.4f}, S={current_safety_weight:.4f} | Freq: every {current_frequency} cycles")
            print(f"  Updates: {physics_updates}/{total_cycles} cycles ({physics_ratio:.1f}%)")

    def _log_final_physics_metrics(self):
        """Log final physics-informed training metrics"""
        print("\nPhysics-Informed Training Analysis:")
        print("=" * 50)
        
        if len(self.physics_loss_history) > 10:  # Need at least 10 data points
            final_physics_loss = np.mean(list(self.physics_loss_history)[-50:])  # Average last 50
            initial_physics_loss = np.mean(list(self.physics_loss_history)[:50])  # Average first 50
            
            if initial_physics_loss > 1e-6:  # Avoid division by very small numbers
                physics_improvement = (initial_physics_loss - final_physics_loss) / abs(initial_physics_loss) * 100
            else:
                physics_improvement = 0.0
                
            print(f"AdaptivePINN Dynamics Learning:")
            print(f"  Initial avg physics loss: {initial_physics_loss:.6f}")
            print(f"  Final avg physics loss: {final_physics_loss:.6f}")
            print(f"  Physics improvement: {physics_improvement:.1f}%")
            print(f"  Physics data points: {len(self.physics_loss_history)}")
        else:
            print(f"AdaptivePINN Dynamics Learning:")
            print(f"  Insufficient data for improvement calculation")
            print(f"  Physics data points: {len(self.physics_loss_history)}")
            if len(self.physics_loss_history) > 0:
                final_physics_loss = np.mean(list(self.physics_loss_history))
                print(f"  Average physics loss: {final_physics_loss:.6f}")
        
        if len(self.safety_loss_history) > 10:  # Need at least 10 data points
            final_safety_loss = np.mean(list(self.safety_loss_history)[-50:])   # Average last 50
            initial_safety_loss = np.mean(list(self.safety_loss_history)[:50])  # Average first 50
            
            if initial_safety_loss > 1e-6:  # Avoid division by very small numbers
                safety_improvement = (initial_safety_loss - final_safety_loss) / abs(initial_safety_loss) * 100
            else:
                safety_improvement = 0.0
                
            print(f"Control Barrier Function Safety:")
            print(f"  Initial avg safety loss: {initial_safety_loss:.6f}")
            print(f"  Final avg safety loss: {final_safety_loss:.6f}")
            print(f"  Safety improvement: {safety_improvement:.1f}%")
            print(f"  Safety data points: {len(self.safety_loss_history)}")
        else:
            print(f"Control Barrier Function Safety:")
            print(f"  Insufficient data for improvement calculation")
            print(f"  Safety data points: {len(self.safety_loss_history)}")
            if len(self.safety_loss_history) > 0:
                final_safety_loss = np.mean(list(self.safety_loss_history))
                print(f"  Average safety loss: {final_safety_loss:.6f}")
        
        # Corrected final physics update statistics
        total_cycles = self.agent.total_training_cycles
        physics_updates = self.agent.physics_update_count
        physics_ratio = (physics_updates / total_cycles * 100) if total_cycles > 0 else 0
        
        print(f"Physics Update Statistics:")
        print(f"  Total training cycles: {total_cycles}")
        print(f"  Physics updates: {physics_updates}")
        print(f"  Physics update ratio: {physics_ratio:.1f}%")
        
        # Expected vs actual frequency analysis
        if hasattr(self.agent, 'current_physics_update_frequency'):
            expected_ratio = 100.0 / self.agent.current_physics_update_frequency
            print(f"  Expected ratio (final frequency): {expected_ratio:.1f}%")
            
        if hasattr(self.agent, 'current_phase'):
            print(f"  Final phase: {self.agent.current_phase}")
            
        # Analysis of why physics improvement might be low
        if len(self.physics_loss_history) < 50:
            print(f"  Note: Limited physics data for reliable improvement calculation")
        
        print("=" * 50)

def create_trainer(config):
    """Factory function to create PIATSG trainer"""
    return PIATSGTrainer(config)