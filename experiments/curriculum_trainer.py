"""
Curriculum Learning Training Script
Progressively trains PIATSG on increasing episode horizons: 5s → 15s → 30s → 60s → 120s
Addresses long-term stability issues identified in extended horizon evaluation
"""

import os
import sys
import json
import time
import torch
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.environment import (
    initialize_simulation, reset_simulation, get_observation,
    step_simulation, apply_action, compute_reward, check_done
)
from core.agent import PIATSGAgent
from utils.config import configure_device, TrainingConfig
from training.evaluation import evaluate_agent


class CurriculumStage:
    """Represents a curriculum training stage"""
    
    def __init__(
        self,
        name: str,
        episode_length: float,
        num_episodes: int,
        success_threshold: float,
        evaluation_episodes: int = 10
    ):
        self.name = name
        self.episode_length = episode_length  # seconds
        self.num_episodes = num_episodes
        self.success_threshold = success_threshold  # mean RMSE threshold
        self.evaluation_episodes = evaluation_episodes
        self.dt = 0.01  # 100Hz
        self.max_steps = int(episode_length / self.dt)
        
    def __repr__(self):
        return (f"Stage(name={self.name}, length={self.episode_length}s, "
                f"episodes={self.num_episodes}, threshold={self.success_threshold}m)")


class CurriculumTrainer:
    """Curriculum learning trainer for extended horizon training"""
    
    def __init__(
        self,
        output_dir: str = "curriculum_training",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_checkpoint: str = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        self.dt = 0.01
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "evaluations").mkdir(exist_ok=True)
        
        # Define curriculum stages
        self.stages = [
            CurriculumStage("stage1_5s", 5.0, 1000, 0.05),      # Baseline
            CurriculumStage("stage2_15s", 15.0, 1500, 0.15),    # 3x extension
            CurriculumStage("stage3_30s", 30.0, 2000, 0.35),    # 6x extension
            CurriculumStage("stage4_60s", 60.0, 2500, 0.65),    # 12x extension
            CurriculumStage("stage5_120s", 120.0, 3000, 1.0),   # 24x extension
        ]
        
        self.current_stage_idx = 0
        self.total_episodes = 0
        
        # Initialize simulation
        initialize_simulation()
        
        # Initialize agent
        device_config, batch_size, buffer_size, _ = configure_device()
        self.config = TrainingConfig(device_config, batch_size, buffer_size)
        
        if load_checkpoint:
            print(f"Loading checkpoint from {load_checkpoint}...")
            self.agent = self._load_checkpoint(load_checkpoint)
        else:
            self.agent = PIATSGAgent(self.config)
        
        # Training state
        self.stage_history = []
        self.best_stage_performance = {}
        
        print("\n" + "="*80)
        print("CURRICULUM LEARNING TRAINER INITIALIZED")
        print("="*80)
        print(f"\nCurriculum stages:")
        for i, stage in enumerate(self.stages, 1):
            print(f"  {i}. {stage}")
        print(f"\nOutput directory: {self.output_dir}")
        print("="*80 + "\n")
    
    def train(self):
        """Execute full curriculum training"""
        
        training_start = time.time()
        
        for stage_idx, stage in enumerate(self.stages):
            self.current_stage_idx = stage_idx
            
            print("\n" + "="*80)
            print(f"STAGE {stage_idx + 1}/{len(self.stages)}: {stage.name.upper()}")
            print("="*80)
            print(f"Episode length: {stage.episode_length}s")
            print(f"Training episodes: {stage.num_episodes}")
            print(f"Success threshold: {stage.success_threshold}m RMSE")
            print("="*80 + "\n")
            
            # Train this stage
            stage_start = time.time()
            stage_results = self._train_stage(stage)
            stage_duration = time.time() - stage_start
            
            # Record stage results
            stage_results['duration_hours'] = stage_duration / 3600
            self.stage_history.append(stage_results)
            
            # Save stage checkpoint
            self._save_checkpoint(stage, stage_results)
            
            # Evaluate on all previous horizons
            self._comprehensive_evaluation(stage)
            
            # Check if stage succeeded
            if stage_results['final_rmse'] <= stage.success_threshold:
                print(f"\n✅ Stage {stage_idx + 1} SUCCEEDED!")
                print(f"   Final RMSE: {stage_results['final_rmse']:.4f}m")
                print(f"   Threshold: {stage.success_threshold}m")
                print(f"   Duration: {stage_duration/3600:.2f} hours")
            else:
                print(f"\n⚠️  Stage {stage_idx + 1} completed but did not meet threshold")
                print(f"   Final RMSE: {stage_results['final_rmse']:.4f}m")
                print(f"   Threshold: {stage.success_threshold}m")
                print(f"   Continuing to next stage anyway...")
        
        # Training complete
        training_duration = time.time() - training_start
        
        print("\n" + "="*80)
        print("CURRICULUM TRAINING COMPLETE")
        print("="*80)
        print(f"Total training time: {training_duration/3600:.2f} hours")
        print(f"Total episodes: {self.total_episodes}")
        
        # Save final training summary
        self._save_training_summary(training_duration)
        
        return self.agent, self.stage_history
    
    def _train_stage(self, stage: CurriculumStage) -> dict:
        """Train a single curriculum stage"""
        
        episode_rewards = []
        episode_rmses = []
        episode_completions = []
        
        stage_start_episode = self.total_episodes
        
        for episode_idx in range(stage.num_episodes):
            self.total_episodes += 1
            
            # Run episode
            episode_reward, episode_steps, episode_rmse, completed = self._run_episode(
                stage.max_steps,
                stage.episode_length
            )
            
            episode_rewards.append(episode_reward)
            episode_rmses.append(episode_rmse)
            episode_completions.append(completed)
            
            # Periodic logging
            if (episode_idx + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_rmse = np.mean(episode_rmses[-50:])
                completion_rate = np.mean(episode_completions[-50:]) * 100
                
                print(f"Episode {episode_idx + 1}/{stage.num_episodes} | "
                      f"Reward: {avg_reward:.0f} | "
                      f"RMSE: {avg_rmse:.4f}m | "
                      f"Completion: {completion_rate:.1f}%")
            
            # Save intermediate checkpoint
            if (episode_idx + 1) % 500 == 0:
                self._save_intermediate_checkpoint(stage, episode_idx + 1)
        
        # Stage summary
        results = {
            'stage_name': stage.name,
            'episode_length': stage.episode_length,
            'episodes_trained': stage.num_episodes,
            'final_rmse': np.mean(episode_rmses[-100:]),
            'best_rmse': np.min(episode_rmses),
            'avg_reward': np.mean(episode_rewards[-100:]),
            'completion_rate': np.mean(episode_completions[-100:]),
            'rmse_history': episode_rmses,
            'reward_history': episode_rewards
        }
        
        return results
    
    def _run_episode(self, max_steps: int, episode_length: float) -> tuple:
        """Run single training episode"""
        
        # Reset with slight randomization for robustness
        reset_simulation(randomize=True)
        state = get_observation()
        
        episode_reward = 0.0
        episode_positions = []
        reference_position = np.array([0.0, 0.0, 1.0])
        
        # Reset actor history
        self.agent.actor.reset_dt_history()
        
        for step in range(max_steps):
            # Select action
            action = self.agent.select_action(state, deterministic=False)
            
            # Apply and step
            apply_action(action, state)
            step_simulation()
            
            # Get next state and reward
            next_state = get_observation()
            reward = compute_reward(next_state)
            done = check_done(next_state)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Update actor history
            self.agent.actor.update_dt_history(state, action, reward)
            
            # Training updates
            if (len(self.agent.memory) > max(self.config.batch_size, 1000) and
                step % self.config.training_frequency == 0):
                
                for _ in range(self.config.updates_per_training):
                    update_physics = (self.agent.total_training_cycles % 2 == 0)
                    self.agent.update(
                        batch_size=self.config.batch_size,
                        update_physics_components=update_physics
                    )
            
            # Track position for RMSE
            episode_positions.append(next_state[:3])
            
            episode_reward += reward
            state = next_state
            
            # Check early termination
            if done:
                completed = False
                break
        else:
            completed = True
        
        # Compute episode RMSE
        positions = np.array(episode_positions)
        errors = np.linalg.norm(positions - reference_position, axis=1)
        episode_rmse = np.sqrt(np.mean(errors ** 2))
        
        return episode_reward, step + 1, episode_rmse, completed
    
    def _comprehensive_evaluation(self, current_stage: CurriculumStage):
        """Evaluate on all horizons up to current stage"""
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EVALUATION - Stage: {current_stage.name}")
        print(f"{'='*80}")
        
        horizons_to_test = [s.episode_length for s in self.stages 
                           if s.episode_length <= current_stage.episode_length]
        
        eval_results = {}
        
        for horizon in horizons_to_test:
            print(f"\nEvaluating {horizon}s horizon...")
            
            rmses = []
            completions = []
            
            for eval_ep in range(10):
                reset_simulation(randomize=False)
                state = get_observation()
                self.agent.actor.reset_dt_history()
                
                positions = []
                max_steps = int(horizon / self.dt)
                
                for step in range(max_steps):
                    with torch.no_grad():
                        action = self.agent.select_action(state, deterministic=True)
                    
                    apply_action(action, state)
                    step_simulation()
                    
                    next_state = get_observation()
                    done = check_done(next_state)
                    
                    positions.append(next_state[:3])
                    state = next_state
                    
                    if done:
                        completions.append(False)
                        break
                else:
                    completions.append(True)
                
                # Compute RMSE
                positions = np.array(positions)
                reference = np.array([0.0, 0.0, 1.0])
                errors = np.linalg.norm(positions - reference, axis=1)
                rmse = np.sqrt(np.mean(errors ** 2))
                rmses.append(rmse)
            
            avg_rmse = np.mean(rmses)
            completion_rate = np.mean(completions) * 100
            
            print(f"  RMSE: {avg_rmse:.4f}m ± {np.std(rmses):.4f}m")
            print(f"  Completion rate: {completion_rate:.1f}%")
            
            eval_results[f'{horizon}s'] = {
                'rmse_mean': avg_rmse,
                'rmse_std': np.std(rmses),
                'completion_rate': completion_rate
            }
        
        # Save evaluation results
        eval_file = self.output_dir / "evaluations" / f"{current_stage.name}_eval.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"\nEvaluation results saved to: {eval_file}")
        
        return eval_results
    
    def _save_checkpoint(self, stage: CurriculumStage, results: dict):
        """Save stage checkpoint"""
        
        checkpoint_path = self.output_dir / "checkpoints" / f"{stage.name}_final.pth"
        
        checkpoint = {
            'actor': self.agent.actor.state_dict(),
            'critic1': self.agent.critic1.state_dict(),
            'critic2': self.agent.critic2.state_dict(),
            'pinn': self.agent.adaptive_pinn.state_dict(),
            'operator': self.agent.neural_operator.state_dict(),
            'safety': self.agent.safety_constraint.state_dict(),
            'stage_name': stage.name,
            'episode_length': stage.episode_length,
            'total_episodes': self.total_episodes,
            'results': results
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_intermediate_checkpoint(self, stage: CurriculumStage, episode_num: int):
        """Save intermediate checkpoint during stage"""
        
        checkpoint_path = self.output_dir / "checkpoints" / f"{stage.name}_ep{episode_num}.pth"
        
        checkpoint = {
            'actor': self.agent.actor.state_dict(),
            'stage_name': stage.name,
            'episode_length': stage.episode_length,
            'total_episodes': self.total_episodes,
            'stage_episode': episode_num
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load agent from checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Initialize agent
        device_config, batch_size, buffer_size, _ = configure_device()
        config = TrainingConfig(device_config, batch_size, buffer_size)
        agent = PIATSGAgent(config)
        
        # Load state dicts
        if 'actor' in checkpoint:
            agent.actor.load_state_dict(checkpoint['actor'])
        if 'critic1' in checkpoint:
            agent.critic1.load_state_dict(checkpoint['critic1'])
        if 'critic2' in checkpoint:
            agent.critic2.load_state_dict(checkpoint['critic2'])
        if 'pinn' in checkpoint:
            agent.adaptive_pinn.load_state_dict(checkpoint['pinn'])
        if 'operator' in checkpoint:
            agent.neural_operator.load_state_dict(checkpoint['operator'])
        if 'safety' in checkpoint:
            agent.safety_constraint.load_state_dict(checkpoint['safety'])
        
        if 'total_episodes' in checkpoint:
            self.total_episodes = checkpoint['total_episodes']
        
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
        if 'stage_name' in checkpoint:
            print(f"   Stage: {checkpoint['stage_name']}")
            print(f"   Episode length: {checkpoint.get('episode_length', 'unknown')}s")
            print(f"   Total episodes: {self.total_episodes}")
        
        return agent
    
    def _save_training_summary(self, total_duration: float):
        """Save comprehensive training summary"""
        
        summary = {
            'total_duration_hours': total_duration / 3600,
            'total_episodes': self.total_episodes,
            'stages': self.stage_history,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Training summary saved: {summary_file}")
        
        # Also create human-readable summary
        text_summary = self.output_dir / "training_summary.txt"
        with open(text_summary, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CURRICULUM TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total Duration: {total_duration/3600:.2f} hours\n")
            f.write(f"Total Episodes: {self.total_episodes}\n")
            f.write(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Stage Results:\n")
            f.write("-"*80 + "\n")
            
            for i, stage_result in enumerate(self.stage_history, 1):
                f.write(f"\nStage {i}: {stage_result['stage_name']}\n")
                f.write(f"  Episode Length: {stage_result['episode_length']}s\n")
                f.write(f"  Episodes Trained: {stage_result['episodes_trained']}\n")
                f.write(f"  Final RMSE: {stage_result['final_rmse']:.4f}m\n")
                f.write(f"  Best RMSE: {stage_result['best_rmse']:.4f}m\n")
                f.write(f"  Avg Reward: {stage_result['avg_reward']:.0f}\n")
                f.write(f"  Completion Rate: {stage_result['completion_rate']*100:.1f}%\n")
                f.write(f"  Duration: {stage_result['duration_hours']:.2f} hours\n")
        
        print(f"📄 Text summary saved: {text_summary}")


def main():
    """Main execution"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Curriculum Learning Training")
    parser.add_argument('--output', type=str, default='curriculum_training',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CurriculumTrainer(
        output_dir=args.output,
        device=args.device,
        load_checkpoint=args.load_checkpoint
    )
    
    # Run curriculum training
    try:
        agent, history = trainer.train()
        print("\n✅ Curriculum training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Checkpoints have been saved")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()