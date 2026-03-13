"""
PIATSG Framework - Agent Evaluation
Physics-Informed Adaptive Transformers with Safety Guarantees

Comprehensive evaluation metrics for PIATSG agent performance.
"""

import numpy as np
from simulation.environment import reset_simulation, get_observation, apply_action, step_simulation, check_done

def evaluate_agent(agent, episodes=5):
    """Evaluate agent performance with comprehensive metrics"""
    results = {
        'precisions_10cm': [],
        'precisions_5cm': [],
        'precisions_2cm': [],
        'physics_scores': [],
        'safety_scores': []
    }
    
    for ep in range(episodes):
        # Add slight randomization to avoid identical episodes
        reset_simulation(randomize=True)
        obs = get_observation()
        episode_length = 0
        max_episode_length = 1000
        
        positions = []
        velocities = []
        actions_taken = []
        angular_velocities = []
        quaternions = []
        
        for step in range(max_episode_length):
            action = agent.select_action(obs, deterministic=True)
            actions_taken.append(action.copy())
            apply_action(action, obs)
            step_simulation()
            obs = get_observation()
            
            positions.append(obs[0:3].copy())
            velocities.append(obs[7:10].copy())
            angular_velocities.append(obs[10:13].copy())
            quaternions.append(obs[3:7].copy())
            episode_length += 1
            
            if check_done(obs):
                break
        
        # Calculate metrics
        positions = np.array(positions)
        velocities = np.array(velocities)
        angular_velocities = np.array(angular_velocities)
        quaternions = np.array(quaternions)
        actions_taken = np.array(actions_taken)
        target_pos = np.array([0.0, 0.0, 1.0])
        
        distances = np.linalg.norm(positions - target_pos, axis=1)
        
        # Precision metrics (percentage of time within distance thresholds)
        precision_10cm = np.mean(distances < 0.1) * 100
        precision_5cm = np.mean(distances < 0.05) * 100
        precision_2cm = np.mean(distances < 0.02) * 100
        
        # Stricter physics score (smooth motion quality)
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        angular_vel_magnitudes = np.linalg.norm(angular_velocities, axis=1)
        
        # Penalize high velocities and angular velocities more strictly
        smooth_velocity = np.mean(vel_magnitudes < 0.3)  # Stricter threshold
        smooth_angular = np.mean(angular_vel_magnitudes < 0.4)  # Stricter threshold
        
        # Check for jerky motion (high acceleration)
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0)
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            smooth_acceleration = np.mean(acc_magnitudes < 0.1)  # New metric
        else:
            smooth_acceleration = 1.0
            
        physics_score = (smooth_velocity * 0.4 + smooth_angular * 0.4 + smooth_acceleration * 0.2) * 100
        
        # Natural safety score (no artificial cap)
        altitude_violations = np.sum((positions[:, 2] <= 0.35) | (positions[:, 2] >= 1.65))
        position_violations = np.sum((np.abs(positions[:, 0]) >= 1.15) | (np.abs(positions[:, 1]) >= 1.15))
        
        # Check for excessive tilt
        qw_values = quaternions[:, 0]
        tilt_violations = np.sum(np.abs(qw_values) < 0.85)  # Stricter tilt check
        
        total_violations = altitude_violations + position_violations + tilt_violations
        violation_rate = total_violations / len(positions)
        
        # Natural safety score calculation
        safety_score = max(0.0, 100.0 * (1 - violation_rate * 2))  # No cap!
        
        results['precisions_10cm'].append(precision_10cm)
        results['precisions_5cm'].append(precision_5cm)
        results['precisions_2cm'].append(precision_2cm)
        results['physics_scores'].append(physics_score)
        results['safety_scores'].append(safety_score)
        
        # Detailed verification output
        final_pos = positions[-1]
        final_distance = np.linalg.norm(final_pos - target_pos)
        action_std = np.std(actions_taken, axis=0)
        action_mean = np.mean(actions_taken, axis=0)
        
        print(f"    Episode {ep} verification:")
        print(f"      Final position: {final_pos}")
        print(f"      Final distance: {final_distance:.3f}m")
        print(f"      Action stats: mean={action_mean}, std={action_std}")
        print(f"      Precision rates: 10cm={precision_10cm:.1f}%, "
              f"5cm={precision_5cm:.1f}%, 2cm={precision_2cm:.1f}%")
        print(f"      Physics: {physics_score:.1f}%, Safety: {safety_score:.1f}%")
    
    # Compute mean results
    results['mean_precision_10cm'] = np.mean(results['precisions_10cm'])
    results['mean_precision_5cm'] = np.mean(results['precisions_5cm']) 
    results['mean_precision_2cm'] = np.mean(results['precisions_2cm'])
    results['mean_physics_score'] = np.mean(results['physics_scores'])
    results['mean_safety_score'] = np.mean(results['safety_scores'])
    
    return results

def print_evaluation_summary(results):
    """Print a summary of evaluation results"""
    print(f"Evaluation Summary:")
    print(f"  Precision (10cm): {results['mean_precision_10cm']:.1f}%")
    print(f"  Super-precision (5cm): {results['mean_precision_5cm']:.1f}%")
    print(f"  Ultra-precision (2cm): {results['mean_precision_2cm']:.1f}%")
    print(f"  Physics score: {results['mean_physics_score']:.1f}%")
    print(f"  Safety score: {results['mean_safety_score']:.1f}%")

def evaluate_and_save_best(agent, episode, best_scores, checkpoint_dir):
    """Evaluate agent and save best models based on different metrics"""
    print(f"Evaluating at episode {episode}...")
    eval_results = evaluate_agent(agent, episodes=5)
    
    print_evaluation_summary(eval_results)
    
    # Track and save best performance
    model_saved = False
    
    if eval_results['mean_precision_10cm'] > best_scores['precision_10cm']:
        best_scores['precision_10cm'] = eval_results['mean_precision_10cm']
        agent.save_model(f'{checkpoint_dir}/best_precision_10cm.pth')
        print(f"New best precision (10cm): {best_scores['precision_10cm']:.1f}%")
        model_saved = True
    
    if eval_results['mean_precision_5cm'] > best_scores['precision_5cm']:
        best_scores['precision_5cm'] = eval_results['mean_precision_5cm']
        agent.save_model(f'{checkpoint_dir}/best_precision_5cm.pth')
        print(f"New best super-precision (5cm): {best_scores['precision_5cm']:.1f}%")
    
    if eval_results['mean_precision_2cm'] > best_scores['precision_2cm']:
        best_scores['precision_2cm'] = eval_results['mean_precision_2cm']
        agent.save_model(f'{checkpoint_dir}/best_precision_2cm.pth')
        print(f"New best ultra-precision (2cm): {best_scores['precision_2cm']:.1f}%")
    
    if eval_results['mean_physics_score'] > best_scores['physics']:
        best_scores['physics'] = eval_results['mean_physics_score']
        agent.save_model(f'{checkpoint_dir}/best_physics.pth')
        print(f"New best physics: {best_scores['physics']:.1f}%")
    
    if eval_results['mean_safety_score'] > best_scores['safety']:
        best_scores['safety'] = eval_results['mean_safety_score']
        agent.save_model(f'{checkpoint_dir}/best_safety.pth')
        print(f"New best safety: {best_scores['safety']:.1f}%")
    
    return eval_results, model_saved

def create_training_summary(episode, total_time, best_scores):
    """Create a comprehensive training summary"""
    print(f"\nTraining Summary:")
    print(f"=" * 60)
    print(f"Episodes completed: {episode}")
    print(f"Total training time: {total_time/3600:.1f} hours")
    print(f"Training speed: {episode/total_time:.2f} episodes/second")
    print(f"\nBest Performance Achieved:")
    print(f"  Precision (10cm): {best_scores['precision_10cm']:.1f}%")
    print(f"  Super-precision (5cm): {best_scores['precision_5cm']:.1f}%") 
    print(f"  Ultra-precision (2cm): {best_scores['precision_2cm']:.1f}%")
    print(f"  Physics score: {best_scores['physics']:.1f}%")
    print(f"  Safety score: {best_scores['safety']:.1f}%")
    print(f"=" * 60)