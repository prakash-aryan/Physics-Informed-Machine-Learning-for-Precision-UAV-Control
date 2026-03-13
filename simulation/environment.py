"""
PIATSG Framework - MuJoCo Simulation Environment
Physics-Informed Adaptive Transformers with Safety Guarantees

MuJoCo simulation setup and control functions for UAV dynamics.
"""

import os
import threading
import numpy as np
import mujoco
import mujoco.viewer

# Global simulation variables
model = None
data = None
sim_lock = threading.Lock()
training_active = True

def initialize_simulation():
    """Initialize MuJoCo simulation with enhanced physics settings"""
    global model, data
    
    # Look for XML files in assets directory
    asset_dir = os.path.join(os.path.dirname(__file__), 'assets')
    
    xml_path = os.path.join(asset_dir, "scene_sac.xml")
    if not os.path.exists(xml_path):
        xml_path = os.path.join(asset_dir, "scene_piatsg.xml")
    if not os.path.exists(xml_path):
        # Fallback to current directory
        xml_path = "scene_sac.xml"
        if not os.path.exists(xml_path):
            xml_path = "scene_piatsg.xml"
    
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Cannot find scene XML file. Looked for: {xml_path}")
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Enhanced physics settings for better physics scores
    model.opt.timestep = 0.001  # Balanced timestep
    model.opt.iterations = 200  # More iterations for better physics
    model.opt.tolerance = 1e-7  # Tight tolerance for physics accuracy
    
    print(f"MuJoCo model loaded: {model.nq} pos, {model.nv} vel, {model.nu} actuators")
    return model, data

def reset_simulation(randomize=True):
    """Reset simulation state with better initial conditions"""
    global data
    with sim_lock:
        mujoco.mj_resetData(model, data)
        
        # Initialize at better starting conditions
        if model.nq >= 7:
            if randomize:
                data.qpos[0] = np.random.uniform(-0.03, 0.03)  # x - small variation
                data.qpos[1] = np.random.uniform(-0.03, 0.03)  # y - small variation
                data.qpos[2] = np.random.uniform(0.95, 1.05)   # z - close to target
                data.qpos[3] = 1.0  # qw - upright
                data.qpos[4:7] = np.random.uniform(-0.01, 0.01, 3)  # small quaternion perturbation
            else:
                data.qpos[0] = 0.0
                data.qpos[1] = 0.0
                data.qpos[2] = 1.0  # Start exactly at target
                data.qpos[3] = 1.0
                data.qpos[4:7] = np.zeros(3)
        
        # Start with minimal velocities
        data.qvel[:] = (np.random.uniform(-0.01, 0.01, data.qvel.shape[0]) 
                       if randomize else np.zeros(data.qvel.shape[0]))
        
        # Better initial hover thrust
        data.ctrl[0] = 0.268  # Base hover thrust
        data.ctrl[1:4] = 0.0  # No torques initially
        
        mujoco.mj_forward(model, data)

def get_observation():
    """Get enhanced 18-dimensional observation from MuJoCo"""
    with sim_lock:
        # Position (3) + quaternion (4) + velocity (3) + angular velocity (3) + controls (4) + target (1)
        obs = np.zeros(18)
        
        if model.nq >= 7:
            obs[0:3] = data.qpos[0:3]  # position
            obs[3:7] = data.qpos[3:7]  # quaternion
        
        if model.nv >= 6:
            obs[7:10] = data.qvel[0:3]  # linear velocity
            obs[10:13] = data.qvel[3:6]  # angular velocity
        
        obs[13:17] = data.ctrl[0:4]  # current controls
        obs[17] = 1.0  # target altitude
        
        return obs

def step_simulation():
    """Step the simulation forward"""
    with sim_lock:
        mujoco.mj_step(model, data)

def apply_action(action, obs):
    """Apply action to simulation with stability considerations"""
    with sim_lock:
        # Better base hover thrust and safer action processing
        base_thrust = 0.272  # Increased for better hovering
        
        # Conservative action processing
        thrust_adjustment = np.clip(action[0], -0.06, 0.06)  # Small adjustments only
        roll_torque = np.clip(action[1], -0.15, 0.15)       # Reduced
        pitch_torque = np.clip(action[2], -0.15, 0.15)      # Reduced
        yaw_torque = np.clip(action[3], -0.10, 0.10)        # Reduced
        
        # Ensure thrust never goes too low
        final_thrust = base_thrust + thrust_adjustment
        final_thrust = np.clip(final_thrust, 0.25, 0.32)  # Force reasonable thrust range
        
        # Apply controls
        data.ctrl[0] = final_thrust
        data.ctrl[1] = roll_torque
        data.ctrl[2] = pitch_torque
        data.ctrl[3] = yaw_torque
        
        # Debug: Check for problematic control values
        current_pos = obs[0:3] if len(obs) >= 3 else [0, 0, 0]
        
        # Only warn about crashes occasionally and when thrust is actually problematic
        if (current_pos[2] < 0.6 and final_thrust < 0.26 and np.random.random() < 0.01):
            action_magnitude = np.linalg.norm(action)
            print(f"    Warning: Low altitude {current_pos[2]:.3f}m, "
                  f"final_thrust={final_thrust:.3f}, raw_action[0]={action[0]:.3f}, "
                  f"action_mag={action_magnitude:.3f}")

def compute_reward(obs):
    """Compute reward with enhanced guidance for training"""
    position = obs[0:3]
    quaternion = obs[3:7]
    velocity = obs[7:10]
    angular_velocity = obs[10:13]
    
    target_pos = np.array([0.0, 0.0, 1.0])
    distance = np.linalg.norm(position - target_pos)
    
    # Precision reward (exponential falloff for tight control)
    precision_reward = 15000 * np.exp(-25 * distance**2)
    
    # Physics reward (smooth motion)
    vel_magnitude = np.linalg.norm(velocity)
    angular_vel_magnitude = np.linalg.norm(angular_velocity)
    physics_reward = (8000 * np.exp(-10 * vel_magnitude**2) * 
                     np.exp(-5 * angular_vel_magnitude**2))
    
    # Safety reward (orientation and bounds)
    qw, qx, qy, qz = quaternion
    qw = np.clip(qw, -1, 1)
    tilt_penalty = 5000 * (1 - abs(qw))  # Penalize tilting
    
    # Progressive altitude penalties
    altitude_penalty = 0
    if position[2] < 0.3:  # Very low - immediate danger
        altitude_penalty = -25000
    elif position[2] < 0.5:  # Low but not critical
        altitude_penalty = -8000
    elif position[2] > 2.0:  # Too high
        altitude_penalty = -15000
    elif position[2] > 1.5:  # Moderately high
        altitude_penalty = -5000
    
    # Position bounds safety  
    position_penalty = 0
    if abs(position[0]) > 1.2 or abs(position[1]) > 1.2:
        position_penalty = -10000
    elif abs(position[0]) > 1.0 or abs(position[1]) > 1.0:
        position_penalty = -3000
    
    # Training phase bonuses
    survival_bonus = 0
    if position[2] > 0.6:  # Staying airborne
        survival_bonus = 5000
    if position[2] > 0.8:  # Getting closer to target region
        survival_bonus = 8000
    
    # Altitude bonus for being near 1.0m
    altitude_bonus = 0
    if 0.7 <= position[2] <= 1.3:  # Good region
        altitude_bonus = 12000 * np.exp(-20 * (position[2] - 1.0)**2)
    
    # Hovering bonus - reward for staying in good region
    hovering_bonus = 0
    if distance < 0.2 and vel_magnitude < 0.3:  # Close and stable
        hovering_bonus = 8000
    elif distance < 0.5 and vel_magnitude < 0.5:  # Reasonable
        hovering_bonus = 3000
    
    total_reward = (precision_reward + physics_reward - tilt_penalty + 
                   altitude_penalty + position_penalty + altitude_bonus + 
                   hovering_bonus + survival_bonus)
    
    # Debug very low rewards occasionally
    if total_reward < -20000 and np.random.random() < 0.001:
        print(f"    Debug: Low reward {total_reward:.0f}, pos={position}, "
              f"alt_pen={altitude_penalty}, dist={distance:.3f}")
    
    return total_reward

def check_done(obs):
    """Check if episode should terminate"""
    position = obs[0:3]
    
    # Terminal conditions - only for severe violations
    if position[2] < 0.2:  # Very low altitude
        return True
    if position[2] > 2.5:  # Very high altitude
        return True
    if abs(position[0]) > 1.5 or abs(position[1]) > 1.5:  # Far position bounds
        return True
    
    return False

def launch_viewer():
    """Launch MuJoCo viewer for visualization"""
    global training_active
    
    if model is None or data is None:
        raise RuntimeError("Simulation must be initialized before launching viewer")
    
    with mujoco.viewer.launch_passive(model, data) as viewer_handle:
        while viewer_handle.is_running() and training_active:
            with sim_lock:
                viewer_handle.sync()
            import time
            time.sleep(0.01)
    
    training_active = False

def cleanup_simulation():
    """Clean up simulation resources"""
    global training_active
    training_active = False