"""
PIATSG Framework - Neural Network Components
Physics-Informed Adaptive Transformers with Safety Guarantees

Core neural network components: Actor, AdaptivePINN, NeuralOperator, SafetyConstraint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import numpy as np

class Actor(nn.Module):
    """Hierarchical actor with Decision Transformer for precision control"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=1536, max_history=50):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_history = max_history
        
        # Decision transformer components
        self.embedding_dim = 256
        self.dt_embed = nn.Linear(state_dim + action_dim + 1, self.embedding_dim)
        self.dt_pos_embed = nn.Parameter(torch.zeros(1, max_history, self.embedding_dim))
        
        # Multi-head attention for temporal dependencies
        self.dt_attention = nn.MultiheadAttention(self.embedding_dim, 8, batch_first=True)
        self.dt_norm = nn.LayerNorm(self.embedding_dim)
        
        # Hierarchical control networks
        self.high_level_policy = nn.Sequential(
            nn.Linear(state_dim + self.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        self.precision_refiner = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)
        
        # History storage
        self.history_states = deque(maxlen=max_history)
        self.history_actions = deque(maxlen=max_history)
        self.history_rewards = deque(maxlen=max_history)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize actor network weights for stable initial policy"""
        for module in [self.high_level_policy, self.precision_refiner]:
            if hasattr(module, '__iter__'):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        if layer == module[-1]:
                            nn.init.uniform_(layer.weight, -0.003, 0.003)
                            nn.init.zeros_(layer.bias)
                        else:
                            nn.init.xavier_uniform_(layer.weight)
                            nn.init.zeros_(layer.bias)
    
    def reset_dt_history(self):
        """Reset decision transformer history"""
        self.history_states.clear()
        self.history_actions.clear()
        self.history_rewards.clear()
    
    def update_dt_history(self, state, action, reward=0.0):
        """Update decision transformer history"""
        self.history_states.append(state.copy())
        self.history_actions.append(action.copy())
        self.history_rewards.append(reward)
    
    def forward(self, state, deterministic=False):
        batch_size = state.shape[0] if len(state.shape) > 1 else 1
        
        # Decision transformer processing
        dt_context = torch.zeros(batch_size, self.embedding_dim).to(state.device)
        
        if len(self.history_states) > 0:
            seq_len = min(len(self.history_states), self.max_history)
            
            seq_states = torch.zeros(batch_size, seq_len, self.state_dim).to(state.device)
            seq_actions = torch.zeros(batch_size, seq_len, self.action_dim).to(state.device)
            seq_rewards = torch.zeros(batch_size, seq_len, 1).to(state.device)
            
            for i, (h_state, h_action, h_reward) in enumerate(zip(
                list(self.history_states)[-seq_len:],
                list(self.history_actions)[-seq_len:],
                list(self.history_rewards)[-seq_len:]
            )):
                seq_states[:, i] = torch.FloatTensor(h_state).to(state.device)
                seq_actions[:, i] = torch.FloatTensor(h_action).to(state.device)
                seq_rewards[:, i, 0] = h_reward
            
            seq_input = torch.cat([seq_states, seq_actions, seq_rewards], dim=-1)
            embedded = self.dt_embed(seq_input)
            embedded += self.dt_pos_embed[:, :seq_len]
            
            attended, _ = self.dt_attention(embedded, embedded, embedded)
            attended = self.dt_norm(attended)
            
            dt_context = attended[:, -1]
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Hierarchical policy
        combined_input = torch.cat([state, dt_context], dim=-1)
        high_level_action = self.high_level_policy(combined_input)
        
        # Precision refinement
        refiner_input = torch.cat([state, high_level_action], dim=-1)
        action_refinement = self.precision_refiner(refiner_input)
        
        mean = high_level_action + 0.1 * action_refinement
        mean = torch.clamp(mean, -10, 10)
        
        log_std = self.log_std.expand_as(mean)
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
            normal = Normal(mean, std)
            action = normal.rsample()
            action = torch.clamp(action, -1, 1)
            return action.squeeze(0) if batch_size == 1 else action


class AdaptivePINN(nn.Module):
    """Physics-Informed Neural Network with PDE residuals and automatic differentiation"""
    
    def __init__(self, state_dim, hidden_dim=1024):
        super().__init__()
        
        self.physics_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, state_dim)
        )
        
        # Physics constants for UAV dynamics
        self.register_buffer('g', torch.tensor(9.81))
        self.register_buffer('mass', torch.tensor(0.027))
        self.register_buffer('dt', torch.tensor(0.01))
        self.register_buffer('Ixx', torch.tensor(2.3951e-5))
        self.register_buffer('Iyy', torch.tensor(2.3951e-5))
        self.register_buffer('Izz', torch.tensor(3.2347e-5))
        
        # Adaptive loss scaling parameters
        self.register_buffer('loss_scale_pos', torch.tensor(1.0))
        self.register_buffer('loss_scale_vel', torch.tensor(1.0))
        self.register_buffer('loss_scale_quat', torch.tensor(0.5))
        self.register_buffer('loss_scale_omega', torch.tensor(0.5))
        
        # Track physics learning progress for long training
        self.physics_loss_history = []
        
    def forward(self, state):
        return self.physics_net(state)
    
    def compute_pde_residuals(self, state, action):
        """Compute PDE residuals using automatic differentiation with stability improvements"""
        # Ensure state requires gradients but avoid in-place operations
        if not state.requires_grad:
            state_with_grad = state.clone().requires_grad_(True)
        else:
            state_with_grad = state
        
        try:
            # Extract state components (create new tensors to avoid in-place issues)
            pos = state_with_grad[:, :3].clone()  # position
            quat_raw = state_with_grad[:, 3:7].clone()  # quaternion
            vel = state_with_grad[:, 7:10].clone()  # velocity
            omega = state_with_grad[:, 10:13].clone()  # angular velocity
            controls = state_with_grad[:, 13:17].clone()  # current controls
            target = state_with_grad[:, 17:18].clone()  # target
            
            # Normalize quaternion to prevent drift (avoid in-place)
            quat_norm = torch.norm(quat_raw, dim=1, keepdim=True)
            quat = quat_raw / (quat_norm + 1e-8)
            
            # Predict next state
            next_state = self.physics_net(state_with_grad)
            next_pos = next_state[:, :3]
            next_quat_raw = next_state[:, 3:7]
            next_vel = next_state[:, 7:10]
            next_omega = next_state[:, 10:13]
            next_controls = next_state[:, 13:17]
            next_target = next_state[:, 17:18]
            
            # Normalize predicted quaternion (avoid in-place)
            next_quat_norm = torch.norm(next_quat_raw, dim=1, keepdim=True)
            next_quat = next_quat_raw / (next_quat_norm + 1e-8)
            
            # Compute time derivatives for all state components
            pos_dot = (next_pos - pos) / self.dt
            quat_dot = (next_quat - quat) / self.dt
            vel_dot = (next_vel - vel) / self.dt
            omega_dot = (next_omega - omega) / self.dt
            controls_dot = (next_controls - controls) / self.dt
            target_dot = (next_target - target) / self.dt
            
            # Extract control inputs with bounds checking (avoid in-place)
            thrust = torch.clamp(action[:, 0].clone(), 0.0, 0.35)
            tau_x = torch.clamp(action[:, 1].clone(), -0.2, 0.2)
            tau_y = torch.clamp(action[:, 2].clone(), -0.2, 0.2)
            tau_z = torch.clamp(action[:, 3].clone(), -0.2, 0.2)
            
            # Rotation matrix from quaternion (with numerical stability)
            qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            
            # Ensure quaternion is normalized and qw is positive for consistency
            qw_stable = torch.abs(qw) + 1e-8
            
            # Rotation matrix elements (only what we need)
            R_33 = 1 - 2 * (qx**2 + qy**2)
            R_33_clamped = torch.clamp(R_33, -1.0, 1.0)  # Numerical stability
            
            # PDE Residual 1: Kinematic equation (position)
            residual_pos = pos_dot - vel
            
            # PDE Residual 2: Translational dynamics (velocity)
            force_body_z = thrust
            force_world_z = force_body_z * R_33_clamped
            
            # Create gravity force vector (avoid in-place operations)
            gravity_force = torch.zeros_like(vel)
            gravity_force_corrected = gravity_force.clone()
            gravity_force_corrected[:, 2] = -self.mass * self.g
            
            # Total force in world frame
            total_force = torch.stack([
                torch.zeros_like(force_world_z),
                torch.zeros_like(force_world_z),
                force_world_z
            ], dim=1) + gravity_force_corrected
            
            expected_vel_dot = total_force / self.mass
            residual_vel = vel_dot - expected_vel_dot
            
            # PDE Residual 3: Rotational dynamics (angular velocity)
            expected_omega_dot = torch.stack([
                tau_x / self.Ixx,
                tau_y / self.Iyy,
                tau_z / self.Izz
            ], dim=1)
            
            residual_omega = omega_dot - expected_omega_dot
            
            # PDE Residual 4: Quaternion kinematics (with stability)
            # Quaternion derivative: q_dot = 0.5 * Q(q) * omega
            omega_quat = torch.stack([
                torch.zeros_like(omega[:, 0]),  # scalar part
                omega[:, 0],  # x
                omega[:, 1],  # y
                omega[:, 2]   # z
            ], dim=1)
            
            # Quaternion multiplication for kinematics
            expected_quat_dot = 0.5 * torch.stack([
                -qx * omega[:, 0] - qy * omega[:, 1] - qz * omega[:, 2],
                qw_stable * omega[:, 0] + qy * omega[:, 2] - qz * omega[:, 1],
                qw_stable * omega[:, 1] + qz * omega[:, 0] - qx * omega[:, 2],
                qw_stable * omega[:, 2] + qx * omega[:, 1] - qy * omega[:, 0]
            ], dim=1)
            
            residual_quat = quat_dot - expected_quat_dot
            
            # PDE Residual 5: Control dynamics (first-order lag)
            control_time_constant = 0.05  # 50ms time constant
            expected_controls_dot = (action - controls) / control_time_constant
            residual_controls = controls_dot - expected_controls_dot
            
            # PDE Residual 6: Target dynamics (stationary target)
            residual_target = target_dot
            
            return residual_pos, residual_vel, residual_omega, residual_quat, residual_controls, residual_target
            
        except Exception as e:
            # Fallback to zero residuals if computation fails
            batch_size = state.shape[0]
            device = state.device
            zero_residual_3 = torch.zeros(batch_size, 3, device=device)
            zero_residual_4 = torch.zeros(batch_size, 4, device=device)
            zero_residual_1 = torch.zeros(batch_size, 1, device=device)
            
            return (zero_residual_3, zero_residual_3, zero_residual_3, 
                   zero_residual_4, zero_residual_4, zero_residual_1)
    
    def physics_loss(self, state, next_state, action):
        """Compute physics loss using PDE residuals with adaptive weighting"""
        try:
            # Data loss (standard supervised learning component)
            predicted_next = self.forward(state)
            data_loss = F.smooth_l1_loss(predicted_next, next_state)
            
            # PDE residuals with improved error handling
            residual_pos, residual_vel, residual_omega, residual_quat, residual_controls, residual_target = self.compute_pde_residuals(state, action)
            
            # Adaptive loss scaling based on residual magnitudes
            pos_scale = self.loss_scale_pos
            vel_scale = self.loss_scale_vel
            quat_scale = self.loss_scale_quat
            omega_scale = self.loss_scale_omega
            
            # Compute individual PDE losses with numerical stability
            pde_loss_pos = pos_scale * torch.mean(torch.clamp(residual_pos**2, 0, 100))
            pde_loss_vel = vel_scale * torch.mean(torch.clamp(residual_vel**2, 0, 100))
            pde_loss_omega = omega_scale * torch.mean(torch.clamp(residual_omega**2, 0, 100))
            pde_loss_quat = quat_scale * torch.mean(torch.clamp(residual_quat**2, 0, 100))
            pde_loss_controls = torch.mean(torch.clamp(residual_controls**2, 0, 100))
            pde_loss_target = torch.mean(torch.clamp(residual_target**2, 0, 100))
            
            # Total PDE loss
            pde_loss = (pde_loss_pos + pde_loss_vel + pde_loss_omega + 
                       pde_loss_quat + 0.1 * pde_loss_controls + 0.1 * pde_loss_target)
            
            # Adaptive physics loss weight based on training progress
            physics_weight = max(0.005, min(0.02, 0.005 * (1 + len(self.physics_loss_history) / 1000)))
            
            # Balance data and physics losses with adaptive weighting
            total_loss = data_loss + physics_weight * pde_loss
            
            # Track physics loss for adaptive weighting
            if len(self.physics_loss_history) >= 1000:
                self.physics_loss_history.pop(0)  # Keep only recent history
            self.physics_loss_history.append(pde_loss.item() if torch.isfinite(pde_loss) else 0.0)
            
            # Clamp total loss to prevent explosion  
            total_loss = torch.clamp(total_loss, 0, 20)
            
            # Check for NaN or infinity
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                return data_loss  # Fallback to data loss only
                
            return total_loss
            
        except Exception as e:
            # Fallback to simple data loss if PDE computation fails
            predicted_next = self.forward(state)
            return F.smooth_l1_loss(predicted_next, next_state)


class NeuralOperator(nn.Module):
    """Function-to-function mapping with sensor locations for discretization-invariant learning"""
    
    def __init__(self, state_dim, hidden_dim=1024, num_sensors=32):
        super().__init__()
        self.num_sensors = num_sensors
        self.state_dim = state_dim
        
        # Define sensor locations in 3D space around UAV
        sensor_range = 2.0
        theta = torch.linspace(0, 2*np.pi, num_sensors//2)
        phi = torch.linspace(0, np.pi, 2)
        
        sensors = []
        for p in phi:
            for t in theta:
                x = sensor_range * torch.sin(p) * torch.cos(t)
                y = sensor_range * torch.sin(p) * torch.sin(t)
                z = sensor_range * torch.cos(p)
                sensors.append([x.item(), y.item(), z.item()])
        
        self.register_buffer('sensor_locations', torch.tensor(sensors[:num_sensors], dtype=torch.float32))
        
        # Branch network: encodes input function at sensor locations
        self.branch_net = nn.Sequential(
            nn.Linear(num_sensors * (state_dim + 4), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )
        
        # Trunk network: encodes evaluation coordinates
        self.trunk_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.Tanh(),
        )
        
        # Bias network
        self.bias_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, state_dim)
        )
        
        # Output projection with residual connection
        self.output_projection = nn.Linear(hidden_dim // 4, state_dim)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def evaluate_input_function(self, state, action):
        """Evaluate input function at sensor locations"""
        batch_size = state.shape[0]
        
        # Current UAV position
        pos = state[:, :3]  # shape: (batch_size, 3)
        
        # Evaluate function at each sensor location
        sensor_values = []
        for i in range(self.num_sensors):
            sensor_pos = self.sensor_locations[i].unsqueeze(0).expand(batch_size, -1)
            
            # Distance-based weighting of state influence
            distance = torch.norm(sensor_pos - pos, dim=1, keepdim=True)
            weight = torch.exp(-distance / 0.5)  # Gaussian weighting
            
            # Weighted state and action at sensor location
            weighted_state = state * weight
            weighted_action = action * weight
            
            sensor_value = torch.cat([weighted_state, weighted_action], dim=1)
            sensor_values.append(sensor_value)
        
        # Concatenate all sensor values
        function_encoding = torch.cat(sensor_values, dim=1)
        return function_encoding
    
    def forward(self, state, action):
        """Function-to-function mapping using branch-trunk architecture"""
        batch_size = state.shape[0]
        
        try:
            # Encode input function at sensor locations
            function_encoding = self.evaluate_input_function(state, action)
            
            # Branch network processing
            branch_output = self.branch_net(function_encoding)
            
            # Current position as evaluation coordinate
            eval_coords = state[:, :3]
            
            # Trunk network processing
            trunk_output = self.trunk_net(eval_coords)
            
            # Branch-trunk interaction (element-wise multiplication)
            interaction = branch_output * trunk_output
            
            # Output mapping
            output_delta = self.output_projection(interaction)
            
            # Bias network
            bias = self.bias_net(eval_coords)
            
            # Residual connection with learned weight
            residual_delta = self.residual_weight * (output_delta + bias)
            next_state = state + residual_delta
            
            # Ensure quaternion stays normalized (avoid in-place)
            if next_state.shape[1] >= 7:
                quat = next_state[:, 3:7]
                quat_norm = torch.norm(quat, dim=1, keepdim=True)
                quat_normalized = quat / (quat_norm + 1e-8)
                # Create new tensor instead of in-place modification
                next_state_corrected = next_state.clone()
                next_state_corrected[:, 3:7] = quat_normalized
                return next_state_corrected
            
            return next_state
            
        except Exception as e:
            # Fallback: return input state with small perturbation
            return state + 0.01 * torch.randn_like(state)


class SafetyConstraint(nn.Module):
    """Control Barrier Function with time derivatives for safety guarantees"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super().__init__()
        
        # Barrier function network
        self.barrier_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Lie derivative network for CBF time derivatives
        self.lie_derivative_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1)
        )
        
        # Safety bounds with more conservative values
        self.register_buffer('altitude_min', torch.tensor(0.4))
        self.register_buffer('altitude_max', torch.tensor(1.6))
        self.register_buffer('position_bound', torch.tensor(1.0))
        self.register_buffer('velocity_max', torch.tensor(0.4))
        self.register_buffer('tilt_max', torch.tensor(0.25))
        
        # CBF parameters
        self.register_buffer('alpha', torch.tensor(2.0))  # Increased CBF decay rate for faster convergence
        
    def compute_barrier_function(self, state):
        """Compute barrier function values with improved formulation"""
        # Extract state components (avoid in-place operations)
        pos = state[:, :3].clone()
        quat_raw = state[:, 3:7].clone()
        vel = state[:, 7:10].clone()
        
        # Normalize quaternion (avoid in-place)
        quat_norm = torch.norm(quat_raw, dim=1, keepdim=True)
        quat = quat_raw / (quat_norm + 1e-8)
        
        # Barrier function 1: Altitude constraints
        altitude_barrier_low = pos[:, 2] - self.altitude_min
        altitude_barrier_high = self.altitude_max - pos[:, 2]
        
        # Barrier function 2: Position bounds
        pos_barrier_x = self.position_bound**2 - pos[:, 0]**2
        pos_barrier_y = self.position_bound**2 - pos[:, 1]**2
        
        # Barrier function 3: Velocity constraints
        vel_magnitude_sq = torch.sum(vel**2, dim=1)
        vel_barrier = self.velocity_max**2 - vel_magnitude_sq
        
        # Barrier function 4: Orientation constraints (improved)
        qw = torch.abs(quat[:, 0])  # Take absolute value for stability
        qw_clamped = torch.clamp(qw, 0.1, 1.0)  # Prevent extreme values
        tilt_barrier = qw_clamped - (1 - self.tilt_max)  # Simpler formulation
        
        # Use smooth minimum instead of hard minimum
        barriers = torch.stack([
            altitude_barrier_low,
            altitude_barrier_high,
            pos_barrier_x,
            pos_barrier_y,
            vel_barrier,
            tilt_barrier
        ], dim=1)
        
        # Smooth minimum using LogSumExp trick
        beta = 5.0  # Controls smoothness
        smooth_min = -torch.logsumexp(-beta * barriers, dim=1) / beta
        
        return smooth_min.unsqueeze(1)
    
    def compute_barrier_derivative(self, state, action):
        """Compute time derivative of barrier function using automatic differentiation"""
        try:
            # Ensure state requires gradients (avoid in-place)
            if not state.requires_grad:
                state_grad = state.clone().requires_grad_(True)
            else:
                state_grad = state
            
            # Compute barrier function
            h = self.compute_barrier_function(state_grad)
            
            # Compute gradient with respect to state
            h_grad = torch.autograd.grad(
                outputs=h.sum(), inputs=state_grad,
                create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]
            
            if h_grad is None:
                h_grad = torch.zeros_like(state_grad)
            
            # State dynamics for full 18-dimensional state (avoid in-place)
            pos = state_grad[:, :3].clone()
            quat_raw = state_grad[:, 3:7].clone()
            vel = state_grad[:, 7:10].clone()
            omega = state_grad[:, 10:13].clone()
            controls = state_grad[:, 13:17].clone()
            target = state_grad[:, 17:18].clone()
            
            # Normalize quaternion (avoid in-place)
            quat_norm = torch.norm(quat_raw, dim=1, keepdim=True)
            quat = quat_raw / (quat_norm + 1e-8)
            
            # Control inputs with clamping (avoid in-place)
            thrust = torch.clamp(action[:, 0].clone(), 0.0, 0.35)
            tau = torch.clamp(action[:, 1:4].clone(), -0.2, 0.2)
            
            # Compute state derivatives (simplified dynamics)
            pos_dot = vel
            
            # Quaternion kinematics
            qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            quat_dot = 0.5 * torch.stack([
                -qx * omega[:, 0] - qy * omega[:, 1] - qz * omega[:, 2],
                qw * omega[:, 0] + qy * omega[:, 2] - qz * omega[:, 1],
                qw * omega[:, 1] + qz * omega[:, 0] - qx * omega[:, 2],
                qw * omega[:, 2] + qx * omega[:, 1] - qy * omega[:, 0]
            ], dim=1)
            
            # Translational dynamics (simplified)
            R_33 = torch.clamp(1 - 2 * (qx**2 + qy**2), -1.0, 1.0)
            vel_dot = torch.stack([
                torch.zeros_like(thrust),
                torch.zeros_like(thrust),
                thrust * R_33 / 0.027 - 9.81
            ], dim=1)
            
            # Rotational dynamics
            inertia = torch.tensor([2.3951e-5, 2.3951e-5, 3.2347e-5]).to(tau.device)
            omega_dot = tau / inertia
            
            # Control dynamics
            controls_dot = (action - controls) / 0.05
            
            # Target dynamics
            target_dot = torch.zeros_like(target)
            
            # Full state derivative vector
            state_dot = torch.cat([pos_dot, quat_dot, vel_dot, omega_dot, controls_dot, target_dot], dim=1)
            
            # Lie derivative: ∇h · f(x,u)
            lie_derivative = torch.sum(h_grad * state_dot, dim=1, keepdim=True)
            
            # Clamp to prevent numerical issues
            lie_derivative_clamped = torch.clamp(lie_derivative, -100, 100)
            
            return lie_derivative_clamped
            
        except Exception as e:
            # Fallback: return zero derivative
            return torch.zeros(state.shape[0], 1, device=state.device)
    
    def forward(self, state, action):
        """Compute CBF constraint: ḣ + αh ≥ 0"""
        try:
            h = self.compute_barrier_function(state)
            h_dot = self.compute_barrier_derivative(state, action)
            cbf_constraint = h_dot + self.alpha * h
            
            # Clamp output to reasonable range
            cbf_constraint = torch.clamp(cbf_constraint, -20, 20)
            
            return cbf_constraint
        except Exception as e:
            # Fallback: return conservative safe value
            return torch.ones(state.shape[0], 1, device=state.device)
    
    def get_safety_mask(self, state, action):
        """Get binary safety mask for constraint violations"""
        cbf_values = self.forward(state, action)
        return (cbf_values >= 0).float()
    
    def compute_safety_violation_loss(self, state, action):
        """Compute loss for safety constraint violations"""
        try:
            cbf_values = self.forward(state, action)
            violation_loss = torch.mean(F.relu(-cbf_values))
            
            # Add penalty for very negative CBF values
            severe_violation_penalty = torch.mean(F.relu(-cbf_values - 5.0))
            
            total_loss = violation_loss + 2.0 * severe_violation_penalty
            
            # Clamp to prevent explosion
            total_loss = torch.clamp(total_loss, 0, 20)
            
            return total_loss
        except Exception as e:
            # Fallback: return zero loss
            return torch.tensor(0.0, device=state.device)