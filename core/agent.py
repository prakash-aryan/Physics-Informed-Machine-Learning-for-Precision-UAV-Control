"""
PIATSG Framework - Main Agent
Physics-Informed Adaptive Transformers with Safety Guarantees
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .components import Actor, AdaptivePINN, NeuralOperator, SafetyConstraint
from .buffer import ReplayBuffer

class PIATSGAgent:
    """PIATSG Agent with physics-informed components"""
    
    def __init__(self, config):
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.device = config.device
        self.config = config
        
        # Initialize mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda')
        
        # PIATSG components
        self.actor = Actor(
            self.state_dim, 
            self.action_dim, 
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        # Critic networks
        self.critic1 = self._create_critic().to(self.device)
        self.critic2 = self._create_critic().to(self.device)
        
        # Target networks
        self.target_critic1 = self._create_critic().to(self.device)
        self.target_critic2 = self._create_critic().to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Physics-informed components
        self.adaptive_pinn = AdaptivePINN(self.state_dim, hidden_dim=1024).to(self.device)
        self.neural_operator = NeuralOperator(self.state_dim, hidden_dim=1024).to(self.device)
        self.safety_constraint = SafetyConstraint(
            self.state_dim, 
            self.action_dim, 
            hidden_dim=1024
        ).to(self.device)
        
        # Adaptive learning rates for physics components
        self.base_pinn_lr = config.pinn_lr
        self.base_operator_lr = config.operator_lr
        self.base_safety_lr = config.safety_lr
        
        # Aggressive optimizers for physics components
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.critic_lr)
        
        # Use AdamW with higher learning rates for physics
        self.pinn_optimizer = optim.AdamW(
            self.adaptive_pinn.parameters(), 
            lr=self.base_pinn_lr * 2.0,  # Start higher
            weight_decay=1e-5
        )
        self.operator_optimizer = optim.AdamW(
            self.neural_operator.parameters(), 
            lr=self.base_operator_lr * 2.0,
            weight_decay=1e-5
        )
        self.safety_optimizer = optim.AdamW(
            self.safety_constraint.parameters(), 
            lr=self.base_safety_lr * 2.0,
            weight_decay=1e-5
        )
        
        # Cosine annealing with warm restarts for physics
        self.pinn_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.pinn_optimizer, T_0=500, T_mult=2, eta_min=1e-6
        )
        self.operator_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.operator_optimizer, T_0=500, T_mult=2, eta_min=1e-6
        )
        self.safety_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.safety_optimizer, T_0=500, T_mult=2, eta_min=1e-6
        )
        
        # SAC parameters
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = config.target_entropy
        
        # Replay buffer
        self.memory = ReplayBuffer(config.buffer_size, config.batch_size, self.device)
        
        # Training parameters
        self.tau = config.tau
        self.gamma = config.gamma
        
        # Aggressive physics loss weights
        self.initial_physics_weight = 0.02    # Start higher
        self.initial_safety_weight = 0.01     # Start higher
        self.min_physics_weight = 0.005       # Higher minimum
        self.min_safety_weight = 0.002        # Higher minimum
        self.max_physics_weight = 0.2         # Much higher maximum
        self.max_safety_weight = 0.08         # Much higher maximum
        self.physics_loss_weight = self.initial_physics_weight
        self.safety_loss_weight = self.initial_safety_weight
        
        # Physics improvement tracking
        self.physics_improvement_tracker = {
            'baseline_pinn_loss': None,
            'baseline_safety_loss': None,
            'best_pinn_loss': float('inf'),
            'best_safety_loss': float('inf'),
            'stagnation_counter': 0,
            'improvement_window': [],
            'force_aggressive_mode': False
        }
        
        # Component activation
        self.min_buffer_for_physics = 8000   # Earlier activation
        self.max_grad_norm = 1.0
        self.physics_grad_norm = 0.5         # Allow larger gradients
        
        # Aggressive update frequency
        self.initial_physics_update_frequency = 30  # More frequent
        self.min_physics_update_frequency = 2       # Very frequent
        self.current_physics_update_frequency = self.initial_physics_update_frequency
        
        # Performance tracking
        self.update_count = 0
        self.physics_update_count = 0
        self.total_training_cycles = 0
        
        # Phase tracking
        self.current_phase = "Phase 1: RL Stabilization"
        self.episode_progress = 0.0
        
        # Aggressive mode control
        self.aggressive_mode_threshold = 100  # Activate after 100 stagnant updates
        self.physics_boost_factor = 1.0
        
        # Print initialization info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"PIATSG Agent initialized for AGGRESSIVE PHYSICS LEARNING:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Physics weight range: {self.min_physics_weight:.6f} -> {self.max_physics_weight:.6f}")
        print(f"  Safety weight range: {self.min_safety_weight:.6f} -> {self.max_safety_weight:.6f}")
        print(f"  Physics gradient clipping: {self.physics_grad_norm}")
        print(f"  Physics frequency range: {self.min_physics_update_frequency} -> {self.initial_physics_update_frequency} cycles")
        print(f"  Aggressive learning mode: ENABLED")
        if torch.cuda.is_available():
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def _create_critic(self):
        """Create critic network architecture"""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def parameters(self):
        """Get all parameters for counting"""
        params = []
        params.extend(self.actor.parameters())
        params.extend(self.critic1.parameters())
        params.extend(self.critic2.parameters())
        params.extend(self.adaptive_pinn.parameters())
        params.extend(self.neural_operator.parameters())
        params.extend(self.safety_constraint.parameters())
        return params
    
    def _track_physics_improvement(self, pinn_loss, safety_loss):
        """Track physics improvement and activate aggressive mode if stagnant"""
        pit = self.physics_improvement_tracker
        
        # Initialize baselines
        if pit['baseline_pinn_loss'] is None:
            pit['baseline_pinn_loss'] = pinn_loss
            pit['baseline_safety_loss'] = safety_loss
            return
        
        # Update best losses
        if pinn_loss < pit['best_pinn_loss']:
            pit['best_pinn_loss'] = pinn_loss
            pit['stagnation_counter'] = 0
        else:
            pit['stagnation_counter'] += 1
            
        if safety_loss < pit['best_safety_loss']:
            pit['best_safety_loss'] = safety_loss
        
        # Track improvement rate
        pinn_improvement = (pit['baseline_pinn_loss'] - pinn_loss) / (pit['baseline_pinn_loss'] + 1e-8)
        safety_improvement = (pit['baseline_safety_loss'] - safety_loss) / (pit['baseline_safety_loss'] + 1e-8)
        
        pit['improvement_window'].append((pinn_improvement, safety_improvement))
        if len(pit['improvement_window']) > 50:
            pit['improvement_window'].pop(0)
        
        # Check for stagnation
        if pit['stagnation_counter'] > self.aggressive_mode_threshold:
            if not pit['force_aggressive_mode']:
                print(f"ACTIVATING AGGRESSIVE PHYSICS LEARNING MODE")
                pit['force_aggressive_mode'] = True
                self.physics_boost_factor = 2.0
                
                # Boost learning rates
                for param_group in self.pinn_optimizer.param_groups:
                    param_group['lr'] = min(param_group['lr'] * 2.0, self.base_pinn_lr * 5)
                for param_group in self.operator_optimizer.param_groups:
                    param_group['lr'] = min(param_group['lr'] * 2.0, self.base_operator_lr * 5)
                for param_group in self.safety_optimizer.param_groups:
                    param_group['lr'] = min(param_group['lr'] * 2.0, self.base_safety_lr * 5)
        
        # Check if we should exit aggressive mode
        elif pit['force_aggressive_mode'] and len(pit['improvement_window']) >= 20:
            recent_pinn_imp = np.mean([x[0] for x in pit['improvement_window'][-20:]])
            recent_safety_imp = np.mean([x[1] for x in pit['improvement_window'][-20:]])
            
            if recent_pinn_imp > 0.02 and recent_safety_imp > 0.02:
                print(f"Exiting aggressive mode - improvement detected")
                pit['force_aggressive_mode'] = False
                self.physics_boost_factor = 1.0
                pit['stagnation_counter'] = 0
    
    def _update_physics_weights(self):
        """Aggressive physics weight adjustment"""
        # Calculate episode progress
        total_updates_expected = self.config.num_episodes * (self.config.max_steps_per_episode // self.config.training_frequency) * self.config.updates_per_training
        episode_progress = min(self.update_count / total_updates_expected, 1.0)
        
        pit = self.physics_improvement_tracker
        
        # Three-phase curriculum with aggressive progression
        if episode_progress < 0.15:  # Shorter Phase 1
            # Phase 1: Quick RL stabilization
            decay_factor = 0.5
            frequency_factor = 1.0
            phase_name = "Phase 1: RL Stabilization"
        elif episode_progress < 0.5:  # Shorter Phase 2
            # Phase 2: Aggressive physics introduction
            phase_progress = (episode_progress - 0.15) / 0.35
            decay_factor = 0.5 + phase_progress * 3.5  # Rapid increase to 4.0
            frequency_factor = 1.0 - phase_progress * 0.85  # Rapid frequency increase
            phase_name = "Phase 2: Progressive Learning"
        else:
            # Phase 3: Maximum physics influence
            phase_progress = (episode_progress - 0.5) / 0.5
            
            # Base decay starts high and can go higher
            base_decay = 4.0 + phase_progress * 2.0  # 4.0 to 6.0
            
            # Apply boost if in aggressive mode
            if pit['force_aggressive_mode']:
                base_decay *= 1.5
                
            decay_factor = base_decay * self.physics_boost_factor
            frequency_factor = 0.15 - phase_progress * 0.08  # Very frequent updates
            phase_name = "Phase 3: Physics Refinement"
        
        # Set weights aggressively
        self.physics_loss_weight = np.clip(
            self.initial_physics_weight * decay_factor,
            self.min_physics_weight,
            self.max_physics_weight
        )
        self.safety_loss_weight = np.clip(
            self.initial_safety_weight * decay_factor,
            self.min_safety_weight,
            self.max_safety_weight
        )
        
        # Very frequent updates in aggressive mode
        if pit['force_aggressive_mode']:
            frequency_factor *= 0.5  # Double the frequency
            
        new_frequency = max(
            int(self.initial_physics_update_frequency * frequency_factor),
            self.min_physics_update_frequency
        )
        self.current_physics_update_frequency = new_frequency
        
        # Store phase info
        self.current_phase = phase_name
        self.episode_progress = episode_progress
    
    def select_action(self, state, deterministic=False):
        """Select action with physics-guided exploration"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            raw_action = self.actor(state_tensor, deterministic=deterministic)
            
            # Physics-guided exploration in non-deterministic mode
            if not deterministic and len(self.memory) > self.min_buffer_for_physics:
                # Get physics prediction
                state_expanded = state_tensor.unsqueeze(0)
                action_expanded = raw_action.unsqueeze(0)
                
                # Predict next state with physics
                physics_next = self.adaptive_pinn(state_expanded)
                operator_next = self.neural_operator(state_expanded, action_expanded)
                
                # Blend predictions
                physics_blend = 0.3 if self.physics_improvement_tracker['force_aggressive_mode'] else 0.1
                predicted_next = (1 - physics_blend) * physics_next + physics_blend * operator_next
                
                # Compute physics-based action adjustment
                target_pos = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                current_pos = state_tensor[:3]
                predicted_pos = predicted_next[0, :3]
                
                # Direction towards target
                pos_error = target_pos - predicted_pos
                action_adjustment = pos_error * 0.1  # Small adjustment
                
                # Apply adjustment to thrust and moments
                raw_action = raw_action + torch.cat([
                    action_adjustment[2:3] * 0.2,  # Thrust adjustment
                    action_adjustment[:2] * 0.1,   # Roll/pitch adjustment
                    torch.zeros(1, device=self.device)  # No yaw adjustment
                ])
            
            # Apply safety constraint filtering
            if len(self.memory) > 8000:
                state_for_cbf = state_tensor.clone().requires_grad_(True)
                action_for_cbf = raw_action.clone().requires_grad_(True)
                
                with torch.enable_grad():
                    safety_logits = self.safety_constraint(state_for_cbf.unsqueeze(0), action_for_cbf.unsqueeze(0))
                    safety_mask = (safety_logits > 0).float()
                    
                    if safety_mask.item() < 0.5:
                        raw_action = raw_action * 0.8
            
            # Action clipping
            action = torch.clamp(raw_action, -0.6, 0.6)
            
            # Training phase scaling
            if len(self.memory) < 3000:
                action = action * 0.5
            elif len(self.memory) < 8000:
                action = action * 0.75
        
        return action.cpu().numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=None, update_physics_components=True):
        """Update all components with aggressive physics learning"""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if len(self.memory) < batch_size:
            return None
        
        self.update_count += 1
        self.total_training_cycles += 1
        
        # Update physics weights
        self._update_physics_weights()
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Check for NaN
        if (torch.isnan(states).any() or torch.isnan(actions).any() or 
            torch.isnan(rewards).any() or torch.isnan(next_states).any()):
            print("Warning: NaN detected in training data, skipping update")
            return None
        
        losses = {}
        
        try:
            # Update Critics
            critic_losses = self._update_critics(states, actions, rewards, next_states, dones)
            losses.update(critic_losses)
            
            # Update Actor with physics-informed loss
            actor_loss = self._update_actor_with_physics(states, actions, next_states)
            losses['actor_loss'] = actor_loss
            
            # Update Alpha
            alpha_loss = self._update_alpha(states)
            losses['alpha_loss'] = alpha_loss
            
            # Aggressive physics updates
            should_update_physics = (
                update_physics_components and 
                len(self.memory) > self.min_buffer_for_physics and
                (self.total_training_cycles % self.current_physics_update_frequency == 0 or
                self.physics_improvement_tracker['force_aggressive_mode'])  # Force updates in aggressive mode
            )
            
            if should_update_physics:
                self.physics_update_count += 1
                
                # Multiple physics updates if in aggressive mode
                num_physics_updates = 3 if self.physics_improvement_tracker['force_aggressive_mode'] else 1
                
                # Initialize combined losses
                combined_pinn_loss = 0.0
                combined_operator_loss = 0.0
                combined_safety_loss = 0.0
                
                for update_idx in range(num_physics_updates):
                    # Sample fresh batch for each physics update to avoid graph issues
                    if update_idx > 0:
                        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
                    
                    physics_losses = self._update_physics_components(states, actions, next_states)
                    
                    # Accumulate losses for reporting
                    combined_pinn_loss += physics_losses.get('pinn_loss', 0.0)
                    combined_operator_loss += physics_losses.get('operator_loss', 0.0)
                    combined_safety_loss += physics_losses.get('safety_loss', 0.0)
                
                # Average the losses
                losses['pinn_loss'] = combined_pinn_loss / num_physics_updates
                losses['operator_loss'] = combined_operator_loss / num_physics_updates
                losses['safety_loss'] = combined_safety_loss / num_physics_updates
                
                # Track improvement
                self._track_physics_improvement(losses['pinn_loss'], losses['safety_loss'])
            else:
                losses.update({'pinn_loss': 0.0, 'operator_loss': 0.0, 'safety_loss': 0.0})
            
            # Update target networks
            self._update_target_networks()
            
        except Exception as e:
            print(f"Warning: Error in update: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return losses

    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks"""
        with torch.amp.autocast('cuda', enabled=True):
            # Target computation
            with torch.no_grad():
                next_actions = self.actor(next_states, deterministic=True)
                next_actions = torch.clamp(next_actions, -1, 1)
                
                target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1))
                target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1))
                target_q = torch.min(target_q1, target_q2)
                
                alpha = torch.clamp(self.log_alpha.exp(), 0.01, 5.0)
                next_log_probs = -0.5 * torch.sum((next_actions ** 2), dim=1, keepdim=True)
                target_q = target_q - alpha * next_log_probs
                target_q = rewards + (1 - dones) * self.gamma * target_q
                target_q = torch.clamp(target_q, -100, 100)
            
            # Current Q values
            current_q1 = self.critic1(torch.cat([states, actions], dim=1))
            current_q2 = self.critic2(torch.cat([states, actions], dim=1))
            
            if torch.isnan(current_q1).any() or torch.isnan(current_q2).any():
                print("Warning: NaN in Q values")
                return {'critic1_loss': 0.0, 'critic2_loss': 0.0}
            
            # Critic losses
            critic1_loss = F.smooth_l1_loss(current_q1, target_q)
            critic2_loss = F.smooth_l1_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        if torch.isfinite(critic1_loss):
            self.scaler.scale(critic1_loss).backward()
            self.scaler.unscale_(self.critic1_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
            self.scaler.step(self.critic1_optimizer)
        
        self.critic2_optimizer.zero_grad()
        if torch.isfinite(critic2_loss):
            self.scaler.scale(critic2_loss).backward()
            self.scaler.unscale_(self.critic2_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
            self.scaler.step(self.critic2_optimizer)
        
        self.scaler.update()
        
        return {
            'critic1_loss': critic1_loss.item() if torch.isfinite(critic1_loss) else 0.0,
            'critic2_loss': critic2_loss.item() if torch.isfinite(critic2_loss) else 0.0
        }
    
    def _update_actor_with_physics(self, states, actions, next_states):
        """Update actor with physics-informed objectives"""
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
            
            # Add physics-informed objectives
            if len(self.memory) > self.min_buffer_for_physics:
                # Physics consistency loss
                predicted_next = self.adaptive_pinn(states)
                physics_error = F.mse_loss(predicted_next, next_states)
                
                # Operator consistency loss  
                operator_next = self.neural_operator(states, new_actions)
                operator_error = F.mse_loss(operator_next, next_states)
                
                # Safety constraint loss
                safety_violation = self.safety_constraint.compute_safety_violation_loss(states, new_actions)
                
                # Combine with aggressive weights
                physics_weight = self.physics_loss_weight * 2.0  # Double the influence on actor
                actor_loss = (actor_loss + 
                             physics_weight * (physics_error + operator_error) +
                             self.safety_loss_weight * safety_violation)
            
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
    
    def _update_alpha(self, states):
        """Update alpha parameter"""
        with torch.no_grad():
            actions_alpha = self.actor(states, deterministic=False)
            actions_alpha = torch.clamp(actions_alpha, -1, 1)
            log_probs_alpha = -0.5 * torch.sum((actions_alpha ** 2), dim=1, keepdim=True)
        
        alpha_loss = -(self.log_alpha * (log_probs_alpha.detach() + self.target_entropy)).mean()
        
        if torch.isfinite(alpha_loss):
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
            self.alpha_optimizer.step()
        
        return alpha_loss.item() if torch.isfinite(alpha_loss) else 0.0
    
    def _update_physics_components(self, states, actions, next_states):
        """Aggressive physics component updates"""
        losses = {}
        
        try:
            # PINN update with strong gradients
            self.pinn_optimizer.zero_grad()
            
            # Clone inputs to ensure fresh computation graph
            states_pinn = states.clone().detach().requires_grad_(False)
            actions_pinn = actions.clone().detach().requires_grad_(False)
            next_states_pinn = next_states.clone().detach().requires_grad_(False)
            
            pinn_loss = self.adaptive_pinn.physics_loss(states_pinn, next_states_pinn, actions_pinn)
            
            # Minimal regularization to allow learning
            l2_weight = 1e-6
            l2_reg_pinn = torch.tensor(0.0, device=states.device)
            for param in self.adaptive_pinn.parameters():
                l2_reg_pinn = l2_reg_pinn + torch.norm(param)
            
            pinn_loss_total = pinn_loss + l2_weight * l2_reg_pinn
            
            if torch.isfinite(pinn_loss_total):
                pinn_loss_total.backward()
                # Allow larger gradients
                torch.nn.utils.clip_grad_norm_(self.adaptive_pinn.parameters(), self.physics_grad_norm)
                self.pinn_optimizer.step()
                self.pinn_scheduler.step()
                losses['pinn_loss'] = pinn_loss.item()
            else:
                losses['pinn_loss'] = 0.0
            
            # Neural Operator update with fresh graph
            self.operator_optimizer.zero_grad()
            
            # Clone again for operator to ensure fresh graph
            states_op = states.clone().detach().requires_grad_(False)
            actions_op = actions.clone().detach().requires_grad_(False)
            next_states_op = next_states.clone().detach().requires_grad_(False)
            
            operator_pred = self.neural_operator(states_op, actions_op)
            operator_loss = F.smooth_l1_loss(operator_pred, next_states_op)
            
            l2_reg_op = torch.tensor(0.0, device=states.device)
            for param in self.neural_operator.parameters():
                l2_reg_op = l2_reg_op + torch.norm(param)
                
            operator_loss_total = operator_loss + l2_weight * l2_reg_op
            
            if torch.isfinite(operator_loss_total):
                operator_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.neural_operator.parameters(), self.physics_grad_norm)
                self.operator_optimizer.step()
                self.operator_scheduler.step()
                losses['operator_loss'] = operator_loss.item()
            else:
                losses['operator_loss'] = 0.0
            
            # Safety Constraint update with fresh graph
            self.safety_optimizer.zero_grad()
            
            # Clone for safety to ensure fresh graph
            states_safe = states.clone().detach().requires_grad_(False)
            actions_safe = actions.clone().detach().requires_grad_(False)
            
            batch_size = states.shape[0]
            
            # Aggressive sampling for maximum learning
            safe_size = batch_size // 4
            unsafe_size = batch_size // 2
            boundary_size = batch_size - safe_size - unsafe_size
            
            indices = torch.randperm(batch_size)
            safe_indices = indices[:safe_size]
            unsafe_indices = indices[safe_size:safe_size + unsafe_size]
            boundary_indices = indices[safe_size + unsafe_size:]
            
            safe_loss = torch.tensor(0.0, device=states.device)
            unsafe_loss = torch.tensor(0.0, device=states.device)
            boundary_loss = torch.tensor(0.0, device=states.device)
            
            # Large safety margin to force learning
            safety_margin = 0.3 + min(self.episode_progress * 0.3, 0.3)
            
            if len(safe_indices) > 0:
                safe_logits = self.safety_constraint(states_safe[safe_indices], actions_safe[safe_indices])
                safe_loss = F.relu(-safe_logits + safety_margin).mean()
            
            # Very aggressive unsafe exploration
            aggression_factor = 2.0 + min(self.episode_progress * 1.0, 1.0)
            if len(unsafe_indices) > 0:
                unsafe_actions = actions_safe[unsafe_indices] * aggression_factor
                unsafe_actions_clamped = torch.clamp(unsafe_actions, -1, 1)
                unsafe_logits = self.safety_constraint(states_safe[unsafe_indices], unsafe_actions_clamped)
                unsafe_loss = F.relu(unsafe_logits + safety_margin).mean()
            
            # High noise boundary exploration
            boundary_noise = 0.2 + min(self.episode_progress * 0.2, 0.2)
            if len(boundary_indices) > 0:
                boundary_actions = actions_safe[boundary_indices] + boundary_noise * torch.randn_like(actions_safe[boundary_indices])
                boundary_actions_clamped = torch.clamp(boundary_actions, -1, 1)
                boundary_logits = self.safety_constraint(states_safe[boundary_indices], boundary_actions_clamped)
                boundary_loss = torch.abs(boundary_logits).mean()
            
            safety_loss = safe_loss + unsafe_loss + 0.5 * boundary_loss
            
            l2_reg_safety = torch.tensor(0.0, device=states.device)
            for param in self.safety_constraint.parameters():
                l2_reg_safety = l2_reg_safety + torch.norm(param)
                
            safety_loss_total = safety_loss + l2_weight * l2_reg_safety
            
            if torch.isfinite(safety_loss_total):
                safety_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.safety_constraint.parameters(), self.physics_grad_norm)
                self.safety_optimizer.step()
                self.safety_scheduler.step()
                losses['safety_loss'] = safety_loss.item()
            else:
                losses['safety_loss'] = 0.0
                
        except Exception as e:
            print(f"Warning: Error in physics components: {e}")
            losses = {'pinn_loss': 0.0, 'operator_loss': 0.0, 'safety_loss': 0.0}
        
        return losses

    def _update_target_networks(self):
        """Update target networks using soft updates"""
        with torch.no_grad():
            for tp1, p1, tp2, p2 in zip(
                self.target_critic1.parameters(), self.critic1.parameters(),
                self.target_critic2.parameters(), self.critic2.parameters()
            ):
                tp1.data.mul_(1 - self.tau).add_(p1.data, alpha=self.tau)
                tp2.data.mul_(1 - self.tau).add_(p2.data, alpha=self.tau)
    
    def save_model(self, filepath):
        """Save all model components"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'pinn': self.adaptive_pinn.state_dict(),
            'operator': self.neural_operator.state_dict(),
            'safety': self.safety_constraint.state_dict(),
            'log_alpha': self.log_alpha,
            'config': self.config,
            'update_count': self.update_count,
            'physics_update_count': self.physics_update_count,
            'total_training_cycles': self.total_training_cycles,
            'physics_loss_weight': self.physics_loss_weight,
            'safety_loss_weight': self.safety_loss_weight,
            'physics_improvement_tracker': self.physics_improvement_tracker
        }, filepath)
    
    def load_model(self, filepath):
        """Load all model components"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.adaptive_pinn.load_state_dict(checkpoint['pinn'])
        self.neural_operator.load_state_dict(checkpoint['operator'])
        self.safety_constraint.load_state_dict(checkpoint['safety'])
        self.log_alpha = checkpoint['log_alpha']
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        if 'physics_update_count' in checkpoint:
            self.physics_update_count = checkpoint['physics_update_count']
        if 'total_training_cycles' in checkpoint:
            self.total_training_cycles = checkpoint['total_training_cycles']
        if 'physics_loss_weight' in checkpoint:
            self.physics_loss_weight = checkpoint['physics_loss_weight']
        if 'safety_loss_weight' in checkpoint:
            self.safety_loss_weight = checkpoint['safety_loss_weight']
        if 'physics_improvement_tracker' in checkpoint:
            self.physics_improvement_tracker = checkpoint['physics_improvement_tracker']