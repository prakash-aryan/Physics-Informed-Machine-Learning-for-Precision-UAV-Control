# core/balanced_agent.py (complete fixed version)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .agent import PIATSGAgent
from utils.relobalo import ReLoBRaLo

class BalancedPIATSGAgent(PIATSGAgent):
    """PIATSG Agent with ReLoBRaLo gradient balancing"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Gradient balancing
        self.gradient_balancer = ReLoBRaLo(
            num_components=4,  # actor, pinn, operator, safety
            alpha=0.95,
            lookback_window=20
        )
        
        print("BalancedPIATSG Agent initialized with ReLoBRaLo gradient balancing")
    
    def update(self, batch_size):
        """Update all networks with gradient balancing"""
        if len(self.memory) < batch_size:
            return None
        
        try:
            # Sample batch
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
            
            # Convert to tensors
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device).unsqueeze(1)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device).unsqueeze(1)
            
            # Update critics (standard SAC)
            critic_losses = self._update_critics(states, actions, rewards, next_states, dones)
            
            # Compute gradients for all components
            grad_norms = self._compute_component_gradients(states, actions, next_states)
            
            # Update weights based on gradient norms
            balanced_weights = self.gradient_balancer.update_weights(grad_norms)
            
            # Apply balanced weights
            actor_weight = balanced_weights[0]
            pinn_weight = balanced_weights[1]
            operator_weight = balanced_weights[2]
            safety_weight = balanced_weights[3]
            
            # Update actor with balanced physics
            actor_loss = self._update_actor_balanced(
                states, actions, next_states,
                actor_weight, pinn_weight, operator_weight, safety_weight
            )
            
            # Update alpha
            alpha_loss = self._update_alpha(states)
            
            # Collect losses
            losses = {
                'critic1_loss': critic_losses['critic1_loss'],
                'critic2_loss': critic_losses['critic2_loss'],
                'actor_loss': actor_loss,
                'alpha_loss': alpha_loss,
                'pinn_loss': 0.0,
                'operator_loss': 0.0,
                'safety_loss': 0.0,
                'actor_weight': actor_weight,
                'pinn_weight': pinn_weight,
                'operator_weight': operator_weight,
                'safety_weight': safety_weight
            }
            
            # Physics components update (with curriculum)
            self.total_training_cycles += 1
            if self.total_training_cycles % self.current_physics_update_frequency == 0:
                self.physics_update_count += 1
                
                physics_losses = self._update_physics_components(states, actions, next_states)
                losses.update(physics_losses)
                
                self._track_physics_improvement(losses['pinn_loss'], losses['safety_loss'])
            
            # Update target networks
            self._update_target_networks()
            
        except Exception as e:
            print(f"Warning: Error in update: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return losses

# core/balanced_agent.py (fix the _update_critics method)

    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks - FIXED version"""
        with torch.amp.autocast('cuda', enabled=True):
            # FIX: Ensure rewards and dones have correct shape [batch_size, 1]
            if rewards.dim() > 2:
                rewards = rewards.squeeze(-1)
            if dones.dim() > 2:
                dones = dones.squeeze(-1)
            
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
                
                # Compute TD target
                target_q = rewards + (1 - dones) * self.gamma * target_q
                target_q = torch.clamp(target_q, -100, 100)
            
            # Current Q values
            current_q1 = self.critic1(torch.cat([states, actions], dim=1))
            current_q2 = self.critic2(torch.cat([states, actions], dim=1))
            
            if torch.isnan(current_q1).any() or torch.isnan(current_q2).any():
                print("Warning: NaN in Q values")
                return {'critic1_loss': 0.0, 'critic2_loss': 0.0}
            
            # Compute critic losses
            critic1_loss = F.smooth_l1_loss(current_q1, target_q)
            critic2_loss = F.smooth_l1_loss(current_q2, target_q)
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        if torch.isfinite(critic1_loss):
            self.scaler.scale(critic1_loss).backward()
            self.scaler.unscale_(self.critic1_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
            self.scaler.step(self.critic1_optimizer)
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        if torch.isfinite(critic2_loss):
            self.scaler.scale(critic2_loss).backward()
            self.scaler.unscale_(self.critic2_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
            self.scaler.step(self.critic2_optimizer)
        
        # Update scaler once after both critics
        self.scaler.update()
        
        return {
            'critic1_loss': critic1_loss.item() if torch.isfinite(critic1_loss) else 0.0,
            'critic2_loss': critic2_loss.item() if torch.isfinite(critic2_loss) else 0.0
        }

    def _compute_component_gradients(self, states, actions, next_states):
        """Compute gradient norms for each component without stepping optimizers"""
        grad_norms = []
        
        # 1. Actor gradient norm
        self.actor_optimizer.zero_grad()
        actor_actions = self.actor(states, deterministic=True)
        actor_actions = torch.clamp(actor_actions, -1, 1)
        q1 = self.critic1(torch.cat([states, actor_actions], dim=1))
        q2 = self.critic2(torch.cat([states, actor_actions], dim=1))
        q_min = torch.min(q1, q2)
        actor_loss = -q_min.mean()
        actor_loss.backward(retain_graph=True)
        actor_norm = self.gradient_balancer.compute_gradient_norm(self.actor.parameters())
        grad_norms.append(actor_norm)
        self.actor_optimizer.zero_grad()
        
        # 2. PINN gradient norm
        self.pinn_optimizer.zero_grad()
        pinn_loss = self.adaptive_pinn.physics_loss(states, next_states, actions)
        if torch.isfinite(pinn_loss):
            pinn_loss.backward(retain_graph=True)
            pinn_norm = self.gradient_balancer.compute_gradient_norm(self.adaptive_pinn.parameters())
            grad_norms.append(pinn_norm)
        else:
            grad_norms.append(0.0)
        self.pinn_optimizer.zero_grad()
        
        # 3. Operator gradient norm
        self.operator_optimizer.zero_grad()
        operator_pred = self.neural_operator(states, actions)
        operator_loss = torch.nn.functional.smooth_l1_loss(operator_pred, next_states)
        operator_loss.backward(retain_graph=True)
        operator_norm = self.gradient_balancer.compute_gradient_norm(self.neural_operator.parameters())
        grad_norms.append(operator_norm)
        self.operator_optimizer.zero_grad()
        
        # 4. Safety gradient norm
        self.safety_optimizer.zero_grad()
        safety_barrier = self.safety_constraint(states, actions)
        safety_loss = torch.relu(-safety_barrier).mean()
        if torch.isfinite(safety_loss) and safety_loss > 0:
            safety_loss.backward(retain_graph=True)
            safety_norm = self.gradient_balancer.compute_gradient_norm(self.safety_constraint.parameters())
            grad_norms.append(safety_norm)
        else:
            grad_norms.append(0.0)
        self.safety_optimizer.zero_grad()
        
        return grad_norms

    def _update_actor_balanced(self, states, actions, next_states, 
                              actor_weight, pinn_weight, operator_weight, safety_weight):
        """Update actor with gradient-level balancing"""
        
        # Compute actor loss and gradients
        self.actor_optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=True):
            actor_actions = self.actor(states, deterministic=True)
            actor_actions = torch.clamp(actor_actions, -1, 1)
            
            q1 = self.critic1(torch.cat([states, actor_actions], dim=1))
            q2 = self.critic2(torch.cat([states, actor_actions], dim=1))
            q_min = torch.min(q1, q2)
            
            alpha = torch.clamp(self.log_alpha.exp(), 0.01, 5.0)
            log_probs = -0.5 * torch.sum((actor_actions ** 2), dim=1, keepdim=True)
            
            actor_loss = (alpha * log_probs - q_min).mean()
        
        # Backward for actor component
        self.scaler.scale(actor_loss).backward(retain_graph=True)
        
        # Scale actor gradients by weight
        for p in self.actor.parameters():
            if p.grad is not None:
                p.grad.data.mul_(actor_weight)
        
        # Store scaled actor gradients
        actor_grads = {name: p.grad.clone() if p.grad is not None else None 
                      for name, p in self.actor.named_parameters()}
        
        # Zero out gradients
        self.actor_optimizer.zero_grad()
        
        # Compute and accumulate PINN gradients
        with torch.amp.autocast('cuda', enabled=True):
            pinn_loss = self.adaptive_pinn.physics_loss(states, next_states, actions)
        
        if torch.isfinite(pinn_loss):
            self.scaler.scale(pinn_loss).backward(retain_graph=True)
            
            # Scale and accumulate PINN gradients
            for name, p in self.actor.named_parameters():
                if p.grad is not None:
                    if actor_grads[name] is not None:
                        actor_grads[name].add_(p.grad.data * pinn_weight)
                    else:
                        actor_grads[name] = p.grad.data * pinn_weight
        
        self.actor_optimizer.zero_grad()
        
        # Compute and accumulate Operator gradients
        with torch.amp.autocast('cuda', enabled=True):
            operator_pred = self.neural_operator(states, actions)
            operator_loss = torch.nn.functional.smooth_l1_loss(operator_pred, next_states)
        
        self.scaler.scale(operator_loss).backward(retain_graph=True)
        
        # Scale and accumulate Operator gradients
        for name, p in self.actor.named_parameters():
            if p.grad is not None:
                if actor_grads[name] is not None:
                    actor_grads[name].add_(p.grad.data * operator_weight)
                else:
                    actor_grads[name] = p.grad.data * operator_weight
        
        self.actor_optimizer.zero_grad()
        
        # Compute and accumulate Safety gradients
        with torch.amp.autocast('cuda', enabled=True):
            safety_barrier = self.safety_constraint(states, actions)
            safety_loss = torch.relu(-safety_barrier).mean()
        
        if torch.isfinite(safety_loss) and safety_loss > 0:
            self.scaler.scale(safety_loss).backward(retain_graph=True)
            
            # Scale and accumulate Safety gradients
            for name, p in self.actor.named_parameters():
                if p.grad is not None:
                    if actor_grads[name] is not None:
                        actor_grads[name].add_(p.grad.data * safety_weight)
                    else:
                        actor_grads[name] = p.grad.data * safety_weight
        
        self.actor_optimizer.zero_grad()
        
        # Apply accumulated scaled gradients
        for name, p in self.actor.named_parameters():
            if actor_grads[name] is not None:
                p.grad = actor_grads[name]
        
        # Clip and step
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        
        return actor_loss.item() if torch.isfinite(actor_loss) else 0.0