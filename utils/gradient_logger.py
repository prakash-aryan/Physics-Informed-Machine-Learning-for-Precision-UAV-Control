# utils/gradient_logger.py
import torch
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

class GradientLogger:
    """Logs and analyzes gradients from multiple loss components"""
    
    def __init__(self, log_dir='gradient_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.gradient_history = defaultdict(list)
        self.alignment_history = []
        self.step_count = 0
        
    def compute_gradient_stats(self, model, loss_name):
        """Compute gradient magnitude and direction for a model"""
        total_norm = 0.0
        grad_vector = []
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_vector.extend(p.grad.detach().flatten().cpu().numpy())
        
        total_norm = total_norm ** 0.5
        
        return {
            'magnitude': total_norm,
            'vector': np.array(grad_vector) if grad_vector else np.array([])
        }
    
    def log_component_gradients(self, agent, states, actions, next_states):
        """Log gradients for each loss component"""
        self.step_count += 1
        component_grads = {}
        
        # Actor loss gradient
        agent.actor_optimizer.zero_grad()
        actor_actions = agent.actor(states, deterministic=True)
        actor_actions = torch.clamp(actor_actions, -1, 1)
        q1 = agent.critic1(torch.cat([states, actor_actions], dim=1))
        q2 = agent.critic2(torch.cat([states, actor_actions], dim=1))
        q_min = torch.min(q1, q2)
        actor_loss = -q_min.mean()
        actor_loss.backward(retain_graph=True)
        component_grads['actor'] = self.compute_gradient_stats(agent.actor, 'actor')
        agent.actor_optimizer.zero_grad()
        
        # PINN gradient
        agent.pinn_optimizer.zero_grad()
        pinn_loss = agent.adaptive_pinn.physics_loss(states, next_states, actions)
        if torch.isfinite(pinn_loss):
            pinn_loss.backward(retain_graph=True)
            component_grads['pinn'] = self.compute_gradient_stats(agent.adaptive_pinn, 'pinn')
        agent.pinn_optimizer.zero_grad()
        
        # Operator gradient
        agent.operator_optimizer.zero_grad()
        operator_pred = agent.neural_operator(states, actions)
        operator_loss = torch.nn.functional.smooth_l1_loss(operator_pred, next_states)
        operator_loss.backward(retain_graph=True)
        component_grads['operator'] = self.compute_gradient_stats(agent.neural_operator, 'operator')
        agent.operator_optimizer.zero_grad()
        
        # Safety gradient
        agent.safety_optimizer.zero_grad()
        safety_barrier = agent.safety_constraint(states, actions)
        safety_loss = torch.relu(-safety_barrier).mean()
        if torch.isfinite(safety_loss) and safety_loss > 0:
            safety_loss.backward(retain_graph=True)
            component_grads['safety'] = self.compute_gradient_stats(agent.safety_constraint, 'safety')
        agent.safety_optimizer.zero_grad()
        
        # Store magnitudes
        for comp, stats in component_grads.items():
            self.gradient_history[f'{comp}_magnitude'].append(stats['magnitude'])
        
        # Compute alignment
        alignment = self.compute_gradient_alignment(component_grads)
        self.alignment_history.append(alignment)
        
        return component_grads, alignment
    
    def compute_gradient_alignment(self, component_grads):
        """Compute cosine similarity between gradient vectors"""
        vectors = {k: v['vector'] for k, v in component_grads.items() if len(v['vector']) > 0}
        
        if len(vectors) < 2:
            return {}
        
        alignments = {}
        components = list(vectors.keys())
        
        for i in range(len(components)):
            for j in range(i+1, len(components)):
                comp1, comp2 = components[i], components[j]
                v1, v2 = vectors[comp1], vectors[comp2]
                
                # Pad to same length
                max_len = max(len(v1), len(v2))
                v1_pad = np.pad(v1, (0, max_len - len(v1)))
                v2_pad = np.pad(v2, (0, max_len - len(v2)))
                
                # Cosine similarity
                dot_product = np.dot(v1_pad, v2_pad)
                norm1 = np.linalg.norm(v1_pad)
                norm2 = np.linalg.norm(v2_pad)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    alignments[f'{comp1}_vs_{comp2}'] = float(similarity)
        
        return alignments
    
    def save_logs(self, filename='gradient_analysis.json'):
        """Save gradient logs to file"""
        # Convert numpy types to Python types for JSON serialization
        gradient_mags = {}
        for k, v in self.gradient_history.items():
            gradient_mags[k] = [float(x) for x in v]
        
        data = {
            'gradient_magnitudes': gradient_mags,
            'alignment_scores': self.alignment_history,
            'total_steps': self.step_count
        }
        
        with open(self.log_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Gradient logs saved to {self.log_dir / filename}")