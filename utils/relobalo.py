# utils/relobalo.py (update the clipping range and algorithm)
import torch
import numpy as np
from collections import deque

class ReLoBRaLo:
    """
    Relative Loss Balancing with Random Lookback
    Dynamically balances gradients across multiple loss components
    """
    
    def __init__(self, num_components, alpha=0.95, lookback_window=10):
        self.num_components = num_components
        self.alpha = alpha
        self.lookback_window = lookback_window
        
        self.grad_norm_history = [deque(maxlen=lookback_window) for _ in range(num_components)]
        self.ema_grad_norms = np.ones(num_components)
        
        # Start with equal weights
        self.weights = np.ones(num_components)
        self.update_count = 0
        
    def compute_gradient_norm(self, parameters):
        """Compute L2 norm of gradients"""
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return np.sqrt(total_norm)
    
    def update_weights(self, grad_norms):
        """Update loss weights based on gradient norms"""
        self.update_count += 1
        
        # Store gradient norms
        for i, norm in enumerate(grad_norms):
            self.grad_norm_history[i].append(norm)
            
            # Update EMA
            if self.update_count == 1:
                self.ema_grad_norms[i] = max(norm, 1e-8)
            else:
                self.ema_grad_norms[i] = (self.alpha * self.ema_grad_norms[i] + 
                                         (1 - self.alpha) * max(norm, 1e-8))
        
        # Target: average of all EMA norms (democratic balancing)
        target_norm = np.mean(self.ema_grad_norms)
        
        # If target is too small, use max instead
        if target_norm < 0.1:
            target_norm = np.max(self.ema_grad_norms)
        
        # Update weights: w_i = target / (ema_i + eps)
        eps = 1e-8
        for i in range(self.num_components):
            self.weights[i] = target_norm / (self.ema_grad_norms[i] + eps)
        
        # Clip to much wider range to handle vanishing/exploding gradients
        self.weights = np.clip(self.weights, 0.01, 1000.0)
        
        return self.weights
    
    def get_weights(self):
        """Get current loss weights"""
        return self.weights
    
    def get_statistics(self):
        """Get balancing statistics"""
        return {
            'weights': self.weights.tolist(),
            'ema_grad_norms': self.ema_grad_norms.tolist(),
            'update_count': self.update_count
        }