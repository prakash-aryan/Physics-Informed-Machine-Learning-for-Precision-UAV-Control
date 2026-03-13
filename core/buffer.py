"""
PIATSG Framework - High-Performance Replay Buffer
Physics-Informed Adaptive Transformers with Safety Guarantees

Memory-optimized replay buffer with pinned memory for maximum training speed.
"""

import torch
import numpy as np
from collections import deque

class ReplayBuffer:
    """Fast replay buffer with memory pinning for maximum speed"""
    
    def __init__(self, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
        # Pre-allocate pinned memory tensors for maximum speed
        self.pinned_states = torch.zeros(batch_size, 18, pin_memory=True)
        self.pinned_actions = torch.zeros(batch_size, 4, pin_memory=True)
        self.pinned_rewards = torch.zeros(batch_size, 1, pin_memory=True)
        self.pinned_next_states = torch.zeros(batch_size, 18, pin_memory=True)
        self.pinned_dones = torch.zeros(batch_size, 1, pin_memory=True)
        
    def push(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Fast sampling with pre-allocated pinned memory"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Use pre-allocated pinned memory when possible
        if batch_size <= self.batch_size:
            states_tensor = self.pinned_states[:batch_size]
            actions_tensor = self.pinned_actions[:batch_size]
            rewards_tensor = self.pinned_rewards[:batch_size]
            next_states_tensor = self.pinned_next_states[:batch_size]
            dones_tensor = self.pinned_dones[:batch_size]
        else:
            # Fallback for larger batches
            states_tensor = torch.zeros(batch_size, 18, pin_memory=True)
            actions_tensor = torch.zeros(batch_size, 4, pin_memory=True)
            rewards_tensor = torch.zeros(batch_size, 1, pin_memory=True)
            next_states_tensor = torch.zeros(batch_size, 18, pin_memory=True)
            dones_tensor = torch.zeros(batch_size, 1, pin_memory=True)
        
        # Vectorized data loading
        for i, idx in enumerate(indices):
            states_tensor[i] = torch.from_numpy(self.buffer[idx][0])
            actions_tensor[i] = torch.from_numpy(self.buffer[idx][1])
            rewards_tensor[i, 0] = self.buffer[idx][2]
            next_states_tensor[i] = torch.from_numpy(self.buffer[idx][3])
            dones_tensor[i, 0] = self.buffer[idx][4]
        
        return (
            states_tensor.to(self.device, non_blocking=True),
            actions_tensor.to(self.device, non_blocking=True),
            rewards_tensor.to(self.device, non_blocking=True),
            next_states_tensor.to(self.device, non_blocking=True),
            dones_tensor.to(self.device, non_blocking=True)
        )
    
    def __len__(self):
        return len(self.buffer)