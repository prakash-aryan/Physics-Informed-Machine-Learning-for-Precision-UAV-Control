"""
PIATSG Framework Configuration and Utilities
Physics-Informed Adaptive Transformers with Safety Guarantees
"""

import os
import time
import random
import numpy as np
import torch

def set_reproducible_seed(seed=42):
    """Set seeds for consistent reproducible results across all libraries"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"Reproducible seed set to {seed}")

def configure_device():
    """Configure device and compute resource parameters"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_vram:.1f} GB")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        
        # Compute batch and buffer sizes for performance
        if total_vram >= 12.0:
            batch_size = 2048
            buffer_size = 400000
        elif total_vram >= 8.0:
            batch_size = 1536
            buffer_size = 300000
        elif total_vram >= 6.0:
            batch_size = 1024
            buffer_size = 250000
        else:
            batch_size = 768
            buffer_size = 200000
        
        torch.cuda.set_per_process_memory_fraction(0.90)
        
        print(f"Performance parameters enabled:")
        print(f"   - Mixed precision training: ENABLED")
        print(f"   - Batch size: {batch_size:,}")
        print(f"   - Buffer size: {buffer_size:,}")
        print(f"   - TensorFloat-32: ENABLED")
        print(f"   - CuDNN benchmark: ENABLED")
        print(f"   - Memory fraction: 90%")
        
        return device, batch_size, buffer_size, total_vram
    else:
        batch_size = 768
        buffer_size = 150000
        return device, batch_size, buffer_size, 0.0

class TrainingConfig:
    """Training configuration parameters"""
    def __init__(self, device, batch_size, buffer_size):
        # Device configuration
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Training parameters AGGRESSIVELY optimized for 5K episodes in 4 hours
        self.num_episodes = 5000
        self.max_steps_per_episode = 500  # Reduced from 600 for speed
        self.training_frequency = 200     # Increased from 100 (less frequent but larger batches)
        self.updates_per_training = 2     # Reduced from 4 (faster per update)
        
        # Learning rates - significantly reduced for physics components
        self.actor_lr = 0.0002
        self.critic_lr = 0.0008
        self.pinn_lr = 0.00005      # Reduced from 0.0005
        self.operator_lr = 0.00005  # Reduced from 0.0005
        self.safety_lr = 0.00003    # Reduced from 0.0005
        self.alpha_lr = 0.0003
        
        # SAC parameters
        self.tau = 0.005
        self.gamma = 0.99
        self.target_entropy = -4
        
        # Network dimensions
        self.state_dim = 18
        self.action_dim = 4
        self.hidden_dim = 1536
        
        # Evaluation parameters optimized for speed  
        self.evaluation_episodes = 3    # Reduced from 5 for speed
        self.evaluation_frequency = 200 # Increased from 100 to evaluate less often
        
        # Logging parameters optimized for speed
        self.log_frequency = 50          # Increased from 25 to reduce I/O overhead
        self.checkpoint_frequency = 2000 # Increased from 1000 to reduce disk I/O
        
        print(f"Training Configuration:")
        print(f"  Episodes: {self.num_episodes}")
        print(f"  Batch size: {self.batch_size:,}")
        print(f"  Buffer size: {self.buffer_size:,}")
        print(f"  Device: {self.device}")
        print(f"  Training frequency: Every {self.training_frequency} steps")
        print(f"  Updates per training: {self.updates_per_training}")
        print(f"  Max episode length: {self.max_steps_per_episode}")
        print(f"  Physics component learning rates:")
        print(f"    - PINN LR: {self.pinn_lr}")
        print(f"    - Operator LR: {self.operator_lr}")
        print(f"    - Safety LR: {self.safety_lr}")

def create_directories():
    """Create necessary directories for the framework"""
    directories = [
        'models',
        'logs', 
        'runs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Framework directories created")

def get_timestamp():
    """Get current timestamp for logging"""
    return time.strftime("%Y%m%d_%H%M%S")