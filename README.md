# PIATSG Framework: Physics-Informed Adaptive Transformers with Safety Guarantees

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3+-green.svg)](https://mujoco.org/)

Official implementation of the paper:

> **Physics-informed machine learning for precision Unmanned Aerial Vehicle control: Adaptive transformers with safety guarantees**
>
> Prakash Aryan, Sebastiano Panichella
>
> *Engineering Applications of Artificial Intelligence*, Volume 172, 2026
>
> [DOI: 10.1016/j.engappai.2026.114379](https://doi.org/10.1016/j.engappai.2026.114379)

## Overview

PIATSG is a multi-component physics-informed framework for precision UAV control that integrates Physics-Informed Neural Networks (PINNs), Neural Operators (DeepONet), Decision Transformers, and Control Barrier Functions (CBFs) within a Soft Actor-Critic architecture. The framework achieves **91.9% precision within 10cm tolerance** while maintaining **99.88% physics consistency** and **100% safety compliance**.

### Key Results

| Metric | Value |
|--------|-------|
| 10 cm precision | 91.9% |
| 5 cm precision | 55.3% |
| 2 cm precision | 5.5% |
| Physics consistency | 99.88% |
| Safety compliance | 100.0% |
| Training time | 5.6 h (consumer GPU) |
| Improvement over best baseline (PPO) | +38.6 pp |

## Architecture

The framework integrates four physics-informed components within a unified SAC-based architecture (6.18M total parameters):

- **AdaptivePINN** (1.25M params) — Encodes Newton-Euler dynamics via automatic differentiation, enforcing physics consistency through PDE residual losses
- **Neural Operator** (366K params) — DeepONet branch-trunk architecture for function-to-function state prediction with residual connections
- **Decision Transformer** (2.46M params) — 8-head attention over 50-step history for temporal context generation at 100 Hz control frequency
- **Safety CBF** (246K params) — Control Barrier Functions with QP projection ensuring flight envelope constraints (altitude, position, velocity, tilt)

## Project Structure

```
piatsg_framework/
├── main.py                        # Training entry point
├── core/
│   ├── agent.py                   # PIATSGAgent with multi-component integration
│   ├── balanced_agent.py          # Extended agent with ReLoBRaLo gradient balancing
│   ├── components.py              # AdaptivePINN, NeuralOperator, SafetyCBF, DecisionTransformer, Actor
│   └── buffer.py                  # GPU-optimized replay buffer
├── simulation/
│   ├── environment.py             # MuJoCo environment (Crazyflie 2.0, 13D state, 4D action)
│   └── assets/                    # Quadrotor model meshes and scene XMLs
├── training/
│   ├── trainer.py                 # Three-phase curriculum training loop
│   └── evaluation.py              # Precision, physics, and safety metrics
├── utils/
│   ├── config.py                  # TrainingConfig with hardware auto-detection
│   ├── gradient_logger.py         # Component gradient tracking
│   └── relobalo.py                # Relative Loss Balancing with Random Lookback
├── experiments/
│   ├── ablation_study.py          # 8-configuration ablation (Table 4)
│   ├── curriculum_trainer.py      # Progressive horizon training (5s-120s)
│   ├── extended_horizon_evaluation.py  # Long-term stability analysis
│   ├── spatiotemporal_profiler.py      # Trajectory profiling with phase identification
│   ├── spatio_generate_paper_figure.py # Publication figure generation
│   ├── gradient_analysis.py            # Standard PIATSG gradient logging
│   ├── gradient_analysis_balanced.py   # Balanced gradient logging
│   └── generate_comparison_figure.py   # Gradient comparison plots
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/yourusername/piatsg_framework.git
cd piatsg_framework

# Create environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import mujoco; import torch; from core.agent import PIATSGAgent; print('OK')"
```

### Hardware Requirements

- **Minimum**: NVIDIA GPU with 6GB+ VRAM, 16GB RAM
- **Recommended**: NVIDIA RTX 3070+ (8GB+ VRAM), 32GB RAM
- **Paper results**: NVIDIA GeForce RTX 5070 Ti (16.6GB VRAM)

The framework auto-detects GPU memory and scales batch/buffer sizes accordingly.

## Usage

### Training

```bash
# Default training (5000 episodes, with viewer)
python main.py --episodes 5000

# Headless training (faster)
python main.py --episodes 5000 --no-viewer

# Custom configuration
python main.py --episodes 3000 --seed 42 --batch-size 1024 --buffer-size 200000
```

Training uses a three-phase curriculum (Section 3.7 of the paper):
1. **Phase 1** (eps 0-1667): RL stabilization with minimal physics weights
2. **Phase 2** (eps 1668-3334): Progressive physics integration via linear scheduling
3. **Phase 3** (eps 3335-5000): Physics refinement with maximum constraint weights

### Ablation Study

Reproduces Table 4 from the paper — systematic evaluation across 8 component configurations:

```bash
# Full ablation study (8 configs, 20 episodes each)
python experiments/ablation_study.py --model models/best_physics.pth --episodes 20

# Single configuration test
python experiments/ablation_study.py --model models/best_physics.pth --single "Without Safety"
```

Configurations: Full PIATSG, w/o PINN, w/o NeuralOp, w/o Safety, w/o DT, w/o PINN+NeuralOp, w/o PINN+NeuralOp+Safety, Baseline RL.

### Curriculum Training

Progressive horizon training (5s -> 15s -> 30s -> 60s -> 120s):

```bash
python experiments/curriculum_trainer.py --output curriculum_training
```

### Extended Horizon Evaluation

Tests long-term stability at 5s, 30s, 60s, and 120s horizons:

```bash
python experiments/extended_horizon_evaluation.py --model models/best_precision_5cm.pth
```

### Gradient Analysis

Logs and compares gradient magnitudes across components:

```bash
# Standard PIATSG
python experiments/gradient_analysis.py

# With ReLoBRaLo gradient balancing
python experiments/gradient_analysis_balanced.py

# Generate comparison figures (requires gradient logs)
python experiments/generate_comparison_figure.py
```

### Spatiotemporal Profiling

Episode trajectory analysis with maneuver phase identification:

```bash
python experiments/spatiotemporal_profiler.py --model models/best_precision_5cm.pth
python experiments/spatio_generate_paper_figure.py --model models/best_precision_5cm.pth
```

## Reproducing Paper Results

```bash
# Step 1: Train full PIATSG (5000 episodes, ~5.6h)
python main.py --episodes 5000 --seed 42 --no-viewer

# Step 2: Run ablation study (Table 4)
python experiments/ablation_study.py --model models/best_physics.pth --episodes 20 --seed 42

# Step 3: Extended horizon evaluation
python experiments/extended_horizon_evaluation.py --model models/best_precision_5cm.pth

# Step 4: Spatiotemporal analysis (Fig. 16)
python experiments/spatio_generate_paper_figure.py --model models/best_precision_5cm.pth
```

## Configuration

Key hyperparameters in `utils/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `actor_lr` | 3e-4 | Actor learning rate (Adam) |
| `critic_lr` | 3e-4 | Critic learning rate (Adam) |
| `pinn_lr` | 5e-5 | PINN learning rate (AdamW) |
| `operator_lr` | 5e-5 | Neural Operator learning rate (AdamW) |
| `safety_lr` | 3e-5 | Safety CBF learning rate (AdamW) |
| `hidden_dim` | 1536 | Hidden layer dimension |
| `batch_size` | 2048 | Training batch size (auto-scaled) |
| `buffer_size` | 400000 | Replay buffer capacity (auto-scaled) |
| `tau` | 0.005 | Soft target update rate |
| `gamma` | 0.99 | Discount factor |
| `alpha` | 0.2 | SAC entropy temperature |

## Citation

```bibtex
@article{aryan2026piatsg,
  title={Physics-informed machine learning for precision Unmanned Aerial Vehicle control: Adaptive transformers with safety guarantees},
  author={Aryan, Prakash and Panichella, Sebastiano},
  journal={Engineering Applications of Artificial Intelligence},
  volume={172},
  pages={114379},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.engappai.2026.114379}
}
```

