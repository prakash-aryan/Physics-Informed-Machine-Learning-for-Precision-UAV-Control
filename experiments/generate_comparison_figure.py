# experiments/generate_comparison_figure.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_gradient_data(json_path):
    """Load gradient data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def plot_gradient_comparison(without_balance_path, with_balance_path, save_path):
    """Generate side-by-side comparison of gradient magnitudes"""
    
    # Load data
    without_data = load_gradient_data(without_balance_path)
    with_data = load_gradient_data(with_balance_path)
    
    # Setup publication style
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    components = ['actor', 'pinn', 'operator', 'safety']
    colors = {'actor': '#1f77b4', 'pinn': '#ff7f0e', 'operator': '#2ca02c', 'safety': '#d62728'}
    
    # Plot without balancing (left)
    ax1 = axes[0]
    for comp in components:
        key = f'{comp}_magnitude'
        if key in without_data['gradient_magnitudes']:
            data = without_data['gradient_magnitudes'][key]
            ax1.plot(data, label=comp.upper(), alpha=0.8, linewidth=2, color=colors[comp])
    
    ax1.set_xlabel('Training Step', fontsize=11)
    ax1.set_ylabel('Gradient Magnitude', fontsize=11)
    ax1.set_title('(a) Without Gradient Balancing', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    
    # Add text annotation showing imbalance
    ax1.text(0.05, 0.95, 'PINN dominates\n(~10x actor)', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot with balancing (right)
    ax2 = axes[1]
    for comp in components:
        key = f'{comp}_magnitude'
        if key in with_data['gradient_magnitudes']:
            data = with_data['gradient_magnitudes'][key]
            ax2.plot(data, label=comp.upper(), alpha=0.8, linewidth=2, color=colors[comp])
    
    ax2.set_xlabel('Training Step', fontsize=11)
    ax2.set_ylabel('Gradient Magnitude', fontsize=11)
    ax2.set_title('(b) With ReLoBRaLo Balancing', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    
    # Add text annotation showing improvement
    ax2.text(0.05, 0.95, 'Balanced gradients\n(weights adapt)', 
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison figure saved to {save_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("GRADIENT STATISTICS")
    print("="*60)
    
    print("\nWithout Balancing:")
    for comp in components:
        key = f'{comp}_magnitude'
        if key in without_data['gradient_magnitudes']:
            data = without_data['gradient_magnitudes'][key]
            valid_data = [x for x in data if x > 0]
            if valid_data:
                print(f"  {comp.upper():8s}: mean={np.mean(valid_data):.3f}, "
                      f"max={np.max(valid_data):.3f}, min={np.min(valid_data):.3f}")
    
    print("\nWith ReLoBRaLo:")
    for comp in components:
        key = f'{comp}_magnitude'
        if key in with_data['gradient_magnitudes']:
            data = with_data['gradient_magnitudes'][key]
            valid_data = [x for x in data if x > 0]
            if valid_data:
                print(f"  {comp.upper():8s}: mean={np.mean(valid_data):.3f}, "
                      f"max={np.max(valid_data):.3f}, min={np.min(valid_data):.3f}")
    
    print("\n" + "="*60)

def plot_alignment_comparison(without_balance_path, with_balance_path, save_path):
    """Generate comparison of gradient alignment scores"""
    
    # Load data
    without_data = load_gradient_data(without_balance_path)
    with_data = load_gradient_data(with_balance_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot without balancing (left)
    ax1 = axes[0]
    if without_data['alignment_scores']:
        all_keys = set()
        for align_dict in without_data['alignment_scores']:
            all_keys.update(align_dict.keys())
        
        for key in all_keys:
            values = [d.get(key, np.nan) for d in without_data['alignment_scores']]
            ax1.plot(values, label=key.replace('_', ' '), alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Training Step', fontsize=11)
    ax1.set_ylabel('Cosine Similarity', fontsize=11)
    ax1.set_title('(a) Without Gradient Balancing', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    
    # Plot with balancing (right)
    ax2 = axes[1]
    if with_data['alignment_scores']:
        all_keys = set()
        for align_dict in with_data['alignment_scores']:
            all_keys.update(align_dict.keys())
        
        for key in all_keys:
            values = [d.get(key, np.nan) for d in with_data['alignment_scores']]
            ax2.plot(values, label=key.replace('_', ' '), alpha=0.7, linewidth=1.5)
    
    ax2.set_xlabel('Training Step', fontsize=11)
    ax2.set_ylabel('Cosine Similarity', fontsize=11)
    ax2.set_title('(b) With ReLoBRaLo Balancing', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9, fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Alignment comparison saved to {save_path}")

if __name__ == '__main__':
    # Paths to JSON files
    without_balance = 'gradient_logs/full_piatsg/full_piatsg_gradients.json'
    with_balance = 'gradient_logs/balanced_piatsg/balanced_piatsg_gradients.json'
    
    # Generate figures
    plot_gradient_comparison(
        without_balance, 
        with_balance,
        'figures/gradient_magnitude_comparison.pdf'
    )
    
    plot_alignment_comparison(
        without_balance,
        with_balance,
        'figures/gradient_alignment_comparison.pdf'
    )
    
    print("\nFigures generated successfully!")
    print("Use gradient_magnitude_comparison.pdf for the main paper figure.")