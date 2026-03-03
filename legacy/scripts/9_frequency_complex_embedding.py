"""
Visualization 9: Frequency Function Impact on Kähler Complex Plane Embedding

Demonstrates how different frequency functions affect the complex plane
embedding z(q) = exp(i·ω·q) and the resulting Kähler manifold structure.

Mathematical Background:
-------------------------
Kähler manifold structure depends on how frequencies map positions to
complex planes:

1. Embedding: z_k(q) = e^(i·ω_k·q) for each frequency component k
2. Rotation speed: Higher ω → faster rotation → more "winding"
3. Kähler metric: g(z_1, z_2) = ⟨z_1, z_2⟩ preserves angles

Different frequency distributions create different embedding geometries:
- TCFMamba (a=3): Low frequencies dominate → smooth, slow-varying embedding
- Transformer: Spread frequencies → complex, fast-varying embedding

Visualization shows:
1. Complex plane trajectories for sample frequencies
2. Embedding space coverage comparison
3. Distance preservation in embedding space

Reference: math.md Section 4 (Kähler structure, embedding properties)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from base import LAPEEncoder, setup_plot_style, save_figure
from base.plot_utils import POWER_COLORS, COLORS
from base.viz_logger import VizLogger


def visualize_frequency_complex_embedding(output_dir: Path = None):
    """
    Generate visualization of frequency impact on complex embedding.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = VizLogger('9_frequency_complex_embedding')
    logger.set_description('Frequency Impact on Kähler Complex Plane Embedding')
    logger.add_finding('Different frequency distributions create different embedding geometries', 'theoretical')
    logger.add_finding('TCFMamba (a=3): Low frequencies dominate → smooth, slow-varying embedding', 'empirical')
    logger.log_parameter('embedding_formula', 'z_k(q) = e^(i·ω_k·q)')
    logger.log_parameter('kahler_metric', 'g(z_1, z_2) = ⟨z_1, z_2⟩ preserves angles')
    logger.log_parameter('dim', 64)
    
    setup_plot_style(figsize=(14, 12))
    
    # Position values
    q_values = np.linspace(0, 1000, 300)
    
    # Select frequencies to visualize
    freq_indices = [0, 8, 16, 24]  # Low, med-low, med-high, high
    
    # Three power exponents
    powers = [-1.0, 1.0, 3.0]
    titles = ['Transformer (a=-1)', 'Linear (a=1)', 'TCFMamba (a=3)']
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    
    for row_idx, (power, title) in enumerate(zip(powers, titles)):
        encoder = LAPEEncoder(dim=64, power=power)
        
        # Plot 1: Complex plane trajectories (left column)
        ax_complex = axes[row_idx, 0]
        
        colors_freq = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        for idx, freq_idx in enumerate(freq_indices):
            # Get frequency value
            omega = encoder.frequencies[freq_idx]
            
            # Compute complex embedding for each position
            z_real = np.cos(omega * q_values / encoder.scale)
            z_imag = np.sin(omega * q_values / encoder.scale)
            
            # Plot trajectory with color gradient
            ax_complex.plot(z_real, z_imag, linewidth=1.5, 
                          label=f'k={freq_idx}, ω={omega:.4f}',
                          color=colors_freq[idx], alpha=0.8)
            
            # Mark starting point
            ax_complex.scatter([1], [0], color=colors_freq[idx], s=80, zorder=5)
        
        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', 
                           linestyle='--', alpha=0.3)
        ax_complex.add_patch(circle)
        
        ax_complex.set_xlim([-1.3, 1.3])
        ax_complex.set_ylim([-1.3, 1.3])
        ax_complex.set_aspect('equal')
        ax_complex.axhline(y=0, color='gray', linewidth=0.5, alpha=0.3)
        ax_complex.axvline(x=0, color='gray', linewidth=0.5, alpha=0.3)
        ax_complex.set_xlabel('Real: cos(ω·q)', fontsize=10)
        ax_complex.set_ylabel('Imag: sin(ω·q)', fontsize=10)
        ax_complex.set_title(f'{title}\nComplex Plane Trajectories', fontsize=11)
        ax_complex.legend(fontsize=8, loc='upper right')
        ax_complex.grid(True, alpha=0.2)
        
        # Plot 2: Phase accumulation (right column)
        ax_phase = axes[row_idx, 1]
        
        for idx, freq_idx in enumerate(freq_indices):
            omega = encoder.frequencies[freq_idx]
            
            # Phase = ω·q (mod 2π)
            phase = (omega * q_values / encoder.scale) % (2 * np.pi)
            
            ax_phase.plot(q_values, phase, linewidth=2,
                         label=f'k={freq_idx}, ω={omega:.4f}',
                         color=colors_freq[idx], alpha=0.8)
        
        ax_phase.set_xlabel('Position q (meters)', fontsize=10)
        ax_phase.set_ylabel('Phase φ = ω·q (mod 2π)', fontsize=10)
        ax_phase.set_title(f'{title}\nPhase Accumulation', fontsize=11)
        ax_phase.legend(fontsize=8, loc='upper left')
        ax_phase.grid(True, alpha=0.3)
        ax_phase.set_ylim([0, 2*np.pi])
        
        # Add 2π reference line
        ax_phase.axhline(y=2*np.pi, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    
    plt.suptitle('Frequency Function Impact on Complex Plane Embedding\n' +
                 'z(q) = e^(i·ω·q): Different frequencies create different embedding geometries',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_figure(fig, '9_frequency_complex_embedding', output_dir)
    logger.log_figure(output_dir / '9_frequency_complex_embedding.png',
                     title='Frequency Complex Plane Embedding',
                     fig_type='multi_subplot')
    plt.close(fig)
    
    # Log embedding data
    q_values = np.linspace(0, 1000, 300)
    freq_indices = [0, 8, 16, 24]
    powers = [-1.0, 1.0, 3.0]
    titles = ['Transformer (a=-1)', 'Linear (a=1)', 'TCFMamba (a=3)']
    
    for power, title in zip(powers, titles):
        encoder = LAPEEncoder(dim=64, power=power)
        freqs = encoder.frequencies
        
        for idx in freq_indices:
            if idx < len(freqs):
                freq = freqs[idx]
                z_real = np.cos(freq * q_values)
                z_imag = np.sin(freq * q_values)
                phase = np.mod(freq * q_values, 2 * np.pi)
                
                logger.log_series(f'embedding_a{power}_freq{idx}_real', q_values, z_real,
                                 x_label='Position q', y_label='Re(z)',
                                 metadata={'power': power, 'freq_idx': idx, 'freq': freq})
                logger.log_series(f'embedding_a{power}_freq{idx}_phase', q_values, phase,
                                 x_label='Position q', y_label='Phase',
                                 metadata={'power': power, 'freq_idx': idx, 'freq': freq})
        
        # Log frequency statistics
        logger.log_metric(f'freq_mean_a{power}', float(np.mean(np.abs(freqs))),
                         context={'power': power, 'title': title})
        logger.log_metric(f'freq_max_a{power}', float(np.max(np.abs(freqs))),
                         context={'power': power, 'title': title})
    
    # Log comparison
    logger.log_comparison('embedding_geometries', [
        {'label': 'Transformer (a=-1)', 'value': -1.0, 'properties': {
            'geometry': 'complex_fast', 'embedding_type': 'high_winding', 'smoothness': 'low'}},
        {'label': 'Linear (a=1)', 'value': 1.0, 'properties': {
            'geometry': 'balanced', 'embedding_type': 'uniform', 'smoothness': 'medium'}},
        {'label': 'TCFMamba (a=3)', 'value': 3.0, 'properties': {
            'geometry': 'smooth_slow', 'embedding_type': 'low_frequency_dominant', 'smoothness': 'high'}}
    ])
    
    # Create rotation speed comparison
    create_rotation_comparison(output_dir)
    logger.log_figure(output_dir / '9_rotation_speed_comparison.png',
                     title='Rotation Speed Comparison',
                     fig_type='comparison')
    
    # Save logger data
    logger.save()
    print("Visualization 9 completed: Frequency Complex Embedding")


def create_rotation_comparison(output_dir: Path):
    """Create comparison of rotation speeds across frequency distributions."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    dim = 64
    powers = [-1.0, 1.0, 3.0]
    labels = ['Transformer', 'Linear', 'TCFMamba']
    
    for ax, power, label in zip(axes, powers, labels):
        encoder = LAPEEncoder(dim=dim, power=power)
        
        # Rotation speed = |ω| (angular velocity)
        freqs = np.abs(encoder.frequencies)
        k = np.arange(len(freqs))
        
        # Plot rotation speed
        ax.bar(k, freqs, color=POWER_COLORS.get(power, COLORS['dark']), 
              alpha=0.7, edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Frequency Index k', fontsize=10)
        ax.set_ylabel('|ω_k| (Angular Velocity)', fontsize=10)
        ax.set_title(f'{label} (a={power})\nRotation Speed Distribution', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        avg_speed = np.mean(freqs)
        max_speed = np.max(freqs)
        ax.text(0.95, 0.95, f'Mean: {avg_speed:.3f}\nMax: {max_speed:.3f}',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Rotation Speed Comparison: |ω_k| = Angular Velocity in Complex Plane',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    save_figure(fig, '9_rotation_speed_comparison', output_dir)
    plt.close(fig)


if __name__ == '__main__':
    visualize_frequency_complex_embedding()
