"""
Visualization 6: Kernel Real Part Attenuation Curves

Compares the decay characteristics of kernel real part Re(K(Δq))
for different power exponents.

Mathematical Background:
-------------------------
Real part of kernel: Re(K(Δq)) = Σ cos(ω_k · Δq)

Asymptotic behavior (Δq → ∞):
    For power-law frequencies ω_k ~ k^a:
        |Re(K)| ~ O(1/(Δq)^(1/a))
        
TCFMamba (a=3): O(1/(Δq)^(1/3)) - slow decay, maintains similarity at distance
Transformer (a=-1): Exponential decay - rapid locality

Advantage for POI recommendation:
- a=3 provides smooth attenuation over short-to-medium distances
- Matches typical POI visit patterns (nearby locations, not just immediate neighbors)

Reference: math.md Section 3 (Asymptotic behavior, power-law frequency analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from base import KernelFunction, FrequencyFunction, setup_plot_style, save_figure
from base.plot_utils import COLORS, POWER_COLORS
from base.viz_logger import VizLogger


def visualize_kernel_attenuation(output_dir: Path = None, html_dir: Path = None):
    """
    Generate kernel real part attenuation visualization.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    if html_dir is None:
        html_dir = Path(__file__).parent.parent / "html"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = VizLogger('6_kernel_real_attenuation')
    logger.set_description('Kernel Real Part Attenuation Curves (Asymptotic Analysis)')
    logger.add_finding('Asymptotic decay: |Re(K)| ~ O(1/(Δq)^(1/a))', 'theoretical')
    logger.add_finding('TCFMamba (a=3) provides slow decay suitable for POI recommendation', 'empirical')
    logger.add_finding('Transformer (a=-1) shows exponential decay with strong locality', 'comparison')
    logger.log_parameter('formula', 'Re(K(Δq)) = Σ cos(ω_k · Δq)')
    logger.log_parameter('asymptotic', '|Re(K)| ~ O(1/(Δq)^(1/a))')
    logger.log_parameter('dim', 64)
    logger.log_parameter('powers', [-1.0, 0.5, 1.0, 3.0])
    
    setup_plot_style(figsize=(14, 10))
    
    # Parameters
    dim = 64
    powers = [-1.0, 0.5, 1.0, 3.0]
    labels = ['Transformer (a=-1)', 'Sublinear (a=0.5)', 
              'Linear (a=1)', 'TCFMamba (a=3)']
    
    # Position difference range
    delta_q = np.linspace(0, 2000, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Full range linear scale
    ax1 = axes[0, 0]
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        kernel = KernelFunction(freq_func)
        K_real = kernel.compute_real(delta_q)
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax1.plot(delta_q, K_real, linewidth=2, label=label, color=color, alpha=0.8)
    
    ax1.set_xlabel('Position Difference Δq (meters)', fontsize=11)
    ax1.set_ylabel('Re(K(Δq))', fontsize=11)
    ax1.set_title('Kernel Real Part (Full Range)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    
    # Plot 2: Short range (0-500m) - POI relevant
    ax2 = axes[0, 1]
    delta_q_short = np.linspace(0, 500, 500)
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        kernel = KernelFunction(freq_func)
        K_real = kernel.compute_real(delta_q_short)
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax2.plot(delta_q_short, K_real, linewidth=2.5, label=label, color=color)
    
    ax2.set_xlabel('Position Difference Δq (meters)', fontsize=11)
    ax2.set_ylabel('Re(K(Δq))', fontsize=11)
    ax2.set_title('Kernel Real Part (Short Range: 0-500m)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    
    # Plot 3: Log-log scale for asymptotic analysis
    ax3 = axes[1, 0]
    delta_q_log = np.logspace(1, 4, 500)  # 10 to 10000 meters
    for power, label in zip(powers, labels):
        if power <= 0:
            continue  # Skip negative powers for log plot
        
        freq_func = FrequencyFunction(dim, power)
        kernel = KernelFunction(freq_func)
        K_real = np.abs(kernel.compute_real(delta_q_log))
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax3.loglog(delta_q_log, K_real + 0.1, linewidth=2, label=label, color=color)
        
        # Theoretical asymptotic line
        asymptotic = 1.0 / (delta_q_log ** (1.0/power))
        ax3.loglog(delta_q_log, asymptotic, '--', linewidth=1.5, 
                  color=color, alpha=0.5, label=f'{label} (theory)')
    
    ax3.set_xlabel('Position Difference Δq (meters, log)', fontsize=11)
    ax3.set_ylabel('|Re(K(Δq))| (log)', fontsize=11)
    ax3.set_title('Asymptotic Decay (Log-Log Scale)', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Half-life analysis (distance to reach 50% of max)
    ax4 = axes[1, 1]
    
    half_lives = []
    valid_powers = []
    valid_labels = []
    
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        kernel = KernelFunction(freq_func)
        K_real = kernel.compute_real(delta_q)
        
        max_val = K_real[0]
        half_val = max_val / 2
        
        # Find where kernel drops to 50%
        below_half = np.where(K_real < half_val)[0]
        if len(below_half) > 0:
            half_life = delta_q[below_half[0]]
            half_lives.append(half_life)
            valid_powers.append(power)
            valid_labels.append(label)
    
    colors_bar = [POWER_COLORS.get(p, COLORS['dark']) for p in valid_powers]
    bars = ax4.bar(range(len(valid_labels)), half_lives, color=colors_bar, alpha=0.7)
    ax4.set_xticks(range(len(valid_labels)))
    ax4.set_xticklabels([l.split('(')[0].strip() for l in valid_labels], rotation=15, ha='right')
    ax4.set_ylabel('Distance to 50% Kernel Value (meters)', fontsize=11)
    ax4.set_title('Kernel Decay Half-Life', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, half_lives):
        height = bar.get_height()
        ax4.annotate(f'{val:.0f}m',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Kernel Real Part Attenuation: Impact of Power Exponent\n' +
                 'Re(K(Δq)) = Σ cos(ω_k · Δq)  |  Decay: O(1/(Δq)^(1/a))',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_figure(fig, '6_kernel_real_attenuation', output_dir)
    logger.log_figure(output_dir / '6_kernel_real_attenuation.png',
                     title='Kernel Real Part Attenuation',
                     fig_type='multi_subplot')
    plt.close(fig)
    
    # Log all decay curves
    delta_q_full = np.linspace(0, 2000, 1000)
    delta_q_short = np.linspace(0, 500, 500)
    delta_q_log = np.logspace(1, 4, 500)
    
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        kernel = KernelFunction(freq_func)
        
        # Full range
        K_real_full = kernel.compute_real(delta_q_full)
        logger.log_series(f'decay_full_a{power}', delta_q_full, K_real_full,
                         x_label='Δq (meters)', y_label='Re(K(Δq))',
                         metadata={'power': power, 'range': 'full', 'label': label})
        
        # Short range
        K_real_short = kernel.compute_real(delta_q_short)
        logger.log_series(f'decay_short_a{power}', delta_q_short, K_real_short,
                         x_label='Δq (meters)', y_label='Re(K(Δq))',
                         metadata={'power': power, 'range': '0-500m', 'label': label})
        
        # Log half-life metric
        max_val = K_real_full[0]
        half_val = max_val / 2
        below_half = np.where(K_real_full < half_val)[0]
        if len(below_half) > 0:
            half_life = delta_q_full[below_half[0]]
            logger.log_metric(f'half_life_{power}', float(half_life),
                            unit='meters',
                            context={'power': power, 'label': label})
        
        # Log key values at specific distances
        for distance in [100, 500, 1000, 1500]:
            idx = int(distance / 2000 * len(delta_q_full))
            if idx < len(K_real_full):
                logger.log_metric(f'kernel_at_{distance}m_a{power}', float(K_real_full[idx]),
                                context={'power': power, 'distance': distance})
    
    # Log comparison
    logger.log_comparison('attenuation_comparison', [
        {'label': 'Transformer (a=-1)', 'value': -1.0, 'properties': {
            'decay_type': 'exponential', 'use_case': 'local attention', 'poi_suitability': 'low'}},
        {'label': 'Sublinear (a=0.5)', 'value': 0.5, 'properties': {
            'decay_type': 'slow_power', 'use_case': 'broad context', 'poi_suitability': 'medium'}},
        {'label': 'Linear (a=1)', 'value': 1.0, 'properties': {
            'decay_type': 'power_1', 'use_case': 'balanced', 'poi_suitability': 'medium'}},
        {'label': 'TCFMamba (a=3)', 'value': 3.0, 'properties': {
            'decay_type': 'power_1/3', 'use_case': 'POI recommendation', 'poi_suitability': 'high'}}
    ])
    
    # Save logger data
    logger.save()
    print("Visualization 6 completed: Kernel Real Part Attenuation")


if __name__ == '__main__':
    visualize_kernel_attenuation()
