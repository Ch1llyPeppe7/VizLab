"""
Visualization 7: Bochner Discrete Measure Pulse Diagram

Visualizes the discrete non-negative measure μ = Σ δ_{ω_k} that
generates the kernel via Bochner's theorem.

Mathematical Background:
-------------------------
Bochner's Theorem: K is positive definite ⟺ K(Δq) = ∫ exp(i·ω·Δq) dμ(ω)

For LAPE with discrete frequencies:
    μ = Σ_{k=1}^m δ_{ω_k}  (sum of Dirac deltas at frequency points)
    K(Δq) = Σ_{k=1}^m exp(i·ω_k·Δq)
    
This visualization shows:
1. The discrete measure μ as impulses at each frequency ω_k
2. How TCFMamba (a=3) concentrates measure at low frequencies
3. Comparison with Transformer (geometric) distribution

Key Insight:
The measure distribution directly determines the kernel properties.
TCFMamba's low-frequency dense distribution creates smooth,
long-range similarity kernels suitable for POI recommendation.

Reference: math.md Section 3 (Bochner theorem, discrete measure representation)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from base import FrequencyFunction, setup_plot_style, save_figure
from base.plot_utils import POWER_COLORS, COLORS
from base.viz_logger import VizLogger


def visualize_bochner_measure(output_dir: Path = None):
    """
    Generate Bochner discrete measure visualization.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = VizLogger('7_bochner_measure_visual')
    logger.set_description('Bochner Discrete Measure Pulse Diagram')
    logger.add_finding('Discrete measure μ = Σ δ_{ω_k} generates kernel via Bochner theorem', 'theoretical')
    logger.add_finding('TCFMamba concentrates measure at low frequencies for smooth kernels', 'empirical')
    logger.log_parameter('bochner_theorem', 'K(Δq) = ∫ exp(i·ω·Δq) dμ(ω) = Σ exp(i·ω_k·Δq)')
    logger.log_parameter('measure_formula', 'μ = Σ_{k=1}^m δ_{ω_k}')
    logger.log_parameter('dim', 64)
    
    setup_plot_style(figsize=(14, 10))
    
    # Parameters
    dim = 64
    powers = [-1.0, 1.0, 3.0]
    labels = ['Transformer (a=-1)', 'Linear (a=1)', 'TCFMamba (a=3)']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Stem plot of discrete measure
    ax1 = axes[0, 0]
    
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        freqs = freq_func.get_frequencies()
        k = np.arange(len(freqs))
        
        # All impulses have weight 1 (equal contribution)
        weights = np.ones_like(freqs)
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        # Matplotlib stem markerfmt only accepts simple formats, not hex colors
        # Use standard marker style without color specification
        markerline, stemlines, baseline = ax1.stem(
            k, weights, linefmt=color, markerfmt='o', 
            basefmt=' ', label=label
        )
        # Apply colors separately after creation
        plt.setp(markerline, 'color', color)
        plt.setp(stemlines, 'alpha', 0.5)
        plt.setp(markerline, 'markersize', 4)
    
    ax1.set_xlabel('Frequency Index k', fontsize=11)
    ax1.set_ylabel('Measure Weight μ(ω_k)', fontsize=11)
    ax1.set_title('Discrete Measure μ = Σ δ_{ω_k} (Equal Weights)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.5])
    
    # Plot 2: Frequency position stem plot
    ax2 = axes[0, 1]
    
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        freqs = freq_func.get_frequencies()
        k = np.arange(len(freqs))
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax2.scatter(k, freqs, color=color, s=30, label=label, alpha=0.7, zorder=3)
        
        # Connect with lines to show progression
        ax2.plot(k, freqs, color=color, linewidth=1, alpha=0.3)
    
    ax2.set_xlabel('Frequency Index k', fontsize=11)
    ax2.set_ylabel('Frequency Position ω_k', fontsize=11)
    ax2.set_title('Frequency Distribution (Frequency Space)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Measure density histogram
    ax3 = axes[1, 0]
    
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        freqs = np.abs(freq_func.get_frequencies())
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax3.hist(freqs, bins=20, alpha=0.4, label=label, color=color, 
                edgecolor='white', linewidth=0.5)
    
    ax3.set_xlabel('Frequency Value |ω|', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Measure Density Distribution', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative measure (showing how measure accumulates)
    ax4 = axes[1, 1]
    
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        freqs = np.abs(freq_func.get_frequencies())
        k = np.arange(len(freqs))
        
        # Cumulative measure
        cumulative = np.cumsum(np.ones_like(freqs))
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax4.plot(k, cumulative, linewidth=2.5, label=label, color=color, marker='o',
                markersize=3, markevery=8)
    
    ax4.set_xlabel('Frequency Index k', fontsize=11)
    ax4.set_ylabel('Cumulative Measure Σ_{i≤k} μ(ω_i)', fontsize=11)
    ax4.set_title('Cumulative Measure Distribution', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Add formula annotation
    fig.text(0.5, 0.02, 
             r"Bochner's Theorem: $K(\Delta q) = \int e^{i\omega\Delta q} d\mu(\omega)$" +
             r"  |  Discrete: $\mu = \sum_{k=1}^m \delta_{\omega_k}$",
             ha='center', fontsize=12, style='italic', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Bochner Discrete Measure Visualization\n' +
                 'μ = Σ δ_{ω_k}: Non-negative measure generating positive definite kernel',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    save_figure(fig, '7_bochner_measure_visual', output_dir)
    logger.log_figure(output_dir / '7_bochner_measure_visual.png',
                     title='Bochner Discrete Measure',
                     fig_type='multi_subplot')
    plt.close(fig)
    
    # Log measure data
    dim = 64
    powers = [-1.0, 0.5, 1.0, 3.0]
    labels = ['Transformer (a=-1)', 'Sublinear (a=0.5)', 'Linear (a=1)', 'TCFMamba (a=3)']
    
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        freqs = np.abs(freq_func.get_frequencies())
        k = np.arange(len(freqs))
        
        # Log frequency distribution
        logger.log_series(f'frequency_distribution_a{power}', k, freqs,
                         x_label='Frequency Index k', y_label='|ω_k|',
                         metadata={'power': power, 'label': label})
        
        # Log cumulative measure
        cumulative = np.cumsum(np.ones_like(freqs))
        logger.log_series(f'cumulative_measure_a{power}', k, cumulative,
                         x_label='Frequency Index k', y_label='Cumulative μ',
                         metadata={'power': power, 'label': label})
        
        # Log measure statistics
        logger.log_metric(f'freq_range_a{power}', float(np.max(freqs) - np.min(freqs)),
                         context={'power': power, 'label': label})
        logger.log_metric(f'low_freq_density_a{power}', float(np.sum(freqs < np.mean(freqs)) / len(freqs)),
                         unit='ratio',
                         context={'power': power, 'label': label})
    
    # Log comparison
    logger.log_comparison('measure_concentration', [
        {'label': 'Transformer (a=-1)', 'value': -1.0, 'properties': {
            'low_freq_concentration': 'low', 'high_freq_concentration': 'high', 'kernel_type': 'local'}},
        {'label': 'Sublinear (a=0.5)', 'value': 0.5, 'properties': {
            'low_freq_concentration': 'moderate', 'high_freq_concentration': 'moderate', 'kernel_type': 'medium'}},
        {'label': 'Linear (a=1)', 'value': 1.0, 'properties': {
            'low_freq_concentration': 'uniform', 'high_freq_concentration': 'uniform', 'kernel_type': 'periodic'}},
        {'label': 'TCFMamba (a=3)', 'value': 3.0, 'properties': {
            'low_freq_concentration': 'high', 'high_freq_concentration': 'low', 'kernel_type': 'smooth'}}
    ])
    
    # Save logger data
    logger.save()
    print("Visualization 7 completed: Bochner Measure Visualization")


if __name__ == '__main__':
    visualize_bochner_measure()
