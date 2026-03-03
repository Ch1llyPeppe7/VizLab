"""
Visualization 5: Frequency Distribution Comparison for Different Power Exponents

This script compares frequency distributions for different values of power exponent 'a':
- a=3: TCFMamba (superlinear, low-frequency dense)
- a=1: Linear distribution
- a=0.5: Sublinear distribution  
- a=-1: Transformer (geometric distribution)

Mathematical Background:
-------------------------
Frequency function: ω_k = (-k/d)^a

The power exponent 'a' determines:
1. Frequency spacing pattern
2. Kernel decay rate: O(1/(Δq)^(1/a))
3. Spatial resolution characteristics

TCFMamba uses a=3 for:
- Dense low frequencies (capture coarse spatial patterns)
- Sparse high frequencies (capture fine details when needed)
- Smooth kernel decay (better for POI recommendation)

Reference: math.md Section 2 (TCFMamba frequency function definition)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from base import FrequencyFunction, setup_plot_style, save_figure
from base.plot_utils import COLORS, POWER_COLORS, create_interactive_plot
from base.viz_logger import VizLogger


def visualize_frequency_distribution(output_dir: Path = None, html_dir: Path = None):
    """
    Compare frequency distributions for different power exponents.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    if html_dir is None:
        html_dir = Path(__file__).parent.parent / "html"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    
    setup_plot_style(figsize=(14, 10))
    
    # Parameters
    dim = 64
    powers = [-1.0, 0.5, 1.0, 3.0]
    labels = ['Transformer (a=-1)', 'Sublinear (a=0.5)', 
              'Linear (a=1)', 'TCFMamba (a=3)']
    
    # Initialize logger
    logger = VizLogger('5_frequency_distribution_compare')
    logger.set_description('Frequency Distribution Comparison for Different Power Exponents')
    logger.add_finding('Power exponent a determines frequency spacing pattern', 'theoretical')
    logger.add_finding('TCFMamba (a=3): dense low frequencies, sparse high frequencies', 'empirical')
    logger.add_finding('Transformer (a=-1): geometric spacing, exponential decay', 'comparison')
    logger.log_parameter('frequency_formula', 'ω_k = (-k/d)^a')
    logger.log_parameter('dim', dim)
    logger.log_parameter('powers', powers)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Linear scale comparison
    ax1 = axes[0, 0]
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        freqs = freq_func.get_frequencies()
        k = np.arange(len(freqs))
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax1.plot(k, freqs, linewidth=2.5, label=label, color=color, marker='o', 
                markersize=3, markevery=8)
    
    ax1.set_xlabel('Frequency Index k', fontsize=11)
    ax1.set_ylabel('Frequency ω_k', fontsize=11)
    ax1.set_title('Frequency Distribution (Linear Scale)', fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale for better comparison
    ax2 = axes[0, 1]
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        freqs = np.abs(freq_func.get_frequencies()) + 1e-10  # Avoid log(0)
        k = np.arange(len(freqs))
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax2.semilogy(k, freqs, linewidth=2.5, label=label, color=color, marker='o',
                    markersize=3, markevery=8)
    
    ax2.set_xlabel('Frequency Index k', fontsize=11)
    ax2.set_ylabel('|Frequency ω_k| (log scale)', fontsize=11)
    ax2.set_title('Frequency Distribution (Log Scale)', fontsize=12)
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Frequency spacing (derivative approximation)
    ax3 = axes[1, 0]
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        freqs = freq_func.get_frequencies()
        spacing = np.diff(freqs)
        k = np.arange(len(spacing))
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax3.plot(k, np.abs(spacing), linewidth=2, label=label, color=color, alpha=0.8)
    
    ax3.set_xlabel('Frequency Index k', fontsize=11)
    ax3.set_ylabel('|ω_{k+1} - ω_k| (Frequency Spacing)', fontsize=11)
    ax3.set_title('Frequency Spacing', fontsize=12)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Cumulative frequency distribution
    ax4 = axes[1, 1]
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        freqs = np.abs(freq_func.get_frequencies())
        cumsum = np.cumsum(freqs)
        cumsum_normalized = cumsum / cumsum[-1]  # Normalize to [0, 1]
        k = np.arange(len(freqs))
        
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax4.plot(k, cumsum_normalized, linewidth=2.5, label=label, color=color)
    
    ax4.set_xlabel('Frequency Index k', fontsize=11)
    ax4.set_ylabel('Cumulative Frequency Ratio', fontsize=11)
    ax4.set_title('Cumulative Frequency Distribution', fontsize=12)
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    # Add formula annotation
    fig.text(0.5, 0.02, r'Frequency Function: $\omega_k = (-k/d)^a$' + 
             r'  |  TCFMamba: $a=3$, Transformer: $a=-1$',
             ha='center', fontsize=12, style='italic')
    
    plt.suptitle('Frequency Distribution Comparison: Impact of Power Exponent a\n' +
                 'ω_k = (-k/d)^a determines frequency spacing and kernel behavior',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save figure
    save_figure(fig, '5_frequency_distribution_compare', output_dir)
    logger.log_figure(output_dir / '5_frequency_distribution_compare.png',
                     title='Frequency Distribution Comparison',
                     fig_type='multi_subplot')
    plt.close(fig)
    
    # Log frequency data series
    for power, label in zip(powers, labels):
        freq_func = FrequencyFunction(dim, power)
        freqs = freq_func.get_frequencies()
        k = np.arange(len(freqs))
        
        logger.log_series(f'frequency_linear_a{power}', k, freqs,
                         x_label='Frequency Index k', y_label='ω_k',
                         metadata={'power': power, 'label': label, 'scale': 'linear'})
        
        # Log spacing
        spacing = np.abs(np.diff(freqs))
        logger.log_series(f'frequency_spacing_a{power}', k[:-1], spacing,
                         x_label='Frequency Index k', y_label='Δω',
                         metadata={'power': power, 'label': label})
        
        # Log statistics
        logger.log_metric(f'freq_max_a{power}', float(np.max(freqs)),
                         context={'power': power, 'label': label})
        logger.log_metric(f'freq_min_a{power}', float(np.min(freqs)),
                         context={'power': power, 'label': label})
        logger.log_metric(f'freq_mean_a{power}', float(np.mean(freqs)),
                         context={'power': power, 'label': label})
        
        # Log array
        logger.log_array(f'frequencies_a{power}', freqs,
                        metadata={'power': power, 'label': label})
    
    # Log comparison
    logger.log_comparison('frequency_pattern_comparison', [
        {'label': 'Transformer (a=-1)', 'value': -1.0, 'properties': {
            'spacing': 'geometric', 'low_freq_density': 'sparse', 'high_freq_density': 'dense'}},
        {'label': 'Sublinear (a=0.5)', 'value': 0.5, 'properties': {
            'spacing': 'sublinear', 'low_freq_density': 'moderate', 'high_freq_density': 'moderate'}},
        {'label': 'Linear (a=1)', 'value': 1.0, 'properties': {
            'spacing': 'linear', 'low_freq_density': 'uniform', 'high_freq_density': 'uniform'}},
        {'label': 'TCFMamba (a=3)', 'value': 3.0, 'properties': {
            'spacing': 'superlinear', 'low_freq_density': 'dense', 'high_freq_density': 'sparse'}}
    ])
    
    # Create interactive HTML
    create_interactive_frequency_html(html_dir, dim, powers, labels)
    logger.log_figure(html_dir / '5_frequency_distribution_compare.html',
                     title='Interactive Frequency Distribution',
                     fig_type='interactive_html')
    
    # Save logger data
    logger.save()
    print("Visualization 5 completed: Frequency Distribution Comparison")


def create_interactive_frequency_html(html_dir: Path, dim: int, powers: list, labels: list):
    """Create interactive HTML for frequency comparison."""
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not available, skipping interactive HTML")
        return
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Linear Scale', 'Log Scale', 
                                      'Frequency Spacing', 'Cumulative Distribution'),
                       specs=[[{}, {}], [{}, {}]])
    
    colors_plotly = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    
    for i, (power, label, color) in enumerate(zip(powers, labels, colors_plotly)):
        freq_func = FrequencyFunction(dim, power)
        freqs = freq_func.get_frequencies()
        k = np.arange(len(freqs))
        
        # Linear scale
        fig.add_trace(go.Scatter(x=k, y=freqs, mode='lines+markers',
                                name=label, line=dict(color=color, width=2),
                                marker=dict(size=4)), row=1, col=1)
        
        # Log scale
        fig.add_trace(go.Scatter(x=k, y=np.abs(freqs)+1e-10, mode='lines',
                                name=label, line=dict(color=color, width=2),
                                showlegend=False), row=1, col=2)
        
        # Spacing
        spacing = np.abs(np.diff(freqs))
        fig.add_trace(go.Scatter(x=k[:-1], y=spacing, mode='lines',
                                name=label, line=dict(color=color, width=2),
                                showlegend=False), row=2, col=1)
        
        # Cumulative
        cumsum = np.cumsum(np.abs(freqs))
        cumsum_norm = cumsum / cumsum[-1]
        fig.add_trace(go.Scatter(x=k, y=cumsum_norm, mode='lines',
                                name=label, line=dict(color=color, width=2),
                                showlegend=False), row=2, col=2)
    
    fig.update_layout(
        title='<b>Frequency Distribution Comparison</b><br>ω_k = (-k/d)^a',
        height=700,
        hovermode='closest'
    )
    
    fig.update_yaxes(type='log', row=1, col=2)
    fig.update_yaxes(type='log', row=2, col=1)
    
    html_path = html_dir / "5_frequency_distribution_compare.html"
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    print(f"Saved interactive HTML: {html_path}")


if __name__ == '__main__':
    visualize_frequency_distribution()
