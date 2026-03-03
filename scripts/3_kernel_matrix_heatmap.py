"""
Visualization 3: Positive Definite Kernel Matrix Heatmap

This script visualizes the kernel matrix K_{ij} = Re(K(q_i - q_j))
which is positive definite by Bochner's theorem.

Mathematical Background:
-------------------------
Bochner's Theorem states that a continuous function K is positive definite
if and only if it is the Fourier transform of a non-negative measure:

    K(Δq) = ∫ exp(i·ω·Δq) dμ(ω)

For LAPE with discrete frequencies {ω_k}:
    K(Δq) = Σ_{k=1}^m exp(i·ω_k·Δq)
    
Real Part (similarity metric):
    Re(K(Δq)) = Σ_{k=1}^m cos(ω_k·Δq)

Kernel Matrix:
    K_{ij} = Re(K(q_i - q_j))

Properties of K matrix:
1. Symmetric: K_{ij} = K_{ji}
2. Positive semi-definite: x^T K x ≥ 0 for all x
3. Diagonal dominance: K_{ii} = m (maximum value)

Reference: math.md Section 3 (Bochner theorem, kernel properties)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add base module to path
sys.path.insert(0, str(Path(__file__).parent))
from base import KernelFunction, FrequencyFunction, setup_plot_style, save_figure
from base.plot_utils import create_heatmap_html, COLORS, POWER_COLORS
from base.viz_logger import VizLogger


def visualize_kernel_matrix(output_dir: Path = None, html_dir: Path = None):
    """
    Generate kernel matrix heatmap visualization.
    
    Shows the positive definite kernel matrix for different frequency distributions,
    highlighting the structure imposed by Bochner's theorem.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    if html_dir is None:
        html_dir = Path(__file__).parent.parent / "html"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = VizLogger('3_kernel_matrix_heatmap')
    logger.set_description('Positive Definite Kernel Matrix Heatmap (Bochner Theorem)')
    logger.add_finding('Kernel matrix is symmetric and positive semi-definite by Bochner theorem', 'theoretical')
    logger.add_finding('Diagonal elements K_ii = m (maximum value, total frequency count)', 'empirical')
    logger.add_finding('TCFMamba (a=3) produces smoother kernel structure than Transformer', 'comparison')
    logger.log_parameter('bochner_theorem', 'K(Δq) = Σ exp(i·ω_k·Δq) = ∫ exp(i·ω·Δq) dμ(ω)')
    logger.log_parameter('kernel_matrix_formula', 'K_ij = Re(K(q_i - q_j)) = Σ cos(ω_k · Δq)')
    logger.log_parameter('n_points', 50)
    logger.log_parameter('dim', 64)
    
    setup_plot_style(figsize=(15, 12))
    
    # Generate positions
    n_points = 50
    positions = np.linspace(0, 1000, n_points)  # 0 to 1000 meters
    
    # Compare different power exponents
    powers = [-1.0, 1.0, 3.0]  # Transformer, Linear, TCFMamba
    titles = ['Transformer (a=-1, geometric)', 
              'Linear (a=1)',
              'TCFMamba (a=3, superlinear)']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (power, title) in enumerate(zip(powers, titles)):
        ax = axes[idx]
        
        # Create frequency function and kernel
        freq_func = FrequencyFunction(dim=64, power=power)
        kernel = KernelFunction(freq_func)
        
        # Compute kernel matrix
        K_matrix = kernel.compute_matrix(positions)
        
        # Plot heatmap
        im = ax.imshow(K_matrix, cmap='RdBu_r', aspect='auto', 
                      vmin=-20, vmax=32)  # Symmetric around 0
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Position Index j', fontsize=10)
        ax.set_ylabel('Position Index i', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Re(K_{ij})', fontsize=10)
        
        # Add statistics text
        max_val = np.max(K_matrix)
        min_val = np.min(K_matrix)
        mean_val = np.mean(K_matrix)
        textstr = f'Max: {max_val:.1f}\nMin: {min_val:.1f}\nMean: {mean_val:.1f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
    
    # Fourth subplot: Kernel value vs distance for comparison
    ax4 = axes[3]
    delta_q = np.linspace(0, 1000, 500)
    
    for power, title in zip(powers, titles):
        freq_func = FrequencyFunction(dim=64, power=power)
        kernel = KernelFunction(freq_func)
        
        K_values = kernel.compute_real(delta_q)
        color = POWER_COLORS.get(power, COLORS['dark'])
        ax4.plot(delta_q, K_values, linewidth=2, label=title, color=color)
    
    ax4.set_xlabel('Position Difference Δq (meters)', fontsize=11)
    ax4.set_ylabel('Kernel Value Re(K(Δq))', fontsize=11)
    ax4.set_title('Kernel Decay Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='gray', linewidth=0.5)
    
    plt.suptitle('Positive Definite Kernel Matrix Visualization\n' +
                 'K_{ij} = Σ cos(ω_k · (q_i - q_j))  (Bochner Theorem)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save static figure
    save_figure(fig, '3_kernel_matrix_heatmap', output_dir, formats=['png', 'pdf'])
    logger.log_figure(output_dir / '3_kernel_matrix_heatmap.png',
                     title='Kernel Matrix Comparison',
                     fig_type='multi_heatmap')
    plt.close(fig)
    
    # Create individual detailed heatmaps for HTML
    for power, title in zip(powers, titles):
        freq_func = FrequencyFunction(dim=64, power=power)
        kernel = KernelFunction(freq_func)
        K_matrix = kernel.compute_matrix(positions)
        
        # Log matrix data
        logger.log_matrix(f'kernel_matrix_a{power}', K_matrix,
                         row_labels=[f'pos_{i}' for i in range(len(positions))],
                         col_labels=[f'pos_{j}' for j in range(len(positions))],
                         metadata={'power': power, 'title': title})
        
        # Log key statistics
        logger.log_metric(f'diagonal_mean_a{power}', float(np.mean(np.diag(K_matrix))),
                         context={'power': power, 'expected': 64})
        logger.log_metric(f'matrix_mean_a{power}', float(np.mean(K_matrix)),
                         context={'power': power})
        logger.log_metric(f'matrix_range_a{power}', float(np.max(K_matrix) - np.min(K_matrix)),
                         context={'power': power})
        
        html_filename = f'3_kernel_matrix_a{power}.html'
        create_heatmap_html(K_matrix, 
                           title=f'Kernel Matrix: {title}',
                           x_label='Position j',
                           y_label='Position i',
                           html_filename=html_filename)
        logger.log_figure(html_dir / html_filename,
                         title=f'Interactive Kernel Matrix (a={power})',
                         fig_type='interactive_heatmap')
    
    # Log kernel decay curves
    delta_q = np.linspace(0, 1000, 500)
    for power, title in zip(powers, titles):
        freq_func = FrequencyFunction(dim=64, power=power)
        kernel = KernelFunction(freq_func)
        K_values = kernel.compute_real(delta_q)
        logger.log_series(f'kernel_decay_a{power}', delta_q, K_values,
                         x_label='Position Difference Δq (meters)',
                         y_label='Re(K(Δq))',
                         metadata={'power': power, 'title': title})
    
    # Log comparison data
    logger.log_comparison('power_exponent_comparison', [
        {'label': 'Transformer (a=-1)', 'value': -1.0, 'properties': {'pattern': 'geometric', 'locality': 'strong'}},
        {'label': 'Linear (a=1)', 'value': 1.0, 'properties': {'pattern': 'linear', 'locality': 'moderate'}},
        {'label': 'TCFMamba (a=3)', 'value': 3.0, 'properties': {'pattern': 'superlinear', 'locality': 'smooth'}}
    ])
    
    # Create comparison HTML
    create_comparison_html(html_dir, powers, titles, positions)
    logger.log_figure(html_dir / '3_kernel_matrix_comparison.html',
                     title='Kernel Matrix Interactive Comparison',
                     fig_type='interactive_comparison')
    
    # Save logger data
    logger.save()
    print("Visualization 3 completed: Kernel Matrix Heatmap")


def create_comparison_html(html_dir: Path, powers: list, titles: list, positions: np.ndarray):
    """Create HTML page comparing all kernel matrices side by side."""
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Kernel Matrix Comparison</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: Georgia, serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .math-note {
            background: #ecf0f1;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .plot-container {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }
        .comparison-plot {
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Visualization 3: Positive Definite Kernel Matrix</h1>
        <div class="math-note">
            <b>Bochner's Theorem:</b> K is positive definite ⟺ K(Δq) = ∫ exp(i·ω·Δq) dμ(ω)<br>
            <b>Discrete Form:</b> K_{ij} = Σ cos(ω_k · Δq), where Δq = q_i - q_j<br>
            <b>Properties:</b> Symmetric, positive semi-definite, diagonal dominance
        </div>
        <div class="grid">
"""
    
    # Add each heatmap
    for power, title in zip(powers, titles):
        freq_func = FrequencyFunction(dim=64, power=power)
        kernel = KernelFunction(freq_func)
        K_matrix = kernel.compute_matrix(positions)
        
        # Create plotly heatmap
        fig_html = f"""
            <div class="plot-container">
                <h3>{title}</h3>
                <div id="heatmap_{power}"></div>
            </div>
            <script>
                var data_{power} = [{{
                    z: {K_matrix.tolist()},
                    type: 'heatmap',
                    colorscale: 'RdBu',
                    zmid: 0,
                    hovertemplate: 'i: %{{x}}<br>j: %{{y}}<br>K: %{{z:.2f}}<extra></extra>'
                }}];
                var layout_{power} = {{
                    title: '{title}',
                    width: 500,
                    height: 450,
                    xaxis: {{title: 'Position j'}},
                    yaxis: {{title: 'Position i'}}
                }};
                Plotly.newPlot('heatmap_{power}', data_{power}, layout_{power}, {{responsive: true}});
            </script>
"""
        html_content += fig_html
    
    # Add comparison plot at bottom
    html_content += """
        </div>
        <div class="comparison-plot">
            <h3>Kernel Decay Comparison (Cross-section)</h3>
            <div id="comparison_plot"></div>
        </div>
    </div>
    <script>
"""
    
    # Add comparison plot data
    delta_q = np.linspace(0, 1000, 200)
    traces = []
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    for power, title, color in zip(powers, titles, colors):
        freq_func = FrequencyFunction(dim=64, power=power)
        kernel = KernelFunction(freq_func)
        K_values = kernel.compute_real(delta_q)
        
        traces.append(f"""
        {{
            x: {delta_q.tolist()},
            y: {K_values.tolist()},
            mode: 'lines',
            name: '{title}',
            line: {{color: '{color}', width: 2}}
        }}""")
    
    html_content += f"""
        var comp_traces = [{','.join(traces)}];
        var comp_layout = {{
            title: 'Kernel Decay vs Position Difference',
            xaxis: {{title: 'Position Difference Δq (meters)'}},
            yaxis: {{title: 'Kernel Value Re(K(Δq))'}},
            hovermode: 'closest',
            width: 900,
            height: 400
        }};
        Plotly.newPlot('comparison_plot', comp_traces, comp_layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    html_path = html_dir / "3_kernel_matrix_comparison.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved comparison HTML: {html_path}")


if __name__ == '__main__':
    visualize_kernel_matrix()
