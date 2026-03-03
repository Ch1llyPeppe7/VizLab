"""
Visualization 1: LAPE Complex Plane Embedding (Kähler Structure Foundation)

This script visualizes the complex plane embedding z(q) = exp(i·ω·q) 
which forms the foundation of the Kähler structure in LAPE encoding.

Mathematical Background:
-------------------------
The complex embedding maps a position q to a point on the unit circle:
    z(q) = e^(i·ω·q) = cos(ω·q) + i·sin(ω·q)

Properties:
1. |z(q)| = 1 (always on unit circle)
2. As q increases, z rotates at angular velocity ω
3. Higher frequency ω → faster rotation → higher spatial resolution

Kähler Structure Connection:
-----------------------------
The Kähler manifold is constructed from m copies of the complex plane C.
Each frequency ω_k corresponds to one complex dimension, and the
embedding Φ(q) = (z_1(q), z_2(q), ..., z_m(q)) maps positions to
the product manifold ⊕^m C.

The inner product ⟨Φ(q_1), Φ(q_2)⟩ = Σ exp(i·ω_k·(q_1-q_2)) = K(Δq)
is exactly the kernel function defined in math.md.

Reference: math.md Section 4 (Kähler structure theorem)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add base module to path
sys.path.insert(0, str(Path(__file__).parent))
from base import LAPEEncoder, setup_plot_style, save_figure
from base.plot_utils import generate_html_visualization, COLORS
from base.viz_logger import VizLogger

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def visualize_complex_embedding(output_dir: Path = None, html_dir: Path = None):
    """
    Generate complex plane embedding visualization.
    
    Shows how positions map to points on the unit circle in complex plane,
    and how different frequencies produce different rotation speeds.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    if html_dir is None:
        html_dir = Path(__file__).parent.parent / "html"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = VizLogger('1_lape_complex_plane')
    logger.set_description('LAPE Complex Plane Embedding (Kähler Structure Foundation)')
    logger.add_finding('Complex embedding z(q) = e^(i·ω·q) maps positions to unit circle', 'theoretical')
    logger.add_finding('Higher frequency ω → faster rotation → higher spatial resolution', 'empirical')
    logger.log_parameter('formula', 'z(q) = cos(ω·q) + i·sin(ω·q)')
    logger.log_parameter('kahler_structure', 'Φ(q) = (z_1(q), ..., z_m(q))')
    
    setup_plot_style(figsize=(14, 10))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Generate position values
    q_values = np.linspace(0, 1000, 500)  # Spatial coordinates (meters)
    
    # Different frequencies to visualize
    frequencies = [0.01, 0.05, 0.1, 0.3]  # Low to high frequency
    colors = [COLORS['primary'], COLORS['success'], COLORS['warning'], COLORS['secondary']]
    
    # Plot 1: Complex plane trajectories for different frequencies
    ax1 = fig.add_subplot(2, 2, 1)
    for freq, color in zip(frequencies, colors):
        # z(q) = exp(i·ω·q) = cos(ω·q) + i·sin(ω·q)
        z_real = np.cos(freq * q_values)
        z_imag = np.sin(freq * q_values)
        
        # Plot trajectory with color gradient
        ax1.plot(z_real, z_imag, color=color, linewidth=1.5, 
                label=f'ω = {freq}', alpha=0.8)
        
        # Mark starting point
        ax1.scatter([1], [0], color=color, s=100, zorder=5, marker='o')
    
    ax1.set_xlabel('Real: cos(ω·q)', fontsize=11)
    ax1.set_ylabel('Imag: sin(ω·q)', fontsize=11)
    ax1.set_title('Complex Plane Trajectories\n$z(q) = e^{i\\omega q}$', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Add unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax1.add_patch(circle)
    
    # Plot 2: Real part vs position (cosine wave)
    ax2 = fig.add_subplot(2, 2, 2)
    for freq, color in zip(frequencies, colors):
        z_real = np.cos(freq * q_values)
        ax2.plot(q_values, z_real, color=color, linewidth=2, label=f'ω = {freq}')
    
    ax2.set_xlabel('Position $q$ (meters)', fontsize=11)
    ax2.set_ylabel('Real: $cos(\\omega \\cdot q)$', fontsize=11)
    ax2.set_title('Real Part vs Position\n(Cosine Encoding)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    
    # Plot 3: Imaginary part vs position (sine wave)
    ax3 = fig.add_subplot(2, 2, 3)
    for freq, color in zip(frequencies, colors):
        z_imag = np.sin(freq * q_values)
        ax3.plot(q_values, z_imag, color=color, linewidth=2, label=f'ω = {freq}')
    
    ax3.set_xlabel('Position $q$ (meters)', fontsize=11)
    ax3.set_ylabel('Imag: $sin(\\omega \\cdot q)$', fontsize=11)
    ax3.set_title('Imaginary Part vs Position\n(Sine Encoding)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linewidth=0.5)
    
    # Plot 4: Rotation speed (phase angle) vs position
    ax4 = fig.add_subplot(2, 2, 4)
    for freq, color in zip(frequencies, colors):
        phase = np.mod(freq * q_values, 2 * np.pi)  # Phase angle [0, 2π]
        ax4.plot(q_values, phase, color=color, linewidth=2, label=f'ω = {freq}')
    
    ax4.set_xlabel('Position $q$ (meters)', fontsize=11)
    ax4.set_ylabel('Phase Angle $\phi = \\omega \\cdot q$ (rad)', fontsize=11)
    ax4.set_title('Phase Rotation Speed\n$\\phi = \\omega q$', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 2*np.pi])
    
    plt.suptitle('LAPE Complex Plane Embedding (Kähler Structure Foundation)\n' + 
                 '$z(q) = e^{i\\omega q} = \\cos(\\omega q) + i\\sin(\\omega q)$',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save static figure
    save_figure(fig, '1_lape_complex_plane', output_dir)
    logger.log_figure(output_dir / '1_lape_complex_plane.png', 
                     title='LAPE Complex Plane Embedding',
                     fig_type='multi_subplot')
    plt.close(fig)
    
    # Create interactive HTML version
    if PLOTLY_AVAILABLE:
        create_interactive_complex_embedding(html_dir)
        logger.log_figure(html_dir / '1_lape_complex_plane.html',
                         title='Interactive LAPE Complex Plane',
                         fig_type='interactive_html')
    
    # Log data series
    for freq in frequencies:
        z_real = np.cos(freq * q_values)
        z_imag = np.sin(freq * q_values)
        phase = np.mod(freq * q_values, 2 * np.pi)
        logger.log_series(f'frequency_{freq}_real', q_values, z_real,
                         x_label='Position q (meters)', y_label='cos(ω·q)',
                         metadata={'frequency': freq, 'component': 'real'})
        logger.log_series(f'frequency_{freq}_imag', q_values, z_imag,
                         x_label='Position q (meters)', y_label='sin(ω·q)',
                         metadata={'frequency': freq, 'component': 'imaginary'})
        logger.log_series(f'frequency_{freq}_phase', q_values, phase,
                         x_label='Position q (meters)', y_label='phase (rad)',
                         metadata={'frequency': freq, 'component': 'phase'})
    
    # Log comparison data
    logger.log_comparison('rotation_speeds', [
        {'label': 'ω=0.01', 'value': 0.01, 'properties': {'period': 628, 'resolution': 'low'}},
        {'label': 'ω=0.05', 'value': 0.05, 'properties': {'period': 125, 'resolution': 'medium'}},
        {'label': 'ω=0.1', 'value': 0.1, 'properties': {'period': 62, 'resolution': 'high'}},
        {'label': 'ω=0.3', 'value': 0.3, 'properties': {'period': 21, 'resolution': 'very_high'}}
    ])
    
    # Save logger data
    logger.save()
    print("Visualization 1 completed: LAPE Complex Plane Embedding")


def create_interactive_complex_embedding(html_dir: Path):
    """Create interactive HTML version with Plotly."""
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Complex Plane Trajectories z(q) = e^(iωq)',
            'Real Part: cos(ω·q)',
            'Imaginary Part: sin(ω·q)',
            'Phase Rotation: φ = ω·q'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    q_values = np.linspace(0, 1000, 500)
    frequencies = [0.01, 0.05, 0.1, 0.3]
    colors_plotly = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    for i, (freq, color) in enumerate(zip(frequencies, colors_plotly)):
        z_real = np.cos(freq * q_values)
        z_imag = np.sin(freq * q_values)
        phase = np.mod(freq * q_values, 2 * np.pi)
        
        # Complex plane (unit circle trajectory)
        fig.add_trace(
            go.Scatter(x=z_real, y=z_imag, mode='lines',
                      name=f'ω = {freq}',
                      line=dict(color=color, width=2),
                      hovertemplate='Re: %{x:.4f}<br>Im: %{y:.4f}<extra></extra>',
                      showlegend=True),
            row=1, col=1
        )
        
        # Real part
        fig.add_trace(
            go.Scatter(x=q_values, y=z_real, mode='lines',
                      name=f'ω = {freq}',
                      line=dict(color=color, width=2),
                      showlegend=False),
            row=1, col=2
        )
        
        # Imaginary part
        fig.add_trace(
            go.Scatter(x=q_values, y=z_imag, mode='lines',
                      name=f'ω = {freq}',
                      line=dict(color=color, width=2),
                      showlegend=False),
            row=2, col=1
        )
        
        # Phase
        fig.add_trace(
            go.Scatter(x=q_values, y=phase, mode='lines',
                      name=f'ω = {freq}',
                      line=dict(color=color, width=2),
                      showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(
        title=dict(
            text='<b>LAPE Complex Plane Embedding</b><br><sup>Kähler Structure Foundation: z(q) = e^(iωq)</sup>',
            font=dict(size=18)
        ),
        height=800,
        hovermode='closest',
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Real: cos(ω·q)', row=1, col=1)
    fig.update_yaxes(title_text='Imag: sin(ω·q)', row=1, col=1)
    fig.update_xaxes(title_text='Position q (meters)', row=1, col=2)
    fig.update_yaxes(title_text='cos(ω·q)', row=1, col=2)
    fig.update_xaxes(title_text='Position q (meters)', row=2, col=1)
    fig.update_yaxes(title_text='sin(ω·q)', row=2, col=1)
    fig.update_xaxes(title_text='Position q (meters)', row=2, col=2)
    fig.update_yaxes(title_text='Phase φ (rad)', range=[0, 2*np.pi], row=2, col=2)
    
    # Save HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LAPE Complex Plane Embedding</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: Georgia, serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .math-note {{ background: #ecf0f1; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }}
        #plot {{ width: 100%; height: 800px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Visualization 1: LAPE Complex Plane Embedding</h1>
        <div class="math-note">
            <b>Mathematical Formula:</b> z(q) = e^(i·ω·q) = cos(ω·q) + i·sin(ω·q)<br>
            <b>Kähler Structure:</b> Each frequency ω maps positions to unit circle in complex plane.<br>
            <b>Key Insight:</b> Higher ω → faster rotation → higher spatial resolution.
        </div>
        <div id="plot"></div>
    </div>
    <script>
        var figure = {fig.to_json()};
        Plotly.newPlot('plot', figure.data, figure.layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    html_path = html_dir / "1_lape_complex_plane.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved interactive HTML: {html_path}")


if __name__ == '__main__':
    visualize_complex_embedding()
