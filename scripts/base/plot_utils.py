"""
Visualization utilities for generating interactive HTML and static plots.

This module provides helper functions for:
1. Setting up consistent plot styles
2. Saving figures in multiple formats
3. Generating interactive HTML visualizations with Plotly
4. Creating animations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. Interactive HTML visualizations will not be available.")


# Default output paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
DEFAULT_HTML_DIR = Path(__file__).parent.parent.parent / "html"


def setup_plot_style(style: str = "seaborn-v0_8-whitegrid", 
                     figsize: Tuple[int, int] = (10, 6),
                     dpi: int = 150):
    """
    Setup consistent matplotlib style for all visualizations.
    
    Args:
        style: Matplotlib style name
        figsize: Default figure size (width, height) in inches
        dpi: Default DPI for saving figures
    """
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Use LaTeX-style math rendering
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'serif'


def save_figure(fig: plt.Figure, 
                filename: str, 
                output_dir: Optional[Union[str, Path]] = None,
                formats: List[str] = None,
                dpi: int = 300,
                bbox_inches: str = 'tight') -> List[Path]:
    """
    Save matplotlib figure in multiple formats.
    
    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        output_dir: Output directory (default: visualization/output/)
        formats: List of formats to save (default: ['png', 'pdf'])
        dpi: Resolution for raster formats
        bbox_inches: Bounding box parameter
        
    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ['png', 'pdf']
    
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                   facecolor='white', edgecolor='none')
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_paths


def create_interactive_plot(data_dict: Dict[str, Any],
                           title: str = "Interactive Visualization",
                           x_label: str = "X",
                           y_label: str = "Y",
                           html_filename: Optional[str] = None) -> Optional[str]:
    """
    Create interactive HTML plot using Plotly.
    
    Args:
        data_dict: Dictionary with keys 'x', 'y', 'name', 'mode', etc.
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        html_filename: Output HTML filename (default: auto-generated)
        
    Returns:
        Path to saved HTML file or None if Plotly unavailable
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available. Cannot create interactive plot.")
        return None
    
    fig = go.Figure()
    
    # Add traces
    if isinstance(data_dict, dict) and 'traces' in data_dict:
        for trace in data_dict['traces']:
            fig.add_trace(go.Scatter(
                x=trace.get('x', []),
                y=trace.get('y', []),
                mode=trace.get('mode', 'lines'),
                name=trace.get('name', ''),
                line=trace.get('line', dict(width=2)),
                hovertemplate=trace.get('hovertemplate', '%{y:.4f}<extra></extra>')
            ))
    else:
        # Single trace
        fig.add_trace(go.Scatter(
            x=data_dict.get('x', []),
            y=data_dict.get('y', []),
            mode=data_dict.get('mode', 'lines'),
            name=data_dict.get('name', ''),
            line=dict(width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save HTML
    if html_filename is None:
        import hashlib
        hash_str = hashlib.md5(title.encode()).hexdigest()[:8]
        html_filename = f"interactive_{hash_str}.html"
    
    html_path = DEFAULT_HTML_DIR / html_filename
    html_path.parent.mkdir(parents=True, exist_ok=True)
    
    pyo.plot(fig, filename=str(html_path), auto_open=False)
    print(f"Saved interactive plot: {html_path}")
    
    return str(html_path)


def generate_html_visualization(fig_dict: Dict[str, Any],
                                html_path: Union[str, Path],
                                title: str = "Visualization") -> str:
    """
    Generate standalone HTML file with embedded Plotly visualization.
    
    This creates a completely self-contained HTML file that can be
    opened directly in a browser without a server.
    
    Args:
        fig_dict: Plotly figure dictionary
        html_path: Output HTML file path
        title: Page title
        
    Returns:
        Path to saved HTML file
    """
    if not PLOTLY_AVAILABLE:
        # Fallback: create simple HTML with matplotlib image
        return _generate_simple_html(fig_dict, html_path, title)
    
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate full HTML with embedded plotly
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: 'Georgia', serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .math-note {{
            background: #ecf0f1;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
            font-size: 14px;
        }}
        #plot {{
            width: 100%;
            height: 600px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div id="plot"></div>
    </div>
    <script>
        var figure = {fig_dict};
        Plotly.newPlot('plot', figure.data, figure.layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved HTML visualization: {html_path}")
    return str(html_path)


def _generate_simple_html(fig_dict: Dict[str, Any], 
                         html_path: Union[str, Path],
                         title: str) -> str:
    """Fallback HTML generation without Plotly."""
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Georgia', serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
        }}
        .note {{
            background: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="note">
            <p>Static visualization saved. Install Plotly for interactive features:</p>
            <code>pip install plotly</code>
        </div>
    </div>
</body>
</html>
"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(html_path)


def create_heatmap_html(matrix: np.ndarray,
                       title: str = "Kernel Matrix",
                       x_label: str = "Position i",
                       y_label: str = "Position j",
                       colorscale: str = "RdBu",
                       html_filename: str = "heatmap.html") -> str:
    """
    Create interactive heatmap visualization.
    
    Args:
        matrix: 2D array to visualize
        title: Plot title
        x_label, y_label: Axis labels
        colorscale: Plotly colorscale name
        html_filename: Output filename
        
    Returns:
        Path to saved HTML file
    """
    if not PLOTLY_AVAILABLE:
        # Create static matplotlib version
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.colorbar(im, ax=ax, label='Kernel Value')
        
        output_path = DEFAULT_OUTPUT_DIR / f"{html_filename.replace('.html', '.png')}"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(output_path)
    
    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale=colorscale,
        zmid=0,
        hovertemplate='i: %{x}<br>j: %{y}<br>Kernel: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=800,
        height=700
    )
    
    html_path = DEFAULT_HTML_DIR / html_filename
    html_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    print(f"Saved heatmap: {html_path}")
    
    return str(html_path)


# Color palettes for consistent styling
COLORS = {
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'dark': '#2c3e50',
    'gray': '#95a5a6',
    'light': '#ecf0f1'
}

# Power exponent color mapping
POWER_COLORS = {
    -1.0: '#e74c3c',   # Transformer - red
    0.5: '#f39c12',    # Sublinear - orange
    1.0: '#2ecc71',    # Linear - green
    3.0: '#3498db',    # TCFMamba - blue
}


def get_power_color(power: float) -> str:
    """Get color for a specific power exponent."""
    return POWER_COLORS.get(power, COLORS['dark'])


def add_math_annotation(ax, text: str, loc: str = 'lower right', fontsize: int = 10):
    """Add mathematical annotation to plot."""
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.05, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
