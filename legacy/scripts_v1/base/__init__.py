"""
Base utilities for LAPE mathematical visualizations.

This module provides pure mathematical implementations of LAPE (Location-Aware Position Encoding)
formulas without requiring model training or weight loading.
"""

from .lape_math import (
    FrequencyFunction,
    KernelFunction,
    LAPEEncoder,
    SphericalTransform
)
from .plot_utils import (
    setup_plot_style,
    save_figure,
    create_interactive_plot,
    generate_html_visualization,
    COLORS,
    POWER_COLORS
)
from .viz_logger import (
    VizLogger,
    VizReportGenerator,
    quick_log
)

__all__ = [
    'FrequencyFunction',
    'KernelFunction', 
    'LAPEEncoder',
    'SphericalTransform',
    'setup_plot_style',
    'save_figure',
    'create_interactive_plot',
    'generate_html_visualization',
    'COLORS',
    'POWER_COLORS',
    'VizLogger',
    'VizReportGenerator',
    'quick_log'
]
