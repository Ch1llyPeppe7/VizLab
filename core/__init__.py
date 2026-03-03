
"""
VizLab Core — 通用核心库

提供数学工具、绘图工具、位置编码注册表、结构化日志等基础设施。

Usage:
    from core import get_pe, get_all_pe, PEConfig
    from core import setup_plot_style, save_figure
    from core import VizLogger
    from core.math_utils import compute_lyapunov_exponent, layer_norm
"""

# 位置编码注册表
from .pe_registry import (
    PEConfig,
    PositionEncoding,
    SinusoidalPE,
    RoPE,
    ALiBi,
    LAPE,
    get_pe,
    get_all_pe,
    list_pe,
)

# 数学工具
from .math_utils import (
    FrequencyFunction,
    KernelFunction,
    random_orthogonal_matrix,
    random_weight_matrix,
    compute_lyapunov_exponent,
    compute_phase_space,
    spectral_entropy,
    mutual_information_discrete,
    effective_rank,
    activation_fn,
    layer_norm,
    simulate_feedforward_layer,
)

# 绘图工具
from .plot_utils import (
    setup_plot_style,
    save_figure,
    add_math_annotation,
    create_plotly_figure,
    save_plotly_html,
    create_heatmap_html,
    generate_report_html,
    get_pe_color,
    get_output_dir,
    get_html_dir,
    COLORS,
    PE_COLORS,
    POWER_COLORS,
    plotly_available,
)

# 日志
from .viz_logger import (
    VizLogger,
    quick_log,
)

__version__ = "2.0.0"

__all__ = [
    # PE
    'PEConfig', 'PositionEncoding',
    'SinusoidalPE', 'RoPE', 'ALiBi', 'LAPE',
    'get_pe', 'get_all_pe', 'list_pe',
    # Math
    'FrequencyFunction', 'KernelFunction',
    'random_orthogonal_matrix', 'random_weight_matrix',
    'compute_lyapunov_exponent', 'compute_phase_space',
    'spectral_entropy', 'mutual_information_discrete',
    'effective_rank', 'activation_fn', 'layer_norm',
    'simulate_feedforward_layer',
    # Plot
    'setup_plot_style', 'save_figure', 'add_math_annotation',
    'create_plotly_figure', 'save_plotly_html',
    'create_heatmap_html', 'generate_report_html',
    'get_pe_color', 'get_output_dir', 'get_html_dir',
    'COLORS', 'PE_COLORS', 'POWER_COLORS', 'plotly_available',
    # Logger
    'VizLogger', 'quick_log',
]
