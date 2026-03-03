
"""
Visualization Utilities — 统一绘图工具

支持 Matplotlib（静态）、Plotly（交互式 HTML）、以及通用 HTML 模板生成。
升级自原 scripts/base/plot_utils.py，增强了主题管理和交互式输出。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any
import warnings
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. Interactive HTML visualizations will not be available.")

# ============================================================
#  路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
HTML_DIR = PROJECT_ROOT / "html"


def get_output_dir(module: str = None) -> Path:
    """获取输出目录，按模块组织"""
    d = OUTPUT_DIR / module if module else OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_html_dir(module: str = None) -> Path:
    """获取 HTML 输出目录"""
    d = HTML_DIR / module if module else HTML_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
#  配色方案
# ============================================================

# 通用调色板
COLORS = {
    'primary':   '#3498db',
    'secondary': '#e74c3c',
    'success':   '#2ecc71',
    'warning':   '#f39c12',
    'info':      '#9b59b6',
    'dark':      '#2c3e50',
    'gray':      '#95a5a6',
    'light':     '#ecf0f1',
}

# PE 方案专用配色
PE_COLORS = {
    'sinusoidal': '#3498db',   # 蓝
    'rope':       '#e74c3c',   # 红
    'alibi':      '#2ecc71',   # 绿
    'lape':       '#f39c12',   # 橙
}

# 向后兼容
POWER_COLORS = {
    -1.0: '#e74c3c',
    0.5:  '#f39c12',
    1.0:  '#2ecc71',
    3.0:  '#3498db',
}


def get_pe_color(pe_name: str) -> str:
    """获取 PE 方案的标准颜色"""
    return PE_COLORS.get(pe_name.lower(), COLORS['dark'])


def get_power_color(power: float) -> str:
    """获取幂指数的标准颜色（向后兼容）"""
    return POWER_COLORS.get(power, COLORS['dark'])


# ============================================================
#  Matplotlib 样式
# ============================================================

def setup_plot_style(style: str = "seaborn-v0_8-whitegrid",
                     figsize: Tuple[int, int] = (10, 6),
                     dpi: int = 150,
                     use_latex: bool = False):
    """
    配置全局 Matplotlib 样式。
    
    Args:
        style: Matplotlib 样式名
        figsize: 默认图形尺寸
        dpi: 默认 DPI
        use_latex: 是否启用 LaTeX 渲染
    """
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use('seaborn-v0_8')
    
    plt.rcParams.update({
        'figure.figsize': figsize,
        'figure.dpi': dpi,
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'legend.fontsize': 10,
        'figure.titlesize': 17,
        'mathtext.fontset': 'stix',
        'font.family': 'serif',
    })
    
    if use_latex:
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
        })


def save_figure(fig: plt.Figure, 
                filename: str, 
                module: str = None,
                formats: List[str] = None,
                dpi: int = 300) -> List[Path]:
    """
    保存 Matplotlib 图形。
    
    Args:
        fig: 图形对象
        filename: 基本文件名（不含扩展名）
        module: 模块名（用于组织输出目录）
        formats: 保存格式列表
        dpi: 分辨率
    Returns:
        保存的文件路径列表
    """
    if formats is None:
        formats = ['png', 'pdf']
    
    output_dir = get_output_dir(module)
    saved = []
    
    for fmt in formats:
        fp = output_dir / f"{filename}.{fmt}"
        fig.savefig(fp, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        saved.append(fp)
        print(f"  ✓ Saved: {fp}")
    
    return saved


def add_math_annotation(ax: plt.Axes, text: str, 
                        loc: str = 'lower right', fontsize: int = 10):
    """在图上添加数学公式注释"""
    locs = {
        'lower right': (0.95, 0.05, 'bottom', 'right'),
        'upper right': (0.95, 0.95, 'top', 'right'),
        'lower left':  (0.05, 0.05, 'bottom', 'left'),
        'upper left':  (0.05, 0.95, 'top', 'left'),
    }
    x, y, va, ha = locs.get(loc, (0.95, 0.05, 'bottom', 'right'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment=va, horizontalalignment=ha, bbox=props)


# ============================================================
#  Plotly 交互式可视化
# ============================================================

def plotly_available() -> bool:
    """检查 Plotly 是否可用"""
    return PLOTLY_AVAILABLE


def create_plotly_figure(title: str = "",
                         x_label: str = "X",
                         y_label: str = "Y",
                         template: str = 'plotly_white',
                         width: int = 900,
                         height: int = 600) -> 'go.Figure':
    """
    创建预配置的 Plotly Figure。
    
    Returns:
        go.Figure 实例
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive plots")
    
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template=template,
        width=width,
        height=height,
        hovermode='closest',
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99
        )
    )
    return fig


def save_plotly_html(fig: 'go.Figure', filename: str,
                     module: str = None,
                     include_mathjax: bool = True) -> Path:
    """
    保存 Plotly 图形为独立 HTML 文件。
    
    Args:
        fig: Plotly Figure
        filename: 文件名（含 .html 扩展名）
        module: 模块名
        include_mathjax: 是否包含 MathJax（用于公式渲染）
    Returns:
        保存的文件路径
    """
    html_dir = get_html_dir(module)
    fp = html_dir / filename
    
    fig.write_html(
        str(fp),
        include_plotlyjs='cdn',
        include_mathjax='cdn' if include_mathjax else False,
        full_html=True
    )
    print(f"  ✓ Saved HTML: {fp}")
    return fp


def create_heatmap_html(matrix: np.ndarray,
                        title: str = "Heatmap",
                        x_label: str = "X", y_label: str = "Y",
                        colorscale: str = "RdBu",
                        module: str = None,
                        filename: str = "heatmap.html") -> Path:
    """创建交互式热力图 HTML"""
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available, falling back to static plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        return save_figure(fig, filename.replace('.html', ''), module, ['png'])[0]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix, colorscale=colorscale, zmid=0,
        hovertemplate='i: %{x}<br>j: %{y}<br>Value: %{z:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=x_label, yaxis_title=y_label,
        width=800, height=700
    )
    return save_plotly_html(fig, filename, module)


# ============================================================
#  HTML 模板生成
# ============================================================

def generate_report_html(title: str, 
                         sections: List[Dict[str, str]],
                         module: str = None,
                         filename: str = "report.html") -> Path:
    """
    生成数学可视化报告 HTML。
    
    Args:
        title: 页面标题
        sections: 段落列表，每个元素 {'title': ..., 'content': ..., 'plot_id': ...}
        module: 模块名
        filename: 输出文件名
    Returns:
        HTML 文件路径
    """
    sections_html = ""
    for sec in sections:
        sections_html += f"""
        <section>
            <h2>{sec.get('title', '')}</h2>
            <div class="math-note">{sec.get('content', '')}</div>
            {'<div id="' + sec['plot_id'] + '" class="plot-container"></div>' if 'plot_id' in sec else ''}
        </section>
        """
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{title} — VizLab</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: 'Georgia', 'Noto Serif SC', serif;
            margin: 0; padding: 20px;
            background: #f7f7f7; color: #333;
        }}
        .container {{
            max-width: 1100px; margin: 0 auto;
            background: white; padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 12px;
        }}
        h2 {{ color: #34495e; }}
        .math-note {{
            background: #f0f4f8;
            padding: 16px 20px;
            border-left: 4px solid #3498db;
            margin: 16px 0;
            line-height: 1.8;
        }}
        .plot-container {{
            width: 100%;
            min-height: 500px;
            margin: 20px 0;
        }}
        section {{
            margin-bottom: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        {sections_html}
    </div>
</body>
</html>"""
    
    html_dir = get_html_dir(module)
    fp = html_dir / filename
    fp.write_text(html, encoding='utf-8')
    print(f"  ✓ Saved report: {fp}")
    return fp
