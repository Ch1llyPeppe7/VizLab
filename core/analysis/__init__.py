"""
core.analysis — 通用数学分析库

此模块提供与领域无关的数学分析工具，可用于 PE 分析、Transformer 内部分析、
或任何其他需要类似数学工具的场景。

子模块:
    - geometry: 微分几何工具（曲率、挠率、度量张量等）
    - spectral: 频谱分析工具（FFT、PSD、谱熵等）
    - information: 信息论工具（熵、互信息、Fisher 信息等）
    - manifold: 流形可视化工具（PCA、t-SNE、UMAP 等）

Usage:
    from core.analysis import geometry, spectral, information, manifold
    
    # 或直接导入函数
    from core.analysis.geometry import curvature, torsion
    from core.analysis.spectral import fft_power_spectrum
"""

from . import geometry
from . import spectral
from . import information
from . import manifold

# 所有子模块都已实现
__all__ = [
    'geometry',
    'spectral',
    'information',
    'manifold',
]
