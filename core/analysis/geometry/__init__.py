"""
core.analysis.geometry — 微分几何分析工具

提供与领域无关的微分几何计算函数，适用于任何高维参数曲线的分析。

核心函数:
    - compute_derivatives: 数值微分（中心差分）
    - metric_tensor: 度量张量 g(t) = ||γ'(t)||²
    - arc_length: 累积弧长 s(t) = ∫√g dt
    - curvature: Frenet-Serret 曲率 κ(t)
    - torsion: 挠率 τ(t)
    - christoffel_connection: Christoffel-like 联络结构

数学背景:
    将离散数据 X ∈ ℝ^{N×d} 视为高维空间中的参数曲线 γ: [0, N-1] → ℝ^d，
    运用微分几何工具分析其几何结构。

Example:
    >>> import numpy as np
    >>> from core.analysis.geometry import curvature, compute_derivatives
    >>> 
    >>> # 生成一条 3D 螺旋线
    >>> t = np.linspace(0, 4*np.pi, 100)
    >>> data = np.column_stack([np.cos(t), np.sin(t), t/10])
    >>> 
    >>> # 计算曲率
    >>> d1, d2, _ = compute_derivatives(data)
    >>> kappa = curvature(d1, d2)
"""

from .derivatives import compute_derivatives, compute_derivatives_1d
from .metric import metric_tensor, arc_length, geodesic_distance
from .curvature import curvature, torsion, curvature_2d
from .connections import christoffel_connection, curvature_spectrum

__all__ = [
    # 微分
    'compute_derivatives',
    'compute_derivatives_1d',
    # 度量
    'metric_tensor',
    'arc_length',
    'geodesic_distance',
    # 曲率
    'curvature',
    'torsion',
    'curvature_2d',
    # 联络
    'christoffel_connection',
    'curvature_spectrum',
]
