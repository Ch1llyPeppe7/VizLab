"""
度量张量与弧长计算

提供黎曼度量相关的计算函数。
"""

import numpy as np
from typing import Optional
from scipy.integrate import cumulative_trapezoid


def metric_tensor(
    d1: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    计算参数曲线的度量张量（1D 参数退化为标量度量）。
    
    对于参数曲线 γ: ℝ → ℝ^d，度量张量为:
        g(t) = ||γ'(t)||² = Σ_k (dγ_k/dt)²
    
    这等价于 Fisher-Rao 度量的特例。
    
    Args:
        d1: [N, d] 一阶导数 γ'(t)
        eps: 数值稳定项
    
    Returns:
        [N,] 度量张量值
    
    物理意义:
        - g(t) 大 → 曲线在 t 处"拉伸"严重 → 位置分辨率高
        - g(t) 小 → 曲线压缩 → 相邻位置难以区分
    
    Example:
        >>> from core.analysis.geometry import compute_derivatives, metric_tensor
        >>> data = np.random.randn(100, 64)
        >>> d1, _, _ = compute_derivatives(data)
        >>> g = metric_tensor(d1)
        >>> print(g.shape)  # (100,)
    """
    d1 = np.atleast_2d(d1)
    return np.sum(d1 ** 2, axis=1) + eps


def arc_length(
    g: np.ndarray,
    t: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算累积弧长。
    
    弧长定义为:
        s(t) = ∫₀ᵗ √g(τ) dτ
    
    Args:
        g: [N,] 度量张量
        t: [N,] 参数值（可选，默认为 [0, 1, ..., N-1]）
    
    Returns:
        [N,] 累积弧长，s[0] = 0
    
    Example:
        >>> g = np.ones(100)  # 均匀度量
        >>> s = arc_length(g)
        >>> print(s[-1])  # ≈ 99 (线性增长)
    """
    g = np.atleast_1d(g)
    N = len(g)
    
    if t is None:
        t = np.arange(N, dtype=float)
    
    speed = np.sqrt(g)  # ds/dt = √g
    
    # 数值积分（梯形法）
    s = np.zeros(N)
    if N > 1:
        s[1:] = cumulative_trapezoid(speed, t)
    
    return s


def geodesic_distance(
    g: np.ndarray,
    t: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算测地距离矩阵。
    
    测地距离 d(i, j) = |s(i) - s(j)|，其中 s 为弧长。
    
    Args:
        g: [N,] 度量张量
        t: [N,] 参数值（可选）
    
    Returns:
        [N, N] 测地距离矩阵
    
    Example:
        >>> g = np.ones(10)
        >>> D = geodesic_distance(g)
        >>> print(D[0, 5])  # ≈ 5.0
    """
    s = arc_length(g, t)
    # 广播计算距离矩阵
    return np.abs(s[:, None] - s[None, :])


def speed(g: np.ndarray) -> np.ndarray:
    """
    计算弧长速率 ds/dt = √g(t)。
    
    Args:
        g: [N,] 度量张量
    
    Returns:
        [N,] 弧长速率
    """
    return np.sqrt(np.maximum(g, 0))


def is_unit_speed(
    g: np.ndarray,
    tol: float = 0.1
) -> bool:
    """
    检查曲线是否为单位速度参数化。
    
    单位速度参数化意味着 g(t) ≈ 1，即 ||γ'(t)|| ≈ 1。
    
    Args:
        g: [N,] 度量张量
        tol: 容差
    
    Returns:
        是否满足单位速度条件
    """
    return np.allclose(g, 1.0, atol=tol)
