"""
曲率与挠率计算

提供 Frenet-Serret 框架下的曲率和挠率计算函数。
"""

import numpy as np
from typing import Tuple


def curvature(
    d1: np.ndarray,
    d2: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    计算高维参数曲线的 Frenet-Serret 曲率。
    
    曲率公式（适用于任意维度）:
        κ(t) = √(||γ'||² ||γ''||² - (γ'·γ'')²) / ||γ'||³
    
    Args:
        d1: [N, d] 一阶导数 γ'(t)
        d2: [N, d] 二阶导数 γ''(t)
        eps: 数值稳定项
    
    Returns:
        [N,] 曲率
    
    几何意义:
        - κ 大 → 曲线急转弯 → 局部信息变化剧烈
        - κ 小 → 曲线平直 → 信息缓慢变化
        - 对于圆，κ = 1/半径
    
    Example:
        >>> # 单位圆 (x, y) = (cos(t), sin(t))
        >>> t = np.linspace(0, 2*np.pi, 100)
        >>> data = np.column_stack([np.cos(t), np.sin(t)])
        >>> d1, d2, _ = compute_derivatives(data)
        >>> kappa = curvature(d1, d2)
        >>> print(np.mean(kappa))  # ≈ 1.0
    """
    d1 = np.atleast_2d(d1)
    d2 = np.atleast_2d(d2)
    
    norm_d1_sq = np.sum(d1 ** 2, axis=1)  # ||γ'||²
    norm_d2_sq = np.sum(d2 ** 2, axis=1)  # ||γ''||²
    dot_d1_d2 = np.sum(d1 * d2, axis=1)   # γ'·γ''
    
    # 分子: √(||γ'||² ||γ''||² - (γ'·γ'')²)
    # 这是 ||γ' × γ''|| 的高维推广
    numerator = np.sqrt(np.maximum(norm_d1_sq * norm_d2_sq - dot_d1_d2**2, 0))
    
    # 分母: ||γ'||³
    denominator = np.power(norm_d1_sq + eps, 1.5)
    
    return numerator / (denominator + eps)


def torsion(
    d1: np.ndarray,
    d2: np.ndarray,
    d3: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    计算高维参数曲线的挠率。
    
    挠率衡量曲线离开密切平面的程度。
    
    3D 公式:
        τ(t) = (γ' × γ'') · γ''' / ||γ' × γ''||²
    
    高维推广使用 Gram 行列式近似。
    
    Args:
        d1: [N, d] 一阶导数 γ'(t)
        d2: [N, d] 二阶导数 γ''(t)
        d3: [N, d] 三阶导数 γ'''(t)
        eps: 数值稳定项
    
    Returns:
        [N,] 挠率
    
    几何意义:
        - τ = 0 → 曲线是平面曲线
        - τ ≠ 0 → 曲线在 3D+ 空间中螺旋
    """
    d1 = np.atleast_2d(d1)
    d2 = np.atleast_2d(d2)
    d3 = np.atleast_2d(d3)
    
    N = d1.shape[0]
    tau = np.zeros(N)
    
    for i in range(N):
        # 构建 3×d 矩阵 [γ'; γ''; γ''']
        M = np.vstack([d1[i], d2[i], d3[i]])  # [3, d]
        
        # Gram 矩阵 G = M @ M.T, shape [3, 3]
        G = M @ M.T
        
        # det(G) 的符号表示螺旋方向
        det_G = np.linalg.det(G)
        
        # 分母 = ||γ' × γ''||²
        norm_d1_sq = np.sum(d1[i] ** 2)
        norm_d2_sq = np.sum(d2[i] ** 2)
        dot_d1_d2 = np.sum(d1[i] * d2[i])
        cross_sq = norm_d1_sq * norm_d2_sq - dot_d1_d2**2
        
        if cross_sq > eps:
            # 近似挠率（使用 Gram 行列式的根号）
            tau[i] = np.sign(det_G) * np.sqrt(np.abs(det_G)) / (cross_sq + eps)
        else:
            tau[i] = 0.0
    
    return tau


def curvature_2d(
    d1: np.ndarray,
    d2: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    计算 2D 曲线的曲率（带符号）。
    
    2D 曲率公式:
        κ = (x'y'' - y'x'') / (x'² + y'²)^{3/2}
    
    Args:
        d1: [N, 2] 一阶导数 (x', y')
        d2: [N, 2] 二阶导数 (x'', y'')
        eps: 数值稳定项
    
    Returns:
        [N,] 带符号的曲率
        - 正值: 逆时针弯曲
        - 负值: 顺时针弯曲
    
    Example:
        >>> # 单位圆
        >>> t = np.linspace(0, 2*np.pi, 100)
        >>> data = np.column_stack([np.cos(t), np.sin(t)])
        >>> d1, d2, _ = compute_derivatives(data)
        >>> kappa = curvature_2d(d1, d2)
        >>> print(np.mean(kappa))  # ≈ 1.0 (逆时针)
    """
    d1 = np.atleast_2d(d1)
    d2 = np.atleast_2d(d2)
    
    if d1.shape[1] != 2 or d2.shape[1] != 2:
        raise ValueError("curvature_2d requires 2D data")
    
    # x'y'' - y'x''
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    
    # (x'² + y'²)^{3/2}
    norm_d1_sq = np.sum(d1 ** 2, axis=1)
    denominator = np.power(norm_d1_sq + eps, 1.5)
    
    return cross / (denominator + eps)


def curvature_radius(kappa: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    计算曲率半径 R = 1/κ。
    
    Args:
        kappa: [N,] 曲率
        eps: 防止除零
    
    Returns:
        [N,] 曲率半径
    """
    return 1.0 / (np.abs(kappa) + eps)


def mean_curvature(kappa: np.ndarray) -> float:
    """
    计算平均曲率。
    
    Args:
        kappa: [N,] 曲率
    
    Returns:
        平均曲率值
    """
    return float(np.mean(np.abs(kappa)))


def total_curvature(kappa: np.ndarray, ds: np.ndarray = None) -> float:
    """
    计算总曲率 ∫|κ|ds。
    
    Args:
        kappa: [N,] 曲率
        ds: [N,] 弧长微元（可选，默认为 1）
    
    Returns:
        总曲率
    """
    if ds is None:
        ds = np.ones_like(kappa)
    return float(np.sum(np.abs(kappa) * ds))
