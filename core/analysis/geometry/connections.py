"""
联络结构与曲率谱分析

提供 Christoffel 符号和曲率谱的计算函数。
"""

import numpy as np
from typing import Tuple, Optional
from .derivatives import compute_derivatives


def christoffel_connection(
    data: np.ndarray,
    h: float = 1.0,
    eps: float = 1e-12
) -> np.ndarray:
    """
    计算 Christoffel-like 联络结构。
    
    对于 1D 参数曲线，计算每个维度的"局部曲率贡献":
        Γ_k(t) = |d²γ_k/dt²| / (|dγ_k/dt| + ε)
    
    这反映了每个维度在各位置的"联络强度"。
    
    Args:
        data: [N, d] 数据矩阵
        h: 差分步长
        eps: 数值稳定项
    
    Returns:
        [N, d] Christoffel-like 矩阵
    
    用途:
        - 可视化哪些维度在哪些位置"转向"最剧烈
        - 分析编码的各维度特性
    
    Example:
        >>> data = np.random.randn(100, 64)
        >>> Gamma = christoffel_connection(data)
        >>> # 可以用热力图可视化 Gamma
    """
    d1, d2, _ = compute_derivatives(data, h=h)
    
    # Γ_k(t) = |d²γ_k/dt²| / (|dγ_k/dt| + ε)
    christoffel = np.abs(d2) / (np.abs(d1) + eps)
    
    return christoffel


def curvature_spectrum(
    data: np.ndarray,
    h: float = 1.0,
    pair_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算曲率谱：各 2D 子空间的曲率贡献。
    
    将高维数据分解为多个 2D 子空间（如 [dim_0, dim_1], [dim_2, dim_3], ...），
    计算每个子空间的平均曲率。
    
    Args:
        data: [N, d] 数据矩阵，d 应为偶数
        h: 差分步长
        pair_indices: [m, 2] 可选，指定哪些维度配对
                      默认为 [[0,1], [2,3], ...]
    
    Returns:
        subspace_idx: [m,] 子空间索引
        kappa_spectrum: [m,] 各子空间的平均曲率
    
    用途:
        - 分析 Sinusoidal/RoPE 等 PE 的频率-曲率关系
        - 各 2D 子空间对应不同频率，曲率 ≈ 频率
    
    Example:
        >>> # 对于 64 维数据，有 32 个 2D 子空间
        >>> data = np.random.randn(100, 64)
        >>> idx, kappa = curvature_spectrum(data)
        >>> print(len(kappa))  # 32
    """
    data = np.atleast_2d(data)
    N, d = data.shape
    
    # 默认配对方式: [0,1], [2,3], ...
    if pair_indices is None:
        m = d // 2
        pair_indices = np.array([[2*k, 2*k+1] for k in range(m)])
    else:
        pair_indices = np.atleast_2d(pair_indices)
        m = pair_indices.shape[0]
    
    kappa_spectrum = np.zeros(m)
    eps = 1e-12
    
    for k in range(m):
        i, j = pair_indices[k]
        
        # 提取 2D 子空间
        data_2d = data[:, [i, j]]  # [N, 2]
        
        # 计算导数
        d1, d2, _ = compute_derivatives(data_2d, h=h)
        
        # 2D 曲率: |x'y'' - y'x''| / (x'² + y'²)^{3/2}
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        norm_d1_sq = np.sum(d1 ** 2, axis=1)
        kappa_2d = np.abs(cross) / (np.power(norm_d1_sq + eps, 1.5))
        
        kappa_spectrum[k] = np.mean(kappa_2d)
    
    subspace_idx = np.arange(m)
    return subspace_idx, kappa_spectrum


def connection_divergence(
    data: np.ndarray,
    h: float = 1.0
) -> np.ndarray:
    """
    计算联络的散度（沿参数方向）。
    
    这可以用于检测曲线的"加速度变化率"。
    
    Args:
        data: [N, d] 数据矩阵
        h: 差分步长
    
    Returns:
        [N, d] 联络散度
    """
    Gamma = christoffel_connection(data, h)
    
    # 计算 Γ 的一阶导数
    dGamma, _, _ = compute_derivatives(Gamma, h=h)
    
    return dGamma


def parallel_transport_deviation(
    data: np.ndarray,
    vector: np.ndarray,
    h: float = 1.0
) -> np.ndarray:
    """
    计算向量沿曲线平行移动的偏差。
    
    在弯曲空间中，平行移动会导致向量旋转。
    此函数估计这种旋转的程度。
    
    Args:
        data: [N, d] 曲线数据
        vector: [d,] 初始向量
        h: 差分步长
    
    Returns:
        [N,] 各点处的偏差角度（弧度）
    
    注意:
        这是一个简化的近似，假设度量接近欧氏度量。
    """
    d1, d2, _ = compute_derivatives(data, h=h)
    
    # 切向量方向
    tangent = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-12)
    
    # 向量在切向量方向的投影
    projection = np.sum(vector * tangent, axis=1)
    
    # 偏差 ≈ 累积的角度变化
    # 使用相邻切向量的夹角近似
    cos_angle = np.sum(tangent[:-1] * tangent[1:], axis=1)
    cos_angle = np.clip(cos_angle, -1, 1)
    angles = np.arccos(cos_angle)
    
    # 累积角度
    cumulative_deviation = np.zeros(len(data))
    cumulative_deviation[1:] = np.cumsum(angles)
    
    return cumulative_deviation
