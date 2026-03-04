"""
数值微分工具

提供中心差分法计算高维曲线的各阶导数。
"""

import numpy as np
from typing import Tuple, Optional


def compute_derivatives(
    data: np.ndarray,
    h: float = 1.0,
    boundary: str = 'reflect'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算离散数据的一阶、二阶、三阶导数（中心差分法）。
    
    将 data[i] 视为参数曲线 γ(t) 在 t=i 处的采样值，
    使用中心差分近似导数。
    
    Args:
        data: [N, d] 数据矩阵，N 为样本数，d 为维度
        h: 差分步长（默认为 1.0，即相邻索引的间隔）
        boundary: 边界处理方式
            - 'reflect': 反射边界（默认）
            - 'constant': 常数外推
            - 'wrap': 周期边界
    
    Returns:
        d1: [N, d] 一阶导数 γ'(t)
        d2: [N, d] 二阶导数 γ''(t)
        d3: [N, d] 三阶导数 γ'''(t)
    
    公式:
        一阶: γ'(t) ≈ (γ(t+h) - γ(t-h)) / 2h
        二阶: γ''(t) ≈ (γ(t+h) - 2γ(t) + γ(t-h)) / h²
        三阶: γ'''(t) ≈ (γ(t+2h) - 2γ(t+h) + 2γ(t-h) - γ(t-2h)) / 2h³
    
    Example:
        >>> data = np.random.randn(100, 64)
        >>> d1, d2, d3 = compute_derivatives(data, h=1.0)
        >>> print(d1.shape)  # (100, 64)
    """
    data = np.atleast_2d(data)
    N, d = data.shape
    
    # 创建填充后的数据以处理边界
    if boundary == 'reflect':
        # 反射填充
        padded = np.pad(data, ((2, 2), (0, 0)), mode='reflect')
    elif boundary == 'constant':
        # 常数填充（使用边界值）
        padded = np.pad(data, ((2, 2), (0, 0)), mode='edge')
    elif boundary == 'wrap':
        # 周期填充
        padded = np.pad(data, ((2, 2), (0, 0)), mode='wrap')
    else:
        raise ValueError(f"Unknown boundary mode: {boundary}")
    
    # 提取各偏移位置的数据
    gamma = padded[2:N+2]           # γ(t)
    gamma_plus = padded[3:N+3]      # γ(t+h)
    gamma_minus = padded[1:N+1]     # γ(t-h)
    gamma_plus2 = padded[4:N+4]     # γ(t+2h)
    gamma_minus2 = padded[0:N]      # γ(t-2h)
    
    # 一阶导数: (γ(t+h) - γ(t-h)) / 2h
    d1 = (gamma_plus - gamma_minus) / (2 * h)
    
    # 二阶导数: (γ(t+h) - 2γ(t) + γ(t-h)) / h²
    d2 = (gamma_plus - 2 * gamma + gamma_minus) / (h ** 2)
    
    # 三阶导数: (γ(t+2h) - 2γ(t+h) + 2γ(t-h) - γ(t-2h)) / 2h³
    d3 = (gamma_plus2 - 2*gamma_plus + 2*gamma_minus - gamma_minus2) / (2 * h**3)
    
    return d1, d2, d3


def compute_derivatives_1d(
    signal: np.ndarray,
    h: float = 1.0,
    boundary: str = 'reflect'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算一维信号的各阶导数。
    
    Args:
        signal: [N,] 一维信号
        h: 差分步长
        boundary: 边界处理方式
    
    Returns:
        d1, d2, d3: [N,] 各阶导数
    
    Example:
        >>> signal = np.sin(np.linspace(0, 2*np.pi, 100))
        >>> d1, d2, d3 = compute_derivatives_1d(signal)
        >>> # d1 ≈ cos, d2 ≈ -sin
    """
    signal = np.atleast_1d(signal).reshape(-1, 1)
    d1, d2, d3 = compute_derivatives(signal, h, boundary)
    return d1.ravel(), d2.ravel(), d3.ravel()


def gradient_magnitude(d1: np.ndarray) -> np.ndarray:
    """
    计算梯度幅值（一阶导数的范数）。
    
    Args:
        d1: [N, d] 一阶导数
    
    Returns:
        [N,] 梯度幅值 ||γ'(t)||
    """
    return np.sqrt(np.sum(d1 ** 2, axis=1))
