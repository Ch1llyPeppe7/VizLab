"""
Entropy functions — Shannon entropy, differential entropy, joint entropy

适用于离散概率分布和连续数据的熵估计
"""

import numpy as np
from typing import Optional, Union


def shannon_entropy(
    p: np.ndarray,
    base: float = np.e,
    eps: float = 1e-12
) -> float:
    """
    计算离散概率分布的香农熵
    
    H(X) = -Σ p(x) log p(x)
    
    Parameters
    ----------
    p : np.ndarray
        概率分布（需满足 sum=1，所有元素非负）
    base : float
        对数的底，默认 e (nats)；使用 2 得到 bits
    eps : float
        数值稳定性常数
        
    Returns
    -------
    float
        熵值
    """
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]  # 只考虑正概率
    p = p / (p.sum() + eps)  # 归一化
    
    log_p = np.log(p + eps) / np.log(base)
    return -np.sum(p * log_p)


def differential_entropy(
    data: np.ndarray,
    method: str = 'kde',
    n_bins: int = 50,
    bandwidth: Optional[float] = None,
    base: float = np.e
) -> float:
    """
    估计连续随机变量的微分熵
    
    h(X) = -∫ f(x) log f(x) dx
    
    Parameters
    ----------
    data : np.ndarray
        1D 数据样本
    method : str
        估计方法: 'histogram', 'kde' (kernel density)
    n_bins : int
        histogram 方法的 bin 数量
    bandwidth : float, optional
        kde 的带宽，None 时使用 Scott's rule
    base : float
        对数底
        
    Returns
    -------
    float
        微分熵估计值
    """
    data = np.asarray(data).flatten()
    n = len(data)
    
    if method == 'histogram':
        # 直方图法
        counts, bin_edges = np.histogram(data, bins=n_bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        # 避免 log(0)
        counts = counts[counts > 0]
        log_counts = np.log(counts) / np.log(base)
        return -np.sum(counts * log_counts) * bin_width
    
    elif method == 'kde':
        # 核密度估计
        from scipy.stats import gaussian_kde
        
        if data.std() < 1e-10:
            return 0.0  # 常数数据
            
        kde = gaussian_kde(data, bw_method=bandwidth)
        # Monte Carlo 积分
        x_eval = np.linspace(data.min(), data.max(), 1000)
        f_x = kde(x_eval)
        f_x = f_x[f_x > 0]
        log_f = np.log(f_x) / np.log(base)
        dx = (data.max() - data.min()) / 1000
        return -np.sum(f_x * log_f) * dx
    
    else:
        raise ValueError(f"Unknown method: {method}")


def joint_entropy(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 30,
    base: float = np.e
) -> float:
    """
    计算两个变量的联合熵
    
    H(X, Y) = -Σ p(x,y) log p(x,y)
    
    Parameters
    ----------
    x, y : np.ndarray
        两个同长度的 1D 数组
    n_bins : int
        每个维度的 bin 数量
    base : float
        对数底
        
    Returns
    -------
    float
        联合熵
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # 2D 直方图
    hist2d, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)
    
    # 计算概率 (density → probability)
    dx = (x.max() - x.min()) / n_bins
    dy = (y.max() - y.min()) / n_bins
    p = hist2d * dx * dy
    
    p = p.flatten()
    p = p[p > 0]
    p = p / (p.sum() + 1e-12)
    
    log_p = np.log(p) / np.log(base)
    return -np.sum(p * log_p)
