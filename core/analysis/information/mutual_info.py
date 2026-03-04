"""
Mutual Information — 互信息及条件互信息

I(X; Y) = H(X) + H(Y) - H(X, Y)
"""

import numpy as np
from typing import Optional


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 30,
    base: float = np.e,
    method: str = 'histogram'
) -> float:
    """
    计算两个随机变量的互信息
    
    I(X; Y) = H(X) + H(Y) - H(X, Y)
    
    Parameters
    ----------
    x, y : np.ndarray
        两个同长度的 1D 数组
    n_bins : int
        每个维度的 bin 数量
    base : float
        对数底
    method : str
        'histogram' 或 'knn' (k-nearest neighbors)
        
    Returns
    -------
    float
        互信息值（非负）
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if method == 'histogram':
        return _mi_histogram(x, y, n_bins, base)
    elif method == 'knn':
        return _mi_knn(x, y, base)
    else:
        raise ValueError(f"Unknown method: {method}")


def _mi_histogram(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    base: float
) -> float:
    """直方图法计算互信息"""
    eps = 1e-12
    
    # 边缘分布
    hist_x, _ = np.histogram(x, bins=n_bins, density=True)
    hist_y, _ = np.histogram(y, bins=n_bins, density=True)
    
    # 联合分布
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)
    
    # 归一化为概率
    dx = (x.max() - x.min()) / n_bins
    dy = (y.max() - y.min()) / n_bins
    
    p_x = hist_x * dx
    p_x = p_x / (p_x.sum() + eps)
    
    p_y = hist_y * dy
    p_y = p_y / (p_y.sum() + eps)
    
    p_xy = hist_xy * dx * dy
    p_xy = p_xy / (p_xy.sum() + eps)
    
    # H(X)
    p_x_nz = p_x[p_x > 0]
    H_x = -np.sum(p_x_nz * np.log(p_x_nz) / np.log(base))
    
    # H(Y)
    p_y_nz = p_y[p_y > 0]
    H_y = -np.sum(p_y_nz * np.log(p_y_nz) / np.log(base))
    
    # H(X, Y)
    p_xy_flat = p_xy.flatten()
    p_xy_nz = p_xy_flat[p_xy_flat > 0]
    H_xy = -np.sum(p_xy_nz * np.log(p_xy_nz) / np.log(base))
    
    mi = H_x + H_y - H_xy
    return max(0, mi)  # 理论上非负


def _mi_knn(
    x: np.ndarray,
    y: np.ndarray,
    base: float,
    k: int = 3
) -> float:
    """
    KNN 法计算互信息（Kraskov estimator）
    
    参考: Kraskov et al. (2004) "Estimating mutual information"
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
        # sklearn 返回 nats
        mi = mutual_info_regression(
            x.reshape(-1, 1), y, 
            n_neighbors=k, 
            random_state=42
        )[0]
        return mi / np.log(base) * np.log(np.e)
    except ImportError:
        # fallback to histogram
        return _mi_histogram(x, y, n_bins=30, base=base)


def conditional_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_bins: int = 20,
    base: float = np.e
) -> float:
    """
    计算条件互信息
    
    I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
    
    Parameters
    ----------
    x, y, z : np.ndarray
        三个同长度的 1D 数组
    n_bins : int
        每个维度的 bin 数量
    base : float
        对数底
        
    Returns
    -------
    float
        条件互信息（理论上非负）
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()
    
    eps = 1e-12
    
    def hist_entropy(data, bins):
        """计算多维直方图的熵"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        hist, _ = np.histogramdd(data, bins=bins, density=True)
        # 计算 bin 体积
        vol = 1.0
        for i in range(data.shape[1]):
            vol *= (data[:, i].max() - data[:, i].min()) / bins
        
        p = hist.flatten() * vol
        p = p[p > 0]
        p = p / (p.sum() + eps)
        return -np.sum(p * np.log(p) / np.log(base))
    
    # H(Z)
    H_z = hist_entropy(z, n_bins)
    
    # H(X, Z)
    H_xz = hist_entropy(np.column_stack([x, z]), n_bins)
    
    # H(Y, Z)
    H_yz = hist_entropy(np.column_stack([y, z]), n_bins)
    
    # H(X, Y, Z)
    H_xyz = hist_entropy(np.column_stack([x, y, z]), n_bins)
    
    cmi = H_xz + H_yz - H_xyz - H_z
    return max(0, cmi)
