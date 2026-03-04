"""
PCA — 主成分分析

提供 PCA 投影、方差解释和加载矩阵
"""

import numpy as np
from typing import Tuple, Optional


def pca_projection(
    data: np.ndarray,
    n_components: int = 3,
    center: bool = True,
    return_model: bool = False
) -> np.ndarray:
    """
    对高维数据进行 PCA 降维投影
    
    Parameters
    ----------
    data : np.ndarray
        输入数据 [N, d]，N 个样本，d 维特征
    n_components : int
        目标维度
    center : bool
        是否中心化数据
    return_model : bool
        是否返回 (投影, 均值, 主成分)
        
    Returns
    -------
    np.ndarray
        降维后的数据 [N, n_components]
        如果 return_model=True，返回 (projected, mean, components)
    """
    data = np.asarray(data, dtype=np.float64)
    N, d = data.shape
    
    n_components = min(n_components, d, N)
    
    # 中心化
    if center:
        mean = data.mean(axis=0)
        X = data - mean
    else:
        mean = np.zeros(d)
        X = data
    
    # SVD 分解
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # 主成分
    components = Vt[:n_components]  # [n_components, d]
    
    # 投影
    projected = X @ components.T  # [N, n_components]
    
    if return_model:
        return projected, mean, components
    return projected


def pca_explained_variance(
    data: np.ndarray,
    n_components: Optional[int] = None,
    center: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 PCA 的解释方差比
    
    Parameters
    ----------
    data : np.ndarray
        输入数据 [N, d]
    n_components : int, optional
        计算前多少个主成分，None 表示全部
    center : bool
        是否中心化
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (explained_variance_ratio, cumulative_variance_ratio)
    """
    data = np.asarray(data, dtype=np.float64)
    N, d = data.shape
    
    if n_components is None:
        n_components = min(N, d)
    
    # 中心化
    if center:
        X = data - data.mean(axis=0)
    else:
        X = data
    
    # SVD
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    
    # 方差 = 奇异值的平方 / (N-1)
    explained_variance = (S ** 2) / (N - 1)
    total_var = explained_variance.sum()
    
    explained_ratio = explained_variance[:n_components] / total_var
    cumulative_ratio = np.cumsum(explained_ratio)
    
    return explained_ratio, cumulative_ratio


def pca_loadings(
    data: np.ndarray,
    n_components: int = 3,
    center: bool = True
) -> np.ndarray:
    """
    计算 PCA 加载矩阵（主成分与原特征的相关性）
    
    Parameters
    ----------
    data : np.ndarray
        输入数据 [N, d]
    n_components : int
        主成分数量
    center : bool
        是否中心化
        
    Returns
    -------
    np.ndarray
        加载矩阵 [n_components, d]
    """
    data = np.asarray(data, dtype=np.float64)
    N, d = data.shape
    
    n_components = min(n_components, d, N)
    
    if center:
        X = data - data.mean(axis=0)
    else:
        X = data
    
    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # 加载 = V * sqrt(方差)
    # 这里简化为返回 V 的前 n_components 行
    loadings = Vt[:n_components]
    
    return loadings


def reconstruct_from_pca(
    projected: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray
) -> np.ndarray:
    """
    从 PCA 投影重建原始数据
    
    Parameters
    ----------
    projected : np.ndarray
        降维数据 [N, k]
    mean : np.ndarray
        原始数据均值 [d,]
    components : np.ndarray
        主成分 [k, d]
        
    Returns
    -------
    np.ndarray
        重建数据 [N, d]
    """
    return projected @ components + mean
