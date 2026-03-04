"""
Fisher Information — Fisher 信息矩阵

用于度量参数空间中的"信息量"和参数敏感性
"""

import numpy as np
from typing import Optional, Callable


def fisher_information_matrix(
    embeddings: np.ndarray,
    h: float = 1.0,
    eps: float = 1e-12
) -> np.ndarray:
    """
    计算嵌入轨迹的 Fisher 信息矩阵
    
    对于参数化曲线 γ(t): ℝ → ℝᵈ，Fisher 信息为:
    I(t) = (dγ/dt)ᵀ (dγ/dt) = ||γ'(t)||²
    
    对于多维情形，计算 d×d 的 Fisher 矩阵:
    F_ij(t) = Σ_k (∂γ_k/∂θ_i)(∂γ_k/∂θ_j)
    
    这里简化为单参数情形，返回每个位置的 Fisher 信息标量
    
    Parameters
    ----------
    embeddings : np.ndarray
        嵌入矩阵 [N, d]，N 个位置，d 维嵌入
    h : float
        数值微分步长（位置间隔）
    eps : float
        数值稳定性
        
    Returns
    -------
    np.ndarray
        Fisher 信息数组 [N,]
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    N, d = embeddings.shape
    
    # 使用中心差分计算一阶导数
    d1 = np.zeros_like(embeddings)
    
    # 边界: 前向/后向差分
    d1[0] = (embeddings[1] - embeddings[0]) / h
    d1[-1] = (embeddings[-1] - embeddings[-2]) / h
    
    # 内部: 中心差分
    d1[1:-1] = (embeddings[2:] - embeddings[:-2]) / (2 * h)
    
    # Fisher 信息 = ||γ'||²
    fisher = np.sum(d1 ** 2, axis=1)
    
    return fisher


def fisher_information_scalar(
    embeddings: np.ndarray,
    h: float = 1.0
) -> float:
    """
    计算整体平均 Fisher 信息
    
    Parameters
    ----------
    embeddings : np.ndarray
        嵌入矩阵 [N, d]
    h : float
        位置步长
        
    Returns
    -------
    float
        平均 Fisher 信息
    """
    return np.mean(fisher_information_matrix(embeddings, h))


def fisher_information_from_gradients(
    gradients: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    从预计算的梯度直接计算 Fisher 信息
    
    当已有 ∂f/∂θ 时使用此函数
    
    Parameters
    ----------
    gradients : np.ndarray
        梯度矩阵 [N, d]，每行是一个样本的梯度向量
        
    Returns
    -------
    np.ndarray
        Fisher 信息数组 [N,]
    """
    return np.sum(gradients ** 2, axis=1)


def fisher_rao_distance(
    embeddings: np.ndarray,
    i: int,
    j: int,
    h: float = 1.0
) -> float:
    """
    计算两个位置之间的 Fisher-Rao 距离（测地距离）
    
    d_FR(i, j) = ∫_i^j √(F(t)) dt
    
    Parameters
    ----------
    embeddings : np.ndarray
        嵌入矩阵 [N, d]
    i, j : int
        起始和终止位置索引
    h : float
        位置步长
        
    Returns
    -------
    float
        Fisher-Rao 测地距离
    """
    if i > j:
        i, j = j, i
    
    fisher = fisher_information_matrix(embeddings, h)
    
    # 数值积分
    sqrt_fisher = np.sqrt(fisher[i:j+1])
    distance = np.trapz(sqrt_fisher) * h
    
    return distance
