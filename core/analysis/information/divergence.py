"""
Divergence measures — KL 散度, JS 散度

用于度量概率分布之间的差异
"""

import numpy as np
from typing import Optional


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: float = np.e,
    eps: float = 1e-12
) -> float:
    """
    计算 KL 散度 D_KL(P || Q)
    
    D_KL(P || Q) = Σ p(x) log(p(x) / q(x))
    
    Parameters
    ----------
    p, q : np.ndarray
        两个概率分布（需非负且和为1）
    base : float
        对数底
    eps : float
        数值稳定性
        
    Returns
    -------
    float
        KL 散度（非负）
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # 归一化
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    
    # 只在 p > 0 处计算
    mask = p > eps
    kl = np.sum(p[mask] * np.log((p[mask] + eps) / (q[mask] + eps))) / np.log(base)
    
    return max(0, kl)


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: float = np.e,
    eps: float = 1e-12
) -> float:
    """
    计算 Jensen-Shannon 散度
    
    JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    其中 M = 0.5 * (P + Q)
    
    Parameters
    ----------
    p, q : np.ndarray
        两个概率分布
    base : float
        对数底
    eps : float
        数值稳定性
        
    Returns
    -------
    float
        JS 散度（范围 [0, log(2)]）
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # 归一化
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    
    # 中间分布
    m = 0.5 * (p + q)
    
    js = 0.5 * kl_divergence(p, m, base, eps) + 0.5 * kl_divergence(q, m, base, eps)
    return js


def symmetric_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: float = np.e,
    eps: float = 1e-12
) -> float:
    """
    对称 KL 散度
    
    D_sym(P, Q) = 0.5 * (D_KL(P || Q) + D_KL(Q || P))
    
    Parameters
    ----------
    p, q : np.ndarray
        两个概率分布
    base : float
        对数底
    eps : float
        数值稳定性
        
    Returns
    -------
    float
        对称 KL 散度
    """
    return 0.5 * (kl_divergence(p, q, base, eps) + kl_divergence(q, p, base, eps))


def total_variation_distance(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-12
) -> float:
    """
    Total Variation 距离
    
    TV(P, Q) = 0.5 * Σ |p(x) - q(x)|
    
    Parameters
    ----------
    p, q : np.ndarray
        两个概率分布
    eps : float
        数值稳定性
        
    Returns
    -------
    float
        TV 距离（范围 [0, 1]）
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    
    return 0.5 * np.sum(np.abs(p - q))


def hellinger_distance(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-12
) -> float:
    """
    Hellinger 距离
    
    H(P, Q) = (1/√2) * √(Σ (√p - √q)²)
    
    Parameters
    ----------
    p, q : np.ndarray
        两个概率分布
    eps : float
        数值稳定性
        
    Returns
    -------
    float
        Hellinger 距离（范围 [0, 1]）
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
