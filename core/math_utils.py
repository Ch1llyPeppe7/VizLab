
"""
Core Mathematical Utilities — 通用数学工具函数

包含原 lape_math.py 的核心功能（向后兼容），以及扩展的通用数学工具。

Modules:
    - FrequencyFunction: 频率函数生成器（向后兼容）
    - KernelFunction: 核函数计算（向后兼容）
    - 通用线性代数 / 信号处理 / 动力系统工具
"""

import numpy as np
from typing import Union, Tuple, Optional, Callable, List
from scipy import linalg, signal
import warnings


# ============================================================
#  向后兼容：原 lape_math.py 中的类
# ============================================================

class FrequencyFunction:
    """
    频率函数生成器（向后兼容 lape_math.py）。
    
    ω_k = (-k/d)^a,  k = 0, 1, ..., m-1
    """
    
    def __init__(self, dim: int = 64, power: float = 3.0):
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")
        self.dim = dim
        self.m = dim // 2
        self.power = power
        
        k = np.arange(self.m)
        d_float = float(dim)
        base = -k / d_float
        with np.errstate(divide='ignore', invalid='ignore'):
            freqs = np.sign(base) * np.abs(base) ** power
            freqs[0] = 0.0
        self.frequencies = freqs
        
    def get_frequencies(self) -> np.ndarray:
        return self.frequencies.copy()


class KernelFunction:
    """核函数（向后兼容 lape_math.py）。"""
    
    def __init__(self, freq_func: FrequencyFunction):
        self.freq_func = freq_func
        self.frequencies = freq_func.get_frequencies()
        self.m = len(self.frequencies)
        
    def compute(self, delta_q: Union[float, np.ndarray]) -> complex:
        phases = np.outer(self.frequencies, np.atleast_1d(delta_q))
        kernel_values = np.sum(np.exp(1j * phases), axis=0)
        if np.isscalar(delta_q) or len(np.atleast_1d(delta_q)) == 1:
            return complex(kernel_values[0])
        return kernel_values
    
    def compute_real(self, delta_q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        phases = np.outer(self.frequencies, np.atleast_1d(delta_q))
        real_values = np.sum(np.cos(phases), axis=0)
        if np.isscalar(delta_q) or len(np.atleast_1d(delta_q)) == 1:
            return float(real_values[0])
        return real_values
    
    def compute_matrix(self, positions: np.ndarray) -> np.ndarray:
        delta_q = positions[:, None] - positions[None, :]
        K_matrix = self.compute_real(delta_q.flatten()).reshape(delta_q.shape)
        return K_matrix


# ============================================================
#  扩展：通用数学工具
# ============================================================

def random_orthogonal_matrix(n: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    生成 n×n 随机正交矩阵（Haar 分布），用于模拟神经网络权重。
    
    通过 QR 分解 + 符号修正确保均匀分布。
    
    Args:
        n: 矩阵维度
        rng: numpy 随机数生成器
    Returns:
        [n, n] 正交矩阵
    """
    if rng is None:
        rng = np.random.default_rng()
    H = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(H)
    # 修正符号使分布均匀
    d = np.diag(R)
    Q *= np.sign(d)
    return Q


def random_weight_matrix(n: int, m: int = None, scale: float = 1.0,
                         rng: np.random.Generator = None) -> np.ndarray:
    """
    生成随机权重矩阵（模拟神经网络初始化）。
    
    使用 Glorot/Xavier 初始化：W ~ N(0, 2/(n+m))
    
    Args:
        n: 输入维度
        m: 输出维度（默认等于 n）
        scale: 缩放因子
        rng: 随机数生成器
    Returns:
        [n, m] 权重矩阵
    """
    if m is None:
        m = n
    if rng is None:
        rng = np.random.default_rng()
    std = scale * np.sqrt(2.0 / (n + m))
    return rng.standard_normal((n, m)) * std


def compute_lyapunov_exponent(trajectory: np.ndarray, dt: float = 1.0) -> float:
    """
    估算离散动力系统的最大 Lyapunov 指数。
    
    使用相邻点距离的指数增长率估算：
        λ ≈ (1/T) Σ log|δx(t+1)/δx(t)|
    
    Args:
        trajectory: [T, d] 状态轨迹
        dt: 时间步长
    Returns:
        最大 Lyapunov 指数（正值 = 混沌）
    """
    T = trajectory.shape[0] - 1
    if T < 2:
        return 0.0
    
    diffs = np.diff(trajectory, axis=0)  # [T, d]
    norms = np.linalg.norm(diffs, axis=1)  # [T,]
    
    # 避免 log(0)
    norms = np.maximum(norms, 1e-15)
    
    # 估算指数增长率
    log_ratios = np.diff(np.log(norms))
    return float(np.mean(log_ratios) / dt)


def compute_phase_space(signal_1d: np.ndarray, delay: int = 1, 
                        embedding_dim: int = 2) -> np.ndarray:
    """
    延时嵌入构造相空间（Takens 定理）。
    
    从一维时间序列构造高维相空间轨迹：
        X(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]
    
    Args:
        signal_1d: [T,] 一维信号
        delay: 延时 τ
        embedding_dim: 嵌入维度 d
    Returns:
        [T', embedding_dim] 相空间轨迹
    """
    T = len(signal_1d)
    n_points = T - (embedding_dim - 1) * delay
    if n_points <= 0:
        raise ValueError(f"Signal too short for delay={delay}, dim={embedding_dim}")
    
    phase = np.zeros((n_points, embedding_dim))
    for i in range(embedding_dim):
        phase[:, i] = signal_1d[i * delay : i * delay + n_points]
    return phase


def spectral_entropy(psd: np.ndarray) -> float:
    """
    计算功率谱密度的谱熵。
    
    H = -Σ p_k log(p_k)，其中 p_k = PSD_k / Σ PSD
    
    高谱熵 → 频率分布均匀（如白噪声）
    低谱熵 → 能量集中在少数频率（如正弦波）
    
    Args:
        psd: 功率谱密度数组
    Returns:
        归一化谱熵 [0, 1]
    """
    psd = np.abs(psd)
    psd = psd / (psd.sum() + 1e-15)
    psd = psd[psd > 0]
    H = -np.sum(psd * np.log2(psd))
    H_max = np.log2(len(psd)) if len(psd) > 0 else 1.0
    return float(H / H_max) if H_max > 0 else 0.0


def mutual_information_discrete(x: np.ndarray, y: np.ndarray, 
                                n_bins: int = 50) -> float:
    """
    估算两个连续变量之间的互信息（离散化方法）。
    
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Args:
        x: [N,] 变量 X
        y: [N,] 变量 Y
        n_bins: 直方图 bin 数
    Returns:
        互信息估计值（bits）
    """
    c_xy = np.histogram2d(x, y, bins=n_bins)[0]
    c_x = np.histogram(x, bins=n_bins)[0]
    c_y = np.histogram(y, bins=n_bins)[0]
    
    def _entropy(counts):
        p = counts / (counts.sum() + 1e-15)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    return float(_entropy(c_x) + _entropy(c_y) - _entropy(c_xy))


def effective_rank(matrix: np.ndarray) -> float:
    """
    矩阵有效秩（基于奇异值的谱熵）。
    
    erank(A) = exp(H(σ))，其中 σ 是归一化奇异值分布
    
    Args:
        matrix: [m, n] 矩阵
    Returns:
        有效秩（连续值）
    """
    svs = np.linalg.svd(matrix, compute_uv=False)
    svs = svs[svs > 1e-10]
    p = svs / svs.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def condition_number_profile(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    矩阵的条件数谱：逐步截断奇异值后的条件数变化。
    
    用于分析矩阵的数值稳定性。
    
    Args:
        matrix: [m, n] 矩阵
    Returns:
        (ranks, condition_numbers) 两个数组
    """
    svs = np.linalg.svd(matrix, compute_uv=False)
    svs = svs[svs > 1e-15]
    ranks = np.arange(1, len(svs) + 1)
    cond_nums = svs[0] / svs
    return ranks, cond_nums


def activation_fn(x: np.ndarray, name: str = 'gelu') -> np.ndarray:
    """
    常用激活函数（纯 numpy 实现）。
    
    Args:
        x: 输入
        name: 'relu' | 'gelu' | 'silu' | 'tanh' | 'sigmoid'
    Returns:
        激活后的值
    """
    if name == 'relu':
        return np.maximum(0, x)
    elif name == 'gelu':
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    elif name == 'silu':  # swish
        return x / (1 + np.exp(-x))
    elif name == 'tanh':
        return np.tanh(x)
    elif name == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    else:
        raise ValueError(f"Unknown activation: {name}")


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization（纯 numpy）。
    
    LN(x) = (x - μ) / √(σ² + ε)
    
    Args:
        x: [..., d] 输入（在最后一维上归一化）
        eps: 数值稳定项
    Returns:
        归一化结果
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def simulate_feedforward_layer(x: np.ndarray, dim_ff: int = None,
                               activation: str = 'gelu',
                               rng: np.random.Generator = None) -> np.ndarray:
    """
    模拟 Transformer FFN 层的前向传播。
    
    FFN(x) = W_2 · σ(W_1 · x + b_1) + b_2
    
    Args:
        x: [N, d] 输入
        dim_ff: 中间维度（默认 4d）
        activation: 激活函数名
        rng: 随机数生成器
    Returns:
        [N, d] 输出
    """
    d = x.shape[-1]
    if dim_ff is None:
        dim_ff = 4 * d
    if rng is None:
        rng = np.random.default_rng()
    
    W1 = random_weight_matrix(d, dim_ff, rng=rng)
    b1 = rng.standard_normal(dim_ff) * 0.01
    W2 = random_weight_matrix(dim_ff, d, rng=rng)
    b2 = rng.standard_normal(d) * 0.01
    
    h = activation_fn(x @ W1 + b1, name=activation)
    return h @ W2 + b2
