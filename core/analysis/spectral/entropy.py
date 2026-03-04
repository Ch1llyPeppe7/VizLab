"""
频谱特征计算

提供谱熵、谱平坦度、谱质心等频谱特征。
"""

import numpy as np
from typing import Optional


def spectral_entropy(
    psd: np.ndarray,
    normalize: bool = True,
    eps: float = 1e-12
) -> float:
    """
    计算谱熵（Spectral Entropy）。
    
    谱熵衡量功率谱的"平坦度"或"复杂度":
        H = -Σ p(f) log p(f)
    
    其中 p(f) = PSD(f) / Σ PSD 是归一化的功率谱。
    
    Args:
        psd: 功率谱密度（可以是未归一化的）
        normalize: 是否将结果归一化到 [0, 1]
        eps: 数值稳定项
    
    Returns:
        谱熵值
        - 高谱熵 → 白噪声（均匀分布）
        - 低谱熵 → 周期信号（集中分布）
    
    Example:
        >>> # 白噪声的谱熵接近 1
        >>> white_noise = np.random.randn(1000)
        >>> freqs, psd = fft_power_spectrum(white_noise)
        >>> H = spectral_entropy(psd, normalize=True)
        >>> print(f"White noise entropy: {H:.3f}")  # ≈ 1.0
    """
    psd = np.asarray(psd).flatten()
    
    # 归一化为概率分布
    psd_sum = np.sum(psd)
    if psd_sum < eps:
        return 0.0
    
    p = psd / psd_sum
    
    # 避免 log(0)
    p = np.clip(p, eps, 1.0)
    
    # Shannon 熵
    H = -np.sum(p * np.log(p))
    
    if normalize:
        # 最大熵 = log(N)
        H_max = np.log(len(p))
        H = H / (H_max + eps)
    
    return float(H)


def spectral_flatness(
    psd: np.ndarray,
    eps: float = 1e-12
) -> float:
    """
    计算谱平坦度（Spectral Flatness / Wiener Entropy）。
    
    谱平坦度 = 几何平均 / 算术平均
    
    取值范围 [0, 1]:
        - 1 → 完全平坦（白噪声）
        - 0 → 非常尖锐（纯音）
    
    Args:
        psd: 功率谱密度
        eps: 数值稳定项
    
    Returns:
        谱平坦度
    """
    psd = np.asarray(psd).flatten()
    psd = np.maximum(psd, eps)
    
    # 几何平均 = exp(mean(log(psd)))
    geometric_mean = np.exp(np.mean(np.log(psd)))
    
    # 算术平均
    arithmetic_mean = np.mean(psd)
    
    return float(geometric_mean / (arithmetic_mean + eps))


def spectral_centroid(
    psd: np.ndarray,
    freqs: np.ndarray
) -> float:
    """
    计算谱质心（Spectral Centroid）。
    
    谱质心 = Σ f · PSD(f) / Σ PSD(f)
    
    是功率谱的"重心"，反映信号的"亮度"。
    
    Args:
        psd: 功率谱密度
        freqs: 频率轴
    
    Returns:
        谱质心频率
    """
    psd = np.asarray(psd).flatten()
    freqs = np.asarray(freqs).flatten()
    
    psd_sum = np.sum(psd)
    if psd_sum < 1e-12:
        return 0.0
    
    centroid = np.sum(freqs * psd) / psd_sum
    
    return float(centroid)


def spectral_spread(
    psd: np.ndarray,
    freqs: np.ndarray,
    centroid: Optional[float] = None
) -> float:
    """
    计算谱扩展度（Spectral Spread）。
    
    谱扩展度 = √(Σ (f - centroid)² · PSD(f) / Σ PSD(f))
    
    是功率谱围绕质心的"标准差"。
    
    Args:
        psd: 功率谱密度
        freqs: 频率轴
        centroid: 预计算的谱质心（可选）
    
    Returns:
        谱扩展度
    """
    psd = np.asarray(psd).flatten()
    freqs = np.asarray(freqs).flatten()
    
    if centroid is None:
        centroid = spectral_centroid(psd, freqs)
    
    psd_sum = np.sum(psd)
    if psd_sum < 1e-12:
        return 0.0
    
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum)
    
    return float(spread)


def spectral_rolloff(
    psd: np.ndarray,
    freqs: np.ndarray,
    threshold: float = 0.85
) -> float:
    """
    计算谱滚降点（Spectral Rolloff）。
    
    谱滚降点是包含 threshold 比例总功率的最低频率。
    
    Args:
        psd: 功率谱密度
        freqs: 频率轴
        threshold: 功率阈值（默认 0.85）
    
    Returns:
        滚降频率
    """
    psd = np.asarray(psd).flatten()
    freqs = np.asarray(freqs).flatten()
    
    # 累积功率
    cumsum = np.cumsum(psd)
    total = cumsum[-1]
    
    if total < 1e-12:
        return freqs[0]
    
    # 找到超过阈值的第一个索引
    idx = np.searchsorted(cumsum, threshold * total)
    idx = min(idx, len(freqs) - 1)
    
    return float(freqs[idx])


def spectral_slope(
    psd: np.ndarray,
    freqs: np.ndarray
) -> float:
    """
    计算谱斜率（Spectral Slope）。
    
    对 log(PSD) vs log(freq) 做线性回归，返回斜率。
    
    Args:
        psd: 功率谱密度
        freqs: 频率轴
    
    Returns:
        谱斜率
        - 粉红噪声: 斜率 ≈ -1
        - 白噪声: 斜率 ≈ 0
    """
    psd = np.asarray(psd).flatten()
    freqs = np.asarray(freqs).flatten()
    
    # 忽略零频率
    mask = freqs > 0
    psd = psd[mask]
    freqs = freqs[mask]
    
    if len(psd) < 2:
        return 0.0
    
    # 对数变换
    log_freqs = np.log(freqs + 1e-12)
    log_psd = np.log(psd + 1e-12)
    
    # 线性回归
    slope = np.polyfit(log_freqs, log_psd, 1)[0]
    
    return float(slope)
