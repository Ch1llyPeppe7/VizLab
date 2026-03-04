"""
功率谱密度估计

提供 Welch、周期图等 PSD 估计方法。
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal as sp_signal


def welch_psd(
    data: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    axis: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Welch 方法估计功率谱密度。
    
    Welch 方法通过分段加窗平均来减少方差，
    是最常用的 PSD 估计方法之一。
    
    Args:
        data: 输入数据
        fs: 采样频率 (Hz)
        nperseg: 每段长度（默认为 N//8）
        noverlap: 段重叠长度（默认为 nperseg//2）
        window: 窗函数名称
        axis: 计算轴
    
    Returns:
        freqs: 频率轴
        psd: 功率谱密度
    
    Example:
        >>> t = np.linspace(0, 10, 10000)
        >>> signal = np.sin(2*np.pi*10*t) + np.random.randn(len(t))*0.5
        >>> freqs, psd = welch_psd(signal, fs=1000)
    """
    data = np.asarray(data)
    N = data.shape[axis]
    
    if nperseg is None:
        nperseg = min(256, N)
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    freqs, psd = sp_signal.welch(
        data, fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        axis=axis
    )
    
    return freqs, psd


def periodogram(
    data: np.ndarray,
    fs: float = 1.0,
    window: str = 'boxcar',
    axis: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算周期图（未平均的 PSD 估计）。
    
    周期图是最简单的 PSD 估计，但方差较大。
    
    Args:
        data: 输入数据
        fs: 采样频率
        window: 窗函数
        axis: 计算轴
    
    Returns:
        freqs: 频率轴
        psd: 周期图
    """
    freqs, psd = sp_signal.periodogram(
        data, fs=fs,
        window=window,
        axis=axis
    )
    
    return freqs, psd


def bandpower(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    relative: bool = False
) -> float:
    """
    计算指定频带的功率。
    
    Args:
        psd: 功率谱密度
        freqs: 频率轴
        fmin: 频带下限
        fmax: 频带上限
        relative: 是否返回相对功率（占总功率的比例）
    
    Returns:
        频带功率
    
    Example:
        >>> # 计算 alpha 波段 (8-13 Hz) 的功率
        >>> freqs, psd = welch_psd(eeg_signal, fs=256)
        >>> alpha_power = bandpower(psd, freqs, 8, 13, relative=True)
    """
    psd = np.asarray(psd)
    freqs = np.asarray(freqs)
    
    # 找到频带内的索引
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    
    # 频率分辨率
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    
    # 积分（梯形法近似）
    band_power = np.trapz(psd[idx], freqs[idx])
    
    if relative:
        total_power = np.trapz(psd, freqs)
        return band_power / (total_power + 1e-12)
    
    return band_power


def multitaper_psd(
    data: np.ndarray,
    fs: float = 1.0,
    NW: float = 4.0,
    axis: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    多锥度（Multitaper）PSD 估计。
    
    使用多个正交锥度函数来减少估计方差，
    特别适合短数据段的分析。
    
    Args:
        data: 输入数据
        fs: 采样频率
        NW: 时间-带宽积（通常为 2-4）
        axis: 计算轴
    
    Returns:
        freqs: 频率轴
        psd: 多锥度 PSD
    
    Note:
        需要 scipy >= 1.9.0 以使用 dpss 窗
    """
    data = np.asarray(data)
    N = data.shape[axis]
    
    # 计算锥度数
    K = int(2 * NW) - 1
    
    try:
        # 使用 DPSS（离散椭球序列）窗
        tapers = sp_signal.windows.dpss(N, NW, K)
    except AttributeError:
        # 回退到 Welch 方法
        return welch_psd(data, fs=fs, axis=axis)
    
    # 对每个锥度计算周期图并平均
    psd_list = []
    for taper in tapers:
        tapered_data = data * taper
        freqs = np.fft.rfftfreq(N, d=1.0/fs)
        fft_vals = np.fft.rfft(tapered_data, axis=axis)
        psd_list.append(np.abs(fft_vals) ** 2)
    
    # 平均
    psd = np.mean(psd_list, axis=0)
    
    return freqs, psd
