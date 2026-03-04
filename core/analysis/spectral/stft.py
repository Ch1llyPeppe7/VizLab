"""
短时傅里叶变换 (STFT)

提供时频分析工具。
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal as sp_signal


def stft_analysis(
    data: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算短时傅里叶变换 (STFT)。
    
    STFT 将信号分解为时间-频率表示，
    适合分析非平稳信号。
    
    Args:
        data: [N,] 输入信号
        fs: 采样频率
        nperseg: 每段长度
        noverlap: 重叠长度（默认 nperseg//2）
        window: 窗函数
    
    Returns:
        freqs: [n_freqs,] 频率轴
        times: [n_times,] 时间轴
        Zxx: [n_freqs, n_times] STFT 系数（复数）
    
    Example:
        >>> # 分析啁啾信号
        >>> t = np.linspace(0, 10, 10000)
        >>> chirp = np.sin(2*np.pi * (10 + t) * t)
        >>> freqs, times, Zxx = stft_analysis(chirp, fs=1000)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    freqs, times, Zxx = sp_signal.stft(
        data, fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window
    )
    
    return freqs, times, Zxx


def spectrogram(
    data: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    log_scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算功率谱图（Spectrogram）。
    
    功率谱图 = |STFT|²，展示能量在时间-频率平面上的分布。
    
    Args:
        data: [N,] 输入信号
        fs: 采样频率
        nperseg: 每段长度
        noverlap: 重叠长度
        window: 窗函数
        log_scale: 是否返回对数功率 (dB)
    
    Returns:
        freqs: 频率轴
        times: 时间轴
        Sxx: [n_freqs, n_times] 功率谱图
    
    Example:
        >>> freqs, times, Sxx = spectrogram(signal, fs=1000)
        >>> plt.pcolormesh(times, freqs, Sxx)
        >>> plt.ylabel('Frequency [Hz]')
        >>> plt.xlabel('Time [sec]')
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    freqs, times, Sxx = sp_signal.spectrogram(
        data, fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window
    )
    
    if log_scale:
        Sxx = 10 * np.log10(Sxx + 1e-12)
    
    return freqs, times, Sxx


def istft(
    Zxx: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    逆短时傅里叶变换 (ISTFT)。
    
    从 STFT 系数重建原始信号。
    
    Args:
        Zxx: [n_freqs, n_times] STFT 系数
        fs: 采样频率
        nperseg: 每段长度
        noverlap: 重叠长度
        window: 窗函数
    
    Returns:
        times: 时间轴
        signal: 重建信号
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    times, signal = sp_signal.istft(
        Zxx, fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window
    )
    
    return times, signal


def instantaneous_frequency(
    Zxx: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray
) -> np.ndarray:
    """
    从 STFT 计算瞬时频率。
    
    在每个时间点找到最大能量对应的频率。
    
    Args:
        Zxx: STFT 系数
        times: 时间轴
        freqs: 频率轴
    
    Returns:
        [n_times,] 瞬时频率
    """
    # 功率谱
    power = np.abs(Zxx) ** 2
    
    # 每个时间点的最大频率索引
    max_freq_idx = np.argmax(power, axis=0)
    
    return freqs[max_freq_idx]


def time_frequency_ridges(
    Sxx: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    n_ridges: int = 1
) -> list:
    """
    提取时频脊线（能量最大的轨迹）。
    
    Args:
        Sxx: 功率谱图
        freqs: 频率轴
        times: 时间轴
        n_ridges: 提取的脊线数量
    
    Returns:
        ridges: list of (times, frequencies) 元组
    """
    ridges = []
    Sxx_work = Sxx.copy()
    
    for _ in range(n_ridges):
        # 每个时间点的最大频率
        max_idx = np.argmax(Sxx_work, axis=0)
        ridge_freqs = freqs[max_idx]
        
        ridges.append((times.copy(), ridge_freqs.copy()))
        
        # 抑制已提取的脊线
        for t_idx, f_idx in enumerate(max_idx):
            # 在该频率附近抑制
            f_low = max(0, f_idx - 3)
            f_high = min(len(freqs), f_idx + 4)
            Sxx_work[f_low:f_high, t_idx] = 0
    
    return ridges
