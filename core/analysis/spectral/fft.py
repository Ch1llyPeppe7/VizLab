"""
FFT 相关函数

提供基于 FFT 的频谱分析工具。
"""

import numpy as np
from typing import Tuple, Optional


def fft_power_spectrum(
    signal: np.ndarray,
    fs: float = 1.0,
    axis: int = 0,
    normalized: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算信号的 FFT 功率谱。
    
    Args:
        signal: 输入信号
            - 1D: [N,] → 单信号
            - 2D: [N, C] → N 个样本，C 个通道（沿 axis 做 FFT）
        fs: 采样频率 (Hz)
        axis: FFT 计算轴
        normalized: 是否归一化（使总功率为 1）
    
    Returns:
        freqs: [N//2+1,] 频率轴（仅正频率）
        psd: 功率谱密度，形状取决于输入
    
    Example:
        >>> t = np.linspace(0, 1, 1000)
        >>> signal = np.sin(2*np.pi*50*t)  # 50Hz 正弦波
        >>> freqs, psd = fft_power_spectrum(signal, fs=1000)
        >>> peak_freq = freqs[np.argmax(psd)]
        >>> print(f"Peak frequency: {peak_freq} Hz")  # ≈ 50
    """
    signal = np.asarray(signal)
    N = signal.shape[axis]
    
    # 计算 FFT（仅正频率部分）
    fft_vals = np.fft.rfft(signal, axis=axis)
    
    # 功率谱 = |FFT|²
    psd = np.abs(fft_vals) ** 2
    
    # 频率轴
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    
    # 归一化
    if normalized:
        psd = psd / (np.sum(psd, axis=axis, keepdims=True) + 1e-12)
    
    return freqs, psd


def fft_amplitude(
    signal: np.ndarray,
    fs: float = 1.0,
    axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算信号的 FFT 幅度谱。
    
    Args:
        signal: 输入信号
        fs: 采样频率
        axis: FFT 计算轴
    
    Returns:
        freqs: 频率轴
        amplitude: 幅度谱 |FFT|
    """
    signal = np.asarray(signal)
    N = signal.shape[axis]
    
    fft_vals = np.fft.rfft(signal, axis=axis)
    amplitude = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    
    return freqs, amplitude


def fft_phase(
    signal: np.ndarray,
    fs: float = 1.0,
    axis: int = 0,
    unwrap: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算信号的 FFT 相位谱。
    
    Args:
        signal: 输入信号
        fs: 采样频率
        axis: FFT 计算轴
        unwrap: 是否展开相位（消除 2π 跳变）
    
    Returns:
        freqs: 频率轴
        phase: 相位谱（弧度）
    """
    signal = np.asarray(signal)
    N = signal.shape[axis]
    
    fft_vals = np.fft.rfft(signal, axis=axis)
    phase = np.angle(fft_vals)
    
    if unwrap:
        phase = np.unwrap(phase, axis=axis)
    
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    
    return freqs, phase


def fft_2d(
    data: np.ndarray,
    normalized: bool = True
) -> np.ndarray:
    """
    计算 2D FFT 功率谱。
    
    适用于图像或 2D 数据矩阵。
    
    Args:
        data: [H, W] 2D 数据
        normalized: 是否归一化
    
    Returns:
        [H, W] 2D 功率谱（中心化）
    """
    fft_2d = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft_2d)
    psd = np.abs(fft_shift) ** 2
    
    if normalized:
        psd = psd / (np.sum(psd) + 1e-12)
    
    return psd


def dominant_frequency(
    signal: np.ndarray,
    fs: float = 1.0,
    n_peaks: int = 1
) -> np.ndarray:
    """
    找出信号的主导频率。
    
    Args:
        signal: [N,] 输入信号
        fs: 采样频率
        n_peaks: 返回的峰值数量
    
    Returns:
        [n_peaks,] 主导频率
    """
    freqs, psd = fft_power_spectrum(signal, fs=fs, normalized=False)
    
    # 忽略直流分量
    psd[0] = 0
    
    # 找到最大的 n_peaks 个频率
    peak_indices = np.argsort(psd)[-n_peaks:][::-1]
    
    return freqs[peak_indices]
