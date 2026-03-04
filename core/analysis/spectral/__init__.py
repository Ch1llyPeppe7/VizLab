"""
core.analysis.spectral — 频谱分析工具

提供与领域无关的频谱分析函数，适用于任何信号或时序数据。

核心函数:
    - fft_power_spectrum: FFT 功率谱
    - welch_psd: Welch 功率谱密度估计
    - spectral_entropy: 谱熵
    - stft_analysis: 短时傅里叶变换
    - bandpower: 频带功率

Example:
    >>> import numpy as np
    >>> from core.analysis.spectral import fft_power_spectrum, spectral_entropy
    >>> 
    >>> # 生成测试信号
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*20*t)
    >>> 
    >>> freqs, psd = fft_power_spectrum(signal, fs=1000)
    >>> entropy = spectral_entropy(psd)
"""

from .fft import fft_power_spectrum, fft_phase, fft_amplitude
from .psd import welch_psd, periodogram, bandpower
from .entropy import spectral_entropy, spectral_flatness, spectral_centroid
from .stft import stft_analysis, spectrogram

__all__ = [
    # FFT
    'fft_power_spectrum',
    'fft_phase',
    'fft_amplitude',
    # PSD
    'welch_psd',
    'periodogram',
    'bandpower',
    # 熵与特征
    'spectral_entropy',
    'spectral_flatness',
    'spectral_centroid',
    # STFT
    'stft_analysis',
    'spectrogram',
]
