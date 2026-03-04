#!/usr/bin/env python3
"""
core.analysis 示例脚本

演示如何使用 geometry, spectral, information, manifold 四个模块分析高维数据。

Usage:
    python core/analysis/examples/demo_analysis.py
"""

import numpy as np
import sys
from pathlib import Path

# 确保可以导入 core 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.analysis import geometry, spectral, information, manifold


def create_sinusoidal_pe(positions, dim):
    """生成正弦位置编码"""
    pe = np.zeros((len(positions), dim))
    for k in range(dim // 2):
        omega = 1 / (10000 ** (2 * k / dim))
        pe[:, 2*k] = np.sin(positions * omega)
        pe[:, 2*k+1] = np.cos(positions * omega)
    return pe


def demo_geometry():
    """演示 geometry 模块"""
    print("\n" + "=" * 60)
    print("  📐 Geometry 模块演示")
    print("=" * 60)
    
    # 创建正弦位置编码
    positions = np.arange(256)
    pe = create_sinusoidal_pe(positions, dim=64)
    print(f"\n  输入数据: PE shape = {pe.shape}")
    
    # 计算导数
    d1, d2, d3 = geometry.compute_derivatives(pe, h=1.0)
    print(f"  导数形状: d1={d1.shape}, d2={d2.shape}, d3={d3.shape}")
    
    # 计算曲率
    kappa = geometry.curvature(d1, d2)
    print(f"\n  曲率 κ(p):")
    print(f"    - 平均值: {np.mean(kappa):.6f}")
    print(f"    - 最大值: {np.max(kappa):.6f}")
    print(f"    - 最小值: {np.min(kappa):.6f}")
    
    # 计算挠率
    tau = geometry.torsion(d1, d2, d3)
    print(f"\n  挠率 τ(p):")
    print(f"    - 平均绝对值: {np.mean(np.abs(tau)):.6f}")
    
    # 计算度量张量 (使用已计算的导数)
    g = geometry.metric_tensor(d1)
    print(f"\n  度量张量 g(p) = ||γ'||²:")
    print(f"    - 平均值: {np.mean(g):.4f}")
    
    # 计算弧长 (使用度量张量)
    s = geometry.arc_length(g)
    print(f"\n  弧长 s(p):")
    print(f"    - 总弧长: {s[-1]:.2f}")
    
    return pe


def demo_spectral():
    """演示 spectral 模块"""
    print("\n" + "=" * 60)
    print("  🎵 Spectral 模块演示")
    print("=" * 60)
    
    # 创建测试信号: 50Hz + 120Hz 混合信号
    fs = 1000  # 采样频率
    t = np.linspace(0, 1, fs)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    signal += 0.2 * np.random.randn(len(t))  # 加噪声
    
    print(f"\n  输入信号: 长度 = {len(signal)}, fs = {fs} Hz")
    print(f"  信号内容: 50Hz + 120Hz + 噪声")
    
    # FFT 功率谱
    freqs, psd = spectral.fft_power_spectrum(signal, fs=fs)
    top_freqs = freqs[np.argsort(psd)[-5:]]
    print(f"\n  FFT 功率谱:")
    print(f"    - 频谱形状: {psd.shape}")
    print(f"    - 主要频率 (top 5): {top_freqs}")
    
    # Welch PSD
    freqs_w, psd_w = spectral.welch_psd(signal, fs=fs, nperseg=256)
    print(f"\n  Welch PSD:")
    print(f"    - 频谱形状: {psd_w.shape}")
    
    # 谱熵
    H = spectral.spectral_entropy(psd)
    print(f"\n  谱熵: {H:.4f}")
    print(f"    (低熵 → 周期性强; 高熵 → 宽带/噪声)")
    
    # STFT
    f, t_stft, Sxx = spectral.spectrogram(signal, fs=fs, nperseg=128)
    print(f"\n  STFT 谱图:")
    print(f"    - 形状: {Sxx.shape} (频率 × 时间)")
    
    return signal


def demo_information():
    """演示 information 模块"""
    print("\n" + "=" * 60)
    print("  📊 Information 模块演示")
    print("=" * 60)
    
    # Shannon 熵
    uniform = np.ones(8) / 8
    peaked = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.004, 0.001])
    
    H_uniform = information.shannon_entropy(uniform, base=2)
    H_peaked = information.shannon_entropy(peaked, base=2)
    
    print(f"\n  Shannon 熵:")
    print(f"    - 均匀分布 (8类): {H_uniform:.2f} bits (理论最大 = log2(8) = 3)")
    print(f"    - 尖峰分布: {H_peaked:.2f} bits")
    
    # KL 散度
    kl = information.kl_divergence(peaked, uniform, base=2)
    print(f"\n  KL 散度 D_KL(peaked || uniform): {kl:.2f} bits")
    
    # JS 散度
    js = information.js_divergence(peaked, uniform, base=2)
    print(f"  JS 散度: {js:.2f} bits")
    
    # 互信息
    np.random.seed(42)
    x = np.random.randn(500)
    y_corr = x + 0.3 * np.random.randn(500)  # 相关
    y_indep = np.random.randn(500)            # 独立
    
    mi_corr = information.mutual_information(x, y_corr)
    mi_indep = information.mutual_information(x, y_indep)
    
    print(f"\n  互信息:")
    print(f"    - I(X; Y_corr): {mi_corr:.3f} (相关变量)")
    print(f"    - I(X; Y_indep): {mi_indep:.3f} (独立变量，应接近 0)")
    
    # Fisher 信息
    positions = np.arange(100)
    pe = create_sinusoidal_pe(positions, dim=32)
    fisher = information.fisher_information_matrix(pe, h=1.0)
    
    print(f"\n  Fisher 信息矩阵:")
    print(f"    - 平均值: {np.mean(fisher):.4f}")


def demo_manifold():
    """演示 manifold 模块"""
    print("\n" + "=" * 60)
    print("  🌐 Manifold 模块演示")
    print("=" * 60)
    
    # 创建高维数据
    N, d = 100, 64
    t = np.linspace(0, 4 * np.pi, N)
    
    # 在前3维创建螺旋，其余为噪声
    data = np.random.randn(N, d) * 0.1
    data[:, 0] = np.cos(t)
    data[:, 1] = np.sin(t)
    data[:, 2] = t / (4 * np.pi)
    
    print(f"\n  输入数据: shape = {data.shape}")
    print(f"  数据结构: 前3维为螺旋轨迹，其余为噪声")
    
    # PCA
    proj_3d = manifold.pca_projection(data, n_components=3)
    explained, cumulative = manifold.pca_explained_variance(data, n_components=5)
    
    print(f"\n  PCA 降维:")
    print(f"    - 输出形状: {proj_3d.shape}")
    print(f"    - 前5主成分解释方差: {explained.round(3)}")
    print(f"    - 累积解释方差: {cumulative[-1]:.1%}")
    
    # t-SNE
    proj_2d = manifold.tsne_embed(data, n_components=2, perplexity=15)
    print(f"\n  t-SNE 嵌入:")
    print(f"    - 输出形状: {proj_2d.shape}")
    
    # 轨迹长度
    length_orig = manifold.compute_trajectory_length(data)
    length_pca = manifold.compute_trajectory_length(proj_3d)
    
    print(f"\n  轨迹长度:")
    print(f"    - 原始空间: {length_orig:.2f}")
    print(f"    - PCA 空间: {length_pca:.2f}")


def main():
    print("\n" + "█" * 60)
    print("  core.analysis — 通用数学分析库演示")
    print("█" * 60)
    
    # 1. Geometry
    pe = demo_geometry()
    
    # 2. Spectral
    signal = demo_spectral()
    
    # 3. Information
    demo_information()
    
    # 4. Manifold
    demo_manifold()
    
    print("\n" + "=" * 60)
    print("  ✅ 所有模块演示完成!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
