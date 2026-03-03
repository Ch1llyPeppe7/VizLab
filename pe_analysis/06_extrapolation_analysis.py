#!/usr/bin/env python3
"""
06 — PE 外推性与长度泛化分析

核心思想:
    Transformer 的"训练长度"限制是位置编码的根本问题。
    不同 PE 方案在超出训练长度时的行为截然不同：
    
    1. Sinusoidal PE:
       - 编码值始终有界 (sin/cos ∈ [-1,1])
       - 但超出训练范围的位置可能映射到未学习的频率组合
       - 理论上可外推，实际上注意力模式崩溃
    
    2. RoPE:
       - 同样有界，但旋转角度持续增大
       - 高频子空间先失效（θ_k * p 超出周期）
       - NTK-Aware / YaRN 等方法通过调整基频改善外推
    
    3. ALiBi:
       - 线性外推性最好（偏置只是 -m*|i-j|）
       - 不需要特殊处理即可外推
    
    4. LAPE:
       - 幂律频率 → 低频成分多 → 外推性更好？

Output:
    output/pe_analysis/   → 静态图 (PNG/PDF)
    html/pe_analysis/     → 交互式 HTML

Usage:
    python -m pe_analysis.06_extrapolation_analysis
    python run.py pe_analysis.extrapolation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    PEConfig, get_pe, get_all_pe, SinusoidalPE, RoPE, ALiBi, LAPE,
    setup_plot_style, save_figure, add_math_annotation,
    save_plotly_html, generate_report_html, get_pe_color,
    PE_COLORS, VizLogger,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

MODULE = "pe_analysis"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NTK-Aware RoPE 实现
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ntk_aware_frequencies(dim, base=10000, scale=1.0):
    """
    NTK-Aware RoPE 频率调整。
    
    将基频从 base 调整为 base * scale^{d/(d-2)}:
        θ'_k = 1 / (base' ^ {2k/d})
    
    Args:
        dim: 编码维度
        base: 原始基频
        scale: 外推缩放因子 (e.g., 2.0 表示 2 倍外推)
    Returns:
        调整后的频率序列
    """
    new_base = base * scale ** (dim / (dim - 2))
    return 1.0 / (new_base ** (2 * np.arange(dim // 2) / dim))


def yarn_frequencies(dim, base=10000, scale=1.0, alpha=1.0, beta=32.0):
    """
    YaRN (Yet another RoPE extensioN) 频率调整。
    
    对高频和低频使用不同的缩放策略：
    - 高频 (λ < β): 不缩放
    - 低频 (λ > α * scale): 线性缩放
    - 中间频率: 插值
    """
    orig_freqs = 1.0 / (base ** (2 * np.arange(dim // 2) / dim))
    wavelengths = 2 * np.pi / orig_freqs
    
    adjusted = np.zeros_like(orig_freqs)
    for i, (freq, wl) in enumerate(zip(orig_freqs, wavelengths)):
        if wl < beta:
            # 高频: 不缩放
            adjusted[i] = freq
        elif wl > alpha * scale:
            # 低频: 线性缩放
            adjusted[i] = freq / scale
        else:
            # 中间: 插值
            t = (wl - beta) / (alpha * scale - beta)
            adjusted[i] = freq * (1 - t) + (freq / scale) * t
    
    return adjusted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 编码值在外推时的行为
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_encoding_extrapolation(pe_dict: dict, train_len: int = 128):
    """
    展示各 PE 编码在超出训练长度时的值域变化。
    
    选定几个关键维度, 绘制编码值随位置的变化。
    训练范围 [0, train_len) 用实线, 外推范围用虚线。
    """
    test_len = train_len * 4
    positions = np.arange(test_len)
    dims_to_show = [0, 1, 4, 5]  # 高频和中频维度
    
    setup_plot_style()
    pe_names = list(pe_dict.keys())
    n_pe = len(pe_names)
    n_dims = len(dims_to_show)
    
    fig, axes = plt.subplots(n_pe, n_dims, figsize=(4 * n_dims, 3.5 * n_pe))
    fig.suptitle(f"PE Encoding Values: Train [0,{train_len}) vs Extrapolation [{train_len},{test_len})\n"
                 "Solid = train range | Dashed = extrapolation",
                 fontsize=14, fontweight='bold', y=1.02)
    
    for row, name in enumerate(pe_names):
        pe = pe_dict[name]
        # 使用更大的 max_len 生成编码
        config_ext = PEConfig(dim=pe.dim, max_len=test_len)
        pe_ext = type(pe)(config=config_ext)
        enc = pe_ext.encode(positions)
        
        for col, d in enumerate(dims_to_show):
            ax = axes[row, col]
            color = get_pe_color(name)
            
            # 训练范围
            ax.plot(positions[:train_len], enc[:train_len, d],
                    color=color, linewidth=1.5, alpha=0.9)
            # 外推范围
            ax.plot(positions[train_len:], enc[train_len:, d],
                    color=color, linewidth=1.5, ls='--', alpha=0.6)
            # 分界线
            ax.axvline(x=train_len, color='red', ls=':', alpha=0.5)
            
            if row == 0:
                ax.set_title(f"dim {d}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{name}", fontsize=10, fontweight='bold')
            if row == n_pe - 1:
                ax.set_xlabel("Position")
    
    plt.tight_layout()
    save_figure(fig, "06_encoding_extrapolation", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. 核函数外推行为
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_kernel_extrapolation(pe_dict: dict, train_len: int = 128):
    """
    展示各 PE 的核函数 K(delta) 在超出训练范围时的衰减行为。
    
    Panel 1: K(delta) 在 [0, 4*train_len] 范围内的全貌
    Panel 2: 对数尺度查看远距离衰减
    Panel 3: 相对于训练范围内的核值, 外推区域的核值比率
    """
    test_len = train_len * 4
    deltas = np.arange(0, test_len)
    
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    for name, pe in pe_dict.items():
        color = get_pe_color(name)
        kernels = np.array([pe.kernel(d) for d in deltas])
        
        # Panel 1: 核函数全貌
        axes[0].plot(deltas[:train_len], kernels[:train_len],
                     color=color, linewidth=1.5, alpha=0.9, label=name)
        axes[0].plot(deltas[train_len:], kernels[train_len:],
                     color=color, linewidth=1.5, ls='--', alpha=0.5)
    
    axes[0].axvline(x=train_len, color='red', ls=':', alpha=0.5, label='Train boundary')
    axes[0].set_xlabel("Position difference $\\Delta$")
    axes[0].set_ylabel("$K(\\Delta)$")
    axes[0].set_title("Kernel Function: Train vs Extrapolation",
                      fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=8)
    
    # Panel 2: |K(delta)| 的包络线 (对数)
    for name, pe in pe_dict.items():
        color = get_pe_color(name)
        kernels = np.array([pe.kernel(d) for d in deltas])
        env = np.maximum(np.abs(kernels), 1e-15)
        axes[1].semilogy(deltas, env, color=color, linewidth=1.5,
                         alpha=0.7, label=name)
    
    axes[1].axvline(x=train_len, color='red', ls=':', alpha=0.5)
    axes[1].set_xlabel("$\\Delta$")
    axes[1].set_ylabel("$|K(\\Delta)|$")
    axes[1].set_title("Kernel Envelope (log scale)", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=8)
    
    # Panel 3: 核值在外推区域的变异系数
    for name, pe in pe_dict.items():
        color = get_pe_color(name)
        kernels = np.array([pe.kernel(d) for d in deltas])
        # 滑动窗口标准差
        window = 16
        stds = []
        centers = []
        for start in range(0, len(kernels) - window, window // 2):
            stds.append(np.std(kernels[start:start + window]))
            centers.append(start + window // 2)
        axes[2].plot(centers, stds, color=color, linewidth=1.5,
                     label=name, alpha=0.8)
    
    axes[2].axvline(x=train_len, color='red', ls=':', alpha=0.5)
    axes[2].set_xlabel("Position $\\Delta$")
    axes[2].set_ylabel("Local Std (window=16)")
    axes[2].set_title("Kernel Variability", fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, "06_kernel_extrapolation", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. 注意力窗口在外推时的变化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_attention_extrapolation(pe_dict: dict, train_len: int = 128):
    """
    模拟 attention 在外推序列上的行为。
    
    Panel 1: 纯位置注意力在不同序列长度下的热力图
    Panel 2: 有效窗口随序列长度的变化
    """
    lengths = [train_len, train_len * 2, train_len * 4]
    
    pe_names = list(pe_dict.keys())
    n_pe = len(pe_names)
    n_lens = len(lengths)
    
    setup_plot_style()
    fig, axes = plt.subplots(n_pe, n_lens, figsize=(5 * n_lens, 4 * n_pe))
    fig.suptitle("Attention Pattern at Different Sequence Lengths\n"
                 f"Train length = {train_len}",
                 fontsize=16, fontweight='bold', y=1.02)
    
    dim = pe_dict[pe_names[0]].dim
    
    for row, name in enumerate(pe_names):
        pe = pe_dict[name]
        
        for col, L in enumerate(lengths):
            ax = axes[row, col]
            positions = np.arange(L)
            
            # 构造位置偏置
            config_ext = PEConfig(dim=dim, max_len=L)
            pe_ext = type(pe)(config=config_ext)
            
            if name in ['sinusoidal', 'lape']:
                enc = pe_ext.encode(positions)
                bias = (enc @ enc.T) / np.sqrt(dim)
            elif name == 'rope':
                # 向量化: kernel_matrix 内部用 np.cos 广播
                bias = pe_ext.kernel_matrix(positions) / np.sqrt(dim)
            elif name == 'alibi':
                bias = -pe_ext.slopes[0] * np.abs(
                    positions[:, None] - positions[None, :]).astype(float)
            else:
                bias = np.zeros((L, L))
            
            attn = softmax(bias, axis=-1)
            
            # 只显示前 train_len x train_len 的区域 + 外推区域的边界
            n_show = min(L, train_len * 2)
            im = ax.imshow(attn[:n_show, :n_show], cmap='hot', aspect='equal',
                          interpolation='nearest', vmin=0)
            
            if L > train_len:
                ax.axvline(x=train_len - 0.5, color='cyan', ls='--', linewidth=1.5)
                ax.axhline(y=train_len - 0.5, color='cyan', ls='--', linewidth=1.5)
            
            if col == 0:
                ax.set_ylabel(f"{name}", fontsize=11, fontweight='bold')
            if row == 0:
                ax.set_title(f"L = {L}" + 
                             (" (train)" if L == train_len else f" ({L//train_len}x)"),
                             fontsize=11)
    
    plt.tight_layout()
    save_figure(fig, "06_attention_extrapolation", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. NTK-Aware / YaRN 外推技术对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_rope_extensions(train_len: int = 128, dim: int = 64):
    """
    对比 RoPE 的外推增强技术:
    - Original RoPE
    - NTK-Aware (缩放基频)
    - YaRN (混合频率缩放)
    - Linear Scaling (直接缩放位置)
    """
    test_len = train_len * 4
    positions = np.arange(test_len)
    scale = test_len / train_len  # 4x
    
    # 频率计算
    orig_freqs = 1.0 / (10000 ** (2 * np.arange(dim // 2) / dim))
    ntk_freqs = ntk_aware_frequencies(dim, scale=scale)
    yarn_freqs = yarn_frequencies(dim, scale=scale)
    linear_freqs = orig_freqs  # 线性缩放不改变频率, 只缩放位置
    
    methods = {
        'Original RoPE': orig_freqs,
        'NTK-Aware': ntk_freqs,
        'YaRN': yarn_freqs,
        'Linear Scaling': linear_freqs,
    }
    method_colors = {
        'Original RoPE': '#E74C3C',
        'NTK-Aware': '#3498DB',
        'YaRN': '#2ECC71',
        'Linear Scaling': '#9B59B6',
    }
    
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"RoPE Extension Techniques (scale={scale:.0f}x)\n"
                 f"Train length={train_len}, Test length={test_len}",
                 fontsize=16, fontweight='bold')
    
    # Panel 1: 频率谱对比
    ax1 = axes[0, 0]
    for method_name, freqs in methods.items():
        color = method_colors[method_name]
        ax1.semilogy(range(len(freqs)), freqs, 'o-', color=color,
                     linewidth=1.5, markersize=3, label=method_name)
    ax1.set_xlabel("Frequency index $k$")
    ax1.set_ylabel("$\\theta_k$")
    ax1.set_title("Frequency Spectrum Comparison", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    
    # Panel 2: 核函数对比
    ax2 = axes[0, 1]
    deltas = np.arange(0, test_len)
    for method_name, freqs in methods.items():
        color = method_colors[method_name]
        if method_name == 'Linear Scaling':
            # 线性缩放: 位置 / scale
            kernels = [np.mean(np.cos(freqs * d / scale)) for d in deltas]
        else:
            kernels = [np.mean(np.cos(freqs * d)) for d in deltas]
        ax2.plot(deltas, kernels, color=color, linewidth=1.5,
                 alpha=0.7, label=method_name)
    ax2.axvline(x=train_len, color='red', ls=':', alpha=0.5, label='Train boundary')
    ax2.set_xlabel("$\\Delta$")
    ax2.set_ylabel("$K(\\Delta)$")
    ax2.set_title("Kernel Function Comparison", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7)
    
    # Panel 3: 注意力热力图 (各方法在外推长度下)
    ax3 = axes[1, 0]
    L = train_len * 2
    pos = np.arange(L)
    attn_methods = {}
    for method_name, freqs in methods.items():
        # 向量化: 利用 Toeplitz 结构, delta = j - i
        delta = pos[:, None] - pos[None, :]  # [L, L]
        if method_name == 'Linear Scaling':
            # phases[k, i, j] = freqs[k] * delta[i,j] / scale
            phases = freqs[:, None, None] * delta[None, :, :] / scale  # [m, L, L]
        else:
            phases = freqs[:, None, None] * delta[None, :, :]  # [m, L, L]
        bias = np.mean(np.cos(phases), axis=0) / np.sqrt(dim)  # [L, L]
        attn = softmax(bias, axis=-1)
        attn_methods[method_name] = attn
    
    # 绘制中心行的注意力分布 (query at center)
    center = L // 2
    for method_name, attn in attn_methods.items():
        color = method_colors[method_name]
        ax3.plot(pos, attn[center], color=color, linewidth=1.5,
                 alpha=0.8, label=method_name)
    ax3.axvline(x=train_len, color='red', ls=':', alpha=0.3)
    ax3.set_xlabel("Key position")
    ax3.set_ylabel("Attention weight")
    ax3.set_title(f"Attention from query={center} (L={L})", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7)
    
    # Panel 4: 注意力熵对比
    ax4 = axes[1, 1]
    for method_name, attn in attn_methods.items():
        color = method_colors[method_name]
        entropies = []
        for i in range(L):
            p = attn[i]
            p = p[p > 0]
            entropies.append(-np.sum(p * np.log2(p)))
        ax4.plot(pos, entropies, color=color, linewidth=1.5,
                 alpha=0.8, label=method_name)
    ax4.axvline(x=train_len, color='red', ls=':', alpha=0.3)
    ax4.set_xlabel("Query position")
    ax4.set_ylabel("Entropy (bits)")
    ax4.set_title("Attention Entropy Comparison", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=7)
    
    plt.tight_layout()
    save_figure(fig, "06_rope_extensions", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. 数值稳定性分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_numerical_stability(pe_dict: dict, train_len: int = 128):
    """
    分析 PE 编码在极长位置时的数值稳定性。
    
    Panel 1: 编码向量范数随位置的变化
    Panel 2: 编码值的方差随位置的变化
    Panel 3: 相邻位置编码差异的范数
    """
    test_len = train_len * 8
    positions = np.arange(test_len)
    dim = pe_dict[list(pe_dict.keys())[0]].dim
    
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    for name, pe in pe_dict.items():
        color = get_pe_color(name)
        config_ext = PEConfig(dim=dim, max_len=test_len)
        pe_ext = type(pe)(config=config_ext)
        enc = pe_ext.encode(positions)
        
        # Panel 1: 范数
        norms = np.linalg.norm(enc, axis=1)
        axes[0].plot(positions, norms, color=color, linewidth=1.5,
                     label=name, alpha=0.8)
        
        # Panel 2: 方差 (逐位置的维度方差)
        variances = np.var(enc, axis=1)
        axes[1].plot(positions, variances, color=color, linewidth=1.5,
                     label=name, alpha=0.8)
        
        # Panel 3: 相邻位置差异
        diffs = np.linalg.norm(np.diff(enc, axis=0), axis=1)
        axes[2].plot(positions[1:], diffs, color=color, linewidth=1.5,
                     label=name, alpha=0.8)
    
    for ax in axes:
        ax.axvline(x=train_len, color='red', ls=':', alpha=0.5, label='Train boundary')
    
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("$\\|PE(p)\\|_2$")
    axes[0].set_title("Encoding Vector Norm", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=7)
    
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Var across dimensions")
    axes[1].set_title("Per-Position Variance", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=7)
    
    axes[2].set_xlabel("Position")
    axes[2].set_ylabel("$\\|PE(p+1) - PE(p)\\|_2$")
    axes[2].set_title("Adjacent Position Difference", fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=7)
    
    plt.tight_layout()
    save_figure(fig, "06_numerical_stability", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. 综合仪表盘
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_extrapolation_dashboard(pe_dict: dict, train_len: int = 128):
    """综合外推性评估仪表盘"""
    names = list(pe_dict.keys())
    colors = [get_pe_color(n) for n in names]
    dim = pe_dict[names[0]].dim
    
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"PE Extrapolation — Dashboard (train={train_len})",
                 fontsize=18, fontweight='bold')
    
    x = np.arange(len(names))
    test_ratios = [1, 2, 4, 8]
    
    # Panel 1: 核衰减速率 (在 2x 范围)
    ax1 = axes[0, 0]
    decay_rates = []
    for name, pe in pe_dict.items():
        k_train = abs(pe.kernel(train_len))
        k_test = abs(pe.kernel(train_len * 2))
        if k_train > 1e-15:
            decay_rates.append(k_test / k_train)
        else:
            decay_rates.append(0.0)
    bars = ax1.bar(x, decay_rates, 0.6, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("|K(2L)| / |K(L)|")
    ax1.set_title("Kernel Decay at 2x Extrapolation", fontsize=12)
    for bar, val in zip(bars, decay_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel 2: 编码范数稳定性 (std/mean over positions)
    ax2 = axes[0, 1]
    stability = []
    for name, pe in pe_dict.items():
        config_ext = PEConfig(dim=dim, max_len=train_len * 4)
        pe_ext = type(pe)(config=config_ext)
        enc = pe_ext.encode(np.arange(train_len * 4))
        norms = np.linalg.norm(enc, axis=1)
        if np.mean(norms) > 1e-15:
            stability.append(np.std(norms) / np.mean(norms))
        else:
            stability.append(0.0)
    bars = ax2.bar(x, stability, 0.6, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("CoV (Norm)")
    ax2.set_title("Encoding Norm Stability\n(low = stable)", fontsize=12)
    for bar, val in zip(bars, stability):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
    
    # Panel 3: 有效注意力范围 (在不同长度下)
    ax3 = axes[1, 0]
    bar_w = 0.2
    for li, ratio in enumerate(test_ratios):
        L = train_len * ratio
        eff_windows = []
        for name, pe in pe_dict.items():
            pos = np.arange(L)
            config_ext = PEConfig(dim=dim, max_len=L)
            pe_ext = type(pe)(config=config_ext)
            if name in ['sinusoidal', 'lape']:
                enc = pe_ext.encode(pos)
                bias = (enc @ enc.T) / np.sqrt(dim)
            elif name == 'rope':
                # 向量化: kernel_matrix 内部用 np.cos 广播
                bias = pe_ext.kernel_matrix(pos) / np.sqrt(dim)
            elif name == 'alibi':
                bias = -pe_ext.slopes[0] * np.abs(
                    pos[:, None] - pos[None, :]).astype(float)
            else:
                bias = np.zeros((L, L))
            attn = softmax(bias, axis=-1)
            # 中间位置的有效窗口
            center = L // 2
            cum = 0.0
            for w in range(L):
                left = max(0, center - w)
                right = min(L, center + w + 1)
                cum = np.sum(attn[center, left:right])
                if cum >= 0.5:
                    eff_windows.append(2 * w + 1)
                    break
            else:
                eff_windows.append(L)
        
        ax3.bar(x + li * bar_w, eff_windows, bar_w,
                color=[c for c in colors], alpha=0.3 + li * 0.2,
                edgecolor='black', linewidth=0.5,
                label=f'{ratio}x' if li == 0 else None)
    
    ax3.set_xticks(x + bar_w * 1.5)
    ax3.set_xticklabels(names)
    ax3.set_ylabel("Window Size (50%)")
    ax3.set_title("Effective Window at Different Lengths\n"
                  f"(1x, 2x, 4x, 8x of L={train_len})", fontsize=12)
    
    # Panel 4: 外推性评分 (综合)
    ax4 = axes[1, 1]
    scores = []
    for i, name in enumerate(names):
        # 综合评分: 低范数变异 + 高核衰减稳定 + 合理窗口
        norm_score = 1.0 / (1.0 + stability[i] * 100)
        decay_score = min(1.0, decay_rates[i])
        score = (norm_score + decay_score) / 2
        scores.append(score)
    bars = ax4.bar(x, scores, 0.6, color=colors, edgecolor='black', alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.set_ylabel("Composite Score")
    ax4.set_title("Extrapolation Capability Score", fontsize=12)
    ax4.set_ylim(0, 1.1)
    for bar, val in zip(bars, scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.92)
    # 直接保存 (避免 bbox_inches='tight' 在某些 matplotlib 版本下渲染崩溃)
    from core.plot_utils import get_output_dir
    output_dir = get_output_dir(MODULE)
    for fmt in ['png', 'pdf']:
        fp = output_dir / f"06_extrapolation_dashboard.{fmt}"
        fig.savefig(fp, dpi=200, facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {fp}")
    plt.close(fig)
    
    if HAS_PLOTLY:
        pfig = make_subplots(rows=2, cols=2,
            subplot_titles=["Kernel Decay", "Norm Stability",
                           "Window Size", "Composite Score"])
        for i, n in enumerate(names):
            color = get_pe_color(n)
            pfig.add_trace(go.Bar(x=[n], y=[decay_rates[i]], marker_color=color,
                                  name=n, legendgroup=n, showlegend=True),
                          row=1, col=1)
            pfig.add_trace(go.Bar(x=[n], y=[stability[i]], marker_color=color,
                                  name=n, legendgroup=n, showlegend=False),
                          row=1, col=2)
            pfig.add_trace(go.Bar(x=[n], y=[scores[i]], marker_color=color,
                                  name=n, legendgroup=n, showlegend=False),
                          row=2, col=2)
        pfig.update_layout(template="plotly_white", width=1000, height=700,
                          title=f"PE Extrapolation Dashboard (train_len={train_len})")
        save_plotly_html(pfig, "06_extrapolation_dashboard.html", MODULE)
    
    return {name: {'decay': d, 'stability': s, 'score': sc}
            for name, d, s, sc in zip(names, decay_rates, stability, scores)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HTML 报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_extrapolation_report():
    sections = [
        {
            'title': '核心问题',
            'content': (
                'Transformer 的"训练长度"限制源自位置编码。'
                '不同 PE 方案在超出训练长度时行为迥异：'
                '<ul>'
                '<li><strong>Sinusoidal</strong>: 有界但可能映射到未训练区域</li>'
                '<li><strong>RoPE</strong>: 有界但高频失效; NTK/YaRN 可改善</li>'
                '<li><strong>ALiBi</strong>: 线性外推, 最自然</li>'
                '<li><strong>LAPE</strong>: 幂律频率衰减可能有利于外推</li>'
                '</ul>'
            )
        },
        {
            'title': '1. 编码值外推',
            'content': '展示各 PE 编码在超出训练范围时的值域变化。'
        },
        {
            'title': '2. 核函数外推',
            'content': '核函数 \\(K(\\Delta)\\) 在远距离的衰减行为决定外推能力。'
        },
        {
            'title': '3. 注意力外推',
            'content': '模拟不同序列长度下的 attention pattern 变化。'
        },
        {
            'title': '4. RoPE 外推技术',
            'content': (
                '对比 NTK-Aware、YaRN、Linear Scaling 等 RoPE 外推增强方法：'
                '<br>NTK-Aware: \\(\\text{base}\' = \\text{base} \\cdot s^{d/(d-2)}\\)'
                '<br>YaRN: 混合高低频缩放策略'
                '<br>Linear: 位置 \\(p\' = p / s\\)'
            )
        },
        {
            'title': '5. 数值稳定性',
            'content': '分析编码向量范数、方差、相邻差异在极长位置时的稳定性。'
        },
    ]
    generate_report_html(
        title="06 — PE Extrapolation & Length Generalization",
        sections=sections,
        module=MODULE,
        filename="06_extrapolation_report.html"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  06 — PE Extrapolation & Length Generalization")
    print("=" * 60)
    
    dim = 64
    train_len = 128
    
    config = PEConfig(dim=dim, max_len=train_len)
    pe_dict = get_all_pe(config=config)
    logger = VizLogger("pe_analysis", "06_extrapolation_analysis")
    
    # 1. 编码值外推
    print("\n  [1/6] Encoding extrapolation...")
    plot_encoding_extrapolation(pe_dict, train_len)
    print("        Saved 06_encoding_extrapolation")
    logger.log_metric("encoding_extrapolation", "completed")
    
    # 2. 核函数外推
    print("\n  [2/6] Kernel extrapolation...")
    plot_kernel_extrapolation(pe_dict, train_len)
    print("        Saved 06_kernel_extrapolation")
    logger.log_metric("kernel_extrapolation", "completed")
    
    # 3. 注意力外推
    print("\n  [3/6] Attention extrapolation...")
    plot_attention_extrapolation(pe_dict, train_len)
    print("        Saved 06_attention_extrapolation")
    logger.log_metric("attention_extrapolation", "completed")
    
    # 4. RoPE 外推技术
    print("\n  [4/6] RoPE extension techniques...")
    plot_rope_extensions(train_len, dim)
    print("        Saved 06_rope_extensions")
    logger.log_metric("rope_extensions", "completed")
    
    # 5. 数值稳定性
    print("\n  [5/6] Numerical stability...")
    plot_numerical_stability(pe_dict, train_len)
    print("        Saved 06_numerical_stability")
    logger.log_metric("numerical_stability", "completed")
    
    # 6. 综合仪表盘
    print("\n  [6/6] Extrapolation dashboard...")
    results = plot_extrapolation_dashboard(pe_dict, train_len)
    print("        Saved 06_extrapolation_dashboard")
    for name, r in results.items():
        logger.log_metric(f"score_{name}", r['score'])
        print(f"        {name}: score={r['score']:.3f}")
    
    # 报告
    print("\n  Generating report...")
    generate_extrapolation_report()
    
    logger.save()
    
    print("\n" + "=" * 60)
    print("  06_extrapolation_analysis complete!")
    print("  Static:      output/pe_analysis/06_*")
    print("  Interactive:  html/pe_analysis/06_*")
    print("=" * 60)


if __name__ == "__main__":
    main()
