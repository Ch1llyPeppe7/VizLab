#!/usr/bin/env python3
"""
05 — PE 对 Attention Pattern 的影响

核心思想:
    位置编码通过不同机制影响 attention score 矩阵 A(i,j):
    
    1. Sinusoidal PE (加性):
       A(i,j) = (q_i + PE_i)^T (k_j + PE_j) / sqrt(d)
              = q_i^T k_j / sqrt(d)                    # 内容注意力
              + q_i^T PE_j / sqrt(d)                    # 查询-位置交叉
              + PE_i^T k_j / sqrt(d)                    # 位置-键交叉
              + PE_i^T PE_j / sqrt(d)                   # 纯位置偏置 ← 这项是关键
    
    2. RoPE (乘性):
       A(i,j) = (R(i)q_i)^T (R(j)k_j) / sqrt(d)
              = q_i^T R(j-i)^T k_j / sqrt(d)           # 只依赖相对位置差!
       纯位置项: K(j-i) = (1/m) sum_k cos(theta_k * (j-i))
    
    3. ALiBi (偏置):
       A(i,j) = q_i^T k_j / sqrt(d) - m * |i-j|       # 线性距离惩罚
    
    4. LAPE:
       类似 Sinusoidal, 但频率衰减更快, 导致更尖锐的局部偏置

数学分析:
    - Attention 热力图: 展示纯位置偏置矩阵
    - 局部性分析: 有效注意力窗口大小 (FWHM)
    - 因果性: 各 PE 对单向/双向注意力的影响
    - 多头注意力: 不同头的位置偏置差异

Output:
    output/pe_analysis/   → 静态图 (PNG/PDF)
    html/pe_analysis/     → 交互式 HTML

Usage:
    python -m pe_analysis.05_attention_patterns
    python run.py pe_analysis.attention
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
#  Attention Score 构造
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_pe_attention_bias(pe_name: str, pe_instance, positions: np.ndarray,
                               dim: int, n_heads: int = 8,
                               rng: np.random.Generator = None):
    """
    计算各 PE 方案引入的纯位置注意力偏置矩阵。
    
    返回:
        bias_matrix: [N, N] 纯位置偏置（softmax 前）
        attn_matrix: [N, N] 经 softmax 后的注意力分布（无内容项）
        multi_head_bias: [n_heads, N, N] 多头偏置（仅对部分 PE 有意义）
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(positions)
    
    if pe_name in ['sinusoidal', 'lape']:
        # 加性 PE: 纯位置偏置 = PE_i^T PE_j / sqrt(d)
        enc = pe_instance.encode(positions)  # [N, d]
        bias = (enc @ enc.T) / np.sqrt(dim)  # [N, N]
        
        # 多头: 每个头看 d/n_heads 个维度
        head_dim = dim // n_heads
        multi_head = np.zeros((n_heads, N, N))
        for h in range(n_heads):
            start = h * head_dim
            end = start + head_dim
            enc_h = enc[:, start:end]
            multi_head[h] = (enc_h @ enc_h.T) / np.sqrt(head_dim)
        
    elif pe_name == 'rope':
        # RoPE: 位置偏置 = K(j-i) = (1/m) sum_k cos(theta_k * (j-i))
        bias = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                bias[i, j] = pe_instance.kernel(positions[j] - positions[i])
        bias /= np.sqrt(dim)
        
        # 多头: 每个头使用不同的频率子集
        freqs = pe_instance.get_frequencies()
        head_dim = dim // n_heads
        n_freq_per_head = head_dim // 2
        multi_head = np.zeros((n_heads, N, N))
        for h in range(n_heads):
            f_start = h * n_freq_per_head
            f_end = min(f_start + n_freq_per_head, len(freqs))
            head_freqs = freqs[f_start:f_end]
            for i in range(N):
                for j in range(N):
                    delta = positions[j] - positions[i]
                    multi_head[h, i, j] = np.mean(np.cos(head_freqs * delta))
            multi_head[h] /= np.sqrt(head_dim)
        
    elif pe_name == 'alibi':
        # ALiBi: -m * |i-j|
        slopes = pe_instance.slopes
        n_heads = len(slopes)
        # 默认取第一个 head 的 slope
        m = slopes[0]
        bias = -m * np.abs(positions[:, None] - positions[None, :]).astype(float)
        
        # 多头: 每个头有不同的 slope
        multi_head = np.zeros((n_heads, N, N))
        for h in range(n_heads):
            multi_head[h] = -slopes[h] * np.abs(
                positions[:, None] - positions[None, :]).astype(float)
    
    else:
        bias = np.zeros((N, N))
        multi_head = np.zeros((n_heads, N, N))
    
    # Softmax (逐行)
    attn = softmax(bias, axis=-1)
    
    return bias, attn, multi_head


def compute_full_attention(pe_name: str, pe_instance, positions: np.ndarray,
                            dim: int, rng: np.random.Generator = None):
    """
    计算完整 attention score（含内容项 + 位置项）。
    
    A(i,j) = content_score(i,j) + position_bias(i,j)
    
    内容项使用随机 q/k 模拟 (模拟预训练后的状态)。
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(positions)
    
    # 随机 q/k (模拟内容)
    Q = rng.standard_normal((N, dim)) * 0.1
    K = rng.standard_normal((N, dim)) * 0.1
    
    content_score = (Q @ K.T) / np.sqrt(dim)  # [N, N]
    
    # 加上位置偏置
    bias, _, _ = compute_pe_attention_bias(pe_name, pe_instance, positions, dim)
    
    full_score = content_score + bias
    full_attn = softmax(full_score, axis=-1)
    
    return content_score, bias, full_attn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_effective_window(attn_row: np.ndarray, center: int,
                             threshold: float = 0.5):
    """
    计算有效注意力窗口大小 (基于累积概率)。
    
    从中心位置出发, 找到包含 threshold 概率质量的最小窗口。
    
    Args:
        attn_row: [N,] 一行注意力分布 (归一化)
        center: 中心位置
        threshold: 阈值
    Returns:
        window_size: 有效窗口大小 (对称)
    """
    N = len(attn_row)
    cumulative = 0.0
    for w in range(N):
        left = max(0, center - w)
        right = min(N, center + w + 1)
        cumulative = np.sum(attn_row[left:right])
        if cumulative >= threshold:
            return 2 * w + 1
    return N


def compute_attention_entropy(attn_row: np.ndarray):
    """
    注意力分布的熵 (bits)。
    
    高熵 → 均匀注意力 (global)
    低熵 → 集中注意力 (local)
    """
    p = attn_row[attn_row > 0]
    return -np.sum(p * np.log2(p))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 纯位置偏置热力图
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_position_bias_heatmaps(pe_dict: dict, positions: np.ndarray,
                                 dim: int):
    """
    可视化各 PE 方案的纯位置偏置矩阵。
    
    上排: 原始偏置 (softmax 前)
    下排: softmax 后的注意力分布
    """
    pe_names = list(pe_dict.keys())
    n_pe = len(pe_names)
    
    setup_plot_style()
    fig, axes = plt.subplots(2, n_pe, figsize=(5 * n_pe, 9))
    fig.suptitle("Pure Position Bias in Attention Score\n"
                 "Top: raw bias | Bottom: after softmax",
                 fontsize=16, fontweight='bold', y=1.02)
    
    for idx, name in enumerate(pe_names):
        pe = pe_dict[name]
        bias, attn, _ = compute_pe_attention_bias(name, pe, positions, dim)
        
        # 上排: 原始偏置
        ax1 = axes[0, idx]
        im1 = ax1.imshow(bias, cmap='RdBu_r', aspect='equal',
                         interpolation='nearest')
        ax1.set_title(f"{name}\n(raw bias)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Key position $j$")
        if idx == 0:
            ax1.set_ylabel("Query position $i$")
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 下排: softmax 后
        ax2 = axes[1, idx]
        im2 = ax2.imshow(attn, cmap='hot', aspect='equal',
                         interpolation='nearest', vmin=0)
        ax2.set_title(f"{name}\n(softmax)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Key position $j$")
        if idx == 0:
            ax2.set_ylabel("Query position $i$")
        plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    save_figure(fig, "05_position_bias_heatmaps", MODULE)
    plt.close(fig)
    
    # Plotly 交互式
    if HAS_PLOTLY:
        pfig = make_subplots(rows=1, cols=n_pe,
                             subplot_titles=[f"{n} — Position Bias" for n in pe_names])
        for idx, name in enumerate(pe_names):
            pe = pe_dict[name]
            bias, _, _ = compute_pe_attention_bias(name, pe, positions, dim)
            pfig.add_trace(go.Heatmap(
                z=bias, colorscale='RdBu_r', name=name,
                showscale=(idx == n_pe - 1),
            ), row=1, col=idx + 1)
        pfig.update_layout(title="Pure Position Bias (Interactive)",
                          template="plotly_white",
                          width=350 * n_pe, height=400)
        save_plotly_html(pfig, "05_position_bias.html", MODULE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. 位置偏置一维剖面 (选定 query 位置)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_bias_profiles(pe_dict: dict, positions: np.ndarray, dim: int):
    """
    从选定 query 位置出发, 绘制对所有 key 位置的偏置曲线。
    
    Panel 1: 原始偏置 (对所有 key)
    Panel 2: softmax 后的注意力分布
    Panel 3: 偏置差 (bias[i,j] - bias[i,i]) vs |j-i|
    """
    query_positions = [0, len(positions) // 4, len(positions) // 2]
    n_queries = len(query_positions)
    
    setup_plot_style()
    fig, axes = plt.subplots(n_queries, 3, figsize=(18, 4 * n_queries))
    fig.suptitle("Position Bias Profiles from Selected Query Positions",
                 fontsize=16, fontweight='bold', y=1.02)
    
    for q_idx, q_pos in enumerate(query_positions):
        for name, pe in pe_dict.items():
            color = get_pe_color(name)
            bias, attn, _ = compute_pe_attention_bias(name, pe, positions, dim)
            
            # Panel 1: 原始偏置
            axes[q_idx, 0].plot(positions, bias[q_pos], color=color,
                                linewidth=1.5, label=name, alpha=0.8)
            
            # Panel 2: softmax 后
            axes[q_idx, 1].plot(positions, attn[q_pos], color=color,
                                linewidth=1.5, label=name, alpha=0.8)
            
            # Panel 3: 偏置差 vs 位置差
            delta_bias = bias[q_pos] - bias[q_pos, q_pos]
            delta_pos = np.abs(positions - positions[q_pos])
            # 按位置差排序
            sort_idx = np.argsort(delta_pos)
            axes[q_idx, 2].plot(delta_pos[sort_idx], delta_bias[sort_idx],
                                color=color, linewidth=1.5, label=name, alpha=0.8)
        
        axes[q_idx, 0].set_title(f"Raw Bias (query={q_pos})", fontsize=11)
        axes[q_idx, 0].set_xlabel("Key position $j$")
        axes[q_idx, 0].set_ylabel("Bias value")
        axes[q_idx, 0].legend(fontsize=7)
        axes[q_idx, 0].axvline(x=positions[q_pos], color='gray', ls=':', alpha=0.5)
        
        axes[q_idx, 1].set_title(f"Attention (query={q_pos})", fontsize=11)
        axes[q_idx, 1].set_xlabel("Key position $j$")
        axes[q_idx, 1].set_ylabel("Attention weight")
        axes[q_idx, 1].legend(fontsize=7)
        axes[q_idx, 1].axvline(x=positions[q_pos], color='gray', ls=':', alpha=0.5)
        
        axes[q_idx, 2].set_title(f"Bias Decay (query={q_pos})", fontsize=11)
        axes[q_idx, 2].set_xlabel("$|j - i|$")
        axes[q_idx, 2].set_ylabel("$\\Delta$ Bias")
        axes[q_idx, 2].legend(fontsize=7)
    
    plt.tight_layout()
    save_figure(fig, "05_bias_profiles", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. 多头注意力偏置对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_multi_head_bias(pe_dict: dict, positions: np.ndarray, dim: int,
                          n_heads: int = 8):
    """
    可视化多头注意力中各头的位置偏置差异。
    
    不同的头看到不同的位置偏置:
    - RoPE: 低频率头 → 长距离依赖; 高频率头 → 局部依赖
    - ALiBi: 不同 slope 的头 → 不同衰减速率
    - Sinusoidal: 不同维度子集 → 不同频率组合
    """
    # 只选2个代表性PE
    pe_to_show = ['rope', 'alibi']
    heads_to_show = [0, n_heads // 4, n_heads // 2, n_heads - 1]
    n_show_heads = len(heads_to_show)
    
    setup_plot_style()
    fig, axes = plt.subplots(len(pe_to_show), n_show_heads,
                              figsize=(4.5 * n_show_heads, 4 * len(pe_to_show)))
    fig.suptitle(f"Multi-Head Position Bias ({n_heads} heads)\n"
                 f"Each head sees different frequency/slope components",
                 fontsize=16, fontweight='bold', y=1.02)
    
    for row, name in enumerate(pe_to_show):
        if name not in pe_dict:
            continue
        pe = pe_dict[name]
        _, _, multi_head = compute_pe_attention_bias(
            name, pe, positions, dim, n_heads=n_heads)
        
        for col, h in enumerate(heads_to_show):
            ax = axes[row, col]
            bias_h = multi_head[h]
            # Softmax
            attn_h = softmax(bias_h, axis=-1)
            
            im = ax.imshow(attn_h, cmap='hot', aspect='equal',
                          interpolation='nearest', vmin=0)
            if col == 0:
                ax.set_ylabel(f"{name}", fontsize=11, fontweight='bold')
            if row == 0:
                ax.set_title(f"Head {h}", fontsize=11)
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    save_figure(fig, "05_multi_head_bias", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. 有效窗口大小 & 注意力熵分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_window_and_entropy(pe_dict: dict, positions: np.ndarray, dim: int):
    """
    Panel 1: 有效窗口大小 (50% 累积概率窗口)
    Panel 2: 注意力熵 (bits)
    Panel 3: 完整注意力 (含内容项) vs 纯位置偏置的熵对比
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    window_results = {}
    
    for name, pe in pe_dict.items():
        color = get_pe_color(name)
        bias, attn, _ = compute_pe_attention_bias(name, pe, positions, dim)
        
        # 有效窗口大小
        windows = []
        for i in range(len(positions)):
            w = compute_effective_window(attn[i], i, threshold=0.5)
            windows.append(w)
        
        # 注意力熵
        entropies = [compute_attention_entropy(attn[i]) for i in range(len(positions))]
        
        # 完整注意力 (含内容项)
        _, _, full_attn = compute_full_attention(name, pe, positions, dim)
        full_entropies = [compute_attention_entropy(full_attn[i])
                          for i in range(len(positions))]
        
        window_results[name] = {
            'mean_window': float(np.mean(windows)),
            'mean_entropy': float(np.mean(entropies)),
            'mean_full_entropy': float(np.mean(full_entropies)),
        }
        
        # Panel 1
        axes[0].plot(positions, windows, color=color, linewidth=1.5,
                     label=f'{name} (mean={np.mean(windows):.0f})', alpha=0.8)
        # Panel 2
        axes[1].plot(positions, entropies, color=color, linewidth=1.5,
                     label=f'{name}', alpha=0.8)
        # Panel 3
        axes[2].scatter([np.mean(entropies)], [np.mean(full_entropies)],
                       color=color, s=100, zorder=5, label=name,
                       edgecolors='black', linewidth=1.5)
    
    axes[0].set_xlabel("Query position $i$")
    axes[0].set_ylabel("Effective Window Size (50%)")
    axes[0].set_title("Effective Attention Window\n(positions containing 50% probability)",
                      fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8)
    
    axes[1].set_xlabel("Query position $i$")
    axes[1].set_ylabel("Attention Entropy (bits)")
    axes[1].set_title("Attention Entropy\n(low = focused, high = uniform)",
                      fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=8)
    max_entropy = np.log2(len(positions))
    axes[1].axhline(y=max_entropy, color='gray', ls='--', alpha=0.3,
                    label=f'Uniform ({max_entropy:.1f} bits)')
    
    axes[2].set_xlabel("Position-only Entropy (bits)")
    axes[2].set_ylabel("Full Attention Entropy (bits)")
    axes[2].set_title("Position vs Full Attention Entropy",
                      fontsize=12, fontweight='bold')
    axes[2].legend()
    # y=x 参考线
    max_val = max(axes[2].get_xlim()[1], axes[2].get_ylim()[1])
    axes[2].plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "05_window_entropy", MODULE)
    plt.close(fig)
    
    return window_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. 内容 vs 位置贡献分解
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_content_vs_position(pe_dict: dict, positions: np.ndarray, dim: int):
    """
    可视化 attention score 中内容项与位置项的相对贡献。
    
    展示:
    - 内容分数矩阵 (q^T k / sqrt(d))
    - 位置偏置矩阵
    - 相加后的完整分数矩阵
    - 各项的能量占比
    """
    pe_names = list(pe_dict.keys())
    n_pe = len(pe_names)
    
    setup_plot_style()
    fig, axes = plt.subplots(n_pe, 3, figsize=(15, 4 * n_pe))
    fig.suptitle("Attention Score Decomposition: Content + Position\n"
                 "Left: Content | Center: Position Bias | Right: Combined",
                 fontsize=16, fontweight='bold', y=1.02)
    
    rng = np.random.default_rng(42)
    
    for row, name in enumerate(pe_names):
        pe = pe_dict[name]
        content, bias, full_attn = compute_full_attention(
            name, pe, positions, dim, rng=np.random.default_rng(42))
        
        # 内容分数
        im1 = axes[row, 0].imshow(content, cmap='RdBu_r', aspect='equal',
                                   interpolation='nearest')
        axes[row, 0].set_title(f"{name} — Content" if row == 0 else name)
        if row == 0:
            axes[row, 0].set_title("Content Score\n$q_i^T k_j / \\sqrt{d}$")
        plt.colorbar(im1, ax=axes[row, 0], shrink=0.8)
        
        # 位置偏置
        im2 = axes[row, 1].imshow(bias, cmap='RdBu_r', aspect='equal',
                                   interpolation='nearest')
        if row == 0:
            axes[row, 1].set_title("Position Bias")
        plt.colorbar(im2, ax=axes[row, 1], shrink=0.8)
        
        # 完整注意力 (softmax 后)
        im3 = axes[row, 2].imshow(full_attn, cmap='hot', aspect='equal',
                                   interpolation='nearest', vmin=0)
        if row == 0:
            axes[row, 2].set_title("Full Attention\n(softmax)")
        plt.colorbar(im3, ax=axes[row, 2], shrink=0.8)
        
        # 标注能量占比
        content_energy = np.mean(content ** 2)
        bias_energy = np.mean(bias ** 2)
        total_energy = content_energy + bias_energy
        if total_energy > 0:
            content_ratio = content_energy / total_energy
            bias_ratio = bias_energy / total_energy
        else:
            content_ratio = 0.5
            bias_ratio = 0.5
        
        axes[row, 0].text(0.02, 0.02, f"E={content_energy:.4f}\n({content_ratio:.0%})",
                          transform=axes[row, 0].transAxes, fontsize=8,
                          color='white', va='bottom',
                          bbox=dict(facecolor='black', alpha=0.6))
        axes[row, 1].text(0.02, 0.02, f"E={bias_energy:.4f}\n({bias_ratio:.0%})",
                          transform=axes[row, 1].transAxes, fontsize=8,
                          color='white', va='bottom',
                          bbox=dict(facecolor='black', alpha=0.6))
    
    plt.tight_layout()
    save_figure(fig, "05_content_vs_position", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. Causal Mask 与位置偏置的交互
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_causal_attention(pe_dict: dict, positions: np.ndarray, dim: int):
    """
    可视化因果掩码 (causal mask) 与位置偏置的交互。
    
    因果掩码: A(i,j) = -inf if j > i (只能看到过去的 token)
    
    展示有/无因果掩码时的注意力分布差异。
    """
    N = len(positions)
    causal_mask = np.triu(np.ones((N, N)) * (-1e9), k=1)  # 上三角 = -inf
    
    pe_names = list(pe_dict.keys())
    n_pe = len(pe_names)
    
    setup_plot_style()
    fig, axes = plt.subplots(2, n_pe, figsize=(5 * n_pe, 9))
    fig.suptitle("Causal vs Bidirectional Attention\n"
                 "Top: Bidirectional | Bottom: Causal (autoregressive)",
                 fontsize=16, fontweight='bold', y=1.02)
    
    for idx, name in enumerate(pe_names):
        pe = pe_dict[name]
        bias, _, _ = compute_pe_attention_bias(name, pe, positions, dim)
        
        # 双向
        attn_bi = softmax(bias, axis=-1)
        axes[0, idx].imshow(attn_bi, cmap='hot', aspect='equal',
                           interpolation='nearest', vmin=0)
        axes[0, idx].set_title(f"{name} — Bidirectional", fontsize=11, fontweight='bold')
        
        # 因果
        bias_causal = bias + causal_mask
        attn_causal = softmax(bias_causal, axis=-1)
        axes[1, idx].imshow(attn_causal, cmap='hot', aspect='equal',
                           interpolation='nearest', vmin=0)
        axes[1, idx].set_title(f"{name} — Causal", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, "05_causal_attention", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  综合仪表盘
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_attention_dashboard(pe_dict: dict, positions: np.ndarray,
                              dim: int, window_results: dict):
    """
    综合仪表盘:
    Panel 1: 有效窗口大小柱状图
    Panel 2: 注意力熵柱状图
    Panel 3: 位置偏置能量
    Panel 4: 位置 vs 内容能量占比
    """
    names = list(pe_dict.keys())
    colors = [get_pe_color(n) for n in names]
    
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PE Attention Pattern — Dashboard",
                 fontsize=18, fontweight='bold')
    
    x = np.arange(len(names))
    bar_w = 0.6
    
    # Panel 1: 窗口大小
    ax1 = axes[0, 0]
    wins = [window_results[n]['mean_window'] for n in names]
    bars = ax1.bar(x, wins, bar_w, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Mean Window Size")
    ax1.set_title("Effective Attention Window (50%)", fontsize=12)
    ax1.axhline(y=len(positions), color='gray', ls='--', alpha=0.3,
                label=f'Full seq ({len(positions)})')
    ax1.legend(fontsize=8)
    for bar, val in zip(bars, wins):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel 2: 注意力熵
    ax2 = axes[0, 1]
    ents = [window_results[n]['mean_entropy'] for n in names]
    bars = ax2.bar(x, ents, bar_w, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Entropy (bits)")
    ax2.set_title("Position-Only Attention Entropy", fontsize=12)
    max_ent = np.log2(len(positions))
    ax2.axhline(y=max_ent, color='gray', ls='--', alpha=0.3,
                label=f'Uniform ({max_ent:.1f})')
    ax2.legend(fontsize=8)
    for bar, val in zip(bars, ents):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel 3: 位置偏置能量
    ax3 = axes[1, 0]
    bias_energies = []
    for name in names:
        pe = pe_dict[name]
        bias, _, _ = compute_pe_attention_bias(name, pe, positions, dim)
        bias_energies.append(np.mean(bias ** 2))
    bars = ax3.bar(x, bias_energies, bar_w, color=colors, edgecolor='black', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylabel("Mean $\\|\\text{Bias}\\|^2$")
    ax3.set_title("Position Bias Energy", fontsize=12)
    for bar, val in zip(bars, bias_energies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                 f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Panel 4: 内容 vs 位置能量堆叠图
    ax4 = axes[1, 1]
    content_ratios = []
    position_ratios = []
    for name in names:
        pe = pe_dict[name]
        content, bias, _ = compute_full_attention(
            name, pe, positions, dim, rng=np.random.default_rng(42))
        c_e = np.mean(content ** 2)
        b_e = np.mean(bias ** 2)
        total = c_e + b_e
        if total > 0:
            content_ratios.append(c_e / total)
            position_ratios.append(b_e / total)
        else:
            content_ratios.append(0.5)
            position_ratios.append(0.5)
    
    ax4.bar(x, content_ratios, bar_w, color='#2196F3', alpha=0.7,
            label='Content', edgecolor='black')
    ax4.bar(x, position_ratios, bar_w, bottom=content_ratios,
            color='#FF9800', alpha=0.7, label='Position', edgecolor='black')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.set_ylabel("Energy Ratio")
    ax4.set_title("Content vs Position Energy Ratio", fontsize=12)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    save_figure(fig, "05_attention_dashboard", MODULE)
    plt.close(fig)
    
    # Plotly 仪表盘
    if HAS_PLOTLY:
        pfig = make_subplots(rows=2, cols=2,
            subplot_titles=["Effective Window", "Attention Entropy",
                           "Bias Energy", "Content/Position Ratio"])
        for i, n in enumerate(names):
            color = get_pe_color(n)
            pfig.add_trace(go.Bar(x=[n], y=[wins[i]], marker_color=color,
                                  name=n, legendgroup=n, showlegend=True),
                          row=1, col=1)
            pfig.add_trace(go.Bar(x=[n], y=[ents[i]], marker_color=color,
                                  name=n, legendgroup=n, showlegend=False),
                          row=1, col=2)
            pfig.add_trace(go.Bar(x=[n], y=[bias_energies[i]], marker_color=color,
                                  name=n, legendgroup=n, showlegend=False),
                          row=2, col=1)
            pfig.add_trace(go.Bar(x=[n], y=[position_ratios[i]], marker_color=color,
                                  name=n, legendgroup=n, showlegend=False),
                          row=2, col=2)
        pfig.update_layout(template="plotly_white", width=1000, height=700,
                          title="PE Attention Pattern — Dashboard")
        save_plotly_html(pfig, "05_attention_dashboard.html", MODULE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HTML 报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_attention_report():
    sections = [
        {
            'title': '核心思想',
            'content': (
                '不同 PE 方案通过不同机制影响 attention score 矩阵：'
                '<ul>'
                '<li><strong>Sinusoidal/LAPE</strong>: \\(A(i,j) \\ni PE_i^T PE_j / \\sqrt{d}\\) — 加性位置偏置</li>'
                '<li><strong>RoPE</strong>: \\(K(\\Delta) = \\frac{1}{m}\\sum_k \\cos(\\theta_k \\Delta)\\) — 仅依赖相对位置差</li>'
                '<li><strong>ALiBi</strong>: \\(-m \\cdot |i-j|\\) — 线性距离惩罚</li>'
                '</ul>'
            )
        },
        {
            'title': '1. 纯位置偏置热力图',
            'content': '展示各 PE 方案在 attention 中引入的纯位置偏置矩阵，不含内容项。'
        },
        {
            'title': '2. 偏置剖面',
            'content': '从选定 query 位置出发，观察对所有 key 的偏置曲线和衰减特性。'
        },
        {
            'title': '3. 多头注意力',
            'content': (
                '不同注意力头看到不同的位置偏置：'
                '<br>RoPE 低频头 -> 长距离, 高频头 -> 局部；'
                '<br>ALiBi 不同 slope -> 不同衰减速率。'
            )
        },
        {
            'title': '4. 有效窗口与熵',
            'content': (
                '有效注意力窗口 = 包含 50% 概率质量的最小区间。'
                '<br>注意力熵反映分布的均匀程度。'
            )
        },
        {
            'title': '5. 内容 vs 位置分解',
            'content': 'Attention score 可分解为内容项 + 位置项，分析各自的能量贡献。'
        },
        {
            'title': '6. 因果注意力',
            'content': '对比双向注意力与因果 (autoregressive) 注意力在不同 PE 下的模式差异。'
        },
    ]
    generate_report_html(
        title="05 — PE Attention Pattern Analysis",
        sections=sections,
        module=MODULE,
        filename="05_attention_report.html"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  05 — PE Attention Pattern Analysis")
    print("  How Position Encodings Shape Attention Distributions")
    print("=" * 60)
    
    # 配置
    dim = 64
    seq_len = 64        # 中等长度 (热力图清晰)
    n_heads = 8
    
    config = PEConfig(dim=dim, max_len=seq_len)
    positions = np.arange(seq_len)
    
    pe_dict = get_all_pe(config=config)
    logger = VizLogger("pe_analysis", "05_attention_patterns")
    
    # ── 1. 纯位置偏置热力图 ──
    print("\n  [1/6] Position bias heatmaps...")
    plot_position_bias_heatmaps(pe_dict, positions, dim)
    print("        Saved 05_position_bias_heatmaps")
    logger.log_metric("bias_heatmaps", "completed")
    
    # ── 2. 偏置剖面 ──
    print("\n  [2/6] Bias profiles...")
    plot_bias_profiles(pe_dict, positions, dim)
    print("        Saved 05_bias_profiles")
    logger.log_metric("bias_profiles", "completed")
    
    # ── 3. 多头注意力 ──
    print("\n  [3/6] Multi-head bias...")
    plot_multi_head_bias(pe_dict, positions, dim, n_heads=n_heads)
    print("        Saved 05_multi_head_bias")
    logger.log_metric("multi_head", "completed")
    
    # ── 4. 有效窗口 & 熵 ──
    print("\n  [4/6] Window size & entropy analysis...")
    window_results = plot_window_and_entropy(pe_dict, positions, dim)
    print("        Saved 05_window_entropy")
    for name, r in window_results.items():
        logger.log_metric(f"window_{name}", r['mean_window'])
        logger.log_metric(f"entropy_{name}", r['mean_entropy'])
        print(f"        {name}: window={r['mean_window']:.0f}, "
              f"entropy={r['mean_entropy']:.2f} bits")
    
    # ── 5. 内容 vs 位置 ──
    print("\n  [5/6] Content vs position decomposition...")
    plot_content_vs_position(pe_dict, positions, dim)
    print("        Saved 05_content_vs_position")
    logger.log_metric("content_vs_position", "completed")
    
    # ── 6. 因果注意力 ──
    print("\n  [6/6] Causal attention...")
    plot_causal_attention(pe_dict, positions, dim)
    print("        Saved 05_causal_attention")
    logger.log_metric("causal_attention", "completed")
    
    # ── 仪表盘 ──
    print("\n  Dashboard...")
    plot_attention_dashboard(pe_dict, positions, dim, window_results)
    print("        Saved 05_attention_dashboard")
    
    # ── 报告 ──
    print("\n  Generating HTML report...")
    generate_attention_report()
    
    logger.save()
    
    print("\n" + "=" * 60)
    print("  05_attention_patterns complete!")
    print("  Static:      output/pe_analysis/05_*")
    print("  Interactive:  html/pe_analysis/05_*")
    print("=" * 60)


if __name__ == "__main__":
    main()
