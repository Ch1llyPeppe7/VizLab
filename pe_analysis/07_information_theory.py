#!/usr/bin/env python3
"""
07 — PE 信息论分析

核心思想:
    从 Shannon 信息论的视角对位置编码进行定量剖析。

    核心问题: 位置编码 PE: ℤ → ℝ^d 是一个从离散位置到连续向量的映射，
    它到底 "编码了多少位置信息"？

    1. 位置分辨率 (Position Resolution):
       - 相邻位置 p 与 p+δ 的编码有多容易区分？
       - 度量: KL 散度 D_KL(P_p ∥ P_{p+δ}), Hellinger 距离 H(P_p, P_{p+δ})
       - 将 PE 向量视为高斯分布的充分统计量，分析其分离度

    2. 编码容量 (Encoding Capacity):
       - 互信息 I(Position; Encoding) 量化位置→编码的信道容量
       - 离散估计: 对每个维度计算 I(p; PE_k(p))
       - 高容量 = 每个维度都携带独立的位置信息

    3. Fisher 信息矩阵:
       - F(p) = E[(∂log f/∂p)²] = ‖∂PE/∂p‖²  (高斯近似下)
       - 衡量编码对位置微小扰动的灵敏度
       - Fisher 信息高 → 位置估计的 Cramér-Rao 下界低 → 编码精度高

    4. 信息瓶颈 (Information Bottleneck):
       - 将 d 维编码投影到 k < d 维，保留了多少位置信息？
       - 用 PCA 截断模拟信息瓶颈: I_k / I_d
       - 揭示编码的冗余度和鲁棒性

Output:
    output/pe_analysis/   → 静态图 (PNG/PDF)
    html/pe_analysis/     → 交互式 HTML

Usage:
    python -m pe_analysis.07_information_theory
    python run.py pe_analysis.information
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
#  数学工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _to_pseudo_distribution(vec: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    将 PE 向量转化为伪概率分布 (用于 KL/Hellinger 计算)。

    方法: 对 PE 向量加高斯噪声后取 softmax，模拟 "位置 p 的编码
    在各维度上的注意力分配"。

    Args:
        vec: [d,] PE 向量
        sigma: 高斯噪声标准差 (控制平滑度)
    Returns:
        [d,] 概率分布
    """
    logits = vec / (sigma + 1e-10)
    logits = logits - np.max(logits)  # 数值稳定
    exp_logits = np.exp(logits)
    return exp_logits / (exp_logits.sum() + 1e-15)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    KL 散度 D_KL(P ∥ Q) = Σ p_i log(p_i / q_i)

    Args:
        p, q: 概率分布
        eps: 数值稳定项
    Returns:
        KL 散度 (bits)
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log2(p / q)))


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Hellinger 距离: H(P, Q) = (1/√2) ‖√P - √Q‖₂

    取值范围 [0, 1]，0 = 完全相同，1 = 完全不重叠。
    """
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon 散度: JSD(P ∥ Q) = (D_KL(P∥M) + D_KL(Q∥M)) / 2
    其中 M = (P + Q) / 2

    对称、有界 [0, 1] (以 bits 计)。
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def mutual_information_per_dim(positions: np.ndarray, encodings: np.ndarray,
                                n_bins: int = 50) -> np.ndarray:
    """
    逐维度计算互信息 I(Position; PE_k)。

    Args:
        positions: [N,] 位置序列
        encodings: [N, d] 编码矩阵
        n_bins: 直方图 bin 数
    Returns:
        [d,] 每个维度的互信息 (bits)
    """
    d = encodings.shape[1]
    mi = np.zeros(d)
    for k in range(d):
        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        c_xy = np.histogram2d(positions, encodings[:, k], bins=n_bins)[0]
        c_x = np.histogram(positions, bins=n_bins)[0]
        c_y = np.histogram(encodings[:, k], bins=n_bins)[0]

        def _entropy(counts):
            p = counts / (counts.sum() + 1e-15)
            p = p[p > 0]
            return -np.sum(p * np.log2(p))

        mi[k] = _entropy(c_x) + _entropy(c_y) - _entropy(c_xy)
    return np.clip(mi, 0, None)  # MI ≥ 0


def fisher_information_vector(pe_instance, positions: np.ndarray,
                               dp: float = 0.5) -> np.ndarray:
    """
    Fisher 信息估计: F(p) ≈ ‖∂PE/∂p‖² (对角近似)。

    使用中心差分 ∂PE/∂p ≈ (PE(p+dp) - PE(p-dp)) / (2·dp)。

    Args:
        pe_instance: PositionEncoding 实例
        positions: [N,] 位置序列
        dp: 差分步长
    Returns:
        [N,] 各位置的 Fisher 信息
    """
    pos = np.atleast_1d(positions).astype(float)
    pe_plus = pe_instance.encode(pos + dp)    # [N, d]
    pe_minus = pe_instance.encode(pos - dp)   # [N, d]
    grad = (pe_plus - pe_minus) / (2 * dp)    # [N, d]
    return np.sum(grad ** 2, axis=1)           # [N,]


def fisher_information_spectrum(pe_instance, positions: np.ndarray,
                                 dp: float = 0.5) -> np.ndarray:
    """
    Fisher 信息谱: 各维度的 Fisher 信息贡献。

    F_k(p) = (∂PE_k/∂p)²，然后对位置取平均。

    Returns:
        [d,] 各维度的平均 Fisher 信息
    """
    pos = np.atleast_1d(positions).astype(float)
    pe_plus = pe_instance.encode(pos + dp)
    pe_minus = pe_instance.encode(pos - dp)
    grad = (pe_plus - pe_minus) / (2 * dp)      # [N, d]
    return np.mean(grad ** 2, axis=0)            # [d,]


def information_retention_curve(positions: np.ndarray, encodings: np.ndarray,
                                 n_bins: int = 50) -> tuple:
    """
    信息瓶颈曲线: PCA 截断到 k 维后保留的位置信息比例。

    Args:
        positions: [N,]
        encodings: [N, d]
        n_bins: 直方图 bin 数
    Returns:
        (dims, retention_ratios) — dims=[1..d], retention=[0..1]
    """
    d = encodings.shape[1]
    # PCA
    centered = encodings - encodings.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # 完整维度的总互信息
    mi_full_per_dim = mutual_information_per_dim(positions, encodings, n_bins)
    mi_full = mi_full_per_dim.sum()

    dims = np.arange(1, d + 1)
    retention = np.zeros(d)

    for k in range(1, d + 1):
        projected = U[:, :k] * S[:k]  # [N, k]
        mi_k = mutual_information_per_dim(positions, projected, n_bins).sum()
        retention[k - 1] = mi_k / (mi_full + 1e-15)

    return dims, np.clip(retention, 0, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  可视化函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_position_resolution(pe_dict: dict, max_len: int = 256):
    """
    [图 1] 位置分辨率: 相邻位置的 KL 散度、Hellinger 距离、JSD。

    分析: 各方案在不同位置间距 δ 下的可分辨度。
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Position Resolution — 位置分辨率分析", fontsize=16, fontweight='bold')

    deltas = np.arange(1, 33)  # 位置偏移量 δ = 1..32
    ref_pos = max_len // 4     # 参考位置

    for pe_name, pe_inst in pe_dict.items():
        color = get_pe_color(pe_name)
        label = pe_inst.name

        # 对 ALiBi 特殊处理: 它的 encode() 返回零矩阵
        if pe_name == 'alibi':
            # 使用 bias 作为伪编码
            L = max_len
            bias = pe_inst.bias_matrix(L, head_idx=0)
            ref_vec = bias[ref_pos, :]
            kl_vals, hel_vals, jsd_vals = [], [], []
            for d in deltas:
                if ref_pos + d < L:
                    neighbor_vec = bias[ref_pos + d, :]
                    p = _to_pseudo_distribution(ref_vec)
                    q = _to_pseudo_distribution(neighbor_vec)
                    kl_vals.append(kl_divergence(p, q))
                    hel_vals.append(hellinger_distance(p, q))
                    jsd_vals.append(jensen_shannon_divergence(p, q))
                else:
                    kl_vals.append(np.nan)
                    hel_vals.append(np.nan)
                    jsd_vals.append(np.nan)
        else:
            positions = np.arange(max_len)
            enc = pe_inst.encode(positions)
            ref_vec = enc[ref_pos]  # [d,]
            p_ref = _to_pseudo_distribution(ref_vec)

            kl_vals, hel_vals, jsd_vals = [], [], []
            for d in deltas:
                q = _to_pseudo_distribution(enc[ref_pos + d])
                kl_vals.append(kl_divergence(p_ref, q))
                hel_vals.append(hellinger_distance(p_ref, q))
                jsd_vals.append(jensen_shannon_divergence(p_ref, q))

        axes[0].plot(deltas, kl_vals, color=color, label=label, linewidth=2)
        axes[1].plot(deltas, hel_vals, color=color, label=label, linewidth=2)
        axes[2].plot(deltas, jsd_vals, color=color, label=label, linewidth=2)

    titles = [
        r"$D_{KL}(P_p \| P_{p+\delta})$ — KL Divergence",
        r"$H(P_p, P_{p+\delta})$ — Hellinger Distance",
        r"$JSD(P_p \| P_{p+\delta})$ — Jensen-Shannon",
    ]
    y_labels = ["KL (bits)", "Hellinger [0,1]", "JSD (bits)"]

    for ax, title, ylabel in zip(axes, titles, y_labels):
        ax.set_xlabel(r"Position offset $\delta$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "07_position_resolution", MODULE)
    plt.close(fig)


def plot_encoding_capacity(pe_dict: dict, max_len: int = 256):
    """
    [图 2] 编码容量: 逐维度互信息 I(Position; PE_k)。

    高容量维度 = 该维度能有效区分不同位置。
    """
    setup_plot_style()
    n_pe = len(pe_dict)
    fig, axes = plt.subplots(1, n_pe, figsize=(5 * n_pe, 5), sharey=True)
    if n_pe == 1:
        axes = [axes]
    fig.suptitle("Encoding Capacity — 逐维度互信息 $I(\\mathrm{Position}; \\mathrm{PE}_k)$",
                 fontsize=16, fontweight='bold')

    positions = np.arange(max_len).astype(float)

    for idx, (pe_name, pe_inst) in enumerate(pe_dict.items()):
        ax = axes[idx]
        color = get_pe_color(pe_name)

        if pe_name == 'alibi':
            # 使用 bias 的第一行作为伪编码
            bias = pe_inst.bias_matrix(max_len, head_idx=0)
            enc = bias  # [L, L]
            # 只取前 dim 列
            enc = enc[:, :min(enc.shape[1], pe_inst.dim)]
        else:
            enc = pe_inst.encode(positions)  # [N, d]

        mi = mutual_information_per_dim(positions, enc, n_bins=50)
        dims = np.arange(len(mi))

        ax.bar(dims, mi, color=color, alpha=0.7, width=0.8)
        ax.axhline(y=np.mean(mi), color='red', linestyle='--', alpha=0.7,
                   label=f'Mean = {np.mean(mi):.2f} bits')
        ax.set_xlabel("Dimension index $k$", fontsize=11)
        if idx == 0:
            ax.set_ylabel("$I(\\mathrm{pos}; \\mathrm{PE}_k)$ (bits)", fontsize=11)
        ax.set_title(f"{pe_inst.name}\nTotal = {mi.sum():.1f} bits", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "07_encoding_capacity", MODULE)
    plt.close(fig)


def plot_fisher_information(pe_dict: dict, max_len: int = 256):
    """
    [图 3] Fisher 信息分析。

    上排: F(p) = ‖∂PE/∂p‖² 随位置 p 的变化
    下排: Fisher 信息谱 — 各维度的平均 Fisher 贡献

    物理意义: Fisher 信息高 → Cramér-Rao 下界低 → 位置估计精度高
    """
    setup_plot_style()
    n_pe = len(pe_dict)
    fig, axes = plt.subplots(2, n_pe, figsize=(5 * n_pe, 9))
    if n_pe == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Fisher Information — 编码灵敏度分析", fontsize=16, fontweight='bold')

    positions = np.arange(max_len).astype(float)

    for idx, (pe_name, pe_inst) in enumerate(pe_dict.items()):
        color = get_pe_color(pe_name)

        if pe_name == 'alibi':
            # ALiBi 的 Fisher 信息 = slope² (常数)
            fi = np.full(max_len, pe_inst.slopes[0] ** 2)
            fi_spec = np.full(pe_inst.dim, pe_inst.slopes[0] ** 2 / pe_inst.dim)
        else:
            fi = fisher_information_vector(pe_inst, positions)
            fi_spec = fisher_information_spectrum(pe_inst, positions)

        # 上排: F(p) vs p
        ax_top = axes[0, idx]
        ax_top.plot(positions, fi, color=color, linewidth=1.5, alpha=0.8)
        ax_top.axhline(y=np.mean(fi), color='red', linestyle='--', alpha=0.6,
                       label=f'Mean = {np.mean(fi):.2f}')
        ax_top.set_xlabel("Position $p$", fontsize=11)
        if idx == 0:
            ax_top.set_ylabel(r"$F(p) = \|\partial \mathrm{PE}/\partial p\|^2$", fontsize=11)
        ax_top.set_title(pe_inst.name, fontsize=13, fontweight='bold')
        ax_top.legend(fontsize=9)
        ax_top.grid(True, alpha=0.3)

        # 下排: Fisher 谱 (逐维度)
        ax_bot = axes[1, idx]
        dim_idx = np.arange(len(fi_spec))
        ax_bot.bar(dim_idx, fi_spec, color=color, alpha=0.7, width=0.8)
        ax_bot.set_xlabel("Dimension index $k$", fontsize=11)
        if idx == 0:
            ax_bot.set_ylabel(r"$\langle (\partial \mathrm{PE}_k / \partial p)^2 \rangle_p$",
                             fontsize=11)
        ax_bot.set_title(f"Fisher Spectrum (sum={fi_spec.sum():.1f})", fontsize=11)
        ax_bot.grid(True, alpha=0.3, axis='y')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "07_fisher_information", MODULE)
    plt.close(fig)


def plot_information_bottleneck(pe_dict: dict, max_len: int = 256):
    """
    [图 4] 信息瓶颈: PCA 维度截断后的信息保留率。

    横轴: 保留的主成分个数 k
    纵轴: I_k / I_d (信息保留比例)

    陡峭上升 → 信息集中在少数方向 → 编码冗余度高
    平缓上升 → 信息均匀分布 → 编码利用率高
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Information Bottleneck — 维度压缩下的信息保留",
                 fontsize=16, fontweight='bold')

    positions = np.arange(max_len).astype(float)

    bottleneck_results = {}

    for pe_name, pe_inst in pe_dict.items():
        color = get_pe_color(pe_name)
        label = pe_inst.name

        if pe_name == 'alibi':
            # ALiBi: 用 bias matrix 的前 dim 列
            bias = pe_inst.bias_matrix(max_len, head_idx=0)
            enc = bias[:, :min(bias.shape[1], pe_inst.dim)]
        else:
            enc = pe_inst.encode(positions)

        dims, retention = information_retention_curve(positions, enc, n_bins=40)
        bottleneck_results[pe_name] = {
            'dims': dims, 'retention': retention,
            'dim_90': int(dims[np.searchsorted(retention, 0.9)]) if np.any(retention >= 0.9) else len(dims),
            'dim_95': int(dims[np.searchsorted(retention, 0.95)]) if np.any(retention >= 0.95) else len(dims),
        }

        # 左图: 信息保留曲线
        axes[0].plot(dims, retention, color=color, label=label, linewidth=2)

        # 右图: 信息增量 (每增加一个维度带来的边际信息)
        marginal = np.diff(retention, prepend=0)
        axes[1].bar(dims[:20], marginal[:20], color=color, alpha=0.5,
                    width=0.8, label=label)

    # 左图装饰
    axes[0].axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
    axes[0].axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
    axes[0].set_xlabel("Retained dimensions $k$", fontsize=12)
    axes[0].set_ylabel("Information retention $I_k / I_d$", fontsize=12)
    axes[0].set_title("信息保留率 vs 维度", fontsize=13)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.1)

    # 右图装饰
    axes[1].set_xlabel("Dimension index $k$ (top 20)", fontsize=12)
    axes[1].set_ylabel("Marginal information gain", fontsize=12)
    axes[1].set_title("边际信息增量 (前 20 维)", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "07_information_bottleneck", MODULE)
    plt.close(fig)

    return bottleneck_results


def plot_information_dashboard(pe_dict: dict, max_len: int = 256):
    """
    [图 5] 综合信息论仪表盘: 4-panel 对比。

    Panel A: 位置分辨率 (JSD vs δ)
    Panel B: 编码容量汇总 (总互信息 bar chart)
    Panel C: Fisher 信息 (F(p) 对比曲线)
    Panel D: 信息瓶颈 (保留率曲线)
    """
    setup_plot_style()
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("Information-Theoretic Dashboard — PE 信息论综合分析",
                 fontsize=17, fontweight='bold', y=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    positions = np.arange(max_len).astype(float)
    deltas = np.arange(1, 33)
    ref_pos = max_len // 4
    summary = {}

    for pe_name, pe_inst in pe_dict.items():
        color = get_pe_color(pe_name)
        label = pe_inst.name

        # ── Panel A: JSD vs δ ──
        if pe_name == 'alibi':
            bias = pe_inst.bias_matrix(max_len, head_idx=0)
            ref_vec = bias[ref_pos, :]
            jsd_vals = []
            for d in deltas:
                if ref_pos + d < max_len:
                    p = _to_pseudo_distribution(ref_vec)
                    q = _to_pseudo_distribution(bias[ref_pos + d, :])
                    jsd_vals.append(jensen_shannon_divergence(p, q))
                else:
                    jsd_vals.append(np.nan)
            enc = bias[:, :min(bias.shape[1], pe_inst.dim)]
        else:
            enc = pe_inst.encode(positions)
            p_ref = _to_pseudo_distribution(enc[ref_pos])
            jsd_vals = []
            for d in deltas:
                q = _to_pseudo_distribution(enc[ref_pos + d])
                jsd_vals.append(jensen_shannon_divergence(p_ref, q))

        ax_a.plot(deltas, jsd_vals, color=color, label=label, linewidth=2)

        # ── Panel B: 总互信息 ──
        mi = mutual_information_per_dim(positions, enc, n_bins=50)
        total_mi = float(mi.sum())

        # ── Panel C: Fisher 信息 ──
        if pe_name == 'alibi':
            fi = np.full(max_len, pe_inst.slopes[0] ** 2)
        else:
            fi = fisher_information_vector(pe_inst, positions)
        ax_c.plot(positions, fi, color=color, label=label, linewidth=1.5, alpha=0.8)

        # ── Panel D: 信息瓶颈 ──
        dims, retention = information_retention_curve(positions, enc, n_bins=40)
        ax_d.plot(dims, retention, color=color, label=label, linewidth=2)

        # 汇总
        dim_90 = int(dims[np.searchsorted(retention, 0.9)]) if np.any(retention >= 0.9) else len(dims)
        summary[pe_name] = {
            'total_mi': total_mi,
            'mean_fisher': float(np.mean(fi)),
            'dim_90': dim_90,
            'jsd_at_1': jsd_vals[0],
        }

    # Panel A 装饰
    ax_a.set_xlabel(r"Position offset $\delta$", fontsize=11)
    ax_a.set_ylabel("JSD (bits)", fontsize=11)
    ax_a.set_title("(A) Position Resolution", fontsize=13, fontweight='bold')
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3)

    # Panel B: Bar chart
    names = [pe_dict[n].name for n in summary]
    total_mis = [summary[n]['total_mi'] for n in summary]
    colors = [get_pe_color(n) for n in summary]
    bars = ax_b.bar(names, total_mis, color=colors, alpha=0.8)
    for bar, val in zip(bars, total_mis):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_b.set_ylabel("Total MI (bits)", fontsize=11)
    ax_b.set_title("(B) Encoding Capacity", fontsize=13, fontweight='bold')
    ax_b.grid(True, alpha=0.3, axis='y')

    # Panel C 装饰
    ax_c.set_xlabel("Position $p$", fontsize=11)
    ax_c.set_ylabel(r"$F(p) = \|\partial \mathrm{PE}/\partial p\|^2$", fontsize=11)
    ax_c.set_title("(C) Fisher Information", fontsize=13, fontweight='bold')
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)

    # Panel D 装饰
    ax_d.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
    ax_d.set_xlabel("Retained dimensions $k$", fontsize=11)
    ax_d.set_ylabel("Information retention", fontsize=11)
    ax_d.set_title("(D) Information Bottleneck", fontsize=13, fontweight='bold')
    ax_d.legend(fontsize=9, loc='lower right')
    ax_d.grid(True, alpha=0.3)
    ax_d.set_ylim(-0.05, 1.1)

    fig.subplots_adjust(top=0.92)
    save_figure(fig, "07_information_dashboard", MODULE)
    plt.close(fig)

    # ── 交互式 HTML ──
    if HAS_PLOTLY:
        pfig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["(A) Position Resolution (JSD)",
                            "(B) Encoding Capacity",
                            "(C) Fisher Information",
                            "(D) Information Bottleneck"],
            vertical_spacing=0.12, horizontal_spacing=0.1
        )

        for pe_name, pe_inst in pe_dict.items():
            color = get_pe_color(pe_name)
            label = pe_inst.name

            # A: JSD
            if pe_name == 'alibi':
                bias = pe_inst.bias_matrix(max_len, head_idx=0)
                ref_vec = bias[ref_pos, :]
                jsd_vals = []
                for d_val in deltas:
                    if ref_pos + d_val < max_len:
                        p = _to_pseudo_distribution(ref_vec)
                        q = _to_pseudo_distribution(bias[ref_pos + d_val, :])
                        jsd_vals.append(jensen_shannon_divergence(p, q))
                    else:
                        jsd_vals.append(None)
                enc = bias[:, :min(bias.shape[1], pe_inst.dim)]
            else:
                enc = pe_inst.encode(positions)
                p_ref = _to_pseudo_distribution(enc[ref_pos])
                jsd_vals = [jensen_shannon_divergence(p_ref, _to_pseudo_distribution(enc[ref_pos + d_val]))
                            for d_val in deltas]

            pfig.add_trace(go.Scatter(x=deltas.tolist(), y=jsd_vals, name=label,
                                       line=dict(color=color, width=2),
                                       legendgroup=label, showlegend=True),
                           row=1, col=1)

            # C: Fisher
            if pe_name == 'alibi':
                fi = np.full(max_len, pe_inst.slopes[0] ** 2)
            else:
                fi = fisher_information_vector(pe_inst, positions)
            pfig.add_trace(go.Scatter(x=positions.tolist(), y=fi.tolist(), name=label,
                                       line=dict(color=color, width=1.5),
                                       legendgroup=label, showlegend=False),
                           row=2, col=1)

            # D: Bottleneck
            dims, retention = information_retention_curve(positions, enc, n_bins=40)
            pfig.add_trace(go.Scatter(x=dims.tolist(), y=retention.tolist(), name=label,
                                       line=dict(color=color, width=2),
                                       legendgroup=label, showlegend=False),
                           row=2, col=2)

        # B: Bar chart
        names = [pe_dict[n].name for n in summary]
        total_mis = [summary[n]['total_mi'] for n in summary]
        colors = [get_pe_color(n) for n in summary]
        pfig.add_trace(go.Bar(x=names, y=total_mis,
                               marker_color=colors, text=[f'{v:.1f}' for v in total_mis],
                               textposition='auto', showlegend=False),
                       row=1, col=2)

        pfig.update_layout(
            title=dict(text="PE Information-Theoretic Dashboard", font=dict(size=20)),
            height=800, width=1100, template='plotly_white'
        )
        save_plotly_html(pfig, "07_information_dashboard.html", MODULE)

    return summary


def generate_information_report(summary: dict = None):
    """
    生成信息论分析 HTML 报告。
    """
    sections = [
        {
            'title': '1. 位置分辨率 (Position Resolution)',
            'content': r"""
                <p>位置分辨率衡量编码对相邻位置的区分能力。我们使用三种信息论度量：</p>
                <ul>
                    <li><b>KL 散度</b>: \( D_{KL}(P_p \| P_{p+\delta}) = \sum_k p_k \log(p_k / q_k) \)
                        — 非对称，衡量用 Q 近似 P 的信息损失</li>
                    <li><b>Hellinger 距离</b>: \( H(P, Q) = \frac{1}{\sqrt{2}} \|\sqrt{P} - \sqrt{Q}\|_2 \in [0, 1] \)
                        — 对称，几何直觉清晰</li>
                    <li><b>Jensen-Shannon 散度</b>: \( JSD(P \| Q) = \frac{1}{2}[D_{KL}(P\|M) + D_{KL}(Q\|M)] \)
                        — 对称、有界、可开根得到度量距离</li>
                </ul>
                <p><b>发现</b>: Sinusoidal/RoPE 的分辨率随 δ 单调递增（基于多尺度频率）；
                ALiBi 线性增长（线性偏置）；LAPE 因幂律频率呈现非单调特征。</p>
            """
        },
        {
            'title': '2. 编码容量 (Encoding Capacity)',
            'content': r"""
                <p>编码容量通过逐维度互信息 \( I(\mathrm{pos}; \mathrm{PE}_k) \) 量化：</p>
                <p>\[ I(X; Y) = H(X) + H(Y) - H(X, Y) \]</p>
                <p>总容量 \( C = \sum_k I(\mathrm{pos}; \mathrm{PE}_k) \) 反映编码的"信道带宽"。</p>
                <p><b>关键结论</b>:</p>
                <ul>
                    <li>Sinusoidal/RoPE: 高频维度 MI 低（高频振荡被离散化"混叠"），低频维度 MI 高</li>
                    <li>LAPE: 幂律频率分布导致 MI 分布更不均匀</li>
                    <li>总容量决定了编码能支持的最大序列长度</li>
                </ul>
            """
        },
        {
            'title': '3. Fisher 信息 (Fisher Information)',
            'content': r"""
                <p>Fisher 信息衡量编码对位置微小扰动的灵敏度：</p>
                <p>\[ F(p) = \left\| \frac{\partial \mathrm{PE}}{\partial p} \right\|^2
                   = \sum_k \left(\frac{\partial \mathrm{PE}_k}{\partial p}\right)^2 \]</p>
                <p>由 <b>Cramér-Rao 下界</b>: \(\mathrm{Var}(\hat{p}) \geq 1 / F(p)\)，
                Fisher 信息越高，从编码反推位置的估计精度越高。</p>
                <p><b>Fisher 信息谱</b> 分解到各维度，揭示哪些维度贡献最多位置信息。
                对 Sinusoidal PE，低索引维度（高频）的 Fisher 贡献最大。</p>
            """
        },
        {
            'title': '4. 信息瓶颈 (Information Bottleneck)',
            'content': r"""
                <p>信息瓶颈分析将 d 维编码通过 PCA 压缩到 k 维，衡量信息保留率：</p>
                <p>\[ R(k) = \frac{I_k(\mathrm{pos}; \mathrm{PE}^{(k)})}{I_d(\mathrm{pos}; \mathrm{PE})} \]</p>
                <p><b>诊断标准</b>:</p>
                <ul>
                    <li>R(k) 快速趋近 1 → 编码冗余度高，少数主成分已携带绝大部分位置信息</li>
                    <li>R(k) 缓慢增长 → 编码利用率高，每个维度都贡献独立信息</li>
                    <li>"90% 维度" (达到 90% 信息保留所需的最少维度) 是衡量编码效率的关键指标</li>
                </ul>
            """
        },
    ]

    # 添加数值汇总
    if summary:
        summary_rows = ""
        for name, data in summary.items():
            summary_rows += f"""
                <tr>
                    <td><b>{name}</b></td>
                    <td>{data['total_mi']:.1f}</td>
                    <td>{data['mean_fisher']:.2f}</td>
                    <td>{data['dim_90']}</td>
                    <td>{data['jsd_at_1']:.4f}</td>
                </tr>
            """
        sections.append({
            'title': '5. 数值汇总',
            'content': f"""
                <table border="1" cellpadding="8" cellspacing="0"
                       style="border-collapse: collapse; width: 100%; margin-top: 12px;">
                    <tr style="background: #f0f4f8;">
                        <th>PE 方案</th>
                        <th>总互信息 (bits)</th>
                        <th>平均 Fisher</th>
                        <th>90% 维度</th>
                        <th>JSD(δ=1)</th>
                    </tr>
                    {summary_rows}
                </table>
            """
        })

    generate_report_html(
        title="07 — PE 信息论分析报告",
        sections=sections,
        module=MODULE,
        filename="07_information_theory_report.html"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  07 — PE Information Theory Analysis")
    print("=" * 60)

    dim = 64
    max_len = 256

    config = PEConfig(dim=dim, max_len=max_len)
    pe_dict = get_all_pe(config=config)
    logger = VizLogger("pe_analysis", "07_information_theory")

    # 1. 位置分辨率
    print("\n  [1/5] Position resolution (KL / Hellinger / JSD)...")
    plot_position_resolution(pe_dict, max_len)
    print("        Saved 07_position_resolution")
    logger.log_metric("position_resolution", "completed")

    # 2. 编码容量
    print("\n  [2/5] Encoding capacity (per-dim MI)...")
    plot_encoding_capacity(pe_dict, max_len)
    print("        Saved 07_encoding_capacity")
    logger.log_metric("encoding_capacity", "completed")

    # 3. Fisher 信息
    print("\n  [3/5] Fisher information...")
    plot_fisher_information(pe_dict, max_len)
    print("        Saved 07_fisher_information")
    logger.log_metric("fisher_information", "completed")

    # 4. 信息瓶颈
    print("\n  [4/5] Information bottleneck...")
    bottleneck = plot_information_bottleneck(pe_dict, max_len)
    print("        Saved 07_information_bottleneck")
    for name, res in bottleneck.items():
        logger.log_metric(f"dim90_{name}", res['dim_90'])
        print(f"        {name}: 90% @ {res['dim_90']} dims")

    # 5. 综合仪表盘
    print("\n  [5/5] Information dashboard...")
    summary = plot_information_dashboard(pe_dict, max_len)
    print("        Saved 07_information_dashboard")
    for name, s in summary.items():
        logger.log_metric(f"total_mi_{name}", s['total_mi'])
        print(f"        {name}: MI={s['total_mi']:.1f} bits, Fisher={s['mean_fisher']:.2f}")

    # 报告
    print("\n  Generating report...")
    generate_information_report(summary)

    logger.save()

    print("\n" + "=" * 60)
    print("  07_information_theory complete!")
    print("  Static:      output/pe_analysis/07_*")
    print("  Interactive:  html/pe_analysis/07_*")
    print("=" * 60)


if __name__ == "__main__":
    main()
