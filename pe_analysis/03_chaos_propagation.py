#!/usr/bin/env python3
"""
03 — PE 深层传播混沌性分析 (Chaos Propagation Analysis)

核心假设 (The Chaos Hypothesis):
    加性绝对位置编码 (Sinusoidal PE) 仅在第一层注入位置信息。
    随着前馈传播（线性变换 + 非线性激活），位置信号逐层退化为混沌/噪声，
    丧失位置区分能力。

    乘性相对位置编码 (RoPE) 在每一层独立施加 SO(2) 旋转变换，
    因此能在深层保持位置信息的结构性。

数学工具：
    - Lyapunov 指数: 量化相邻位置轨迹的指数发散率
    - 相空间重构 (Takens 定理): 揭示位置信号在高维相空间中的几何退化
    - 距离矩阵保持度: 量化逐层核函数的结构保持
    - 谱熵演化: 频率结构随层数的退化

模拟架构：
    x_0 = Embed(token) + PE(p)                  # 加性 PE
    x_{l+1} = LayerNorm(x_l + FFN(x_l))         # 残差 + FFN + LN
    
    对于 RoPE：
    q_l = RoPE(x_l, p),  k_l = RoPE(x_l, p)    # 每层独立旋转

Output:
    output/pe_analysis/   → 静态图 (PNG/PDF)
    html/pe_analysis/     → 交互式 HTML

Usage:
    python -m pe_analysis.03_chaos_propagation
    python run.py pe_analysis.chaos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal as sp_signal
from scipy.spatial.distance import pdist, squareform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    PEConfig, get_pe, get_all_pe, SinusoidalPE, RoPE, ALiBi, LAPE,
    setup_plot_style, save_figure, add_math_annotation,
    save_plotly_html, generate_report_html, get_pe_color,
    PE_COLORS, VizLogger,
    spectral_entropy, compute_lyapunov_exponent, compute_phase_space,
    simulate_feedforward_layer, layer_norm, activation_fn,
    random_weight_matrix, effective_rank,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

MODULE = "pe_analysis"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  模拟引擎：多层 Transformer 前馈传播
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TransformerSimulator:
    """
    纯数学 Transformer 前馈传播模拟器。
    
    模拟 L 层 Transformer 的前馈传播过程（不含 self-attention），
    用于研究位置编码信号在深层的退化行为。
    
    架构：
        x_{l+1} = LN(x_l + FFN(x_l))
        FFN(x) = W2 · σ(W1 · x + b1) + b2
    
    对于加性 PE (Sinusoidal, LAPE)：
        x_0 = random_embed + PE(p)   → 只在第 0 层注入
        
    对于乘性 PE (RoPE)：
        x_0 = random_embed           → 每层通过旋转注入位置
        在分析时，对 x_l 施加 RoPE 旋转后再测量
    """
    
    def __init__(self, dim: int = 64, n_layers: int = 24,
                 ff_mult: int = 4, activation: str = 'gelu',
                 use_residual: bool = True, use_layernorm: bool = True,
                 seed: int = 42):
        self.dim = dim
        self.n_layers = n_layers
        self.ff_mult = ff_mult
        self.activation = activation
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.rng = np.random.default_rng(seed)
        
        # 预生成所有层的权重（确保每次运行可复现）
        self.layers = []
        dim_ff = dim * ff_mult
        for _ in range(n_layers):
            W1 = random_weight_matrix(dim, dim_ff, rng=self.rng)
            b1 = self.rng.standard_normal(dim_ff) * 0.01
            W2 = random_weight_matrix(dim_ff, dim, rng=self.rng)
            b2 = self.rng.standard_normal(dim) * 0.01
            self.layers.append((W1, b1, W2, b2))
    
    def forward_one_layer(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """单层前向传播: LN(x + FFN(x))"""
        W1, b1, W2, b2 = self.layers[layer_idx]
        h = activation_fn(x @ W1 + b1, name=self.activation)
        ffn_out = h @ W2 + b2
        
        if self.use_residual:
            x = x + ffn_out
        else:
            x = ffn_out
        
        if self.use_layernorm:
            x = layer_norm(x)
        
        return x
    
    def propagate_additive(self, pe_encoding: np.ndarray,
                           n_layers: int = None) -> list:
        """
        加性 PE 传播：PE 只在第 0 层加到 embedding 上。
        
        Args:
            pe_encoding: [N, dim] 位置编码
            n_layers: 传播层数（默认全部）
        Returns:
            list of [N, dim] — 每层的输出（含第 0 层）
        """
        if n_layers is None:
            n_layers = self.n_layers
        
        # 初始 embedding = 随机 token embed + PE
        token_embed = self.rng.standard_normal(pe_encoding.shape) * 0.1
        x = token_embed + pe_encoding  # 加性注入
        
        trajectory = [x.copy()]
        for l in range(min(n_layers, self.n_layers)):
            x = self.forward_one_layer(x, l)
            trajectory.append(x.copy())
        
        return trajectory
    
    def propagate_rope(self, rope_pe: RoPE, positions: np.ndarray,
                       n_layers: int = None) -> list:
        """
        RoPE 乘性传播：每层独立施加旋转。
        
        在实际 Transformer 中，RoPE 作用在 q/k 上。
        这里我们模拟的是"RoPE 每层重新施加旋转后的等效信号"。
        
        Args:
            rope_pe: RoPE 实例
            positions: [N,] 位置序列
            n_layers: 传播层数
        Returns:
            list of [N, dim] — 每层经 RoPE 旋转后的输出
        """
        if n_layers is None:
            n_layers = self.n_layers
        
        # 初始 embedding（无位置信息）
        token_embed = self.rng.standard_normal(
            (len(positions), self.dim)) * 0.1
        x = token_embed.copy()
        
        # 第 0 层：施加 RoPE 旋转
        x_rotated = rope_pe.apply_rotary(x, positions)
        trajectory = [x_rotated.copy()]
        
        for l in range(min(n_layers, self.n_layers)):
            x = self.forward_one_layer(x, l)
            # 每层重新施加 RoPE 旋转
            x_rotated = rope_pe.apply_rotary(x, positions)
            trajectory.append(x_rotated.copy())
        
        return trajectory
    
    def propagate_alibi(self, positions: np.ndarray,
                        n_layers: int = None) -> list:
        """
        ALiBi 传播：每层的 attention bias 独立施加。
        ALiBi 不修改 embedding，这里只跟踪原始 embedding 的演化。
        """
        if n_layers is None:
            n_layers = self.n_layers
        
        token_embed = self.rng.standard_normal(
            (len(positions), self.dim)) * 0.1
        x = token_embed.copy()
        
        trajectory = [x.copy()]
        for l in range(min(n_layers, self.n_layers)):
            x = self.forward_one_layer(x, l)
            trajectory.append(x.copy())
        
        return trajectory


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_position_distinguishability(embeddings: np.ndarray) -> float:
    """
    计算位置区分度：相邻位置 embedding 差异与整体差异的比值。
    
    D = mean(||e_i - e_{i+1}||) / std(flatten(E))
    
    高 D → 位置间有清晰区分
    低 D → 位置混成一团
    """
    diffs = np.diff(embeddings, axis=0)
    mean_diff = np.mean(np.linalg.norm(diffs, axis=1))
    overall_std = np.std(embeddings)
    return float(mean_diff / (overall_std + 1e-15))


def compute_kernel_preservation(K_original: np.ndarray, 
                                 K_current: np.ndarray) -> float:
    """
    核函数保持度：当前层的核矩阵与原始核矩阵的 Frobenius 相似度。
    
    Preservation = 1 - ||K_orig - K_curr||_F / (||K_orig||_F + ε)
    
    值域 [0, 1]，1 = 完美保持，0 = 完全退化
    """
    diff = np.linalg.norm(K_original - K_current, 'fro')
    norm = np.linalg.norm(K_original, 'fro')
    return float(max(0, 1.0 - diff / (norm + 1e-15)))


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """计算位置 embedding 的成对距离矩阵"""
    return squareform(pdist(embeddings, metric='euclidean'))


def compute_rank_correlation(K1: np.ndarray, K2: np.ndarray) -> float:
    """
    Spearman 秩相关系数：衡量两个核矩阵的排序保持度。
    """
    from scipy.stats import spearmanr
    v1 = K1[np.triu_indices_from(K1, k=1)]
    v2 = K2[np.triu_indices_from(K2, k=1)]
    if np.std(v1) < 1e-15 or np.std(v2) < 1e-15:
        return 0.0
    corr, _ = spearmanr(v1, v2)
    return float(corr) if not np.isnan(corr) else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 位置信号逐层退化可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_signal_degradation(trajectories: dict, positions: np.ndarray,
                            config: PEConfig, layers_to_show: list = None):
    """
    可视化位置编码信号在逐层传播后的退化过程。
    
    展示选定维度上的信号波形如何随层数变化。
    对于 Sinusoidal PE，初始是清晰的正弦波，深层应退化为噪声。
    对于 RoPE，由于每层重新施加旋转，应保持周期结构。
    """
    if layers_to_show is None:
        layers_to_show = [0, 3, 6, 12, 18, 23]
    
    n_show = len(layers_to_show)
    pe_names = list(trajectories.keys())
    n_pe = len(pe_names)
    
    setup_plot_style()
    fig, axes = plt.subplots(n_pe, n_show, figsize=(4 * n_show, 3.5 * n_pe))
    fig.suptitle("Position Signal Degradation Through Layers\n"
                 "(Dimension 0 — highest frequency component)",
                 fontsize=16, fontweight='bold', y=1.02)
    
    dim_to_show = 0  # 最高频维度
    
    for row, name in enumerate(pe_names):
        traj = trajectories[name]
        color = get_pe_color(name)
        
        for col, layer in enumerate(layers_to_show):
            if layer >= len(traj):
                layer = len(traj) - 1
            ax = axes[row, col] if n_pe > 1 else axes[col]
            
            signal = traj[layer][:, dim_to_show]
            ax.plot(positions[:len(signal)], signal, color=color, linewidth=1.0, alpha=0.8)
            
            if col == 0:
                ax.set_ylabel(f"{name}", fontsize=11, fontweight='bold')
            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=11)
            if row == n_pe - 1:
                ax.set_xlabel("Position")
            
            # 标注信号统计量
            std_val = np.std(signal)
            ax.text(0.95, 0.95, f"σ={std_val:.3f}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    save_figure(fig, "03_signal_degradation", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. 位置区分度 & 核保持度 — 逐层衰减曲线
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_distinguishability_decay(trajectories: dict, pe_dict: dict,
                                  positions: np.ndarray):
    """
    绘制两个关键指标随层数的衰减曲线：
    
    Panel 1: 位置区分度 D(l)
        D(l) = mean(||e_i^l - e_{i+1}^l||) / std(E^l)
        衡量相邻位置在第 l 层是否仍可区分
    
    Panel 2: 核函数保持度 P(l) 
        P(l) = 1 - ||K_0 - K_l||_F / ||K_0||_F
        衡量第 l 层的距离结构是否保持原始 PE 的模式
    
    Panel 3: Spearman 秩相关
        衡量位置排序关系是否被保持
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    results = {}
    
    for name, traj in trajectories.items():
        color = get_pe_color(name)
        n_layers = len(traj)
        layers = np.arange(n_layers)
        
        # 位置区分度
        dist_scores = [compute_position_distinguishability(t) for t in traj]
        
        # 核矩阵保持度
        K0 = traj[0] @ traj[0].T  # 初始核矩阵
        K0 /= (np.linalg.norm(K0, 'fro') + 1e-15)
        preservation = []
        rank_corrs = []
        for t in traj:
            Kl = t @ t.T
            Kl /= (np.linalg.norm(Kl, 'fro') + 1e-15)
            preservation.append(compute_kernel_preservation(K0, Kl))
            rank_corrs.append(compute_rank_correlation(K0, Kl))
        
        results[name] = {
            'distinguishability': dist_scores,
            'preservation': preservation,
            'rank_correlation': rank_corrs,
        }
        
        # Panel 1
        axes[0].plot(layers, dist_scores, 'o-', color=color, linewidth=2,
                     markersize=3, label=name, alpha=0.85)
        # Panel 2
        axes[1].plot(layers, preservation, 's-', color=color, linewidth=2,
                     markersize=3, label=name, alpha=0.85)
        # Panel 3
        axes[2].plot(layers, rank_corrs, '^-', color=color, linewidth=2,
                     markersize=3, label=name, alpha=0.85)
    
    axes[0].set_xlabel("Layer $l$")
    axes[0].set_ylabel("Position Distinguishability $D(l)$")
    axes[0].set_title("Position Distinguishability Decay", fontsize=13, fontweight='bold')
    axes[0].legend()
    add_math_annotation(axes[0],
        r"$D(l) = \frac{\langle\|e_i^l - e_{i+1}^l\|\rangle}{\mathrm{std}(E^l)}$",
        loc='upper right', fontsize=9)
    
    axes[1].set_xlabel("Layer $l$")
    axes[1].set_ylabel("Kernel Preservation $P(l)$")
    axes[1].set_title("Kernel Structure Preservation", fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].set_ylim(-0.1, 1.1)
    
    axes[2].set_xlabel("Layer $l$")
    axes[2].set_ylabel("Spearman $\\rho(l)$")
    axes[2].set_title("Distance Rank Correlation", fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].set_ylim(-0.2, 1.1)
    axes[2].axhline(y=0, color='gray', ls='--', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "03_distinguishability_decay", MODULE)
    plt.close(fig)
    
    # Plotly 交互式
    if HAS_PLOTLY:
        pfig = make_subplots(rows=1, cols=3,
            subplot_titles=["Position Distinguishability",
                           "Kernel Preservation",
                           "Rank Correlation"])
        for name in trajectories:
            color = get_pe_color(name)
            r = results[name]
            layers = list(range(len(r['distinguishability'])))
            pfig.add_trace(go.Scatter(
                x=layers, y=r['distinguishability'],
                mode='lines+markers', name=name,
                line=dict(color=color, width=2),
                legendgroup=name,
            ), row=1, col=1)
            pfig.add_trace(go.Scatter(
                x=layers, y=r['preservation'],
                mode='lines+markers', name=name + ' (pres)',
                line=dict(color=color, width=2, dash='dot'),
                legendgroup=name, showlegend=False,
            ), row=1, col=2)
            pfig.add_trace(go.Scatter(
                x=layers, y=r['rank_correlation'],
                mode='lines+markers', name=name + ' (rank)',
                line=dict(color=color, width=2, dash='dash'),
                legendgroup=name, showlegend=False,
            ), row=1, col=3)
        pfig.update_layout(template="plotly_white", width=1200, height=450,
                          title="Position Information Decay Through Layers")
        save_plotly_html(pfig, "03_distinguishability_decay.html", MODULE)
    
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. Lyapunov 指数分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_lyapunov_analysis(trajectories: dict, positions: np.ndarray):
    """
    计算每种 PE 方案在逐层传播中的 Lyapunov 指数。
    
    Lyapunov 指数 λ 衡量相邻位置轨迹的指数发散率：
        ||δx(l)|| ~ ||δx(0)|| · e^{λl}
    
    - λ > 0: 混沌（位置信号发散）
    - λ ≈ 0: 边缘稳定
    - λ < 0: 收敛（位置信号坍缩）
    
    方法：
        选取多组相邻位置对 (p, p+1)，跟踪它们在各层的 embedding 距离。
        λ ≈ (1/L) Σ_l log(||x^l_{p+1} - x^l_p|| / ||x^{l-1}_{p+1} - x^{l-1}_p||)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Lyapunov Exponent Analysis — Chaos Detection",
                 fontsize=16, fontweight='bold')
    
    lyapunov_results = {}
    
    for idx, (name, traj) in enumerate(trajectories.items()):
        ax = axes[idx // 2, idx % 2]
        color = get_pe_color(name)
        n_layers = len(traj)
        
        # 选取多对相邻位置
        n_pairs = min(20, len(positions) - 1)
        pair_indices = np.linspace(0, len(positions) - 2, n_pairs, dtype=int)
        
        all_lyapunovs = []
        
        for pi in pair_indices:
            # 逐层距离
            distances = []
            for l in range(n_layers):
                d = np.linalg.norm(traj[l][pi] - traj[l][pi + 1])
                distances.append(d)
            distances = np.array(distances)
            distances = np.maximum(distances, 1e-15)
            
            # 累积 Lyapunov
            log_dists = np.log(distances)
            cumulative_lyap = np.diff(log_dists)
            all_lyapunovs.append(cumulative_lyap)
            
            # 绘制距离演化 (半透明)
            ax.semilogy(range(n_layers), distances, color=color, alpha=0.15, linewidth=0.5)
        
        # 平均 Lyapunov
        all_lyap = np.array(all_lyapunovs)
        mean_lyap = np.mean(all_lyap, axis=0)
        overall_lyap = np.mean(mean_lyap)
        lyapunov_results[name] = {
            'overall': overall_lyap,
            'per_layer': mean_lyap,
            'distances': distances,
        }
        
        # 均值线
        mean_dists = np.mean([np.maximum(
            [np.linalg.norm(traj[l][pi] - traj[l][pi+1]) for l in range(n_layers)], 1e-15)
            for pi in pair_indices], axis=0)
        ax.semilogy(range(n_layers), mean_dists, color=color, linewidth=2.5, label='Mean distance')
        
        ax.set_xlabel("Layer $l$")
        ax.set_ylabel("$\\|\\delta x(l)\\|$")
        ax.set_title(f"{name}  —  $\\lambda \\approx$ {overall_lyap:.4f}", fontsize=12)
        ax.legend(fontsize=8)
        
        # 标注混沌/稳定
        if overall_lyap > 0.01:
            ax.text(0.5, 0.02, "CHAOTIC (lambda > 0)", transform=ax.transAxes,
                    ha='center', fontsize=10, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
        elif overall_lyap < -0.01:
            ax.text(0.5, 0.02, "CONVERGENT (lambda < 0)", transform=ax.transAxes,
                    ha='center', fontsize=10, color='green', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='honeydew'))
        else:
            ax.text(0.5, 0.02, "MARGINAL (lambda ~ 0)", transform=ax.transAxes,
                    ha='center', fontsize=10, color='orange', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    save_figure(fig, "03_lyapunov_analysis", MODULE)
    plt.close(fig)
    
    # Plotly
    if HAS_PLOTLY:
        pfig = go.Figure()
        for name in trajectories:
            color = get_pe_color(name)
            r = lyapunov_results[name]
            layers = list(range(len(r['per_layer'])))
            pfig.add_trace(go.Scatter(
                x=layers, y=r['per_layer'].tolist(),
                mode='lines+markers', name=f"{name} (λ={r['overall']:.4f})",
                line=dict(color=color, width=2),
            ))
        pfig.update_layout(
            title="Per-Layer Lyapunov Exponent",
            xaxis_title="Layer", yaxis_title="Δlog(||δx||)",
            template="plotly_white", width=900, height=500,
        )
        pfig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        save_plotly_html(pfig, "03_lyapunov.html", MODULE)
    
    return lyapunov_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. 相空间重构 (Phase Space Portrait)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_phase_space(trajectories: dict, positions: np.ndarray):
    """
    使用 Takens 延时嵌入定理在相空间中重构位置编码信号的几何结构。
    
    Takens 定理：
        对于一维观测信号 s(t)，构造延时向量：
        X(t) = [s(t), s(t+τ), s(t+2τ)]
        
        如果原系统维度为 d_A，则只要嵌入维度 m > 2d_A，
        重构的相空间与原始吸引子微分同胚。
    
    预期：
        - Sinusoidal (第 0 层): 清晰的椭圆/Lissajous 图案
        - Sinusoidal (深层): 退化为无结构的噪声团
        - RoPE (深层): 仍保持椭圆结构
    """
    layers_to_show = [0, 6, 12, 23]
    pe_names = list(trajectories.keys())
    n_pe = len(pe_names)
    n_layers = len(layers_to_show)
    
    setup_plot_style()
    fig, axes = plt.subplots(n_pe, n_layers, figsize=(4.5 * n_layers, 4 * n_pe))
    fig.suptitle("Phase Space Reconstruction (Takens Embedding)\n"
                 "Dimension 0, delay τ=1, embedding dim=2",
                 fontsize=16, fontweight='bold', y=1.02)
    
    delay = 1
    embed_dim = 2
    dim_to_analyze = 0
    
    for row, name in enumerate(pe_names):
        traj = trajectories[name]
        color = get_pe_color(name)
        
        for col, layer in enumerate(layers_to_show):
            if layer >= len(traj):
                layer = len(traj) - 1
            ax = axes[row, col] if n_pe > 1 else axes[col]
            
            signal_1d = traj[layer][:, dim_to_analyze]
            
            try:
                phase = compute_phase_space(signal_1d, delay=delay,
                                           embedding_dim=embed_dim)
                # 用位置索引着色
                scatter = ax.scatter(phase[:, 0], phase[:, 1],
                                     c=np.arange(len(phase)),
                                     cmap='viridis', s=8, alpha=0.6)
                ax.plot(phase[:, 0], phase[:, 1], color=color, alpha=0.2, linewidth=0.5)
            except ValueError:
                ax.text(0.5, 0.5, "Signal too short", transform=ax.transAxes,
                        ha='center', fontsize=9)
            
            if col == 0:
                ax.set_ylabel(f"{name}\n$s(t+\\tau)$", fontsize=10, fontweight='bold')
            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=11)
            if row == n_pe - 1:
                ax.set_xlabel("$s(t)$")
    
    plt.tight_layout()
    save_figure(fig, "03_phase_space", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. 谱熵演化 — 频率结构随层数的退化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_spectral_evolution(trajectories: dict, positions: np.ndarray):
    """
    跟踪每层输出的谱熵变化。
    
    对于加性 PE：
        初始层谱熵低（信号以少数离散频率为主），
        随着层数增加，FFN 的非线性混合导致谱熵上升（趋向白噪声）。
    
    对于 RoPE：
        由于每层重新施加旋转，频率结构应保持稳定，谱熵不显著上升。
    
    同时跟踪有效秩 (effective rank) 的变化。
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    evolution_data = {}
    
    for name, traj in trajectories.items():
        color = get_pe_color(name)
        n_layers = len(traj)
        layers = np.arange(n_layers)
        
        entropies = []
        e_ranks = []
        mean_psds = []
        
        for l in range(n_layers):
            emb = traj[l]  # [N, dim]
            
            # 谱熵：对每维做 FFT，取平均 PSD，计算谱熵
            fft_vals = np.fft.rfft(emb, axis=0)
            mean_psd = np.mean(np.abs(fft_vals) ** 2, axis=1)
            entropies.append(spectral_entropy(mean_psd))
            mean_psds.append(mean_psd)
            
            # 有效秩
            e_ranks.append(effective_rank(emb))
        
        evolution_data[name] = {
            'spectral_entropy': entropies,
            'effective_rank': e_ranks,
            'mean_psds': mean_psds,
        }
        
        # Panel 1: 谱熵
        axes[0].plot(layers, entropies, 'o-', color=color, linewidth=2,
                     markersize=3, label=name)
        
        # Panel 2: 有效秩
        axes[1].plot(layers, e_ranks, 's-', color=color, linewidth=2,
                     markersize=3, label=name)
    
    axes[0].set_xlabel("Layer $l$")
    axes[0].set_ylabel("Normalized Spectral Entropy $\\tilde{H}_s$")
    axes[0].set_title("Spectral Entropy Evolution", fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=1.0, color='gray', ls='--', alpha=0.3, label='White noise')
    add_math_annotation(axes[0],
        r"$\tilde{H}_s \to 1$: chaos" + "\n" + r"$\tilde{H}_s \to 0$: structured",
        loc='center right', fontsize=8)
    
    axes[1].set_xlabel("Layer $l$")
    axes[1].set_ylabel("Effective Rank")
    axes[1].set_title("Embedding Effective Rank Evolution", fontsize=13, fontweight='bold')
    axes[1].legend()
    
    # Panel 3: 选定层的 PSD 对比 (Sinusoidal vs RoPE)
    ax3 = axes[2]
    show_layers = [0, 6, 12, 23]
    cmap = plt.cm.coolwarm
    for name in ['sinusoidal', 'rope']:
        if name not in evolution_data:
            continue
        ls = '-' if name == 'sinusoidal' else '--'
        for i, l in enumerate(show_layers):
            if l >= len(evolution_data[name]['mean_psds']):
                continue
            psd = evolution_data[name]['mean_psds'][l]
            freqs = np.fft.rfftfreq(len(trajectories[name][l]))
            alpha = 1.0 - i * 0.2
            color_l = get_pe_color(name)
            ax3.semilogy(freqs[1:], np.maximum(psd[1:], 1e-30),
                        linestyle=ls, color=color_l, alpha=alpha,
                        linewidth=1.5, label=f'{name} L{l}')
    
    ax3.set_xlabel("Normalized frequency")
    ax3.set_ylabel("Mean PSD")
    ax3.set_title("PSD Evolution: Sinusoidal(—) vs RoPE(--)", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, ncol=2)
    
    plt.tight_layout()
    save_figure(fig, "03_spectral_evolution", MODULE)
    plt.close(fig)
    
    # Plotly 交互式
    if HAS_PLOTLY:
        pfig = make_subplots(rows=1, cols=2,
            subplot_titles=["Spectral Entropy Evolution",
                           "Effective Rank Evolution"])
        for name in trajectories:
            color = get_pe_color(name)
            r = evolution_data[name]
            layers = list(range(len(r['spectral_entropy'])))
            pfig.add_trace(go.Scatter(
                x=layers, y=r['spectral_entropy'],
                mode='lines+markers', name=name,
                line=dict(color=color, width=2), legendgroup=name,
            ), row=1, col=1)
            pfig.add_trace(go.Scatter(
                x=layers, y=r['effective_rank'],
                mode='lines+markers', name=name + ' (rank)',
                line=dict(color=color, width=2, dash='dot'),
                legendgroup=name, showlegend=False,
            ), row=1, col=2)
        pfig.update_layout(template="plotly_white", width=1100, height=450,
                          title="Spectral Structure Evolution Through Layers")
        save_plotly_html(pfig, "03_spectral_evolution.html", MODULE)
    
    return evolution_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. 距离矩阵热力图 — 核结构退化可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_distance_matrix_evolution(trajectories: dict, positions: np.ndarray):
    """
    可视化位置 embedding 的成对距离矩阵在逐层传播中的退化。
    
    对于一个好的 PE，距离矩阵应呈现平滑的 Toeplitz 结构
    （相同位置差的 pair 有相同距离）。
    随着混沌化，这个结构应被破坏。
    """
    layers_to_show = [0, 6, 12, 23]
    
    # 只取前 64 个位置（避免热力图过大）
    n_pos = min(64, len(positions))
    
    pe_names = list(trajectories.keys())
    n_pe = len(pe_names)
    n_layers = len(layers_to_show)
    
    setup_plot_style()
    fig, axes = plt.subplots(n_pe, n_layers, figsize=(4 * n_layers, 3.5 * n_pe))
    fig.suptitle("Distance Matrix Evolution — Kernel Structure Degradation",
                 fontsize=16, fontweight='bold', y=1.02)
    
    for row, name in enumerate(pe_names):
        traj = trajectories[name]
        
        for col, layer in enumerate(layers_to_show):
            if layer >= len(traj):
                layer = len(traj) - 1
            ax = axes[row, col] if n_pe > 1 else axes[col]
            
            emb = traj[layer][:n_pos]
            D = compute_distance_matrix(emb)
            
            # 归一化到 [0, 1]
            D_norm = D / (D.max() + 1e-15)
            
            im = ax.imshow(D_norm, cmap='RdYlBu_r', aspect='equal',
                          interpolation='nearest', vmin=0, vmax=1)
            
            if col == 0:
                ax.set_ylabel(f"{name}", fontsize=10, fontweight='bold')
            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=11)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 左下角标注 Toeplitz 偏差
            # Toeplitz 矩阵的每条对角线应该是常数
            toeplitz_err = 0.0
            for diag in range(1, min(10, n_pos)):
                diag_vals = np.diag(D_norm, diag)
                if len(diag_vals) > 1:
                    toeplitz_err += np.std(diag_vals)
            toeplitz_err /= min(10, n_pos)
            
            ax.text(0.05, 0.05, f"T-err: {toeplitz_err:.3f}",
                    transform=ax.transAxes, fontsize=7, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    plt.tight_layout()
    save_figure(fig, "03_distance_matrix_evolution", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. 综合混沌指标仪表盘
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_chaos_dashboard(lyapunov_results: dict, decay_results: dict,
                         evolution_data: dict, trajectories: dict):
    """
    绘制综合混沌指标仪表盘：
    
    - 最终 Lyapunov 指数柱状图
    - 最终谱熵柱状图  
    - 位置区分度保持率（末层/初层）
    - 核保持度末层值
    """
    names = list(lyapunov_results.keys())
    
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Chaos Hypothesis — Comprehensive Dashboard",
                 fontsize=18, fontweight='bold')
    
    colors = [get_pe_color(n) for n in names]
    x = np.arange(len(names))
    bar_width = 0.6
    
    # Panel 1: Lyapunov 指数
    ax1 = axes[0, 0]
    lyap_vals = [lyapunov_results[n]['overall'] for n in names]
    bars1 = ax1.bar(x, lyap_vals, bar_width, color=colors, edgecolor='black', alpha=0.85)
    ax1.axhline(y=0, color='red', ls='--', linewidth=1, alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("Lyapunov Exponent $\\lambda$")
    ax1.set_title("Mean Lyapunov Exponent\n($\\lambda > 0$ ⇒ Chaos)", fontsize=12)
    for bar, val in zip(bars1, lyap_vals):
        color_txt = 'red' if val > 0 else 'green'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=10, fontweight='bold', color=color_txt)
    
    # Panel 2: 最终谱熵
    ax2 = axes[0, 1]
    final_entropy = [evolution_data[n]['spectral_entropy'][-1] for n in names]
    init_entropy = [evolution_data[n]['spectral_entropy'][0] for n in names]
    
    x2 = np.arange(len(names))
    bars2a = ax2.bar(x2 - 0.15, init_entropy, 0.3, color=colors, alpha=0.4,
                     edgecolor='black', label='Layer 0')
    bars2b = ax2.bar(x2 + 0.15, final_entropy, 0.3, color=colors, alpha=0.9,
                     edgecolor='black', label='Final Layer')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel("Spectral Entropy $\\tilde{H}_s$")
    ax2.set_title("Spectral Entropy: Initial vs Final Layer", fontsize=12)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    for bar, val in zip(bars2b, final_entropy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Panel 3: 位置区分度保持率
    ax3 = axes[1, 0]
    d_init = [decay_results[n]['distinguishability'][0] for n in names]
    d_final = [decay_results[n]['distinguishability'][-1] for n in names]
    retention = [d_final[i] / (d_init[i] + 1e-15) for i in range(len(names))]
    
    bars3 = ax3.bar(x, retention, bar_width, color=colors, edgecolor='black', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=10)
    ax3.set_ylabel("Retention Ratio $D_{final}/D_{init}$")
    ax3.set_title("Position Distinguishability Retention", fontsize=12)
    for bar, val in zip(bars3, retention):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel 4: 核保持度末层值
    ax4 = axes[1, 1]
    final_pres = [decay_results[n]['preservation'][-1] for n in names]
    final_rank = [decay_results[n]['rank_correlation'][-1] for n in names]
    
    x4 = np.arange(len(names))
    bars4a = ax4.bar(x4 - 0.15, final_pres, 0.3, color=colors, alpha=0.7,
                     edgecolor='black', label='Kernel Preservation')
    bars4b = ax4.bar(x4 + 0.15, final_rank, 0.3, color=colors, alpha=0.9,
                     edgecolor='black', hatch='//', label='Rank Correlation')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(names, fontsize=10)
    ax4.set_ylabel("Score")
    ax4.set_title("Final Layer: Kernel Preservation & Rank Correlation", fontsize=12)
    ax4.legend(fontsize=8)
    ax4.set_ylim(-0.2, 1.2)
    
    plt.tight_layout()
    save_figure(fig, "03_chaos_dashboard", MODULE)
    plt.close(fig)
    
    # Plotly 仪表盘
    if HAS_PLOTLY:
        pfig = make_subplots(rows=2, cols=2,
            subplot_titles=["Lyapunov Exponent", "Spectral Entropy (Init vs Final)",
                           "Distinguishability Retention", "Kernel Preservation"])
        for i, n in enumerate(names):
            color = get_pe_color(n)
            pfig.add_trace(go.Bar(x=[n], y=[lyapunov_results[n]['overall']],
                marker_color=color, name=n, legendgroup=n,
                showlegend=(i==0 or True)), row=1, col=1)
            pfig.add_trace(go.Bar(x=[n], y=[final_entropy[i]],
                marker_color=color, name=n, legendgroup=n,
                showlegend=False), row=1, col=2)
            pfig.add_trace(go.Bar(x=[n], y=[retention[i]],
                marker_color=color, name=n, legendgroup=n,
                showlegend=False), row=2, col=1)
            pfig.add_trace(go.Bar(x=[n], y=[final_pres[i]],
                marker_color=color, name=n, legendgroup=n,
                showlegend=False), row=2, col=2)
        pfig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        pfig.update_layout(template="plotly_white", width=1000, height=700,
                          title="Chaos Hypothesis — Dashboard")
        save_plotly_html(pfig, "03_chaos_dashboard.html", MODULE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  综合报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_chaos_report():
    """生成综合 HTML 报告"""
    sections = [
        {
            'title': '核心假设 (The Chaos Hypothesis)',
            'content': (
                '<strong>假设</strong>：加性绝对位置编码（Sinusoidal PE）仅在第一层注入位置信息。'
                '随着前馈网络的层层传播（线性变换 + 非线性激活 + 残差连接），'
                '位置信号逐渐退化为混沌/噪声，丧失位置区分能力。'
                '<br><br>'
                '相反，乘性相对位置编码（RoPE）在<strong>每一层</strong>独立施加 SO(2) 旋转变换，'
                '因此能在深层保持位置信息的结构性。'
                '<br><br>'
                '<strong>数学机制</strong>：'
                '<ul>'
                '<li>加性 PE：\\(x_0 = \\text{Embed} + PE(p)\\)，之后 PE 作为初始扰动被 FFN 非线性混合</li>'
                '<li>乘性 PE：\\(q_l = R(p) \\cdot x_l\\)，每层独立旋转保证位置信号的幺正不变性</li>'
                '</ul>'
            )
        },
        {
            'title': '1. 位置信号逐层退化',
            'content': (
                '展示 PE 编码信号（最高频维度）在逐层传播后的波形变化。'
                '<br><br>对于 Sinusoidal PE，初始是清晰的正弦波，'
                '经过多层 FFN 后应退化为无结构的噪声。'
                'RoPE 由于每层重新施加旋转，应保持周期结构。'
            )
        },
        {
            'title': '2. 位置区分度 & 核保持度衰减',
            'content': (
                '两个关键定量指标：'
                '<br><br>'
                '\\[D(l) = \\frac{\\langle\\|e_i^l - e_{i+1}^l\\|\\rangle}{\\mathrm{std}(E^l)}\\]'
                '位置区分度衡量相邻位置在第 \\(l\\) 层是否仍可区分。'
                '<br><br>'
                '\\[P(l) = 1 - \\frac{\\|K_0 - K_l\\|_F}{\\|K_0\\|_F}\\]'
                '核保持度衡量距离结构是否保持原始 PE 的模式。'
            )
        },
        {
            'title': '3. Lyapunov 指数 — 混沌检测',
            'content': (
                'Lyapunov 指数 \\(\\lambda\\) 衡量相邻位置轨迹的指数发散率：'
                '\\[\\|\\delta x(l)\\| \\sim \\|\\delta x(0)\\| \\cdot e^{\\lambda l}\\]'
                '<ul>'
                '<li>\\(\\lambda > 0\\)：<span style="color:red">混沌</span>（位置信号发散）</li>'
                '<li>\\(\\lambda \\approx 0\\)：边缘稳定</li>'
                '<li>\\(\\lambda < 0\\)：<span style="color:green">收敛</span>（位置信号坍缩）</li>'
                '</ul>'
            )
        },
        {
            'title': '4. 相空间重构 (Takens 定理)',
            'content': (
                '使用延时嵌入定理在相空间中重构位置编码信号的几何结构。'
                '<br><br>'
                '构造延时向量：\\(X(t) = [s(t), s(t+\\tau)]\\)'
                '<br><br>'
                '预期：初始层呈现清晰的 Lissajous 图案（结构化），'
                '深层退化为无结构的噪声团（混沌化）。'
            )
        },
        {
            'title': '5. 谱熵演化',
            'content': (
                '跟踪每层输出的谱熵 \\(\\tilde{H}_s\\) 变化。'
                '<br><br>'
                '\\(\\tilde{H}_s \\to 1\\) 表示趋向白噪声（混沌），'
                '\\(\\tilde{H}_s\\) 保持低值表示频率结构得以保持。'
                '<br><br>'
                '同时跟踪有效秩 (effective rank)，衡量 embedding 空间的有效利用维度。'
            )
        },
        {
            'title': '6. 距离矩阵退化',
            'content': (
                '可视化位置 embedding 的成对距离矩阵在逐层传播中的退化。'
                '<br><br>'
                '一个好的 PE 应该让距离矩阵呈现平滑的 Toeplitz 结构'
                '（相同位置差的 pair 有相同距离）。'
                '随着混沌化，这个结构应被破坏。'
                '<br><br>'
                '用 Toeplitz 误差 (T-err) 量化：每条对角线元素的标准差之和。'
            )
        },
        {
            'title': '结论',
            'content': (
                '<strong>综合仪表盘</strong>通过四个维度验证混沌假设：'
                '<ol>'
                '<li>Lyapunov 指数 — 动力系统稳定性</li>'
                '<li>谱熵 — 频率结构退化</li>'
                '<li>位置区分度 — 功能性退化</li>'
                '<li>核保持度 — 几何结构退化</li>'
                '</ol>'
                '如果假设成立，加性 PE 应在所有指标上表现出显著退化，'
                '而 RoPE 由于每层独立的旋转注入，应保持结构完整性。'
            )
        },
    ]
    
    generate_report_html(
        title="03 — Chaos Propagation Analysis: The Chaos Hypothesis",
        sections=sections,
        module=MODULE,
        filename="03_chaos_propagation_report.html"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  03 — Chaos Propagation Analysis")
    print("  The Chaos Hypothesis: Additive PE → Chaos in Deep Layers")
    print("=" * 60)
    
    # 配置
    dim = 64
    seq_len = 128       # 位置序列长度
    n_layers = 24       # 模拟 24 层 Transformer
    seed = 42
    
    config = PEConfig(dim=dim, max_len=seq_len)
    positions = np.arange(seq_len)
    
    # 获取 PE 实例
    pe_dict = get_all_pe(config=config)
    rope_pe = pe_dict['rope']
    
    # 日志
    logger = VizLogger("pe_analysis", "03_chaos_propagation")
    
    # ── 初始化模拟器 ──
    print("\n  [0/7] Initializing Transformer simulator...")
    print(f"        dim={dim}, layers={n_layers}, seq_len={seq_len}, seed={seed}")
    sim = TransformerSimulator(dim=dim, n_layers=n_layers, seed=seed)
    
    # ── 生成各 PE 方案的传播轨迹 ──
    print("\n  [0/7] Generating propagation trajectories...")
    trajectories = {}
    
    # Sinusoidal PE (加性)
    print("        → Sinusoidal PE (additive)...")
    sin_pe = pe_dict['sinusoidal']
    sin_enc = sin_pe.encode(positions)
    sim_sin = TransformerSimulator(dim=dim, n_layers=n_layers, seed=seed)
    trajectories['sinusoidal'] = sim_sin.propagate_additive(sin_enc)
    
    # RoPE (乘性)
    print("        → RoPE (multiplicative)...")
    sim_rope = TransformerSimulator(dim=dim, n_layers=n_layers, seed=seed)
    trajectories['rope'] = sim_rope.propagate_rope(rope_pe, positions)
    
    # ALiBi (无 embedding PE)
    print("        → ALiBi (bias-only, no embedding PE)...")
    sim_alibi = TransformerSimulator(dim=dim, n_layers=n_layers, seed=seed)
    trajectories['alibi'] = sim_alibi.propagate_alibi(positions)
    
    # LAPE (加性)
    print("        → LAPE (additive)...")
    lape_pe = pe_dict['lape']
    lape_enc = lape_pe.encode(positions)
    sim_lape = TransformerSimulator(dim=dim, n_layers=n_layers, seed=seed)
    trajectories['lape'] = sim_lape.propagate_additive(lape_enc)
    
    print(f"        ✓ Generated {len(trajectories)} trajectories, "
          f"each with {n_layers + 1} layers")
    
    # ── 1. 信号退化可视化 ──
    print("\n  [1/7] Signal degradation visualization...")
    plot_signal_degradation(trajectories, positions, config,
                           layers_to_show=[0, 3, 6, 12, 18, 23])
    print("  ✓ Saved 03_signal_degradation")
    logger.log_metric("signal_degradation", "completed")
    
    # ── 2. 区分度衰减曲线 ──
    print("\n  [2/7] Distinguishability & kernel preservation decay...")
    decay_results = plot_distinguishability_decay(trajectories, pe_dict, positions)
    print("  ✓ Saved 03_distinguishability_decay")
    for name in trajectories:
        logger.log_metric(f"final_distinguishability_{name}",
                         decay_results[name]['distinguishability'][-1])
        logger.log_metric(f"final_preservation_{name}",
                         decay_results[name]['preservation'][-1])
    
    # ── 3. Lyapunov 指数 ──
    print("\n  [3/7] Lyapunov exponent analysis...")
    lyapunov_results = plot_lyapunov_analysis(trajectories, positions)
    print("  ✓ Saved 03_lyapunov_analysis")
    for name, r in lyapunov_results.items():
        logger.log_metric(f"lyapunov_{name}", r['overall'])
        status = "CHAOTIC" if r['overall'] > 0.01 else (
            "CONVERGENT" if r['overall'] < -0.01 else "MARGINAL")
        print(f"        {name}: λ = {r['overall']:.6f} ({status})")
    
    # ── 4. 相空间重构 ──
    print("\n  [4/7] Phase space reconstruction...")
    plot_phase_space(trajectories, positions)
    print("  ✓ Saved 03_phase_space")
    logger.log_metric("phase_space", "completed")
    
    # ── 5. 谱熵演化 ──
    print("\n  [5/7] Spectral entropy evolution...")
    evolution_data = plot_spectral_evolution(trajectories, positions)
    print("  ✓ Saved 03_spectral_evolution")
    for name in trajectories:
        logger.log_metric(f"final_spectral_entropy_{name}",
                         evolution_data[name]['spectral_entropy'][-1])
        logger.log_metric(f"final_effective_rank_{name}",
                         evolution_data[name]['effective_rank'][-1])
    
    # ── 6. 距离矩阵退化 ──
    print("\n  [6/7] Distance matrix evolution...")
    plot_distance_matrix_evolution(trajectories, positions)
    print("  ✓ Saved 03_distance_matrix_evolution")
    logger.log_metric("distance_matrix", "completed")
    
    # ── 7. 综合仪表盘 ──
    print("\n  [7/7] Chaos hypothesis dashboard...")
    plot_chaos_dashboard(lyapunov_results, decay_results,
                        evolution_data, trajectories)
    print("  ✓ Saved 03_chaos_dashboard")
    
    # ── 报告 ──
    print("\n  Generating HTML report...")
    generate_chaos_report()
    
    # 保存日志
    logger.save()
    
    # ── 总结 ──
    print("\n" + "=" * 60)
    print("  ✅ 03_chaos_propagation 完成！")
    print("  📊 静态图: output/pe_analysis/03_*")
    print("  🌐 交互图: html/pe_analysis/03_*")
    print("\n  ── 混沌假设验证结果 ──")
    for name in trajectories:
        lyap = lyapunov_results[name]['overall']
        entropy_init = evolution_data[name]['spectral_entropy'][0]
        entropy_final = evolution_data[name]['spectral_entropy'][-1]
        dist_retention = (decay_results[name]['distinguishability'][-1] /
                         (decay_results[name]['distinguishability'][0] + 1e-15))
        print(f"  {name:15s}: λ={lyap:+.4f}  "
              f"H_s: {entropy_init:.3f}→{entropy_final:.3f}  "
              f"D_retention: {dist_retention:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
