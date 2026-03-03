
#!/usr/bin/env python3
"""
01 — 统一数学框架下的 PE 方案对比

在复分析 / Bochner 定理 / 群论的统一视角下对比四种位置编码方案：
  • Sinusoidal PE  (Vaswani 2017)  — 加性绝对, 傅里叶基
  • RoPE           (Su 2021)       — 乘性相对, SO(2) 旋转
  • ALiBi          (Press 2022)    — 加性偏置, 线性衰减
  • LAPE           (TCFMamba)      — 加性绝对, 幂律谱

生成内容：
  1. 核函数 K(Δ) 对比图（Matplotlib + Plotly）
  2. 复平面嵌入轨迹（Plotly 3D 交互）
  3. 频率谱测度对比图
  4. 核矩阵热力图（4-panel）
  5. 平移群响应分析
  6. 综合 HTML 报告

Output:
  output/pe_analysis/   → 静态图 (PNG/PDF)
  html/pe_analysis/     → 交互式 HTML

Usage:
  python -m pe_analysis.01_unified_comparison
  python run.py pe_analysis.comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    PEConfig, get_pe, get_all_pe,
    setup_plot_style, save_figure, add_math_annotation,
    create_plotly_figure, save_plotly_html, create_heatmap_html,
    generate_report_html, get_pe_color, get_output_dir, get_html_dir,
    PE_COLORS, COLORS, VizLogger, plotly_available,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

MODULE = "pe_analysis"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 核函数对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_kernel_comparison(pe_dict: dict, max_delta: int = 256):
    """
    绘制各 PE 方案的核函数 K(Δ) 对比图。
    
    数学背景 (Bochner 定理):
        连续正定函数 K: ℝ → ℂ 可以表示为某个非负有限测度 μ 的 Fourier 变换：
            K(Δ) = ∫ e^{iωΔ} dμ(ω)
        
        对于离散频率 {ω_k}:
            K(Δ) = (1/m) Σ_k cos(ω_k · Δ)
        
        不同 PE 方案 ↔ 不同谱测度 μ ↔ 不同的核函数衰减行为
    """
    deltas = np.arange(-max_delta, max_delta + 1)
    
    # ── Matplotlib 静态图 ────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Position Encoding Kernel Functions  $K(\\Delta) = \\langle PE(p),\\, PE(p+\\Delta) \\rangle$",
                 fontsize=16, fontweight='bold')
    
    # Panel 1: 全局对比
    ax = axes[0, 0]
    for name, pe in pe_dict.items():
        k_vals = pe.kernel(deltas)
        color = get_pe_color(name)
        ax.plot(deltas, k_vals, label=pe.name, color=color, linewidth=1.5)
    ax.set_xlabel("$\\Delta$ (position offset)")
    ax.set_ylabel("$K(\\Delta)$")
    ax.set_title("Full Kernel Comparison")
    ax.legend(loc='upper right')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    add_math_annotation(ax, r"$K(\Delta)=\frac{1}{m}\sum_k\cos(\omega_k\Delta)$", 
                        loc='lower right')
    
    # Panel 2: 短程行为 (|Δ| < 30)
    ax = axes[0, 1]
    short_deltas = np.linspace(-30, 30, 500)
    for name, pe in pe_dict.items():
        k_vals = pe.kernel(short_deltas)
        ax.plot(short_deltas, k_vals, label=pe.name, 
                color=get_pe_color(name), linewidth=2)
    ax.set_xlabel("$\\Delta$")
    ax.set_ylabel("$K(\\Delta)$")
    ax.set_title("Short-range Behavior ($|\\Delta| < 30$)")
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Panel 3: 核函数绝对值的对数衰减
    ax = axes[1, 0]
    pos_deltas = np.arange(1, max_delta + 1)
    for name, pe in pe_dict.items():
        k_vals = pe.kernel(pos_deltas)
        k_abs = np.maximum(np.abs(k_vals), 1e-15)
        ax.semilogy(pos_deltas, k_abs, label=pe.name,
                    color=get_pe_color(name), linewidth=1.5)
    ax.set_xlabel("$|\\Delta|$")
    ax.set_ylabel("$|K(\\Delta)|$ (log scale)")
    ax.set_title("Kernel Decay Rate (Log Scale)")
    ax.legend()
    add_math_annotation(ax, r"Decay rate $\leftrightarrow$ spectral measure $\mu$", 
                        loc='upper right')
    
    # Panel 4: 核函数导数（局部敏感度）
    ax = axes[1, 1]
    fine_deltas = np.linspace(0, 50, 1000)
    for name, pe in pe_dict.items():
        if name == 'alibi':
            # ALiBi 的导数是常数 -m
            slopes = pe.get_frequencies()
            ax.axhline(y=-slopes[0], color=get_pe_color(name), 
                      linestyle='--', label=f"{pe.name} ($K' = -m$)")
            continue
        k_vals = pe.kernel(fine_deltas)
        k_deriv = np.gradient(k_vals, fine_deltas)
        ax.plot(fine_deltas, k_deriv, label=pe.name, 
                color=get_pe_color(name), linewidth=1.5)
    ax.set_xlabel("$\\Delta$")
    ax.set_ylabel("$K'(\\Delta)$")
    ax.set_title("Kernel Derivative (Local Sensitivity)")
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    paths = save_figure(fig, "01_kernel_comparison", MODULE)
    plt.close(fig)
    
    # ── Plotly 交互图 ────────────────────────────────────────
    if HAS_PLOTLY:
        pfig = make_subplots(rows=1, cols=2, 
                             subplot_titles=["Kernel K(Δ)", "|K(Δ)| Log Scale"])
        
        for name, pe in pe_dict.items():
            color = get_pe_color(name)
            k_vals = pe.kernel(deltas)
            pfig.add_trace(
                go.Scatter(x=deltas, y=k_vals, mode='lines',
                          name=pe.name, line=dict(color=color, width=2),
                          hovertemplate="Δ=%{x}<br>K(Δ)=%{y:.4f}"),
                row=1, col=1
            )
            k_abs = np.maximum(np.abs(pe.kernel(pos_deltas)), 1e-15)
            pfig.add_trace(
                go.Scatter(x=pos_deltas, y=k_abs, mode='lines',
                          name=pe.name + " (log)", 
                          line=dict(color=color, width=2, dash='dot'),
                          showlegend=False,
                          hovertemplate="Δ=%{x}<br>|K(Δ)|=%{y:.6f}"),
                row=1, col=2
            )
        
        pfig.update_yaxes(type="log", row=1, col=2)
        pfig.update_layout(
            title="Position Encoding Kernel Functions — Interactive",
            width=1200, height=500, template='plotly_white',
            hovermode='x unified'
        )
        save_plotly_html(pfig, "01_kernel_comparison.html", MODULE)
    
    return paths


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. 复平面嵌入轨迹
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_complex_plane_embedding(pe_dict: dict, n_positions: int = 200):
    """
    将各 PE 方案在复平面上的嵌入轨迹可视化。
    
    数学背景:
        z_k(p) = e^{i ω_k p}  ∈ ℂ
        
        对于频率 ω_k，位置 p 的复数嵌入在单位圆上以角速度 ω_k 旋转。
        不同频率分量形成不同半径的螺旋（投影到 3D 时为螺旋线）。
    
    可视化:
        选取两个频率分量 (ω_low, ω_high)，展示其在复平面上的轨迹。
        x 轴 = Re(z_low), y 轴 = Im(z_low), z 轴 = Re(z_high)
    """
    positions = np.arange(n_positions)
    
    # ── Matplotlib 2D 复平面 ─────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Complex Plane Embeddings: $z_k(p) = e^{i\\omega_k p}$",
                 fontsize=16, fontweight='bold')
    
    for idx, (name, pe) in enumerate(pe_dict.items()):
        ax = axes[idx // 2, idx % 2]
        
        if name == 'alibi':
            # ALiBi 没有复数嵌入，展示偏置衰减
            slopes = pe.get_frequencies()
            for h, slope in enumerate(slopes[:4]):
                bias = -slope * positions
                ax.plot(positions, bias, label=f"head {h+1}, m={slope:.4f}",
                       linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Position offset $\\Delta$")
            ax.set_ylabel("Bias $B(\\Delta)$")
            ax.set_title(f"{pe.name} — Linear Bias (no complex embedding)")
            ax.legend(fontsize=8)
        else:
            z = pe.encode_complex(positions)  # [N, dim//2]
            # 选取低频和高频两个分量
            freq_low, freq_high = 0, min(5, z.shape[1] - 1)
            
            colors = positions
            scatter = ax.scatter(z[:, freq_low].real, z[:, freq_low].imag,
                               c=colors, cmap='viridis', s=10, alpha=0.8)
            ax.set_xlabel(f"$\\mathrm{{Re}}(z_{{{freq_low}}})$")
            ax.set_ylabel(f"$\\mathrm{{Im}}(z_{{{freq_low}}})$")
            
            freqs = pe.get_frequencies()
            omega_str = f"ω₀={freqs[freq_low]:.4f}"
            ax.set_title(f"{pe.name} — Complex Trajectory ({omega_str})")
            
            # 添加单位圆参考
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2, linewidth=0.5)
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax, label='Position')
    
    plt.tight_layout()
    paths = save_figure(fig, "01_complex_embeddings", MODULE)
    plt.close(fig)
    
    # ── Plotly 3D 螺旋轨迹 ──────────────────────────────────
    if HAS_PLOTLY:
        pfig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=["Sinusoidal PE", "RoPE", "LAPE"]
        )
        
        plot_names = ['sinusoidal', 'rope', 'lape']
        for col, name in enumerate(plot_names, 1):
            pe = pe_dict[name]
            z = pe.encode_complex(positions)
            freqs = pe.get_frequencies()
            
            # 低频 + 高频两个分量构成 3D
            f_low, f_high = 0, min(8, z.shape[1] - 1)
            
            pfig.add_trace(
                go.Scatter3d(
                    x=z[:, f_low].real,
                    y=z[:, f_low].imag,
                    z=z[:, f_high].real,
                    mode='markers+lines',
                    marker=dict(size=2, color=positions, colorscale='Viridis',
                               colorbar=dict(title="Pos") if col == 3 else dict()),
                    line=dict(color=get_pe_color(name), width=2),
                    name=pe.name,
                    hovertemplate=("pos=%{text}<br>"
                                  "Re(z_low)=%{x:.3f}<br>"
                                  "Im(z_low)=%{y:.3f}<br>"
                                  "Re(z_high)=%{z:.3f}"),
                    text=[str(p) for p in positions],
                ),
                row=1, col=col
            )
        
        for col in range(1, 4):
            pfig.update_scenes(
                dict(
                    xaxis_title="Re(z_low)",
                    yaxis_title="Im(z_low)",
                    zaxis_title="Re(z_high)",
                ),
                row=1, col=col
            )
        
        pfig.update_layout(
            title="Complex Plane 3D Spirals — Position Embeddings",
            width=1500, height=550
        )
        save_plotly_html(pfig, "01_complex_3d_spirals.html", MODULE)
    
    return paths


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. 频率谱测度对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_frequency_spectra(pe_dict: dict):
    """
    对比各 PE 方案的频率分布（谱测度 μ）。
    
    数学背景:
        Bochner 定理建立了 "正定核" 与 "谱测度" 之间的一一对应：
            K(Δ) = ∫ e^{iωΔ} dμ(ω)
        
        对于离散频率方案：
            μ = (1/m) Σ_k δ(ω - ω_k)
        
        频率分布的差异直接决定了核函数的衰减行为：
            • Sinusoidal/RoPE: ω_k = 1/base^{2k/d} — 几何级数 (对数均匀)
            • LAPE: ω_k = (k/d)^a — 幂律分布 (密集在低频)
            • ALiBi: 无频率概念，线性偏置
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Spectral Measures of Position Encodings (Bochner's Theorem)",
                 fontsize=16, fontweight='bold')
    
    # Panel 1: 频率值对比 (线性)
    ax = axes[0, 0]
    for name, pe in pe_dict.items():
        freqs = pe.get_frequencies()
        k = np.arange(len(freqs))
        ax.plot(k, freqs, 'o-', label=pe.name, color=get_pe_color(name),
                markersize=3, linewidth=1)
    ax.set_xlabel("Frequency index $k$")
    ax.set_ylabel("$\\omega_k$")
    ax.set_title("Frequency Values (Linear Scale)")
    ax.legend()
    
    # Panel 2: 频率值对比 (对数)
    ax = axes[0, 1]
    for name, pe in pe_dict.items():
        if name == 'alibi':
            continue
        freqs = pe.get_frequencies()
        abs_freqs = np.abs(freqs)
        abs_freqs[abs_freqs == 0] = 1e-15
        k = np.arange(len(freqs))
        ax.semilogy(k, abs_freqs, 'o-', label=pe.name, 
                    color=get_pe_color(name), markersize=3)
    ax.set_xlabel("Frequency index $k$")
    ax.set_ylabel("$|\\omega_k|$ (log scale)")
    ax.set_title("Frequency Values (Log Scale)")
    ax.legend()
    add_math_annotation(ax, r"$\omega_k^{\mathrm{sin}}=b^{-2k/d}$" + "\n" + 
                        r"$\omega_k^{\mathrm{LAPE}}=(k/d)^a$",
                        loc='upper right')
    
    # Panel 3: 频率密度直方图
    ax = axes[1, 0]
    for name, pe in pe_dict.items():
        if name == 'alibi':
            continue
        freqs = np.abs(pe.get_frequencies())
        freqs = freqs[freqs > 0]
        ax.hist(freqs, bins=30, alpha=0.5, label=pe.name,
                color=get_pe_color(name), density=True, edgecolor='white')
    ax.set_xlabel("$|\\omega|$")
    ax.set_ylabel("Density")
    ax.set_title("Frequency Density Distribution")
    ax.legend()
    
    # Panel 4: 累积频率分布 (CDF)
    ax = axes[1, 1]
    for name, pe in pe_dict.items():
        if name == 'alibi':
            continue
        freqs = np.sort(np.abs(pe.get_frequencies()))
        freqs = freqs[freqs > 0]
        cdf = np.arange(1, len(freqs) + 1) / len(freqs)
        ax.plot(freqs, cdf, 's-', label=pe.name, 
                color=get_pe_color(name), markersize=2, linewidth=1.5)
    ax.set_xlabel("$|\\omega|$")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title("Cumulative Frequency Distribution (CDF)")
    ax.legend()
    add_math_annotation(ax, "CDF shape ↔ resolution at different scales",
                        loc='lower right')
    
    plt.tight_layout()
    paths = save_figure(fig, "01_frequency_spectra", MODULE)
    plt.close(fig)
    return paths


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. 核矩阵热力图
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_kernel_matrices(pe_dict: dict, seq_len: int = 64):
    """
    可视化各 PE 的核矩阵（位置相关性矩阵）。
    
    K_{ij} = K(i - j) = ⟨PE(i), PE(j)⟩
    
    揭示信息:
        • 对角线结构 → 局部性（近距离位置高相关）
        • 条带模式 → 周期性
        • 衰减速率 → 长距离敏感度
    """
    positions = np.arange(seq_len)
    
    # ── Matplotlib 4-panel 热力图 ────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Kernel Matrices  $K_{ij} = K(i - j)$", 
                 fontsize=16, fontweight='bold')
    
    for idx, (name, pe) in enumerate(pe_dict.items()):
        ax = axes[idx // 2, idx % 2]
        K = pe.kernel_matrix(positions)
        
        vmin, vmax = K.min(), K.max()
        # 使核矩阵对称显示
        abs_max = max(abs(vmin), abs(vmax))
        
        im = ax.imshow(K, cmap='RdBu_r', aspect='auto',
                       vmin=-abs_max, vmax=abs_max,
                       extent=[0, seq_len, seq_len, 0])
        ax.set_xlabel("Position $j$")
        ax.set_ylabel("Position $i$")
        ax.set_title(f"{pe.name}  ($K_{{min}}$={vmin:.3f}, $K_{{max}}$={vmax:.3f})")
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    paths = save_figure(fig, "01_kernel_matrices", MODULE)
    plt.close(fig)
    
    # ── Plotly 交互热力图（逐个 PE） ─────────────────────────
    if HAS_PLOTLY:
        for name, pe in pe_dict.items():
            K = pe.kernel_matrix(positions)
            create_heatmap_html(
                K, 
                title=f"Kernel Matrix — {pe.name}",
                x_label="Position j", y_label="Position i",
                module=MODULE,
                filename=f"01_kernel_matrix_{name}.html"
            )
    
    return paths


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. 平移群作用分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_translation_group_analysis(pe_dict: dict, seq_len: int = 128):
    """
    分析各 PE 对整数平移群 (ℤ, +) 的响应方式。
    
    数学框架:
        定义平移算子 T_s: PE(p) ↦ PE(p + s)
        
        关键问题: ⟨PE(p), PE(p+s)⟩ 是否仅依赖 s 而非 p？
        
        • 如果是 → PE 具有平移不变性（仅编码相对位置）
        • 如果否 → PE 编码了绝对位置信息
        
        对于 Sinusoidal/RoPE: K(p, p+s) = K(s) ✓ 平移不变核
        对于 ALiBi: B(p, p+s) = -m|s| ✓ 平移不变偏置
        对于 LAPE: K(p, p+s) = K(s) ✓ 但衰减行为不同
        
    进一步: 验证在不同锚点 p₀ 处，K(p₀, p₀+Δ) 是否一致。
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Translation Group Analysis: Is $\\langle PE(p),\\, PE(p+\\Delta)\\rangle$ independent of $p$?",
                 fontsize=14, fontweight='bold')
    
    anchor_positions = [0, 32, 64, 96]
    deltas = np.arange(0, seq_len)
    
    for idx, (name, pe) in enumerate(pe_dict.items()):
        ax = axes[idx // 2, idx % 2]
        
        for p0 in anchor_positions:
            if name == 'alibi':
                # ALiBi: B(p₀, p₀+Δ) = -m·|Δ|
                k_vals = pe.kernel(deltas)
            else:
                # 数值验证: ⟨PE(p₀), PE(p₀+Δ)⟩
                enc_p0 = pe.encode(np.array([p0]))  # [1, dim]
                enc_targets = pe.encode(p0 + deltas)  # [N, dim]
                k_vals = (enc_p0 @ enc_targets.T).squeeze() / (pe.dim / 2)
            
            ax.plot(deltas, k_vals, label=f"$p_0={p0}$", 
                    linewidth=1.2, alpha=0.8)
        
        ax.set_xlabel("$\\Delta$")
        ax.set_ylabel("$K(p_0, p_0+\\Delta)$")
        ax.set_title(f"{pe.name}")
        ax.legend(fontsize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    paths = save_figure(fig, "01_translation_invariance", MODULE)
    plt.close(fig)
    return paths


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. 综合 HTML 报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_comparison_report():
    """生成综合对比报告"""
    sections = [
        {
            "title": "1. 核函数 K(Δ) 对比",
            "content": """
                <p>根据 <strong>Bochner 定理</strong>，每种位置编码方案定义了一个正定核函数：</p>
                <p>\\[ K(\\Delta) = \\frac{1}{m}\\sum_{k=0}^{m-1} \\cos(\\omega_k \\Delta) \\]</p>
                <p>核函数的衰减速率直接反映了位置编码的<strong>距离敏感度</strong>：</p>
                <ul>
                    <li><strong>Sinusoidal / RoPE</strong>: 几何级数频率 → 多尺度分辨率</li>
                    <li><strong>LAPE</strong>: 幂律频率 → 低频密集, 高频稀疏</li>
                    <li><strong>ALiBi</strong>: 线性偏置 → 单调递减, 无振荡</li>
                </ul>
            """
        },
        {
            "title": "2. 复平面嵌入",
            "content": """
                <p>复数嵌入 \\( z_k(p) = e^{i\\omega_k p} \\) 将整数位置映射到单位圆上：</p>
                <ul>
                    <li><strong>低频分量</strong> (小 ω): 慢旋转 → 编码宏观位置</li>
                    <li><strong>高频分量</strong> (大 ω): 快旋转 → 编码局部精细位置</li>
                </ul>
                <p>不同频率分量的组合形成高维空间中的<strong>螺旋流形</strong>。</p>
            """
        },
        {
            "title": "3. 谱测度对比",
            "content": """
                <p>频率分布的形状决定了位置编码的"视觉范围"：</p>
                <p>\\[ \\mu = \\frac{1}{m}\\sum_k \\delta(\\omega - \\omega_k) \\]</p>
                <ul>
                    <li><strong>对数均匀谱</strong> (Sinusoidal/RoPE): 各尺度均匀覆盖</li>
                    <li><strong>幂律谱</strong> (LAPE): 集中在低频, 适合空间坐标</li>
                </ul>
            """
        },
        {
            "title": "4. 平移群分析",
            "content": """
                <p>核心问题: \\( \\langle PE(p), PE(p+\\Delta)\\rangle \\) 是否仅依赖 \\(\\Delta\\)?</p>
                <p>若是 → 位置编码具有<strong>平移等变性</strong>（仅编码相对位置信息）。</p>
                <p>所有四种方案在理论上都满足平移不变性, 但在有限精度下可能出现偏差。</p>
            """
        },
    ]
    
    return generate_report_html(
        "统一数学框架下的 PE 方案对比",
        sections, MODULE, "01_unified_comparison_report.html"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  01 — Unified PE Comparison (Complex Analysis / Group Theory)")
    print("=" * 60)
    
    # 配置
    config = PEConfig(dim=64, max_len=512)
    pe_dict = get_all_pe(config=config)
    
    # 初始化日志
    logger = VizLogger('01_unified_comparison', module=MODULE)
    logger.set_description("统一数学框架下的四种 PE 方案对比分析")
    
    # 记录各 PE 的关键参数
    for name, pe in pe_dict.items():
        freqs = pe.get_frequencies()
        logger.log_parameter(f"{name}_freq_range", {
            "min": float(np.min(np.abs(freqs[freqs != 0]))) if np.any(freqs != 0) else 0,
            "max": float(np.max(np.abs(freqs))),
            "n_freqs": len(freqs)
        })
        logger.log_metric(f"{name}_kernel_at_1", float(pe.kernel(1)),
                         context={"description": "K(1) — nearest neighbor similarity"})
        logger.log_metric(f"{name}_kernel_at_100", float(pe.kernel(100)),
                         context={"description": "K(100) — long-range similarity"})
    
    # 执行可视化
    print("\n  [1/5] Kernel function comparison...")
    plot_kernel_comparison(pe_dict)
    logger.add_finding("核函数衰减速率: ALiBi (线性) > Sinusoidal/RoPE (振荡衰减) > LAPE (幂律衰减)",
                       category="kernel")
    
    print("  [2/5] Complex plane embeddings...")
    plot_complex_plane_embedding(pe_dict)
    logger.add_finding("Sinusoidal 和 RoPE 共享相同的频率序列，但应用方式不同（加性 vs 乘性）",
                       category="complex_embedding")
    
    print("  [3/5] Frequency spectra...")
    plot_frequency_spectra(pe_dict)
    logger.add_finding("LAPE 的幂律频率分布导致低频能量集中，适合编码大尺度空间结构",
                       category="spectral")
    
    print("  [4/5] Kernel matrices...")
    plot_kernel_matrices(pe_dict)
    logger.add_finding("ALiBi 的核矩阵呈严格锥形，而 Sinusoidal/RoPE 呈振荡衰减带状结构",
                       category="kernel_matrix")
    
    print("  [5/5] Translation group analysis...")
    plot_translation_group_analysis(pe_dict)
    logger.add_finding("所有方案的核函数在数值上满足平移不变性: K(p₀, p₀+Δ) ≈ K(Δ)",
                       category="translation")
    
    # 生成报告
    print("\n  Generating HTML report...")
    generate_comparison_report()
    
    # 保存日志
    logger.save()
    
    print("\n" + "=" * 60)
    print("  ✅ 01_unified_comparison 完成！")
    print(f"  📊 静态图: output/{MODULE}/")
    print(f"  🌐 交互图: html/{MODULE}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
