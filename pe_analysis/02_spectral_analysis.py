#!/usr/bin/env python3
"""
02 — 位置编码谱分析 (Spectral Analysis)

从傅里叶分析 / 功率谱密度 / 信息论的视角深度剖析四种位置编码方案的频域特性。

数学背景：
  Bochner 定理告诉我们，正定核函数 K(Δ) 可以表示为某个非负有限测度 μ 的 Fourier 变换：
      K(Δ) = ∫ e^{iωΔ} dμ(ω)
  
  因此，不同 PE 方案的本质区别在于它们的**谱测度 μ(dω)** 不同。
  本脚本通过多种频域工具揭示这些差异。

生成内容：
  1. 频率基底分布对比 (Frequency Basis)
  2. PE 编码的 FFT 功率谱 (Encoding PSD)
  3. 核函数的 Welch PSD 估计 (Kernel PSD)
  4. 谱熵对比 (Spectral Entropy)
  5. 2D 频域能量热力图 (Spectrogram)
  6. 时频联合分析 (STFT)
  7. 综合 HTML 报告

Output:
  output/pe_analysis/   → 静态图 (PNG/PDF)
  html/pe_analysis/     → 交互式 HTML

Usage:
  python -m pe_analysis.02_spectral_analysis
  python run.py pe_analysis.spectral
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal as sp_signal
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
    spectral_entropy,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

MODULE = "pe_analysis"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 频率基底分布对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_frequency_basis(pe_dict: dict, config: PEConfig):
    """
    对比各 PE 方案的频率基底 ω_k 的分布模式。

    数学意义：
        - Sinusoidal / RoPE: ω_k = 1/base^{2k/d} — 几何级数 (等比递减)
        - LAPE: ω_k = (-k/d)^a — 幂律分布
        - ALiBi: 无显式频率，使用 slope 作为替代
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Frequency Basis Distribution  $\\omega_k$ of Each PE Scheme",
                 fontsize=16, fontweight='bold')

    for idx, (name, pe) in enumerate(pe_dict.items()):
        ax = axes[idx // 2, idx % 2]
        freqs = pe.get_frequencies()
        k = np.arange(len(freqs))
        color = get_pe_color(name)

        # 茎叶图
        markerline, stemlines, baseline = ax.stem(k, np.abs(freqs), linefmt='-', markerfmt='o', basefmt='k-')
        markerline.set_color(color)
        markerline.set_markersize(4)
        stemlines.set_color(color)
        stemlines.set_alpha(0.6)

        ax.set_xlabel("Frequency index $k$")
        ax.set_ylabel("$|\\omega_k|$")
        ax.set_title(f"{pe.name}", fontsize=13)
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-8)

        # 标注分布类型
        if 'sinusoidal' in name.lower() or 'rope' in name.lower():
            add_math_annotation(ax, r"$\omega_k = \frac{1}{10000^{2k/d}}$", loc='upper right', fontsize=9)
        elif 'lape' in name.lower():
            add_math_annotation(ax, r"$\omega_k = \left(-\frac{k}{d}\right)^a$", loc='upper right', fontsize=9)
        elif 'alibi' in name.lower():
            add_math_annotation(ax, r"$m_h = 2^{-8h/H}$  (slopes)", loc='upper right', fontsize=9)

    plt.tight_layout()
    save_figure(fig, "02_frequency_basis", MODULE)
    plt.close(fig)

    # Plotly 交互式
    if HAS_PLOTLY:
        pfig = go.Figure()
        for name, pe in pe_dict.items():
            freqs = pe.get_frequencies()
            k = np.arange(len(freqs))
            color = get_pe_color(name)
            pfig.add_trace(go.Scatter(
                x=k, y=np.abs(freqs) + 1e-15,
                mode='lines+markers', name=pe.name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate='k=%{x}<br>|ω_k|=%{y:.6e}<extra>' + pe.name + '</extra>'
            ))
        pfig.update_layout(
            title="Frequency Basis |ω_k| (log scale)",
            xaxis_title="Frequency index k",
            yaxis_title="|ω_k|",
            yaxis_type="log",
            template="plotly_white",
            width=900, height=550,
        )
        save_plotly_html(pfig, "02_frequency_basis.html", MODULE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. PE 编码信号的 FFT 功率谱
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_encoding_fft(pe_dict: dict, config: PEConfig, seq_len: int = 512):
    """
    对各 PE 的实际编码矩阵 PE(p) 做 FFT，分析其功率谱。

    方法：
        对于每个 PE 方案：
        1. 生成 encoding 矩阵 [seq_len, dim]
        2. 对每一维 (column) 做 FFT
        3. 取功率谱 |FFT|² 并在维度上取平均 → 平均 PSD

    数学意义：
        FFT 揭示了 PE 编码作为"位置信号"时的频率能量分布。
        - Sinusoidal PE 的每一维是单一频率的正弦/余弦 → PSD 是 delta 函数
        - RoPE 同理（虽然乘性施加，但其"等价编码"也是正弦/余弦）
        - LAPE 的幂律频率导致不同的 PSD 形状
    """
    positions = np.arange(seq_len)

    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FFT Power Spectrum of PE Encoding Signals",
                 fontsize=16, fontweight='bold')

    fft_results = {}

    for idx, (name, pe) in enumerate(pe_dict.items()):
        ax = axes[idx // 2, idx % 2]
        color = get_pe_color(name)

        enc = pe.encode(positions)  # [seq_len, dim]

        # 对每一维做 FFT
        fft_vals = np.fft.rfft(enc, axis=0)  # [seq_len//2+1, dim]
        psd = np.abs(fft_vals) ** 2           # 功率谱

        # 频率轴 (归一化)
        freqs_fft = np.fft.rfftfreq(seq_len)

        # 平均 PSD (跨维度)
        mean_psd = psd.mean(axis=1)

        # 存储结果
        fft_results[name] = {
            'freqs': freqs_fft,
            'mean_psd': mean_psd,
            'psd_matrix': psd,
        }

        # 绘制平均 PSD (clamp to avoid log(0) warning)
        mean_psd_safe = np.maximum(mean_psd, 1e-30)
        ax.semilogy(freqs_fft[1:], mean_psd_safe[1:], color=color, linewidth=1.5, label='Mean PSD')

        # 叠加几个单维的 PSD (半透明)
        dims_to_show = [0, config.dim // 4, config.dim // 2, config.dim - 1]
        for d_idx in dims_to_show:
            if d_idx < psd.shape[1]:
                ax.semilogy(freqs_fft[1:], np.maximum(psd[1:, d_idx], 1e-30), alpha=0.2, color=color, linewidth=0.5)

        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Power $|\\hat{PE}(f)|^2$")
        ax.set_title(f"{pe.name}", fontsize=13)
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, "02_encoding_fft", MODULE)
    plt.close(fig)

    # 综合对比图
    setup_plot_style()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for name, pe in pe_dict.items():
        color = get_pe_color(name)
        r = fft_results[name]
        ax2.semilogy(r['freqs'][1:], np.maximum(r['mean_psd'][1:], 1e-30), color=color,
                     linewidth=2, label=pe.name)
    ax2.set_xlabel("Normalized frequency")
    ax2.set_ylabel("Mean Power Spectral Density")
    ax2.set_title("Averaged FFT Power Spectrum — All PE Schemes", fontsize=14, fontweight='bold')
    ax2.legend()
    add_math_annotation(ax2, r"$\mathrm{PSD}(f) = \frac{1}{d}\sum_{k=1}^{d}|\hat{PE}_k(f)|^2$",
                        loc='upper right')
    plt.tight_layout()
    save_figure(fig2, "02_encoding_fft_comparison", MODULE)
    plt.close(fig2)

    # Plotly
    if HAS_PLOTLY:
        pfig = go.Figure()
        for name, pe in pe_dict.items():
            color = get_pe_color(name)
            r = fft_results[name]
            pfig.add_trace(go.Scatter(
                x=r['freqs'][1:], y=r['mean_psd'][1:],
                mode='lines', name=pe.name,
                line=dict(color=color, width=2),
                hovertemplate='f=%{x:.4f}<br>PSD=%{y:.4e}<extra>' + pe.name + '</extra>'
            ))
        pfig.update_layout(
            title="Mean FFT Power Spectrum of PE Encodings",
            xaxis_title="Normalized Frequency",
            yaxis_title="Power Spectral Density",
            yaxis_type="log",
            template="plotly_white",
            width=950, height=550,
        )
        save_plotly_html(pfig, "02_encoding_fft.html", MODULE)

    return fft_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. 核函数的 Welch PSD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_kernel_psd(pe_dict: dict, max_delta: int = 1024):
    """
    用 Welch 方法估计核函数 K(Δ) 的功率谱密度。

    数学背景：
        核函数 K(Δ) 本身是一个关于位置差 Δ 的信号。
        对它做谱分析，得到的就是 Bochner 定理中的谱测度 μ(ω) 的数值近似：
            PSD_K(ω) ≈ μ(ω)

        因此，Welch PSD 直接揭示了 Bochner 谱测度的形状。
    """
    deltas = np.arange(max_delta)

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Welch PSD
    ax1 = axes[0]
    # Panel 2: 累积谱分布函数
    ax2 = axes[1]

    psd_results = {}

    for name, pe in pe_dict.items():
        color = get_pe_color(name)
        k_vals = pe.kernel(deltas.astype(float))

        # Welch PSD
        f_welch, psd_welch = sp_signal.welch(k_vals, fs=1.0, nperseg=min(256, len(k_vals)))
        psd_results[name] = {'f': f_welch, 'psd': psd_welch}

        ax1.semilogy(f_welch, psd_welch, color=color, linewidth=2, label=pe.name)

        # 累积谱分布 (CSD)
        csd = np.cumsum(psd_welch)
        csd /= csd[-1] + 1e-15
        ax2.plot(f_welch, csd, color=color, linewidth=2, label=pe.name)

    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("PSD (Welch)")
    ax1.set_title("Kernel PSD  ≈  Bochner Spectral Measure $\\mu(\\omega)$", fontsize=13)
    ax1.legend()
    add_math_annotation(ax1, r"$K(\Delta) = \int e^{i\omega\Delta}\,d\mu(\omega)$",
                        loc='upper right', fontsize=9)

    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Cumulative Spectral Distribution")
    ax2.set_title("Cumulative Spectral Distribution Function", fontsize=13)
    ax2.legend()
    ax2.axhline(y=0.5, color='gray', ls='--', alpha=0.5)
    ax2.axhline(y=0.9, color='gray', ls='--', alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "02_kernel_psd", MODULE)
    plt.close(fig)

    # Plotly
    if HAS_PLOTLY:
        pfig = make_subplots(rows=1, cols=2,
                             subplot_titles=["Kernel Welch PSD (≈ Bochner μ(ω))",
                                             "Cumulative Spectral Distribution"])
        for name, pe in pe_dict.items():
            color = get_pe_color(name)
            r = psd_results[name]
            pfig.add_trace(go.Scatter(
                x=r['f'], y=r['psd'],
                mode='lines', name=pe.name,
                line=dict(color=color, width=2),
                legendgroup=name,
                hovertemplate='f=%{x:.4f}<br>PSD=%{y:.4e}<extra></extra>'
            ), row=1, col=1)

            csd = np.cumsum(r['psd'])
            csd /= csd[-1] + 1e-15
            pfig.add_trace(go.Scatter(
                x=r['f'], y=csd,
                mode='lines', name=pe.name + ' (CSD)',
                line=dict(color=color, width=2, dash='dot'),
                legendgroup=name, showlegend=False,
            ), row=1, col=2)

        pfig.update_yaxes(type="log", row=1, col=1)
        pfig.update_layout(template="plotly_white", width=1100, height=500,
                           title="Kernel Function Spectral Analysis")
        save_plotly_html(pfig, "02_kernel_psd.html", MODULE)

    return psd_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. 谱熵对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_spectral_entropy(pe_dict: dict, config: PEConfig):
    """
    对比各 PE 方案在不同维度/序列长度下的谱熵。

    谱熵定义：
        H_s = -Σ p_k log₂(p_k),  p_k = PSD_k / Σ PSD
        归一化: H̃_s = H_s / log₂(N)

    物理意义：
        H̃_s → 1: 能量均匀分布在所有频率上 (白噪声)
        H̃_s → 0: 能量集中在极少数频率 (纯正弦波)

    PE 方案的预期：
        - Sinusoidal: 低谱熵 (每一维是单频信号)
        - LAPE: 中等谱熵 (幂律频率分布)
        - ALiBi: 高谱熵 (线性衰减的宽频谱)
    """
    # 4a: 不同维度下的谱熵
    dims_to_test = [16, 32, 64, 128, 256, 512]
    seq_len = 512

    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: 各维度下的编码谱熵
    ax1 = axes[0]
    entropy_data = {}
    for name in pe_dict:
        entropies = []
        for dim in dims_to_test:
            cfg = PEConfig(dim=dim, max_len=seq_len)
            pe = get_pe(name, config=cfg)
            enc = pe.encode(np.arange(seq_len))
            # 对每一维做 FFT，计算平均谱熵
            fft_vals = np.fft.rfft(enc, axis=0)
            mean_psd = np.mean(np.abs(fft_vals) ** 2, axis=1)
            entropies.append(spectral_entropy(mean_psd))
        entropy_data[name] = entropies
        color = get_pe_color(name)
        ax1.plot(dims_to_test, entropies, 'o-', color=color, linewidth=2,
                 label=pe_dict[name].name, markersize=6)

    ax1.set_xlabel("Embedding dimension $d$")
    ax1.set_ylabel("Normalized Spectral Entropy $\\tilde{H}_s$")
    ax1.set_title("Encoding Spectral Entropy vs. Dimension", fontsize=12)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # Panel 2: 不同序列长度下的谱熵
    ax2 = axes[1]
    seq_lens = [64, 128, 256, 512, 1024, 2048]
    for name in pe_dict:
        entropies_seq = []
        for sl in seq_lens:
            pe = pe_dict[name]
            positions = np.arange(sl)
            enc = pe.encode(positions)
            fft_vals = np.fft.rfft(enc, axis=0)
            mean_psd = np.mean(np.abs(fft_vals) ** 2, axis=1)
            entropies_seq.append(spectral_entropy(mean_psd))
        color = get_pe_color(name)
        ax2.plot(seq_lens, entropies_seq, 's-', color=color, linewidth=2,
                 label=pe_dict[name].name, markersize=5)

    ax2.set_xlabel("Sequence length $N$")
    ax2.set_ylabel("Normalized Spectral Entropy $\\tilde{H}_s$")
    ax2.set_title("Encoding Spectral Entropy vs. Seq Length", fontsize=12)
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.set_xscale('log', base=2)

    # Panel 3: 核函数的谱熵
    ax3 = axes[2]
    kernel_entropies = {}
    deltas_range = [128, 256, 512, 1024, 2048]
    for name in pe_dict:
        k_entropies = []
        for max_d in deltas_range:
            deltas = np.arange(max_d).astype(float)
            k_vals = pe_dict[name].kernel(deltas)
            f_w, psd_w = sp_signal.welch(k_vals, fs=1.0, nperseg=min(256, max_d))
            k_entropies.append(spectral_entropy(psd_w))
        kernel_entropies[name] = k_entropies
        color = get_pe_color(name)
        ax3.plot(deltas_range, k_entropies, '^-', color=color, linewidth=2,
                 label=pe_dict[name].name, markersize=5)

    ax3.set_xlabel("Kernel window size $\\Delta_{max}$")
    ax3.set_ylabel("Kernel Spectral Entropy")
    ax3.set_title("Kernel Spectral Entropy vs. Window Size", fontsize=12)
    ax3.legend()
    ax3.set_ylim(0, 1.05)

    plt.tight_layout()
    save_figure(fig, "02_spectral_entropy", MODULE)
    plt.close(fig)

    return entropy_data, kernel_entropies


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. 2D 频谱热力图
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_spectrogram_2d(pe_dict: dict, config: PEConfig, seq_len: int = 512):
    """
    生成每种 PE 编码的 2D 频谱热力图：维度 × 频率 → 功率。

    这展示了每个编码维度中的频率能量分布。
    对于 Sinusoidal PE，每一维只有一个频率分量，因此热力图上会呈现清晰的对角线结构。
    """
    positions = np.arange(seq_len)

    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2D Spectrogram: Encoding Dimension × Frequency",
                 fontsize=16, fontweight='bold')

    for idx, (name, pe) in enumerate(pe_dict.items()):
        ax = axes[idx // 2, idx % 2]

        enc = pe.encode(positions)  # [seq_len, dim]
        # 对每一维做 FFT
        fft_vals = np.fft.rfft(enc, axis=0)  # [seq_len//2+1, dim]
        psd = np.abs(fft_vals) ** 2
        freqs_fft = np.fft.rfftfreq(seq_len)

        # 取 log 以便可视化 (避免 log(0))
        log_psd = np.log10(psd + 1e-15)

        im = ax.imshow(log_psd.T, aspect='auto', origin='lower',
                       extent=[freqs_fft[0], freqs_fft[-1], 0, config.dim],
                       cmap='inferno', interpolation='nearest')
        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Encoding dimension")
        ax.set_title(f"{pe.name}", fontsize=13)
        plt.colorbar(im, ax=ax, label='log₁₀(PSD)', shrink=0.8)

    plt.tight_layout()
    save_figure(fig, "02_spectrogram_2d", MODULE)
    plt.close(fig)

    # Plotly 交互式 (逐个方案)
    if HAS_PLOTLY:
        for name, pe in pe_dict.items():
            enc = pe.encode(positions)
            fft_vals = np.fft.rfft(enc, axis=0)
            psd = np.abs(fft_vals) ** 2
            freqs_fft = np.fft.rfftfreq(seq_len)
            log_psd = np.log10(psd + 1e-15)

            pfig = go.Figure(data=go.Heatmap(
                z=log_psd.T,
                x=freqs_fft,
                y=np.arange(config.dim),
                colorscale='Inferno',
                colorbar=dict(title='log₁₀(PSD)'),
                hovertemplate='Freq: %{x:.4f}<br>Dim: %{y}<br>log₁₀(PSD): %{z:.2f}<extra></extra>'
            ))
            pfig.update_layout(
                title=f"2D Spectrogram — {pe.name}",
                xaxis_title="Normalized Frequency",
                yaxis_title="Encoding Dimension",
                template="plotly_white",
                width=850, height=600,
            )
            save_plotly_html(pfig, f"02_spectrogram_{name}.html", MODULE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. 时频联合分析 (STFT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_stft_analysis(pe_dict: dict, config: PEConfig, seq_len: int = 512):
    """
    对 PE 编码的代表维度做短时傅里叶变换 (STFT)。

    数学定义：
        STFT{x[n]}(m, ω) = Σ_n x[n] w[n-m] e^{-iωn}

    这揭示了 PE 编码在位置轴上的频率成分是否变化（对于 Sinusoidal PE 应该是常数，
    因为每一维是恒定频率的正弦波）。

    选取的代表维度：
        - dim 0 (最高频)
        - dim d//2 (中频)
        - dim d-2 (最低频)
    """
    positions = np.arange(seq_len)
    representative_dims = [0, config.dim // 2, config.dim - 2]
    dim_labels = ['High-freq (k=0)', f'Mid-freq (k={config.dim//2})', f'Low-freq (k={config.dim-2})']

    setup_plot_style()
    n_pe = len(pe_dict)
    n_dims = len(representative_dims)
    fig, axes = plt.subplots(n_pe, n_dims, figsize=(5 * n_dims, 4 * n_pe))
    fig.suptitle("STFT of PE Encoding — Time-Frequency Analysis",
                 fontsize=16, fontweight='bold', y=1.02)

    for row, (name, pe) in enumerate(pe_dict.items()):
        enc = pe.encode(positions)  # [seq_len, dim]
        for col, (d_idx, d_label) in enumerate(zip(representative_dims, dim_labels)):
            ax = axes[row, col] if n_pe > 1 else axes[col]
            sig = enc[:, d_idx]

            # STFT
            nperseg = min(64, seq_len // 4)
            f_stft, t_stft, Zxx = sp_signal.stft(sig, fs=1.0, nperseg=nperseg,
                                                   noverlap=nperseg // 2)
            ax.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud',
                          cmap='magma')
            ax.set_ylabel("Frequency")
            if row == n_pe - 1:
                ax.set_xlabel("Position")
            if col == 0:
                ax.set_ylabel(f"{pe.name}\nFrequency")
            if row == 0:
                ax.set_title(d_label, fontsize=11)

    plt.tight_layout()
    save_figure(fig, "02_stft_analysis", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. 频谱衰减速率分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_spectral_decay(pe_dict: dict, config: PEConfig, seq_len: int = 512):
    """
    分析 PE 编码功率谱的衰减速率。

    数学背景：
        如果 PSD(f) ~ f^{-β}，则 β 称为谱指数。
        - β ≈ 0: 白噪声
        - β ≈ 1: 1/f 噪声 (粉红噪声)
        - β ≈ 2: 布朗噪声
        - β > 2: 高度光滑的信号

    通过 log-log 拟合 PSD 的衰减斜率来估算 β。
    """
    positions = np.arange(seq_len)

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]  # log-log PSD
    ax2 = axes[1]  # 衰减指数 β 柱状图

    betas = {}

    for name, pe in pe_dict.items():
        color = get_pe_color(name)
        enc = pe.encode(positions)

        # 平均 PSD
        fft_vals = np.fft.rfft(enc, axis=0)
        mean_psd = np.mean(np.abs(fft_vals) ** 2, axis=1)
        freqs = np.fft.rfftfreq(seq_len)

        # 去掉 DC 分量
        f_pos = freqs[1:]
        psd_pos = mean_psd[1:]

        # log-log 绘图
        ax1.loglog(f_pos, psd_pos, color=color, linewidth=2, label=pe.name)

        # 拟合 β (log-log 线性回归)
        mask = psd_pos > 1e-15
        if mask.sum() > 10:
            log_f = np.log10(f_pos[mask])
            log_psd = np.log10(psd_pos[mask])
            coeffs = np.polyfit(log_f, log_psd, 1)
            beta = -coeffs[0]  # PSD ~ f^{-β}
            betas[name] = beta

            # 拟合线
            fit_line = 10 ** (coeffs[0] * np.log10(f_pos) + coeffs[1])
            ax1.loglog(f_pos, fit_line, '--', color=color, alpha=0.5,
                       label=f'{pe.name} fit: β={beta:.2f}')
        else:
            betas[name] = 0.0

    ax1.set_xlabel("Frequency (log scale)")
    ax1.set_ylabel("PSD (log scale)")
    ax1.set_title("Power Spectral Density — Log-Log Scale", fontsize=13)
    ax1.legend(fontsize=8)
    add_math_annotation(ax1, r"$\mathrm{PSD}(f) \sim f^{-\beta}$", loc='upper right')

    # 柱状图
    names_list = list(betas.keys())
    beta_vals = [betas[n] for n in names_list]
    colors_list = [get_pe_color(n) for n in names_list]
    bars = ax2.bar(range(len(names_list)), beta_vals, color=colors_list, alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(names_list)))
    ax2.set_xticklabels([pe_dict[n].name for n in names_list], fontsize=10)
    ax2.set_ylabel("Spectral Decay Exponent $\\beta$")
    ax2.set_title("Spectral Decay Exponent (higher = smoother signal)", fontsize=13)
    ax2.axhline(y=1, color='gray', ls='--', alpha=0.5, label='1/f noise')
    ax2.axhline(y=2, color='gray', ls=':', alpha=0.5, label='Brownian')
    ax2.legend(fontsize=8)

    # 在柱子上方标注数值
    for bar, val in zip(bars, beta_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, "02_spectral_decay", MODULE)
    plt.close(fig)

    return betas


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  综合报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_spectral_report():
    """生成综合 HTML 报告"""
    sections = [
        {
            'title': '1. 频率基底分布',
            'content': (
                '不同 PE 方案使用不同的频率基底 {ω_k}。Sinusoidal PE 和 RoPE 使用相同的几何级数 '
                '\\(\\omega_k = 1/10000^{2k/d}\\)，而 LAPE 使用幂律分布 \\(\\omega_k = (-k/d)^a\\)。'
                'ALiBi 没有显式频率，但其斜率 \\(m_h = 2^{-8h/H}\\) 扮演了类似角色。'
                '<br><br>频率分布的形状直接决定了核函数的衰减行为——这是 Bochner 定理的核心推论。'
            )
        },
        {
            'title': '2. FFT 功率谱分析',
            'content': (
                '对 PE 编码矩阵的每一维做 FFT，得到的功率谱揭示了位置信号的频率能量分布。'
                '<br><br>Sinusoidal PE 的每一维是纯正弦/余弦波，因此其 PSD 应该是 delta 函数族。'
                'LAPE 的幂律频率导致更集中的频率分布。'
                '<br><br>公式：\\(\\mathrm{PSD}(f) = \\frac{1}{d}\\sum_{k=1}^{d}|\\hat{PE}_k(f)|^2\\)'
            )
        },
        {
            'title': '3. 核函数的 Bochner 谱测度',
            'content': (
                '根据 Bochner 定理，正定核函数 \\(K(\\Delta)\\) 可以分解为：'
                '\\[K(\\Delta) = \\int e^{i\\omega\\Delta}\\,d\\mu(\\omega)\\]'
                '其中 \\(\\mu(\\omega)\\) 是非负有限谱测度。用 Welch 方法估计核函数的 PSD，'
                '就是在数值上近似这个谱测度。'
                '<br><br>累积谱分布函数 (CSD) 进一步展示了能量如何在频率轴上累积。'
            )
        },
        {
            'title': '4. 谱熵与信息论',
            'content': (
                '谱熵 \\(\\tilde{H}_s = -\\sum p_k \\log_2 p_k / \\log_2 N\\) 量化了频率能量分布的均匀性。'
                '<br><br>物理直觉：'
                '<ul>'
                '<li>\\(\\tilde{H}_s \\to 1\\): 白噪声（所有频率均分能量）</li>'
                '<li>\\(\\tilde{H}_s \\to 0\\): 纯正弦波（能量集中在单一频率）</li>'
                '</ul>'
                '谱熵越低，说明 PE 编码的频率结构越清晰——有助于位置区分。'
            )
        },
        {
            'title': '5. 2D 频谱热力图',
            'content': (
                '以编码维度为纵轴、频率为横轴，用热力图展示每一维的频率能量分布。'
                '<br><br>Sinusoidal PE 会呈现清晰的"对角线"结构——每一维恰好对应一个特征频率。'
                'LAPE 的幂律频率分布会导致热力图上呈现不同的聚集模式。'
            )
        },
        {
            'title': '6. 时频联合分析 (STFT)',
            'content': (
                '短时傅里叶变换 (STFT) 揭示了 PE 编码在位置轴上的频率成分是否随位置变化。'
                '<br><br>对于 Sinusoidal PE，每一维是恒定频率的正弦波，因此 STFT 应该是水平直线。'
                '这体现了位置编码的**平移不变性**。'
            )
        },
        {
            'title': '7. 谱衰减指数',
            'content': (
                '如果 PSD 在 log-log 坐标下呈线性衰减 \\(\\mathrm{PSD}(f) \\sim f^{-\\beta}\\)，'
                '则 \\(\\beta\\) 称为谱指数：'
                '<ul>'
                '<li>\\(\\beta \\approx 0\\): 白噪声</li>'
                '<li>\\(\\beta \\approx 1\\): 1/f 噪声（粉红噪声）</li>'
                '<li>\\(\\beta \\approx 2\\): 布朗噪声</li>'
                '<li>\\(\\beta > 2\\): 非常光滑的信号</li>'
                '</ul>'
                '谱衰减指数衡量了 PE 编码的"光滑度"，与其在深层传播中的稳定性可能相关。'
            )
        },
    ]

    generate_report_html(
        title="02 — Position Encoding Spectral Analysis",
        sections=sections,
        module=MODULE,
        filename="02_spectral_analysis_report.html"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  02 — Position Encoding Spectral Analysis")
    print("=" * 60)

    # 配置
    config = PEConfig(dim=64, max_len=512)
    pe_dict = get_all_pe(config=config)

    # 日志
    logger = VizLogger("pe_analysis", "02_spectral_analysis")

    # ── 1. 频率基底 ──
    print("\n  [1/7] Frequency basis distribution...")
    plot_frequency_basis(pe_dict, config)
    logger.log_metric("frequency_basis", "completed")

    # ── 2. FFT 功率谱 ──
    print("\n  [2/7] Encoding FFT power spectrum...")
    fft_results = plot_encoding_fft(pe_dict, config)
    logger.log_metric("encoding_fft", "completed")

    # ── 3. 核函数 PSD ──
    print("\n  [3/7] Kernel Welch PSD...")
    psd_results = plot_kernel_psd(pe_dict)
    logger.log_metric("kernel_psd", "completed")

    # ── 4. 谱熵 ──
    print("\n  [4/7] Spectral entropy comparison...")
    entropy_data, kernel_entropies = plot_spectral_entropy(pe_dict, config)
    for name in pe_dict:
        logger.log_metric(f"spectral_entropy_{name}", entropy_data[name][-1] if name in entropy_data else 0.0)

    # ── 5. 2D 频谱热力图 ──
    print("\n  [5/7] 2D spectrogram heatmaps...")
    plot_spectrogram_2d(pe_dict, config)
    logger.log_metric("spectrogram_2d", "completed")

    # ── 6. STFT ──
    print("\n  [6/7] STFT time-frequency analysis...")
    plot_stft_analysis(pe_dict, config)
    logger.log_metric("stft_analysis", "completed")

    # ── 7. 谱衰减 ──
    print("\n  [7/7] Spectral decay analysis...")
    betas = plot_spectral_decay(pe_dict, config)
    for name, beta in betas.items():
        logger.log_metric(f"spectral_decay_beta_{name}", beta)

    # ── 报告 ──
    print("\n  Generating HTML report...")
    generate_spectral_report()

    # 保存日志
    logger.save()

    print("\n" + "=" * 60)
    print("  ✅ 02_spectral_analysis 完成！")
    print("  📊 静态图: output/pe_analysis/")
    print("  🌐 交互图: html/pe_analysis/")
    print("=" * 60)


if __name__ == "__main__":
    main()
