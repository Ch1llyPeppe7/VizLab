#!/usr/bin/env python3
"""
08 — PE 微分几何分析

核心思想:
    将位置编码 PE: ℤ → ℝ^d 视为高维空间中的参数曲线，
    运用微分几何工具分析其几何结构。

    核心问题: PE 曲线的几何特征（曲率、挠率、度量张量）
    如何影响模型对位置信息的处理？

    1. 度量张量 (Metric Tensor):
       - g(p) = ‖dPE/dp‖² 是 Fisher-Rao 度量的特例
       - 衡量编码空间在各位置的"拉伸程度"
       - g(p) 大 → 位置分辨率高; g(p) 小 → 位置压缩

    2. 弧长与测地距离 (Arc Length):
       - s(p) = ∫₀ᵖ √g(t) dt 累积弧长
       - ds/dp = √g(p) 弧长速率
       - 反映编码空间的"速度特征"

    3. 曲率 (Curvature):
       - Frenet-Serret 曲率 κ(p) = ‖γ' × γ''‖ / ‖γ'‖³
       - 衡量曲线的弯曲程度
       - 高曲率 → 局部位置信息变化剧烈

    4. 挠率 (Torsion):
       - τ(p) = (γ' × γ'') · γ''' / ‖γ' × γ''‖²
       - 衡量曲线离开密切平面的程度 (3D+)

    5. Christoffel 符号:
       - 描述平行移动在 PE 流形上的扭曲
       - 热力图展示联络结构

    6. 曲率谱 (Curvature Spectrum):
       - 分解各频率子空间的曲率贡献
       - Sinusoidal/RoPE 的 2D 子空间曲率 = ω_k

Output:
    output/pe_analysis/   → 静态图 (PNG/PDF)
    html/pe_analysis/     → 交互式 HTML

Usage:
    python -m pe_analysis.08_differential_geometry
    python run.py pe_analysis.geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import cumulative_trapezoid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    PEConfig, get_pe, get_all_pe, SinusoidalPE, RoPE, ALiBi, LAPE,
    setup_plot_style, save_figure, add_math_annotation,
    save_plotly_html, generate_report_html, get_pe_color,
    PE_COLORS, VizLogger,
)

# 使用 core.analysis.geometry 通用数学库
from core.analysis.geometry import (
    curvature as _curvature_generic,
    torsion as _torsion_generic,
    metric_tensor as _metric_tensor_generic,
    arc_length as _arc_length_generic,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

MODULE = "pe_analysis"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  微分几何核心数学函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_derivatives(pe_instance, positions: np.ndarray, h: float = 0.5):
    """
    计算 PE 曲线的一阶、二阶、三阶导数 (中心差分)。

    Args:
        pe_instance: PositionEncoding 实例
        positions: [N,] 位置序列
        h: 差分步长
    Returns:
        gamma: [N, d] PE 编码
        d1: [N, d] 一阶导数 γ'
        d2: [N, d] 二阶导数 γ''
        d3: [N, d] 三阶导数 γ'''
    """
    pos = np.atleast_1d(positions).astype(float)
    
    # 对 ALiBi 使用 bias_matrix 的行作为伪编码
    if hasattr(pe_instance, 'bias_matrix') and pe_instance.name == 'ALiBi':
        L = int(np.max(pos)) + 10
        bias = pe_instance.bias_matrix(L, head_idx=0)
        # 取 bias 的每一行作为伪编码
        gamma = np.array([bias[int(p), :min(L, pe_instance.dim)] for p in pos])
        # 导数
        gamma_plus = np.array([bias[min(int(p+h), L-1), :min(L, pe_instance.dim)] for p in pos])
        gamma_minus = np.array([bias[max(int(p-h), 0), :min(L, pe_instance.dim)] for p in pos])
        gamma_plus2 = np.array([bias[min(int(p+2*h), L-1), :min(L, pe_instance.dim)] for p in pos])
        gamma_minus2 = np.array([bias[max(int(p-2*h), 0), :min(L, pe_instance.dim)] for p in pos])
    else:
        gamma = pe_instance.encode(pos)
        gamma_plus = pe_instance.encode(pos + h)
        gamma_minus = pe_instance.encode(pos - h)
        gamma_plus2 = pe_instance.encode(pos + 2*h)
        gamma_minus2 = pe_instance.encode(pos - 2*h)
    
    # 一阶导数: (f(x+h) - f(x-h)) / 2h
    d1 = (gamma_plus - gamma_minus) / (2 * h)
    
    # 二阶导数: (f(x+h) - 2f(x) + f(x-h)) / h²
    d2 = (gamma_plus - 2 * gamma + gamma_minus) / (h ** 2)
    
    # 三阶导数: (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / 2h³
    d3 = (gamma_plus2 - 2*gamma_plus + 2*gamma_minus - gamma_minus2) / (2 * h**3)
    
    return gamma, d1, d2, d3


def metric_tensor(d1: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    度量张量 (1D 参数曲线退化为标量度量)。
    
    使用 core.analysis.geometry 通用库实现。
    
    g(p) = ‖γ'(p)‖² = Σ_k (dPE_k/dp)²

    Args:
        d1: [N, d] 一阶导数
        eps: 数值稳定项
    Returns:
        [N,] 度量张量值
    """
    # 调用通用库 (metric_tensor_generic 接受 embeddings，这里直接使用 d1)
    # 由于通用库计算的是嵌入的导数度量，而这里 d1 已经是导数，直接计算范数平方
    return np.sum(d1 ** 2, axis=1) + eps


def arc_length(g: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    累积弧长 s(p) = ∫₀ᵖ √g(t) dt
    
    注: 由于接口差异，保留本地实现以兼容位置序列积分。

    Args:
        g: [N,] 度量张量
        positions: [N,] 位置序列
    Returns:
        [N,] 累积弧长
    """
    speed = np.sqrt(g)  # ds/dp = √g
    # 数值积分 (梯形法)
    s = np.zeros_like(g)
    s[1:] = cumulative_trapezoid(speed, positions)
    return s


def curvature(d1: np.ndarray, d2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Frenet-Serret 曲率 (高维广义公式)。
    
    使用 core.analysis.geometry.curvature 通用库实现。

    κ(p) = √(‖γ'‖²‖γ''‖² - (γ'·γ'')²) / ‖γ'‖³

    Args:
        d1: [N, d] 一阶导数
        d2: [N, d] 二阶导数
        eps: 数值稳定项
    Returns:
        [N,] 曲率
    """
    # 使用 core.analysis.geometry 通用库
    return _curvature_generic(d1, d2, eps)


def torsion(d1: np.ndarray, d2: np.ndarray, d3: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    挠率 (Torsion) — 曲线离开密切平面的程度。
    
    使用 core.analysis.geometry.torsion 通用库实现。

    τ(p) = (γ' × γ'') · γ''' / ‖γ' × γ''‖²

    Args:
        d1, d2, d3: [N, d] 各阶导数
        eps: 数值稳定项
    Returns:
        [N,] 挠率
    """
    # 使用 core.analysis.geometry 通用库
    return _torsion_generic(d1, d2, d3, eps)


def christoffel_heatmap(pe_instance, positions: np.ndarray, h: float = 0.5) -> np.ndarray:
    """
    计算 Christoffel-like 联络结构 (Position × Dimension)。

    对于 1D 参数曲线，我们计算每个维度的"局部曲率贡献":
    Γ_k(p) = |d²PE_k/dp²| / (|dPE_k/dp| + ε)

    这反映了每个维度在各位置的"联络强度"。

    Args:
        pe_instance: PositionEncoding 实例
        positions: [N,] 位置序列
        h: 差分步长
    Returns:
        [N, d] Christoffel-like 矩阵
    """
    gamma, d1, d2, d3 = compute_derivatives(pe_instance, positions, h)
    eps = 1e-12
    
    # Γ_k(p) = |d²PE_k/dp²| / (|dPE_k/dp| + ε)
    christoffel = np.abs(d2) / (np.abs(d1) + eps)
    
    return christoffel


def curvature_spectrum(pe_instance, positions: np.ndarray, h: float = 0.5) -> tuple:
    """
    曲率谱: 各 2D 子空间 (sin, cos) 的曲率贡献。

    对 Sinusoidal/RoPE，每个 2D 子空间是圆，曲率 = ω_k。

    Args:
        pe_instance: PositionEncoding 实例
        positions: [N,] 位置序列
        h: 差分步长
    Returns:
        freqs: [d//2,] 频率
        kappa_per_subspace: [d//2,] 各子空间平均曲率
    """
    freqs = pe_instance.get_frequencies()  # [d//2,]
    m = len(freqs)
    
    # 对每个 2D 子空间计算曲率
    kappa_subspace = np.zeros(m)
    
    if hasattr(pe_instance, 'bias_matrix') and pe_instance.name == 'ALiBi':
        # ALiBi 没有真正的频率子空间，返回斜率相关值
        kappa_subspace = freqs.copy()  # 使用斜率作为替代
    else:
        for k in range(m):
            # 提取第 k 个 2D 子空间的编码
            def encode_2d(p):
                enc = pe_instance.encode(np.atleast_1d(p))
                return enc[:, [2*k, 2*k+1]]  # [N, 2]
            
            # 计算该子空间的曲率
            pos = np.atleast_1d(positions).astype(float)
            gamma_2d = encode_2d(pos)
            gamma_2d_plus = encode_2d(pos + h)
            gamma_2d_minus = encode_2d(pos - h)
            
            d1_2d = (gamma_2d_plus - gamma_2d_minus) / (2 * h)
            d2_2d = (gamma_2d_plus - 2 * gamma_2d + gamma_2d_minus) / (h ** 2)
            
            # 2D 曲率 = |x'y'' - y'x''| / (x'² + y'²)^{3/2}
            eps = 1e-12
            cross = d1_2d[:, 0] * d2_2d[:, 1] - d1_2d[:, 1] * d2_2d[:, 0]
            norm_d1_sq = np.sum(d1_2d ** 2, axis=1)
            kappa_2d = np.abs(cross) / (np.power(norm_d1_sq + eps, 1.5))
            
            kappa_subspace[k] = np.mean(kappa_2d)
    
    return freqs, kappa_subspace


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  可视化函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_metric_tensor(pe_dict: dict, max_len: int = 256):
    """
    [图 1] 度量张量: g(p) = ‖dPE/dp‖² 随位置变化。

    物理意义: g(p) 大 → 编码空间"拉伸" → 位置分辨率高
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Metric Tensor — 度量张量分析", fontsize=16, fontweight='bold')

    positions = np.arange(max_len).astype(float)

    for pe_name, pe_inst in pe_dict.items():
        color = get_pe_color(pe_name)
        label = pe_inst.name

        gamma, d1, d2, d3 = compute_derivatives(pe_inst, positions)
        g = metric_tensor(d1)

        # 左图: g(p) vs p
        axes[0].plot(positions, g, color=color, label=label, linewidth=2, alpha=0.85)
        
        # 右图: log(g(p)) vs p (更清晰地展示指数差异)
        axes[1].semilogy(positions, g, color=color, label=label, linewidth=2, alpha=0.85)

    axes[0].set_xlabel("Position $p$", fontsize=12)
    axes[0].set_ylabel(r"$g(p) = \|\mathrm{d}PE/\mathrm{d}p\|^2$", fontsize=12)
    axes[0].set_title("度量张量 (线性坐标)", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Position $p$", fontsize=12)
    axes[1].set_ylabel(r"$g(p)$ (log scale)", fontsize=12)
    axes[1].set_title("度量张量 (对数坐标)", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, which='both')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "08_metric_tensor", MODULE)
    plt.close(fig)


def plot_arc_length(pe_dict: dict, max_len: int = 256):
    """
    [图 2] 弧长与测地距离。

    上排: 累积弧长 s(p)
    下排: 弧长速率 ds/dp = √g(p)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Arc Length & Geodesic Speed — 弧长与测地速度", fontsize=16, fontweight='bold')

    positions = np.arange(max_len).astype(float)
    
    arc_length_data = {}

    for pe_name, pe_inst in pe_dict.items():
        color = get_pe_color(pe_name)
        label = pe_inst.name

        gamma, d1, d2, d3 = compute_derivatives(pe_inst, positions)
        g = metric_tensor(d1)
        s = arc_length(g, positions)
        speed = np.sqrt(g)
        
        arc_length_data[pe_name] = {'s': s, 'speed': speed}

        # 上左: 累积弧长 s(p)
        axes[0, 0].plot(positions, s, color=color, label=label, linewidth=2)
        
        # 上右: 归一化弧长 s(p)/s(max)
        s_norm = s / (s[-1] + 1e-12)
        axes[0, 1].plot(positions, s_norm, color=color, label=label, linewidth=2)
        
        # 下左: 弧长速率 ds/dp
        axes[1, 0].plot(positions, speed, color=color, label=label, linewidth=2, alpha=0.85)
        
        # 下右: 速率对数
        axes[1, 1].semilogy(positions, speed, color=color, label=label, linewidth=2, alpha=0.85)

    # 参考线: 理想线性增长 s = p
    axes[0, 0].plot(positions, positions, 'k--', alpha=0.3, label='Linear ref.')
    
    axes[0, 0].set_xlabel("Position $p$", fontsize=11)
    axes[0, 0].set_ylabel(r"$s(p) = \int_0^p \sqrt{g(t)} \mathrm{d}t$", fontsize=11)
    axes[0, 0].set_title("累积弧长", fontsize=12)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel("Position $p$", fontsize=11)
    axes[0, 1].set_ylabel("$s(p) / s_{max}$", fontsize=11)
    axes[0, 1].set_title("归一化弧长", fontsize=12)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Position $p$", fontsize=11)
    axes[1, 0].set_ylabel(r"$\mathrm{d}s/\mathrm{d}p = \sqrt{g(p)}$", fontsize=11)
    axes[1, 0].set_title("弧长速率 (线性)", fontsize=12)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Position $p$", fontsize=11)
    axes[1, 1].set_ylabel(r"$\mathrm{d}s/\mathrm{d}p$ (log scale)", fontsize=11)
    axes[1, 1].set_title("弧长速率 (对数)", fontsize=12)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, which='both')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "08_arc_length", MODULE)
    plt.close(fig)
    
    return arc_length_data


def plot_curvature_analysis(pe_dict: dict, max_len: int = 256):
    """
    [图 3] 曲率分析: κ(p) 和 τ(p)。

    曲率高 → 曲线急转弯 → 局部位置信息变化剧烈
    挠率 → 曲线离开密切平面的程度
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Curvature & Torsion Analysis — 曲率与挠率", fontsize=16, fontweight='bold')

    positions = np.arange(max_len).astype(float)
    
    curvature_data = {}

    for pe_name, pe_inst in pe_dict.items():
        color = get_pe_color(pe_name)
        label = pe_inst.name

        gamma, d1, d2, d3 = compute_derivatives(pe_inst, positions)
        kappa = curvature(d1, d2)
        tau = torsion(d1, d2, d3)
        
        curvature_data[pe_name] = {
            'mean_kappa': float(np.mean(kappa)),
            'max_kappa': float(np.max(kappa)),
            'mean_tau': float(np.mean(np.abs(tau))),
        }

        # 上左: 曲率 κ(p)
        axes[0, 0].plot(positions, kappa, color=color, label=label, linewidth=1.5, alpha=0.85)
        
        # 上右: 曲率对数
        axes[0, 1].semilogy(positions, kappa + 1e-12, color=color, label=label, linewidth=1.5, alpha=0.85)
        
        # 下左: 挠率 τ(p)
        axes[1, 0].plot(positions, tau, color=color, label=label, linewidth=1.5, alpha=0.85)
        
        # 下右: 挠率绝对值
        axes[1, 1].semilogy(positions, np.abs(tau) + 1e-12, color=color, label=label, linewidth=1.5, alpha=0.85)

    axes[0, 0].set_xlabel("Position $p$", fontsize=11)
    axes[0, 0].set_ylabel(r"Curvature $\kappa(p)$", fontsize=11)
    axes[0, 0].set_title("曲率 (线性)", fontsize=12)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Position $p$", fontsize=11)
    axes[0, 1].set_ylabel(r"$\kappa(p)$ (log scale)", fontsize=11)
    axes[0, 1].set_title("曲率 (对数)", fontsize=12)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, which='both')

    axes[1, 0].set_xlabel("Position $p$", fontsize=11)
    axes[1, 0].set_ylabel(r"Torsion $\tau(p)$", fontsize=11)
    axes[1, 0].set_title("挠率 (带符号)", fontsize=12)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Position $p$", fontsize=11)
    axes[1, 1].set_ylabel(r"$|\tau(p)|$ (log scale)", fontsize=11)
    axes[1, 1].set_title("挠率绝对值 (对数)", fontsize=12)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, which='both')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "08_curvature_torsion", MODULE)
    plt.close(fig)
    
    return curvature_data


def plot_christoffel_heatmap(pe_dict: dict, max_len: int = 128):
    """
    [图 4] Christoffel 符号热力图。

    展示各 PE 方案的 (Position × Dimension) 联络结构。
    """
    setup_plot_style()
    n_pe = len(pe_dict)
    fig, axes = plt.subplots(1, n_pe, figsize=(5 * n_pe, 6))
    if n_pe == 1:
        axes = [axes]
    fig.suptitle("Christoffel-like Connection Structure — 联络结构热力图",
                 fontsize=16, fontweight='bold')

    positions = np.arange(max_len).astype(float)

    for idx, (pe_name, pe_inst) in enumerate(pe_dict.items()):
        ax = axes[idx]
        
        christoffel = christoffel_heatmap(pe_inst, positions)
        
        # 取对数以便可视化
        log_christoffel = np.log10(christoffel + 1e-12)
        
        im = ax.imshow(log_christoffel.T, aspect='auto', origin='lower',
                       cmap='viridis', extent=[0, max_len, 0, christoffel.shape[1]])
        ax.set_xlabel("Position $p$", fontsize=11)
        ax.set_ylabel("Dimension $k$", fontsize=11)
        ax.set_title(pe_inst.name, fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label=r"$\log_{10}(\Gamma_k(p))$")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "08_christoffel_heatmap", MODULE)
    plt.close(fig)


def plot_curvature_spectrum(pe_dict: dict, max_len: int = 256):
    """
    [图 5] 曲率谱: 各频率子空间的曲率贡献。

    对 Sinusoidal/RoPE: 2D 子空间曲率 ≈ ω_k
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvature Spectrum — 曲率谱分析", fontsize=16, fontweight='bold')

    positions = np.arange(max_len).astype(float)

    for pe_name, pe_inst in pe_dict.items():
        color = get_pe_color(pe_name)
        label = pe_inst.name

        freqs, kappa_spec = curvature_spectrum(pe_inst, positions)
        dim_idx = np.arange(len(freqs))

        # 左图: 曲率谱 vs 维度索引
        axes[0].plot(dim_idx, kappa_spec, 'o-', color=color, label=label,
                     linewidth=1.5, markersize=4, alpha=0.85)
        
        # 右图: 曲率谱 vs 频率 (log-log)
        positive_idx = freqs > 1e-10
        if np.any(positive_idx):
            axes[1].loglog(freqs[positive_idx], kappa_spec[positive_idx], 'o-',
                          color=color, label=label, linewidth=1.5, markersize=4, alpha=0.85)

    # 理论参考线: κ = ω (对于单位圆)
    axes[1].loglog([1e-4, 1], [1e-4, 1], 'k--', alpha=0.3, label=r'$\kappa = \omega$ (unit circle)')

    axes[0].set_xlabel("Subspace index $k$", fontsize=11)
    axes[0].set_ylabel(r"Mean curvature $\langle\kappa_k\rangle$", fontsize=11)
    axes[0].set_title("曲率谱 vs 子空间索引", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r"Frequency $\omega_k$", fontsize=11)
    axes[1].set_ylabel(r"Mean curvature $\langle\kappa_k\rangle$", fontsize=11)
    axes[1].set_title("曲率谱 vs 频率 (log-log)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, which='both')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "08_curvature_spectrum", MODULE)
    plt.close(fig)


def plot_geometry_dashboard(pe_dict: dict, max_len: int = 256):
    """
    [图 6] 微分几何综合仪表盘。

    4-panel: 度量张量 | 弧长 | 曲率 | 曲率谱
    """
    setup_plot_style()
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    fig.suptitle("Differential Geometry Dashboard — PE 微分几何综合分析",
                 fontsize=17, fontweight='bold', y=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    positions = np.arange(max_len).astype(float)
    summary = {}

    for pe_name, pe_inst in pe_dict.items():
        color = get_pe_color(pe_name)
        label = pe_inst.name

        gamma, d1, d2, d3 = compute_derivatives(pe_inst, positions)
        g = metric_tensor(d1)
        s = arc_length(g, positions)
        kappa = curvature(d1, d2)
        freqs, kappa_spec = curvature_spectrum(pe_inst, positions)

        # Panel A: 度量张量
        ax_a.semilogy(positions, g, color=color, label=label, linewidth=2)
        
        # Panel B: 累积弧长
        ax_b.plot(positions, s, color=color, label=label, linewidth=2)
        
        # Panel C: 曲率
        ax_c.semilogy(positions, kappa + 1e-12, color=color, label=label, linewidth=1.5, alpha=0.85)
        
        # Panel D: 曲率谱
        dim_idx = np.arange(len(freqs))
        ax_d.semilogy(dim_idx, kappa_spec + 1e-12, 'o-', color=color, label=label,
                      linewidth=1.5, markersize=3, alpha=0.85)

        summary[pe_name] = {
            'mean_metric': float(np.mean(g)),
            'total_arc_length': float(s[-1]),
            'mean_curvature': float(np.mean(kappa)),
            'max_curvature': float(np.max(kappa)),
        }

    # Panel A 装饰
    ax_a.set_xlabel("Position $p$", fontsize=11)
    ax_a.set_ylabel(r"$g(p)$ (log scale)", fontsize=11)
    ax_a.set_title("(A) Metric Tensor", fontsize=13, fontweight='bold')
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3, which='both')

    # Panel B 装饰
    ax_b.set_xlabel("Position $p$", fontsize=11)
    ax_b.set_ylabel(r"$s(p)$", fontsize=11)
    ax_b.set_title("(B) Arc Length", fontsize=13, fontweight='bold')
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3)

    # Panel C 装饰
    ax_c.set_xlabel("Position $p$", fontsize=11)
    ax_c.set_ylabel(r"$\kappa(p)$ (log scale)", fontsize=11)
    ax_c.set_title("(C) Curvature", fontsize=13, fontweight='bold')
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3, which='both')

    # Panel D 装饰
    ax_d.set_xlabel("Subspace index $k$", fontsize=11)
    ax_d.set_ylabel(r"$\langle\kappa_k\rangle$ (log scale)", fontsize=11)
    ax_d.set_title("(D) Curvature Spectrum", fontsize=13, fontweight='bold')
    ax_d.legend(fontsize=9)
    ax_d.grid(True, alpha=0.3, which='both')

    fig.subplots_adjust(top=0.92)
    save_figure(fig, "08_geometry_dashboard", MODULE)
    plt.close(fig)

    # ── 交互式 Plotly ──
    if HAS_PLOTLY:
        pfig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["(A) Metric Tensor", "(B) Arc Length",
                            "(C) Curvature", "(D) Curvature Spectrum"],
            vertical_spacing=0.12, horizontal_spacing=0.1
        )

        for pe_name, pe_inst in pe_dict.items():
            color = get_pe_color(pe_name)
            label = pe_inst.name

            gamma, d1, d2, d3 = compute_derivatives(pe_inst, positions)
            g = metric_tensor(d1)
            s = arc_length(g, positions)
            kappa = curvature(d1, d2)
            freqs, kappa_spec = curvature_spectrum(pe_inst, positions)

            # A: Metric tensor
            pfig.add_trace(go.Scatter(x=positions.tolist(), y=g.tolist(), name=label,
                                       line=dict(color=color, width=2),
                                       legendgroup=label, showlegend=True),
                           row=1, col=1)

            # B: Arc length
            pfig.add_trace(go.Scatter(x=positions.tolist(), y=s.tolist(), name=label,
                                       line=dict(color=color, width=2),
                                       legendgroup=label, showlegend=False),
                           row=1, col=2)

            # C: Curvature
            pfig.add_trace(go.Scatter(x=positions.tolist(), y=kappa.tolist(), name=label,
                                       line=dict(color=color, width=1.5),
                                       legendgroup=label, showlegend=False),
                           row=2, col=1)

            # D: Curvature spectrum
            dim_idx = np.arange(len(freqs))
            pfig.add_trace(go.Scatter(x=dim_idx.tolist(), y=kappa_spec.tolist(), name=label,
                                       mode='lines+markers',
                                       line=dict(color=color, width=1.5),
                                       marker=dict(size=4),
                                       legendgroup=label, showlegend=False),
                           row=2, col=2)

        pfig.update_yaxes(type="log", row=1, col=1)
        pfig.update_yaxes(type="log", row=2, col=1)
        pfig.update_yaxes(type="log", row=2, col=2)

        pfig.update_layout(
            title=dict(text="PE Differential Geometry Dashboard", font=dict(size=20)),
            height=800, width=1100, template='plotly_white'
        )
        save_plotly_html(pfig, "08_geometry_dashboard.html", MODULE)

    return summary


def generate_geometry_report(summary: dict = None):
    """
    生成微分几何分析 HTML 报告。
    """
    sections = [
        {
            'title': '1. 度量张量 (Metric Tensor)',
            'content': r"""
                <p>度量张量描述编码空间的"拉伸程度"：</p>
                <p>\[ g(p) = \left\| \frac{\mathrm{d}PE}{\mathrm{d}p} \right\|^2 
                   = \sum_k \left(\frac{\partial PE_k}{\partial p}\right)^2 \]</p>
                <p><b>物理意义</b>:</p>
                <ul>
                    <li>\( g(p) \) 大 → 编码空间在位置 \( p \) 附近"拉伸"严重 → 位置分辨率高</li>
                    <li>\( g(p) \) 小 → 压缩 → 相邻位置难以区分</li>
                    <li>与 Fisher 信息等价: \( g(p) = F(p) \) (Fisher-Rao 度量)</li>
                </ul>
                <p><b>发现</b>: Sinusoidal/RoPE 的度量张量近似常数（等距映射）；
                ALiBi 严格常数；LAPE 因幂律频率呈现非均匀分布。</p>
            """
        },
        {
            'title': '2. 弧长与测地距离 (Arc Length)',
            'content': r"""
                <p>弧长衡量编码空间中的"真实距离"：</p>
                <p>\[ s(p) = \int_0^p \sqrt{g(t)} \, \mathrm{d}t \]</p>
                <p>弧长速率 \( \mathrm{d}s/\mathrm{d}p = \sqrt{g(p)} \) 反映编码曲线的"速度"。</p>
                <p><b>诊断标准</b>:</p>
                <ul>
                    <li>线性增长 → 等速参数化 → 均匀位置采样</li>
                    <li>亚线性增长 → 后端位置"减速" → 长距离外推困难</li>
                    <li>超线性增长 → 后端位置"加速" → 高位置分辨率</li>
                </ul>
            """
        },
        {
            'title': '3. 曲率 (Curvature)',
            'content': r"""
                <p>Frenet-Serret 曲率衡量曲线的弯曲程度：</p>
                <p>\[ \kappa(p) = \frac{\sqrt{\|\gamma'\|^2\|\gamma''\|^2 - (\gamma' \cdot \gamma'')^2}}{\|\gamma'\|^3} \]</p>
                <p><b>几何意义</b>:</p>
                <ul>
                    <li>曲率高 → 曲线急转弯 → 局部位置信息变化剧烈</li>
                    <li>曲率低 → 曲线平直 → 位置信息缓慢变化</li>
                    <li>对于圆，曲率 = 1/半径；Sinusoidal 的各子空间是圆</li>
                </ul>
            """
        },
        {
            'title': '4. 挠率 (Torsion)',
            'content': r"""
                <p>挠率衡量曲线离开密切平面的程度（仅 3D+ 有意义）：</p>
                <p>\[ \tau(p) = \frac{(\gamma' \times \gamma'') \cdot \gamma'''}{\|\gamma' \times \gamma''\|^2} \]</p>
                <p><b>意义</b>:</p>
                <ul>
                    <li>挠率为零 → 曲线是平面曲线</li>
                    <li>挠率非零 → 曲线在 3D 空间中螺旋</li>
                    <li>PE 曲线在高维空间中，挠率反映其"螺旋程度"</li>
                </ul>
            """
        },
        {
            'title': '5. Christoffel 符号与联络',
            'content': r"""
                <p>Christoffel 符号描述平行移动在流形上的扭曲：</p>
                <p>\[ \Gamma^k_{ij} = \frac{1}{2} g^{kl}\left(\frac{\partial g_{il}}{\partial x^j} 
                   + \frac{\partial g_{jl}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^l}\right) \]</p>
                <p>对 1D 参数曲线，我们计算"局部曲率贡献"：</p>
                <p>\[ \Gamma_k(p) = \frac{|d^2 PE_k/dp^2|}{|d PE_k/dp| + \epsilon} \]</p>
                <p>热力图展示各维度在各位置的联络强度，揭示 PE 的内在几何结构。</p>
            """
        },
        {
            'title': '6. 曲率谱 (Curvature Spectrum)',
            'content': r"""
                <p>曲率谱将总曲率分解到各频率子空间：</p>
                <p>\[ \kappa_k = \text{Mean curvature of } (\sin(\omega_k p), \cos(\omega_k p)) \text{ subspace} \]</p>
                <p><b>理论预测</b>:</p>
                <ul>
                    <li>Sinusoidal/RoPE: 每个 2D 子空间是圆，曲率 \( \kappa_k = \omega_k \)</li>
                    <li>LAPE: 幂律频率 → 幂律曲率谱</li>
                    <li>ALiBi: 无周期结构 → 退化为线性</li>
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
                    <td>{data['mean_metric']:.2f}</td>
                    <td>{data['total_arc_length']:.1f}</td>
                    <td>{data['mean_curvature']:.4f}</td>
                    <td>{data['max_curvature']:.4f}</td>
                </tr>
            """
        sections.append({
            'title': '7. 数值汇总',
            'content': f"""
                <table border="1" cellpadding="8" cellspacing="0"
                       style="border-collapse: collapse; width: 100%; margin-top: 12px;">
                    <tr style="background: #f0f4f8;">
                        <th>PE 方案</th>
                        <th>平均度量</th>
                        <th>总弧长</th>
                        <th>平均曲率</th>
                        <th>最大曲率</th>
                    </tr>
                    {summary_rows}
                </table>
            """
        })

    generate_report_html(
        title="08 — PE 微分几何分析报告",
        sections=sections,
        module=MODULE,
        filename="08_differential_geometry_report.html"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  08 — PE Differential Geometry Analysis")
    print("=" * 60)

    dim = 64
    max_len = 256

    config = PEConfig(dim=dim, max_len=max_len)
    pe_dict = get_all_pe(config=config)
    logger = VizLogger("pe_analysis", "08_differential_geometry")

    # 1. 度量张量
    print("\n  [1/6] Metric tensor analysis...")
    plot_metric_tensor(pe_dict, max_len)
    print("        Saved 08_metric_tensor")
    logger.log_metric("metric_tensor", "completed")

    # 2. 弧长
    print("\n  [2/6] Arc length analysis...")
    arc_data = plot_arc_length(pe_dict, max_len)
    print("        Saved 08_arc_length")
    for name, data in arc_data.items():
        logger.log_metric(f"total_arc_{name}", float(data['s'][-1]))
        print(f"        {name}: total arc length = {data['s'][-1]:.1f}")

    # 3. 曲率分析
    print("\n  [3/6] Curvature & torsion analysis...")
    curvature_data = plot_curvature_analysis(pe_dict, max_len)
    print("        Saved 08_curvature_torsion")
    for name, data in curvature_data.items():
        logger.log_metric(f"mean_kappa_{name}", data['mean_kappa'])
        print(f"        {name}: mean κ = {data['mean_kappa']:.4f}, max κ = {data['max_kappa']:.4f}")

    # 4. Christoffel 热力图
    print("\n  [4/6] Christoffel heatmap...")
    plot_christoffel_heatmap(pe_dict, max_len=128)
    print("        Saved 08_christoffel_heatmap")
    logger.log_metric("christoffel_heatmap", "completed")

    # 5. 曲率谱
    print("\n  [5/6] Curvature spectrum...")
    plot_curvature_spectrum(pe_dict, max_len)
    print("        Saved 08_curvature_spectrum")
    logger.log_metric("curvature_spectrum", "completed")

    # 6. 综合仪表盘
    print("\n  [6/6] Geometry dashboard...")
    summary = plot_geometry_dashboard(pe_dict, max_len)
    print("        Saved 08_geometry_dashboard")
    for name, s in summary.items():
        print(f"        {name}: metric={s['mean_metric']:.2f}, arc={s['total_arc_length']:.1f}")

    # 报告
    print("\n  Generating report...")
    generate_geometry_report(summary)

    logger.save()

    print("\n" + "=" * 60)
    print("  08_differential_geometry complete!")
    print("  Static:       output/pe_analysis/08_*")
    print("  Interactive:  html/pe_analysis/08_*")
    print("=" * 60)


if __name__ == "__main__":
    main()
