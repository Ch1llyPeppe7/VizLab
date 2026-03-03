#!/usr/bin/env python3
"""
04 — PE Manifold Visualization (位置编码几何流形可视化)

核心思想:
    不同的位置编码方案将位置映射到高维空间中的不同几何流形：
    - Sinusoidal PE: 位于高维超环面 T^{d/2} 上 (各频率独立旋转构成的乘积环面)
    - RoPE:         在每个 2D 子空间独立旋转, 形成螺旋线 (helix) 结构
    - ALiBi:        不改变 embedding 空间, 几何上退化为点
    - LAPE:         改变频率衰减, 产生压缩/扭曲的环面

    在深层 FFN 传播后, 加性 PE 的流形结构逐渐坍缩,
    而 RoPE 由于每层重新施加旋转, 保持流形的几何完整性。

数学工具:
    - PCA: 主成分分析, 揭示全局线性结构
    - t-SNE: 局部结构保持降维 (perplexity ∝ 局部邻域大小)
    - UMAP: 拓扑数据分析启发的降维 (保持全局 + 局部)
    - MLE 内在维度: 基于 k-NN 的最大似然估计 (Levina & Bickel, 2005)
    - 测地距离: k-NN 图上的最短路径 vs 欧氏直线距离

Output:
    output/pe_analysis/   → 静态图 (PNG/PDF)
    html/pe_analysis/     → 交互式 HTML

Usage:
    python -m pe_analysis.04_manifold_visualization
    python run.py pe_analysis.manifold
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    PEConfig, get_pe, get_all_pe, SinusoidalPE, RoPE, ALiBi, LAPE,
    setup_plot_style, save_figure, add_math_annotation,
    save_plotly_html, generate_report_html, get_pe_color,
    PE_COLORS, VizLogger,
    spectral_entropy, effective_rank,
    random_weight_matrix, activation_fn, layer_norm,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

MODULE = "pe_analysis"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  简化模拟引擎 (复用 03 的设计)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimpleTransformerSim:
    """简化版 Transformer 前馈传播模拟器 (用于流形分析)"""
    
    def __init__(self, dim=64, n_layers=24, seed=42):
        self.dim = dim
        self.n_layers = n_layers
        self.rng = np.random.default_rng(seed)
        dim_ff = dim * 4
        self.layers = []
        for _ in range(n_layers):
            W1 = random_weight_matrix(dim, dim_ff, rng=self.rng)
            b1 = self.rng.standard_normal(dim_ff) * 0.01
            W2 = random_weight_matrix(dim_ff, dim, rng=self.rng)
            b2 = self.rng.standard_normal(dim) * 0.01
            self.layers.append((W1, b1, W2, b2))
    
    def forward_one(self, x, layer_idx):
        W1, b1, W2, b2 = self.layers[layer_idx]
        h = activation_fn(x @ W1 + b1, name='gelu')
        return layer_norm(x + h @ W2 + b2)
    
    def propagate_additive(self, pe_enc, n_layers=None):
        if n_layers is None:
            n_layers = self.n_layers
        token = self.rng.standard_normal(pe_enc.shape) * 0.1
        x = token + pe_enc
        traj = [x.copy()]
        for l in range(min(n_layers, self.n_layers)):
            x = self.forward_one(x, l)
            traj.append(x.copy())
        return traj
    
    def propagate_rope(self, rope_pe, positions, n_layers=None):
        if n_layers is None:
            n_layers = self.n_layers
        token = self.rng.standard_normal((len(positions), self.dim)) * 0.1
        x = token.copy()
        x_rot = rope_pe.apply_rotary(x, positions)
        traj = [x_rot.copy()]
        for l in range(min(n_layers, self.n_layers)):
            x = self.forward_one(x, l)
            x_rot = rope_pe.apply_rotary(x, positions)
            traj.append(x_rot.copy())
        return traj
    
    def propagate_plain(self, positions, n_layers=None):
        if n_layers is None:
            n_layers = self.n_layers
        token = self.rng.standard_normal((len(positions), self.dim)) * 0.1
        x = token.copy()
        traj = [x.copy()]
        for l in range(min(n_layers, self.n_layers)):
            x = self.forward_one(x, l)
            traj.append(x.copy())
        return traj


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def estimate_intrinsic_dimension(X, k=10):
    """
    MLE 内在维度估计 (Levina & Bickel, 2005)
    
    对每个点, 基于 k 近邻距离的对数比值估计局部维度:
        d_hat = [ (1/(k-1)) sum_{j=1}^{k-1} log(T_k / T_j) ]^{-1}
    
    其中 T_j 是到第 j 近邻的距离。
    
    Args:
        X: [N, D] 数据矩阵
        k: 近邻数
    Returns:
        (mean_dim, per_point_dims) — 平均维度和每点的局部维度估计
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # 去掉自身 (距离=0)
    distances = np.maximum(distances, 1e-15)
    
    T_k = distances[:, -1]  # 第 k 近邻距离
    log_ratios = np.log(T_k[:, None] / distances[:, :-1])  # [N, k-1]
    mean_log_ratios = np.mean(log_ratios, axis=1)  # [N,]
    
    per_point = 1.0 / (mean_log_ratios + 1e-15)
    per_point = np.clip(per_point, 0, X.shape[1])  # 上界为原始维度
    
    return float(np.mean(per_point)), per_point


def compute_geodesic_vs_euclidean(X, k=8):
    """
    计算测地距离 (k-NN 图最短路径) 与欧氏距离的比较。
    
    测地距离 / 欧氏距离 > 1 表示数据在弯曲流形上。
    比值越大, 流形弯曲程度越高。
    
    Args:
        X: [N, D] 数据矩阵
        k: k-NN 图的邻居数
    Returns:
        (geo_dists, euc_dists, ratios) 三个矩阵
    """
    N = X.shape[0]
    
    # 1. 构建 k-NN 图
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nn.kneighbors(X)
    
    # 2. 构建稀疏距离矩阵
    from scipy.sparse import lil_matrix
    graph = lil_matrix((N, N))
    for i in range(N):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            graph[i, j] = d
            graph[j, i] = d
    
    # 3. 最短路径 (Dijkstra)
    geo_dists = shortest_path(graph.tocsr(), directed=False)
    
    # 4. 欧氏距离
    euc_dists = squareform(pdist(X))
    
    # 5. 比值 (避免除零)
    mask = euc_dists > 1e-10
    ratios = np.ones_like(euc_dists)
    ratios[mask] = geo_dists[mask] / euc_dists[mask]
    
    return geo_dists, euc_dists, ratios


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. PCA 降维 — 2D/3D 几何结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_pca_geometry(encodings: dict, positions: np.ndarray):
    """
    PCA 降维到 2D 和 3D, 展示各 PE 方案的几何结构。
    
    预期:
    - Sinusoidal: 高维环面 T^{d/2} 的 PCA 投影 → 椭圆/Lissajous 曲线
    - RoPE:       与 Sinusoidal 结构相似 (频率相同), 但 encode() 返回 [cos,sin]
    - ALiBi:      不改变 embedding, 无几何结构 (点/随机云)
    - LAPE:       不同频率衰减导致的压缩环面
    """
    setup_plot_style()
    fig = plt.figure(figsize=(20, 10))
    
    pe_names = list(encodings.keys())
    n_pe = len(pe_names)
    
    for idx, name in enumerate(pe_names):
        enc = encodings[name]  # [N, dim]
        color = get_pe_color(name)
        
        # 检查是否为全零/常数矩阵 (如 ALiBi)
        is_degenerate = np.std(enc) < 1e-10
        
        ax2d = fig.add_subplot(2, n_pe, idx + 1)
        ax3d = fig.add_subplot(2, n_pe, n_pe + idx + 1, projection='3d')
        
        if is_degenerate:
            # ALiBi: 全零编码, 无法做 PCA
            ax2d.text(0.5, 0.5, f"{name}\n(No embedding PE;\nbias-only scheme)",
                      transform=ax2d.transAxes, ha='center', va='center',
                      fontsize=11, fontweight='bold', color='gray')
            ax2d.set_title(f"{name}\nPCA 2D (degenerate)", fontsize=11, fontweight='bold')
            ax3d.text2D(0.5, 0.5, "No geometry\n(bias-only)",
                        transform=ax3d.transAxes, ha='center', va='center',
                        fontsize=10, color='gray')
            ax3d.set_title(f"{name} PCA 3D", fontsize=10)
            continue
        
        # PCA 2D
        pca2 = PCA(n_components=2)
        proj2 = pca2.fit_transform(enc)
        
        scatter = ax2d.scatter(proj2[:, 0], proj2[:, 1],
                               c=positions[:len(enc)], cmap='viridis',
                               s=15, alpha=0.7, edgecolors='none')
        ax2d.set_title(f"{name}\nPCA 2D (var: {pca2.explained_variance_ratio_.sum():.2%})",
                       fontsize=11, fontweight='bold')
        ax2d.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
        ax2d.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
        plt.colorbar(scatter, ax=ax2d, label='Position', shrink=0.8)
        
        # PCA 3D
        pca3 = PCA(n_components=min(3, enc.shape[1]))
        proj3 = pca3.fit_transform(enc)
        
        if proj3.shape[1] >= 3:
            ax3d.scatter(proj3[:, 0], proj3[:, 1], proj3[:, 2],
                         c=positions[:len(enc)], cmap='viridis',
                         s=8, alpha=0.6)
            ax3d.set_xlabel(f"PC1")
            ax3d.set_ylabel(f"PC2")
            ax3d.set_zlabel(f"PC3")
        else:
            ax3d.scatter(proj3[:, 0], proj3[:, 1], np.zeros(len(proj3)),
                         c=positions[:len(enc)], cmap='viridis',
                         s=8, alpha=0.6)
        ax3d.set_title(f"{name} PCA 3D", fontsize=10)
    
    fig.suptitle("PE Manifold Geometry — PCA Projection\n"
                 "Color = position index (dark=0, bright=max)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "04_pca_geometry", MODULE)
    plt.close(fig)
    
    # Plotly 3D 交互图
    if HAS_PLOTLY:
        pfig = make_subplots(
            rows=1, cols=n_pe,
            specs=[[{'type': 'scatter3d'}] * n_pe],
            subplot_titles=pe_names,
        )
        for idx, name in enumerate(pe_names):
            enc = encodings[name]
            pca3 = PCA(n_components=min(3, enc.shape[1]))
            proj3 = pca3.fit_transform(enc)
            color_arr = positions[:len(enc)]
            if proj3.shape[1] < 3:
                proj3 = np.column_stack([proj3, np.zeros(len(proj3))])
            pfig.add_trace(go.Scatter3d(
                x=proj3[:, 0], y=proj3[:, 1], z=proj3[:, 2],
                mode='markers+lines',
                marker=dict(size=3, color=color_arr, colorscale='Viridis',
                            showscale=(idx == n_pe - 1)),
                line=dict(color=get_pe_color(name), width=1),
                name=name,
            ), row=1, col=idx + 1)
        pfig.update_layout(
            title="PE Manifold — 3D PCA (Interactive)",
            template="plotly_white", width=400 * n_pe, height=500,
        )
        save_plotly_html(pfig, "04_pca_3d.html", MODULE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. t-SNE / UMAP 降维对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_tsne_umap(encodings: dict, positions: np.ndarray):
    """
    t-SNE 和 UMAP 降维对比, 展示各 PE 方案的局部/全局结构。
    
    t-SNE 擅长保持局部邻域结构 (但可能扭曲全局距离);
    UMAP 同时保持局部和全局拓扑。
    
    两者联合使用可以判断 PE 流形的拓扑性质:
    - 如果 t-SNE 和 UMAP 都展示出连续环状结构 → 流形是闭合的 (如 S^1)
    - 如果 UMAP 保持连续线段 → 流形是开的 (如 R^1 的子集)
    """
    pe_names = list(encodings.keys())
    n_pe = len(pe_names)
    n_methods = 2 if HAS_UMAP else 1
    
    setup_plot_style()
    fig, axes = plt.subplots(n_methods, n_pe, figsize=(5 * n_pe, 5 * n_methods))
    if n_methods == 1:
        axes = axes[np.newaxis, :]
    
    method_names = ['t-SNE']
    if HAS_UMAP:
        method_names.append('UMAP')
    
    for idx, name in enumerate(pe_names):
        enc = encodings[name]
        pos_colors = positions[:len(enc)]
        is_degenerate = np.std(enc) < 1e-10
        
        # t-SNE
        if is_degenerate:
            # 对退化编码(如ALiBi)加微小扰动以避免t-SNE崩溃
            enc_tsne = enc + np.random.default_rng(42).standard_normal(enc.shape) * 1e-6
        else:
            enc_tsne = enc
        perplexity = min(30, len(enc_tsne) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                     max_iter=1000)
        proj_tsne = tsne.fit_transform(enc_tsne)
        
        ax = axes[0, idx]
        scatter = ax.scatter(proj_tsne[:, 0], proj_tsne[:, 1],
                             c=pos_colors, cmap='viridis',
                             s=15, alpha=0.7, edgecolors='none')
        ax.set_title(f"{name} — t-SNE", fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("t-SNE", fontsize=12)
        plt.colorbar(scatter, ax=ax, label='Position', shrink=0.8)
        
        # UMAP
        if HAS_UMAP:
            enc_umap = enc if not is_degenerate else enc + np.random.default_rng(42).standard_normal(enc.shape) * 1e-6
            reducer = umap.UMAP(n_components=2, n_neighbors=15,
                                min_dist=0.1, random_state=42)
            proj_umap = reducer.fit_transform(enc_umap)
            
            ax = axes[1, idx]
            scatter = ax.scatter(proj_umap[:, 0], proj_umap[:, 1],
                                 c=pos_colors, cmap='viridis',
                                 s=15, alpha=0.7, edgecolors='none')
            ax.set_title(f"{name} — UMAP", fontsize=11, fontweight='bold')
            if idx == 0:
                ax.set_ylabel("UMAP", fontsize=12)
            plt.colorbar(scatter, ax=ax, label='Position', shrink=0.8)
    
    methods_str = " & ".join(method_names)
    fig.suptitle(f"PE Manifold — {methods_str} Visualization\n"
                 f"Color = position index",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "04_tsne_umap", MODULE)
    plt.close(fig)
    
    # Plotly 交互版
    if HAS_PLOTLY:
        pfig = make_subplots(rows=1, cols=n_pe,
                             subplot_titles=[f"{n} — t-SNE" for n in pe_names])
        for idx, name in enumerate(pe_names):
            enc = encodings[name]
            if np.std(enc) < 1e-10:
                enc = enc + np.random.default_rng(42).standard_normal(enc.shape) * 1e-6
            perplexity = min(30, len(enc) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            proj = tsne.fit_transform(enc)
            pfig.add_trace(go.Scatter(
                x=proj[:, 0], y=proj[:, 1], mode='markers',
                marker=dict(size=5, color=positions[:len(enc)],
                            colorscale='Viridis',
                            showscale=(idx == n_pe - 1)),
                name=name,
            ), row=1, col=idx + 1)
        pfig.update_layout(title="PE Manifold — t-SNE (Interactive)",
                          template="plotly_white",
                          width=350 * n_pe, height=400)
        save_plotly_html(pfig, "04_tsne.html", MODULE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. 深层传播后的流形坍缩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_manifold_collapse(trajectories: dict, positions: np.ndarray):
    """
    PCA 跟踪 PE 流形在深层 FFN 传播后的几何变化。
    
    选取关键层 (0, 6, 12, 23), 对每层的 embedding 做 PCA 投影。
    
    预期:
    - Sinusoidal: 初始层有清晰几何结构, 深层坍缩为无结构点云
    - RoPE:       由于每层重新旋转, 保持结构
    - ALiBi:      始终无结构
    - LAPE:       类似 Sinusoidal 但衰减更快
    """
    layers_to_show = [0, 6, 12, 23]
    pe_names = list(trajectories.keys())
    n_pe = len(pe_names)
    n_layers = len(layers_to_show)
    
    setup_plot_style()
    fig, axes = plt.subplots(n_pe, n_layers, figsize=(4.5 * n_layers, 4 * n_pe))
    fig.suptitle("Manifold Collapse Through Layers — PCA 2D Projection\n"
                 "Color = position index",
                 fontsize=16, fontweight='bold', y=1.02)
    
    for row, name in enumerate(pe_names):
        traj = trajectories[name]
        
        for col, layer in enumerate(layers_to_show):
            if layer >= len(traj):
                layer = len(traj) - 1
            
            ax = axes[row, col] if n_pe > 1 else axes[col]
            emb = traj[layer]
            
            # PCA 2D
            pca = PCA(n_components=2)
            proj = pca.fit_transform(emb)
            var_ratio = pca.explained_variance_ratio_.sum()
            
            scatter = ax.scatter(proj[:, 0], proj[:, 1],
                                 c=positions[:len(emb)], cmap='viridis',
                                 s=12, alpha=0.6, edgecolors='none')
            
            if col == 0:
                ax.set_ylabel(f"{name}", fontsize=11, fontweight='bold')
            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=11)
            
            # 标注解释方差比
            ax.text(0.05, 0.95, f"Var: {var_ratio:.1%}",
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # 标注有效秩
            er = effective_rank(emb)
            ax.text(0.05, 0.82, f"eRank: {er:.1f}",
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    save_figure(fig, "04_manifold_collapse", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. Explained Variance & 内在维度
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_intrinsic_dimension(encodings: dict, trajectories: dict,
                             positions: np.ndarray):
    """
    分析各 PE 方案的内在维度:
    
    Panel 1: PCA 累积解释方差曲线
        快速下降 → 低内在维度 (数据集中在低维子空间)
        缓慢下降 → 高内在维度
    
    Panel 2: MLE 内在维度估计 (初始编码)
        各 PE 的本征维度
    
    Panel 3: 内在维度随层数的变化
        加性 PE 的维度应随深度增加 (从结构化退化为高维噪声)
        RoPE 应保持稳定
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    dim_results = {}
    
    # Panel 1: PCA 累积方差
    ax1 = axes[0]
    for name, enc in encodings.items():
        color = get_pe_color(name)
        n_comp = min(enc.shape)
        pca = PCA(n_components=n_comp)
        pca.fit(enc)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax1.plot(range(1, n_comp + 1), cumvar, '-', color=color,
                 linewidth=2, label=name)
        
        # 标注 90% 和 99% 方差对应的维度
        for threshold in [0.9, 0.99]:
            dim_at = np.searchsorted(cumvar, threshold) + 1
            ax1.axvline(x=dim_at, color=color, ls=':', alpha=0.3)
    
    ax1.axhline(y=0.9, color='gray', ls='--', alpha=0.4, label='90%')
    ax1.axhline(y=0.99, color='gray', ls='-.', alpha=0.4, label='99%')
    ax1.set_xlabel("Number of Principal Components")
    ax1.set_ylabel("Cumulative Explained Variance")
    ax1.set_title("PCA Explained Variance", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.05)
    
    # Panel 2: MLE 内在维度 (初始编码)
    ax2 = axes[1]
    names = list(encodings.keys())
    colors = [get_pe_color(n) for n in names]
    int_dims = []
    for name in names:
        enc = encodings[name]
        k = min(10, len(enc) - 1)
        mean_dim, _ = estimate_intrinsic_dimension(enc, k=k)
        int_dims.append(mean_dim)
        dim_results[name] = {'initial_id': mean_dim}
    
    bars = ax2.bar(range(len(names)), int_dims, color=colors,
                   edgecolor='black', alpha=0.85)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel("Estimated Intrinsic Dimension")
    ax2.set_title("MLE Intrinsic Dimension\n(Levina & Bickel, 2005)",
                  fontsize=13, fontweight='bold')
    for bar, val in zip(bars, int_dims):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel 3: 内在维度随层数变化
    ax3 = axes[2]
    for name, traj in trajectories.items():
        color = get_pe_color(name)
        layer_dims = []
        layers = []
        for l in range(0, len(traj), max(1, len(traj) // 12)):
            emb = traj[l]
            k = min(10, len(emb) - 1)
            mean_dim, _ = estimate_intrinsic_dimension(emb, k=k)
            layer_dims.append(mean_dim)
            layers.append(l)
        ax3.plot(layers, layer_dims, 'o-', color=color, linewidth=2,
                 markersize=4, label=name)
        dim_results[name]['layer_dims'] = layer_dims
        dim_results[name]['layers'] = layers
    
    ax3.set_xlabel("Layer $l$")
    ax3.set_ylabel("Intrinsic Dimension")
    ax3.set_title("Intrinsic Dimension Evolution", fontsize=13, fontweight='bold')
    ax3.legend()
    
    plt.tight_layout()
    save_figure(fig, "04_intrinsic_dimension", MODULE)
    plt.close(fig)
    
    return dim_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. 测地距离 vs 欧氏距离
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_geodesic_analysis(encodings: dict, positions: np.ndarray):
    """
    比较测地距离 (k-NN 图最短路径) 与欧氏距离。
    
    Panel 1: 测地距离 vs 欧氏距离散点图
        如果点落在 y=x 线上 → 数据在平坦空间
        如果点在 y=x 上方 → 数据在弯曲流形上
    
    Panel 2: 测地/欧氏距离比值的分布
        比值 ≈ 1 → 平坦
        比值 >> 1 → 高度弯曲
    
    Panel 3: 相邻位置的测地距离 vs 位置差
        应该是单调的 (位置差越大, 测地距离越大)
        如果不单调 → 流形有折叠/扭曲
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    geo_results = {}
    
    for idx, (name, enc) in enumerate(encodings.items()):
        color = get_pe_color(name)
        
        # 只取前 100 个位置 (测地距离计算耗时)
        n_pts = min(100, len(enc))
        enc_sub = enc[:n_pts]
        
        try:
            geo_d, euc_d, ratios = compute_geodesic_vs_euclidean(enc_sub, k=8)
            
            # 取上三角
            mask = np.triu(np.ones(n_pts, dtype=bool), k=1)
            geo_flat = geo_d[mask]
            euc_flat = euc_d[mask]
            ratio_flat = ratios[mask]
            
            # 过滤 inf 值
            valid = np.isfinite(geo_flat) & np.isfinite(euc_flat)
            geo_flat = geo_flat[valid]
            euc_flat = euc_flat[valid]
            ratio_flat = ratio_flat[valid]
            
            mean_ratio = np.mean(ratio_flat) if len(ratio_flat) > 0 else 1.0
            geo_results[name] = {
                'mean_ratio': mean_ratio,
                'median_ratio': float(np.median(ratio_flat)) if len(ratio_flat) > 0 else 1.0,
            }
            
        except Exception as e:
            print(f"    Warning: Geodesic computation failed for {name}: {e}")
            geo_results[name] = {'mean_ratio': 1.0, 'median_ratio': 1.0}
            ax = axes[idx // 2, idx % 2]
            ax.text(0.5, 0.5, f"{name}\n(degenerate / failed)",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color='gray')
            ax.set_title(f"{name}", fontsize=11, fontweight='bold')
            continue
        
        # 子图: 散点图 (geo vs euc)
        ax = axes[idx // 2, idx % 2]
        
        if len(geo_flat) == 0 or len(euc_flat) == 0:
            ax.text(0.5, 0.5, f"{name}\n(all zero distances)",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color='gray')
            ax.set_title(f"{name}\n(degenerate)", fontsize=11, fontweight='bold')
            continue
        
        # 子采样以避免过多点
        n_show = min(5000, len(geo_flat))
        sample_idx = np.random.default_rng(42).choice(len(geo_flat), n_show, replace=False)
        
        ax.scatter(euc_flat[sample_idx], geo_flat[sample_idx],
                   color=color, s=2, alpha=0.3)
        
        # y=x 参考线
        max_val = max(euc_flat.max(), geo_flat.max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='y=x')
        
        ax.set_xlabel("Euclidean Distance")
        ax.set_ylabel("Geodesic Distance")
        ax.set_title(f"{name}\nMean Geo/Euc = {mean_ratio:.3f}",
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
    
    fig.suptitle("Geodesic vs Euclidean Distance\n"
                 "Points above y=x → curved manifold",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "04_geodesic_analysis", MODULE)
    plt.close(fig)
    
    # 柱状图: 平均比值
    setup_plot_style()
    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    names = list(geo_results.keys())
    colors = [get_pe_color(n) for n in names]
    mean_ratios = [geo_results[n]['mean_ratio'] for n in names]
    
    bars = ax.bar(range(len(names)), mean_ratios, color=colors,
                  edgecolor='black', alpha=0.85)
    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5, label='Flat space')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Mean Geodesic / Euclidean Ratio")
    ax.set_title("Manifold Curvature Indicator\n"
                 "(Ratio > 1 implies curved manifold)",
                 fontsize=13, fontweight='bold')
    ax.legend()
    for bar, val in zip(bars, mean_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig2, "04_geodesic_ratios", MODULE)
    plt.close(fig2)
    
    return geo_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. 综合仪表盘
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_manifold_dashboard(dim_results: dict, geo_results: dict,
                            encodings: dict):
    """
    综合仪表盘: 流形几何的关键指标一览。
    
    Panel 1: 内在维度 (初始编码)
    Panel 2: 测地/欧氏比值
    Panel 3: PCA 90% 方差维度
    Panel 4: 有效秩
    """
    names = list(dim_results.keys())
    colors = [get_pe_color(n) for n in names]
    
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PE Manifold Geometry — Dashboard",
                 fontsize=18, fontweight='bold')
    
    x = np.arange(len(names))
    bar_w = 0.6
    
    # Panel 1: 内在维度
    ax1 = axes[0, 0]
    id_vals = [dim_results[n]['initial_id'] for n in names]
    bars = ax1.bar(x, id_vals, bar_w, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Intrinsic Dimension")
    ax1.set_title("MLE Intrinsic Dimension", fontsize=12)
    for bar, val in zip(bars, id_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel 2: 测地/欧氏比值
    ax2 = axes[0, 1]
    geo_vals = [geo_results.get(n, {}).get('mean_ratio', 1.0) for n in names]
    bars = ax2.bar(x, geo_vals, bar_w, color=colors, edgecolor='black', alpha=0.85)
    ax2.axhline(y=1.0, color='gray', ls='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Geo/Euc Ratio")
    ax2.set_title("Manifold Curvature", fontsize=12)
    for bar, val in zip(bars, geo_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel 3: PCA 90% 方差维度
    ax3 = axes[1, 0]
    pca_dims = []
    for name in names:
        enc = encodings[name]
        n_comp = min(enc.shape)
        pca = PCA(n_components=n_comp)
        pca.fit(enc)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dim90 = int(np.searchsorted(cumvar, 0.9)) + 1
        pca_dims.append(dim90)
    bars = ax3.bar(x, pca_dims, bar_w, color=colors, edgecolor='black', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylabel("Dimensions for 90% Variance")
    ax3.set_title("PCA Dimensionality (90% Threshold)", fontsize=12)
    for bar, val in zip(bars, pca_dims):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel 4: 有效秩
    ax4 = axes[1, 1]
    eranks = [effective_rank(encodings[n]) for n in names]
    bars = ax4.bar(x, eranks, bar_w, color=colors, edgecolor='black', alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.set_ylabel("Effective Rank")
    ax4.set_title("Embedding Effective Rank", fontsize=12)
    for bar, val in zip(bars, eranks):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, "04_manifold_dashboard", MODULE)
    plt.close(fig)
    
    # Plotly 仪表盘
    if HAS_PLOTLY:
        pfig = make_subplots(rows=2, cols=2,
            subplot_titles=["Intrinsic Dimension", "Manifold Curvature",
                           "PCA 90% Dim", "Effective Rank"])
        for i, n in enumerate(names):
            color = get_pe_color(n)
            pfig.add_trace(go.Bar(x=[n], y=[id_vals[i]], marker_color=color,
                                  name=n, legendgroup=n,
                                  showlegend=True), row=1, col=1)
            pfig.add_trace(go.Bar(x=[n], y=[geo_vals[i]], marker_color=color,
                                  name=n, legendgroup=n,
                                  showlegend=False), row=1, col=2)
            pfig.add_trace(go.Bar(x=[n], y=[pca_dims[i]], marker_color=color,
                                  name=n, legendgroup=n,
                                  showlegend=False), row=2, col=1)
            pfig.add_trace(go.Bar(x=[n], y=[eranks[i]], marker_color=color,
                                  name=n, legendgroup=n,
                                  showlegend=False), row=2, col=2)
        pfig.update_layout(template="plotly_white", width=1000, height=700,
                          title="PE Manifold Geometry — Dashboard")
        save_plotly_html(pfig, "04_manifold_dashboard.html", MODULE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. Singular Value Spectrum — 奇异值谱
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_singular_value_spectrum(encodings: dict, trajectories: dict,
                                 positions: np.ndarray):
    """
    分析各 PE 编码矩阵的奇异值谱。
    
    Panel 1: 初始编码的奇异值谱 (归一化)
    Panel 2: 初始 vs 深层的奇异值谱对比
    
    奇异值分布反映了数据的维度结构:
    - 几个大奇异值 + 快速衰减 → 低秩结构 (数据在低维子空间)
    - 平坦分布 → 数据均匀分布在各维度
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: 初始编码的 SVD
    ax1 = axes[0]
    for name, enc in encodings.items():
        color = get_pe_color(name)
        svs = np.linalg.svd(enc, compute_uv=False)
        if svs[0] < 1e-15:
            # 退化矩阵 (如 ALiBi 全零编码)
            ax1.axhline(y=1e-15, color=color, ls=':', alpha=0.5, label=f'{name} (degenerate)')
            continue
        svs_norm = svs / svs[0]  # 归一化
        ax1.semilogy(range(1, len(svs) + 1), svs_norm, '-', color=color,
                     linewidth=2, label=name, alpha=0.85)
    
    ax1.set_xlabel("Singular Value Index $k$")
    ax1.set_ylabel("Normalized Singular Value $\\sigma_k / \\sigma_1$")
    ax1.set_title("Singular Value Spectrum — Initial Encoding",
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Layer 0 vs Layer 23 对比
    ax2 = axes[1]
    for name, traj in trajectories.items():
        color = get_pe_color(name)
        # Layer 0
        svs0 = np.linalg.svd(traj[0], compute_uv=False)
        svs0_norm = svs0 / svs0[0]
        ax2.semilogy(range(1, len(svs0) + 1), svs0_norm, '-', color=color,
                     linewidth=2, alpha=0.5, label=f'{name} L0')
        # Final layer
        svs_f = np.linalg.svd(traj[-1], compute_uv=False)
        svs_f_norm = svs_f / svs_f[0]
        ax2.semilogy(range(1, len(svs_f) + 1), svs_f_norm, '--', color=color,
                     linewidth=2, alpha=0.9, label=f'{name} L{len(traj)-1}')
    
    ax2.set_xlabel("Singular Value Index $k$")
    ax2.set_ylabel("Normalized $\\sigma_k / \\sigma_1$")
    ax2.set_title("SVD Spectrum: Layer 0 (solid) vs Final (dashed)",
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "04_svd_spectrum", MODULE)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  综合 HTML 报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_manifold_report():
    """生成综合 HTML 报告"""
    sections = [
        {
            'title': '核心思想',
            'content': (
                '不同的位置编码方案将位置映射到高维空间中的不同<strong>几何流形</strong>：'
                '<ul>'
                '<li><strong>Sinusoidal PE</strong>: 高维超环面 \\(\\mathbb{T}^{d/2}\\)，'
                '由 \\(d/2\\) 个独立频率的正弦/余弦构成</li>'
                '<li><strong>RoPE</strong>: \\(d/2\\) 个独立 SO(2) 旋转构成的螺旋线结构</li>'
                '<li><strong>ALiBi</strong>: 不修改 embedding，几何上无结构</li>'
                '<li><strong>LAPE</strong>: 修改频率衰减率，产生压缩的环面</li>'
                '</ul>'
            )
        },
        {
            'title': '1. PCA 投影 — 全局几何结构',
            'content': (
                '使用主成分分析 (PCA) 将高维 PE 编码投影到 2D/3D 空间。'
                '<br><br>'
                'PCA 找到数据方差最大的方向，保持全局线性结构。'
                '累积解释方差比反映了数据的有效维度：'
                '\\[\\text{var}(k) = \\frac{\\sum_{i=1}^k \\lambda_i}{\\sum_{i=1}^D \\lambda_i}\\]'
            )
        },
        {
            'title': '2. t-SNE & UMAP — 局部/全局拓扑',
            'content': (
                '<strong>t-SNE</strong> (van der Maaten, 2008) 通过最小化高维与低维邻域分布的 KL 散度，'
                '保持局部结构。<br><br>'
                '<strong>UMAP</strong> (McInnes, 2018) 基于拓扑数据分析的 Rips 复形，'
                '同时保持局部和全局拓扑结构。'
            )
        },
        {
            'title': '3. 流形坍缩 — 深层传播效应',
            'content': (
                '跟踪 PE 流形在深层 FFN 传播后的几何变化。'
                '<br><br>'
                '加性 PE 的流形应在深层坍缩（从清晰的环面结构退化为无结构点云），'
                '而 RoPE 由于每层重新施加旋转，保持结构完整性。'
            )
        },
        {
            'title': '4. 内在维度分析',
            'content': (
                '使用 MLE 方法 (Levina & Bickel, 2005) 估计数据的内在维度：'
                '\\[\\hat{d}_k(x) = \\left[\\frac{1}{k-1}\\sum_{j=1}^{k-1}\\log\\frac{T_k(x)}{T_j(x)}\\right]^{-1}\\]'
                '其中 \\(T_j(x)\\) 是到第 \\(j\\) 近邻的距离。'
                '<br><br>'
                '低内在维度 → PE 编码集中在低维子空间（结构化）。'
                '<br>'
                '高内在维度 → PE 编码分布在高维空间（噪声化/混沌化）。'
            )
        },
        {
            'title': '5. 测地距离 vs 欧氏距离',
            'content': (
                '通过 k-NN 图上的最短路径 (Dijkstra) 计算测地距离，'
                '并与欧氏直线距离比较。'
                '<br><br>'
                '测地/欧氏比值 \\(\\approx 1\\) → 数据在平坦空间；<br>'
                '比值 \\(\\gg 1\\) → 数据在弯曲流形上。'
            )
        },
        {
            'title': '6. 奇异值谱分析',
            'content': (
                'PE 编码矩阵的奇异值谱反映维度结构：'
                '<br>'
                '几个大奇异值 + 快速衰减 → 低秩/低维结构；<br>'
                '平坦分布 → 数据均匀填充各维度（全秩）。'
                '<br><br>'
                '深层传播后奇异值谱的变化揭示了 FFN 对 PE 几何结构的影响。'
            )
        },
    ]
    
    generate_report_html(
        title="04 — PE Manifold Visualization: Geometric Structure of Position Encodings",
        sections=sections,
        module=MODULE,
        filename="04_manifold_report.html"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  04 — PE Manifold Visualization")
    print("  Geometric Structure of Position Encodings")
    print("=" * 60)
    
    # 配置
    dim = 64
    seq_len = 256       # 更多位置以获得更好的流形展示
    n_layers = 24
    seed = 42
    
    config = PEConfig(dim=dim, max_len=seq_len)
    positions = np.arange(seq_len)
    
    # 获取 PE 实例
    pe_dict = get_all_pe(config=config)
    
    # 日志
    logger = VizLogger("pe_analysis", "04_manifold_visualization")
    
    # ── 生成编码 ──
    print("\n  [0/7] Generating PE encodings...")
    encodings = {}
    for name, pe in pe_dict.items():
        encodings[name] = pe.encode(positions)
        print(f"        {name}: shape={encodings[name].shape}")
    
    # ── 生成深层传播轨迹 ──
    print("\n  [0/7] Generating deep propagation trajectories...")
    sim = SimpleTransformerSim(dim=dim, n_layers=n_layers, seed=seed)
    trajectories = {}
    
    print("        -> Sinusoidal (additive)...")
    sim1 = SimpleTransformerSim(dim=dim, n_layers=n_layers, seed=seed)
    trajectories['sinusoidal'] = sim1.propagate_additive(encodings['sinusoidal'])
    
    print("        -> RoPE (multiplicative)...")
    sim2 = SimpleTransformerSim(dim=dim, n_layers=n_layers, seed=seed)
    trajectories['rope'] = sim2.propagate_rope(pe_dict['rope'], positions)
    
    print("        -> ALiBi (plain)...")
    sim3 = SimpleTransformerSim(dim=dim, n_layers=n_layers, seed=seed)
    trajectories['alibi'] = sim3.propagate_plain(positions)
    
    print("        -> LAPE (additive)...")
    sim4 = SimpleTransformerSim(dim=dim, n_layers=n_layers, seed=seed)
    trajectories['lape'] = sim4.propagate_additive(encodings['lape'])
    
    print(f"        Done. {len(trajectories)} trajectories generated.")
    
    # ── 1. PCA 几何结构 ──
    print("\n  [1/7] PCA geometry visualization...")
    plot_pca_geometry(encodings, positions)
    print("        Saved 04_pca_geometry")
    logger.log_metric("pca_geometry", "completed")
    
    # ── 2. t-SNE / UMAP ──
    print("\n  [2/7] t-SNE / UMAP visualization...")
    plot_tsne_umap(encodings, positions)
    print("        Saved 04_tsne_umap")
    logger.log_metric("tsne_umap", "completed")
    logger.log_metric("has_umap", HAS_UMAP)
    
    # ── 3. 流形坍缩 ──
    print("\n  [3/7] Manifold collapse through layers...")
    plot_manifold_collapse(trajectories, positions)
    print("        Saved 04_manifold_collapse")
    logger.log_metric("manifold_collapse", "completed")
    
    # ── 4. 内在维度 ──
    print("\n  [4/7] Intrinsic dimension analysis...")
    dim_results = plot_intrinsic_dimension(encodings, trajectories, positions)
    print("        Saved 04_intrinsic_dimension")
    for name, r in dim_results.items():
        logger.log_metric(f"intrinsic_dim_{name}", r['initial_id'])
        print(f"        {name}: ID = {r['initial_id']:.2f}")
    
    # ── 5. 测地距离 ──
    print("\n  [5/7] Geodesic vs Euclidean distance...")
    geo_results = plot_geodesic_analysis(encodings, positions)
    print("        Saved 04_geodesic_analysis")
    for name, r in geo_results.items():
        logger.log_metric(f"geo_euc_ratio_{name}", r['mean_ratio'])
        print(f"        {name}: Geo/Euc = {r['mean_ratio']:.4f}")
    
    # ── 6. 奇异值谱 ──
    print("\n  [6/7] Singular value spectrum...")
    plot_singular_value_spectrum(encodings, trajectories, positions)
    print("        Saved 04_svd_spectrum")
    logger.log_metric("svd_spectrum", "completed")
    
    # ── 7. 综合仪表盘 ──
    print("\n  [7/7] Manifold dashboard...")
    plot_manifold_dashboard(dim_results, geo_results, encodings)
    print("        Saved 04_manifold_dashboard")
    
    # ── 报告 ──
    print("\n  Generating HTML report...")
    generate_manifold_report()
    
    # 保存日志
    logger.save()
    
    # ── 总结 ──
    print("\n" + "=" * 60)
    print("  04_manifold_visualization complete!")
    print("  Static:      output/pe_analysis/04_*")
    print("  Interactive:  html/pe_analysis/04_*")
    print("\n  -- Manifold Geometry Summary --")
    for name in encodings:
        id_val = dim_results[name]['initial_id']
        geo_val = geo_results.get(name, {}).get('mean_ratio', float('nan'))
        er = effective_rank(encodings[name])
        print(f"  {name:15s}: ID={id_val:5.2f}  "
              f"Geo/Euc={geo_val:.3f}  eRank={er:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
