"""
Trajectory visualization — 轨迹可视化工具

用于绘制嵌入空间中的 2D/3D 轨迹
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt


def trajectory_3d(
    data: np.ndarray,
    colors: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "3D Trajectory",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    show_colorbar: bool = True,
    marker_size: float = 20,
    line_alpha: float = 0.3,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False
) -> Optional[plt.Figure]:
    """
    绘制 3D 轨迹可视化
    
    Parameters
    ----------
    data : np.ndarray
        3D 坐标 [N, 3] 或高维数据（自动 PCA 降维）
    colors : np.ndarray, optional
        每个点的颜色值 [N,]
    labels : List[str], optional
        每个点的标签（用于 hover）
    title : str
        图表标题
    figsize : Tuple[int, int]
        图表大小
    cmap : str
        颜色映射
    show_colorbar : bool
        是否显示颜色条
    marker_size : float
        标记大小
    line_alpha : float
        连线透明度
    ax : plt.Axes, optional
        现有的 3D 坐标轴
    return_fig : bool
        是否返回图表对象
        
    Returns
    -------
    Optional[plt.Figure]
        如果 return_fig=True，返回图表对象
    """
    data = np.asarray(data, dtype=np.float64)
    
    # 如果维度 > 3，自动 PCA 降维
    if data.shape[1] > 3:
        from .pca import pca_projection
        data = pca_projection(data, n_components=3)
    elif data.shape[1] < 3:
        # 补零到 3D
        padding = np.zeros((data.shape[0], 3 - data.shape[1]))
        data = np.hstack([data, padding])
    
    N = data.shape[0]
    
    # 默认颜色为位置索引
    if colors is None:
        colors = np.arange(N)
    
    # 创建或使用现有图表
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # 绘制连线
    ax.plot(data[:, 0], data[:, 1], data[:, 2], 
            alpha=line_alpha, color='gray', linewidth=0.5)
    
    # 绘制散点
    scatter = ax.scatter(
        data[:, 0], data[:, 1], data[:, 2],
        c=colors, cmap=cmap, s=marker_size,
        edgecolors='white', linewidth=0.5
    )
    
    if show_colorbar:
        plt.colorbar(scatter, ax=ax, label='Position')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)
    
    if return_fig:
        return fig
    else:
        plt.tight_layout()
        return None


def trajectory_2d(
    data: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "2D Trajectory",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    show_colorbar: bool = True,
    marker_size: float = 30,
    line_alpha: float = 0.3,
    show_arrows: bool = True,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False
) -> Optional[plt.Figure]:
    """
    绘制 2D 轨迹可视化
    
    Parameters
    ----------
    data : np.ndarray
        2D 坐标 [N, 2] 或高维数据（自动 PCA 降维）
    colors : np.ndarray, optional
        每个点的颜色值 [N,]
    title : str
        图表标题
    figsize : Tuple[int, int]
        图表大小
    cmap : str
        颜色映射
    show_colorbar : bool
        是否显示颜色条
    marker_size : float
        标记大小
    line_alpha : float
        连线透明度
    show_arrows : bool
        是否显示方向箭头
    ax : plt.Axes, optional
        现有的坐标轴
    return_fig : bool
        是否返回图表对象
        
    Returns
    -------
    Optional[plt.Figure]
        如果 return_fig=True，返回图表对象
    """
    data = np.asarray(data, dtype=np.float64)
    
    # 如果维度 > 2，自动 PCA 降维
    if data.shape[1] > 2:
        from .pca import pca_projection
        data = pca_projection(data, n_components=2)
    elif data.shape[1] < 2:
        padding = np.zeros((data.shape[0], 2 - data.shape[1]))
        data = np.hstack([data, padding])
    
    N = data.shape[0]
    
    if colors is None:
        colors = np.arange(N)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # 绘制连线
    ax.plot(data[:, 0], data[:, 1], 
            alpha=line_alpha, color='gray', linewidth=0.5)
    
    # 绘制散点
    scatter = ax.scatter(
        data[:, 0], data[:, 1],
        c=colors, cmap=cmap, s=marker_size,
        edgecolors='white', linewidth=0.5
    )
    
    # 绘制方向箭头
    if show_arrows and N > 10:
        step = max(1, N // 10)
        for i in range(0, N - 1, step):
            dx = data[i + 1, 0] - data[i, 0]
            dy = data[i + 1, 1] - data[i, 1]
            ax.annotate('', xy=(data[i + 1, 0], data[i + 1, 1]),
                       xytext=(data[i, 0], data[i, 1]),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
    
    if show_colorbar:
        plt.colorbar(scatter, ax=ax, label='Position')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='datalim')
    
    if return_fig:
        return fig
    else:
        plt.tight_layout()
        return None


def compute_trajectory_length(
    data: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    计算轨迹的总长度
    
    Parameters
    ----------
    data : np.ndarray
        轨迹坐标 [N, d]
    metric : str
        距离度量: 'euclidean', 'manhattan', 'cosine'
        
    Returns
    -------
    float
        轨迹总长度
    """
    data = np.asarray(data, dtype=np.float64)
    N = data.shape[0]
    
    if N < 2:
        return 0.0
    
    if metric == 'euclidean':
        diffs = np.diff(data, axis=0)
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    elif metric == 'manhattan':
        diffs = np.diff(data, axis=0)
        distances = np.sum(np.abs(diffs), axis=1)
    elif metric == 'cosine':
        distances = []
        for i in range(N - 1):
            norm1 = np.linalg.norm(data[i])
            norm2 = np.linalg.norm(data[i + 1])
            if norm1 > 0 and norm2 > 0:
                cos_sim = np.dot(data[i], data[i + 1]) / (norm1 * norm2)
                distances.append(1 - cos_sim)
            else:
                distances.append(0)
        distances = np.array(distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return float(np.sum(distances))
