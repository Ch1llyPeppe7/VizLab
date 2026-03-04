"""
core.analysis.manifold — 流形可视化与降维工具

提供领域无关的降维和流形分析:
    - pca: 主成分分析
    - tsne: t-SNE 嵌入
    - umap: UMAP 嵌入
    - trajectory: 轨迹可视化工具
"""

from .pca import (
    pca_projection,
    pca_explained_variance,
    pca_loadings,
)
from .tsne import (
    tsne_embed,
    tsne_embed_with_params,
)
from .umap_embed import (
    umap_embed,
    umap_embed_with_params,
)
from .trajectory import (
    trajectory_3d,
    trajectory_2d,
    compute_trajectory_length,
)

__all__ = [
    # PCA
    'pca_projection',
    'pca_explained_variance',
    'pca_loadings',
    # t-SNE
    'tsne_embed',
    'tsne_embed_with_params',
    # UMAP
    'umap_embed',
    'umap_embed_with_params',
    # Trajectory
    'trajectory_3d',
    'trajectory_2d',
    'compute_trajectory_length',
]
