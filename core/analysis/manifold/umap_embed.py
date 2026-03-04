"""
UMAP — Uniform Manifold Approximation and Projection

使用 umap-learn 封装的 UMAP 实现非线性降维
"""

import numpy as np
from typing import Optional, Dict, Any
import warnings


def umap_embed(
    data: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    对高维数据进行 UMAP 降维
    
    Parameters
    ----------
    data : np.ndarray
        输入数据 [N, d]
    n_components : int
        目标维度（通常 2 或 3）
    n_neighbors : int
        邻居数量，控制局部 vs 全局结构
    min_dist : float
        嵌入点之间的最小距离
    random_state : int
        随机种子
        
    Returns
    -------
    np.ndarray
        UMAP 嵌入 [N, n_components]
    """
    try:
        import umap
    except ImportError:
        # 如果 UMAP 不可用，回退到 t-SNE
        warnings.warn("UMAP 不可用，回退到 t-SNE。安装: pip install umap-learn")
        from .tsne import tsne_embed
        return tsne_embed(data, n_components=n_components, random_state=random_state)
    
    data = np.asarray(data, dtype=np.float64)
    N = data.shape[0]
    
    # 自动调整 n_neighbors
    n_neighbors = min(n_neighbors, N - 1)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    return reducer.fit_transform(data)


def umap_embed_with_params(
    data: np.ndarray,
    n_components: int = 2,
    **kwargs
) -> np.ndarray:
    """
    带自定义参数的 UMAP 嵌入
    
    Parameters
    ----------
    data : np.ndarray
        输入数据 [N, d]
    n_components : int
        目标维度
    **kwargs
        传递给 umap.UMAP 的参数:
        - n_neighbors: int (default 15)
        - min_dist: float (default 0.1)
        - metric: str (default 'euclidean')
        - spread: float (default 1.0)
        - random_state: int (default 42)
        
    Returns
    -------
    np.ndarray
        UMAP 嵌入 [N, n_components]
    """
    try:
        import umap
    except ImportError:
        warnings.warn("UMAP 不可用，回退到 t-SNE")
        from .tsne import tsne_embed_with_params
        # 只传递 t-SNE 支持的参数
        tsne_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['perplexity', 'learning_rate', 'n_iter', 'random_state']}
        return tsne_embed_with_params(data, n_components=n_components, **tsne_kwargs)
    
    data = np.asarray(data, dtype=np.float64)
    N = data.shape[0]
    
    # 默认参数
    default_params = {
        'n_neighbors': min(15, N - 1),
        'min_dist': 0.1,
        'metric': 'euclidean',
        'spread': 1.0,
        'random_state': 42
    }
    default_params.update(kwargs)
    
    # 确保 n_neighbors 不超过样本数
    default_params['n_neighbors'] = min(default_params['n_neighbors'], N - 1)
    
    reducer = umap.UMAP(n_components=n_components, **default_params)
    
    return reducer.fit_transform(data)
