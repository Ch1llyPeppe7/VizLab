"""
t-SNE — t-分布随机邻域嵌入

使用 sklearn 封装的 t-SNE 实现非线性降维
"""

import numpy as np
from typing import Optional, Dict, Any


def tsne_embed(
    data: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42
) -> np.ndarray:
    """
    对高维数据进行 t-SNE 降维
    
    Parameters
    ----------
    data : np.ndarray
        输入数据 [N, d]
    n_components : int
        目标维度（通常 2 或 3）
    perplexity : float
        困惑度参数，通常 5-50
    random_state : int
        随机种子
        
    Returns
    -------
    np.ndarray
        t-SNE 嵌入 [N, n_components]
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("需要安装 scikit-learn: pip install scikit-learn")
    
    data = np.asarray(data, dtype=np.float64)
    N = data.shape[0]
    
    # 自动调整 perplexity
    perplexity = min(perplexity, N - 1)
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    
    return tsne.fit_transform(data)


def tsne_embed_with_params(
    data: np.ndarray,
    n_components: int = 2,
    **kwargs
) -> np.ndarray:
    """
    带自定义参数的 t-SNE 嵌入
    
    Parameters
    ----------
    data : np.ndarray
        输入数据 [N, d]
    n_components : int
        目标维度
    **kwargs
        传递给 sklearn.manifold.TSNE 的参数:
        - perplexity: float (default 30)
        - learning_rate: float or 'auto' (default 'auto')
        - n_iter: int (default 1000)
        - metric: str (default 'euclidean')
        - init: str (default 'pca')
        - random_state: int (default 42)
        
    Returns
    -------
    np.ndarray
        t-SNE 嵌入 [N, n_components]
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("需要安装 scikit-learn: pip install scikit-learn")
    
    data = np.asarray(data, dtype=np.float64)
    N = data.shape[0]
    
    # 默认参数
    default_params = {
        'perplexity': min(30.0, N - 1),
        'learning_rate': 'auto',
        'n_iter': 1000,
        'metric': 'euclidean',
        'init': 'pca',
        'random_state': 42
    }
    default_params.update(kwargs)
    
    # 确保 perplexity 不超过样本数
    default_params['perplexity'] = min(default_params['perplexity'], N - 1)
    
    tsne = TSNE(n_components=n_components, **default_params)
    
    return tsne.fit_transform(data)
