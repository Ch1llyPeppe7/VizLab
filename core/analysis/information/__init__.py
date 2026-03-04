"""
core.analysis.information — 信息论分析工具

提供领域无关的信息论度量:
    - entropy: 香农熵、微分熵
    - mutual_info: 互信息
    - fisher_info: Fisher 信息矩阵
    - divergence: KL 散度、JS 散度
"""

from .entropy import (
    shannon_entropy,
    differential_entropy,
    joint_entropy,
)
from .mutual_info import (
    mutual_information,
    conditional_mutual_information,
)
from .fisher import (
    fisher_information_matrix,
    fisher_information_scalar,
)
from .divergence import (
    kl_divergence,
    js_divergence,
)

__all__ = [
    # Entropy
    'shannon_entropy',
    'differential_entropy',
    'joint_entropy',
    # Mutual Information
    'mutual_information',
    'conditional_mutual_information',
    # Fisher Information
    'fisher_information_matrix',
    'fisher_information_scalar',
    # Divergence
    'kl_divergence',
    'js_divergence',
]
