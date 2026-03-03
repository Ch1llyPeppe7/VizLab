
"""
Position Encoding Registry — 统一的位置编码数学实现

提供四种主流 PE 方案的纯数学实现，在统一的复分析/群论框架下进行对比。

Supported Schemes:
    1. Sinusoidal PE  — 加性绝对编码，傅里叶基函数
    2. RoPE           — 乘性相对编码，SO(2) 旋转群
    3. ALiBi          — 加性线性偏置，无可学习参数
    4. LAPE           — 加性绝对编码，幂律频率分布

Mathematical Framework:
    所有 PE 方案都可以从 Bochner 定理的视角统一理解：
    - 核函数 K(Δ) = ⟨PE(p), PE(p+Δ)⟩ 是位置差的函数
    - 不同方案对应不同的谱测度 μ(dω)
    - 几何意义：PE 将整数位置映射到某个流形上的点
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class PEConfig:
    """位置编码配置"""
    dim: int = 64                # embedding 维度（必须为偶数）
    max_len: int = 512           # 最大序列长度
    base: float = 10000.0        # RoPE / Sinusoidal 的频率基数
    power: float = 3.0           # LAPE 的幂指数
    alibi_slopes: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.dim % 2 != 0:
            raise ValueError(f"dim must be even, got {self.dim}")


class PositionEncoding(ABC):
    """位置编码抽象基类"""
    
    def __init__(self, config: PEConfig):
        self.config = config
        self.dim = config.dim
        self.max_len = config.max_len
    
    @abstractmethod
    def encode(self, positions: np.ndarray) -> np.ndarray:
        """
        编码位置序列。
        
        Args:
            positions: 位置数组 [N,] 或标量
        Returns:
            编码矩阵 [N, dim]
        """
        pass
    
    @abstractmethod
    def get_frequencies(self) -> np.ndarray:
        """返回频率序列 [dim//2,]"""
        pass
    
    @abstractmethod
    def kernel(self, delta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        核函数 K(Δ) = ⟨PE(p), PE(p+Δ)⟩ 的解析形式。
        
        Args:
            delta: 位置差（标量或数组）
        Returns:
            核函数值
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def math_description(self) -> str:
        """LaTeX 格式的数学描述"""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """'additive_absolute' | 'multiplicative_relative' | 'additive_bias'"""
        pass

    def encode_complex(self, positions: np.ndarray) -> np.ndarray:
        """
        复数嵌入：z_k(p) = e^{i ω_k p}
        
        Args:
            positions: [N,]
        Returns:
            复数矩阵 [N, dim//2]
        """
        freqs = self.get_frequencies()  # [dim//2,]
        pos = np.atleast_1d(positions).astype(float)
        phases = np.outer(pos, freqs)   # [N, dim//2]
        return np.exp(1j * phases)
    
    def kernel_matrix(self, positions: np.ndarray) -> np.ndarray:
        """
        核矩阵 K_{ij} = K(p_i - p_j)
        
        Args:
            positions: [N,]
        Returns:
            [N, N] 核矩阵
        """
        pos = np.atleast_1d(positions).astype(float)
        delta = pos[:, None] - pos[None, :]  # [N, N]
        return self.kernel(delta)


# ============================================================
#  1. Sinusoidal Position Encoding (Vaswani et al., 2017)
# ============================================================
class SinusoidalPE(PositionEncoding):
    """
    正弦位置编码 — Transformer 原始方案
    
    数学定义：
        PE(p, 2k)   = sin(p / base^{2k/d})
        PE(p, 2k+1) = cos(p / base^{2k/d})
    
    等价地，频率序列为：
        ω_k = 1 / base^{2k/d},  k = 0, 1, ..., d/2 - 1
    
    核函数：
        K(Δ) = (1/m) Σ_k cos(ω_k · Δ)
    
    几何意义：
        将整数位置映射到 d/2 维环面 T^{d/2} 上
    """
    
    def __init__(self, config: PEConfig = None, **kwargs):
        if config is None:
            config = PEConfig(**kwargs)
        super().__init__(config)
        self._frequencies = 1.0 / (self.config.base ** (2 * np.arange(self.dim // 2) / self.dim))
    
    @property
    def name(self) -> str:
        return "Sinusoidal PE"
    
    @property
    def math_description(self) -> str:
        return r"PE(p,2k) = \sin\!\left(\frac{p}{10000^{2k/d}}\right),\quad PE(p,2k\!+\!1) = \cos\!\left(\frac{p}{10000^{2k/d}}\right)"
    
    @property
    def category(self) -> str:
        return "additive_absolute"
    
    def get_frequencies(self) -> np.ndarray:
        return self._frequencies.copy()
    
    def encode(self, positions: np.ndarray) -> np.ndarray:
        pos = np.atleast_1d(positions).astype(float)
        N = len(pos)
        m = self.dim // 2
        
        # phases[n, k] = pos[n] * ω_k
        phases = np.outer(pos, self._frequencies)  # [N, m]
        
        encoding = np.zeros((N, self.dim))
        encoding[:, 0::2] = np.sin(phases)   # even dims
        encoding[:, 1::2] = np.cos(phases)   # odd dims
        return encoding
    
    def kernel(self, delta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """K(Δ) = (1/m) Σ_k cos(ω_k Δ)"""
        delta = np.atleast_1d(delta).astype(float)
        m = self.dim // 2
        phases = np.tensordot(self._frequencies, delta, axes=0)  # [m, ...]
        return np.mean(np.cos(phases), axis=0).squeeze()


# ============================================================
#  2. Rotary Position Embedding (Su et al., 2021)
# ============================================================
class RoPE(PositionEncoding):
    """
    旋转位置编码 — 乘性相对位置编码
    
    数学定义（作用于 query/key 向量对的第 k 个 2D 子空间）：
        R(p, k) = [[cos(p·θ_k), -sin(p·θ_k)],
                    [sin(p·θ_k),  cos(p·θ_k)]]
    
    其中 θ_k = 1 / base^{2k/d}（与 Sinusoidal 相同的频率序列）
    
    核函数（q·k 内积中的位置依赖部分）：
        K(Δ) = (1/m) Σ_k cos(θ_k · Δ)
    
    关键区别：
        - Sinusoidal: PE(p) 被**加到** embedding 上，只在第一层
        - RoPE: R(p) **乘到** q/k 向量上，在**每一层**重新施加
        - 核函数形式相同，但乘性机制保证了位置信息在深层的**结构保持性**
    
    几何意义：
        d/2 个独立的 SO(2) 旋转，形成螺旋流形
    """
    
    def __init__(self, config: PEConfig = None, **kwargs):
        if config is None:
            config = PEConfig(**kwargs)
        super().__init__(config)
        self._frequencies = 1.0 / (self.config.base ** (2 * np.arange(self.dim // 2) / self.dim))
    
    @property
    def name(self) -> str:
        return "RoPE"
    
    @property
    def math_description(self) -> str:
        return r"R_k(p) = \begin{pmatrix}\cos p\theta_k & -\sin p\theta_k \\ \sin p\theta_k & \cos p\theta_k\end{pmatrix},\quad \theta_k = \frac{1}{10000^{2k/d}}"
    
    @property
    def category(self) -> str:
        return "multiplicative_relative"
    
    def get_frequencies(self) -> np.ndarray:
        return self._frequencies.copy()
    
    def encode(self, positions: np.ndarray) -> np.ndarray:
        """
        返回 [cos(p·θ_k), sin(p·θ_k)] 的展开形式。
        注意：实际使用中 RoPE 是作用在 q/k 上的旋转矩阵，
        这里返回的是等价的 "位置嵌入" 以便统一对比。
        """
        pos = np.atleast_1d(positions).astype(float)
        N = len(pos)
        phases = np.outer(pos, self._frequencies)  # [N, m]
        
        encoding = np.zeros((N, self.dim))
        encoding[:, 0::2] = np.cos(phases)
        encoding[:, 1::2] = np.sin(phases)
        return encoding
    
    def rotation_matrix(self, position: float, freq_idx: int) -> np.ndarray:
        """
        返回第 k 个子空间在位置 p 的 2x2 旋转矩阵。
        
        R_k(p) = [[cos(p·θ_k), -sin(p·θ_k)],
                   [sin(p·θ_k),  cos(p·θ_k)]]
        """
        theta = position * self._frequencies[freq_idx]
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])
    
    def apply_rotary(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        将 RoPE 旋转应用到向量 x 上。
        
        Args:
            x: [N, dim] 输入向量（如 query 或 key）
            positions: [N,] 位置序列
        Returns:
            [N, dim] 旋转后的向量
        """
        pos = np.atleast_1d(positions).astype(float)
        phases = np.outer(pos, self._frequencies)  # [N, m]
        cos_p = np.cos(phases)
        sin_p = np.sin(phases)
        
        x_even = x[:, 0::2]
        x_odd  = x[:, 1::2]
        
        out = np.zeros_like(x)
        out[:, 0::2] = x_even * cos_p - x_odd * sin_p
        out[:, 1::2] = x_even * sin_p + x_odd * cos_p
        return out
    
    def kernel(self, delta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """K(Δ) = (1/m) Σ_k cos(θ_k Δ) — 与 Sinusoidal 形式相同"""
        delta = np.atleast_1d(delta).astype(float)
        m = self.dim // 2
        phases = np.tensordot(self._frequencies, delta, axes=0)
        return np.mean(np.cos(phases), axis=0).squeeze()


# ============================================================
#  3. ALiBi (Press et al., 2022)
# ============================================================
class ALiBi(PositionEncoding):
    """
    Attention with Linear Biases — 线性位置偏置
    
    数学定义：
        不修改 embedding，而是在 attention score 上加偏置：
        Attention(q, k) = softmax(q·k^T / √d - m·|i-j|)
    
    其中 m 是头特定的斜率（slope），按几何级数分配：
        m_h = 2^{-8h/H},  h = 1, 2, ..., H
    
    核函数（位置偏置函数）：
        B(Δ) = -m · |Δ|
    
    注意：ALiBi 严格来说没有 "encoding"，只有位置偏置。
    这里为了统一框架，将偏置函数视为核函数。
    
    几何意义：
        锥面上的线性衰减
    """
    
    def __init__(self, config: PEConfig = None, n_heads: int = 8, **kwargs):
        if config is None:
            config = PEConfig(**kwargs)
        super().__init__(config)
        self.n_heads = n_heads
        
        # 几何级数斜率: m_h = 2^{-8h/H}
        if config.alibi_slopes is not None:
            self.slopes = config.alibi_slopes
        else:
            self.slopes = 2.0 ** (-8.0 * np.arange(1, n_heads + 1) / n_heads)
    
    @property
    def name(self) -> str:
        return "ALiBi"
    
    @property
    def math_description(self) -> str:
        return r"\text{score}(i,j) = q_i \cdot k_j / \sqrt{d} - m \cdot |i - j|,\quad m_h = 2^{-8h/H}"
    
    @property
    def category(self) -> str:
        return "additive_bias"
    
    def get_frequencies(self) -> np.ndarray:
        """ALiBi 没有显式频率，返回斜率作为替代"""
        return self.slopes.copy()
    
    def encode(self, positions: np.ndarray) -> np.ndarray:
        """
        ALiBi 不生成位置编码，返回零矩阵。
        真正的位置信息通过 bias_matrix() 注入。
        """
        pos = np.atleast_1d(positions)
        return np.zeros((len(pos), self.dim))
    
    def bias_matrix(self, seq_len: int, head_idx: int = 0) -> np.ndarray:
        """
        生成 ALiBi 偏置矩阵。
        
        B[i,j] = -m_h · |i - j|
        
        Args:
            seq_len: 序列长度
            head_idx: 注意力头索引
        Returns:
            [seq_len, seq_len] 偏置矩阵
        """
        positions = np.arange(seq_len)
        delta = np.abs(positions[:, None] - positions[None, :])
        return -self.slopes[head_idx] * delta
    
    def kernel(self, delta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        B(Δ) = -m · |Δ|（使用第一个头的斜率）
        
        注意：这不是正定核，而是偏置函数。
        """
        delta = np.atleast_1d(delta).astype(float)
        return (-self.slopes[0] * np.abs(delta)).squeeze()


# ============================================================
#  4. LAPE — Location-Aware Position Encoding (TCFMamba)
# ============================================================
class LAPE(PositionEncoding):
    """
    LAPE — 基于幂律频率分布的位置编码
    
    数学定义：
        ω_k = (-k/d)^a,  k = 0, 1, ..., d/2 - 1
        PE(q, 2k)   = sin(q · ω_k / s)
        PE(q, 2k+1) = cos(q · ω_k / s)
    
    其中：
        - a = 3.0 (TCFMamba 的超线性增长)
        - s = 1.0 (无量纲情形)
    
    核函数：
        K(Δ) = (1/m) Σ_k cos(ω_k · Δ/s)
    
    渐近行为：
        |K(Δ)| ~ O(1/(Δ)^{1/a})  as Δ → ∞
    
    几何意义：
        Kähler 流形上的曲线，曲率由幂指数 a 控制
    """
    
    def __init__(self, config: PEConfig = None, scale: float = 1.0, **kwargs):
        if config is None:
            config = PEConfig(**kwargs)
        super().__init__(config)
        self.scale = scale
        self.power = config.power
        
        # 幂律频率: ω_k = (-k/d)^a
        k = np.arange(self.dim // 2)
        base = -k / float(self.dim)
        with np.errstate(divide='ignore', invalid='ignore'):
            freqs = np.sign(base) * np.abs(base) ** self.power
            freqs[0] = 0.0
        self._frequencies = freqs
    
    @property
    def name(self) -> str:
        return f"LAPE (a={self.power})"
    
    @property
    def math_description(self) -> str:
        return r"\omega_k = \left(-\frac{k}{d}\right)^{a},\quad PE(q,2k)=\sin(\omega_k q/s),\quad a=" + str(self.power)
    
    @property
    def category(self) -> str:
        return "additive_absolute"
    
    def get_frequencies(self) -> np.ndarray:
        return self._frequencies.copy()
    
    def encode(self, positions: np.ndarray) -> np.ndarray:
        pos = np.atleast_1d(positions).astype(float)
        N = len(pos)
        m = self.dim // 2
        
        pos_scaled = pos / self.scale
        phases = np.outer(pos_scaled, self._frequencies)  # [N, m]
        
        encoding = np.zeros((N, self.dim))
        encoding[:, 0::2] = np.sin(phases)
        encoding[:, 1::2] = np.cos(phases)
        return encoding
    
    def kernel(self, delta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """K(Δ) = (1/m) Σ_k cos(ω_k · Δ/s)"""
        delta = np.atleast_1d(delta).astype(float)
        m = self.dim // 2
        delta_scaled = delta / self.scale
        phases = np.tensordot(self._frequencies, delta_scaled, axes=0)
        return np.mean(np.cos(phases), axis=0).squeeze()


# ============================================================
#  注册表
# ============================================================

_REGISTRY: Dict[str, type] = {
    'sinusoidal': SinusoidalPE,
    'rope':       RoPE,
    'alibi':      ALiBi,
    'lape':       LAPE,
}


def get_pe(name: str, config: PEConfig = None, **kwargs) -> PositionEncoding:
    """
    工厂函数：按名称获取位置编码实例。
    
    Args:
        name: 'sinusoidal' | 'rope' | 'alibi' | 'lape'
        config: PEConfig 或通过 kwargs 传入
    Returns:
        PositionEncoding 实例
    
    Example:
        >>> pe = get_pe('rope', dim=128)
        >>> embeddings = pe.encode(np.arange(100))
    """
    name_lower = name.lower()
    if name_lower not in _REGISTRY:
        raise ValueError(f"Unknown PE: '{name}'. Available: {list(_REGISTRY.keys())}")
    
    cls = _REGISTRY[name_lower]
    if config is None:
        config = PEConfig(**{k: v for k, v in kwargs.items() 
                            if k in PEConfig.__dataclass_fields__})
    
    # 提取非 config 参数（如 ALiBi 的 n_heads, LAPE 的 scale）
    extra_kwargs = {k: v for k, v in kwargs.items() 
                    if k not in PEConfig.__dataclass_fields__}
    
    return cls(config=config, **extra_kwargs)


def get_all_pe(config: PEConfig = None, **kwargs) -> Dict[str, PositionEncoding]:
    """获取所有 PE 方案的实例字典"""
    if config is None:
        config = PEConfig(**{k: v for k, v in kwargs.items()
                            if k in PEConfig.__dataclass_fields__})
    return {name: get_pe(name, config=config) for name in _REGISTRY}


def list_pe() -> Dict[str, str]:
    """列出所有可用的 PE 方案及其类别"""
    result = {}
    dummy_config = PEConfig()
    for name, cls in _REGISTRY.items():
        instance = cls(config=dummy_config) if name != 'alibi' else cls(config=dummy_config, n_heads=8)
        result[name] = {
            'class': cls.__name__,
            'category': instance.category,
            'description': instance.math_description[:80] + '...'
        }
    return result
