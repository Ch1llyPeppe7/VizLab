# Position Encoding Registry

[![English](https://img.shields.io/badge/lang-English-blue.svg)](PE_README.md)
[![中文](https://img.shields.io/badge/lang-中文-red.svg)](PE_README_zh.md)

**`core/pe_registry.py`** — 位置编码的统一数学实现框架

## 概述

本模块在 **复分析/群论** 的统一框架下实现四种主流位置编码方案，支持：

- 🔢 位置编码生成
- 📊 核函数计算（Bochner 定理视角）
- 🔄 复数嵌入与旋转操作
- 📐 几何/代数特性分析

## 支持的 PE 方案

| 名称 | 类型 | 数学本质 | 几何意义 |
|------|------|----------|----------|
| **Sinusoidal** | 加性绝对 | 傅里叶基函数 | $d/2$ 维环面 $T^{d/2}$ |
| **RoPE** | 乘性相对 | SO(2) 旋转群 | 螺旋流形 |
| **ALiBi** | 加性偏置 | 线性衰减函数 | 锥面 |
| **LAPE** | 加性绝对 | 幂律频率分布 | Kähler 流形 |

## 快速开始

### 1. 获取 PE 实例

```python
from core.pe_registry import get_pe, get_all_pe, PEConfig
import numpy as np

# 方式 1：使用工厂函数 + kwargs
pe = get_pe('rope', dim=128, max_len=512)

# 方式 2：使用配置对象
config = PEConfig(dim=64, max_len=256, base=10000.0)
pe = get_pe('sinusoidal', config=config)

# 方式 3：获取所有方案
all_pes = get_all_pe(dim=128)  # {'sinusoidal': ..., 'rope': ..., 'alibi': ..., 'lape': ...}
```

### 2. 生成位置编码

```python
positions = np.arange(100)  # [0, 1, ..., 99]

# 实数编码 [N, dim]
embeddings = pe.encode(positions)

# 复数编码 [N, dim//2] — 用于频谱分析
complex_emb = pe.encode_complex(positions)
```

### 3. 核函数分析

核函数 $K(\Delta) = \langle PE(p), PE(p+\Delta) \rangle$ 揭示位置编码的内积结构：

```python
# 单点核函数值
k_value = pe.kernel(delta=10)

# 核矩阵 [N, N]
K_matrix = pe.kernel_matrix(positions)
```

## API 详解

### `PositionEncoding` 抽象基类

| 方法/属性 | 返回类型 | 说明 |
|-----------|----------|------|
| `encode(positions)` | `[N, dim]` | 生成位置编码矩阵 |
| `get_frequencies()` | `[dim//2]` | 返回频率序列 $\omega_k$ |
| `kernel(delta)` | `float` 或 `[...]` | 核函数 $K(\Delta)$ |
| `kernel_matrix(positions)` | `[N, N]` | 核矩阵 $K_{ij} = K(p_i - p_j)$ |
| `encode_complex(positions)` | `[N, dim//2]` 复数 | 复数嵌入 $e^{i\omega_k p}$ |
| `name` | `str` | 方案名称 |
| `category` | `str` | 类型：`additive_absolute` \| `multiplicative_relative` \| `additive_bias` |
| `math_description` | `str` | LaTeX 格式数学描述 |

### 特定方法

**RoPE:**
```python
rope = get_pe('rope', dim=64)

# 获取第 k 个子空间的 2x2 旋转矩阵
R = rope.rotation_matrix(position=5, freq_idx=0)

# 将旋转应用到 query/key 向量
x = np.random.randn(100, 64)
x_rotated = rope.apply_rotary(x, positions)
```

**ALiBi:**
```python
alibi = get_pe('alibi', dim=64, n_heads=8)

# 获取注意力偏置矩阵 [seq_len, seq_len]
bias = alibi.bias_matrix(seq_len=100, head_idx=0)
```

## 数学框架

### Bochner 定理视角

所有位置编码都可以从 **正定核函数** 的角度理解：

$$K(\Delta) = \int e^{i\omega\Delta} \mu(d\omega)$$

其中谱测度 $\mu(d\omega)$ 决定了 PE 的性质：

| PE | 谱测度 |
|----|--------|
| Sinusoidal / RoPE | $\mu = \sum_k \delta(\omega - \omega_k)$，几何级数频率 |
| LAPE | $\mu = \sum_k \delta(\omega - \omega_k)$，幂律频率 |
| ALiBi | 非正定（偏置函数而非核函数） |

### 核函数公式

```
Sinusoidal / RoPE:  K(Δ) = (1/m) Σ_k cos(ω_k · Δ)
ALiBi:              B(Δ) = -m · |Δ|  (偏置，非核)
LAPE:               K(Δ) = (1/m) Σ_k cos(ω_k · Δ/s)
```

## 与 `core/analysis` 配合使用

```python
from core.pe_registry import get_all_pe, PEConfig
from core.analysis import geometry, spectral

# 配置
config = PEConfig(dim=128, max_len=256)
positions = np.arange(256)

# 对比多种 PE 的几何特性
for name, pe in get_all_pe(config=config).items():
    emb = pe.encode(positions)
    
    # 微分几何分析
    d1, d2, d3 = geometry.compute_derivatives(emb)
    kappa = geometry.curvature(d1, d2)
    
    # 谱分析
    freqs, psd = spectral.welch_psd(emb)
    
    print(f"{name}: mean_curvature={kappa.mean():.4f}")
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dim` | 64 | embedding 维度（必须为偶数） |
| `max_len` | 512 | 最大序列长度 |
| `base` | 10000.0 | Sinusoidal/RoPE 的频率基数 |
| `power` | 3.0 | LAPE 的幂指数 $a$ |
| `alibi_slopes` | None | ALiBi 自定义斜率（默认几何级数） |
| `n_heads` | 8 | ALiBi 注意力头数 |
| `scale` | 1.0 | LAPE 缩放因子 |

## 扩展新的 PE 方案

```python
from core.pe_registry import PositionEncoding, PEConfig

class MyCustomPE(PositionEncoding):
    """自定义位置编码"""
    
    @property
    def name(self) -> str:
        return "MyCustomPE"
    
    @property
    def math_description(self) -> str:
        return r"PE(p) = ..."
    
    @property
    def category(self) -> str:
        return "additive_absolute"
    
    def get_frequencies(self) -> np.ndarray:
        # 返回频率序列
        return np.arange(self.dim // 2) * 0.1
    
    def encode(self, positions: np.ndarray) -> np.ndarray:
        # 实现编码逻辑
        ...
    
    def kernel(self, delta) -> np.ndarray:
        # 实现核函数
        ...

# 手动注册（可选）
from core.pe_registry import _REGISTRY
_REGISTRY['mycustom'] = MyCustomPE
```

## 文件结构

```
core/
├── pe_registry.py          # PE 核心实现
├── PE_README.md            # 本文档
└── analysis/               # 分析工具库
    ├── geometry/           # 微分几何
    ├── spectral/           # 谱分析
    ├── information/        # 信息论
    └── manifold/           # 流形学习
```
