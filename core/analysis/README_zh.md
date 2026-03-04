# core.analysis — 通用数学分析库

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

> 领域无关的数学分析工具，可用于位置编码分析、Transformer 内部表征分析或任何高维序列数据分析。

## 📦 模块概览

| 模块 | 功能 | 主要函数 |
|------|------|----------|
| `geometry` | 微分几何 | `curvature`, `torsion`, `metric_tensor`, `arc_length` |
| `spectral` | 频谱分析 | `fft_power_spectrum`, `welch_psd`, `spectral_entropy`, `spectrogram` |
| `information` | 信息论 | `shannon_entropy`, `mutual_information`, `kl_divergence`, `fisher_information_matrix` |
| `manifold` | 流形可视化 | `pca_projection`, `tsne_embed`, `umap_embed`, `trajectory_3d` |

## 🚀 快速开始

```python
from core.analysis import geometry, spectral, information, manifold
import numpy as np

# 创建测试数据: 高维嵌入序列 [N, d]
N, d = 100, 64
embeddings = np.random.randn(N, d)
```

---

## 📐 geometry — 微分几何工具

分析高维空间中参数曲线的几何性质。

### 主要函数

#### `curvature(d1, d2, eps=1e-12) -> np.ndarray`
计算 Frenet-Serret 曲率 κ(p)。

```python
from core.analysis.geometry import curvature, compute_derivatives

# 计算导数
d1, d2, d3 = compute_derivatives(embeddings, h=1.0)

# 计算曲率
kappa = curvature(d1, d2)  # [N,]
print(f"平均曲率: {np.mean(kappa):.4f}")
```

**数学定义:**
$$\kappa(p) = \frac{\sqrt{\|\gamma'\|^2\|\gamma''\|^2 - (\gamma' \cdot \gamma'')^2}}{\|\gamma'\|^3}$$

#### `torsion(d1, d2, d3, eps=1e-12) -> np.ndarray`
计算挠率 τ(p)，衡量曲线离开密切平面的程度。

```python
from core.analysis.geometry import torsion

tau = torsion(d1, d2, d3)  # [N,]
```

#### `metric_tensor(embeddings, h=1.0) -> np.ndarray`
计算度量张量 g(p) = ||γ'(p)||²。

```python
from core.analysis.geometry import metric_tensor

g = metric_tensor(embeddings, h=1.0)  # [N,]
```

#### `arc_length(embeddings, h=1.0) -> np.ndarray`
计算累积弧长 s(p) = ∫₀ᵖ √g(t) dt。

```python
from core.analysis.geometry import arc_length

s = arc_length(embeddings, h=1.0)  # [N,]
print(f"总弧长: {s[-1]:.2f}")
```

---

## 🎵 spectral — 频谱分析工具

分析信号的频率特性。

### 主要函数

#### `fft_power_spectrum(signal, fs=1.0) -> Tuple[ndarray, ndarray]`
计算 FFT 功率谱。

```python
from core.analysis.spectral import fft_power_spectrum

# 创建测试信号: 50Hz 正弦波
fs = 1000
t = np.linspace(0, 1, fs)
signal = np.sin(2 * np.pi * 50 * t)

freqs, psd = fft_power_spectrum(signal, fs=fs)
peak_freq = freqs[np.argmax(psd[1:]) + 1]
print(f"主频: {peak_freq:.1f} Hz")
```

#### `welch_psd(signal, fs=1.0, nperseg=256) -> Tuple[ndarray, ndarray]`
使用 Welch 方法估计 PSD。

```python
from core.analysis.spectral import welch_psd

freqs, psd = welch_psd(signal, fs=fs, nperseg=128)
```

#### `spectral_entropy(psd, base=np.e) -> float`
计算谱熵，衡量频谱的"平坦度"。

```python
from core.analysis.spectral import spectral_entropy

H = spectral_entropy(psd)
# H 低 → 周期信号; H 高 → 宽带/噪声信号
```

#### `spectrogram(signal, fs=1.0, nperseg=256) -> Tuple[ndarray, ndarray, ndarray]`
计算短时傅里叶变换谱图。

```python
from core.analysis.spectral import spectrogram

f, t, Sxx = spectrogram(signal, fs=fs)
# Sxx: [n_freqs, n_times]
```

---

## 📊 information — 信息论工具

度量概率分布和随机变量之间的信息关系。

### 主要函数

#### `shannon_entropy(p, base=np.e) -> float`
计算离散概率分布的香农熵。

```python
from core.analysis.information import shannon_entropy

# 均匀分布 → 最大熵
uniform = np.ones(8) / 8
H_uniform = shannon_entropy(uniform, base=2)  # = log2(8) = 3 bits

# 尖峰分布 → 低熵
peaked = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.004, 0.001])
H_peaked = shannon_entropy(peaked, base=2)    # ≈ 0.68 bits
```

#### `mutual_information(x, y, n_bins=30) -> float`
计算两个随机变量的互信息。

```python
from core.analysis.information import mutual_information

x = np.random.randn(500)
y = x + 0.3 * np.random.randn(500)  # 相关
mi = mutual_information(x, y)
# mi 大 → x 和 y 强相关
```

#### `kl_divergence(p, q, base=np.e) -> float`
计算 KL 散度 D_KL(P || Q)。

```python
from core.analysis.information import kl_divergence

kl = kl_divergence(peaked, uniform, base=2)
# kl 大 → 两个分布差异大
```

#### `fisher_information_matrix(embeddings, h=1.0) -> np.ndarray`
计算 Fisher 信息矩阵（单参数情形）。

```python
from core.analysis.information import fisher_information_matrix

fisher = fisher_information_matrix(embeddings, h=1.0)  # [N,]
# Fisher 信息大 → 参数敏感性高
```

---

## 🌐 manifold — 流形可视化工具

高维数据的降维和可视化。

### 主要函数

#### `pca_projection(data, n_components=3) -> np.ndarray`
PCA 降维投影。

```python
from core.analysis.manifold import pca_projection, pca_explained_variance

# 64维 → 3维
proj_3d = pca_projection(embeddings, n_components=3)  # [N, 3]

# 查看解释方差
explained, cumulative = pca_explained_variance(embeddings, n_components=10)
print(f"前3主成分解释方差: {cumulative[2]:.1%}")
```

#### `tsne_embed(data, n_components=2, perplexity=30) -> np.ndarray`
t-SNE 非线性降维。

```python
from core.analysis.manifold import tsne_embed

proj_2d = tsne_embed(embeddings, n_components=2, perplexity=15)  # [N, 2]
```

#### `umap_embed(data, n_components=2, n_neighbors=15) -> np.ndarray`
UMAP 非线性降维（需安装 `umap-learn`）。

```python
from core.analysis.manifold import umap_embed

proj_2d = umap_embed(embeddings, n_components=2)  # [N, 2]
```

#### `trajectory_3d(data, colors=None, title="3D Trajectory")`
绘制 3D 轨迹可视化。

```python
from core.analysis.manifold import trajectory_3d
import matplotlib.pyplot as plt

trajectory_3d(embeddings, colors=np.arange(N), title="Embedding Trajectory")
plt.show()
```

#### `compute_trajectory_length(data, metric='euclidean') -> float`
计算轨迹总长度。

```python
from core.analysis.manifold import compute_trajectory_length

length = compute_trajectory_length(embeddings)
print(f"轨迹长度: {length:.2f}")
```

---

## 🔧 完整示例

### 分析 Position Encoding 的几何特性

```python
import numpy as np
from core.analysis import geometry, spectral, information, manifold

# 1. 生成正弦位置编码
def sinusoidal_pe(positions, dim):
    pe = np.zeros((len(positions), dim))
    for k in range(dim // 2):
        omega = 1 / (10000 ** (2 * k / dim))
        pe[:, 2*k] = np.sin(positions * omega)
        pe[:, 2*k+1] = np.cos(positions * omega)
    return pe

positions = np.arange(256)
pe = sinusoidal_pe(positions, dim=64)

# 2. 微分几何分析
d1, d2, d3 = geometry.compute_derivatives(pe, h=1.0)
kappa = geometry.curvature(d1, d2)
tau = geometry.torsion(d1, d2, d3)
print(f"平均曲率: {np.mean(kappa):.4f}")
print(f"平均挠率: {np.mean(np.abs(tau)):.4f}")

# 3. 频谱分析 (第一维)
freqs, psd = spectral.fft_power_spectrum(pe[:, 0], fs=1.0)
H = spectral.spectral_entropy(psd)
print(f"谱熵: {H:.4f}")

# 4. 流形可视化
proj_3d = manifold.pca_projection(pe, n_components=3)
manifold.trajectory_3d(proj_3d, colors=positions, title="PE Trajectory (PCA)")
```

### 分析 Transformer 表征

```python
# 假设有 Transformer 的隐藏状态 [seq_len, hidden_dim]
hidden_states = ...  # 从模型获取

# 分析几何特性
kappa = geometry.curvature(*geometry.compute_derivatives(hidden_states)[:2])

# 计算互信息 (相邻位置)
mi_adjacent = information.mutual_information(
    hidden_states[:-1].flatten(), 
    hidden_states[1:].flatten()
)

# 降维可视化
proj = manifold.tsne_embed(hidden_states, n_components=2)
```

---

## 📋 API 参考

### geometry 模块

| 函数 | 输入 | 输出 | 描述 |
|------|------|------|------|
| `compute_derivatives(data, h)` | `[N, d]` | `(d1, d2, d3)` | 计算1/2/3阶导数 |
| `curvature(d1, d2, eps)` | `[N, d]` | `[N,]` | Frenet-Serret 曲率 |
| `torsion(d1, d2, d3, eps)` | `[N, d]` | `[N,]` | 挠率 |
| `metric_tensor(data, h)` | `[N, d]` | `[N,]` | 度量张量 |
| `arc_length(data, h)` | `[N, d]` | `[N,]` | 累积弧长 |

### spectral 模块

| 函数 | 输入 | 输出 | 描述 |
|------|------|------|------|
| `fft_power_spectrum(signal, fs)` | `[N,]` | `(freqs, psd)` | FFT 功率谱 |
| `welch_psd(signal, fs, nperseg)` | `[N,]` | `(freqs, psd)` | Welch PSD |
| `spectral_entropy(psd, base)` | `[N,]` | `float` | 谱熵 |
| `spectrogram(signal, fs, nperseg)` | `[N,]` | `(f, t, Sxx)` | STFT 谱图 |

### information 模块

| 函数 | 输入 | 输出 | 描述 |
|------|------|------|------|
| `shannon_entropy(p, base)` | `[N,]` | `float` | 香农熵 |
| `mutual_information(x, y, n_bins)` | `[N,]` | `float` | 互信息 |
| `kl_divergence(p, q, base)` | `[N,]` | `float` | KL 散度 |
| `fisher_information_matrix(data, h)` | `[N, d]` | `[N,]` | Fisher 信息 |

### manifold 模块

| 函数 | 输入 | 输出 | 描述 |
|------|------|------|------|
| `pca_projection(data, n_components)` | `[N, d]` | `[N, k]` | PCA 降维 |
| `tsne_embed(data, n_components)` | `[N, d]` | `[N, k]` | t-SNE 嵌入 |
| `umap_embed(data, n_components)` | `[N, d]` | `[N, k]` | UMAP 嵌入 |
| `trajectory_3d(data, colors)` | `[N, 3]` | `Figure` | 3D 轨迹图 |

---

## 依赖

- `numpy` (必需)
- `scipy` (spectral 模块需要)
- `matplotlib` (manifold 可视化需要)
- `scikit-learn` (t-SNE、互信息 KNN 需要)
- `umap-learn` (可选，UMAP 需要)
