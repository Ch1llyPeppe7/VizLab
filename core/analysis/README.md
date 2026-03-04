# core.analysis — General-Purpose Mathematical Analysis Library

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

> Domain-agnostic mathematical analysis tools applicable to PE analysis, Transformer internals, or any high-dimensional sequential data.

## 📦 Module Overview

| Module | Functionality | Key Functions |
|--------|--------------|---------------|
| `geometry` | Differential Geometry | `curvature`, `torsion`, `metric_tensor`, `arc_length` |
| `spectral` | Spectral Analysis | `fft_power_spectrum`, `welch_psd`, `spectral_entropy`, `spectrogram` |
| `information` | Information Theory | `shannon_entropy`, `mutual_information`, `kl_divergence`, `fisher_information_matrix` |
| `manifold` | Manifold Visualization | `pca_projection`, `tsne_embed`, `umap_embed`, `trajectory_3d` |

## 🚀 Quick Start

```python
from core.analysis import geometry, spectral, information, manifold
import numpy as np

# Create test data: high-dimensional embedding sequence [N, d]
N, d = 100, 64
embeddings = np.random.randn(N, d)
```

---

## 📐 geometry — Differential Geometry Tools

Analyze geometric properties of parametric curves in high-dimensional spaces.

### Key Functions

#### `curvature(d1, d2, eps=1e-12) -> np.ndarray`
Compute Frenet-Serret curvature κ(p).

```python
from core.analysis.geometry import curvature, compute_derivatives

# Compute derivatives
d1, d2, d3 = compute_derivatives(embeddings, h=1.0)

# Compute curvature
kappa = curvature(d1, d2)  # [N,]
print(f"Mean curvature: {np.mean(kappa):.4f}")
```

**Mathematical Definition:**
$$\kappa(p) = \frac{\sqrt{\|\gamma'\|^2\|\gamma''\|^2 - (\gamma' \cdot \gamma'')^2}}{\|\gamma'\|^3}$$

#### `torsion(d1, d2, d3, eps=1e-12) -> np.ndarray`
Compute torsion τ(p), measuring how the curve deviates from its osculating plane.

```python
from core.analysis.geometry import torsion

tau = torsion(d1, d2, d3)  # [N,]
```

#### `metric_tensor(embeddings, h=1.0) -> np.ndarray`
Compute metric tensor g(p) = ||γ'(p)||².

```python
from core.analysis.geometry import metric_tensor

g = metric_tensor(embeddings, h=1.0)  # [N,]
```

#### `arc_length(embeddings, h=1.0) -> np.ndarray`
Compute cumulative arc length s(p) = ∫₀ᵖ √g(t) dt.

```python
from core.analysis.geometry import arc_length

s = arc_length(embeddings, h=1.0)  # [N,]
print(f"Total arc length: {s[-1]:.2f}")
```

---

## 🎵 spectral — Spectral Analysis Tools

Analyze frequency characteristics of signals.

### Key Functions

#### `fft_power_spectrum(signal, fs=1.0) -> Tuple[ndarray, ndarray]`
Compute FFT power spectrum.

```python
from core.analysis.spectral import fft_power_spectrum

# Create test signal: 50Hz sine wave
fs = 1000
t = np.linspace(0, 1, fs)
signal = np.sin(2 * np.pi * 50 * t)

freqs, psd = fft_power_spectrum(signal, fs=fs)
peak_freq = freqs[np.argmax(psd[1:]) + 1]
print(f"Peak frequency: {peak_freq:.1f} Hz")
```

#### `welch_psd(signal, fs=1.0, nperseg=256) -> Tuple[ndarray, ndarray]`
Estimate PSD using Welch's method.

```python
from core.analysis.spectral import welch_psd

freqs, psd = welch_psd(signal, fs=fs, nperseg=128)
```

#### `spectral_entropy(psd, base=np.e) -> float`
Compute spectral entropy, measuring spectrum "flatness".

```python
from core.analysis.spectral import spectral_entropy

H = spectral_entropy(psd)
# Low H → periodic signal; High H → broadband/noise
```

#### `spectrogram(signal, fs=1.0, nperseg=256) -> Tuple[ndarray, ndarray, ndarray]`
Compute short-time Fourier transform spectrogram.

```python
from core.analysis.spectral import spectrogram

f, t, Sxx = spectrogram(signal, fs=fs)
# Sxx: [n_freqs, n_times]
```

---

## 📊 information — Information Theory Tools

Measure information relationships between probability distributions and random variables.

### Key Functions

#### `shannon_entropy(p, base=np.e) -> float`
Compute Shannon entropy of a discrete probability distribution.

```python
from core.analysis.information import shannon_entropy

# Uniform distribution → maximum entropy
uniform = np.ones(8) / 8
H_uniform = shannon_entropy(uniform, base=2)  # = log2(8) = 3 bits

# Peaked distribution → low entropy
peaked = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.004, 0.001])
H_peaked = shannon_entropy(peaked, base=2)    # ≈ 0.68 bits
```

#### `mutual_information(x, y, n_bins=30) -> float`
Compute mutual information between two random variables.

```python
from core.analysis.information import mutual_information

x = np.random.randn(500)
y = x + 0.3 * np.random.randn(500)  # correlated
mi = mutual_information(x, y)
# High MI → x and y are strongly correlated
```

#### `kl_divergence(p, q, base=np.e) -> float`
Compute KL divergence D_KL(P || Q).

```python
from core.analysis.information import kl_divergence

kl = kl_divergence(peaked, uniform, base=2)
# High KL → large difference between distributions
```

#### `fisher_information_matrix(embeddings, h=1.0) -> np.ndarray`
Compute Fisher information matrix (single parameter case).

```python
from core.analysis.information import fisher_information_matrix

fisher = fisher_information_matrix(embeddings, h=1.0)  # [N,]
# High Fisher information → high parameter sensitivity
```

---

## 🌐 manifold — Manifold Visualization Tools

Dimensionality reduction and visualization of high-dimensional data.

### Key Functions

#### `pca_projection(data, n_components=3) -> np.ndarray`
PCA dimensionality reduction.

```python
from core.analysis.manifold import pca_projection, pca_explained_variance

# 64D → 3D
proj_3d = pca_projection(embeddings, n_components=3)  # [N, 3]

# Check explained variance
explained, cumulative = pca_explained_variance(embeddings, n_components=10)
print(f"First 3 PCs explain: {cumulative[2]:.1%}")
```

#### `tsne_embed(data, n_components=2, perplexity=30) -> np.ndarray`
t-SNE nonlinear dimensionality reduction.

```python
from core.analysis.manifold import tsne_embed

proj_2d = tsne_embed(embeddings, n_components=2, perplexity=15)  # [N, 2]
```

#### `umap_embed(data, n_components=2, n_neighbors=15) -> np.ndarray`
UMAP nonlinear dimensionality reduction (requires `umap-learn`).

```python
from core.analysis.manifold import umap_embed

proj_2d = umap_embed(embeddings, n_components=2)  # [N, 2]
```

#### `trajectory_3d(data, colors=None, title="3D Trajectory")`
Plot 3D trajectory visualization.

```python
from core.analysis.manifold import trajectory_3d
import matplotlib.pyplot as plt

trajectory_3d(embeddings, colors=np.arange(N), title="Embedding Trajectory")
plt.show()
```

#### `compute_trajectory_length(data, metric='euclidean') -> float`
Compute total trajectory length.

```python
from core.analysis.manifold import compute_trajectory_length

length = compute_trajectory_length(embeddings)
print(f"Trajectory length: {length:.2f}")
```

---

## 🔧 Complete Example

### Analyzing Position Encoding Geometric Properties

```python
import numpy as np
from core.analysis import geometry, spectral, information, manifold

# 1. Generate sinusoidal position encoding
def sinusoidal_pe(positions, dim):
    pe = np.zeros((len(positions), dim))
    for k in range(dim // 2):
        omega = 1 / (10000 ** (2 * k / dim))
        pe[:, 2*k] = np.sin(positions * omega)
        pe[:, 2*k+1] = np.cos(positions * omega)
    return pe

positions = np.arange(256)
pe = sinusoidal_pe(positions, dim=64)

# 2. Differential geometry analysis
d1, d2, d3 = geometry.compute_derivatives(pe, h=1.0)
kappa = geometry.curvature(d1, d2)
tau = geometry.torsion(d1, d2, d3)
print(f"Mean curvature: {np.mean(kappa):.4f}")
print(f"Mean torsion: {np.mean(np.abs(tau)):.4f}")

# 3. Spectral analysis (first dimension)
freqs, psd = spectral.fft_power_spectrum(pe[:, 0], fs=1.0)
H = spectral.spectral_entropy(psd)
print(f"Spectral entropy: {H:.4f}")

# 4. Manifold visualization
proj_3d = manifold.pca_projection(pe, n_components=3)
manifold.trajectory_3d(proj_3d, colors=positions, title="PE Trajectory (PCA)")
```

### Analyzing Transformer Representations

```python
# Assume Transformer hidden states [seq_len, hidden_dim]
hidden_states = ...  # obtained from model

# Analyze geometric properties
kappa = geometry.curvature(*geometry.compute_derivatives(hidden_states)[:2])

# Compute mutual information (adjacent positions)
mi_adjacent = information.mutual_information(
    hidden_states[:-1].flatten(), 
    hidden_states[1:].flatten()
)

# Dimensionality reduction visualization
proj = manifold.tsne_embed(hidden_states, n_components=2)
```

---

## 📋 API Reference

### geometry Module

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `compute_derivatives(data, h)` | `[N, d]` | `(d1, d2, d3)` | Compute 1st/2nd/3rd derivatives |
| `curvature(d1, d2, eps)` | `[N, d]` | `[N,]` | Frenet-Serret curvature |
| `torsion(d1, d2, d3, eps)` | `[N, d]` | `[N,]` | Torsion |
| `metric_tensor(data, h)` | `[N, d]` | `[N,]` | Metric tensor |
| `arc_length(data, h)` | `[N, d]` | `[N,]` | Cumulative arc length |

### spectral Module

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `fft_power_spectrum(signal, fs)` | `[N,]` | `(freqs, psd)` | FFT power spectrum |
| `welch_psd(signal, fs, nperseg)` | `[N,]` | `(freqs, psd)` | Welch PSD |
| `spectral_entropy(psd, base)` | `[N,]` | `float` | Spectral entropy |
| `spectrogram(signal, fs, nperseg)` | `[N,]` | `(f, t, Sxx)` | STFT spectrogram |

### information Module

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `shannon_entropy(p, base)` | `[N,]` | `float` | Shannon entropy |
| `mutual_information(x, y, n_bins)` | `[N,]` | `float` | Mutual information |
| `kl_divergence(p, q, base)` | `[N,]` | `float` | KL divergence |
| `fisher_information_matrix(data, h)` | `[N, d]` | `[N,]` | Fisher information |

### manifold Module

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `pca_projection(data, n_components)` | `[N, d]` | `[N, k]` | PCA projection |
| `tsne_embed(data, n_components)` | `[N, d]` | `[N, k]` | t-SNE embedding |
| `umap_embed(data, n_components)` | `[N, d]` | `[N, k]` | UMAP embedding |
| `trajectory_3d(data, colors)` | `[N, 3]` | `Figure` | 3D trajectory plot |

---

## Dependencies

- `numpy` (required)
- `scipy` (required for spectral module)
- `matplotlib` (required for manifold visualization)
- `scikit-learn` (required for t-SNE, MI KNN estimation)
- `umap-learn` (optional, required for UMAP)
