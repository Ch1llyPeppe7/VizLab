# Position Encoding Registry

[![English](https://img.shields.io/badge/lang-English-blue.svg)](PE_README.md)
[![中文](https://img.shields.io/badge/lang-中文-red.svg)](PE_README_zh.md)

**`core/pe_registry.py`** — A unified mathematical implementation framework for position encodings.

## Overview

This module implements four mainstream position encoding schemes within a unified **complex analysis / group theory** framework, providing:

- 🔢 Position encoding generation
- 📊 Kernel function computation (Bochner's theorem perspective)
- 🔄 Complex embeddings and rotation operations
- 📐 Geometric and algebraic property analysis

## Supported PE Schemes

| Name | Type | Mathematical Essence | Geometric Interpretation |
|------|------|---------------------|-------------------------|
| **Sinusoidal** | Additive Absolute | Fourier basis functions | d/2-dimensional torus T^{d/2} |
| **RoPE** | Multiplicative Relative | SO(2) rotation group | Helical manifold |
| **ALiBi** | Additive Bias | Linear decay function | Cone |
| **LAPE** | Additive Absolute | Power-law frequency distribution | Kähler manifold |

## Quick Start

### 1. Obtaining PE Instances

```python
from core.pe_registry import get_pe, get_all_pe, PEConfig
import numpy as np

# Method 1: Factory function with kwargs
pe = get_pe('rope', dim=128, max_len=512)

# Method 2: Using configuration object
config = PEConfig(dim=64, max_len=256, base=10000.0)
pe = get_pe('sinusoidal', config=config)

# Method 3: Get all schemes
all_pes = get_all_pe(dim=128)  # {'sinusoidal': ..., 'rope': ..., 'alibi': ..., 'lape': ...}
```

### 2. Generating Position Encodings

```python
positions = np.arange(100)  # [0, 1, ..., 99]

# Real encoding [N, dim]
embeddings = pe.encode(positions)

# Complex encoding [N, dim//2] — for spectral analysis
complex_emb = pe.encode_complex(positions)
```

### 3. Kernel Function Analysis

The kernel function K(Δ) = ⟨PE(p), PE(p+Δ)⟩ reveals the inner product structure of position encodings:

```python
# Single-point kernel value
k_value = pe.kernel(delta=10)

# Kernel matrix [N, N]
K_matrix = pe.kernel_matrix(positions)
```

## API Reference

### `PositionEncoding` Abstract Base Class

| Method/Property | Return Type | Description |
|-----------------|-------------|-------------|
| `encode(positions)` | `[N, dim]` | Generate position encoding matrix |
| `get_frequencies()` | `[dim//2]` | Return frequency sequence ωₖ |
| `kernel(delta)` | `float` or `[...]` | Kernel function K(Δ) |
| `kernel_matrix(positions)` | `[N, N]` | Kernel matrix K_{ij} = K(pᵢ - pⱼ) |
| `encode_complex(positions)` | `[N, dim//2]` complex | Complex embedding e^{iωₖp} |
| `name` | `str` | Scheme name |
| `category` | `str` | Type: `additive_absolute` \| `multiplicative_relative` \| `additive_bias` |
| `math_description` | `str` | LaTeX mathematical description |

### Scheme-Specific Methods

**RoPE:**
```python
rope = get_pe('rope', dim=64)

# Get 2x2 rotation matrix for k-th subspace
R = rope.rotation_matrix(position=5, freq_idx=0)

# Apply rotation to query/key vectors
x = np.random.randn(100, 64)
x_rotated = rope.apply_rotary(x, positions)
```

**ALiBi:**
```python
alibi = get_pe('alibi', dim=64, n_heads=8)

# Get attention bias matrix [seq_len, seq_len]
bias = alibi.bias_matrix(seq_len=100, head_idx=0)
```

## Mathematical Framework

### Bochner's Theorem Perspective

All position encodings can be understood through **positive definite kernel functions**:

$$K(\Delta) = \int e^{i\omega\Delta} \mu(d\omega)$$

where the spectral measure μ(dω) determines PE properties:

| PE | Spectral Measure |
|----|------------------|
| Sinusoidal / RoPE | μ = Σₖ δ(ω - ωₖ), geometric frequency series |
| LAPE | μ = Σₖ δ(ω - ωₖ), power-law frequency series |
| ALiBi | Non-positive-definite (bias function, not kernel) |

### Kernel Function Formulas

```
Sinusoidal / RoPE:  K(Δ) = (1/m) Σₖ cos(ωₖ · Δ)
ALiBi:              B(Δ) = -m · |Δ|  (bias, not kernel)
LAPE:               K(Δ) = (1/m) Σₖ cos(ωₖ · Δ/s)
```

## Integration with `core/analysis`

```python
from core.pe_registry import get_all_pe, PEConfig
from core.analysis import geometry, spectral

# Configuration
config = PEConfig(dim=128, max_len=256)
positions = np.arange(256)

# Compare geometric properties across PE schemes
for name, pe in get_all_pe(config=config).items():
    emb = pe.encode(positions)
    
    # Differential geometry analysis
    d1, d2, d3 = geometry.compute_derivatives(emb)
    kappa = geometry.curvature(d1, d2)
    
    # Spectral analysis
    freqs, psd = spectral.welch_psd(emb)
    
    print(f"{name}: mean_curvature={kappa.mean():.4f}")
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 64 | Embedding dimension (must be even) |
| `max_len` | 512 | Maximum sequence length |
| `base` | 10000.0 | Frequency base for Sinusoidal/RoPE |
| `power` | 3.0 | Power exponent a for LAPE |
| `alibi_slopes` | None | Custom ALiBi slopes (default: geometric series) |
| `n_heads` | 8 | Number of attention heads for ALiBi |
| `scale` | 1.0 | Scale factor for LAPE |

## Extending with Custom PE Schemes

```python
from core.pe_registry import PositionEncoding, PEConfig

class MyCustomPE(PositionEncoding):
    """Custom position encoding implementation"""
    
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
        # Return frequency sequence
        return np.arange(self.dim // 2) * 0.1
    
    def encode(self, positions: np.ndarray) -> np.ndarray:
        # Implement encoding logic
        ...
    
    def kernel(self, delta) -> np.ndarray:
        # Implement kernel function
        ...

# Manual registration (optional)
from core.pe_registry import _REGISTRY
_REGISTRY['mycustom'] = MyCustomPE
```

## File Structure

```
core/
├── pe_registry.py          # PE core implementation
├── PE_README.md            # This document
└── analysis/               # Analysis toolkit
    ├── geometry/           # Differential geometry
    ├── spectral/           # Spectral analysis
    ├── information/        # Information theory
    └── manifold/           # Manifold learning
```
