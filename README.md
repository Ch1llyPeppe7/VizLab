# VizLab — Mathematical Visualization Research Laboratory

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

> A comprehensive mathematical visualization platform for deep learning research, starting from position encoding analysis and extending to differential geometry, spectral analysis, information theory, and Transformer internals.

## 🏗️ Project Structure

```
VizLab/
├── core/                          # Core library
│   ├── pe_registry.py             # Position encoding registry (4 PE schemes)
│   ├── PE_README.md               # PE module documentation
│   ├── math_utils.py              # Mathematical utilities
│   ├── plot_utils.py              # Plotting utilities (Matplotlib/Plotly/HTML)
│   ├── viz_logger.py              # Structured JSON logging
│   ├── analysis/                  # General-purpose analysis toolkit
│   │   ├── geometry/              # Differential geometry (curvature, torsion, metric)
│   │   ├── spectral/              # Spectral analysis (FFT, PSD, entropy, STFT)
│   │   ├── information/           # Information theory (entropy, MI, Fisher, KL)
│   │   └── manifold/              # Manifold learning (PCA, t-SNE, UMAP)
│   └── examples/                  # Example scripts
│
├── pe_analysis/                   # Position encoding research (main scripts)
│   ├── 01_unified_comparison.py   # Unified mathematical framework comparison
│   ├── 02_spectral_analysis.py    # Frequency domain analysis
│   ├── 03_chaos_propagation.py    # Deep propagation chaos analysis ★
│   ├── 04_manifold_visualization.py  # Geometric manifold visualization
│   ├── 05_attention_patterns.py   # Attention pattern analysis
│   ├── 06_extrapolation_analysis.py  # Extrapolation and length generalization
│   ├── 07_information_theory.py   # Information-theoretic analysis
│   └── 08_differential_geometry.py  # Differential geometry (curvature/torsion/metric)
│
├── legacy/                        # Archived content
│   └── scripts_v1/                # Early standalone scripts (superseded by pe_analysis)
│
├── output/                        # Output directory (PDF/PNG)
│   └── pe_analysis/               # PE analysis output charts
├── html/                          # Interactive HTML visualizations
│   └── pe_analysis/               # Plotly interactive charts
├── run.py                         # Unified entry point
├── requirements.txt
└── README.md
```

## 🔬 Research Highlights

### ★ Additive PE Chaos vs. Multiplicative PE Structural Preservation

**Key Insight**: Additive absolute position encodings (e.g., Sinusoidal PE) inject position information only at the first layer. As embeddings propagate through multiple feed-forward layers, positional signals gradually become chaotic. In contrast, multiplicative relative position encodings (e.g., RoPE) re-apply rotations at every layer's attention computation, inherently preserving positional structure.

This repository rigorously simulates and visualizes this phenomenon from a purely mathematical perspective:

- **Lyapunov Exponents**: Quantify the degree of chaos in position signal propagation
- **Phase Space Diagrams**: Visualize orbital evolution of position embeddings
- **Mutual Information Decay**: Compare position information retention across PE schemes

### Unified Mathematical Framework

We analyze four PE schemes from a unified perspective combining complex analysis, group theory, and Bochner's theorem:

| Scheme | Mathematical Essence | Kernel Function | Geometric Structure |
|--------|---------------------|-----------------|---------------------|
| Sinusoidal | Additive, Fourier basis | K(Δ) = Σ cos(ωₖΔ) | Torus T^d |
| RoPE | Multiplicative, SO(2) rotation | K(Δ) = Σ cos(ωₖΔ) | Helical manifold |
| ALiBi | Additive bias, linear decay | K(Δ) = -m·\|Δ\| | Cone |
| LAPE | Additive, power-law frequency | K(Δ) ~ O(Δ^{-1/a}) | Kähler manifold |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run specific module
python run.py pe_analysis          # Run all PE analysis
python run.py pe_analysis.03       # Run only chaos propagation analysis
python run.py --list               # List all available modules

# Run all
python run.py --all
```

## 📦 Dependencies

- **Core**: numpy, matplotlib, scipy
- **Interactive**: plotly
- **Dimensionality Reduction**: scikit-learn, umap-learn
- **Animation**: manim (optional)
- **Signal Processing**: pywt (wavelets, optional)

## 📝 Design Principles

1. **Pure Mathematical Implementation**: All visualizations are based on mathematical derivations, independent of pretrained model weights
2. **Tool Agnostic**: Flexible choice between Matplotlib / Plotly / Manim / D3.js based on context
3. **Structured Logging**: Each visualization automatically outputs JSON data for LLM analysis
4. **Modular Independence**: Each subdirectory can run independently without cross-dependencies

## 📄 License

MIT License
