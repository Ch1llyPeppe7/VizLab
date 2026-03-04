# VizLab — 数学可视化研究实验室

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

> 一个面向深度学习研究的综合性数学可视化平台，以位置编码分析为起点，扩展至微分几何、谱分析、信息论及 Transformer 架构内部机制研究。

## 🏗️ 项目结构

```
VizLab/
├── core/                          # 核心库
│   ├── pe_registry.py             # 位置编码统一注册表 (支持 4 种 PE 方案)
│   ├── PE_README.md               # PE 模块使用文档
│   ├── math_utils.py              # 数学工具函数
│   ├── plot_utils.py              # 绑图工具 (Matplotlib/Plotly/HTML)
│   ├── viz_logger.py              # 结构化 JSON 日志
│   ├── analysis/                  # 通用分析工具库
│   │   ├── geometry/              # 微分几何 (曲率、挠率、度量张量)
│   │   ├── spectral/              # 谱分析 (FFT、PSD、谱熵、STFT)
│   │   ├── information/           # 信息论 (熵、互信息、Fisher 信息、KL 散度)
│   │   └── manifold/              # 流形学习 (PCA、t-SNE、UMAP)
│   └── examples/                  # 示例脚本
│
├── pe_analysis/                   # 位置编码深度研究 (主力脚本)
│   ├── 01_unified_comparison.py   # 统一数学框架对比
│   ├── 02_spectral_analysis.py    # 频域谱分析
│   ├── 03_chaos_propagation.py    # 深层传播混沌性分析 ★
│   ├── 04_manifold_visualization.py  # 几何流形可视化
│   ├── 05_attention_patterns.py   # 注意力模式分析
│   ├── 06_extrapolation_analysis.py  # 外推性与长度泛化
│   ├── 07_information_theory.py   # 信息论分析
│   └── 08_differential_geometry.py  # 微分几何分析 (曲率/挠率/度量)
│
├── legacy/                        # 归档内容
│   └── scripts_v1/                # 早期独立脚本 (已被 pe_analysis 取代)
│
├── output/                        # 输出目录 (PDF/PNG)
│   └── pe_analysis/               # PE 分析输出图表
├── html/                          # 交互式 HTML 可视化
│   └── pe_analysis/               # Plotly 交互式图表
├── run.py                         # 统一运行入口
├── requirements.txt
└── README.md
```

## 🔬 核心研究亮点

### ★ 加性位置编码的混沌性 vs 乘性位置编码的结构保持性

**核心洞察**：加性绝对位置编码（如 Sinusoidal PE）仅在第一层注入位置信息。随着嵌入向量在多层前馈网络中传播，位置信号会逐渐呈现混沌特性。相比之下，乘性相对位置编码（如 RoPE）在每一层的注意力计算中重新施加旋转变换，天然地保持了位置信息的结构完整性。

本仓库从纯数学角度严格模拟并可视化这一现象：

- **Lyapunov 指数**：量化位置信号在多层传播中的混沌程度
- **相空间图**：展示位置嵌入的轨道演化过程
- **互信息衰减曲线**：对比不同 PE 方案的位置信息保持能力

### 统一数学框架

本项目从复分析、群论与 Bochner 定理的统一视角，对比分析四种主流位置编码方案：

| 方案 | 数学本质 | 核函数 | 几何结构 |
|------|----------|--------|----------|
| Sinusoidal | 加性，傅里叶基 | K(Δ) = Σ cos(ωₖΔ) | 环面 T^d |
| RoPE | 乘性，SO(2) 旋转 | K(Δ) = Σ cos(ωₖΔ) | 螺旋流形 |
| ALiBi | 加性偏置，线性衰减 | K(Δ) = -m·\|Δ\| | 锥面 |
| LAPE | 加性，幂律频率 | K(Δ) ~ O(Δ^{-1/a}) | Kähler 流形 |

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行特定模块
python run.py pe_analysis          # 运行全部 PE 分析
python run.py pe_analysis.03       # 仅运行混沌传播分析
python run.py --list               # 列出所有可用模块

# 运行全部分析
python run.py --all
```

## 📦 依赖项

- **核心库**：numpy, matplotlib, scipy
- **交互可视化**：plotly
- **降维算法**：scikit-learn, umap-learn
- **动画渲染**：manim (可选)
- **信号处理**：pywt (小波分析，可选)

## 📝 设计原则

1. **纯数学实现**：所有可视化均基于数学推导，不依赖预训练模型权重
2. **工具无关性**：根据场景灵活选择 Matplotlib / Plotly / Manim / D3.js
3. **结构化日志**：每个可视化自动输出 JSON 数据，便于 LLM 分析与复现
4. **模块独立性**：各子目录可独立运行，无交叉依赖

## 📄 许可证

MIT License
