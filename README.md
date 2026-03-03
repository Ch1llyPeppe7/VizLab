
# VizLab — 数学可视化研究实验室

> 从位置编码的复分析研究出发，扩展为覆盖微分几何、统计学、信号处理、数值分析、Transformer 架构的综合数学可视化平台。

## 🏗️ 项目结构

```
VizLab/
├── core/                          # 通用核心库
│   ├── __init__.py
│   ├── math_utils.py              # 数学工具（频率函数、核函数、编码器）
│   ├── plot_utils.py              # 绘图工具（Matplotlib/Plotly/HTML）
│   ├── viz_logger.py              # 结构化 JSON 日志
│   └── pe_registry.py             # 位置编码统一注册表
│
├── pe_analysis/                   # 位置编码深度研究
│   ├── 01_pe_comparison.py        # PE 方案统一数学框架对比
│   ├── 02_spectral_analysis.py    # 频域谱分析
│   ├── 03_deep_propagation.py     # 深层传播混沌性分析 ★
│   ├── 04_geometric_manifold.py   # 几何流形可视化
│   ├── 05_attention_pattern.py    # Attention Pattern 影响
│   ├── 06_extrapolation.py        # 外推性与长度泛化
│   ├── 07_information_theory.py   # 信息论分析
│   └── README.md
│
├── diff_geometry/                 # 微分几何可视化
│   ├── 01_geodesics.py            # 测地线动画
│   ├── 02_curvature.py            # 高斯曲率热力图
│   ├── 03_parallel_transport.py   # 平行移动
│   ├── 04_lie_group.py            # Lie 群指数映射
│   └── README.md
│
├── statistics/                    # 统计学与概率可视化
│   ├── 01_distributions.py        # 分布族形状探索
│   ├── 02_bayesian_update.py      # 贝叶斯后验更新
│   ├── 03_mcmc.py                 # MCMC 采样轨迹
│   ├── 04_clt.py                  # 中心极限定理
│   └── README.md
│
├── signal_processing/             # 信号处理可视化
│   ├── 01_fourier_series.py       # 傅里叶级数逼近
│   ├── 02_stft.py                 # 短时傅里叶变换
│   ├── 03_wavelets.py             # 小波多分辨率
│   ├── 04_filters.py              # 数字滤波器极零图
│   └── README.md
│
├── numerical_analysis/            # 数值分析可视化
│   ├── 01_root_finding.py         # 迭代法收敛轨迹
│   ├── 02_ode_stability.py        # ODE 稳定性区域
│   ├── 03_interpolation.py        # 插值与 Runge 现象
│   └── README.md
│
├── transformer_internals/         # Transformer 架构可视化
│   ├── 01_attention_rank.py       # Attention 秩分析
│   ├── 02_residual_stream.py      # 残差流可视化
│   ├── 03_layer_norm_geometry.py  # LayerNorm 几何效果
│   └── README.md
│
├── legacy/                        # 原始 LAPE 可视化（归档）
│   └── scripts/                   # 原 scripts/ 目录内容
│
├── output/                        # 输出目录（.gitignore）
├── html/                          # 交互式 HTML（.gitignore）
├── run.py                         # 统一运行入口
├── requirements.txt
└── README.md
```

## 🔬 核心研究亮点

### ★ 加性 PE 深层混沌 vs 乘性 PE 结构保持

**核心洞察**：加性绝对位置编码（Sinusoidal PE）虽然只在第一层注入，但经过多层前馈传播后位置信号会逐渐混沌化；而 RoPE 等乘性相对位置编码在每层的 Attention 计算中重新施加旋转，天然保持了位置信息的结构。

本仓库从纯数学角度严格模拟并可视化这一现象：

- **Lyapunov 指数**量化位置信号在多层传播中的混沌程度
- **相空间图**展示位置 embedding 的轨道演化
- **互信息衰减曲线**对比不同 PE 的位置信息保持能力

### 统一数学框架

从复分析 / 群论 / Bochner 定理的统一视角，对比四种 PE 方案：

| 方案 | 数学本质 | 核函数 | 几何结构 |
|------|----------|--------|----------|
| Sinusoidal | 加性，傅里叶基 | $K(\Delta) = \sum \cos(\omega_k \Delta)$ | 环面 $\mathbb{T}^d$ |
| RoPE | 乘性，$SO(2)$ 旋转 | $K(\Delta) = \sum \cos(\omega_k \Delta)$ | 螺旋流形 |
| ALiBi | 加性偏置，线性衰减 | $K(\Delta) = -m \|\Delta\|$ | 锥面 |
| LAPE | 加性，幂律频率 | $K(\Delta) \sim O(\Delta^{-1/a})$ | Kähler 流形 |

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行特定模块
python run.py pe_analysis          # 位置编码全部分析
python run.py pe_analysis.03       # 仅运行深层传播混沌性分析
python run.py diff_geometry        # 微分几何全部可视化
python run.py --list               # 列出所有可用模块

# 运行全部
python run.py --all
```

## 📦 依赖

- **核心**：numpy, matplotlib, scipy
- **交互**：plotly
- **降维**：scikit-learn, umap-learn
- **动画**：manim (可选)
- **信号处理**：pywt (小波，可选)

## 📝 设计原则

1. **纯数学实现**：所有可视化基于数学推导，不依赖预训练模型权重
2. **工具无关**：根据场景灵活选择 Matplotlib / Plotly / Manim / D3.js
3. **结构化日志**：每个可视化自动输出 JSON 数据，便于 LLM 分析
4. **模块独立**：每个子目录可独立运行，互不依赖
