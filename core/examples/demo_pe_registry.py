#!/usr/bin/env python3
"""
Position Encoding 模块示例

演示如何使用 core/pe_registry.py 进行位置编码分析。
"""

import sys
from pathlib import Path
# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from core.pe_registry import get_pe, get_all_pe, PEConfig, list_pe


def demo_basic_usage():
    """基础用法演示"""
    print("=" * 60)
    print("1. 基础用法")
    print("=" * 60)
    
    # 列出所有可用的 PE 方案
    print("\n可用的 PE 方案:")
    for name, info in list_pe().items():
        print(f"  - {name}: {info['class']} ({info['category']})")
    
    # 创建 RoPE 实例
    rope = get_pe('rope', dim=64, max_len=256)
    print(f"\n创建 PE: {rope.name}")
    print(f"  类别: {rope.category}")
    print(f"  数学描述: {rope.math_description[:60]}...")
    
    # 生成编码
    positions = np.arange(100)
    embeddings = rope.encode(positions)
    print(f"\n编码形状: {embeddings.shape}")
    print(f"  范围: [{embeddings.min():.3f}, {embeddings.max():.3f}]")


def demo_kernel_analysis():
    """核函数分析演示"""
    print("\n" + "=" * 60)
    print("2. 核函数分析")
    print("=" * 60)
    
    config = PEConfig(dim=128)
    all_pes = get_all_pe(config=config)
    
    # 计算核函数衰减
    deltas = np.arange(0, 50)
    
    print("\n核函数 K(Δ) 值 (Δ=0, 10, 20, 30):")
    for name, pe in all_pes.items():
        k_values = pe.kernel(deltas)
        print(f"  {name:12s}: K(0)={k_values[0]:.3f}, K(10)={k_values[10]:.3f}, "
              f"K(20)={k_values[20]:.3f}, K(30)={k_values[30]:.3f}")


def demo_rope_rotation():
    """RoPE 旋转演示"""
    print("\n" + "=" * 60)
    print("3. RoPE 旋转操作")
    print("=" * 60)
    
    rope = get_pe('rope', dim=64)
    
    # 获取旋转矩阵
    R = rope.rotation_matrix(position=5, freq_idx=0)
    print(f"\n位置 5 处第 0 子空间的旋转矩阵:")
    print(f"  [[{R[0,0]:+.4f}, {R[0,1]:+.4f}],")
    print(f"   [{R[1,0]:+.4f}, {R[1,1]:+.4f}]]")
    
    # 验证 det(R) = 1 (SO(2))
    print(f"  det(R) = {np.linalg.det(R):.6f} (应为 1.0)")
    
    # 应用旋转到向量
    x = np.random.randn(10, 64)
    positions = np.arange(10)
    x_rotated = rope.apply_rotary(x, positions)
    
    # 验证范数保持（旋转是等距变换）
    norms_before = np.linalg.norm(x, axis=1)
    norms_after = np.linalg.norm(x_rotated, axis=1)
    print(f"\n旋转前后范数变化: max|Δ| = {np.abs(norms_before - norms_after).max():.2e}")


def demo_alibi_bias():
    """ALiBi 偏置演示"""
    print("\n" + "=" * 60)
    print("4. ALiBi 偏置矩阵")
    print("=" * 60)
    
    alibi = get_pe('alibi', dim=64, n_heads=8)
    
    print(f"\n斜率 (8 heads): {alibi.slopes}")
    
    # 生成偏置矩阵
    bias = alibi.bias_matrix(seq_len=10, head_idx=0)
    print(f"\n偏置矩阵 (head 0, 10x10):")
    print("  对角线值:", bias.diagonal()[:5])
    print("  |i-j|=1 偏置:", bias[0, 1])
    print("  |i-j|=5 偏置:", bias[0, 5])


def demo_complex_embedding():
    """复数嵌入演示"""
    print("\n" + "=" * 60)
    print("5. 复数嵌入")
    print("=" * 60)
    
    pe = get_pe('sinusoidal', dim=64)
    positions = np.arange(10)
    
    # 复数编码
    z = pe.encode_complex(positions)
    print(f"\n复数嵌入形状: {z.shape}")
    print(f"  位置 0: |z| = {np.abs(z[0]).mean():.4f} (应为 1.0)")
    
    # 验证 |e^{iθ}| = 1
    magnitudes = np.abs(z)
    print(f"  所有位置 |z|: mean={magnitudes.mean():.6f}, std={magnitudes.std():.2e}")


def demo_visualization():
    """可视化演示"""
    print("\n" + "=" * 60)
    print("6. 可视化")
    print("=" * 60)
    
    config = PEConfig(dim=64, max_len=256)
    positions = np.arange(100)
    deltas = np.arange(0, 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 编码热力图
    ax = axes[0, 0]
    pe = get_pe('sinusoidal', config=config)
    emb = pe.encode(positions[:50])
    im = ax.imshow(emb, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Position')
    ax.set_title('Sinusoidal PE Heatmap')
    plt.colorbar(im, ax=ax)
    
    # 2. 核函数衰减对比
    ax = axes[0, 1]
    for name, pe in get_all_pe(config=config).items():
        k = pe.kernel(deltas)
        ax.plot(deltas, k, label=name, linewidth=2)
    ax.set_xlabel('Δ (Position Difference)')
    ax.set_ylabel('K(Δ)')
    ax.set_title('Kernel Function Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 频率分布
    ax = axes[1, 0]
    for name, pe in get_all_pe(config=config).items():
        freqs = pe.get_frequencies()
        ax.semilogy(freqs, 'o-', label=name, markersize=3, alpha=0.7)
    ax.set_xlabel('Frequency Index k')
    ax.set_ylabel('ω_k (log scale)')
    ax.set_title('Frequency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. ALiBi 偏置矩阵
    ax = axes[1, 1]
    alibi = get_pe('alibi', dim=64, n_heads=8)
    bias = alibi.bias_matrix(seq_len=50, head_idx=0)
    im = ax.imshow(bias, cmap='Blues_r')
    ax.set_xlabel('Key Position j')
    ax.set_ylabel('Query Position i')
    ax.set_title('ALiBi Bias Matrix (Head 0)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'pe_demo_output.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n图表已保存到: {output_path}")
    plt.close()


def demo_with_analysis_lib():
    """与 core/analysis 库配合使用"""
    print("\n" + "=" * 60)
    print("7. 与 core/analysis 配合使用")
    print("=" * 60)
    
    try:
        from core.analysis import geometry, spectral
    except ImportError:
        print("  [跳过] core.analysis 模块未安装")
        return
    
    config = PEConfig(dim=128, max_len=256)
    positions = np.arange(200)
    
    print("\n各 PE 方案的几何特性:")
    print("-" * 50)
    
    for name, pe in get_all_pe(config=config).items():
        emb = pe.encode(positions)
        
        # 微分几何分析
        d1, d2, d3 = geometry.compute_derivatives(emb)
        kappa = geometry.curvature(d1, d2)
        g = geometry.metric_tensor(d1)
        s = geometry.arc_length(g)
        
        print(f"{name:12s}: curvature={kappa.mean():.4f}, arc_length={s[-1]:.2f}")


def main():
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("  Position Encoding 模块演示")
    print("=" * 60)
    
    demo_basic_usage()
    demo_kernel_analysis()
    demo_rope_rotation()
    demo_alibi_bias()
    demo_complex_embedding()
    demo_visualization()
    demo_with_analysis_lib()
    
    print("\n" + "=" * 60)
    print("  演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
