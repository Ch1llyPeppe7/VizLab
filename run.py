
#!/usr/bin/env python3
"""
VizLab — 统一运行入口

Usage:
    python run.py                           # 列出所有可用模块
    python run.py pe_analysis               # 运行 PE 分析全部脚本
    python run.py pe_analysis.spectral      # 运行单个脚本
    python run.py --list                    # 列出所有可用脚本
    python run.py --check                   # 检查环境依赖
"""

import sys
import importlib
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# ── 模块注册表 ──────────────────────────────────────────────────────────
MODULES = {
    "pe_analysis": {
        "description": "位置编码深度分析",
        "scripts": {
            "comparison":   "pe_analysis.01_unified_comparison",
            "spectral":     "pe_analysis.02_spectral_analysis",
            "chaos":        "pe_analysis.03_chaos_propagation",
            "manifold":     "pe_analysis.04_manifold_visualization",
            "attention":    "pe_analysis.05_attention_patterns",
            "extrapolation":"pe_analysis.06_extrapolation_analysis",
            "information":  "pe_analysis.07_information_theory",
        }
    },
    "diff_geometry": {
        "description": "微分几何可视化",
        "scripts": {}
    },
    "stats_probability": {
        "description": "统计学与概率可视化",
        "scripts": {}
    },
    "signal_processing": {
        "description": "信号处理可视化",
        "scripts": {}
    },
    "numerical_analysis": {
        "description": "数值分析可视化",
        "scripts": {}
    },
    "transformer_internals": {
        "description": "Transformer 架构内部可视化",
        "scripts": {}
    },
}


def check_environment():
    """检查关键依赖"""
    checks = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
        "plotly": "plotly",
        "sklearn": "sklearn",
    }
    optional = {
        "manim": "manim",
        "torch": "torch",
    }
    
    print("=" * 60)
    print("  VizLab 环境检查")
    print("=" * 60)
    
    all_ok = True
    print("\n[必需依赖]")
    for name, pkg in checks.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, '__version__', '?')
            print(f"  ✓ {name:<15} {ver}")
        except ImportError:
            print(f"  ✗ {name:<15} 未安装")
            all_ok = False
    
    print("\n[可选依赖]")
    for name, pkg in optional.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, '__version__', '?')
            print(f"  ✓ {name:<15} {ver}")
        except ImportError:
            print(f"  ○ {name:<15} 未安装（可选）")
    
    # Core 模块
    print("\n[Core 模块]")
    try:
        import core
        print(f"  ✓ core v{core.__version__}")
        pe_list = core.list_pe()
        print(f"    已注册 PE 方案: {', '.join(pe_list)}")
    except Exception as e:
        print(f"  ✗ core 导入失败: {e}")
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ 环境就绪，可以运行所有模块")
    else:
        print("  ⚠️ 部分依赖缺失，请运行: pip install -r requirements.txt")
    print("=" * 60)


def list_all_scripts():
    """列出所有可用脚本"""
    print("\n" + "=" * 60)
    print("  VizLab — 可用模块与脚本")
    print("=" * 60)
    
    for mod_name, mod_info in MODULES.items():
        desc = mod_info["description"]
        scripts = mod_info["scripts"]
        status = f"({len(scripts)} 个脚本)" if scripts else "(待开发)"
        print(f"\n  📦 {mod_name} — {desc} {status}")
        for script_alias, script_path in scripts.items():
            print(f"      ▸ {mod_name}.{script_alias}")
    
    print("\n" + "=" * 60)
    print("  用法: python run.py <模块名>[.脚本名]")
    print("=" * 60 + "\n")


def run_script(module_path: str):
    """运行指定脚本的 main() 函数"""
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        print(f"  ✗ 模块未找到: {module_path}")
        print(f"    错误: {e}")
        sys.exit(1)
    
    if hasattr(mod, 'main'):
        print(f"  ▶ 运行 {module_path}.main()")
        print("-" * 60)
        mod.main()
        print("-" * 60)
        print(f"  ✓ {module_path} 完成")
    else:
        print(f"  ⚠ {module_path} 没有 main() 函数，跳过")


def main():
    parser = argparse.ArgumentParser(
        description="VizLab — 数学可视化研究实验室",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py --list                    列出所有可用脚本
  python run.py --check                   检查环境依赖
  python run.py pe_analysis               运行 PE 分析全部脚本
  python run.py pe_analysis.spectral      运行单个脚本
        """
    )
    parser.add_argument("target", nargs="?", help="目标模块或脚本")
    parser.add_argument("--list", action="store_true", help="列出所有可用脚本")
    parser.add_argument("--check", action="store_true", help="检查环境依赖")
    
    args = parser.parse_args()
    
    if args.check:
        check_environment()
        return
    
    if args.list or args.target is None:
        list_all_scripts()
        return
    
    target = args.target
    parts = target.split(".", 1)
    module_name = parts[0]
    
    if module_name not in MODULES:
        print(f"  ✗ 未知模块: {module_name}")
        print(f"    可用模块: {', '.join(MODULES.keys())}")
        sys.exit(1)
    
    mod_info = MODULES[module_name]
    
    if len(parts) == 1:
        # 运行整个模块的所有脚本
        scripts = mod_info["scripts"]
        if not scripts:
            print(f"  ⚠ {module_name} 暂无可用脚本")
            return
        print(f"\n  ▶ 运行模块 [{module_name}] 的所有脚本 ({len(scripts)} 个)\n")
        for alias, path in scripts.items():
            run_script(path)
            print()
    else:
        # 运行单个脚本
        script_alias = parts[1]
        scripts = mod_info["scripts"]
        if script_alias not in scripts:
            print(f"  ✗ 未知脚本: {target}")
            if scripts:
                print(f"    可用脚本: {', '.join(module_name + '.' + a for a in scripts)}")
            sys.exit(1)
        run_script(scripts[script_alias])


if __name__ == "__main__":
    main()
