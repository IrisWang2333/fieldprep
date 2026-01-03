#!/usr/bin/env python
"""
一键分析束的平衡性 - 直接运行！

在 VS Code 里：
1. 打开这个文件
2. 点击右上角的运行按钮 ▶️
3. 查看报告

会自动分析最新的 bundles.parquet 文件
"""

import sys
from pathlib import Path

# 添加 src 到路径
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_bundle_balance import calculate_balance_metrics, print_balance_report, plot_distribution
from sd311_fieldprep.utils import paths
import geopandas as gpd


def main():
    print("\n" + "="*70)
    print("📊 束平衡性分析")
    print("="*70)

    # ========== 配置区域 - 可以修改这里的值 ==========
    SESSION = "DH"           # 或 "D2DS" - 要分析哪个 session
    GENERATE_PLOT = True     # 是否生成图表
    # ===============================================

    try:
        # 查找束文件
        root, cfg, out_root = paths()
        bundle_dir = out_root / "bundles" / SESSION
        bundle_file = bundle_dir / "bundles.parquet"

        if not bundle_file.exists():
            print(f"\n❌ 错误: 未找到束文件")
            print(f"   预期位置: {bundle_file}")
            print(f"\n💡 请先运行 bundle 命令:")
            print(f"   python cli.py bundle --session {SESSION} --target_addrs 120")
            return

        print(f"\n📂 分析文件: {bundle_file}")

        # 加载数据
        bundles = gpd.read_parquet(bundle_file)
        print(f"   ✓ 加载了 {len(bundles)} 个路段")

        # 计算指标
        print("\n🔍 计算指标...")
        metrics, bundle_stats = calculate_balance_metrics(bundles)

        # 打印报告
        print_balance_report(metrics)

        # 生成图表
        if GENERATE_PLOT:
            output_path = bundle_dir / "bundle_analysis.png"
            plot_distribution(bundle_stats, output_path)
            print(f"\n📊 图表已保存: {output_path}\n")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
