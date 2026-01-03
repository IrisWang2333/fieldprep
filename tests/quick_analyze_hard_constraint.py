#!/usr/bin/env python
"""
🔒 一键分析硬约束束的平衡性 - 直接运行！

在 VS Code 里：
1. 打开这个文件
2. 点击右上角的运行按钮 ▶️
3. 查看报告

会自动分析最新的 bundles_hard_constraint.parquet 文件
"""

import sys
from pathlib import Path

# 添加 src 到路径
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_bundle_balance_hard_constraint import calculate_balance_metrics, print_balance_report, plot_distribution
from sd311_fieldprep.utils import paths
import geopandas as gpd


def main():
    print("\n" + "="*70)
    print("🔒 硬约束束平衡性分析")
    print("="*70)

    # ========== 配置区域 - 可以修改这里的值 ==========
    SESSION = "DH"              # 或 "D2DS" - 要分析哪个 session
    GENERATE_PLOT = True        # 是否生成图表
    TARGET_ADDRS = 80           # 目标地址数（用于硬约束分析）
    HARD_MAX_MULTIPLIER = 1.5   # 检查阈值：1.5x = split threshold (超过此值表示拆分失败)
    # 说明：
    # - Merge 硬约束: 1.1x (88) - 合并时严格限制
    # - Sweep 软约束: 1.1x (88) - 允许少量超出
    # - Split 阈值:   1.5x (120) - 超过此值会被拆分
    # ===============================================

    try:
        # 查找束文件
        root, cfg, out_root = paths()
        bundle_dir = out_root / "bundles" / SESSION
        bundle_file = bundle_dir / "bundles_multibfs_regroup.parquet"

        if not bundle_file.exists():
            print(f"\n❌ 错误: 未找到 Multi-BFS + Regroup 束文件")
            print(f"   预期位置: {bundle_file}")
            print(f"\n💡 请先运行对比测试:")
            print(f"   python tests/quick_comparison_hard_constraint.py")
            print(f"\n可用的束文件:")

            # 列出可用文件
            if bundle_dir.exists():
                for f in bundle_dir.glob("bundles*.parquet"):
                    print(f"   - {f.name}")

            return

        print(f"\n📂 分析文件: {bundle_file}")

        # 加载数据
        bundles = gpd.read_parquet(bundle_file)
        print(f"   ✓ 加载了 {len(bundles)} 个路段")

        # 计算指标（包含硬约束分析）
        print(f"\n🔍 计算指标...")
        print(f"   Target: {TARGET_ADDRS} addresses")
        print(f"   Split threshold: {int(TARGET_ADDRS * HARD_MAX_MULTIPLIER)} addresses ({HARD_MAX_MULTIPLIER}x)")
        metrics, bundle_stats = calculate_balance_metrics(
            bundles,
            target_addrs=TARGET_ADDRS,
            hard_max_multiplier=HARD_MAX_MULTIPLIER
        )

        # 打印报告
        print_balance_report(metrics)

        # 生成图表
        if GENERATE_PLOT:
            output_path = bundle_dir / "bundle_analysis_multibfs_regroup.png"
            hard_max = int(TARGET_ADDRS * HARD_MAX_MULTIPLIER)
            plot_distribution(bundle_stats, output_path, hard_max=hard_max)
            print(f"\n📊 图表已保存: {output_path}\n")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
