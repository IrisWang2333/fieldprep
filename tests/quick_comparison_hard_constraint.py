#!/usr/bin/env python
"""
🔒 Multi-BFS 平衡约束版本对比测试 🔒

对比两个版本的 bundle 算法。

在 VS Code 里：
1. 打开这个文件
2. 点击右上角的运行按钮 ▶️
3. 或者右键 -> Run Python File

对比两个版本：
1. Greedy (无约束) - 标准版本 (5 步)
   - 基于距离合并，无大小约束

2. Multi-BFS (平衡 + 拆分 + 重组) - 新版本 (10 步)
   - Step 3: 连通性合并，无约束，优先合并到最小的 bundle
   - Step 5: 自动拆分 > 1.0x target (80 addresses)
   - Step 6, 8: 硬约束清理 ≤ 1.1x (88 addresses)
   - Step 9: 循环重组不合格的 bundles ([0.9x, 1.1x] = [72, 88])
   - Step 10: Eulerian 验证 & 智能重组
"""

import sys
from pathlib import Path

# 添加 src 到路径
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.utils import load_sources, paths, project_to
import geopandas as gpd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def calculate_bundle_metrics(bundled_gdf):
    """Calculate key metrics for bundle distribution."""
    bundle_stats = bundled_gdf.groupby("bundle_id").agg({
        "sfh_addr_count": "sum",
        "bundle_seg_count": "first"
    }).rename(columns={"sfh_addr_count": "total_addrs"})

    sizes = bundle_stats["total_addrs"].values

    metrics = {
        "n_bundles": len(bundle_stats),
        "mean_size": float(np.mean(sizes)),
        "median_size": float(np.median(sizes)),
        "std_size": float(np.std(sizes)),
        "cv": float(np.std(sizes) / np.mean(sizes)),  # Coefficient of Variation
        "min_size": int(np.min(sizes)),
        "max_size": int(np.max(sizes)),
        "range_ratio": float(np.max(sizes) / np.min(sizes)),
    }

    # Mean Absolute Standardized Difference
    target = np.mean(sizes)
    masd = float(np.mean(np.abs(sizes - target) / target))
    metrics["masd"] = masd

    return metrics, bundle_stats


def plot_comparison(results, output_path=None):
    """Generate comparison visualization - side-by-side histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Define display names for better plot titles
    display_names = {
        "greedy_no_constraint": "GREEDY",
        "multibfs_hard_1.0x_split": "MULTI BFS (Hard Constraint)"
    }

    methods = list(results.keys())
    colors = ['blue', 'green']

    for idx, method in enumerate(methods):
        ax = axes[idx]
        sizes = results[method]["bundle_stats"]["total_addrs"].values

        ax.hist(sizes, bins=20, alpha=0.7, color=colors[idx], edgecolor='black')
        ax.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(sizes):.1f}')
        ax.axvline(np.median(sizes), color='orange', linestyle=':', linewidth=2,
                   label=f'Median: {np.median(sizes):.1f}')

        ax.set_xlabel('Bundle Size (addresses)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        # Use display name if available, otherwise format method name
        title_name = display_names.get(method, method.replace("_", " ").upper())
        ax.set_title(f'{title_name}\nCV={results[method]["metrics"]["cv"]:.3f}, '
                     f'MASD={results[method]["metrics"]["masd"]:.3f}',
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved: {output_path}")
    else:
        plt.show()

    plt.close()


def compare_with_hard_constraint():
    print("\n" + "="*70)
    print("🔒 硬约束版本对比测试")
    print("="*70)

    # ========== 配置区域 ==========
    SESSION = "DH"
    TARGET_ADDRS = 80
    JOIN_TOL_M = 15.0
    SEED = 42
    TAG = "locked"
    SWEEP_FILE = "eligible_b40_m2.parquet"
    MIN_BUNDLE_SFH = 72
    HARD_MAX_MULTIPLIER = 1.1  # 硬约束：最大 110% * target = 88 addresses (Step 6, 8)
    # =============================

    print(f"\n📋 配置:")
    print(f"   Target: {TARGET_ADDRS} addresses")
    print(f"   🔒 Hard max (Step 6, 8): {int(TARGET_ADDRS * HARD_MAX_MULTIPLIER)} addresses ({HARD_MAX_MULTIPLIER}x)")
    print(f"   🔒 Hard min (Step 9): {int(TARGET_ADDRS * 0.9)} addresses (0.9x)")
    print(f"   ✂️  Split threshold (Step 5): {TARGET_ADDRS} addresses (1.0x)")

    try:
        # 加载数据
        print("\n[1/4] 加载数据...")
        streets, addrs, ds, pr = load_sources()
        seg_id = pr["fields"]["streets_segment_id"]

        root, cfg, out_root = paths()
        sweep_root = out_root / "sweep" / TAG

        sweep_file = sweep_root / SWEEP_FILE
        if not sweep_file.exists():
            print(f"\n❌ 错误: 找不到文件 {SWEEP_FILE}")
            return

        segs = gpd.read_parquet(sweep_file)
        print(f"   ✓ 使用文件: {sweep_file.name}")
        print(f"   ✓ 路段数量: {len(segs)}")

        # 准备数据
        print("\n[2/4] 准备数据...")
        work_epsg = int(pr.get("crs", {}).get("working_meters", 26911))
        segs_m = project_to(segs, work_epsg)

        # 运行对比
        print("\n[3/4] 运行两个版本的对比...")
        results = {}

        # 版本 1: Greedy (无约束) - 使用标准 bundle.py
        print("\n>>> [1/2] Greedy (无约束)")
        from sd311_fieldprep.bundle import _build_connected_bundles as greedy_bundle
        bundled_greedy = greedy_bundle(
            segs_m, seg_id, TARGET_ADDRS, JOIN_TOL_M, SEED,
            min_bundle_sfh=MIN_BUNDLE_SFH
        )
        results["greedy_no_constraint"] = bundled_greedy

        # 版本 2: Multi-BFS (平衡 + 拆分 + 重组 + Eulerian) - 使用 bundle_hard_constraint.py
        print(f"\n>>> [2/2] Multi-BFS (平衡合并 + 拆分 1.0x + 清理 {HARD_MAX_MULTIPLIER}x + 重组 + Eulerian 验证)")
        from sd311_fieldprep.bundle_hard_constraint import _build_connected_bundles_multibfs
        bundled_hard = _build_connected_bundles_multibfs(
            segs_m, seg_id, TARGET_ADDRS, JOIN_TOL_M, SEED,
            min_bundle_sfh=MIN_BUNDLE_SFH,
            hard_max_multiplier=HARD_MAX_MULTIPLIER
        )
        results["multibfs_hard_1.0x_split"] = bundled_hard

        # 生成报告
        print("\n[4/4] 生成对比报告和图表...")

        # Calculate metrics for both versions
        comparison_results = {}
        for name, bundled in results.items():
            metrics, bundle_stats = calculate_bundle_metrics(bundled)
            comparison_results[name] = {
                "bundled": bundled,
                "metrics": metrics,
                "bundle_stats": bundle_stats
            }

        print("\n" + "="*70)
        print("📊 对比结果")
        print("="*70)

        print(f"\n{'版本':<30} {'Bundles':>10} {'Mean':>10} {'Max':>10} {'Min':>10} {'CV':>10} {'MASD':>10}")
        print("-" * 90)

        for name in comparison_results.keys():
            metrics = comparison_results[name]["metrics"]
            print(f"{name:<30} {metrics['n_bundles']:>10} {metrics['mean_size']:>10.1f} "
                  f"{metrics['max_size']:>10} {metrics['min_size']:>10} "
                  f"{metrics['cv']:>10.3f} {metrics['masd']:>10.3f}")

        # 保存 Multi-BFS + Regroup 版本
        print("\n💾 保存 Multi-BFS + Regroup 版本结果...")
        bundle_dir = out_root / "bundles" / SESSION
        bundle_dir.mkdir(parents=True, exist_ok=True)

        bundle_file = bundle_dir / "bundles_multibfs_regroup.parquet"
        bundled_hard.to_parquet(bundle_file, index=False)
        print(f"   ✓ 已保存: {bundle_file}")

        # 生成对比图表
        plot_file = bundle_dir / "comparison_greedy_vs_multibfs_regroup.png"
        plot_comparison(comparison_results, plot_file)

        # 生成地图
        map_file = bundle_dir / "bundles_multibfs_regroup_map.html"
        try:
            import folium
            from sd311_fieldprep.utils import folium_map

            print("\n🗺️  生成地图...")
            viz = bundled_hard.dropna(subset=["bundle_id"]).to_crs(4326)
            tooltip_cols = [seg_id, "bundle_id", "bundle_seg_count", "bundle_addr_total"]

            m = folium_map(viz, color_col="bundle_id", tooltip_cols=tooltip_cols)
            m.save(str(map_file))
            print(f"   ✓ 地图已保存: {map_file}")
        except Exception as e:
            print(f"   ⚠️  地图生成失败: {e}")

        print(f"\n{'='*70}")
        print("✅ 对比完成!")
        print(f"{'='*70}")
        print(f"\n关键发现:")
        greedy_metrics = comparison_results['greedy_no_constraint']['metrics']
        hard_metrics = comparison_results['multibfs_hard_1.0x_split']['metrics']
        print(f"  - Greedy 无约束版本:")
        print(f"    Max: {greedy_metrics['max_size']}, CV: {greedy_metrics['cv']:.3f}, MASD: {greedy_metrics['masd']:.3f}")
        print(f"  - Multi-BFS 平衡约束版本:")
        print(f"    Max: {hard_metrics['max_size']}, CV: {hard_metrics['cv']:.3f}, MASD: {hard_metrics['masd']:.3f}")
        print(f"\n💡 说明:")
        print(f"  - Step 3: 连通性合并，无约束，优先合并到最小的 bundle")
        print(f"  - Step 5: 自动拆分 > {TARGET_ADDRS} addresses (1.0x target)")
        print(f"  - Step 6, 8: 硬约束清理 ≤ {int(TARGET_ADDRS * HARD_MAX_MULTIPLIER)} addresses ({HARD_MAX_MULTIPLIER}x)")
        print(f"  - Step 9: 循环重组不合格的 bundles [{int(TARGET_ADDRS * 0.9)}, {int(TARGET_ADDRS * HARD_MAX_MULTIPLIER)}] addresses")
        print(f"  - Step 10: Eulerian 验证 & 智能重组（确保所有 bundles 可行走）")
        print(f"\n📁 输出文件:")
        print(f"  - 对比图表: {plot_file}")
        print(f"  - Bundle 数据: {bundle_file}")
        print(f"  - 地图: {map_file if map_file.exists() else '(未生成)'}")
        print()

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    compare_with_hard_constraint()
