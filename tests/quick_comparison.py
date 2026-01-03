#!/usr/bin/env python
"""
一键对比测试脚本 - 无需命令行参数，直接运行！

在 VS Code 里：
1. 打开这个文件
2. 点击右上角的运行按钮 ▶️
3. 或者右键 -> Run Python File

会自动：
- 使用 DH session, target_addrs=120 的默认配置
- 对比 greedy vs multi_bfs 两种算法
- 生成报告和图表
"""

import sys
from pathlib import Path

# 添加 src 到路径
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 导入对比脚本的主要功能
from test_bundle_comparison import compare_methods, print_comparison_report, plot_comparison, save_detailed_report

from sd311_fieldprep.utils import load_sources, paths, project_to
import geopandas as gpd
from datetime import datetime


def main():
    print("\n" + "="*70)
    print("🚀 快速对比测试 - 贪婪 vs Multi-source BFS")
    print("="*70)

    # ========== 配置区域 - 可以修改这里的值 ==========
    SESSION = "DH"           # 或 "D2DS"
    TARGET_ADDRS = 120       # 目标地址数
    JOIN_TOL_M = 15.0        # 连接容差（米）
    SEED = 42                # 随机种子
    TAG = "locked"           # sweep 标签目录（None = 搜索所有）
    SWEEP_FILE = "eligible_b40_m2.parquet"  # ⭐ 指定要用的 sweep 文件（None = 使用最新）
    MIN_BUNDLE_SFH = 40      # ⭐ 合并小束：小于此地址数的束会被合并（None = 不合并）
    # ===============================================

    print(f"\n📋 配置:")
    print(f"   Session: {SESSION}")
    print(f"   Target addresses: {TARGET_ADDRS}")
    print(f"   Join tolerance: {JOIN_TOL_M}m")
    print(f"   Random seed: {SEED}")
    print(f"   Sweep tag: {TAG if TAG else 'all'}")
    print(f"   Sweep file: {SWEEP_FILE if SWEEP_FILE else 'latest'}")
    print(f"   Min bundle SFH: {MIN_BUNDLE_SFH if MIN_BUNDLE_SFH else 'None (no merging)'}")

    try:
        # 加载数据
        print("\n[1/4] 加载数据...")
        streets, addrs, ds, pr = load_sources()
        seg_id = pr["fields"]["streets_segment_id"]

        root, cfg, out_root = paths()
        sweep_root = out_root / "sweep"
        if TAG:
            sweep_root = sweep_root / TAG

        # ⭐ 如果指定了具体文件，直接使用；否则选最新的
        if SWEEP_FILE:
            sweep_file = sweep_root / SWEEP_FILE
            if not sweep_file.exists():
                print(f"\n❌ 错误: 找不到指定的文件 {SWEEP_FILE}")
                print(f"   路径: {sweep_file}")
                print(f"\n可用文件:")
                for p in sweep_root.glob("eligible_*.parquet"):
                    print(f"   - {p.name}")
                return
            segs = gpd.read_parquet(sweep_file)
            print(f"   ✓ 使用指定文件: {sweep_file.name}")
        else:
            cands = sorted((p for p in sweep_root.rglob("eligible_*.parquet")),
                          key=lambda p: p.stat().st_mtime)
            if not cands:
                print("\n❌ 错误: 没有找到 eligible_*.parquet 文件")
                print("   请先运行 sweep 命令:")
                print("   python cli.py sweep --buffers 25 --mins 6 --tag default")
                return
            latest = cands[-1]
            segs = gpd.read_parquet(latest)
            print(f"   ✓ 使用最新文件: {latest.name}")

        print(f"   ✓ 路段数量: {len(segs)}")

        # 准备数据
        print("\n[2/4] 准备数据...")
        work_epsg = int(pr.get("crs", {}).get("working_meters", 26911))
        segs_m = project_to(segs, work_epsg)
        print(f"   ✓ 投影到 CRS: {work_epsg}")

        # 运行对比
        print("\n[3/4] 运行算法对比...")
        results = compare_methods(segs_m, seg_id, TARGET_ADDRS, JOIN_TOL_M, SEED, MIN_BUNDLE_SFH)

        # 生成报告
        print("\n[4/4] 生成报告...")
        print_comparison_report(results)

        # 保存输出
        output_dir = root / "outputs" / "tests"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = output_dir / f"quick_comparison_{timestamp}.png"
        csv_path = output_dir / f"quick_comparison_{timestamp}.csv"

        plot_comparison(results, plot_path)
        save_detailed_report(results, csv_path)

        # ⭐ 新增：保存 multi_bfs 的结果到 bundles.parquet
        bundle_dir = out_root / "bundles" / SESSION
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_file = bundle_dir / "bundles.parquet"

        multi_bfs_result = results["multi_bfs"]["bundled"]
        multi_bfs_result.to_parquet(bundle_file, index=False)
        print(f"\n💾 已保存 Multi-BFS 结果:")
        print(f"   {bundle_file}")
        print(f"   (可用 quick_analyze.py 分析)")

        # ⭐ 新增：生成地图
        try:
            import folium
            from sd311_fieldprep.utils import folium_map

            print("\n🗺️  生成地图...")
            viz = multi_bfs_result.dropna(subset=["bundle_id"]).to_crs(4326)
            tooltip_cols = [seg_id, "bundle_id", "bundle_seg_count", "bundle_addr_total"]

            m = folium_map(
                viz,
                color_col="bundle_id",
                tooltip_cols=tooltip_cols
            )
            map_file = bundle_dir / "bundles_map.html"
            m.save(str(map_file))
            print(f"   ✓ 地图已保存: {map_file}")
        except Exception as e:
            print(f"   ⚠️  地图生成失败: {e}")

        print(f"\n{'='*70}")
        print("✅ 对比完成!")
        print(f"{'='*70}")
        print(f"\n📁 输出文件:")
        print(f"   图表: {plot_path}")
        print(f"   数据: {csv_path}")
        print(f"   束文件: {bundle_file}")
        print(f"   地图: {bundle_dir / 'bundles_map.html'}")
        print()

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 提示: 请确保已经运行过 sweep 命令")


if __name__ == "__main__":
    main()
