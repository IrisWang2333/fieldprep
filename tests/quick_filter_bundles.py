#!/usr/bin/env python
"""
🔍 过滤 Bundles - 保留 [0.9x, 1.1x] 范围内的 bundles

在 VS Code 里：
1. 打开这个文件
2. 点击右上角的运行按钮 ▶️
3. 查看报告

功能：
- 过滤掉 < 72 或 > 88 addresses 的 bundles
- 生成新的 parquet 文件和地图
- 报告 dropped segments 比例和地址分布代表性
"""

import sys
from pathlib import Path

# 添加 src 到路径
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.utils import paths, load_sources
import geopandas as gpd
import numpy as np
import pandas as pd


def analyze_spatial_representativeness(original_gdf, filtered_gdf):
    """
    分析过滤后的空间代表性。

    检查：
    1. 地理分布：dropped segments 是否集中在某些区域
    2. 地址密度：dropped segments 的地址密度是否与保留的不同
    """
    original_total = len(original_gdf)
    filtered_total = len(filtered_gdf)
    dropped_total = original_total - filtered_total

    # 计算 dropped segments
    dropped_gdf = original_gdf[~original_gdf.index.isin(filtered_gdf.index)]

    print("\n" + "="*70)
    print("📊 空间代表性分析")
    print("="*70)

    # 1. 基本统计
    print(f"\n1️⃣ 基本统计:")
    print(f"   原始 segments: {original_total:,}")
    print(f"   保留 segments: {filtered_total:,}")
    print(f"   丢弃 segments: {dropped_total:,}")
    print(f"   丢弃比例: {dropped_total/original_total*100:.2f}%")

    # 2. 地址统计
    original_addrs = original_gdf['sfh_addr_count'].sum()
    filtered_addrs = filtered_gdf['sfh_addr_count'].sum()
    dropped_addrs = original_addrs - filtered_addrs

    print(f"\n2️⃣ 地址统计:")
    print(f"   原始地址: {int(original_addrs):,}")
    print(f"   保留地址: {int(filtered_addrs):,}")
    print(f"   丢弃地址: {int(dropped_addrs):,}")
    print(f"   丢弃比例: {dropped_addrs/original_addrs*100:.2f}%")

    # 3. 地址密度分布
    print(f"\n3️⃣ 地址密度分析:")

    original_density = original_gdf['sfh_addr_count'].describe()
    filtered_density = filtered_gdf['sfh_addr_count'].describe()
    dropped_density = dropped_gdf['sfh_addr_count'].describe() if len(dropped_gdf) > 0 else None

    print(f"\n   原始 segments 地址密度:")
    print(f"      Mean: {original_density['mean']:.1f}")
    print(f"      Median: {original_density['50%']:.1f}")
    print(f"      Std: {original_density['std']:.1f}")

    print(f"\n   保留 segments 地址密度:")
    print(f"      Mean: {filtered_density['mean']:.1f}")
    print(f"      Median: {filtered_density['50%']:.1f}")
    print(f"      Std: {filtered_density['std']:.1f}")

    if dropped_density is not None:
        print(f"\n   丢弃 segments 地址密度:")
        print(f"      Mean: {dropped_density['mean']:.1f}")
        print(f"      Median: {dropped_density['50%']:.1f}")
        print(f"      Std: {dropped_density['std']:.1f}")

    # 4. 空间分布检查（简化版：基于 centroid 的空间范围）
    print(f"\n4️⃣ 空间分布检查:")

    original_cents = original_gdf.geometry.centroid
    filtered_cents = filtered_gdf.geometry.centroid

    original_bounds = original_gdf.total_bounds
    filtered_bounds = filtered_gdf.total_bounds

    bounds_change = [
        abs(filtered_bounds[i] - original_bounds[i]) / (original_bounds[2] - original_bounds[0])
        for i in range(4)
    ]

    print(f"   原始边界: ({original_bounds[0]:.0f}, {original_bounds[1]:.0f}) - ({original_bounds[2]:.0f}, {original_bounds[3]:.0f})")
    print(f"   保留边界: ({filtered_bounds[0]:.0f}, {filtered_bounds[1]:.0f}) - ({filtered_bounds[2]:.0f}, {filtered_bounds[3]:.0f})")
    print(f"   边界变化: {max(bounds_change)*100:.2f}%")

    # 5. Bundle 大小分布
    print(f"\n5️⃣ Bundle 大小分布:")

    original_bundle_sizes = original_gdf.groupby('bundle_id')['sfh_addr_count'].sum()
    filtered_bundle_sizes = filtered_gdf.groupby('bundle_id')['sfh_addr_count'].sum()

    print(f"   原始 bundles: {len(original_bundle_sizes):,}")
    print(f"      Min: {int(original_bundle_sizes.min())}, Max: {int(original_bundle_sizes.max())}")
    print(f"      Mean: {original_bundle_sizes.mean():.1f}, Median: {original_bundle_sizes.median():.1f}")

    print(f"\n   过滤后 bundles: {len(filtered_bundle_sizes):,}")
    print(f"      Min: {int(filtered_bundle_sizes.min())}, Max: {int(filtered_bundle_sizes.max())}")
    print(f"      Mean: {filtered_bundle_sizes.mean():.1f}, Median: {filtered_bundle_sizes.median():.1f}")

    # 6. 结论
    print(f"\n6️⃣ 代表性结论:")

    addr_loss_pct = dropped_addrs/original_addrs*100
    seg_loss_pct = dropped_total/original_total*100
    bounds_change_pct = max(bounds_change)*100

    issues = []

    if addr_loss_pct > 10:
        issues.append(f"⚠️  地址丢失较多 ({addr_loss_pct:.1f}%)")
    else:
        issues.append(f"✅ 地址丢失可接受 ({addr_loss_pct:.1f}%)")

    if seg_loss_pct > 15:
        issues.append(f"⚠️  路段丢失较多 ({seg_loss_pct:.1f}%)")
    else:
        issues.append(f"✅ 路段丢失可接受 ({seg_loss_pct:.1f}%)")

    if bounds_change_pct > 5:
        issues.append(f"⚠️  空间范围明显变化 ({bounds_change_pct:.1f}%)")
    else:
        issues.append(f"✅ 空间范围基本不变 ({bounds_change_pct:.1f}%)")

    # 密度代表性
    if dropped_density is not None:
        density_diff = abs(dropped_density['mean'] - filtered_density['mean']) / original_density['mean']
        if density_diff > 0.2:
            issues.append(f"⚠️  丢弃的 segments 地址密度明显不同 ({density_diff*100:.1f}%)")
        else:
            issues.append(f"✅ 地址密度分布代表性良好")

    for issue in issues:
        print(f"   {issue}")

    print()


def filter_bundles():
    print("\n" + "="*70)
    print("🔍 过滤 Bundles - 保留 [0.9x, 1.1x] 范围")
    print("="*70)

    # ========== 配置区域 ==========
    SESSION = "DH"
    TARGET_ADDRS = 80
    LOWER_BOUND = TARGET_ADDRS * 0.9  # 72
    UPPER_BOUND = TARGET_ADDRS * 1.1  # 88
    # =============================

    print(f"\n📋 配置:")
    print(f"   Target: {TARGET_ADDRS} addresses")
    print(f"   下限: {int(LOWER_BOUND)} addresses (0.9x)")
    print(f"   上限: {int(UPPER_BOUND)} addresses (1.1x)")

    try:
        # 加载数据
        print("\n[1/4] 加载数据...")
        streets, addrs, ds, pr = load_sources()
        seg_id = pr["fields"]["streets_segment_id"]

        root, cfg, out_root = paths()
        bundle_dir = out_root / "bundles" / SESSION
        bundle_file = bundle_dir / "bundles_multibfs_regroup.parquet"

        if not bundle_file.exists():
            print(f"\n❌ 错误: 未找到文件 {bundle_file}")
            print(f"   请先运行 quick_comparison_hard_constraint.py")
            return

        bundles = gpd.read_parquet(bundle_file)
        print(f"   ✓ 加载了 {len(bundles)} 个路段")
        print(f"   ✓ 包含 {bundles['bundle_id'].nunique()} 个 bundles")

        # 过滤 bundles
        print(f"\n[2/4] 过滤 bundles...")

        # 计算每个 bundle 的总地址数
        bundle_sizes = bundles.groupby('bundle_id')['sfh_addr_count'].sum()

        # 找出符合条件的 bundles
        valid_bundles = bundle_sizes[
            (bundle_sizes >= LOWER_BOUND) &
            (bundle_sizes <= UPPER_BOUND)
        ].index

        # 找出不符合条件的 bundles
        too_small = bundle_sizes[bundle_sizes < LOWER_BOUND]
        too_large = bundle_sizes[bundle_sizes > UPPER_BOUND]

        print(f"   原始 bundles: {len(bundle_sizes)}")
        print(f"   ✅ 符合条件: {len(valid_bundles)} bundles")
        print(f"   ❌ 太小 (< {int(LOWER_BOUND)}): {len(too_small)} bundles")
        print(f"   ❌ 太大 (> {int(UPPER_BOUND)}): {len(too_large)} bundles")

        if len(too_small) > 0:
            print(f"      最小 bundle: {int(too_small.min())} addresses")
        if len(too_large) > 0:
            print(f"      最大 bundle: {int(too_large.max())} addresses")

        # 过滤 GeoDataFrame
        filtered_bundles = bundles[bundles['bundle_id'].isin(valid_bundles)].copy()

        print(f"\n   过滤结果:")
        print(f"   保留路段: {len(filtered_bundles)} / {len(bundles)}")
        print(f"   丢弃路段: {len(bundles) - len(filtered_bundles)}")
        print(f"   丢弃比例: {(len(bundles) - len(filtered_bundles))/len(bundles)*100:.2f}%")

        # 保存过滤后的数据
        print(f"\n[3/4] 保存过滤后的数据...")

        filtered_file = bundle_dir / "bundles_multibfs_regroup_filtered.parquet"
        filtered_bundles.to_parquet(filtered_file, index=False)
        print(f"   ✓ 已保存: {filtered_file}")

        # 生成地图
        print(f"\n[4/5] 生成地图...")

        try:
            import folium
            from sd311_fieldprep.utils import folium_map

            # 保留的 bundles 地图
            map_file = bundle_dir / "bundles_multibfs_regroup_filtered_map.html"

            viz = filtered_bundles.dropna(subset=["bundle_id"]).to_crs(4326)
            tooltip_cols = [seg_id, "bundle_id", "bundle_seg_count", "bundle_addr_total"]

            m = folium_map(viz, color_col="bundle_id", tooltip_cols=tooltip_cols)
            m.save(str(map_file))
            print(f"   ✓ 保留的地图已保存: {map_file}")
        except Exception as e:
            print(f"   ⚠️  保留的地图生成失败: {e}")

        # 筛掉的 bundles 地图
        print(f"\n[5/5] 生成筛掉的 bundles 地图...")

        try:
            # 找出被筛掉的 bundles
            dropped_bundle_ids = bundle_sizes[
                (bundle_sizes < LOWER_BOUND) | (bundle_sizes > UPPER_BOUND)
            ].index
            dropped_bundles = bundles[bundles['bundle_id'].isin(dropped_bundle_ids)].copy()

            if not dropped_bundles.empty:
                import folium
                from sd311_fieldprep.utils import folium_map

                dropped_map_file = bundle_dir / "bundles_multibfs_regroup_dropped_map.html"

                viz_dropped = dropped_bundles.dropna(subset=["bundle_id"]).to_crs(4326)
                tooltip_cols = [seg_id, "bundle_id", "bundle_seg_count", "bundle_addr_total"]

                m_dropped = folium_map(viz_dropped, color_col="bundle_id", tooltip_cols=tooltip_cols)
                m_dropped.save(str(dropped_map_file))
                print(f"   ✓ 筛掉的地图已保存: {dropped_map_file}")
                print(f"   📊 筛掉的 bundles: {len(dropped_bundle_ids)} bundles, {len(dropped_bundles)} segments")
            else:
                print(f"   ℹ️  没有被筛掉的 bundles")
                dropped_map_file = None
        except Exception as e:
            print(f"   ⚠️  筛掉的地图生成失败: {e}")
            dropped_map_file = None

        # 分析空间代表性
        analyze_spatial_representativeness(bundles, filtered_bundles)

        # 最终报告
        print("="*70)
        print("✅ 过滤完成!")
        print("="*70)
        print(f"\n📁 输出文件:")
        print(f"  - 过滤后数据: {filtered_file}")
        print(f"  - 保留的地图: {map_file}")
        if dropped_map_file:
            print(f"  - 筛掉的地图: {dropped_map_file}")
        print()

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    filter_bundles()
