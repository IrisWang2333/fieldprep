#!/usr/bin/env python
"""
🔒 硬约束版本 Bundle 分析工具

Analyze bundle balance from bundles_hard_constraint.parquet files.

Usage:
    python tests/analyze_bundle_balance_hard_constraint.py outputs/bundles/DH/bundles_hard_constraint.parquet
    python tests/analyze_bundle_balance_hard_constraint.py outputs/bundles/DH/bundles_hard_constraint.parquet --plot
    python tests/analyze_bundle_balance_hard_constraint.py outputs/bundles/DH/bundles_hard_constraint.parquet --target 120 --multiplier 1.2
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def calculate_balance_metrics(bundles_df, target_addrs=None, hard_max_multiplier=None):
    """Calculate comprehensive balance metrics for bundles with hard constraint analysis."""
    # Group by bundle_id
    bundle_stats = bundles_df.groupby("bundle_id").agg({
        "sfh_addr_count": ["sum", "count"],
    })
    bundle_stats.columns = ["total_addrs", "seg_count"]

    sizes = bundle_stats["total_addrs"].values
    seg_counts = bundle_stats["seg_count"].values

    # Size distribution metrics
    mean_size = np.mean(sizes)
    median_size = np.median(sizes)
    std_size = np.std(sizes)
    cv = std_size / mean_size if mean_size > 0 else 0

    # Mean Absolute Standardized Difference (MASD)
    masd = np.mean(np.abs(sizes - mean_size) / mean_size) if mean_size > 0 else 0

    # Range metrics
    min_size = np.min(sizes)
    max_size = np.max(sizes)
    range_ratio = max_size / min_size if min_size > 0 else np.inf

    # Percentiles
    p25 = np.percentile(sizes, 25)
    p75 = np.percentile(sizes, 75)
    iqr = p75 - p25

    # Segment count metrics
    singleton_count = np.sum(seg_counts == 1)
    singleton_share = singleton_count / len(bundle_stats)

    # Outlier detection (1.5 * IQR rule)
    lower_bound = p25 - 1.5 * iqr
    upper_bound = p75 + 1.5 * iqr
    outlier_count = np.sum((sizes < lower_bound) | (sizes > upper_bound))
    outlier_share = outlier_count / len(bundle_stats)

    metrics = {
        "n_bundles": len(bundle_stats),
        "total_addresses": np.sum(sizes),
        "mean_size": mean_size,
        "median_size": median_size,
        "std_size": std_size,
        "cv": cv,
        "masd": masd,
        "min_size": min_size,
        "max_size": max_size,
        "range": max_size - min_size,
        "range_ratio": range_ratio,
        "p25": p25,
        "p75": p75,
        "iqr": iqr,
        "singleton_count": singleton_count,
        "singleton_share": singleton_share,
        "outlier_count": outlier_count,
        "outlier_share": outlier_share,
        "mean_seg_count": np.mean(seg_counts),
        "median_seg_count": np.median(seg_counts),
    }

    # 🔒 Hard constraint specific metrics
    if target_addrs is not None and hard_max_multiplier is not None:
        hard_max = int(target_addrs * hard_max_multiplier)
        violations = np.sum(sizes > hard_max)
        violations_share = violations / len(bundle_stats)

        metrics["target_addrs"] = target_addrs
        metrics["hard_max_multiplier"] = hard_max_multiplier
        metrics["hard_max"] = hard_max
        metrics["violations"] = violations
        metrics["violations_share"] = violations_share
        metrics["max_deviation"] = max_size - hard_max if max_size > hard_max else 0
        metrics["max_deviation_pct"] = ((max_size - hard_max) / hard_max * 100) if max_size > hard_max else 0

    return metrics, bundle_stats


def print_balance_report(metrics):
    """Print comprehensive balance report with hard constraint analysis."""
    print("\n" + "="*70)
    print(" 🔒 HARD CONSTRAINT BUNDLE BALANCE ANALYSIS REPORT")
    print("="*70)

    # 🔒 Hard constraint section (if available)
    if "hard_max" in metrics:
        print("\n🔒 HARD CONSTRAINT STATUS:")
        print("-" * 70)
        print(f"  Target Size:          {metrics['target_addrs']:>10} addresses")
        print(f"  Hard Max Multiplier:  {metrics['hard_max_multiplier']:>10.1f}x")
        print(f"  Hard Max Limit:       {metrics['hard_max']:>10} addresses")
        print(f"  Actual Max Size:      {metrics['max_size']:>10.0f} addresses")

        if metrics['violations'] == 0:
            print(f"  Constraint Violations:{' ':>9}✅ NONE (Perfect!)")
        else:
            print(f"  Constraint Violations:{metrics['violations']:>10} bundles "
                  f"({metrics['violations_share']*100:.1f}%)")
            print(f"  Max Deviation:        {metrics['max_deviation']:>10.0f} addresses "
                  f"({metrics['max_deviation_pct']:+.1f}%)")

    print("\n📊 OVERALL STATISTICS:")
    print("-" * 70)
    print(f"  Total Bundles:        {metrics['n_bundles']:>10}")
    print(f"  Total Addresses:      {metrics['total_addresses']:>10.0f}")
    print(f"  Mean Bundle Size:     {metrics['mean_size']:>10.1f} addresses")
    print(f"  Median Bundle Size:   {metrics['median_size']:>10.1f} addresses")

    print("\n📏 SIZE DISTRIBUTION:")
    print("-" * 70)
    print(f"  Standard Deviation:   {metrics['std_size']:>10.1f}")
    print(f"  Coefficient of Variation (CV):  {metrics['cv']:>10.4f}")
    print(f"  Mean Abs. Std. Diff (MASD):     {metrics['masd']:>10.4f}")
    print(f"  Min Size:             {metrics['min_size']:>10.0f} addresses")
    print(f"  25th Percentile:      {metrics['p25']:>10.1f} addresses")
    print(f"  75th Percentile:      {metrics['p75']:>10.1f} addresses")
    print(f"  Max Size:             {metrics['max_size']:>10.0f} addresses")
    print(f"  Range:                {metrics['range']:>10.0f} addresses")
    print(f"  Range Ratio (max/min):{metrics['range_ratio']:>10.2f}x")

    print("\n🎯 QUALITY METRICS:")
    print("-" * 70)

    # CV assessment
    if metrics['cv'] < 0.10:
        cv_rating = "Excellent"
        cv_emoji = "🌟"
    elif metrics['cv'] < 0.15:
        cv_rating = "Very Good"
        cv_emoji = "✅"
    elif metrics['cv'] < 0.20:
        cv_rating = "Good"
        cv_emoji = "👍"
    elif metrics['cv'] < 0.30:
        cv_rating = "Fair"
        cv_emoji = "⚠️"
    else:
        cv_rating = "Poor"
        cv_emoji = "❌"

    print(f"  Balance Quality:      {cv_emoji} {cv_rating} (CV={metrics['cv']:.4f})")

    # MASD assessment
    if metrics['masd'] < 0.10:
        masd_rating = "Excellent"
    elif metrics['masd'] < 0.15:
        masd_rating = "Very Good"
    elif metrics['masd'] < 0.20:
        masd_rating = "Good"
    elif metrics['masd'] < 0.30:
        masd_rating = "Fair"
    else:
        masd_rating = "Poor"

    print(f"  Uniformity:           {masd_rating} (MASD={metrics['masd']:.4f})")

    # Singleton assessment
    if metrics['singleton_share'] < 0.02:
        singleton_rating = "Excellent"
    elif metrics['singleton_share'] < 0.05:
        singleton_rating = "Good"
    elif metrics['singleton_share'] < 0.10:
        singleton_rating = "Fair"
    else:
        singleton_rating = "Poor"

    print(f"  Singleton Bundles:    {metrics['singleton_count']} "
          f"({metrics['singleton_share']*100:.1f}%) - {singleton_rating}")
    print(f"  Outlier Bundles:      {metrics['outlier_count']} "
          f"({metrics['outlier_share']*100:.1f}%)")

    print("\n📦 SEGMENT STATISTICS:")
    print("-" * 70)
    print(f"  Mean Segments/Bundle: {metrics['mean_seg_count']:>10.1f}")
    print(f"  Median Segments/Bundle:{metrics['median_seg_count']:>10.0f}")

    print("\n💡 RECOMMENDATIONS:")
    print("-" * 70)

    # Hard constraint specific recommendations
    if "hard_max" in metrics:
        if metrics['violations'] == 0:
            print("  ✅ Hard constraint is working perfectly - no violations!")
        else:
            print(f"  ⚠️  {metrics['violations']} bundles exceed hard max limit.")
            print(f"     Consider increasing hard_max_multiplier or reviewing merge logic.")

        if metrics['singleton_share'] > 0.05:
            print(f"  💡 {metrics['singleton_count']} singleton bundles remain.")
            print(f"     This may be due to hard constraint preventing merges.")

    if metrics['cv'] > 0.25:
        print("  ⚠️  High CV detected. Consider adjusting target_addrs or algorithm.")
    if metrics['singleton_share'] > 0.10:
        print("  ⚠️  High singleton rate. Consider using --min_bundle_sfh parameter.")
    if metrics['range_ratio'] > 3.0:
        print("  ⚠️  Large size variance. Review settings.")
    if metrics['cv'] < 0.15 and metrics['singleton_share'] < 0.05:
        print("  ✅ Bundle distribution looks excellent!")

    print("\n" + "="*70 + "\n")


def plot_distribution(bundle_stats, output_path=None, hard_max=None):
    """Plot bundle size distribution with hard constraint visualization."""
    sizes = bundle_stats["total_addrs"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("🔒 Hard Constraint Bundle Analysis", fontsize=14, fontweight='bold')

    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(sizes, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(sizes):.1f}')
    ax.axvline(np.median(sizes), color='orange', linestyle=':', linewidth=2,
               label=f'Median: {np.median(sizes):.1f}')

    # 🔒 Add hard max line
    if hard_max is not None:
        ax.axvline(hard_max, color='darkred', linestyle='-', linewidth=2.5,
                   label=f'Hard Max: {hard_max}', alpha=0.8)

    ax.set_xlabel('Bundle Size (addresses)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Bundle Size Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Box plot
    ax = axes[0, 1]
    box = ax.boxplot(sizes, vert=True, patch_artist=True)
    box['boxes'][0].set_facecolor('lightblue')

    # 🔒 Add hard max line
    if hard_max is not None:
        ax.axhline(hard_max, color='darkred', linestyle='-', linewidth=2.5,
                   label=f'Hard Max: {hard_max}', alpha=0.8)
        ax.legend()

    ax.set_ylabel('Bundle Size (addresses)', fontsize=11)
    ax.set_title('Bundle Size Box Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Sorted sizes
    ax = axes[1, 0]
    sorted_sizes = np.sort(sizes)
    ax.plot(sorted_sizes, marker='o', markersize=3, linestyle='-', color='steelblue')
    ax.axhline(np.mean(sizes), color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Mean: {np.mean(sizes):.1f}')

    # 🔒 Add hard max line
    if hard_max is not None:
        ax.axhline(hard_max, color='darkred', linestyle='-', linewidth=2.5,
                   label=f'Hard Max: {hard_max}', alpha=0.8)

    ax.set_xlabel('Bundle Index (sorted)', fontsize=11)
    ax.set_ylabel('Bundle Size (addresses)', fontsize=11)
    ax.set_title('Sorted Bundle Sizes', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Segments per bundle
    ax = axes[1, 1]
    seg_counts = bundle_stats["seg_count"].values
    ax.hist(seg_counts, bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(np.mean(seg_counts), color='darkred', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(seg_counts):.1f}')
    ax.set_xlabel('Segments per Bundle', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Segment Count Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="🔒 Analyze hard constraint bundle balance from bundles_hard_constraint.parquet"
    )
    parser.add_argument("bundle_file", nargs='?',
                        default="outputs/bundles/DH/bundles_hard_constraint.parquet",
                        help="Path to bundles_hard_constraint.parquet file (default: outputs/bundles/DH/bundles_hard_constraint.parquet)")
    parser.add_argument("--plot", action="store_true", help="Generate distribution plot")
    parser.add_argument("--output", help="Output path for plot (PNG)")
    parser.add_argument("--target", type=int, help="Target addresses per bundle (for constraint analysis)")
    parser.add_argument("--multiplier", type=float, default=1.2,
                        help="Hard max multiplier (default: 1.2)")

    args = parser.parse_args()

    bundle_path = Path(args.bundle_file)
    if not bundle_path.exists():
        print(f"❌ Error: File not found: {bundle_path}")
        print(f"\nSearching for bundles_hard_constraint.parquet files...")

        # Try to find the file
        search_paths = [
            Path("outputs/bundles/DH/bundles_hard_constraint.parquet"),
            Path("outputs/bundles/D2DS/bundles_hard_constraint.parquet"),
            Path("../outputs/bundles/DH/bundles_hard_constraint.parquet"),
        ]

        for p in search_paths:
            if p.exists():
                print(f"✓ Found: {p}")
                bundle_path = p
                break
        else:
            print("\n💡 Tip: Run quick_comparison_hard_constraint.py first to generate the file.")
            sys.exit(1)

    print(f"\n🔒 Loading hard constraint bundles from: {bundle_path}")
    bundles = gpd.read_parquet(bundle_path)

    print(f"  Total segments: {len(bundles)}")

    # Calculate metrics
    metrics, bundle_stats = calculate_balance_metrics(
        bundles,
        target_addrs=args.target,
        hard_max_multiplier=args.multiplier
    )

    # Print report
    print_balance_report(metrics)

    # Generate plot if requested
    if args.plot:
        output_path = args.output if args.output else bundle_path.parent / "bundle_analysis_hard_constraint.png"
        hard_max = metrics.get("hard_max", None)
        plot_distribution(bundle_stats, output_path, hard_max=hard_max)


if __name__ == "__main__":
    main()
