#!/usr/bin/env python
"""
Test script to compare greedy vs multi_bfs bundling algorithms.

Usage:
    python tests/test_bundle_comparison.py --session DH --target_addrs 120

This script:
1. Runs both bundling algorithms on the same input
2. Compares bundle size distributions (CV, MASD)
3. Generates comparison reports and visualizations
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure src is importable
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.bundle import _build_connected_bundles
from sd311_fieldprep.utils import load_sources, paths, project_to


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
        "singleton_count": int((bundle_stats["bundle_seg_count"] == 1).sum()),
        "singleton_share": float((bundle_stats["bundle_seg_count"] == 1).mean()),
    }

    # Mean Absolute Standardized Difference
    target = np.mean(sizes)
    masd = float(np.mean(np.abs(sizes - target) / target))
    metrics["masd"] = masd

    return metrics, bundle_stats


def compare_methods(segs_m, seg_id, target_addrs, join_tol_m=15.0, seed=42, min_bundle_sfh=None):
    """Run both bundling methods and compare results."""
    print("\n" + "="*60)
    print("BUNDLE ALGORITHM COMPARISON")
    print("="*60)

    results = {}

    for method in ["greedy", "multi_bfs"]:
        print(f"\n>>> Running {method} algorithm...")
        start_time = datetime.now()

        bundled = _build_connected_bundles(
            segs_m,
            seg_id_col=seg_id,
            target_addrs=target_addrs,
            join_tol_m=join_tol_m,
            seed=seed,
            min_bundle_sfh=min_bundle_sfh,
            method=method
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"    Completed in {elapsed:.2f} seconds")

        metrics, bundle_stats = calculate_bundle_metrics(bundled)
        metrics["runtime_sec"] = elapsed

        results[method] = {
            "bundled": bundled,
            "metrics": metrics,
            "bundle_stats": bundle_stats
        }

    return results


def print_comparison_report(results):
    """Print comparison report to console."""
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)

    metrics_order = [
        "n_bundles", "mean_size", "median_size", "std_size", "cv", "masd",
        "min_size", "max_size", "range_ratio", "singleton_count", "singleton_share",
        "runtime_sec"
    ]

    print(f"\n{'Metric':<25} {'Greedy':>15} {'Multi-BFS':>15} {'Improvement':>15}")
    print("-" * 72)

    for metric in metrics_order:
        greedy_val = results["greedy"]["metrics"][metric]
        multi_val = results["multi_bfs"]["metrics"][metric]

        # Calculate improvement (negative = better for CV, MASD, range_ratio, singleton_share)
        if metric in ["cv", "masd", "range_ratio", "singleton_share", "std_size", "runtime_sec"]:
            improvement = ((greedy_val - multi_val) / greedy_val * 100) if greedy_val != 0 else 0
            improvement_str = f"{improvement:+.1f}%"
        elif metric in ["n_bundles", "singleton_count"]:
            improvement_str = f"{multi_val - greedy_val:+.0f}"
        else:
            improvement_str = "-"

        # Format values
        if metric == "runtime_sec":
            greedy_str = f"{greedy_val:.2f}s"
            multi_str = f"{multi_val:.2f}s"
        elif metric in ["cv", "masd", "singleton_share"]:
            greedy_str = f"{greedy_val:.4f}"
            multi_str = f"{multi_val:.4f}"
        elif metric == "range_ratio":
            greedy_str = f"{greedy_val:.2f}x"
            multi_str = f"{multi_val:.2f}x"
        elif isinstance(greedy_val, float):
            greedy_str = f"{greedy_val:.1f}"
            multi_str = f"{multi_val:.1f}"
        else:
            greedy_str = f"{greedy_val}"
            multi_str = f"{multi_val}"

        print(f"{metric:<25} {greedy_str:>15} {multi_str:>15} {improvement_str:>15}")

    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)

    cv_improvement = ((results["greedy"]["metrics"]["cv"] - results["multi_bfs"]["metrics"]["cv"])
                      / results["greedy"]["metrics"]["cv"] * 100)
    masd_improvement = ((results["greedy"]["metrics"]["masd"] - results["multi_bfs"]["metrics"]["masd"])
                        / results["greedy"]["metrics"]["masd"] * 100)

    print(f"• CV improved by {cv_improvement:.1f}% (lower is better)")
    print(f"• MASD improved by {masd_improvement:.1f}% (lower is better)")
    print(f"• Range ratio: {results['greedy']['metrics']['range_ratio']:.2f}x → "
          f"{results['multi_bfs']['metrics']['range_ratio']:.2f}x")

    if cv_improvement > 30:
        print("\n✓ Multi-BFS shows SIGNIFICANT improvement in balance!")
    elif cv_improvement > 15:
        print("\n✓ Multi-BFS shows moderate improvement in balance.")
    else:
        print("\n⚠ Multi-BFS shows minimal improvement.")


def plot_comparison(results, output_path=None):
    """Generate comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, method in enumerate(["greedy", "multi_bfs"]):
        ax = axes[idx]
        sizes = results[method]["bundle_stats"]["total_addrs"].values

        ax.hist(sizes, bins=20, alpha=0.7, color=['blue', 'green'][idx], edgecolor='black')
        ax.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sizes):.1f}')
        ax.axvline(np.median(sizes), color='orange', linestyle=':', linewidth=2, label=f'Median: {np.median(sizes):.1f}')

        ax.set_xlabel('Bundle Size (addresses)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{method.upper()}\nCV={results[method]["metrics"]["cv"]:.3f}, '
                     f'MASD={results[method]["metrics"]["masd"]:.3f}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def save_detailed_report(results, output_path):
    """Save detailed bundle statistics to CSV."""
    report_data = []

    for method in ["greedy", "multi_bfs"]:
        stats = results[method]["bundle_stats"].copy()
        stats["method"] = method
        stats["bundle_id_orig"] = stats.index
        report_data.append(stats)

    combined = pd.concat(report_data)
    combined.to_csv(output_path, index=False)
    print(f"✓ Detailed report saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare greedy vs multi_bfs bundling algorithms")
    parser.add_argument("--session", choices=["DH", "D2DS"], default="DH",
                        help="Session type (default: DH)")
    parser.add_argument("--target_addrs", type=int, default=120,
                        help="Target addresses per bundle (default: 120)")
    parser.add_argument("--join_tol_m", type=float, default=15.0,
                        help="Join tolerance in meters (default: 15.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--tag", default=None,
                        help="Sweep tag to use (default: most recent)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for reports (default: outputs/tests/)")

    args = parser.parse_args()

    # Load data
    print("\n[1/4] Loading data...")
    streets, addrs, ds, pr = load_sources()
    seg_id = pr["fields"]["streets_segment_id"]

    root, cfg, out_root = paths()
    sweep_root = out_root / "sweep"
    if args.tag:
        sweep_root = sweep_root / args.tag

    cands = sorted((p for p in sweep_root.rglob("eligible_*.parquet")),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("No eligible_*.parquet found. Run sweep first.")

    latest = cands[-1]
    segs = gpd.read_parquet(latest)
    print(f"   Using: {latest}")

    # Prepare data
    print("\n[2/4] Preparing data...")
    work_epsg = int(pr.get("crs", {}).get("working_meters", 26911))
    segs_m = project_to(segs, work_epsg)

    # Run comparison
    print("\n[3/4] Running comparison...")
    results = compare_methods(segs_m, seg_id, args.target_addrs, args.join_tol_m, args.seed)

    # Generate reports
    print("\n[4/4] Generating reports...")
    print_comparison_report(results)

    # Save outputs
    output_dir = Path(args.output_dir) if args.output_dir else (root / "outputs" / "tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"bundle_comparison_{timestamp}.png"
    csv_path = output_dir / f"bundle_comparison_{timestamp}.csv"

    plot_comparison(results, plot_path)
    save_detailed_report(results, csv_path)

    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
