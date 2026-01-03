#!/usr/bin/env python
"""Test the new Eulerian validation + regrouping function"""
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.bundle_hard_constraint import (
    _validate_and_regroup_eulerian,
    _build_segment_neighbor_graph
)

# Load existing bundle file
print("Loading existing bundle file...")
bundles = gpd.read_parquet("outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
print(f"Loaded {len(bundles)} segments, {bundles['bundle_id'].nunique()} bundles\n")

# Test validation + regrouping on a subset that includes Bundle 3480
test_bundle_ids = [3480, 2973, 4870, 2980, 5077, 3040, 2647, 2270]
print(f"Testing validation + regrouping on bundles: {test_bundle_ids}\n")

test_bundles = bundles[bundles['bundle_id'].isin(test_bundle_ids)].copy()
print(f"Test set: {len(test_bundles)} segments, {test_bundles['bundle_id'].nunique()} bundles")
print(f"Total addresses: {test_bundles['sfh_addr_count'].sum():.0f}\n")

# Build neighbor graph
print("Building segment neighbor graph...")
nbrs, keep_idx, _ = _build_segment_neighbor_graph(test_bundles, join_tol_m=15.0)
print(f"Graph: {len(nbrs)} segments\n")

# Run validation + regrouping
print("="*70)
print("RUNNING EULERIAN VALIDATION + REGROUPING")
print("="*70)

validated = _validate_and_regroup_eulerian(
    test_bundles,
    nbrs=nbrs,
    target_addrs=80,
    min_size=72,
    max_size=88,
    seed=42,
    snap_tol=0.5
)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

original_bundles = len(test_bundle_ids)
final_bundles = validated['bundle_id'].dropna().nunique()
unassigned_segments = validated['bundle_id'].isna().sum()
unassigned_addrs = validated.loc[validated['bundle_id'].isna(), 'sfh_addr_count'].sum()

print(f"\nOriginal bundles: {original_bundles}")
print(f"Final bundles: {final_bundles}")
print(f"Unassigned segments: {unassigned_segments}")
print(f"Unassigned addresses: {unassigned_addrs:.0f}")

if final_bundles >= original_bundles - 1:
    print(f"\n✅ SUCCESS: Bundle 3480 was successfully regrouped!")
    print(f"   Most segments were reassigned to valid bundles")
else:
    print(f"\n⚠️  Some bundles were dissolved")

# Show bundle size distribution
print("\nFinal bundle sizes:")
bundle_sizes = validated.groupby('bundle_id')['sfh_addr_count'].sum().sort_values()
for bid, size in bundle_sizes.items():
    n_segs = len(validated[validated['bundle_id'] == bid])
    print(f"  Bundle {int(bid)}: {n_segs} segments, {int(size)} addresses")
