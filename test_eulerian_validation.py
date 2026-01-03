#!/usr/bin/env python
"""Test the new Eulerian validation function on existing bundles"""
import sys
from pathlib import Path
import geopandas as gpd

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.bundle_hard_constraint import _validate_and_filter_eulerian

# Load existing bundle file
print("Loading existing bundle file...")
bundles = gpd.read_parquet("outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
print(f"Loaded {len(bundles)} segments, {bundles['bundle_id'].nunique()} bundles\n")

# Test validation on a subset that includes Bundle 3480
test_bundle_ids = [3480, 2973, 4870, 2980, 5077]
print(f"Testing validation on bundles: {test_bundle_ids}\n")

test_bundles = bundles[bundles['bundle_id'].isin(test_bundle_ids)].copy()
print(f"Test set: {len(test_bundles)} segments, {test_bundles['bundle_id'].nunique()} bundles\n")

# Run validation
print("="*70)
print("RUNNING EULERIAN VALIDATION")
print("="*70)

validated = _validate_and_filter_eulerian(test_bundles, snap_tol=0.5)

print("\n" + "="*70)
print("VALIDATION RESULTS")
print("="*70)

remaining_bundles = validated['bundle_id'].dropna().nunique()
filtered_segments = validated['bundle_id'].isna().sum()

print(f"\nRemaining bundles: {remaining_bundles}")
print(f"Filtered segments: {filtered_segments}")

if filtered_segments > 0:
    print(f"\n✅ SUCCESS: Validation correctly identified and filtered problematic bundles!")
    print(f"   (Expected: Bundle 3480 should be filtered)")
else:
    print(f"\n⚠️  No bundles were filtered (unexpected if Bundle 3480 is in the test set)")

print("\nRemaining bundle IDs:")
print(sorted(validated['bundle_id'].dropna().unique()))
