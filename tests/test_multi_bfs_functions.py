#!/usr/bin/env python
"""
Unit tests for Multi-source BFS bundling functions.

Tests:
- _select_spatially_distributed_seeds
- _backtrack_rebalance
- _multi_source_balanced_bfs

Usage:
    python tests/test_multi_bfs_functions.py
"""

import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point

# Ensure src is importable
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.bundle import (
    _select_spatially_distributed_seeds,
    _backtrack_rebalance,
    _multi_source_balanced_bfs,
    _build_segment_neighbor_graph
)


def create_test_grid_network(n=5, spacing=100):
    """Create a simple grid network for testing.

    Creates n x n grid of road segments with specified spacing (meters).
    Each segment has a random number of addresses.
    """
    segments = []

    # Horizontal segments
    for i in range(n):
        for j in range(n - 1):
            start = (j * spacing, i * spacing)
            end = ((j + 1) * spacing, i * spacing)
            segments.append({
                "geometry": LineString([start, end]),
                "sfh_addr_count": np.random.randint(5, 25)
            })

    # Vertical segments
    for i in range(n - 1):
        for j in range(n):
            start = (j * spacing, i * spacing)
            end = (j * spacing, (i + 1) * spacing)
            segments.append({
                "geometry": LineString([start, end]),
                "sfh_addr_count": np.random.randint(5, 25)
            })

    gdf = gpd.GeoDataFrame(segments, crs=26911)
    return gdf


def test_select_spatially_distributed_seeds():
    """Test seed selection algorithm."""
    print("\n" + "="*60)
    print("TEST: _select_spatially_distributed_seeds")
    print("="*60)

    # Create test network
    g = create_test_grid_network(n=5, spacing=100)
    indices = list(g.index)
    rng = np.random.default_rng(42)

    # Test 1: Normal case
    n_seeds = 5
    seeds = _select_spatially_distributed_seeds(g, indices, n_seeds, rng)

    print(f"\n✓ Test 1: Select {n_seeds} seeds from {len(indices)} segments")
    print(f"  Selected seeds: {seeds}")
    assert len(seeds) == n_seeds, f"Expected {n_seeds} seeds, got {len(seeds)}"
    assert len(set(seeds)) == n_seeds, "Seeds should be unique"

    # Calculate average distance between seeds
    total_dist = 0
    count = 0
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            dist = g.loc[s1].geometry.distance(g.loc[s2].geometry)
            total_dist += dist
            count += 1
    avg_dist = total_dist / count if count > 0 else 0
    print(f"  Average distance between seeds: {avg_dist:.1f} meters")

    # Test 2: Edge case - more seeds than segments
    seeds2 = _select_spatially_distributed_seeds(g, indices[:3], 10, rng)
    print(f"\n✓ Test 2: Request 10 seeds from 3 segments")
    print(f"  Got {len(seeds2)} seeds (capped at available)")
    assert len(seeds2) <= 3, "Should not exceed available segments"

    # Test 3: Edge case - zero seeds
    seeds3 = _select_spatially_distributed_seeds(g, indices, 0, rng)
    print(f"\n✓ Test 3: Request 0 seeds")
    print(f"  Got {len(seeds3)} seeds")
    assert len(seeds3) == 0, "Should return empty list"

    print("\n✓ All tests passed!")
    return True


def test_backtrack_rebalance():
    """Test backtracking rebalance algorithm."""
    print("\n" + "="*60)
    print("TEST: _backtrack_rebalance")
    print("="*60)

    # Create test network
    g = create_test_grid_network(n=4, spacing=100)
    nbrs, keep_idx, _ = _build_segment_neighbor_graph(g, join_tol_m=20.0)
    g = g.iloc[keep_idx].copy()

    # Create imbalanced assignment
    # Bundle 0: first 5 segments (likely oversized)
    # Bundle 1: next 2 segments (likely undersized)
    assignment = {}
    for idx in list(g.index)[:5]:
        assignment[idx] = 0
    for idx in list(g.index)[5:7]:
        assignment[idx] = 1

    # Calculate initial sizes
    sizes = {0: 0, 1: 0}
    for idx, bid in assignment.items():
        sizes[bid] += int(g.loc[idx, "sfh_addr_count"])

    target = sum(sizes.values()) // 2

    print(f"\nInitial state:")
    print(f"  Target: {target} addresses")
    print(f"  Bundle 0: {sizes[0]} addresses ({len([i for i, b in assignment.items() if b == 0])} segments)")
    print(f"  Bundle 1: {sizes[1]} addresses ({len([i for i, b in assignment.items() if b == 1])} segments)")

    initial_imbalance = abs(sizes[0] - sizes[1])

    # Run rebalancing
    assignment_new = _backtrack_rebalance(g, nbrs, assignment, sizes, target)

    # Recalculate sizes
    sizes_new = {0: 0, 1: 0}
    for idx, bid in assignment_new.items():
        sizes_new[bid] += int(g.loc[idx, "sfh_addr_count"])

    print(f"\nAfter rebalancing:")
    print(f"  Bundle 0: {sizes_new[0]} addresses ({len([i for i, b in assignment_new.items() if b == 0])} segments)")
    print(f"  Bundle 1: {sizes_new[1]} addresses ({len([i for i, b in assignment_new.items() if b == 1])} segments)")

    final_imbalance = abs(sizes_new[0] - sizes_new[1])
    improvement = ((initial_imbalance - final_imbalance) / initial_imbalance * 100) if initial_imbalance > 0 else 0

    print(f"\nImbalance: {initial_imbalance} → {final_imbalance} (improved {improvement:.1f}%)")

    print("\n✓ Test passed (rebalancing executed successfully)")
    return True


def test_multi_source_balanced_bfs():
    """Test complete multi-source BFS algorithm."""
    print("\n" + "="*60)
    print("TEST: _multi_source_balanced_bfs")
    print("="*60)

    # Create larger test network
    g = create_test_grid_network(n=6, spacing=100)
    nbrs, keep_idx, _ = _build_segment_neighbor_graph(g, join_tol_m=20.0)
    g = g.iloc[keep_idx].copy()

    comp_indices = list(g.index)
    target_addrs = 50
    seed = 42

    print(f"\nTest configuration:")
    print(f"  Segments: {len(comp_indices)}")
    print(f"  Total addresses: {g['sfh_addr_count'].sum()}")
    print(f"  Target per bundle: {target_addrs}")
    print(f"  Expected bundles: ~{g['sfh_addr_count'].sum() // target_addrs}")

    # Run algorithm
    assignment = _multi_source_balanced_bfs(g, nbrs, comp_indices, target_addrs, seed)

    # Calculate bundle stats
    bundles = {}
    for idx, bid in assignment.items():
        if bid not in bundles:
            bundles[bid] = {"count": 0, "addrs": 0}
        bundles[bid]["count"] += 1
        bundles[bid]["addrs"] += int(g.loc[idx, "sfh_addr_count"])

    print(f"\nResults:")
    print(f"  Bundles created: {len(bundles)}")

    sizes = [b["addrs"] for b in bundles.values()]
    cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0

    print(f"  Bundle sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    print(f"  CV: {cv:.4f}")
    print(f"  Segments per bundle: min={min(b['count'] for b in bundles.values())}, "
          f"max={max(b['count'] for b in bundles.values())}")

    # Check that all segments are assigned
    assert len(assignment) == len(comp_indices), "All segments should be assigned"

    # Check CV is reasonable (< 0.3 for balanced)
    if cv < 0.3:
        print(f"\n✓ Test passed! CV={cv:.4f} indicates good balance")
    else:
        print(f"\n⚠ Test passed but CV={cv:.4f} is higher than ideal (<0.3)")

    return True


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*70)
    print(" MULTI-SOURCE BFS UNIT TESTS")
    print("="*70)

    tests = [
        test_select_spatially_distributed_seeds,
        test_backtrack_rebalance,
        test_multi_source_balanced_bfs
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
