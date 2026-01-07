#!/usr/bin/env python
"""
Optimized Bundle Assignment Using Local Search

Improves upon greedy minimax through iterative bundle swapping.
Guarantees a solution at least as good as greedy, potentially better.

Algorithm:
1. Start with greedy minimax solution
2. Try swapping bundles between all pairs of interviewers
3. Accept swaps that reduce maximum travel time
4. Repeat until no improvement found
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

from sd311_fieldprep.assign_bundles_minimax import (
    assign_bundles_minimax,
    calculate_travel_time,
    get_bundle_centroid,
    calculate_bundle_internal_distance,
    load_interviewer_data,
    get_interviewers_for_date
)


def get_max_travel_time(assignments: Dict[str, Tuple[List[int], float]]) -> float:
    """Get maximum travel time across all interviewers."""
    return max(time for _, time in assignments.values())


def try_swap_bundles(
    interviewer1: Dict,
    interviewer2: Dict,
    bundles1: List[int],
    bundles2: List[int],
    bundle_info: Dict,
    current_max_time: float
) -> Tuple[bool, List[int], List[int], float]:
    """
    Try swapping one bundle between two interviewers.

    Returns:
        (improved, new_bundles1, new_bundles2, new_max_time)
    """
    best_improvement = False
    best_bundles1 = bundles1
    best_bundles2 = bundles2
    best_max_time = current_max_time

    # Try swapping each bundle from interviewer1 with each bundle from interviewer2
    for i, bundle1 in enumerate(bundles1):
        for j, bundle2 in enumerate(bundles2):
            # Create new bundle assignments
            new_bundles1 = bundles1[:i] + [bundle2] + bundles1[i+1:]
            new_bundles2 = bundles2[:j] + [bundle1] + bundles2[j+1:]

            # Calculate new travel times
            _, time1 = calculate_travel_time(
                interviewer1['lat'], interviewer1['lon'],
                new_bundles1, bundle_info
            )
            _, time2 = calculate_travel_time(
                interviewer2['lat'], interviewer2['lon'],
                new_bundles2, bundle_info
            )

            # Check if this swap improves the maximum time
            new_max_time = max(time1, time2)

            if new_max_time < best_max_time:
                best_improvement = True
                best_bundles1 = new_bundles1
                best_bundles2 = new_bundles2
                best_max_time = new_max_time

    return best_improvement, best_bundles1, best_bundles2, best_max_time


def optimize_with_local_search(
    interviewers: List[Dict],
    initial_assignments: Dict[str, Tuple[List[int], float]],
    bundle_info: Dict,
    max_iterations: int = 100
) -> Dict[str, Tuple[List[int], float]]:
    """
    Improve assignments through local search (bundle swapping).

    Args:
        interviewers: List of interviewer dicts
        initial_assignments: Starting assignments (from greedy minimax)
        bundle_info: Bundle information dict
        max_iterations: Maximum number of improvement iterations

    Returns:
        Improved assignments
    """
    # Create interviewer lookup
    interviewer_dict = {i['name']: i for i in interviewers}

    # Start with initial assignments
    current_assignments = {
        name: list(bundles) for name, (bundles, _) in initial_assignments.items()
    }

    current_max_time = get_max_travel_time(initial_assignments)
    initial_max_time = current_max_time

    print(f"[Local Search] Initial maximum time: {current_max_time:.2f} km")

    iteration = 0
    improvements = 0

    while iteration < max_iterations:
        iteration += 1
        improved = False

        # Try swapping bundles between all pairs of interviewers
        interviewer_names = list(current_assignments.keys())

        for i in range(len(interviewer_names)):
            for j in range(i + 1, len(interviewer_names)):
                name1 = interviewer_names[i]
                name2 = interviewer_names[j]

                interviewer1 = interviewer_dict[name1]
                interviewer2 = interviewer_dict[name2]

                bundles1 = current_assignments[name1]
                bundles2 = current_assignments[name2]

                # Try swapping
                swap_improved, new_bundles1, new_bundles2, new_local_max = try_swap_bundles(
                    interviewer1, interviewer2,
                    bundles1, bundles2,
                    bundle_info,
                    current_max_time
                )

                if swap_improved:
                    # Recalculate global maximum time with new assignments
                    temp_assignments = current_assignments.copy()
                    temp_assignments[name1] = new_bundles1
                    temp_assignments[name2] = new_bundles2

                    # Calculate all travel times
                    all_times = []
                    for name, bundles in temp_assignments.items():
                        interviewer = interviewer_dict[name]
                        _, time = calculate_travel_time(
                            interviewer['lat'], interviewer['lon'],
                            bundles, bundle_info
                        )
                        all_times.append(time)

                    new_global_max = max(all_times)

                    if new_global_max < current_max_time:
                        # Accept the swap
                        current_assignments[name1] = new_bundles1
                        current_assignments[name2] = new_bundles2
                        current_max_time = new_global_max
                        improvements += 1
                        improved = True

                        print(f"[Local Search] Iteration {iteration}: Found improvement! "
                              f"New max time: {current_max_time:.2f} km "
                              f"(swapped bundles between {name1} and {name2})")

        if not improved:
            print(f"[Local Search] No more improvements found after {iteration} iterations")
            break

    # Calculate final routes
    final_assignments = {}
    for name, bundles in current_assignments.items():
        interviewer = interviewer_dict[name]
        ordered_bundles, travel_time = calculate_travel_time(
            interviewer['lat'], interviewer['lon'],
            bundles, bundle_info
        )
        final_assignments[name] = (ordered_bundles, travel_time)

    improvement = initial_max_time - current_max_time
    pct_improvement = (improvement / initial_max_time * 100) if initial_max_time > 0 else 0

    print(f"[Local Search] ✓ Optimization complete!")
    print(f"[Local Search] Total improvements: {improvements}")
    print(f"[Local Search] Initial: {initial_max_time:.2f} km → Final: {current_max_time:.2f} km")
    print(f"[Local Search] Improvement: {improvement:.2f} km ({pct_improvement:.1f}%)")

    return final_assignments


def assign_bundles_optimized(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using greedy minimax + local search optimization.

    Returns:
        Dict mapping interviewer name to (ordered_bundles, total_travel_time)
    """
    n_interviewers = len(interviewers)
    n_bundles = len(bundles)

    if n_bundles != n_interviewers * bundles_per_interviewer:
        raise ValueError(
            f"Number of bundles ({n_bundles}) must equal "
            f"number of interviewers ({n_interviewers}) × "
            f"bundles_per_interviewer ({bundles_per_interviewer})"
        )

    print(f"[Optimized Assignment] Step 1: Greedy minimax initialization")
    print(f"{'='*80}")

    # Get initial greedy solution
    initial_assignments = assign_bundles_minimax(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer
    )

    print(f"\n[Optimized Assignment] Step 2: Local search optimization")
    print(f"{'='*80}")

    # Build bundle info
    bundle_info = {}
    for bundle_id in bundles:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        centroid_lat, centroid_lon = get_bundle_centroid(bundle_df)
        internal_dist = calculate_bundle_internal_distance(bundle_df)
        bundle_info[bundle_id] = {
            'lat': centroid_lat,
            'lon': centroid_lon,
            'internal_dist': internal_dist
        }

    # Optimize with local search
    optimized_assignments = optimize_with_local_search(
        interviewers=interviewers,
        initial_assignments=initial_assignments,
        bundle_info=bundle_info,
        max_iterations=100
    )

    return optimized_assignments


def assign_bundles_for_date_optimized(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles to interviewers using optimized algorithm.

    Returns:
        Dict mapping interviewer name to (ordered_bundle_list, total_travel_time)
    """
    # Get interviewers for this date
    interviewer_names = get_interviewers_for_date(date)

    # Load geocoded data
    all_interviewers_df = load_interviewer_data(geocoded_file)

    # Filter to assigned interviewers
    assigned_interviewers = all_interviewers_df[
        all_interviewers_df['name'].isin(interviewer_names)
    ].copy()

    assigned_interviewers = assigned_interviewers.dropna(subset=['lat', 'lon'])

    # Convert to list of dicts
    interviewers = assigned_interviewers[['name', 'email', 'lat', 'lon']].to_dict('records')

    # Assign with optimization
    return assign_bundles_optimized(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer
    )


def main():
    """Test optimized assignment."""
    bundle_file = Path("outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
    bundles_gdf = gpd.read_parquet(bundle_file)

    # Test bundles from Dec 27
    test_bundles = [53, 6015, 3022, 1573, 5417, 1014, 6018, 1602,
                    3262, 1175, 462, 3108, 3630, 2199, 5636, 2176,
                    1711, 1734, 5506, 5313, 2355, 2653, 5682, 3403]

    target_date = "2025-12-27"
    geocoded_file = "data/interviewers_geocoded.csv"

    print(f"{'='*80}")
    print(f"OPTIMIZED ASSIGNMENT TEST")
    print(f"{'='*80}\n")

    assignments = assign_bundles_for_date_optimized(
        date=target_date,
        bundles=test_bundles,
        bundles_gdf=bundles_gdf,
        geocoded_file=geocoded_file,
        bundles_per_interviewer=4
    )

    print(f"\n{'='*80}")
    print(f"FINAL OPTIMIZED RESULTS")
    print(f"{'='*80}\n")

    max_time = 0
    times = []
    for name, (ordered_bundles, travel_time) in sorted(assignments.items()):
        print(f"{name}:")
        print(f"  Route: Home → {' → '.join(map(str, ordered_bundles))}")
        print(f"  Travel time: {travel_time:.2f} km")
        times.append(travel_time)
        max_time = max(max_time, travel_time)

    print(f"\n{'-'*80}")
    print(f"Maximum time: {max_time:.2f} km")
    print(f"Range: {max(times) - min(times):.2f} km")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
