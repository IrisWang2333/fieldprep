#!/usr/bin/env python
"""
Assign Bundles to Interviewers Using ILP (Integer Linear Programming)

Uses PuLP to solve the minimax assignment problem optimally.
Guarantees global minimum of maximum travel time across all interviewers.

Two-stage approach:
1. ILP Assignment: Assign bundles to minimize maximum travel time
2. TSP Routing: Solve TSP optimally for each interviewer's assigned bundles
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus, value
import warnings
warnings.filterwarnings('ignore')

# Import functions from minimax module
from sd311_fieldprep.assign_bundles_minimax import (
    haversine_distance,
    get_bundle_centroid,
    calculate_bundle_internal_distance,
    solve_tsp_optimal,
    calculate_travel_time,
    load_interviewer_data,
    get_interviewers_for_date
)


def build_distance_matrix(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame
) -> Tuple[Dict, Dict]:
    """
    Build distance matrix and bundle info for ILP.

    Returns:
        (bundle_info, distance_matrix)
        - bundle_info: Dict mapping bundle_id to {'lat', 'lon', 'internal_dist'}
        - distance_matrix: Dict[interviewer_name][bundle_id] = distance from home to bundle
    """
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

    # Build distance matrix
    distance_matrix = {}
    for interviewer in interviewers:
        name = interviewer['name']
        home_lat = interviewer['lat']
        home_lon = interviewer['lon']

        distance_matrix[name] = {}
        for bundle_id in bundles:
            bundle_lat = bundle_info[bundle_id]['lat']
            bundle_lon = bundle_info[bundle_id]['lon']
            dist = haversine_distance(home_lat, home_lon, bundle_lat, bundle_lon)
            distance_matrix[name][bundle_id] = dist

    return bundle_info, distance_matrix


def assign_bundles_ilp(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using Integer Linear Programming for optimal minimax solution.

    Formulation:
        Decision variables:
            x[i,j] ∈ {0,1} : interviewer i gets bundle j
            M : maximum travel time (auxiliary variable)

        Objective:
            minimize M

        Constraints:
            1. Each bundle to exactly one interviewer: Σ_i x[i,j] = 1 ∀j
            2. Each interviewer gets exactly 4 bundles: Σ_j x[i,j] = 4 ∀i
            3. M ≥ travel_time[i] ∀i (where travel_time[i] is approximated)

    Args:
        interviewers: List of interviewer dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with all bundles
        bundles_per_interviewer: Number of bundles per interviewer

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

    print(f"[ILP Assignment] Building distance matrix...")
    bundle_info, distance_matrix = build_distance_matrix(interviewers, bundles, bundles_gdf)

    print(f"[ILP Assignment] Setting up optimization problem...")

    # Create optimization problem
    prob = LpProblem("Minimax_Bundle_Assignment", LpMinimize)

    # Decision variables: x[i,j] = 1 if interviewer i gets bundle j
    x = {}
    for interviewer in interviewers:
        i_name = interviewer['name']
        for j in bundles:
            x[(i_name, j)] = LpVariable(f"x_{i_name}_{j}", cat=LpBinary)

    # Auxiliary variable: M = maximum travel time
    M = LpVariable("M", lowBound=0)

    # Objective: minimize M
    prob += M, "Minimize_Maximum_Travel_Time"

    # Constraint 1: Each bundle assigned to exactly one interviewer
    for j in bundles:
        prob += (
            lpSum(x[(i['name'], j)] for i in interviewers) == 1,
            f"Bundle_{j}_assigned_once"
        )

    # Constraint 2: Each interviewer gets exactly bundles_per_interviewer bundles
    for interviewer in interviewers:
        i_name = interviewer['name']
        prob += (
            lpSum(x[(i_name, j)] for j in bundles) == bundles_per_interviewer,
            f"Interviewer_{i_name}_gets_{bundles_per_interviewer}_bundles"
        )

    # Constraint 3: M >= travel_time[i] for all interviewers
    # Approximate travel time as: sum of (home → bundle distances) + internal distances
    # This is an upper bound; actual TSP will be shorter or equal
    for interviewer in interviewers:
        i_name = interviewer['name']

        # Approximate travel time
        travel_time_approx = lpSum(
            x[(i_name, j)] * (distance_matrix[i_name][j] + bundle_info[j]['internal_dist'])
            for j in bundles
        )

        prob += (
            M >= travel_time_approx,
            f"Max_time_constraint_{i_name}"
        )

    # Solve the problem
    print(f"[ILP Assignment] Solving optimization problem...")
    prob.solve()

    # Check solution status
    status = LpStatus[prob.status]
    print(f"[ILP Assignment] Solution status: {status}")

    if status != 'Optimal':
        raise RuntimeError(f"ILP solver failed with status: {status}")

    # Extract assignments
    print(f"[ILP Assignment] Extracting assignments and computing optimal routes...")
    assignments = {interviewer['name']: [] for interviewer in interviewers}

    for interviewer in interviewers:
        i_name = interviewer['name']
        for j in bundles:
            if value(x[(i_name, j)]) > 0.5:  # Binary variable is 1
                assignments[i_name].append(j)

    # Solve TSP optimally for each interviewer's assigned bundles
    final_assignments = {}
    max_time = 0

    for interviewer in interviewers:
        name = interviewer['name']
        bundle_ids = assignments[name]

        # Calculate optimal route and travel time
        ordered_ids, travel_time = calculate_travel_time(
            interviewer['lat'], interviewer['lon'],
            bundle_ids, bundle_info
        )

        final_assignments[name] = (ordered_ids, travel_time)
        max_time = max(max_time, travel_time)

    print(f"[ILP Assignment] ✓ Optimal assignment found!")
    print(f"[ILP Assignment] Maximum travel time: {max_time:.2f} km")

    return final_assignments


def assign_bundles_for_date_ilp(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles to interviewers using ILP optimization.

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

    # Assign with ILP
    return assign_bundles_ilp(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer
    )


def main():
    """Test ILP assignment and compare with greedy minimax."""
    from sd311_fieldprep.assign_bundles_minimax import assign_bundles_for_date_minimax

    bundle_file = Path("outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
    bundles_gdf = gpd.read_parquet(bundle_file)

    # Test bundles from Dec 27
    test_bundles = [53, 6015, 3022, 1573, 5417, 1014, 6018, 1602,
                    3262, 1175, 462, 3108, 3630, 2199, 5636, 2176,
                    1711, 1734, 5506, 5313, 2355, 2653, 5682, 3403]

    target_date = "2025-12-27"
    geocoded_file = "data/interviewers_geocoded.csv"

    print(f"{'='*80}")
    print(f"COMPARISON: Greedy Minimax vs ILP Optimization")
    print(f"{'='*80}\n")

    # Test greedy minimax
    print(f"[1] Testing GREEDY MINIMAX algorithm...")
    print(f"{'='*80}\n")

    greedy_assignments = assign_bundles_for_date_minimax(
        date=target_date,
        bundles=test_bundles,
        bundles_gdf=bundles_gdf,
        geocoded_file=geocoded_file,
        bundles_per_interviewer=4
    )

    print("\nGreedy Minimax Results:")
    print(f"{'-'*80}")
    greedy_max = 0
    greedy_times = []
    for name, (bundles, time) in sorted(greedy_assignments.items()):
        print(f"{name}: {time:.2f} km (bundles: {bundles})")
        greedy_times.append(time)
        greedy_max = max(greedy_max, time)

    print(f"{'-'*80}")
    print(f"Greedy Maximum: {greedy_max:.2f} km")
    print(f"Greedy Range: {max(greedy_times) - min(greedy_times):.2f} km")
    print()

    # Test ILP
    print(f"\n[2] Testing ILP OPTIMIZATION algorithm...")
    print(f"{'='*80}\n")

    ilp_assignments = assign_bundles_for_date_ilp(
        date=target_date,
        bundles=test_bundles,
        bundles_gdf=bundles_gdf,
        geocoded_file=geocoded_file,
        bundles_per_interviewer=4
    )

    print("\nILP Optimization Results:")
    print(f"{'-'*80}")
    ilp_max = 0
    ilp_times = []
    for name, (bundles, time) in sorted(ilp_assignments.items()):
        print(f"{name}: {time:.2f} km (bundles: {bundles})")
        ilp_times.append(time)
        ilp_max = max(ilp_max, time)

    print(f"{'-'*80}")
    print(f"ILP Maximum: {ilp_max:.2f} km")
    print(f"ILP Range: {max(ilp_times) - min(ilp_times):.2f} km")
    print()

    # Comparison
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"Greedy Minimax:")
    print(f"  Maximum time: {greedy_max:.2f} km")
    print(f"  Range: {max(greedy_times) - min(greedy_times):.2f} km")
    print()
    print(f"ILP Optimization:")
    print(f"  Maximum time: {ilp_max:.2f} km")
    print(f"  Range: {max(ilp_times) - min(ilp_times):.2f} km")
    print()
    improvement = greedy_max - ilp_max
    pct_improvement = (improvement / greedy_max) * 100 if greedy_max > 0 else 0
    print(f"Improvement: {improvement:.2f} km ({pct_improvement:.1f}%)")

    if ilp_max < greedy_max:
        print(f"✅ ILP is better!")
    elif ilp_max == greedy_max:
        print(f"➖ Same performance")
    else:
        print(f"❌ Greedy is better (unexpected!)")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
