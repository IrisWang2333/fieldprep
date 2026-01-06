#!/usr/bin/env python
"""
Assign Bundles to Interviewers with Optimal Routing

This module solves a two-stage optimization problem:
1. Assignment: Which bundles go to which interviewer? (Hungarian Algorithm)
2. Routing: What order to visit assigned bundles? (TSP)

Total distance for each interviewer:
Home → Bundle₁ → Bundle₂ → ... → Bundleₙ → Home
(including internal distance within each bundle)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Haversine distance between two points in kilometers.
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth radius in km

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def get_bundle_centroid(bundle_df: gpd.GeoDataFrame) -> Tuple[float, float]:
    """Calculate bundle centroid from segment geometries."""
    union_geom = bundle_df.geometry.unary_union
    centroid = union_geom.centroid

    if bundle_df.crs and not bundle_df.crs.is_geographic:
        bundle_gdf = gpd.GeoDataFrame([{'geometry': centroid}], crs=bundle_df.crs)
        bundle_gdf = bundle_gdf.to_crs('EPSG:4326')
        centroid = bundle_gdf.geometry.iloc[0]

    return (centroid.y, centroid.x)


def calculate_bundle_internal_distance(bundle_df: gpd.GeoDataFrame) -> float:
    """Estimate total distance traveled within bundle."""
    if bundle_df.crs and not bundle_df.crs.is_geographic:
        total_length_m = bundle_df.geometry.length.sum()
        return total_length_m / 1000.0
    else:
        total_length_deg = bundle_df.geometry.length.sum()
        return total_length_deg * 111.0


def solve_tsp_brute_force(distance_matrix: np.ndarray, start_idx: int = 0) -> Tuple[List[int], float]:
    """
    Solve TSP using brute force (for small n).

    Args:
        distance_matrix: n×n matrix of distances between locations
        start_idx: Index of starting location (usually 0 for Home)

    Returns:
        (best_tour, best_distance)
        best_tour: List of indices in visit order [start, i, j, k, ..., start]
        best_distance: Total distance of the tour
    """
    n = len(distance_matrix)

    if n <= 1:
        return [start_idx], 0.0

    # Generate all permutations of locations (excluding start)
    other_locations = [i for i in range(n) if i != start_idx]

    best_tour = None
    best_distance = float('inf')

    # Try all permutations
    for perm in permutations(other_locations):
        # Build tour: start → perm[0] → perm[1] → ... → perm[-1] → start
        tour = [start_idx] + list(perm) + [start_idx]

        # Calculate total distance
        distance = 0
        for i in range(len(tour) - 1):
            distance += distance_matrix[tour[i], tour[i+1]]

        if distance < best_distance:
            best_distance = distance
            best_tour = tour

    return best_tour, best_distance


def solve_tsp_greedy(distance_matrix: np.ndarray, start_idx: int = 0) -> Tuple[List[int], float]:
    """
    Solve TSP using greedy nearest neighbor heuristic (for larger n).

    Not optimal but fast: O(n²) instead of O(n!)
    """
    n = len(distance_matrix)

    if n <= 1:
        return [start_idx], 0.0

    unvisited = set(range(n))
    unvisited.remove(start_idx)

    tour = [start_idx]
    current = start_idx
    total_distance = 0

    # Greedily visit nearest unvisited location
    while unvisited:
        nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
        total_distance += distance_matrix[current, nearest]
        tour.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    # Return to start
    total_distance += distance_matrix[current, start_idx]
    tour.append(start_idx)

    return tour, total_distance


def calculate_route_distance(
    home_lat: float,
    home_lon: float,
    bundle_ids: List[int],
    bundles_gdf: gpd.GeoDataFrame
) -> Tuple[List[int], float]:
    """
    Calculate optimal route for visiting bundles starting and ending at home.

    Args:
        home_lat, home_lon: Interviewer's home coordinates
        bundle_ids: List of bundle IDs to visit
        bundles_gdf: GeoDataFrame with all bundles

    Returns:
        (optimal_order, total_distance)
        optimal_order: Bundle IDs in visit order
        total_distance: Home → B₁ → ... → Bₙ → Home (including internal distances)
    """
    n_bundles = len(bundle_ids)

    if n_bundles == 0:
        return [], 0.0

    # Prepare bundle info
    bundle_info = {}
    for bundle_id in bundle_ids:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        centroid_lat, centroid_lon = get_bundle_centroid(bundle_df)
        internal_dist = calculate_bundle_internal_distance(bundle_df)
        bundle_info[bundle_id] = {
            'lat': centroid_lat,
            'lon': centroid_lon,
            'internal': internal_dist
        }

    # Build distance matrix
    # Locations: [Home, Bundle₁, Bundle₂, ..., Bundleₙ]
    n = n_bundles + 1
    dist_matrix = np.zeros((n, n))

    # Index mapping: 0 = Home, 1 = bundle_ids[0], 2 = bundle_ids[1], ...

    # Distances from/to Home (index 0)
    for i, bundle_id in enumerate(bundle_ids):
        idx = i + 1
        bundle_lat = bundle_info[bundle_id]['lat']
        bundle_lon = bundle_info[bundle_id]['lon']

        # Home ↔ Bundle
        dist = haversine_distance(home_lat, home_lon, bundle_lat, bundle_lon)
        dist_matrix[0, idx] = dist
        dist_matrix[idx, 0] = dist

    # Distances between bundles
    for i, bundle_id_i in enumerate(bundle_ids):
        for j, bundle_id_j in enumerate(bundle_ids):
            if i != j:
                idx_i = i + 1
                idx_j = j + 1

                lat_i = bundle_info[bundle_id_i]['lat']
                lon_i = bundle_info[bundle_id_i]['lon']
                lat_j = bundle_info[bundle_id_j]['lat']
                lon_j = bundle_info[bundle_id_j]['lon']

                dist = haversine_distance(lat_i, lon_i, lat_j, lon_j)
                dist_matrix[idx_i, idx_j] = dist

    # Solve TSP
    if n_bundles <= 4:
        # For small n (≤4), use brute force (optimal)
        tour_indices, route_distance = solve_tsp_brute_force(dist_matrix, start_idx=0)
    else:
        # For larger n, use greedy (fast but suboptimal)
        tour_indices, route_distance = solve_tsp_greedy(dist_matrix, start_idx=0)

    # Convert indices back to bundle IDs
    # tour_indices is like [0, 2, 1, 3, 0] (Home → B₁ → B₀ → B₂ → Home)
    optimal_order = [bundle_ids[idx - 1] for idx in tour_indices[1:-1]]

    # Add internal distances
    total_distance = route_distance
    for bundle_id in bundle_ids:
        total_distance += bundle_info[bundle_id]['internal']

    return optimal_order, total_distance


def calculate_cost_matrix_with_routing(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4
) -> np.ndarray:
    """
    Calculate cost matrix where each cell represents the total route distance
    for an interviewer visiting a specific set of bundles.

    This is exponentially expensive! For 6 interviewers and 24 bundles:
    - Need to evaluate C(24, 4) = 10,626 combinations per interviewer
    - Total: 6 × 10,626 = 63,756 TSP solves

    This is why we use a simpler approximation in practice.
    """
    # This is the theoretically correct but computationally expensive approach
    # We'll implement a heuristic instead (see below)
    raise NotImplementedError("Full routing-aware assignment is too expensive")


def assign_bundles_with_routing_heuristic(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Two-stage heuristic assignment:

    Stage 1: Use simplified cost (average distance from home) for assignment
    Stage 2: For each interviewer's assigned bundles, solve TSP for optimal route

    Returns:
        Dict mapping interviewer name to (ordered_bundles, total_distance)
    """
    n_interviewers = len(interviewers)
    n_bundles = len(bundles)

    # Prepare bundle info
    bundle_info = {}
    for bundle_id in bundles:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        centroid_lat, centroid_lon = get_bundle_centroid(bundle_df)
        internal_dist = calculate_bundle_internal_distance(bundle_df)
        bundle_info[bundle_id] = {
            'lat': centroid_lat,
            'lon': centroid_lon,
            'internal': internal_dist
        }

    # Stage 1: Initial assignment using simplified cost
    # Cost = average distance from home to bundle centroid
    cost_matrix = np.zeros((n_interviewers, n_bundles))

    for i, interviewer in enumerate(interviewers):
        for j, bundle_id in enumerate(bundles):
            bundle_lat = bundle_info[bundle_id]['lat']
            bundle_lon = bundle_info[bundle_id]['lon']

            # Simple cost: distance from home to bundle
            dist = haversine_distance(
                interviewer['lat'], interviewer['lon'],
                bundle_lat, bundle_lon
            )

            # Add internal distance
            cost_matrix[i, j] = dist + bundle_info[bundle_id]['internal']

    # Expand matrix for multiple bundles per interviewer
    expanded_cost = np.repeat(cost_matrix, bundles_per_interviewer, axis=0)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(expanded_cost)

    # Map back to interviewers
    initial_assignments = {interviewer['name']: [] for interviewer in interviewers}
    for row, col in zip(row_ind, col_ind):
        interviewer_idx = row // bundles_per_interviewer
        interviewer_name = interviewers[interviewer_idx]['name']
        bundle_id = bundles[col]
        initial_assignments[interviewer_name].append(bundle_id)

    # Stage 2: For each interviewer, solve TSP to find optimal route
    final_assignments = {}

    for interviewer in interviewers:
        name = interviewer['name']
        assigned_bundles = initial_assignments[name]

        if len(assigned_bundles) > 0:
            # Solve TSP for this interviewer's bundles
            optimal_order, total_distance = calculate_route_distance(
                interviewer['lat'],
                interviewer['lon'],
                assigned_bundles,
                bundles_gdf
            )

            final_assignments[name] = (optimal_order, total_distance)
        else:
            final_assignments[name] = ([], 0.0)

    return final_assignments


def load_interviewer_data(geocoded_file: str) -> pd.DataFrame:
    """Load interviewer data from geocoded CSV."""
    if Path(geocoded_file).exists():
        return pd.read_csv(geocoded_file)
    else:
        raise FileNotFoundError(f"Geocoded file not found: {geocoded_file}")


def get_interviewers_for_date(
    date: str,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> List[str]:
    """Get list of interviewer names assigned for a specific date."""
    assignments_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0'
    assignments_df = pd.read_csv(assignments_url)

    matching_rows = assignments_df[assignments_df['Date'] == date]

    if len(matching_rows) == 0:
        raise ValueError(f"No assignment found for date {date}")

    row = matching_rows.iloc[0]
    return [row['A'], row['B'], row['C'], row['D'], row['E'], row['F']]


def assign_bundles_for_date_with_routing(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles to interviewers with optimal routing.

    Returns:
        Dict mapping interviewer name to (ordered_bundle_list, total_distance)
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

    # Assign with routing
    return assign_bundles_with_routing_heuristic(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer
    )


def main():
    """Test the routing-aware assignment."""
    bundle_file = Path("outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
    bundles_gdf = gpd.read_parquet(bundle_file)

    # Test bundles
    test_bundles = [5693, 2359, 3631, 2046, 6011, 5103, 5501, 5602,
                    5830, 2671, 2221, 5699, 5110, 5503, 2194, 5672,
                    1940, 3025, 3406, 5418, 1326, 1715, 485, 3383]

    target_date = "2026-01-10"
    geocoded_file = "data/interviewers_geocoded.csv"

    print(f"=== Assigning Bundles with Routing for {target_date} ===\n")

    assignments = assign_bundles_for_date_with_routing(
        date=target_date,
        bundles=test_bundles,
        bundles_gdf=bundles_gdf,
        geocoded_file=geocoded_file,
        bundles_per_interviewer=4
    )

    print("\n=== Optimized Assignments with Routing ===\n")

    total_all = 0
    for interviewer_name, (ordered_bundles, total_dist) in sorted(assignments.items()):
        print(f"{interviewer_name}:")
        print(f"  Route: Home → {' → '.join(map(str, ordered_bundles))} → Home")
        print(f"  Total distance: {total_dist:.2f} km")
        total_all += total_dist

    print(f"\n=== Total distance for all interviewers: {total_all:.2f} km ===")


if __name__ == '__main__':
    main()
