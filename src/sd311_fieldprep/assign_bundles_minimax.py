#!/usr/bin/env python
"""
Assign Bundles to Interviewers Using Minimax Optimization

NEW ALGORITHM: Minimize the maximum travel time across all interviewers
This ensures a more fair distribution of workload.

Travel time = home → bundle₁ → bundle₂ → ... → bundleₙ (no return home)
Uses greedy nearest neighbor TSP for routing.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance between two points in kilometers."""
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


def solve_tsp_greedy(home_lat: float, home_lon: float, bundle_locations: List[Tuple[int, float, float]]) -> Tuple[List[int], float]:
    """
    Solve TSP using greedy nearest neighbor heuristic.

    Args:
        home_lat, home_lon: Starting location (home)
        bundle_locations: List of (bundle_id, lat, lon) tuples

    Returns:
        (ordered_bundle_ids, total_distance)
        Total distance = home → B₁ → B₂ → ... → Bₙ (NO return to home)
    """
    if len(bundle_locations) == 0:
        return [], 0.0

    # Start from home
    current_lat, current_lon = home_lat, home_lon
    remaining = list(bundle_locations)
    route = []
    total_distance = 0.0

    # Greedy: always visit nearest unvisited bundle
    while remaining:
        # Find nearest bundle
        nearest_idx = min(range(len(remaining)),
                         key=lambda i: haversine_distance(current_lat, current_lon,
                                                         remaining[i][1], remaining[i][2]))

        nearest_bundle = remaining.pop(nearest_idx)
        bundle_id, bundle_lat, bundle_lon = nearest_bundle

        # Add distance
        dist = haversine_distance(current_lat, current_lon, bundle_lat, bundle_lon)
        total_distance += dist

        # Update current location
        current_lat, current_lon = bundle_lat, bundle_lon
        route.append(bundle_id)

    return route, total_distance


def solve_tsp_optimal(home_lat: float, home_lon: float, bundle_locations: List[Tuple[int, float, float]]) -> Tuple[List[int], float]:
    """
    Solve TSP optimally using brute force (for small n ≤ 7).

    Returns:
        (ordered_bundle_ids, total_distance)
    """
    if len(bundle_locations) == 0:
        return [], 0.0

    if len(bundle_locations) > 7:
        # Too many bundles, use greedy instead
        return solve_tsp_greedy(home_lat, home_lon, bundle_locations)

    best_route = None
    best_distance = float('inf')

    # Try all permutations
    for perm in permutations(bundle_locations):
        # Calculate total distance for this permutation
        current_lat, current_lon = home_lat, home_lon
        distance = 0.0

        for bundle_id, bundle_lat, bundle_lon in perm:
            dist = haversine_distance(current_lat, current_lon, bundle_lat, bundle_lon)
            distance += dist
            current_lat, current_lon = bundle_lat, bundle_lon

        if distance < best_distance:
            best_distance = distance
            best_route = [b[0] for b in perm]

    return best_route, best_distance


def calculate_travel_time(
    home_lat: float,
    home_lon: float,
    bundle_ids: List[int],
    bundle_info: Dict
) -> Tuple[List[int], float]:
    """
    Calculate travel time for an interviewer visiting a set of bundles.

    Travel time = home → bundle₁ → bundle₂ → ... → bundleₙ + internal distances

    Args:
        home_lat, home_lon: Interviewer's home coordinates
        bundle_ids: List of bundle IDs to visit
        bundle_info: Dict mapping bundle_id to {'lat', 'lon', 'internal_dist'}

    Returns:
        (ordered_bundle_ids, total_travel_time in km)
    """
    if len(bundle_ids) == 0:
        return [], 0.0

    # Prepare bundle locations
    bundle_locations = [(bid, bundle_info[bid]['lat'], bundle_info[bid]['lon'])
                       for bid in bundle_ids]

    # Solve TSP
    if len(bundle_ids) <= 7:
        ordered_ids, route_distance = solve_tsp_optimal(home_lat, home_lon, bundle_locations)
    else:
        ordered_ids, route_distance = solve_tsp_greedy(home_lat, home_lon, bundle_locations)

    # Add internal distances
    total_internal = sum(bundle_info[bid]['internal_dist'] for bid in bundle_ids)
    total_travel_time = route_distance + total_internal

    return ordered_ids, total_travel_time


def assign_bundles_minimax(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using minimax optimization (minimize maximum travel time).

    Algorithm:
    1. Build bundle info (centroids, internal distances)
    2. Greedy assignment: For each round, assign one bundle to the interviewer
       with the smallest current travel time
    3. Repeat until all bundles are assigned

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

    # Initialize assignments
    assignments = {interviewer['name']: [] for interviewer in interviewers}

    # Greedy minimax assignment
    remaining_bundles = list(bundles)

    for round_num in range(bundles_per_interviewer):
        print(f"[Minimax Assignment] Round {round_num + 1}/{bundles_per_interviewer}")

        for _ in range(n_interviewers):
            if not remaining_bundles:
                break

            # For each interviewer, calculate their current travel time
            interviewer_times = []
            for interviewer in interviewers:
                name = interviewer['name']
                current_bundles = assignments[name]

                if len(current_bundles) >= bundles_per_interviewer:
                    # Already has enough bundles
                    interviewer_times.append((name, float('inf')))
                    continue

                # Current travel time
                _, current_time = calculate_travel_time(
                    interviewer['lat'], interviewer['lon'],
                    current_bundles, bundle_info
                )
                interviewer_times.append((name, current_time))

            # Find interviewer with minimum current travel time
            min_interviewer_name = min(interviewer_times, key=lambda x: x[1])[0]

            # Find best bundle to assign to this interviewer
            # (bundle that increases their time the least)
            best_bundle = None
            best_new_time = float('inf')

            interviewer = next(i for i in interviewers if i['name'] == min_interviewer_name)
            current_bundles = assignments[min_interviewer_name]

            for bundle_id in remaining_bundles:
                # Calculate new travel time if we add this bundle
                test_bundles = current_bundles + [bundle_id]
                _, new_time = calculate_travel_time(
                    interviewer['lat'], interviewer['lon'],
                    test_bundles, bundle_info
                )

                if new_time < best_new_time:
                    best_new_time = new_time
                    best_bundle = bundle_id

            # Assign best bundle
            if best_bundle is not None:
                assignments[min_interviewer_name].append(best_bundle)
                remaining_bundles.remove(best_bundle)

    # Calculate final routes and travel times
    final_assignments = {}
    for interviewer in interviewers:
        name = interviewer['name']
        bundle_ids = assignments[name]

        ordered_ids, travel_time = calculate_travel_time(
            interviewer['lat'], interviewer['lon'],
            bundle_ids, bundle_info
        )

        final_assignments[name] = (ordered_ids, travel_time)

    return final_assignments


def load_interviewer_data(geocoded_file: str = None) -> pd.DataFrame:
    """Load interviewer data from geocoded CSV."""
    if geocoded_file and Path(geocoded_file).exists():
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


def assign_bundles_for_date_minimax(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str | None = None,
    bundles_per_interviewer: int = 4,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles to interviewers using minimax optimization.

    Args:
        date: Target date
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with bundle geometries
        geocoded_file: DEPRECATED - kept for backward compatibility, not used
        bundles_per_interviewer: Target bundles per interviewer
        sheet_id: Google Sheets ID for interviewer data

    Returns:
        Dict mapping interviewer name to (ordered_bundle_list, total_travel_time)
    """
    # Load interviewers from Google Sheet with geocoding
    from sd311_fieldprep.interviewer_geocoding import get_interviewers_for_date_with_locations

    interviewers = get_interviewers_for_date_with_locations(
        date=date,
        sheet_id=sheet_id
    )

    print(f"[Minimax Assignment] Loaded {len(interviewers)} interviewers for {date}")

    # Assign with minimax
    return assign_bundles_minimax(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer
    )


def main():
    """Test minimax assignment."""
    bundle_file = Path("outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
    bundles_gdf = gpd.read_parquet(bundle_file)

    # Test bundles from Dec 27
    test_bundles = [53, 6015, 3022, 1573, 5417, 1014, 6018, 1602,
                    3262, 1175, 462, 3108, 3630, 2199, 5636, 2176,
                    1711, 1734, 5506, 5313, 2355, 2653, 5682, 3403]

    target_date = "2025-12-27"
    geocoded_file = "data/interviewers_geocoded.csv"

    print(f"=== Minimax Assignment for {target_date} ===\n")

    assignments = assign_bundles_for_date_minimax(
        date=target_date,
        bundles=test_bundles,
        bundles_gdf=bundles_gdf,
        geocoded_file=geocoded_file,
        bundles_per_interviewer=4
    )

    print("\n=== Minimax Results ===\n")

    max_time = 0
    for interviewer_name, (ordered_bundles, travel_time) in sorted(assignments.items()):
        print(f"{interviewer_name}:")
        print(f"  Route: Home → {' → '.join(map(str, ordered_bundles))}")
        print(f"  Total travel time: {travel_time:.2f} km")
        max_time = max(max_time, travel_time)

    print(f"\n=== Maximum travel time: {max_time:.2f} km ===")


if __name__ == '__main__':
    main()
