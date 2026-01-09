#!/usr/bin/env python3
"""
Greedy TSP-Based Bundle Assignment

Direct optimization approach:
1. For each interviewer, greedily select bundles that minimize total TSP travel distance
2. Uses actual TSP route calculation (not just centroids)
3. Ensures bundles assigned to same interviewer are close to each other AND close to interviewer

Algorithm:
- Start with empty assignments for all interviewers
- Repeatedly assign the next bundle to the interviewer that would have the smallest
  increase in TSP travel distance
- This naturally creates compact clusters close to interviewers
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two lat/lon points in km."""
    R = 6371
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def get_bundle_centroid(bundle_df: gpd.GeoDataFrame) -> Tuple[float, float]:
    """Get the centroid of a bundle (lat, lon)."""
    union_geom = bundle_df.geometry.union_all()
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


def calculate_tsp_distance(
    home_lat: float,
    home_lon: float,
    bundle_ids: List[int],
    bundle_info: Dict
) -> float:
    """
    Calculate total TSP travel distance for a set of bundles.
    Uses nearest-neighbor greedy TSP.

    Returns:
        Total travel distance in km (home → bundles → done)
    """
    if not bundle_ids:
        return 0.0

    if len(bundle_ids) == 1:
        bundle_id = bundle_ids[0]
        lat, lon = bundle_info[bundle_id]['lat'], bundle_info[bundle_id]['lon']
        dist = haversine_distance(home_lat, home_lon, lat, lon)
        internal = bundle_info[bundle_id]['internal_dist']
        return dist + internal

    current_lat, current_lon = home_lat, home_lon
    unvisited = set(bundle_ids)
    total_distance = 0.0

    while unvisited:
        nearest_id = None
        nearest_dist = float('inf')

        for bundle_id in unvisited:
            lat = bundle_info[bundle_id]['lat']
            lon = bundle_info[bundle_id]['lon']
            dist = haversine_distance(current_lat, current_lon, lat, lon)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = bundle_id

        total_distance += nearest_dist
        total_distance += bundle_info[nearest_id]['internal_dist']
        unvisited.remove(nearest_id)
        current_lat = bundle_info[nearest_id]['lat']
        current_lon = bundle_info[nearest_id]['lon']

    return total_distance


def calculate_tsp_route(
    home_lat: float,
    home_lon: float,
    bundle_ids: List[int],
    bundle_info: Dict
) -> Tuple[List[int], float]:
    """
    Calculate TSP route and return ordered bundle IDs with total distance.
    """
    if not bundle_ids:
        return [], 0.0

    if len(bundle_ids) == 1:
        bundle_id = bundle_ids[0]
        lat, lon = bundle_info[bundle_id]['lat'], bundle_info[bundle_id]['lon']
        dist = haversine_distance(home_lat, home_lon, lat, lon)
        internal = bundle_info[bundle_id]['internal_dist']
        return [bundle_id], dist + internal

    current_lat, current_lon = home_lat, home_lon
    unvisited = set(bundle_ids)
    route = []
    total_distance = 0.0

    while unvisited:
        nearest_id = None
        nearest_dist = float('inf')

        for bundle_id in unvisited:
            lat = bundle_info[bundle_id]['lat']
            lon = bundle_info[bundle_id]['lon']
            dist = haversine_distance(current_lat, current_lon, lat, lon)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = bundle_id

        route.append(nearest_id)
        total_distance += nearest_dist
        total_distance += bundle_info[nearest_id]['internal_dist']
        unvisited.remove(nearest_id)
        current_lat = bundle_info[nearest_id]['lat']
        current_lon = bundle_info[nearest_id]['lon']

    return route, total_distance


def assign_bundles_greedy_tsp(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using greedy TSP optimization.

    Algorithm:
    - Assign bundles one at a time
    - Each bundle goes to the interviewer that would have the smallest
      total TSP travel distance after adding this bundle
    - This ensures assigned bundles are close to each other AND to interviewer

    Args:
        interviewers: List of dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with bundle geometries
        bundles_per_interviewer: Target number of bundles per interviewer

    Returns:
        Dict mapping interviewer name to (ordered_bundle_ids, travel_distance_km)
    """
    print(f"[Greedy TSP] Step 1: Extract bundle info")

    # Build bundle info
    bundle_info = {}
    for bundle_id in bundles:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        if len(bundle_df) == 0:
            print(f"Warning: Bundle {bundle_id} not found")
            continue
        lat, lon = get_bundle_centroid(bundle_df)
        internal_dist = calculate_bundle_internal_distance(bundle_df)

        bundle_info[bundle_id] = {
            'lat': lat,
            'lon': lon,
            'internal_dist': internal_dist
        }

    print(f"[Greedy TSP] Step 2: Greedy assignment based on TSP distance")

    # Initialize assignments
    assignments = {i['name']: [] for i in interviewers}

    # Sort bundles by some initial order (e.g., by latitude)
    # This doesn't matter much, but can help with consistency
    sorted_bundles = sorted(bundles, key=lambda bid: bundle_info[bid]['lat'] if bid in bundle_info else 0)

    # Greedy assignment
    for bundle_id in sorted_bundles:
        if bundle_id not in bundle_info:
            continue

        # Find interviewer that would have smallest TSP distance after adding this bundle
        best_interviewer = None
        best_distance = float('inf')

        for interviewer in interviewers:
            name = interviewer['name']
            current_bundles = assignments[name]

            # Calculate TSP distance with this bundle added
            test_bundles = current_bundles + [bundle_id]
            test_distance = calculate_tsp_distance(
                interviewer['lat'],
                interviewer['lon'],
                test_bundles,
                bundle_info
            )

            # Penalize if this interviewer already has too many bundles
            # (to encourage balanced assignment)
            if len(current_bundles) >= bundles_per_interviewer:
                test_distance *= 2.0  # Penalty for overassignment

            if test_distance < best_distance:
                best_distance = test_distance
                best_interviewer = name

        if best_interviewer:
            assignments[best_interviewer].append(bundle_id)
            print(f"  Bundle {bundle_id} → {best_interviewer} (TSP distance: {best_distance:.2f} km)")

    print(f"[Greedy TSP] Step 3: Optimize routes with TSP")

    # Calculate final TSP routes
    results = {}
    for interviewer in interviewers:
        name = interviewer['name']
        bundle_ids = assignments[name]

        if not bundle_ids:
            results[name] = ([], 0.0)
            continue

        ordered_ids, travel_dist = calculate_tsp_route(
            interviewer['lat'],
            interviewer['lon'],
            bundle_ids,
            bundle_info
        )

        results[name] = (ordered_ids, travel_dist)
        print(f"  {name}: {len(ordered_ids)} bundles, {travel_dist:.2f} km total")

    return results


def assign_bundles_for_date_greedy_tsp(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str | None = None,
    bundles_per_interviewer: int = 4,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> Dict[str, Tuple[List[int], float]]:
    """
    Wrapper function for greedy TSP bundle assignment.

    Args:
        date: Date string (for loading interviewers)
        bundles: List of bundle IDs
        bundles_gdf: GeoDataFrame with bundle geometries
        geocoded_file: DEPRECATED - kept for backward compatibility
        bundles_per_interviewer: Target bundles per interviewer
        sheet_id: Google Sheets ID for interviewer data

    Returns:
        Dict mapping interviewer name to (ordered_bundles, travel_distance_km)
    """
    from sd311_fieldprep.interviewer_geocoding import get_interviewers_for_date_with_locations

    interviewers = get_interviewers_for_date_with_locations(
        date=date,
        sheet_id=sheet_id
    )

    print(f"[Greedy TSP] Loaded {len(interviewers)} interviewers for {date}")

    return assign_bundles_greedy_tsp(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer
    )


if __name__ == '__main__':
    # Test the algorithm
    from pathlib import Path
    import geopandas as gpd
    import pandas as pd

    bundle_file = Path("outputs/bundles/DH/bundles_multibfs_regroup_filtered_length.parquet")
    bundles_gdf = gpd.read_parquet(bundle_file)

    # Load actual Dec 27 plan
    plan_df = pd.read_csv("outputs/plans/bundles_plan_2025-12-27.csv")
    test_bundles = sorted(plan_df['bundle_id'].tolist())

    print("=== Testing Greedy TSP Assignment ===\n")

    results = assign_bundles_for_date_greedy_tsp(
        date="2025-12-27",
        bundles=test_bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=4
    )

    print("\n=== Results ===")
    max_time = 0
    for name, (bundles, time) in sorted(results.items()):
        print(f"{name}: {time:.2f} km (bundles: {bundles})")
        max_time = max(max_time, time)

    print(f"\nMaximum travel time: {max_time:.2f} km")
