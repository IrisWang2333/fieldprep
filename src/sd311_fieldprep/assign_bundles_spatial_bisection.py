#!/usr/bin/env python3
"""
Recursive Spatial Bisection for Bundle Assignment

This algorithm recursively divides the geographic space into k regions,
ensuring each region has equal number of bundles and is spatially compact.

Algorithm:
1. For k=6, do 3 levels of bisection (2^3 = 8, but we adjust for k=6)
2. At each level, find the optimal split line that:
   - Divides bundles into equal groups
   - Minimizes the sum of distances from bundles to their region center
3. Assign each region to nearest interviewer using Hungarian algorithm
4. Optimize routes with TSP

Advantages over K-means:
- Guarantees equal-sized regions (no rebalancing needed)
- Considers global spatial structure (not greedy)
- More predictable and stable results
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
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


def find_optimal_split(
    bundle_ids: List[int],
    bundle_centroids: Dict[int, Tuple[float, float]],
    target_size: int
) -> Tuple[List[int], List[int]]:
    """
    Find optimal split of bundles into two groups of equal size.

    Strategy:
    1. Try splitting along lat dimension (horizontal line)
    2. Try splitting along lon dimension (vertical line)
    3. Choose the split that creates more compact groups

    Args:
        bundle_ids: List of bundle IDs to split
        bundle_centroids: Dict mapping bundle_id to (lat, lon)
        target_size: Target size for each group

    Returns:
        (group1, group2) - two lists of bundle IDs
    """
    if len(bundle_ids) <= 2:
        mid = len(bundle_ids) // 2
        return bundle_ids[:mid], bundle_ids[mid:]

    # Extract coordinates
    lats = [bundle_centroids[bid][0] for bid in bundle_ids]
    lons = [bundle_centroids[bid][1] for bid in bundle_ids]

    # Try lat split (horizontal line)
    sorted_by_lat = sorted(zip(bundle_ids, lats), key=lambda x: x[1])
    group1_lat = [bid for bid, _ in sorted_by_lat[:target_size]]
    group2_lat = [bid for bid, _ in sorted_by_lat[target_size:]]

    # Calculate compactness (sum of intra-group distances)
    compactness_lat = 0
    for group in [group1_lat, group2_lat]:
        if len(group) <= 1:
            continue
        center_lat = np.mean([bundle_centroids[bid][0] for bid in group])
        center_lon = np.mean([bundle_centroids[bid][1] for bid in group])
        for bid in group:
            lat, lon = bundle_centroids[bid]
            compactness_lat += haversine_distance(lat, lon, center_lat, center_lon)

    # Try lon split (vertical line)
    sorted_by_lon = sorted(zip(bundle_ids, lons), key=lambda x: x[1])
    group1_lon = [bid for bid, _ in sorted_by_lon[:target_size]]
    group2_lon = [bid for bid, _ in sorted_by_lon[target_size:]]

    compactness_lon = 0
    for group in [group1_lon, group2_lon]:
        if len(group) <= 1:
            continue
        center_lat = np.mean([bundle_centroids[bid][0] for bid in group])
        center_lon = np.mean([bundle_centroids[bid][1] for bid in group])
        for bid in group:
            lat, lon = bundle_centroids[bid]
            compactness_lon += haversine_distance(lat, lon, center_lat, center_lon)

    # Choose the split with better compactness (lower is better)
    if compactness_lat <= compactness_lon:
        return group1_lat, group2_lat
    else:
        return group1_lon, group2_lon


def recursive_bisection(
    bundle_ids: List[int],
    bundle_centroids: Dict[int, Tuple[float, float]],
    k: int
) -> List[List[int]]:
    """
    Recursively bisect bundles into k groups of equal size.

    Args:
        bundle_ids: List of bundle IDs to partition
        bundle_centroids: Dict mapping bundle_id to (lat, lon)
        k: Number of groups to create

    Returns:
        List of k groups, each containing bundle IDs
    """
    if k == 1:
        return [bundle_ids]

    if k == 2:
        # Base case: split into two equal groups
        target_size = len(bundle_ids) // 2
        group1, group2 = find_optimal_split(bundle_ids, bundle_centroids, target_size)
        return [group1, group2]

    # Recursive case: split into k groups
    # Strategy: split into two groups, then recursively split each
    k1 = k // 2
    k2 = k - k1

    target_size1 = (len(bundle_ids) * k1) // k

    group1, group2 = find_optimal_split(bundle_ids, bundle_centroids, target_size1)

    # Recursively split each group
    subgroups1 = recursive_bisection(group1, bundle_centroids, k1)
    subgroups2 = recursive_bisection(group2, bundle_centroids, k2)

    return subgroups1 + subgroups2


def calculate_tsp_route(
    home_lat: float,
    home_lon: float,
    bundle_ids: List[int],
    bundle_info: Dict
) -> Tuple[List[int], float]:
    """
    Simple nearest-neighbor TSP for ordering bundles.
    Returns (ordered_bundle_ids, total_travel_distance_km).
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


def assign_bundles_spatial_bisection(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using recursive spatial bisection.

    Args:
        interviewers: List of dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with bundle geometries
        bundles_per_interviewer: Target number of bundles per interviewer

    Returns:
        Dict mapping interviewer name to (ordered_bundle_ids, travel_distance_km)
    """
    n_interviewers = len(interviewers)

    print(f"[Spatial Bisection] Step 1: Extract bundle centroids")

    # Build bundle info
    bundle_info = {}
    bundle_centroids = {}

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
        bundle_centroids[bundle_id] = (lat, lon)

    print(f"[Spatial Bisection] Step 2: Recursive spatial bisection (k={n_interviewers})")

    # Perform recursive bisection
    groups = recursive_bisection(bundles, bundle_centroids, n_interviewers)

    print(f"[Spatial Bisection] Created {len(groups)} groups:")
    for i, group in enumerate(groups):
        print(f"  Group {i}: {len(group)} bundles")

    # Calculate group centers
    print(f"[Spatial Bisection] Step 3: Calculate group centers")
    group_centers = []
    for group in groups:
        lats = [bundle_centroids[bid][0] for bid in group]
        lons = [bundle_centroids[bid][1] for bid in group]
        group_centers.append((np.mean(lats), np.mean(lons)))

    # Calculate group compactness metrics
    for i, (group, center) in enumerate(zip(groups, group_centers)):
        if len(group) <= 1:
            continue

        max_dist = 0
        for bid in group:
            lat, lon = bundle_centroids[bid]
            dist = haversine_distance(lat, lon, center[0], center[1])
            max_dist = max(max_dist, dist)

        print(f"  Group {i}: center=({center[0]:.4f}, {center[1]:.4f}), max_radius={max_dist:.2f} km")

    print(f"[Spatial Bisection] Step 4: Assign groups to interviewers")

    # Hungarian algorithm: assign groups to interviewers
    cost_matrix = np.zeros((n_interviewers, n_interviewers))
    for i, interviewer in enumerate(interviewers):
        for j, (group_lat, group_lon) in enumerate(group_centers):
            dist = haversine_distance(interviewer['lat'], interviewer['lon'],
                                     group_lat, group_lon)
            cost_matrix[i][j] = dist

    interviewer_indices, group_indices = linear_sum_assignment(cost_matrix)

    print(f"[Spatial Bisection] Step 5: Optimize routes with TSP")

    results = {}
    for interviewer_idx, group_idx in zip(interviewer_indices, group_indices):
        interviewer = interviewers[interviewer_idx]
        name = interviewer['name']
        bundle_ids = groups[group_idx]

        if not bundle_ids:
            results[name] = ([], 0.0)
            continue

        ordered_ids, travel_dist = calculate_tsp_route(
            interviewer['lat'], interviewer['lon'],
            bundle_ids, bundle_info
        )

        results[name] = (ordered_ids, travel_dist)
        dist_to_group = cost_matrix[interviewer_idx][group_idx]
        print(f"  {name}: Group {group_idx} ({len(ordered_ids)} bundles, {dist_to_group:.2f} km to center, {travel_dist:.2f} km total)")

    return results


def assign_bundles_for_date_spatial_bisection(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str | None = None,
    bundles_per_interviewer: int = 4,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> Dict[str, Tuple[List[int], float]]:
    """
    Wrapper function for spatial bisection bundle assignment.

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

    print(f"[Spatial Bisection] Loaded {len(interviewers)} interviewers for {date}")

    return assign_bundles_spatial_bisection(
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

    print("=== Testing Spatial Bisection ===\n")

    results = assign_bundles_for_date_spatial_bisection(
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
