#!/usr/bin/env python3
"""
Greedy Compact Clustering for Bundle Assignment

This algorithm optimizes for cluster compactness by growing each interviewer's
cluster incrementally, always adding the nearest unassigned bundle.

Algorithm:
1. Seed: Assign each interviewer their closest bundle
2. Grow: Round-robin through interviewers, each time adding the bundle that is
   closest to ANY of their already-assigned bundles
3. Result: Compact, geographically-concentrated clusters

Advantages over K-means:
- Guarantees compactness (bundles are added based on proximity to existing cluster)
- More intuitive (clusters grow outward from interviewer home)
- Avoids pathological cases where clusters span large distances
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from math import radians, sin, cos, sqrt, atan2
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two lat/lon points in km."""
    R = 6371  # Earth radius in km
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

    # Convert to lat/lon if needed
    if bundle_df.crs and not bundle_df.crs.is_geographic:
        bundle_gdf = gpd.GeoDataFrame([{'geometry': centroid}], crs=bundle_df.crs)
        bundle_gdf = bundle_gdf.to_crs('EPSG:4326')
        centroid = bundle_gdf.geometry.iloc[0]

    return (centroid.y, centroid.x)


def calculate_tsp_route(
    home_lat: float,
    home_lon: float,
    bundle_ids: List[int],
    bundle_centroids: Dict[int, Tuple[float, float]]
) -> Tuple[List[int], float]:
    """
    Simple nearest-neighbor TSP for ordering bundles.
    Returns (ordered_bundle_ids, total_travel_distance_km).
    """
    if not bundle_ids:
        return [], 0.0

    if len(bundle_ids) == 1:
        bundle_id = bundle_ids[0]
        lat, lon = bundle_centroids[bundle_id]
        dist = haversine_distance(home_lat, home_lon, lat, lon)
        return [bundle_id], dist

    # Start from home, greedily visit nearest unvisited bundle
    current_lat, current_lon = home_lat, home_lon
    unvisited = set(bundle_ids)
    route = []
    total_distance = 0.0

    while unvisited:
        # Find nearest unvisited bundle
        nearest_id = None
        nearest_dist = float('inf')

        for bundle_id in unvisited:
            lat, lon = bundle_centroids[bundle_id]
            dist = haversine_distance(current_lat, current_lon, lat, lon)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = bundle_id

        # Visit it
        route.append(nearest_id)
        total_distance += nearest_dist
        unvisited.remove(nearest_id)
        current_lat, current_lon = bundle_centroids[nearest_id]

    return route, total_distance


def assign_bundles_greedy_compact(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles to interviewers using greedy compact clustering.

    Args:
        interviewers: List of dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with bundle geometries
        bundles_per_interviewer: Target number of bundles per interviewer

    Returns:
        Dict mapping interviewer name to (ordered_bundle_ids, travel_distance_km)
    """
    n_interviewers = len(interviewers)

    # Step 1: Extract all bundle centroids
    print(f"[Greedy Compact] Extracting centroids for {len(bundles)} bundles...")
    bundle_centroids = {}
    for bundle_id in bundles:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        if len(bundle_df) == 0:
            print(f"Warning: Bundle {bundle_id} not found in bundles_gdf")
            continue
        lat, lon = get_bundle_centroid(bundle_df)
        bundle_centroids[bundle_id] = (lat, lon)

    # Step 2: Seed - assign each interviewer their closest bundle
    print(f"[Greedy Compact] Seeding: assigning closest bundle to each interviewer...")
    assignments = {interviewer['name']: [] for interviewer in interviewers}
    unassigned = set(bundles)

    for interviewer in interviewers:
        name = interviewer['name']
        home_lat = interviewer['lat']
        home_lon = interviewer['lon']

        # Find closest bundle
        closest_bundle = None
        closest_dist = float('inf')
        for bundle_id in unassigned:
            lat, lon = bundle_centroids[bundle_id]
            dist = haversine_distance(home_lat, home_lon, lat, lon)
            if dist < closest_dist:
                closest_dist = dist
                closest_bundle = bundle_id

        if closest_bundle:
            assignments[name].append(closest_bundle)
            unassigned.remove(closest_bundle)
            print(f"  {name}: seeded with bundle {closest_bundle} ({closest_dist:.2f} km from home)")

    # Step 3: Grow clusters - round-robin through interviewers with distance constraint
    MAX_CLUSTER_DISTANCE_KM = 8.0  # Don't add bundles > 8km from existing cluster

    print(f"[Greedy Compact] Growing clusters (target: {bundles_per_interviewer} bundles each)")
    print(f"[Greedy Compact] Distance constraint: max {MAX_CLUSTER_DISTANCE_KM} km from existing cluster")
    round_num = 1

    while unassigned and any(len(assignments[i['name']]) < bundles_per_interviewer for i in interviewers):
        print(f"  Round {round_num}: {len(unassigned)} bundles remaining")

        made_assignment_this_round = False

        for interviewer in interviewers:
            name = interviewer['name']

            # Skip if this interviewer already has enough bundles
            if len(assignments[name]) >= bundles_per_interviewer:
                continue

            if not unassigned:
                break

            # Find the unassigned bundle closest to ANY of this interviewer's assigned bundles
            closest_bundle = None
            closest_dist = float('inf')

            for bundle_id in unassigned:
                bundle_lat, bundle_lon = bundle_centroids[bundle_id]

                # Calculate distance to each assigned bundle, take minimum
                for assigned_id in assignments[name]:
                    assigned_lat, assigned_lon = bundle_centroids[assigned_id]
                    dist = haversine_distance(bundle_lat, bundle_lon, assigned_lat, assigned_lon)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_bundle = bundle_id

            # Only assign if within distance constraint
            if closest_bundle and closest_dist <= MAX_CLUSTER_DISTANCE_KM:
                assignments[name].append(closest_bundle)
                unassigned.remove(closest_bundle)
                print(f"    {name}: added bundle {closest_bundle} ({closest_dist:.2f} km from cluster)")
                made_assignment_this_round = True
            elif closest_bundle:
                print(f"    {name}: skipped bundle {closest_bundle} ({closest_dist:.2f} km > {MAX_CLUSTER_DISTANCE_KM} km threshold)")

        # If no assignments were made this round due to distance constraint,
        # relax the constraint slightly for next round
        if not made_assignment_this_round and unassigned:
            print(f"  No assignments made due to distance constraint, relaxing threshold...")
            MAX_CLUSTER_DISTANCE_KM += 2.0

        round_num += 1

    # Step 4: Handle any remaining unassigned bundles (edge case)
    if unassigned:
        print(f"[Greedy Compact] Warning: {len(unassigned)} bundles remain unassigned, distributing...")
        for interviewer in interviewers:
            if not unassigned:
                break
            bundle_id = unassigned.pop()
            assignments[interviewer['name']].append(bundle_id)

    # Step 5: TSP optimization for each interviewer
    print(f"[Greedy Compact] Optimizing routes with TSP...")
    results = {}

    for interviewer in interviewers:
        name = interviewer['name']
        bundle_ids = assignments[name]

        if not bundle_ids:
            results[name] = ([], 0.0)
            continue

        ordered_ids, travel_dist = calculate_tsp_route(
            interviewer['lat'], interviewer['lon'],
            bundle_ids, bundle_centroids
        )

        results[name] = (ordered_ids, travel_dist)
        print(f"  {name}: {len(ordered_ids)} bundles, {travel_dist:.2f} km total travel")

    return results


def assign_bundles_for_date_greedy_compact(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str | None = None,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Wrapper function for greedy compact bundle assignment.

    Args:
        date: Date string (for logging)
        bundles: List of bundle IDs
        bundles_gdf: GeoDataFrame with bundle geometries
        geocoded_file: Path to interviewers_geocoded.csv
        bundles_per_interviewer: Target bundles per interviewer

    Returns:
        Dict mapping interviewer name to (ordered_bundles, travel_distance_km)
    """
    # Load interviewer geocoded data
    if geocoded_file and Path(geocoded_file).exists():
        interviewers_df = pd.read_csv(geocoded_file)
        interviewers = []

        for _, row in interviewers_df.iterrows():
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                interviewers.append({
                    'name': row['name'],
                    'lat': row['lat'],
                    'lon': row['lon']
                })

        print(f"[Greedy Compact] Loaded {len(interviewers)} interviewers from {geocoded_file}")
    else:
        raise ValueError("geocoded_file is required and must exist")

    return assign_bundles_greedy_compact(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer
    )


if __name__ == "__main__":
    # Test the algorithm
    import sys

    root = Path(__file__).parent.parent.parent

    # Load bundles
    bundles_file = root / "outputs" / "bundles" / "DH" / "bundles_multibfs_regroup_filtered_length.parquet"
    bundles_gdf = gpd.read_parquet(bundles_file)

    # Get all unique bundle IDs (sample 24 for testing)
    all_bundles = sorted(bundles_gdf['bundle_id'].unique())
    np.random.seed(42)
    test_bundles = list(np.random.choice(all_bundles, size=min(24, len(all_bundles)), replace=False))

    # Load interviewers
    geocoded_file = root / "data" / "interviewers_geocoded.csv"

    # Test assignment
    results = assign_bundles_for_date_greedy_compact(
        date="2025-12-27",
        bundles=test_bundles,
        bundles_gdf=bundles_gdf,
        geocoded_file=str(geocoded_file),
        bundles_per_interviewer=4
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for name, (bundles, travel_dist) in sorted(results.items()):
        print(f"{name}: {bundles} ({travel_dist:.2f} km)")
