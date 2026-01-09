#!/usr/bin/env python3
"""
Edge-First Growth Clustering for Bundle Assignment

This algorithm starts from geographically dispersed "edge" bundles and grows
compact clusters inward, avoiding the central competition problem of standard K-means.

Algorithm:
1. Select k dispersed seed bundles using k-means++ strategy:
   - First seed: random
   - Subsequent seeds: farthest from existing seeds
2. Grow each cluster by iteratively adding nearest unassigned bundle
3. Stop when all bundles assigned
4. Assign each cluster to nearest interviewer using Hungarian algorithm
5. Optimize routes with TSP

Advantages:
- Seeds are naturally dispersed across the geographic space
- Clusters grow inward from edges, maintaining compactness
- No initial dependency on interviewer home locations
- Prevents multiple clusters competing for same central area
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from math import radians, sin, cos, sqrt, atan2
import numpy as np
from scipy.optimize import linear_sum_assignment


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


def select_dispersed_seeds(
    bundles: List[int],
    bundle_centroids: Dict[int, Tuple[float, float]],
    k: int,
    seed: int = 42
) -> List[int]:
    """
    Select k maximally dispersed bundles using k-means++ strategy.

    Returns list of k bundle IDs that are geographically dispersed.
    """
    np.random.seed(seed)

    seeds = []
    unselected = set(bundles)

    # First seed: random
    first_seed = np.random.choice(list(unselected))
    seeds.append(first_seed)
    unselected.remove(first_seed)

    print(f"[Edge Growth] Seed 1: bundle {first_seed}")

    # Subsequent seeds: maximize minimum distance to existing seeds
    for i in range(1, k):
        best_bundle = None
        best_min_dist = -1

        for bundle_id in unselected:
            lat, lon = bundle_centroids[bundle_id]

            # Calculate minimum distance to any existing seed
            min_dist = float('inf')
            for seed_id in seeds:
                seed_lat, seed_lon = bundle_centroids[seed_id]
                dist = haversine_distance(lat, lon, seed_lat, seed_lon)
                min_dist = min(min_dist, dist)

            # Keep track of bundle with maximum min_dist (farthest from all seeds)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_bundle = bundle_id

        seeds.append(best_bundle)
        unselected.remove(best_bundle)
        print(f"[Edge Growth] Seed {i+1}: bundle {best_bundle} ({best_min_dist:.2f} km from nearest seed)")

    return seeds


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

    current_lat, current_lon = home_lat, home_lon
    unvisited = set(bundle_ids)
    route = []
    total_distance = 0.0

    while unvisited:
        nearest_id = None
        nearest_dist = float('inf')

        for bundle_id in unvisited:
            lat, lon = bundle_centroids[bundle_id]
            dist = haversine_distance(current_lat, current_lon, lat, lon)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = bundle_id

        route.append(nearest_id)
        total_distance += nearest_dist
        unvisited.remove(nearest_id)
        current_lat, current_lon = bundle_centroids[nearest_id]

    return route, total_distance


def assign_bundles_edge_growth(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using edge-first growth clustering.

    Args:
        interviewers: List of dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with bundle geometries
        bundles_per_interviewer: Target number of bundles per interviewer

    Returns:
        Dict mapping interviewer name to (ordered_bundle_ids, travel_distance_km)
    """
    n_interviewers = len(interviewers)

    # Step 1: Extract bundle centroids
    print(f"[Edge Growth] Extracting centroids for {len(bundles)} bundles...")
    bundle_centroids = {}
    for bundle_id in bundles:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        if len(bundle_df) == 0:
            print(f"Warning: Bundle {bundle_id} not found")
            continue
        lat, lon = get_bundle_centroid(bundle_df)
        bundle_centroids[bundle_id] = (lat, lon)

    # Step 2: Select dispersed seeds
    print(f"[Edge Growth] Selecting {n_interviewers} dispersed seed bundles...")
    seeds = select_dispersed_seeds(bundles, bundle_centroids, n_interviewers)

    # Step 3: Grow clusters from seeds
    print(f"[Edge Growth] Growing clusters from seeds (target: {bundles_per_interviewer} each)...")

    clusters = [[seed] for seed in seeds]
    unassigned = set(bundles) - set(seeds)

    round_num = 1
    while unassigned:
        print(f"  Round {round_num}: {len(unassigned)} bundles remaining")
        made_assignment = False

        # Round-robin through clusters
        for cluster_idx in range(n_interviewers):
            # Stop if this cluster is full
            if len(clusters[cluster_idx]) >= bundles_per_interviewer:
                continue

            if not unassigned:
                break

            # Find nearest unassigned bundle to ANY bundle in this cluster
            best_bundle = None
            best_dist = float('inf')

            for bundle_id in unassigned:
                lat, lon = bundle_centroids[bundle_id]

                # Calculate min distance to any bundle in cluster
                for cluster_bundle_id in clusters[cluster_idx]:
                    cluster_lat, cluster_lon = bundle_centroids[cluster_bundle_id]
                    dist = haversine_distance(lat, lon, cluster_lat, cluster_lon)
                    if dist < best_dist:
                        best_dist = dist
                        best_bundle = bundle_id

            # Add to cluster
            if best_bundle:
                clusters[cluster_idx].append(best_bundle)
                unassigned.remove(best_bundle)
                made_assignment = True
                print(f"    Cluster {cluster_idx}: added bundle {best_bundle} ({best_dist:.2f} km from cluster)")

        if not made_assignment:
            # Edge case: distribute remaining bundles
            for bundle_id in list(unassigned):
                for cluster_idx in range(n_interviewers):
                    if len(clusters[cluster_idx]) < bundles_per_interviewer:
                        clusters[cluster_idx].append(bundle_id)
                        unassigned.remove(bundle_id)
                        break

        round_num += 1

    # Step 4: Calculate cluster centers
    print(f"[Edge Growth] Calculating cluster centers...")
    cluster_centers = []
    for cluster_bundles in clusters:
        if cluster_bundles:
            lats = [bundle_centroids[bid][0] for bid in cluster_bundles]
            lons = [bundle_centroids[bid][1] for bid in cluster_bundles]
            cluster_centers.append((np.mean(lats), np.mean(lons)))
        else:
            cluster_centers.append((0, 0))

    # Step 5: Assign clusters to interviewers using Hungarian algorithm
    print(f"[Edge Growth] Assigning clusters to interviewers...")
    cost_matrix = np.zeros((n_interviewers, n_interviewers))
    for i, interviewer in enumerate(interviewers):
        for j, (cluster_lat, cluster_lon) in enumerate(cluster_centers):
            dist = haversine_distance(interviewer['lat'], interviewer['lon'],
                                     cluster_lat, cluster_lon)
            cost_matrix[i][j] = dist

    interviewer_indices, cluster_indices = linear_sum_assignment(cost_matrix)

    # Step 6: Calculate cluster compactness metrics
    for cluster_idx, cluster_bundles in enumerate(clusters):
        if len(cluster_bundles) <= 1:
            continue

        # Calculate max pairwise distance (diameter)
        max_dist = 0
        for i in range(len(cluster_bundles)):
            for j in range(i+1, len(cluster_bundles)):
                lat1, lon1 = bundle_centroids[cluster_bundles[i]]
                lat2, lon2 = bundle_centroids[cluster_bundles[j]]
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                max_dist = max(max_dist, dist)

        print(f"  Cluster {cluster_idx}: {len(cluster_bundles)} bundles, diameter {max_dist:.2f} km")

    # Step 7: TSP optimization
    print(f"[Edge Growth] Optimizing routes with TSP...")
    results = {}

    for interviewer_idx, cluster_idx in zip(interviewer_indices, cluster_indices):
        interviewer = interviewers[interviewer_idx]
        name = interviewer['name']
        bundle_ids = clusters[cluster_idx]

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


def assign_bundles_for_date_edge_growth(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str | None = None,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Wrapper function for edge-growth bundle assignment.

    Args:
        date: Date string (for logging)
        bundles: List of bundle IDs
        bundles_gdf: GeoDataFrame with bundle geometries
        geocoded_file: Path to interviewers_geocoded.csv
        bundles_per_interviewer: Target bundles per interviewer

    Returns:
        Dict mapping interviewer name to (ordered_bundles, travel_distance_km)
    """
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

        print(f"[Edge Growth] Loaded {len(interviewers)} interviewers")
    else:
        raise ValueError("geocoded_file is required and must exist")

    return assign_bundles_edge_growth(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer
    )
