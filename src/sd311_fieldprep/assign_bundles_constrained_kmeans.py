#!/usr/bin/env python3
"""
Constrained K-means Clustering for Bundle Assignment

This algorithm improves on standard K-means by enforcing cluster compactness constraints.

Algorithm:
1. Run K-means clustering
2. Check each cluster's diameter (max intra-cluster distance)
3. If diameter > threshold, iteratively reassign worst outlier to nearest other cluster
4. Rebalance to ensure equal bundle counts
5. Assign clusters to interviewers using Hungarian algorithm
6. Optimize routes with TSP

Advantages over standard K-means:
- Enforces maximum cluster diameter constraint
- Prevents pathological dispersed clusters
- Maintains equal bundle distribution
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from math import radians, sin, cos, sqrt, atan2
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


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


def calculate_cluster_diameter(bundle_ids: List[int], bundle_centroids: Dict[int, Tuple[float, float]]) -> float:
    """Calculate maximum pairwise distance within a cluster."""
    if len(bundle_ids) <= 1:
        return 0.0

    max_dist = 0.0
    for i in range(len(bundle_ids)):
        for j in range(i+1, len(bundle_ids)):
            lat1, lon1 = bundle_centroids[bundle_ids[i]]
            lat2, lon2 = bundle_centroids[bundle_ids[j]]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            max_dist = max(max_dist, dist)

    return max_dist


def find_worst_outlier(cluster_bundles: List[int], bundle_centroids: Dict[int, Tuple[float, float]]) -> int:
    """Find the bundle that contributes most to cluster diameter."""
    if len(cluster_bundles) <= 1:
        return None

    # For each bundle, calculate max distance to other bundles in cluster
    max_distances = {}
    for bundle_id in cluster_bundles:
        lat1, lon1 = bundle_centroids[bundle_id]
        max_dist = 0.0
        for other_id in cluster_bundles:
            if other_id != bundle_id:
                lat2, lon2 = bundle_centroids[other_id]
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                max_dist = max(max_dist, dist)
        max_distances[bundle_id] = max_dist

    # Return bundle with largest max distance (worst outlier)
    return max(max_distances, key=max_distances.get)


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


def assign_bundles_constrained_kmeans(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4,
    max_cluster_diameter_km: float = 15.0
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles to interviewers using constrained K-means clustering.

    Args:
        interviewers: List of dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with bundle geometries
        bundles_per_interviewer: Target number of bundles per interviewer
        max_cluster_diameter_km: Maximum allowed cluster diameter in km

    Returns:
        Dict mapping interviewer name to (ordered_bundle_ids, travel_distance_km)
    """
    n_interviewers = len(interviewers)

    # Step 1: Extract bundle centroids
    print(f"[Constrained K-means] Extracting centroids for {len(bundles)} bundles...")
    bundle_centroids = {}
    for bundle_id in bundles:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        if len(bundle_df) == 0:
            print(f"Warning: Bundle {bundle_id} not found")
            continue
        lat, lon = get_bundle_centroid(bundle_df)
        bundle_centroids[bundle_id] = (lat, lon)

    # Step 2: Initial K-means clustering
    print(f"[Constrained K-means] Running initial K-means (k={n_interviewers})...")
    X = np.array([bundle_centroids[bid] for bid in bundles])
    kmeans = KMeans(n_clusters=n_interviewers, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Build cluster assignments
    clusters = [[] for _ in range(n_interviewers)]
    for bundle_id, label in zip(bundles, labels):
        clusters[label].append(bundle_id)

    print(f"[Constrained K-means] Initial cluster sizes: {[len(c) for c in clusters]}")

    # Step 3: Enforce diameter constraint
    print(f"[Constrained K-means] Enforcing max diameter constraint: {max_cluster_diameter_km} km")
    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        made_change = False

        for cluster_idx in range(len(clusters)):
            cluster_bundles = clusters[cluster_idx]
            if len(cluster_bundles) <= 1:
                continue

            diameter = calculate_cluster_diameter(cluster_bundles, bundle_centroids)

            if diameter > max_cluster_diameter_km:
                print(f"  Cluster {cluster_idx}: diameter {diameter:.2f} km > {max_cluster_diameter_km} km")

                # Find worst outlier
                outlier = find_worst_outlier(cluster_bundles, bundle_centroids)

                # Find nearest other cluster
                outlier_lat, outlier_lon = bundle_centroids[outlier]
                best_target_cluster = None
                best_dist = float('inf')

                for target_idx in range(len(clusters)):
                    if target_idx == cluster_idx or len(clusters[target_idx]) == 0:
                        continue

                    # Calculate distance to cluster center
                    target_bundles = clusters[target_idx]
                    target_lats = [bundle_centroids[bid][0] for bid in target_bundles]
                    target_lons = [bundle_centroids[bid][1] for bid in target_bundles]
                    target_center_lat = np.mean(target_lats)
                    target_center_lon = np.mean(target_lons)

                    dist = haversine_distance(outlier_lat, outlier_lon, target_center_lat, target_center_lon)
                    if dist < best_dist:
                        best_dist = dist
                        best_target_cluster = target_idx

                # Move outlier to best target cluster
                if best_target_cluster is not None:
                    clusters[cluster_idx].remove(outlier)
                    clusters[best_target_cluster].append(outlier)
                    print(f"    Moved bundle {outlier} from cluster {cluster_idx} to {best_target_cluster}")
                    made_change = True
                    break

        if not made_change:
            break

        iteration += 1

    # Step 4: Rebalance clusters to equal size
    print(f"[Constrained K-means] Rebalancing clusters...")
    while True:
        sizes = [len(c) for c in clusters]
        if max(sizes) - min(sizes) <= 1:  # Allow difference of 1
            break

        # Find largest and smallest clusters
        largest_idx = sizes.index(max(sizes))
        smallest_idx = sizes.index(min(sizes))

        # Find bundle in largest cluster that is closest to smallest cluster
        largest_bundles = clusters[largest_idx]
        smallest_bundles = clusters[smallest_idx]

        if not smallest_bundles:
            # Just move any bundle
            bundle_to_move = largest_bundles[0]
        else:
            smallest_center_lat = np.mean([bundle_centroids[bid][0] for bid in smallest_bundles])
            smallest_center_lon = np.mean([bundle_centroids[bid][1] for bid in smallest_bundles])

            best_bundle = None
            best_dist = float('inf')
            for bundle_id in largest_bundles:
                lat, lon = bundle_centroids[bundle_id]
                dist = haversine_distance(lat, lon, smallest_center_lat, smallest_center_lon)
                if dist < best_dist:
                    best_dist = dist
                    best_bundle = bundle_id

            bundle_to_move = best_bundle

        clusters[largest_idx].remove(bundle_to_move)
        clusters[smallest_idx].append(bundle_to_move)

    print(f"[Constrained K-means] Final cluster sizes: {[len(c) for c in clusters]}")

    # Step 5: Calculate cluster centers and assign to interviewers
    print(f"[Constrained K-means] Assigning clusters to interviewers...")
    cluster_centers = []
    for cluster_bundles in clusters:
        if cluster_bundles:
            lats = [bundle_centroids[bid][0] for bid in cluster_bundles]
            lons = [bundle_centroids[bid][1] for bid in cluster_bundles]
            cluster_centers.append((np.mean(lats), np.mean(lons)))
        else:
            cluster_centers.append((0, 0))

    # Build cost matrix (interviewer to cluster distance)
    cost_matrix = np.zeros((n_interviewers, n_interviewers))
    for i, interviewer in enumerate(interviewers):
        for j, (cluster_lat, cluster_lon) in enumerate(cluster_centers):
            dist = haversine_distance(interviewer['lat'], interviewer['lon'], cluster_lat, cluster_lon)
            cost_matrix[i][j] = dist

    # Hungarian algorithm for optimal assignment
    interviewer_indices, cluster_indices = linear_sum_assignment(cost_matrix)

    # Step 6: TSP optimization for each interviewer
    print(f"[Constrained K-means] Optimizing routes...")
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

        # Calculate final cluster diameter
        diameter = calculate_cluster_diameter(bundle_ids, bundle_centroids)

        results[name] = (ordered_ids, travel_dist)
        print(f"  {name}: {len(ordered_ids)} bundles, {travel_dist:.2f} km travel, {diameter:.2f} km diameter")

    return results


def assign_bundles_for_date_constrained_kmeans(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str | None = None,
    bundles_per_interviewer: int = 4,
    max_cluster_diameter_km: float = 15.0
) -> Dict[str, Tuple[List[int], float]]:
    """
    Wrapper function for constrained K-means bundle assignment.

    Args:
        date: Date string (for logging)
        bundles: List of bundle IDs
        bundles_gdf: GeoDataFrame with bundle geometries
        geocoded_file: Path to interviewers_geocoded.csv
        bundles_per_interviewer: Target bundles per interviewer
        max_cluster_diameter_km: Maximum cluster diameter in km

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

        print(f"[Constrained K-means] Loaded {len(interviewers)} interviewers")
    else:
        raise ValueError("geocoded_file is required and must exist")

    return assign_bundles_constrained_kmeans(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer,
        max_cluster_diameter_km=max_cluster_diameter_km
    )
