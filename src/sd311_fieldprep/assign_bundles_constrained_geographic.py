#!/usr/bin/env python3
"""
Constrained Geographic Clustering

Algorithm:
1. Create exactly k geographic clusters (k = number of interviewers)
2. Each cluster has exactly n bundles (n = bundles_per_interviewer)
3. Minimize maximum cluster diameter (no outliers!)
4. Assign each cluster to nearest interviewer

Strategy:
- Use iterative k-means with size constraints
- Each iteration:
  1. Assign bundles to nearest cluster center (respecting size limits)
  2. Recompute cluster centers
  3. Check if clusters are compact enough
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from math import radians, sin, cos, sqrt, atan2
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


def calculate_bundle_internal_distance(bundle_df: gpd.GeoDataFrame) -> float:
    """Estimate total distance traveled within bundle."""
    if bundle_df.crs and not bundle_df.crs.is_geographic:
        total_length_m = bundle_df.geometry.length.sum()
        return total_length_m / 1000.0
    else:
        total_length_deg = bundle_df.geometry.length.sum()
        return total_length_deg * 111.0


def calculate_cluster_diameter(bundle_ids: List[int], bundle_info: Dict) -> float:
    """Calculate maximum pairwise distance within cluster."""
    if len(bundle_ids) <= 1:
        return 0.0

    max_dist = 0.0
    for i in range(len(bundle_ids)):
        for j in range(i+1, len(bundle_ids)):
            lat1, lon1 = bundle_info[bundle_ids[i]]['lat'], bundle_info[bundle_ids[i]]['lon']
            lat2, lon2 = bundle_info[bundle_ids[j]]['lat'], bundle_info[bundle_ids[j]]['lon']
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            max_dist = max(max_dist, dist)

    return max_dist


def initialize_cluster_centers_kmeans_plus_plus(
    bundle_ids: List[int],
    bundle_info: Dict,
    k: int
) -> List[Tuple[float, float]]:
    """
    Initialize k cluster centers using k-means++ algorithm.
    This ensures centers are well-spread initially.
    """
    centers = []

    # First center: choose randomly
    first_bundle = np.random.choice(bundle_ids)
    centers.append((bundle_info[first_bundle]['lat'], bundle_info[first_bundle]['lon']))

    # Remaining centers: choose bundles farthest from existing centers
    for _ in range(k - 1):
        max_min_dist = 0
        farthest_bundle = None

        for bundle_id in bundle_ids:
            lat, lon = bundle_info[bundle_id]['lat'], bundle_info[bundle_id]['lon']

            # Find minimum distance to any existing center
            min_dist = min([haversine_distance(lat, lon, c[0], c[1]) for c in centers])

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                farthest_bundle = bundle_id

        centers.append((bundle_info[farthest_bundle]['lat'], bundle_info[farthest_bundle]['lon']))

    return centers


def assign_bundles_constrained_geographic(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4,
    max_iterations: int = 50
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using constrained geographic clustering.

    Algorithm:
    1. Create k tight geographic clusters (k = number of interviewers)
    2. Each cluster has exactly bundles_per_interviewer bundles
    3. Minimize maximum cluster diameter
    4. Assign clusters to nearest interviewers

    Args:
        interviewers: List of dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with bundle geometries
        bundles_per_interviewer: Target number of bundles per interviewer
        max_iterations: Maximum iterations for clustering

    Returns:
        Dict mapping interviewer name to (ordered_bundle_ids, travel_distance_km)
    """
    n_clusters = len(interviewers)

    print(f"[Constrained Geographic] Step 1: Extract bundle info")

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

    print(f"[Constrained Geographic] Step 2: Initialize {n_clusters} cluster centers (k-means++)")

    # Initialize cluster centers using k-means++
    cluster_centers = initialize_cluster_centers_kmeans_plus_plus(
        list(bundle_info.keys()),
        bundle_info,
        n_clusters
    )

    print(f"[Constrained Geographic] Step 3: Iterative clustering with size constraints")

    best_clusters = None
    best_max_diameter = float('inf')

    for iteration in range(max_iterations):
        # Assign bundles to clusters with size constraints
        clusters = [[] for _ in range(n_clusters)]
        unassigned = set(bundle_info.keys())

        # Create distance matrix: bundle x cluster
        distances = np.zeros((len(bundle_info), n_clusters))
        bundle_list = list(bundle_info.keys())

        for i, bundle_id in enumerate(bundle_list):
            lat, lon = bundle_info[bundle_id]['lat'], bundle_info[bundle_id]['lon']
            for j, (center_lat, center_lon) in enumerate(cluster_centers):
                distances[i][j] = haversine_distance(lat, lon, center_lat, center_lon)

        # Greedy assignment: repeatedly assign closest bundle-cluster pair
        # that respects size constraints
        while unassigned:
            # Find minimum distance among unassigned bundles
            min_dist = float('inf')
            best_bundle_idx = None
            best_cluster_idx = None

            for i, bundle_id in enumerate(bundle_list):
                if bundle_id not in unassigned:
                    continue

                for j in range(n_clusters):
                    if len(clusters[j]) >= bundles_per_interviewer:
                        continue

                    if distances[i][j] < min_dist:
                        min_dist = distances[i][j]
                        best_bundle_idx = i
                        best_cluster_idx = j

            if best_bundle_idx is None:
                print(f"Warning: Could not assign all bundles!")
                break

            bundle_id = bundle_list[best_bundle_idx]
            clusters[best_cluster_idx].append(bundle_id)
            unassigned.remove(bundle_id)

        # Recompute cluster centers
        new_centers = []
        for cluster in clusters:
            if len(cluster) == 0:
                # Keep old center if cluster is empty
                new_centers.append(cluster_centers[len(new_centers)])
            else:
                # Center = mean of bundle centroids
                lats = [bundle_info[bid]['lat'] for bid in cluster]
                lons = [bundle_info[bid]['lon'] for bid in cluster]
                new_centers.append((np.mean(lats), np.mean(lons)))

        cluster_centers = new_centers

        # Calculate maximum cluster diameter
        max_diameter = 0
        for cluster in clusters:
            diameter = calculate_cluster_diameter(cluster, bundle_info)
            max_diameter = max(max_diameter, diameter)

        # Track best clustering
        if max_diameter < best_max_diameter:
            best_max_diameter = max_diameter
            best_clusters = [list(c) for c in clusters]

        # Check convergence
        if iteration > 0 and abs(max_diameter - best_max_diameter) < 0.01:
            print(f"  Converged after {iteration + 1} iterations")
            break

        if iteration % 10 == 0:
            print(f"  Iteration {iteration + 1}: max diameter = {max_diameter:.2f} km")

    clusters = best_clusters
    print(f"[Constrained Geographic] Final max cluster diameter: {best_max_diameter:.2f} km")

    # Show cluster statistics
    print(f"[Constrained Geographic] Cluster statistics:")
    for i, cluster in enumerate(clusters):
        diameter = calculate_cluster_diameter(cluster, bundle_info)
        print(f"  Cluster {i}: {len(cluster)} bundles, diameter = {diameter:.2f} km")

    print(f"[Constrained Geographic] Step 4: Assign clusters to interviewers")

    # Calculate cluster centers
    final_centers = []
    for cluster in clusters:
        lats = [bundle_info[bid]['lat'] for bid in cluster]
        lons = [bundle_info[bid]['lon'] for bid in cluster]
        final_centers.append((np.mean(lats), np.mean(lons)))

    # Hungarian algorithm: assign clusters to interviewers
    cost_matrix = np.zeros((len(interviewers), n_clusters))
    for i, interviewer in enumerate(interviewers):
        for j, (center_lat, center_lon) in enumerate(final_centers):
            dist = haversine_distance(
                interviewer['lat'], interviewer['lon'],
                center_lat, center_lon
            )
            cost_matrix[i][j] = dist

    interviewer_indices, cluster_indices = linear_sum_assignment(cost_matrix)

    print(f"[Constrained Geographic] Step 5: Calculate TSP routes")

    # Calculate TSP routes
    def calculate_tsp_route(home_lat, home_lon, bundle_ids, bundle_info):
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
            bundle_ids, bundle_info
        )

        results[name] = (ordered_ids, travel_dist)
        cluster_diameter = calculate_cluster_diameter(bundle_ids, bundle_info)
        dist_to_center = cost_matrix[interviewer_idx][cluster_idx]
        print(f"  {name}: {len(ordered_ids)} bundles, diameter={cluster_diameter:.2f}km, {dist_to_center:.2f}km to center, {travel_dist:.2f}km total")

    return results


def assign_bundles_for_date_constrained_geographic(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str | None = None,
    bundles_per_interviewer: int = 4,
    max_iterations: int = 50,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> Dict[str, Tuple[List[int], float]]:
    """Wrapper function for constrained geographic clustering."""
    from sd311_fieldprep.interviewer_geocoding import get_interviewers_for_date_with_locations

    interviewers = get_interviewers_for_date_with_locations(
        date=date,
        sheet_id=sheet_id
    )

    print(f"[Constrained Geographic] Loaded {len(interviewers)} interviewers for {date}")

    return assign_bundles_constrained_geographic(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer,
        max_iterations=max_iterations
    )
