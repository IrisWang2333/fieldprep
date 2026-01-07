#!/usr/bin/env python
"""
Assign Bundles Using Spatial Clustering

Strategy:
1. First cluster bundles geographically (k-means)
2. Then assign each cluster to the nearest interviewer
3. Optimize within-cluster routing with TSP

This ensures each interviewer gets spatially concentrated bundles,
minimizing travel between bundles.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from sd311_fieldprep.assign_bundles_minimax import (
    haversine_distance,
    get_bundle_centroid,
    calculate_bundle_internal_distance,
    calculate_travel_time,
    load_interviewer_data,
    get_interviewers_for_date
)


def assign_bundles_clustered(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using spatial clustering.

    Algorithm:
    1. Extract bundle centroids
    2. K-means clustering (k = number of interviewers)
    3. Assign each cluster to nearest interviewer
    4. Solve TSP for each interviewer's cluster

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

    print(f"[Clustered Assignment] Step 1: Extract bundle centroids")

    # Build bundle info and extract centroids
    bundle_info = {}
    bundle_centroids = []

    for bundle_id in bundles:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        centroid_lat, centroid_lon = get_bundle_centroid(bundle_df)
        internal_dist = calculate_bundle_internal_distance(bundle_df)

        bundle_info[bundle_id] = {
            'lat': centroid_lat,
            'lon': centroid_lon,
            'internal_dist': internal_dist
        }
        bundle_centroids.append([centroid_lat, centroid_lon])

    # Convert to numpy array for k-means
    X = np.array(bundle_centroids)

    print(f"[Clustered Assignment] Step 2: K-means clustering (k={n_interviewers})")

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_interviewers, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_

    # Group bundles by cluster
    clusters = {i: [] for i in range(n_interviewers)}
    for bundle_id, cluster_id in zip(bundles, cluster_labels):
        clusters[cluster_id].append(bundle_id)

    initial_sizes = [len(clusters[i]) for i in range(n_interviewers)]
    print(f"[Clustered Assignment] Initial cluster sizes: {initial_sizes}")

    # Rebalance clusters to ensure each has exactly bundles_per_interviewer
    print(f"[Clustered Assignment] Rebalancing to {bundles_per_interviewer} bundles per cluster...")

    # Build bundle-to-cluster mapping and distances to cluster centers
    bundle_cluster_distances = {}
    for bundle_id, (lat, lon) in zip(bundles, X):
        # Calculate distance to all cluster centers
        distances_to_centers = []
        for cluster_id in range(n_interviewers):
            center_lat, center_lon = cluster_centers[cluster_id]
            dist = haversine_distance(lat, lon, center_lat, center_lon)
            distances_to_centers.append((cluster_id, dist))
        # Sort by distance (closest first)
        distances_to_centers.sort(key=lambda x: x[1])
        bundle_cluster_distances[bundle_id] = distances_to_centers

    # Iteratively rebalance
    max_iterations = 100
    for iteration in range(max_iterations):
        # Find overcrowded and undercrowded clusters
        overcrowded = [i for i in range(n_interviewers) if len(clusters[i]) > bundles_per_interviewer]
        undercrowded = [i for i in range(n_interviewers) if len(clusters[i]) < bundles_per_interviewer]

        if not overcrowded or not undercrowded:
            break

        # Move bundles from overcrowded to undercrowded
        for over_cluster in overcrowded:
            while len(clusters[over_cluster]) > bundles_per_interviewer and undercrowded:
                # Find the bundle in over_cluster that is closest to an undercrowded cluster
                best_bundle = None
                best_target_cluster = None
                best_distance = float('inf')

                for bundle_id in clusters[over_cluster]:
                    # Find closest undercrowded cluster for this bundle
                    for target_cluster, dist in bundle_cluster_distances[bundle_id]:
                        if target_cluster in undercrowded and len(clusters[target_cluster]) < bundles_per_interviewer:
                            if dist < best_distance:
                                best_distance = dist
                                best_bundle = bundle_id
                                best_target_cluster = target_cluster
                            break

                if best_bundle is not None:
                    # Move bundle
                    clusters[over_cluster].remove(best_bundle)
                    clusters[best_target_cluster].append(best_bundle)

                    # Update undercrowded list
                    if len(clusters[best_target_cluster]) == bundles_per_interviewer:
                        undercrowded.remove(best_target_cluster)
                else:
                    break

    final_sizes = [len(clusters[i]) for i in range(n_interviewers)]
    print(f"[Clustered Assignment] Final cluster sizes: {final_sizes}")

    print(f"[Clustered Assignment] Step 3: Assign clusters to interviewers")

    # Assign each cluster to the nearest interviewer
    # Use Hungarian algorithm (optimal assignment)
    from scipy.optimize import linear_sum_assignment

    # Build cost matrix: cost[i][j] = distance from interviewer i to cluster j center
    cost_matrix = np.zeros((n_interviewers, n_interviewers))

    for i, interviewer in enumerate(interviewers):
        home_lat = interviewer['lat']
        home_lon = interviewer['lon']

        for j in range(n_interviewers):
            cluster_center_lat = cluster_centers[j][0]
            cluster_center_lon = cluster_centers[j][1]
            dist = haversine_distance(home_lat, home_lon, cluster_center_lat, cluster_center_lon)
            cost_matrix[i][j] = dist

    # Solve assignment problem
    interviewer_indices, cluster_indices = linear_sum_assignment(cost_matrix)

    # Create assignments
    assignments = {}
    for interviewer_idx, cluster_idx in zip(interviewer_indices, cluster_indices):
        interviewer = interviewers[interviewer_idx]
        name = interviewer['name']
        assigned_bundles = clusters[cluster_idx]
        assignments[name] = assigned_bundles

        cluster_center = cluster_centers[cluster_idx]
        dist_to_cluster = cost_matrix[interviewer_idx][cluster_idx]
        print(f"  {name}: Cluster {cluster_idx} ({len(assigned_bundles)} bundles, {dist_to_cluster:.2f} km to center)")

    print(f"[Clustered Assignment] Step 4: Optimize TSP routing within each cluster")

    # Calculate optimal routes for each interviewer
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

    print(f"[Clustered Assignment] ✓ Clustering-based assignment complete!")
    print(f"[Clustered Assignment] Maximum travel time: {max_time:.2f} km")

    return final_assignments


def assign_bundles_for_date_clustered(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str,
    bundles_per_interviewer: int = 4
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles to interviewers using spatial clustering.

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

    # Assign with clustering
    return assign_bundles_clustered(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer
    )


def main():
    """Test clustering-based assignment."""
    from sd311_fieldprep.assign_bundles_optimized import assign_bundles_for_date_optimized

    bundle_file = Path("outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
    bundles_gdf = gpd.read_parquet(bundle_file)

    # Test bundles from Dec 27
    test_bundles = [1604, 1412, 2665, 2968, 1736, 1713, 3630, 2192,
                    1935, 2204, 3112, 458, 1923, 1324, 5671, 5687,
                    6006, 813, 900, 5558, 2356, 3403, 3026, 1014]

    target_date = "2025-12-27"
    geocoded_file = "data/interviewers_geocoded.csv"

    print(f"{'='*80}")
    print(f"COMPARISON: Optimized vs Clustered Assignment")
    print(f"{'='*80}\n")

    # Test optimized (current best)
    print(f"[1] Testing OPTIMIZED algorithm (greedy + local search)...")
    print(f"{'='*80}\n")

    optimized_assignments = assign_bundles_for_date_optimized(
        date=target_date,
        bundles=test_bundles,
        bundles_gdf=bundles_gdf,
        geocoded_file=geocoded_file,
        bundles_per_interviewer=4
    )

    print("\nOptimized Results:")
    print(f"{'-'*80}")
    opt_max = 0
    opt_times = []
    for name, (bundles, time) in sorted(optimized_assignments.items()):
        print(f"{name}: {time:.2f} km")
        opt_times.append(time)
        opt_max = max(opt_max, time)

    print(f"{'-'*80}")
    print(f"Optimized Maximum: {opt_max:.2f} km")
    print(f"Optimized Range: {max(opt_times) - min(opt_times):.2f} km")
    print(f"Optimized Average: {np.mean(opt_times):.2f} km")
    print()

    # Test clustered
    print(f"\n[2] Testing CLUSTERED algorithm (spatial clustering)...")
    print(f"{'='*80}\n")

    clustered_assignments = assign_bundles_for_date_clustered(
        date=target_date,
        bundles=test_bundles,
        bundles_gdf=bundles_gdf,
        geocoded_file=geocoded_file,
        bundles_per_interviewer=4
    )

    print("\nClustered Results:")
    print(f"{'-'*80}")
    clust_max = 0
    clust_times = []
    for name, (bundles, time) in sorted(clustered_assignments.items()):
        print(f"{name}: {time:.2f} km")
        clust_times.append(time)
        clust_max = max(clust_max, time)

    print(f"{'-'*80}")
    print(f"Clustered Maximum: {clust_max:.2f} km")
    print(f"Clustered Range: {max(clust_times) - min(clust_times):.2f} km")
    print(f"Clustered Average: {np.mean(clust_times):.2f} km")
    print()

    # Comparison
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"Optimized (greedy + local search):")
    print(f"  Maximum time: {opt_max:.2f} km")
    print(f"  Range: {max(opt_times) - min(opt_times):.2f} km")
    print(f"  Average: {np.mean(opt_times):.2f} km")
    print()
    print(f"Clustered (spatial clustering):")
    print(f"  Maximum time: {clust_max:.2f} km")
    print(f"  Range: {max(clust_times) - min(clust_times):.2f} km")
    print(f"  Average: {np.mean(clust_times):.2f} km")
    print()

    improvement = opt_max - clust_max
    pct_improvement = (improvement / opt_max) * 100 if opt_max > 0 else 0

    print(f"Maximum time change: {improvement:.2f} km ({pct_improvement:.1f}%)")

    if clust_max < opt_max:
        print(f"✅ Clustered is better for max time!")
    elif clust_max == opt_max:
        print(f"➖ Same max time")
    else:
        print(f"❌ Optimized is better for max time")

    avg_improvement = np.mean(opt_times) - np.mean(clust_times)
    print(f"Average time change: {avg_improvement:.2f} km")

    if np.mean(clust_times) < np.mean(opt_times):
        print(f"✅ Clustered has lower average time!")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
