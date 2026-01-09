#!/usr/bin/env python3
"""
Bisection-Aware with Outlier Refinement

Algorithm:
1. Use Bisection-Aware to get initial assignment
2. Post-processing: Iteratively refine by breaking up outlier bundles
   - For each cluster, find the two bundles that are farthest apart
   - Try swapping one of them with bundles from other clusters
   - Accept swap if it reduces max travel distance
3. Repeat until no improvement

This combines global spatial partitioning with local outlier removal.
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


def calculate_tsp_route(
    home_lat: float,
    home_lon: float,
    bundle_ids: List[int],
    bundle_info: Dict
) -> Tuple[List[int], float]:
    """Calculate TSP route using nearest-neighbor greedy algorithm."""
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


def find_outlier_bundles_in_cluster(
    cluster_bundles: List[int],
    bundle_info: Dict
) -> Tuple[int, int, float]:
    """
    Find the two bundles in cluster that are farthest apart.
    Returns: (bundle1_id, bundle2_id, distance)
    """
    if len(cluster_bundles) <= 1:
        return None, None, 0.0

    max_dist = 0.0
    max_pair = (None, None)

    for i in range(len(cluster_bundles)):
        for j in range(i+1, len(cluster_bundles)):
            b1 = cluster_bundles[i]
            b2 = cluster_bundles[j]
            dist = haversine_distance(
                bundle_info[b1]['lat'], bundle_info[b1]['lon'],
                bundle_info[b2]['lat'], bundle_info[b2]['lon']
            )
            if dist > max_dist:
                max_dist = dist
                max_pair = (b1, b2)

    return max_pair[0], max_pair[1], max_dist


def assign_bundles_bisection_refined(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4,
    alpha: float = 0.0,
    max_refine_iterations: int = 50
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using Bisection-Aware with Outlier Refinement.

    Args:
        interviewers: List of dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with bundle geometries
        bundles_per_interviewer: Target number of bundles per interviewer
        alpha: Weight for compactness vs interviewer proximity (for initial bisection)
        max_refine_iterations: Maximum refinement iterations

    Returns:
        Dict mapping interviewer name to (ordered_bundle_ids, travel_distance_km)
    """
    from sd311_fieldprep.assign_bundles_bisection_aware import assign_bundles_bisection_aware

    print(f"[Bisection-Refined] Step 1: Get initial assignment using Bisection-Aware (alpha={alpha})")

    # Get initial assignment from Bisection-Aware
    initial_results = assign_bundles_bisection_aware(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer,
        alpha=alpha
    )

    # Extract assignments (bundle lists, not routes)
    assignments = {name: list(bundle_list) for name, (bundle_list, _) in initial_results.items()}

    # Build bundle info
    print(f"[Bisection-Refined] Step 2: Build bundle info for refinement")
    bundle_info = {}
    for bundle_id in bundles:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        if len(bundle_df) == 0:
            continue
        lat, lon = get_bundle_centroid(bundle_df)
        internal_dist = calculate_bundle_internal_distance(bundle_df)
        bundle_info[bundle_id] = {
            'lat': lat,
            'lon': lon,
            'internal_dist': internal_dist
        }

    # Create interviewer name to interviewer dict mapping
    interviewer_dict = {i['name']: i for i in interviewers}

    # Calculate initial travel distances
    def calculate_all_travel_distances():
        distances = {}
        for name, bundle_list in assignments.items():
            interviewer = interviewer_dict[name]
            _, travel_dist = calculate_tsp_route(
                interviewer['lat'], interviewer['lon'],
                bundle_list, bundle_info
            )
            distances[name] = travel_dist
        return distances

    travel_distances = calculate_all_travel_distances()
    current_max = max(travel_distances.values())

    print(f"[Bisection-Refined] Initial max travel distance: {current_max:.2f} km")

    # Show initial outliers
    print(f"[Bisection-Refined] Initial outlier analysis:")
    for name, bundle_list in sorted(assignments.items()):
        b1, b2, dist = find_outlier_bundles_in_cluster(bundle_list, bundle_info)
        if b1 and b2:
            print(f"  {name}: outlier pair {b1}↔{b2} = {dist:.2f} km")

    print(f"[Bisection-Refined] Step 3: Refine by breaking up outliers (max {max_refine_iterations} iterations)")

    improved = True
    iteration = 0

    while improved and iteration < max_refine_iterations:
        improved = False
        iteration += 1

        # Find the cluster with the worst outlier
        worst_outlier_name = None
        worst_outlier_dist = 0
        worst_outlier_pair = None

        for name, bundle_list in assignments.items():
            b1, b2, dist = find_outlier_bundles_in_cluster(bundle_list, bundle_info)
            if dist > worst_outlier_dist:
                worst_outlier_dist = dist
                worst_outlier_name = name
                worst_outlier_pair = (b1, b2)

        if worst_outlier_name is None:
            break

        # Try swapping one of the outlier bundles with bundles from other clusters
        b1, b2 = worst_outlier_pair

        for outlier_bundle in [b1, b2]:
            # Try swapping with each bundle in each other cluster
            for other_name, other_bundle_list in assignments.items():
                if other_name == worst_outlier_name:
                    continue

                for other_bundle in other_bundle_list:
                    # Try swap
                    assignments[worst_outlier_name].remove(outlier_bundle)
                    assignments[worst_outlier_name].append(other_bundle)
                    assignments[other_name].remove(other_bundle)
                    assignments[other_name].append(outlier_bundle)

                    # Calculate new travel distances for affected clusters
                    _, dist1 = calculate_tsp_route(
                        interviewer_dict[worst_outlier_name]['lat'],
                        interviewer_dict[worst_outlier_name]['lon'],
                        assignments[worst_outlier_name],
                        bundle_info
                    )
                    _, dist2 = calculate_tsp_route(
                        interviewer_dict[other_name]['lat'],
                        interviewer_dict[other_name]['lon'],
                        assignments[other_name],
                        bundle_info
                    )

                    # Calculate new max
                    new_max = max(dist1, dist2,
                                 *[travel_distances[n] for n in travel_distances
                                   if n not in [worst_outlier_name, other_name]])

                    if new_max < current_max:
                        # Keep the swap
                        travel_distances[worst_outlier_name] = dist1
                        travel_distances[other_name] = dist2
                        current_max = new_max
                        improved = True

                        # Calculate new outlier distances
                        b1_new, b2_new, new_outlier_dist = find_outlier_bundles_in_cluster(
                            assignments[worst_outlier_name], bundle_info
                        )

                        print(f"  Iteration {iteration}: Swapped {outlier_bundle} ({worst_outlier_name}) ↔ {other_bundle} ({other_name})")
                        print(f"    Old outlier: {b1}↔{b2} = {worst_outlier_dist:.2f} km")
                        print(f"    New outlier: {b1_new}↔{b2_new} = {new_outlier_dist:.2f} km")
                        print(f"    New max travel: {current_max:.2f} km")
                        break  # Found improvement, restart with new worst outlier
                    else:
                        # Undo swap
                        assignments[worst_outlier_name].remove(other_bundle)
                        assignments[worst_outlier_name].append(outlier_bundle)
                        assignments[other_name].remove(outlier_bundle)
                        assignments[other_name].append(other_bundle)

                if improved:
                    break  # Restart with new worst outlier

            if improved:
                break  # Restart with new worst outlier

    print(f"[Bisection-Refined] Refinement completed after {iteration} iterations")
    print(f"[Bisection-Refined] Final max travel distance: {current_max:.2f} km")

    # Show final outliers
    print(f"[Bisection-Refined] Final outlier analysis:")
    for name, bundle_list in sorted(assignments.items()):
        b1, b2, dist = find_outlier_bundles_in_cluster(bundle_list, bundle_info)
        if b1 and b2:
            print(f"  {name}: outlier pair {b1}↔{b2} = {dist:.2f} km")

    print(f"[Bisection-Refined] Step 4: Calculate final TSP routes")

    # Calculate final TSP routes
    results = {}
    for name, bundle_list in sorted(assignments.items()):
        interviewer = interviewer_dict[name]
        ordered_ids, travel_dist = calculate_tsp_route(
            interviewer['lat'], interviewer['lon'],
            bundle_list, bundle_info
        )
        results[name] = (ordered_ids, travel_dist)
        print(f"  {name}: {len(ordered_ids)} bundles, {travel_dist:.2f} km")

    return results


def assign_bundles_for_date_bisection_refined(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str | None = None,
    bundles_per_interviewer: int = 4,
    alpha: float = 0.0,
    max_refine_iterations: int = 50,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> Dict[str, Tuple[List[int], float]]:
    """Wrapper function for Bisection-Aware with Outlier Refinement."""
    from sd311_fieldprep.interviewer_geocoding import get_interviewers_for_date_with_locations

    interviewers = get_interviewers_for_date_with_locations(
        date=date,
        sheet_id=sheet_id
    )

    print(f"[Bisection-Refined] Loaded {len(interviewers)} interviewers for {date}")

    return assign_bundles_bisection_refined(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer,
        alpha=alpha,
        max_refine_iterations=max_refine_iterations
    )
