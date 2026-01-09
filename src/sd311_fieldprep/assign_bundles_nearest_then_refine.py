#!/usr/bin/env python3
"""
Nearest-Then-Refine Bundle Assignment

Algorithm:
1. Initial assignment: Assign each bundle to its nearest interviewer
2. Balance: If some interviewers have too many/few, swap bundles
3. Refine: Iteratively try swapping bundles between interviewers to reduce max travel time

This should create naturally compact clusters since bundles go to nearest interviewer.
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


def assign_bundles_nearest_then_refine(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4,
    max_refine_iterations: int = 100
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using nearest-then-refine algorithm.

    Args:
        interviewers: List of dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with bundle geometries
        bundles_per_interviewer: Target number of bundles per interviewer
        max_refine_iterations: Maximum iterations for refinement

    Returns:
        Dict mapping interviewer name to (ordered_bundle_ids, travel_distance_km)
    """
    print(f"[Nearest-Refine] Step 1: Extract bundle info")

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

    print(f"[Nearest-Refine] Step 2: Initial assignment (nearest interviewer)")

    # Step 1: Assign each bundle to nearest interviewer
    assignments = {i['name']: [] for i in interviewers}

    for bundle_id in bundles:
        if bundle_id not in bundle_info:
            continue

        bundle_lat = bundle_info[bundle_id]['lat']
        bundle_lon = bundle_info[bundle_id]['lon']

        # Find nearest interviewer
        nearest_interviewer = None
        nearest_dist = float('inf')

        for interviewer in interviewers:
            dist = haversine_distance(
                interviewer['lat'], interviewer['lon'],
                bundle_lat, bundle_lon
            )
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_interviewer = interviewer['name']

        assignments[nearest_interviewer].append(bundle_id)

    # Show initial assignment
    print(f"[Nearest-Refine] Initial assignment:")
    for name, bundle_list in assignments.items():
        print(f"  {name}: {len(bundle_list)} bundles")

    # Step 2: Balance assignments
    print(f"[Nearest-Refine] Step 3: Balance assignments")

    target = bundles_per_interviewer

    # Move bundles from over-assigned to under-assigned interviewers
    for _ in range(50):  # Max balance iterations
        # Find interviewers with too many/few bundles
        over_assigned = [(name, bundles) for name, bundles in assignments.items() if len(bundles) > target]
        under_assigned = [(name, bundles) for name, bundles in assignments.items() if len(bundles) < target]

        if not over_assigned or not under_assigned:
            break

        # Move one bundle from over to under
        from_name, from_bundles = over_assigned[0]
        to_name, to_bundles = under_assigned[0]

        # Find which bundle in from_bundles is closest to to_interviewer
        to_interviewer = next(i for i in interviewers if i['name'] == to_name)

        best_bundle = None
        best_dist = float('inf')

        for bundle_id in from_bundles:
            dist = haversine_distance(
                to_interviewer['lat'], to_interviewer['lon'],
                bundle_info[bundle_id]['lat'], bundle_info[bundle_id]['lon']
            )
            if dist < best_dist:
                best_dist = dist
                best_bundle = bundle_id

        # Move bundle
        assignments[from_name].remove(best_bundle)
        assignments[to_name].append(best_bundle)

    # Show balanced assignment
    print(f"[Nearest-Refine] After balancing:")
    for name, bundle_list in assignments.items():
        print(f"  {name}: {len(bundle_list)} bundles")

    # Step 3: Refine with swaps
    print(f"[Nearest-Refine] Step 4: Refine with swaps (max {max_refine_iterations} iterations)")

    # Calculate initial travel distances
    def calculate_travel_distances():
        distances = {}
        for interviewer in interviewers:
            name = interviewer['name']
            bundle_ids = assignments[name]
            _, travel_dist = calculate_tsp_route(
                interviewer['lat'], interviewer['lon'],
                bundle_ids, bundle_info
            )
            distances[name] = travel_dist
        return distances

    travel_distances = calculate_travel_distances()
    current_max = max(travel_distances.values())

    print(f"[Nearest-Refine] Initial max travel: {current_max:.2f} km")

    improved = True
    iteration = 0

    while improved and iteration < max_refine_iterations:
        improved = False
        iteration += 1

        # Try swapping bundles between every pair of interviewers
        for i in range(len(interviewers)):
            for j in range(i+1, len(interviewers)):
                interviewer1 = interviewers[i]
                interviewer2 = interviewers[j]
                name1 = interviewer1['name']
                name2 = interviewer2['name']

                # Make copies to iterate over (since we're modifying the lists)
                bundles1 = list(assignments[name1])
                bundles2 = list(assignments[name2])

                # Try swapping each pair of bundles
                for bundle1 in bundles1:
                    for bundle2 in bundles2:
                        # Skip if already swapped
                        if bundle1 not in assignments[name1] or bundle2 not in assignments[name2]:
                            continue

                        # Try swap
                        assignments[name1].remove(bundle1)
                        assignments[name1].append(bundle2)
                        assignments[name2].remove(bundle2)
                        assignments[name2].append(bundle1)

                        # Calculate new travel distances
                        _, dist1 = calculate_tsp_route(
                            interviewer1['lat'], interviewer1['lon'],
                            assignments[name1], bundle_info
                        )
                        _, dist2 = calculate_tsp_route(
                            interviewer2['lat'], interviewer2['lon'],
                            assignments[name2], bundle_info
                        )

                        new_max = max(dist1, dist2,
                                     *[travel_distances[n] for n in travel_distances if n not in [name1, name2]])

                        if new_max < current_max:
                            # Keep the swap
                            travel_distances[name1] = dist1
                            travel_distances[name2] = dist2
                            current_max = new_max
                            improved = True
                            print(f"  Iteration {iteration}: Swapped {bundle1} ↔ {bundle2} ({name1} ↔ {name2}), new max: {current_max:.2f} km")
                        else:
                            # Undo swap
                            assignments[name1].remove(bundle2)
                            assignments[name1].append(bundle1)
                            assignments[name2].remove(bundle1)
                            assignments[name2].append(bundle2)

    print(f"[Nearest-Refine] Refinement completed after {iteration} iterations")
    print(f"[Nearest-Refine] Final max travel: {current_max:.2f} km")

    # Step 4: Calculate final TSP routes
    print(f"[Nearest-Refine] Step 5: Calculate final TSP routes")

    results = {}
    for interviewer in interviewers:
        name = interviewer['name']
        bundle_ids = assignments[name]

        if not bundle_ids:
            results[name] = ([], 0.0)
            continue

        ordered_ids, travel_dist = calculate_tsp_route(
            interviewer['lat'], interviewer['lon'],
            bundle_ids, bundle_info
        )

        results[name] = (ordered_ids, travel_dist)
        print(f"  {name}: {len(ordered_ids)} bundles, {travel_dist:.2f} km total")

    return results


def assign_bundles_for_date_nearest_then_refine(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str | None = None,
    bundles_per_interviewer: int = 4,
    max_refine_iterations: int = 100,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> Dict[str, Tuple[List[int], float]]:
    """Wrapper function for nearest-then-refine bundle assignment."""
    from sd311_fieldprep.interviewer_geocoding import get_interviewers_for_date_with_locations

    interviewers = get_interviewers_for_date_with_locations(
        date=date,
        sheet_id=sheet_id
    )

    print(f"[Nearest-Refine] Loaded {len(interviewers)} interviewers for {date}")

    return assign_bundles_nearest_then_refine(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer,
        max_refine_iterations=max_refine_iterations
    )
