#!/usr/bin/env python3
"""
Joint DH+D2DS Bundle Assignment Optimization

Assigns DH and D2DS bundles together to minimize maximum travel distance.
Distance = Home → DH centroid + DH centroid → D2DS centroid
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


def calculate_travel_distance(
    home_lat: float,
    home_lon: float,
    dh_bundle_id: int,
    d2ds_bundle_id: int,
    bundle_info: Dict
) -> float:
    """
    Calculate total travel distance for one interviewer.
    Distance = Home → DH + DH → D2DS
    """
    dh_lat, dh_lon = bundle_info[dh_bundle_id]['lat'], bundle_info[dh_bundle_id]['lon']
    d2ds_lat, d2ds_lon = bundle_info[d2ds_bundle_id]['lat'], bundle_info[d2ds_bundle_id]['lon']

    dist_home_to_dh = haversine_distance(home_lat, home_lon, dh_lat, dh_lon)
    dist_dh_to_d2ds = haversine_distance(dh_lat, dh_lon, d2ds_lat, d2ds_lon)

    return dist_home_to_dh + dist_dh_to_d2ds


def assign_bundles_dh_d2ds_joint(
    interviewers: List[Dict],
    dh_bundles: List[int],
    d2ds_bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    max_refine_iterations: int = 100
) -> Dict[str, Tuple[int, int, float]]:
    """
    Joint DH+D2DS assignment optimization.

    Args:
        interviewers: List of dicts with 'name', 'lat', 'lon'
        dh_bundles: List of DH bundle IDs (must equal len(interviewers))
        d2ds_bundles: List of D2DS bundle IDs (must equal len(interviewers))
        bundles_gdf: GeoDataFrame with bundle geometries
        max_refine_iterations: Maximum iterations for refinement

    Returns:
        Dict mapping interviewer name to (dh_bundle_id, d2ds_bundle_id, travel_distance_km)
    """
    n = len(interviewers)

    if len(dh_bundles) != n or len(d2ds_bundles) != n:
        raise ValueError(f"Must have exactly {n} DH and {n} D2DS bundles for {n} interviewers")

    print(f"[Joint Assignment] Step 1: Extract bundle centroids")

    # Build bundle info (centroids)
    bundle_info = {}
    all_bundles = set(dh_bundles) | set(d2ds_bundles)

    for bundle_id in all_bundles:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        if len(bundle_df) == 0:
            print(f"Warning: Bundle {bundle_id} not found")
            continue
        lat, lon = get_bundle_centroid(bundle_df)
        bundle_info[bundle_id] = {'lat': lat, 'lon': lon}

    print(f"[Joint Assignment] Step 2: Initial assignment (nearest DH + nearest D2DS)")

    # Initial greedy assignment: each interviewer gets nearest available DH and D2DS
    assignments = {}  # {interviewer_name: {'dh': bundle_id, 'd2ds': bundle_id}}

    remaining_dh = list(dh_bundles)
    remaining_d2ds = list(d2ds_bundles)

    for interviewer in interviewers:
        name = interviewer['name']
        lat = interviewer['lat']
        lon = interviewer['lon']

        # Find nearest DH bundle
        best_dh = None
        best_dh_dist = float('inf')
        for dh_id in remaining_dh:
            if dh_id not in bundle_info:
                continue
            dist = haversine_distance(lat, lon, bundle_info[dh_id]['lat'], bundle_info[dh_id]['lon'])
            if dist < best_dh_dist:
                best_dh_dist = dist
                best_dh = dh_id

        # Find nearest D2DS bundle
        best_d2ds = None
        best_d2ds_dist = float('inf')
        for d2ds_id in remaining_d2ds:
            if d2ds_id not in bundle_info:
                continue
            dist = haversine_distance(lat, lon, bundle_info[d2ds_id]['lat'], bundle_info[d2ds_id]['lon'])
            if dist < best_d2ds_dist:
                best_d2ds_dist = dist
                best_d2ds = d2ds_id

        if best_dh is None or best_d2ds is None:
            raise ValueError(f"Could not find bundles for {name}")

        assignments[name] = {'dh': best_dh, 'd2ds': best_d2ds}
        remaining_dh.remove(best_dh)
        remaining_d2ds.remove(best_d2ds)

    # Calculate initial distances
    def calculate_all_distances():
        distances = {}
        for interviewer in interviewers:
            name = interviewer['name']
            dh_id = assignments[name]['dh']
            d2ds_id = assignments[name]['d2ds']
            dist = calculate_travel_distance(
                interviewer['lat'], interviewer['lon'],
                dh_id, d2ds_id, bundle_info
            )
            distances[name] = dist
        return distances

    travel_distances = calculate_all_distances()
    current_max = max(travel_distances.values())

    print(f"[Joint Assignment] Initial assignment:")
    for name in sorted(travel_distances.keys()):
        print(f"  {name}: DH={assignments[name]['dh']}, D2DS={assignments[name]['d2ds']}, dist={travel_distances[name]:.2f} km")
    print(f"[Joint Assignment] Initial max travel: {current_max:.2f} km")

    print(f"[Joint Assignment] Step 3: Refine with swaps (max {max_refine_iterations} iterations)")

    improved = True
    iteration = 0

    while improved and iteration < max_refine_iterations:
        improved = False
        iteration += 1

        # Sort interviewers by distance (longest first)
        sorted_names = sorted(travel_distances.keys(), key=lambda n: travel_distances[n], reverse=True)

        # Try swapping bundles starting from the interviewer with longest distance
        for i, name1 in enumerate(sorted_names):
            for j in range(i+1, len(sorted_names)):
                name2 = sorted_names[j]

                interviewer1 = next(iv for iv in interviewers if iv['name'] == name1)
                interviewer2 = next(iv for iv in interviewers if iv['name'] == name2)

                # Try 4 types of swaps:
                # 1. Swap DH bundles
                # 2. Swap D2DS bundles
                # 3. Swap interviewer1's DH with interviewer2's D2DS
                # 4. Swap interviewer1's D2DS with interviewer2's DH

                swap_types = [
                    ('dh', 'dh'),      # Swap DH bundles
                    ('d2ds', 'd2ds'),  # Swap D2DS bundles
                ]

                for type1, type2 in swap_types:
                    # Save original state
                    orig_bundle1 = assignments[name1][type1]
                    orig_bundle2 = assignments[name2][type2]

                    # Try swap
                    assignments[name1][type1] = orig_bundle2
                    assignments[name2][type2] = orig_bundle1

                    # Calculate new distances for affected interviewers
                    new_dist1 = calculate_travel_distance(
                        interviewer1['lat'], interviewer1['lon'],
                        assignments[name1]['dh'], assignments[name1]['d2ds'],
                        bundle_info
                    )
                    new_dist2 = calculate_travel_distance(
                        interviewer2['lat'], interviewer2['lon'],
                        assignments[name2]['dh'], assignments[name2]['d2ds'],
                        bundle_info
                    )

                    # Calculate new max
                    new_max = max(new_dist1, new_dist2,
                                 *[travel_distances[n] for n in travel_distances if n not in [name1, name2]])

                    if new_max < current_max:
                        # Keep the swap
                        travel_distances[name1] = new_dist1
                        travel_distances[name2] = new_dist2
                        current_max = new_max
                        improved = True
                        print(f"  Iteration {iteration}: Swapped {type1} bundles {orig_bundle1} ↔ {orig_bundle2} ({name1} ↔ {name2}), new max: {current_max:.2f} km")
                        break  # Move to next interviewer pair after successful swap
                    else:
                        # Undo swap
                        assignments[name1][type1] = orig_bundle1
                        assignments[name2][type2] = orig_bundle2

                if improved:
                    break  # Restart from longest distance interviewer

            if improved:
                break

    print(f"[Joint Assignment] Refinement completed after {iteration} iterations")
    print(f"[Joint Assignment] Final max travel: {current_max:.2f} km")

    print(f"[Joint Assignment] Final assignment:")
    results = {}
    for name in sorted(travel_distances.keys()):
        dh_id = assignments[name]['dh']
        d2ds_id = assignments[name]['d2ds']
        dist = travel_distances[name]
        results[name] = (dh_id, d2ds_id, dist)
        print(f"  {name}: DH={dh_id}, D2DS={d2ds_id}, dist={dist:.2f} km")

    return results


def assign_bundles_for_date_dh_d2ds_joint(
    date: str,
    dh_bundles: List[int],
    d2ds_bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    max_refine_iterations: int = 100,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> Dict[str, Tuple[int, int, float]]:
    """Wrapper function for joint DH+D2DS assignment."""
    from sd311_fieldprep.interviewer_geocoding import get_interviewers_for_date_with_locations

    interviewers = get_interviewers_for_date_with_locations(
        date=date,
        sheet_id=sheet_id
    )

    print(f"[Joint Assignment] Loaded {len(interviewers)} interviewers for {date}")

    return assign_bundles_dh_d2ds_joint(
        interviewers=interviewers,
        dh_bundles=dh_bundles,
        d2ds_bundles=d2ds_bundles,
        bundles_gdf=bundles_gdf,
        max_refine_iterations=max_refine_iterations
    )
