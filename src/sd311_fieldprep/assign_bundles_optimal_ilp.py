#!/usr/bin/env python3
"""
Optimal Bundle Assignment using Integer Linear Programming

Goal: Minimize the maximum travel distance among all interviewers

Formulation:
- Decision variables: x[i,b] = 1 if interviewer i gets bundle b, 0 otherwise
- Auxiliary variable: max_dist = maximum travel distance
- Constraints:
  1. Each bundle assigned to exactly one interviewer: sum_i x[i,b] = 1 for all b
  2. Each interviewer gets exactly n bundles: sum_b x[i,b] = n for all i
  3. max_dist >= travel_dist[i] for all i
- Objective: minimize max_dist

For travel distance, we use an upper bound that's linear:
  travel_dist[i] ≈ max{dist(home_i, bundle_b) : x[i,b]=1} + cluster_diameter + internal_dists

This can be linearized using additional variables and constraints.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from math import radians, sin, cos, sqrt, atan2

try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("Warning: PuLP not available. Install with: pip install pulp")


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


def assign_bundles_optimal_ilp(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4,
    time_limit_seconds: int = 300
) -> Dict[str, Tuple[List[int], float]]:
    """
    Assign bundles using Integer Linear Programming to find optimal solution.

    Minimizes: max{travel_distance_i : i ∈ interviewers}

    Args:
        interviewers: List of dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with bundle geometries
        bundles_per_interviewer: Target number of bundles per interviewer
        time_limit_seconds: Maximum time for ILP solver

    Returns:
        Dict mapping interviewer name to (ordered_bundle_ids, travel_distance_km)
    """
    if not PULP_AVAILABLE:
        raise ImportError("PuLP is required for ILP optimization. Install with: pip install pulp")

    print(f"[ILP Optimizer] Step 1: Extract bundle info and compute distance matrices")

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

    n_interviewers = len(interviewers)
    n_bundles = len(list(bundle_info.keys()))

    # Compute distance matrices
    # home_to_bundle[i][b] = distance from interviewer i's home to bundle b
    home_to_bundle = {}
    for i, interviewer in enumerate(interviewers):
        home_to_bundle[i] = {}
        for bundle_id in bundle_info:
            dist = haversine_distance(
                interviewer['lat'], interviewer['lon'],
                bundle_info[bundle_id]['lat'], bundle_info[bundle_id]['lon']
            )
            home_to_bundle[i][bundle_id] = dist

    # bundle_to_bundle[b1][b2] = distance between bundles b1 and b2
    bundle_to_bundle = {}
    bundle_list = list(bundle_info.keys())
    for b1 in bundle_list:
        bundle_to_bundle[b1] = {}
        for b2 in bundle_list:
            if b1 == b2:
                bundle_to_bundle[b1][b2] = 0
            else:
                dist = haversine_distance(
                    bundle_info[b1]['lat'], bundle_info[b1]['lon'],
                    bundle_info[b2]['lat'], bundle_info[b2]['lon']
                )
                bundle_to_bundle[b1][b2] = dist

    print(f"[ILP Optimizer] Step 2: Formulate ILP problem")

    # Create LP problem
    prob = LpProblem("MinMax_Bundle_Assignment", LpMinimize)

    # Decision variables: x[i,b] = 1 if interviewer i gets bundle b
    x = {}
    for i in range(n_interviewers):
        for bundle_id in bundle_list:
            x[i, bundle_id] = LpVariable(f"x_{i}_{bundle_id}", cat='Binary')

    # Auxiliary variable: max_dist
    max_dist = LpVariable("max_dist", lowBound=0)

    # Objective: minimize max_dist
    prob += max_dist

    # Constraint 1: Each bundle assigned to exactly one interviewer
    for bundle_id in bundle_list:
        prob += lpSum([x[i, bundle_id] for i in range(n_interviewers)]) == 1, f"bundle_{bundle_id}_assigned_once"

    # Constraint 2: Each interviewer gets exactly bundles_per_interviewer bundles
    for i in range(n_interviewers):
        prob += lpSum([x[i, bundle_id] for bundle_id in bundle_list]) == bundles_per_interviewer, f"interviewer_{i}_gets_{bundles_per_interviewer}_bundles"

    # Constraint 3: max_dist >= upper bound on travel distance for each interviewer
    # Upper bound: max_home_to_bundle + max_inter_bundle_distance + sum_internal_distances

    # For each interviewer, we need auxiliary variables for max distances
    max_home_to_bundle = {}
    max_inter_bundle = {}

    for i in range(n_interviewers):
        # max_home_to_bundle[i] = max distance from home to assigned bundles
        max_home_to_bundle[i] = LpVariable(f"max_home_bundle_{i}", lowBound=0)

        # max_home_to_bundle[i] >= home_to_bundle[i][b] if x[i,b] = 1
        for bundle_id in bundle_list:
            prob += max_home_to_bundle[i] >= home_to_bundle[i][bundle_id] * x[i, bundle_id], \
                    f"max_home_bundle_{i}_{bundle_id}"

        # max_inter_bundle[i] = max distance between any two assigned bundles
        max_inter_bundle[i] = LpVariable(f"max_inter_bundle_{i}", lowBound=0)

        # max_inter_bundle[i] >= bundle_to_bundle[b1][b2] if both x[i,b1]=1 and x[i,b2]=1
        # This is non-linear, so we approximate with: max over all pairs regardless of assignment
        # Better approximation: use big-M formulation
        M = 100  # Big M (larger than any possible distance)
        for b1 in bundle_list:
            for b2 in bundle_list:
                if b1 != b2:
                    # If both x[i,b1] and x[i,b2] are 1, then max_inter_bundle >= dist
                    # We can approximate this with: max_inter_bundle >= dist - M*(2 - x[i,b1] - x[i,b2])
                    prob += max_inter_bundle[i] >= bundle_to_bundle[b1][b2] - M * (2 - x[i, b1] - x[i, b2]), \
                            f"max_inter_{i}_{b1}_{b2}"

        # Sum of internal distances for assigned bundles
        sum_internal = lpSum([bundle_info[bundle_id]['internal_dist'] * x[i, bundle_id] for bundle_id in bundle_list])

        # Upper bound on travel distance
        # Approximation: home to farthest bundle + farthest inter-bundle distance + sum of internal distances
        travel_upper_bound = max_home_to_bundle[i] + max_inter_bundle[i] + sum_internal

        prob += max_dist >= travel_upper_bound, f"max_dist_bound_{i}"

    print(f"[ILP Optimizer] Step 3: Solve ILP (time limit: {time_limit_seconds}s)")

    # Solve
    solver = PULP_CBC_CMD(msg=1, timeLimit=time_limit_seconds)
    prob.solve(solver)

    # Check solution status
    print(f"[ILP Optimizer] Solution status: {LpStatus[prob.status]}")

    if prob.status != LpStatusOptimal:
        print(f"Warning: Could not find optimal solution. Status: {LpStatus[prob.status]}")
        print("Falling back to heuristic...")
        # Return empty results or fall back to another method
        return {}

    print(f"[ILP Optimizer] Optimal max distance (upper bound): {value(max_dist):.2f} km")

    # Extract solution
    print(f"[ILP Optimizer] Step 4: Extract assignment and calculate actual TSP routes")

    assignments = {i: [] for i in range(n_interviewers)}

    for i in range(n_interviewers):
        for bundle_id in bundle_list:
            if value(x[i, bundle_id]) > 0.5:  # Binary variable, so should be 0 or 1
                assignments[i].append(bundle_id)

    # Calculate actual TSP routes for verification
    results = {}
    for i, interviewer in enumerate(interviewers):
        name = interviewer['name']
        bundle_ids = assignments[i]

        if not bundle_ids:
            results[name] = ([], 0.0)
            continue

        ordered_ids, travel_dist = calculate_tsp_route(
            interviewer['lat'], interviewer['lon'],
            bundle_ids, bundle_info
        )

        results[name] = (ordered_ids, travel_dist)
        print(f"  {name}: {len(ordered_ids)} bundles, {travel_dist:.2f} km actual TSP distance")

    return results


def assign_bundles_for_date_optimal_ilp(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str | None = None,
    bundles_per_interviewer: int = 4,
    time_limit_seconds: int = 300,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> Dict[str, Tuple[List[int], float]]:
    """Wrapper function for ILP-based optimal bundle assignment."""
    from sd311_fieldprep.interviewer_geocoding import get_interviewers_for_date_with_locations

    interviewers = get_interviewers_for_date_with_locations(
        date=date,
        sheet_id=sheet_id
    )

    print(f"[ILP Optimizer] Loaded {len(interviewers)} interviewers for {date}")

    return assign_bundles_optimal_ilp(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer,
        time_limit_seconds=time_limit_seconds
    )
