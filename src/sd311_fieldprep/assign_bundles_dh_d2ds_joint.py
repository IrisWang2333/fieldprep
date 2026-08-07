#!/usr/bin/env python3
"""
Joint DH+D2DS Bundle Assignment Optimization (Two-Phase)

Assigns DH and D2DS bundles together using a two-phase optimization:

Phase 1: Minimize maximum travel distance (minimax)
  - Ensures no interviewer has excessively long travel
  - Uses iterative swaps to reduce the longest distance

Phase 2: Minimize total travel distance (while preserving max)
  - Improves overall fairness and efficiency
  - Only accepts swaps that don't increase the maximum distance
  - Reduces total system-wide travel

Distance = Home → DH centroid + DH centroid → D2DS centroid
"""

import itertools
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from math import radians, sin, cos, sqrt, atan2


# Interviewer pairs who carpool (one drives the other) and must therefore start
# the day from geographically adjacent locations. Names must match the roster /
# Interviewers sheet exactly (case-sensitive).
#
# DISABLED 2026-08-07 per request: the carpool packing pulled the pair (and,
# by taking near-home bundles, other interviewers too) far from their homes on
# spread-out draws, so we reverted to normal 6-person assignment. To re-enable,
# add the pair back, e.g. [("Vicky", "Veronica")].
CARPOOL_PAIRS: List[Tuple[str, str]] = []


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


def _build_bundle_info(bundle_ids, bundles_gdf: gpd.GeoDataFrame) -> Dict:
    """Build {bundle_id: {'lat', 'lon'}} centroids for the given bundle IDs."""
    info = {}
    for bundle_id in bundle_ids:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        if len(bundle_df) == 0:
            print(f"Warning: Bundle {bundle_id} not found when building centroids")
            continue
        lat, lon = get_bundle_centroid(bundle_df)
        info[bundle_id] = {'lat': lat, 'lon': lon}
    return info


def _apply_carpool_constraint(
    result: Dict[str, Tuple[int, int, float]],
    interviewers: List[Dict],
    dh_bundles: List[int],
    d2ds_bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    carpool_pairs: List[Tuple[str, str]] = None,
) -> Dict[str, Tuple[int, int, float]]:
    """
    Re-label the normally-optimized (DH, D2DS) assignments so that a carpool pair
    (one interviewer drives the other) receives the two assignments that minimize
    the average of their DH-to-DH and D2DS-to-D2DS starting-point distances.

    The pair share transportation for the whole day (DH first, then D2DS), so both
    the DH and the D2DS start points should be close. For each candidate pair of
    assignments we take avg(dist(DH_a, DH_b), dist(D2DS_a, D2DS_b)) and pick the
    minimum. The DH-D2DS groupings produced by the normal optimization are preserved
    as fixed units; only which interviewer holds which unit changes. Within the two
    carpool units the orientation is chosen to minimize the pair's own home travel,
    and the other interviewers are re-assigned the remaining units minimizing
    (max, then total) home travel.

    This is a pure re-labeling of already-drawn bundles: the set of DH/D2DS bundles
    and their conditional/random treatment status is unchanged, so it does not
    affect the experiment's treatment composition.
    """
    carpool_pairs = CARPOOL_PAIRS if carpool_pairs is None else carpool_pairs
    names_present = {iv['name'] for iv in interviewers}
    active = [p for p in carpool_pairs if p[0] in names_present and p[1] in names_present]

    if not active:
        return result

    pair = active[0]
    if len(active) > 1:
        print(f"[Carpool] Multiple carpool pairs present; applying only {pair}, ignoring {active[1:]}")
    c1, c2 = pair
    print(f"\n[Carpool] Applying DH-proximity constraint for carpool pair {c1} & {c2}")

    home = {iv['name']: (iv['lat'], iv['lon']) for iv in interviewers}
    bundle_info = _build_bundle_info(set(dh_bundles) | set(d2ds_bundles), bundles_gdf)

    # Fixed (DH, D2DS) units from the normal optimization
    names = list(result.keys())
    unit_list = [(result[name][0], result[name][1]) for name in names]  # (dh, d2ds)

    # Find the two units minimizing the average of the pair's DH-to-DH and
    # D2DS-to-D2DS starting-point distances.
    best = None
    for i in range(len(unit_list)):
        for j in range(i + 1, len(unit_list)):
            dh_i, dh_j = unit_list[i][0], unit_list[j][0]
            d2_i, d2_j = unit_list[i][1], unit_list[j][1]
            if any(b not in bundle_info for b in (dh_i, dh_j, d2_i, d2_j)):
                continue
            dh_d = haversine_distance(
                bundle_info[dh_i]['lat'], bundle_info[dh_i]['lon'],
                bundle_info[dh_j]['lat'], bundle_info[dh_j]['lon'],
            )
            d2_d = haversine_distance(
                bundle_info[d2_i]['lat'], bundle_info[d2_i]['lon'],
                bundle_info[d2_j]['lat'], bundle_info[d2_j]['lon'],
            )
            avg_d = (dh_d + d2_d) / 2
            if best is None or avg_d < best[0]:
                best = (avg_d, i, j, dh_d, d2_d)

    if best is None:
        print("[Carpool] Could not compute starting-point distances; leaving assignment unchanged")
        return result

    avg_dist, i, j, dh_dist, d2_dist = best
    carpool_units = [unit_list[i], unit_list[j]]
    remaining_units = [u for k, u in enumerate(unit_list) if k not in (i, j)]
    other_names = [name for name in names if name not in (c1, c2)]

    print(f"[Carpool] Min avg(DH, D2DS) pair -> {c1} & {c2}: "
          f"DH {carpool_units[0][0]}&{carpool_units[1][0]} ({dh_dist:.2f} km), "
          f"D2DS {carpool_units[0][1]}&{carpool_units[1][1]} ({d2_dist:.2f} km), "
          f"avg {avg_dist:.2f} km")

    def travel(name, unit):
        h = home[name]
        return calculate_travel_distance(h[0], h[1], unit[0], unit[1], bundle_info)

    # Orient the two carpool units between c1 and c2 to minimize their home travel
    if (travel(c1, carpool_units[0]) + travel(c2, carpool_units[1]) <=
            travel(c1, carpool_units[1]) + travel(c2, carpool_units[0])):
        c1_unit, c2_unit = carpool_units[0], carpool_units[1]
    else:
        c1_unit, c2_unit = carpool_units[1], carpool_units[0]

    # Assign remaining units to the other interviewers, minimizing (max, then total)
    best_perm = None
    for perm in itertools.permutations(remaining_units):
        dists = [travel(other_names[k], perm[k]) for k in range(len(other_names))]
        key = (max(dists) if dists else 0.0, sum(dists))
        if best_perm is None or key < best_perm[0]:
            best_perm = (key, perm, dists)

    final = {
        c1: (c1_unit[0], c1_unit[1], travel(c1, c1_unit)),
        c2: (c2_unit[0], c2_unit[1], travel(c2, c2_unit)),
    }
    if other_names:
        _, perm, dists = best_perm
        for k, name in enumerate(other_names):
            final[name] = (perm[k][0], perm[k][1], dists[k])

    print(f"[Carpool] Final assignment after carpool constraint:")
    for name in sorted(final.keys()):
        dh_id, d2ds_id, dist = final[name]
        tag = " <carpool>" if name in (c1, c2) else ""
        print(f"  {name}: DH={dh_id}, D2DS={d2ds_id}, dist={dist:.2f} km{tag}")

    return final


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

    print(f"[Joint Assignment] Phase 1 completed after {iteration} iterations")
    print(f"[Joint Assignment] Phase 1 final max travel: {current_max:.2f} km")

    # ============================================================================
    # Phase 2: Minimize total distance without increasing max
    # ============================================================================
    print(f"\n[Joint Assignment] Step 4: Phase 2 - Minimize total distance (preserving max)")

    current_total = sum(travel_distances.values())
    print(f"[Joint Assignment] Phase 1 total distance: {current_total:.2f} km")

    phase2_improved = True
    phase2_iteration = 0
    phase2_max_iterations = max_refine_iterations  # Same limit as Phase 1

    while phase2_improved and phase2_iteration < phase2_max_iterations:
        phase2_improved = False
        phase2_iteration += 1

        # Try all pairs of interviewers
        for i in range(len(interviewers)):
            for j in range(i+1, len(interviewers)):
                name1 = interviewers[i]['name']
                name2 = interviewers[j]['name']

                interviewer1 = interviewers[i]
                interviewer2 = interviewers[j]

                # Try both swap types
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

                    # Calculate new distances
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

                    # Calculate new max and total
                    new_max = max(new_dist1, new_dist2,
                                 *[travel_distances[n] for n in travel_distances if n not in [name1, name2]])
                    new_total = new_dist1 + new_dist2 + sum(travel_distances[n] for n in travel_distances if n not in [name1, name2])

                    # Accept swap if: max doesn't increase AND total decreases
                    if new_max <= current_max and new_total < current_total:
                        # Keep the swap
                        old_total = current_total
                        travel_distances[name1] = new_dist1
                        travel_distances[name2] = new_dist2
                        current_total = new_total
                        phase2_improved = True

                        max_change = "maintained" if new_max == current_max else f"reduced to {new_max:.2f}"
                        print(f"  Phase 2 Iteration {phase2_iteration}: Swapped {type1} bundles {orig_bundle1} ↔ {orig_bundle2} ({name1} ↔ {name2})")
                        print(f"    Total: {new_total:.2f} km (saved {(old_total - new_total):.2f} km), Max: {max_change}")

                        # Update max if it decreased
                        if new_max < current_max:
                            current_max = new_max

                        break  # Move to next pair after successful swap
                    else:
                        # Undo swap
                        assignments[name1][type1] = orig_bundle1
                        assignments[name2][type2] = orig_bundle2

                if phase2_improved:
                    break

            if phase2_improved:
                break

    print(f"[Joint Assignment] Phase 2 completed after {phase2_iteration} iterations")
    print(f"[Joint Assignment] Final max travel: {current_max:.2f} km")
    print(f"[Joint Assignment] Final total travel: {current_total:.2f} km")

    print(f"\n[Joint Assignment] Final assignment:")
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
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0',
    carpool_pairs: List[Tuple[str, str]] = None,
) -> Dict[str, Tuple[int, int, float]]:
    """Wrapper function for joint DH+D2DS assignment.

    After the normal travel optimization, any configured carpool pair (see
    CARPOOL_PAIRS) that is present on this date is re-assigned the two DH
    starting points closest to each other, so the driver and passenger begin
    the day in the same area.
    """
    from sd311_fieldprep.interviewer_geocoding import get_interviewers_for_date_with_locations

    interviewers = get_interviewers_for_date_with_locations(
        date=date,
        sheet_id=sheet_id
    )

    print(f"[Joint Assignment] Loaded {len(interviewers)} interviewers for {date}")

    result = assign_bundles_dh_d2ds_joint(
        interviewers=interviewers,
        dh_bundles=dh_bundles,
        d2ds_bundles=d2ds_bundles,
        bundles_gdf=bundles_gdf,
        max_refine_iterations=max_refine_iterations
    )

    result = _apply_carpool_constraint(
        result=result,
        interviewers=interviewers,
        dh_bundles=dh_bundles,
        d2ds_bundles=d2ds_bundles,
        bundles_gdf=bundles_gdf,
        carpool_pairs=carpool_pairs,
    )

    return result
