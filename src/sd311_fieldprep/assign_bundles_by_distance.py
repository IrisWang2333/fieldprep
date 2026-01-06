#!/usr/bin/env python
"""
Assign Bundles to Interviewers Based on Distance

This module optimizes bundle assignments by minimizing total travel distance:
1. Distance from interviewer's home to bundle centroid
2. Distance between segments within bundle (travel during survey)

Uses Hungarian algorithm (linear assignment) for optimal matching.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')


def geocode_address(address: str) -> Tuple[float, float]:
    """
    Geocode address to (lat, lon) coordinates.

    For now, uses a simple geocoding approach. In production, use Google Maps API.

    Args:
        address: Street address

    Returns:
        (latitude, longitude) tuple
    """
    # TODO: Implement actual geocoding (Google Maps API or similar)
    # Placeholder: return San Diego center
    print(f"Warning: Using placeholder geocoding for {address}")
    return (32.7157, -117.1611)


def get_bundle_centroid(bundle_df: gpd.GeoDataFrame) -> Tuple[float, float]:
    """
    Calculate bundle centroid from segment geometries.

    Args:
        bundle_df: GeoDataFrame of segments in bundle

    Returns:
        (latitude, longitude) of centroid
    """
    # Union all segments and get centroid
    union_geom = bundle_df.geometry.unary_union
    centroid = union_geom.centroid

    # Convert from projected CRS to lat/lon if needed
    if bundle_df.crs and not bundle_df.crs.is_geographic:
        # Reproject to WGS84
        bundle_gdf = gpd.GeoDataFrame([{'geometry': centroid}], crs=bundle_df.crs)
        bundle_gdf = bundle_gdf.to_crs('EPSG:4326')
        centroid = bundle_gdf.geometry.iloc[0]

    return (centroid.y, centroid.x)  # (lat, lon)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Haversine distance between two points in kilometers.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth radius in km

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def calculate_bundle_internal_distance(bundle_df: gpd.GeoDataFrame) -> float:
    """
    Estimate total distance traveled within bundle.

    Uses simple heuristic: sum of segment lengths (proxy for travel during survey).

    Args:
        bundle_df: GeoDataFrame of segments in bundle

    Returns:
        Total internal distance in kilometers
    """
    # If in projected CRS (meters), convert to km
    if bundle_df.crs and not bundle_df.crs.is_geographic:
        total_length_m = bundle_df.geometry.length.sum()
        return total_length_m / 1000.0  # meters to km
    else:
        # If in lat/lon, approximate
        # Rough approximation: 1 degree ≈ 111 km at equator
        total_length_deg = bundle_df.geometry.length.sum()
        return total_length_deg * 111.0


def calculate_cost_matrix(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    alpha: float = 0.7,
    beta: float = 0.3
) -> np.ndarray:
    """
    Calculate cost matrix for assignment problem.

    Cost = alpha * (home to bundle distance) + beta * (bundle internal distance)

    Args:
        interviewers: List of interviewer dicts with 'name', 'lat', 'lon'
        bundles: List of bundle IDs
        bundles_gdf: GeoDataFrame with all bundles
        alpha: Weight for home-to-bundle distance (default: 0.7)
        beta: Weight for internal bundle distance (default: 0.3)

    Returns:
        Cost matrix (n_interviewers x n_bundles)
    """
    n_interviewers = len(interviewers)
    n_bundles = len(bundles)

    cost_matrix = np.zeros((n_interviewers, n_bundles))

    # Calculate bundle centroids and internal distances
    bundle_info = {}
    for bundle_id in bundles:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        centroid_lat, centroid_lon = get_bundle_centroid(bundle_df)
        internal_dist = calculate_bundle_internal_distance(bundle_df)
        bundle_info[bundle_id] = {
            'centroid': (centroid_lat, centroid_lon),
            'internal_dist': internal_dist
        }

    # Fill cost matrix
    for i, interviewer in enumerate(interviewers):
        interviewer_lat = interviewer['lat']
        interviewer_lon = interviewer['lon']

        for j, bundle_id in enumerate(bundles):
            # Distance from home to bundle
            bundle_lat, bundle_lon = bundle_info[bundle_id]['centroid']
            home_to_bundle = haversine_distance(
                interviewer_lat, interviewer_lon,
                bundle_lat, bundle_lon
            )

            # Bundle internal distance
            internal_dist = bundle_info[bundle_id]['internal_dist']

            # Total cost
            cost_matrix[i, j] = alpha * home_to_bundle + beta * internal_dist

    return cost_matrix


def assign_bundles_optimally(
    interviewers: List[Dict],
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    bundles_per_interviewer: int = 4,
    alpha: float = 0.7,
    beta: float = 0.3
) -> Dict[str, List[int]]:
    """
    Assign bundles to interviewers using Hungarian algorithm.

    Args:
        interviewers: List of interviewer dicts
        bundles: List of available bundle IDs
        bundles_gdf: GeoDataFrame with all bundles
        bundles_per_interviewer: Number of bundles per interviewer (default: 4)
        alpha: Weight for home-to-bundle distance
        beta: Weight for internal bundle distance

    Returns:
        Dict mapping interviewer name to list of assigned bundle IDs
    """
    n_interviewers = len(interviewers)
    n_bundles = len(bundles)

    if n_bundles != n_interviewers * bundles_per_interviewer:
        raise ValueError(
            f"Number of bundles ({n_bundles}) must equal "
            f"number of interviewers ({n_interviewers}) × "
            f"bundles_per_interviewer ({bundles_per_interviewer})"
        )

    # Calculate cost matrix
    cost_matrix = calculate_cost_matrix(interviewers, bundles, bundles_gdf, alpha, beta)

    # Repeat each interviewer bundles_per_interviewer times
    # This allows each interviewer to be assigned multiple bundles
    expanded_cost_matrix = np.repeat(cost_matrix, bundles_per_interviewer, axis=0)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(expanded_cost_matrix)

    # Map assignments back to interviewers
    assignments = {interviewer['name']: [] for interviewer in interviewers}

    for row, col in zip(row_ind, col_ind):
        interviewer_idx = row // bundles_per_interviewer
        interviewer_name = interviewers[interviewer_idx]['name']
        bundle_id = bundles[col]
        assignments[interviewer_name].append(bundle_id)

    return assignments


def load_interviewer_data(
    geocoded_file: str = None,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> pd.DataFrame:
    """
    Load interviewer data from geocoded CSV or Google Sheets.

    Args:
        geocoded_file: Path to geocoded CSV file (if None, will geocode from Google Sheets)
        sheet_id: Google Sheet ID

    Returns:
        DataFrame with interviewer info (name, email, address, lat, lon)
    """
    if geocoded_file and Path(geocoded_file).exists():
        # Load from cached geocoded file
        return pd.read_csv(geocoded_file)

    # Load from Google Sheets and geocode
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    import time

    interviewers_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=42380463'
    interviewers_df = pd.read_csv(interviewers_url)

    # Geocode addresses
    geolocator = Nominatim(user_agent="sd311_fieldprep")
    geocoded_results = []

    for idx, row in interviewers_df.iterrows():
        name = row['Name']
        address = row['Home Address']

        if pd.isna(address):
            geocoded_results.append({
                'name': name,
                'email': row['Email'],
                'address': None,
                'lat': None,
                'lon': None
            })
            continue

        try:
            time.sleep(1)  # Rate limiting
            location = geolocator.geocode(address, timeout=10)

            if location:
                geocoded_results.append({
                    'name': name,
                    'email': row['Email'],
                    'address': address,
                    'lat': location.latitude,
                    'lon': location.longitude
                })
            else:
                geocoded_results.append({
                    'name': name,
                    'email': row['Email'],
                    'address': address,
                    'lat': None,
                    'lon': None
                })
        except GeocoderTimedOut:
            geocoded_results.append({
                'name': name,
                'email': row['Email'],
                'address': address,
                'lat': None,
                'lon': None
            })

    return pd.DataFrame(geocoded_results)


def get_interviewers_for_date(
    date: str,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0'
) -> List[str]:
    """
    Get list of interviewer names assigned for a specific date.

    Args:
        date: Date string in YYYY-MM-DD format
        sheet_id: Google Sheet ID

    Returns:
        List of interviewer names (length 6, positions A-F)
    """
    assignments_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0'
    assignments_df = pd.read_csv(assignments_url)

    # Find row matching the date
    matching_rows = assignments_df[assignments_df['Date'] == date]

    if len(matching_rows) == 0:
        raise ValueError(f"No assignment found for date {date}")

    row = matching_rows.iloc[0]

    # Extract interviewers for positions A-F
    return [row['A'], row['B'], row['C'], row['D'], row['E'], row['F']]


def assign_bundles_for_date(
    date: str,
    bundles: List[int],
    bundles_gdf: gpd.GeoDataFrame,
    geocoded_file: str = None,
    bundles_per_interviewer: int = 4,
    alpha: float = 0.7,
    beta: float = 0.3
) -> Dict[str, List[int]]:
    """
    Assign bundles to interviewers for a specific date based on distance optimization.

    Args:
        date: Date string in YYYY-MM-DD format
        bundles: List of bundle IDs to assign
        bundles_gdf: GeoDataFrame with all bundles
        geocoded_file: Path to geocoded interviewers CSV (optional)
        bundles_per_interviewer: Number of bundles per interviewer
        alpha: Weight for home-to-bundle distance
        beta: Weight for bundle internal distance

    Returns:
        Dict mapping interviewer name to list of assigned bundle IDs
    """
    # Get interviewers assigned for this date
    interviewer_names = get_interviewers_for_date(date)
    print(f"Interviewers for {date}: {interviewer_names}")

    # Load geocoded interviewer data
    all_interviewers_df = load_interviewer_data(geocoded_file=geocoded_file)

    # Filter to only those assigned for this date
    assigned_interviewers = all_interviewers_df[
        all_interviewers_df['name'].isin(interviewer_names)
    ].copy()

    # Drop interviewers without coordinates
    assigned_interviewers = assigned_interviewers.dropna(subset=['lat', 'lon'])

    if len(assigned_interviewers) != 6:
        print(f"\nWarning: Only {len(assigned_interviewers)}/6 interviewers have geocoded addresses")
        print(f"Missing: {set(interviewer_names) - set(assigned_interviewers['name'])}")

    # Convert to list of dicts
    interviewers = assigned_interviewers[['name', 'email', 'lat', 'lon']].to_dict('records')

    # Assign bundles optimally
    return assign_bundles_optimally(
        interviewers=interviewers,
        bundles=bundles,
        bundles_gdf=bundles_gdf,
        bundles_per_interviewer=bundles_per_interviewer,
        alpha=alpha,
        beta=beta
    )


def main():
    """
    Example usage: Assign bundles for Jan 10, 2026 based on Google Sheets data.
    """
    # Load bundles
    bundle_file = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
    bundles_gdf = gpd.read_parquet(bundle_file)

    seg_col = 'iamfloc' if 'iamfloc' in bundles_gdf.columns else 'segment_id'
    bundles_gdf['segment_id'] = bundles_gdf[seg_col].astype(str)

    # Example: Week 1 DH bundles from plan (24 bundles for 6 interviewers = 4 each)
    example_bundles = [5313, 3061, 3262, 1175, 5506, 2330, 1918, 2636,
                       3630, 1571, 2181, 1709, 3106, 3017, 5681, 3406,
                       2167, 486, 1601, 1324, 1928, 5821, 1013, 1544]

    # Assign bundles for Jan 10, 2026
    target_date = "2026-01-10"
    geocoded_file = "data/interviewers_geocoded.csv"

    print(f"=== Assigning Bundles for {target_date} ===\n")

    try:
        assignments = assign_bundles_for_date(
            date=target_date,
            bundles=example_bundles,
            bundles_gdf=bundles_gdf,
            geocoded_file=geocoded_file,
            bundles_per_interviewer=4,
            alpha=0.7,  # 70% weight on home-to-bundle distance
            beta=0.3    # 30% weight on bundle internal distance
        )

        print("\n=== Optimized Bundle Assignments ===")

        # Load all interviewer data for distance calculation
        all_interviewers_df = load_interviewer_data(geocoded_file=geocoded_file)

        for interviewer_name, bundles in sorted(assignments.items()):
            print(f"\n{interviewer_name}: {bundles}")

            # Get interviewer info
            int_row = all_interviewers_df[all_interviewers_df['name'] == interviewer_name].iloc[0]

            total_dist = 0
            for bundle_id in bundles:
                bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
                centroid_lat, centroid_lon = get_bundle_centroid(bundle_df)
                home_to_bundle = haversine_distance(
                    int_row['lat'], int_row['lon'],
                    centroid_lat, centroid_lon
                )
                internal_dist = calculate_bundle_internal_distance(bundle_df)
                total_dist += home_to_bundle + internal_dist

            print(f"  Home address: {int_row['address']}")
            print(f"  Total travel distance: {total_dist:.2f} km")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
