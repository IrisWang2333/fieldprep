#!/usr/bin/env python
"""
Compare different cost functions for Stage 1 assignment.

This demonstrates why the simplified cost function is an approximation
and shows the difference between:
1. Simplified cost (current implementation)
2. Actual routing distance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import geopandas as gpd
import pandas as pd
from sd311_fieldprep.assign_bundles_by_distance import (
    get_bundle_centroid,
    haversine_distance,
    calculate_bundle_internal_distance,
    load_interviewer_data
)


def calculate_simplified_cost(
    interviewer_lat, interviewer_lon,
    bundle_ids, bundles_gdf,
    alpha=0.7, beta=0.3
):
    """
    Simplified cost: Σ[α×d(Home,Bundle) + β×internal(Bundle)]

    This assumes each bundle is visited independently from home.
    """
    total_cost = 0

    for bundle_id in bundle_ids:
        bundle_df = bundles_gdf[bundles_gdf['bundle_id'] == bundle_id]
        centroid_lat, centroid_lon = get_bundle_centroid(bundle_df)
        internal_dist = calculate_bundle_internal_distance(bundle_df)

        # Distance from home to bundle
        home_to_bundle = haversine_distance(
            interviewer_lat, interviewer_lon,
            centroid_lat, centroid_lon
        )

        # Simplified cost
        total_cost += alpha * home_to_bundle + beta * internal_dist

    return total_cost


def calculate_actual_routing_distance(
    interviewer_lat, interviewer_lon,
    bundle_ids, bundles_gdf
):
    """
    Actual routing distance: Home → B₁ → B₂ → ... → Bₙ → Home
    (assuming optimal TSP order)
    """
    from sd311_fieldprep.assign_bundles_with_routing import calculate_route_distance

    optimal_order, total_distance = calculate_route_distance(
        interviewer_lat, interviewer_lon,
        bundle_ids, bundles_gdf
    )

    return total_distance, optimal_order


def main():
    # Load data
    bundle_file = Path("outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
    bundles_gdf = gpd.read_parquet(bundle_file)

    geocoded_file = Path("data/interviewers_geocoded.csv")
    interviewers_df = load_interviewer_data(str(geocoded_file))

    # Test with actual assignment from 2026-01-10
    test_cases = [
        ("Carlie", [5693, 2359, 3631, 2046]),
        ("Rene", [1940, 3025, 3406, 5418]),
        ("David", [6011, 5103, 5501, 5602]),
        ("Jessica D.", [5110, 5503, 2194, 5672]),
        ("Jessica A.", [5830, 2671, 2221, 5699]),
        ("Veronica", [1326, 1715, 485, 3383]),
    ]

    print("=" * 80)
    print("COMPARISON: Simplified Cost vs Actual Routing Distance")
    print("=" * 80)
    print()

    total_simplified = 0
    total_actual = 0

    for interviewer_name, bundle_ids in test_cases:
        # Get interviewer info
        interviewer = interviewers_df[interviewers_df['name'] == interviewer_name].iloc[0]
        lat, lon = interviewer['lat'], interviewer['lon']

        # Calculate simplified cost (α=0.7, β=0.3)
        simplified_cost = calculate_simplified_cost(
            lat, lon, bundle_ids, bundles_gdf,
            alpha=0.7, beta=0.3
        )

        # Calculate actual routing distance
        actual_distance, optimal_order = calculate_actual_routing_distance(
            lat, lon, bundle_ids, bundles_gdf
        )

        # Calculate error
        error = simplified_cost - actual_distance
        error_pct = (error / actual_distance) * 100

        print(f"{interviewer_name}:")
        print(f"  Assigned bundles: {bundle_ids}")
        print(f"  Optimal route:    {optimal_order}")
        print(f"  Simplified cost:  {simplified_cost:.2f} km")
        print(f"  Actual distance:  {actual_distance:.2f} km")
        print(f"  Error:            {error:+.2f} km ({error_pct:+.1f}%)")
        print()

        total_simplified += simplified_cost
        total_actual += actual_distance

    print("=" * 80)
    print(f"TOTAL (all 6 interviewers):")
    print(f"  Simplified cost:  {total_simplified:.2f} km")
    print(f"  Actual distance:  {total_actual:.2f} km")
    print(f"  Error:            {total_simplified - total_actual:+.2f} km")
    print(f"  Error percentage: {((total_simplified - total_actual) / total_actual) * 100:+.1f}%")
    print("=" * 80)
    print()
    print("INTERPRETATION:")
    print()
    print("The simplified cost function overestimates the actual distance because:")
    print("- It counts α×d(Home, Bundle) for EACH bundle")
    print("- In reality, you only go Home→First_Bundle once, and Last_Bundle→Home once")
    print("- The middle segments are Bundle→Bundle, which are often shorter than Bundle→Home")
    print()
    print("However, the simplified cost is still useful for Stage 1 assignment because:")
    print("- It correctly identifies which bundles are 'near' vs 'far' from each interviewer")
    print("- Hungarian algorithm uses it to assign nearby bundles to each interviewer")
    print("- The relative ordering is what matters, not the absolute value")
    print()


if __name__ == '__main__':
    main()
