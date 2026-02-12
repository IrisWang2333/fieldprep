#!/usr/bin/env python
"""
Infer historical segment assignments from actual Google Drive field files.

This script uses ONLY the files from Google Drive (google_drive_actual folder)
to determine the true segment-level treatment assignments.
"""
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def infer_from_google_drive_files(date, google_drive_dir, bundles_gdf):
    """
    Infer segment assignments directly from Google Drive field files.

    Parameters
    ----------
    date : str
        Date in YYYY-MM-DD format
    google_drive_dir : Path
        Path to google_drive_actual/DATE folder
    bundles_gdf : GeoDataFrame
        Bundle definitions with segment_id

    Returns
    -------
    pd.DataFrame
        Inferred segment assignments
    """
    print(f"\n{'='*70}")
    print(f"Inferring segment assignments for {date}")
    print(f"Using Google Drive files from: {google_drive_dir}")
    print(f"{'='*70}")

    sfh_points_file = google_drive_dir / "sfh_points.csv"

    if not sfh_points_file.exists():
        print(f"  ERROR: sfh_points.csv not found in {google_drive_dir}")
        return None

    # Read actual field file (what was given to field workers)
    sfh_points = pd.read_csv(sfh_points_file)
    print(f"  Loaded {len(sfh_points)} addresses from Google Drive sfh_points.csv")

    # Get DH addresses
    dh_addresses = sfh_points[sfh_points['task'] == 'DH'].copy()
    print(f"  DH addresses: {len(dh_addresses)}")

    if len(dh_addresses) == 0:
        print("  No DH addresses found")
        return None

    # Get unique DH segments and bundles
    dh_segments = set(dh_addresses['segment_id'].unique())
    print(f"  DH segments: {len(dh_segments)}")

    # Find which bundles these segments belong to
    dh_bundle_segments = bundles_gdf[bundles_gdf['segment_id'].isin(dh_segments)].copy()
    dh_bundles = sorted(dh_bundle_segments['bundle_id'].unique())
    print(f"  DH bundles (inferred from segments): {dh_bundles}")

    # Count addresses per segment in DH task
    treated_by_segment = dh_addresses.groupby('segment_id').size().to_dict()

    # Get all segments in all DH bundles
    all_dh_segments = bundles_gdf[bundles_gdf['bundle_id'].isin(dh_bundles)]['segment_id'].unique()
    print(f"  Total segments in DH bundles: {len(all_dh_segments)}")

    # Count total addresses per segment from bundle data
    # USE sfh_addr_count (segment-level), NOT bundle_addr_total (bundle-level)!
    segment_total_addresses = bundles_gdf.groupby('segment_id')['sfh_addr_count'].first().to_dict()

    # Infer assignments
    segment_assignments = []

    for segment_id in all_dh_segments:
        treated_addrs = treated_by_segment.get(segment_id, 0)
        total_addrs = segment_total_addresses.get(segment_id, 0)

        if total_addrs == 0:
            # No address data for this segment
            treated_share = 0.0
            dh_arm = 'Control'
        else:
            treated_share = treated_addrs / total_addrs

            # Classify based on treated_share
            if treated_share >= 0.9:  # Close to 100%
                dh_arm = 'Full'
                treated_share = 1.0
            elif 0.4 <= treated_share <= 0.6:  # Close to 50%
                dh_arm = 'Partial'
                treated_share = 0.5
            elif treated_share < 0.1:  # Close to 0%
                dh_arm = 'Control'
                treated_share = 0.0
            else:
                # Ambiguous - use threshold-based classification
                if treated_share > 0.7:
                    dh_arm = 'Full'
                    treated_share = 1.0
                elif treated_share > 0.3:
                    dh_arm = 'Partial'
                    treated_share = 0.5
                else:
                    dh_arm = 'Control'
                    treated_share = 0.0

        # Get bundle_id for this segment
        bundle_row = bundles_gdf[bundles_gdf['segment_id'] == segment_id]
        if len(bundle_row) == 0:
            print(f"  WARNING: Segment {segment_id} not found in bundle file")
            continue

        bundle_id = int(bundle_row['bundle_id'].iloc[0])

        segment_assignments.append({
            'date': date,
            'bundle_id': bundle_id,
            'segment_id': str(segment_id),
            'dh_arm': dh_arm,
            'treated_share': treated_share
        })

    df = pd.DataFrame(segment_assignments)

    print(f"  Inferred {len(df)} segment assignments")
    print(f"    Full: {len(df[df['dh_arm'] == 'Full'])} segments ({len(df[df['dh_arm'] == 'Full']) / len(df) * 100:.1f}%)")
    print(f"    Partial: {len(df[df['dh_arm'] == 'Partial'])} segments ({len(df[df['dh_arm'] == 'Partial']) / len(df) * 100:.1f}%)")
    print(f"    Control: {len(df[df['dh_arm'] == 'Control'])} segments ({len(df[df['dh_arm'] == 'Control']) / len(df) * 100:.1f}%)")

    return df


def main():
    """Infer segment assignments for all dates in google_drive_actual."""
    root = Path(__file__).parent
    plans_dir = root / "outputs" / "plans"
    google_drive_base = root / "outputs" / "routing" / "google_drive_actual"
    bundle_file = root / "outputs" / "bundles" / "DH" / "bundles_multibfs_regroup_filtered_length_3.parquet"

    print(f"{'='*70}")
    print("INFERRING SEGMENT ASSIGNMENTS FROM GOOGLE DRIVE FILES")
    print(f"{'='*70}")

    # Load bundle data
    print("\nLoading bundle data...")
    bundles_gdf = gpd.read_parquet(bundle_file)

    # Ensure segment_id column
    if 'segment_id' not in bundles_gdf.columns:
        if 'iamfloc' in bundles_gdf.columns:
            bundles_gdf['segment_id'] = bundles_gdf['iamfloc'].astype(str)

    # Ensure we have segment address counts
    if 'sfh_addr_count' not in bundles_gdf.columns:
        raise ValueError("Bundle file missing 'sfh_addr_count' column")

    print(f"  Loaded {bundles_gdf['bundle_id'].nunique()} bundles")

    # Find all dates with Google Drive files
    date_dirs = sorted([d for d in google_drive_base.glob("2026-*") if d.is_dir() and d.name >= "2026-01-10"])

    print(f"\nFound {len(date_dirs)} dates with Google Drive files:")
    for d in date_dirs:
        print(f"  - {d.name}")

    results = []

    for date_dir in date_dirs:
        date = date_dir.name

        df = infer_from_google_drive_files(
            date=date,
            google_drive_dir=date_dir,
            bundles_gdf=bundles_gdf
        )

        if df is not None:
            # Save to plans directory
            segment_file = plans_dir / f"segment_assignments_{date}.csv"
            df.to_csv(segment_file, index=False)
            print(f"  ✓ Saved to: {segment_file}")

            # Also save to Google Drive routing folder
            routing_file = date_dir / f"segment_assignments_{date}.csv"
            df.to_csv(routing_file, index=False)
            print(f"  ✓ Saved to: {routing_file}")

            results.append((date, len(df), "SUCCESS"))
        else:
            results.append((date, 0, "SKIPPED"))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for date, count, status in results:
        if status == "SUCCESS":
            print(f"  {date}: ✓ Inferred {count} segment assignments")
        else:
            print(f"  {date}: - {status}")

    print(f"\n✅ Completed! Inferred segment assignments for {len([r for r in results if r[2] == 'SUCCESS'])} dates")

    # Show distribution summary
    print(f"\n{'='*70}")
    print("DISTRIBUTION SUMMARY")
    print(f"{'='*70}")

    for date_dir in date_dirs:
        date = date_dir.name
        segment_file = plans_dir / f"segment_assignments_{date}.csv"

        if segment_file.exists():
            df = pd.read_csv(segment_file)
            full = len(df[df['dh_arm'] == 'Full'])
            partial = len(df[df['dh_arm'] == 'Partial'])
            control = len(df[df['dh_arm'] == 'Control'])
            total = len(df)

            print(f"{date}: {total} segments")
            print(f"  Full: {full} ({full/total*100:.1f}%), Partial: {partial} ({partial/total*100:.1f}%), Control: {control} ({control/total*100:.1f}%)")


if __name__ == "__main__":
    main()
