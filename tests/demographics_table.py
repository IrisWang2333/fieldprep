#!/usr/bin/env python3
"""
Demographics and PCI Table Generator

Creates a comparison table with demographics (income, education, race) and PCI across:
1. City overall (all street segments)
2. SFH-segments (eligible single-family housing segments)
3. Bundle-segments (segments included in bundles)

Spatial join logic:
- Each street segment is assigned to ONE Census Block Group (for population/race)
- Each street segment is assigned to ONE Census Tract (for income/education)
- Assignment based on largest area overlap

Data scope handling:
- Streets: City of San Diego only (from DataSD)
- Census Block Groups: California-wide shapefile → filtered to SD County → clipped to City extent
- Census Tracts: US-wide shapefile → filtered to SD County → clipped to City extent
- Final results represent CITY-level demographics, not County-level
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import zipfile
import tempfile
import shutil

# Add fieldprep src to path
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.utils import paths, load_sources


# ============================================================================
# Configuration
# ============================================================================

DATA_ROOT = Path("/Users/iris/Dropbox/SanDiego311/data/raw")

# Block Group data (2020 Census) - Race/ethnicity
BG_CSV = DATA_ROOT / "nhgis/nhgis_blockgroup_census/nhgis0008_ds258_2020_blck_grp.csv"
BG_SHAPE_ZIP = DATA_ROOT / "nhgis/nhgis0002_shape/nhgis0002_shapefile_tl2023_060_blck_grp_2023.zip"

# Tract data (ACS 2019-2023) - Income/education
TRACT_CSV = DATA_ROOT / "nhgis/nhgis0001_csv_popuraceedu/nhgis0001_ds267_20235_tract.csv"
TRACT_SHAPE_ZIP = DATA_ROOT / "nhgis/nhgis0002_shape/nhgis0002_shapefile_tl2023_us_tract_2023.zip"

# Street segments with PCI
STREETS_SHP = DATA_ROOT / "DataSD/sd_paving_segs_datasd/sd_paving_segs_datasd.shp"


# ============================================================================
# Helper Functions
# ============================================================================

def unzip_shapefile(zip_path: Path, temp_dir: Path):
    """Extract shapefile from zip to temp directory, return path to .shp"""
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(temp_dir)

    # Find the .shp file
    shp_files = list(temp_dir.glob("**/*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {zip_path}")

    # Filter for San Diego County (FIPS 073) if multiple files
    for shp in shp_files:
        # Try to identify San Diego County files
        if "073" in shp.stem or "san_diego" in shp.stem.lower():
            return shp

    return shp_files[0]


def filter_san_diego(gdf, geoid_col="GISJOIN"):
    """Filter to San Diego County only (FIPS 073 or 73)"""
    # Try COUNTYA first (numeric, without leading zero)
    if "COUNTYA" in gdf.columns:
        sd_mask = (gdf["COUNTYA"] == 73) | (gdf["COUNTYA"] == "73")
        if sd_mask.any():
            return gdf[sd_mask].copy()

    # Try COUNTYFP (string, may have leading zero)
    for col in ["COUNTYFP", "COUNTY"]:
        if col in gdf.columns:
            sd_mask = gdf[col].astype(str).isin(["073", "73"])
            if sd_mask.any():
                return gdf[sd_mask].copy()

    # Last resort: GISJOIN pattern
    if geoid_col in gdf.columns:
        sd_mask = gdf[geoid_col].str.contains("G06073", na=False)
        if sd_mask.any():
            return gdf[sd_mask].copy()

    print("  Warning: Could not filter to San Diego County, using all data")
    return gdf.copy()


def spatial_join_largest_overlap(segments_gdf, census_gdf, census_id_col):
    """
    Assign each segment to the census unit with largest area overlap.

    Args:
        segments_gdf: Street segments (must be in same CRS as census_gdf)
        census_gdf: Census geography (Block Group or Tract)
        census_id_col: ID column in census_gdf to join

    Returns:
        GeoDataFrame with census_id_col added
    """
    print(f"  Spatial join: {len(segments_gdf)} segments to {len(census_gdf)} census units...")

    # Ensure same CRS
    if segments_gdf.crs != census_gdf.crs:
        census_gdf = census_gdf.to_crs(segments_gdf.crs)

    # Overlay to get all intersections with area
    overlay = gpd.overlay(
        segments_gdf[[segments_gdf.geometry.name]].reset_index(),
        census_gdf[[census_id_col, census_gdf.geometry.name]],
        how='intersection'
    )

    if overlay.empty:
        raise ValueError("No overlaps found between segments and census geography")

    # Calculate area of each intersection
    overlay['overlap_area'] = overlay.geometry.area

    # For each segment, find census unit with largest overlap
    idx_max = overlay.groupby('index')['overlap_area'].idxmax()
    largest = overlay.loc[idx_max, ['index', census_id_col]].set_index('index')

    # Merge back to original segments
    result = segments_gdf.copy()
    result[census_id_col] = result.index.map(largest[census_id_col])

    matched = result[census_id_col].notna().sum()
    print(f"  → Matched {matched}/{len(result)} segments ({matched/len(result)*100:.1f}%)")

    return result


def calculate_demographics(bg_data, tract_data):
    """
    Calculate demographic variables from Census data.

    Args:
        bg_data: Block Group DataFrame with race columns
        tract_data: Tract DataFrame with income/education columns

    Returns:
        Two DataFrames with calculated variables
    """
    # ---- Block Group: Race/Ethnicity ----
    bg = bg_data.copy()

    # Population
    bg['population'] = bg['U7P001'].fillna(0)

    # Shares (as proportions 0-1)
    total = bg['U7P001'].replace(0, np.nan)
    bg['share_hispanic'] = bg['U7P002'] / total
    bg['share_white_nh'] = bg['U7P005'] / total  # Non-Hispanic White alone
    bg['share_black_nh'] = bg['U7P006'] / total  # Non-Hispanic Black alone
    bg['share_asian_nh'] = bg['U7P008'] / total  # Non-Hispanic Asian alone

    bg_vars = bg[['GISJOIN', 'population', 'share_hispanic', 'share_white_nh',
                  'share_black_nh', 'share_asian_nh']].copy()

    # ---- Tract: Income/Education ----
    tract = tract_data.copy()

    # Per capita income (replace NHGIS missing value code -666666666 with NaN)
    tract['per_capita_income'] = tract['ASRTE001'].replace(-666666666, np.nan)
    # Also replace any other negative values (shouldn't exist for income)
    tract.loc[tract['per_capita_income'] < 0, 'per_capita_income'] = np.nan

    # Education: Share with Bachelor's degree or higher
    # ASP3E001: Total population 25+
    # ASP3E022: Bachelor's degree
    # ASP3E023: Master's degree
    # ASP3E024: Professional school degree
    # ASP3E025: Doctorate degree
    total_25plus = tract['ASP3E001'].replace(0, np.nan)
    college_plus = (
        tract['ASP3E022'].fillna(0) +
        tract['ASP3E023'].fillna(0) +
        tract['ASP3E024'].fillna(0) +
        tract['ASP3E025'].fillna(0)
    )
    tract['share_college'] = college_plus / total_25plus

    tract_vars = tract[['GISJOIN', 'per_capita_income', 'share_college']].copy()

    return bg_vars, tract_vars


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*70)
    print("DEMOGRAPHICS & PCI TABLE GENERATOR")
    print("="*70)

    # ---- 1. Load Street Segments with PCI ----
    print("\n[1/6] Loading street segments...")
    streets = gpd.read_file(STREETS_SHP)
    print(f"  Loaded {len(streets)} street segments")

    # Check for PCI column
    pci_col = None
    for col in ['pci23', 'PCI23', 'PCI', 'pci']:
        if col in streets.columns:
            pci_col = col
            break

    if pci_col:
        streets['PCI'] = pd.to_numeric(streets[pci_col], errors='coerce')
        print(f"  Found PCI column: {pci_col}")
    else:
        print("  Warning: No PCI column found")
        streets['PCI'] = np.nan

    # Get segment ID
    seg_id_col = None
    for col in ['iamfloc', 'segment_id', 'SEGMENT_ID', 'OBJECTID']:
        if col in streets.columns:
            seg_id_col = col
            break

    if seg_id_col:
        streets['segment_id'] = streets[seg_id_col].astype(str)
        print(f"  Found segment ID: {seg_id_col}")
    else:
        streets['segment_id'] = streets.index.astype(str)
        print("  Created segment ID from index")

    # Ensure working CRS (meters for area calculation)
    if streets.crs.to_epsg() != 26911:
        print(f"  Reprojecting from {streets.crs} to EPSG:26911...")
        streets = streets.to_crs("EPSG:26911")

    # ---- 2. Load Census Block Group Data ----
    print("\n[2/6] Loading Census Block Group data (race/ethnicity)...")

    # Load CSV (skip description row at index 1)
    bg_csv = pd.read_csv(BG_CSV, skiprows=[1], low_memory=False)
    print(f"  Loaded {len(bg_csv)} block groups from CSV")

    # Filter to San Diego County (FIPS code 73 in COUNTYA column)
    bg_csv_sd = bg_csv[bg_csv['COUNTYA'] == 73].copy()
    print(f"  Filtered to {len(bg_csv_sd)} San Diego County block groups")

    # Load shapefile
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_shp_path = unzip_shapefile(BG_SHAPE_ZIP, Path(tmpdir))
        bg_shp = gpd.read_file(bg_shp_path)

    bg_shp_sd = filter_san_diego(bg_shp, "GISJOIN")
    print(f"  Loaded {len(bg_shp_sd)} San Diego County block group shapes")

    # Join CSV to shapefile
    bg_geo = bg_shp_sd.merge(bg_csv_sd, on='GISJOIN', how='inner')
    print(f"  Joined to {len(bg_geo)} block groups with data")

    # Reproject to match streets
    bg_geo = bg_geo.to_crs(streets.crs)

    # ⚠️  IMPORTANT: Clip to City extent
    # Streets are City-level (from DataSD), but census data is County-level
    # Create a buffered extent from streets to limit census units to City area
    streets_extent = streets.geometry.unary_union.envelope.buffer(1000)  # 1km buffer
    bg_geo_clipped = bg_geo[bg_geo.geometry.intersects(streets_extent)].copy()
    print(f"  → Clipped to City extent: {len(bg_geo_clipped)} block groups")
    bg_geo = bg_geo_clipped

    # ---- 3. Load Census Tract Data ----
    print("\n[3/6] Loading Census Tract data (income/education)...")

    # Load CSV (skip description row at index 1)
    tract_csv = pd.read_csv(TRACT_CSV, skiprows=[1], low_memory=False)
    print(f"  Loaded {len(tract_csv)} tracts from CSV")

    # Filter to San Diego County (FIPS code 73 in COUNTYA column)
    tract_csv_sd = tract_csv[tract_csv['COUNTYA'] == 73].copy()
    print(f"  Filtered to {len(tract_csv_sd)} San Diego County tracts")

    # Load shapefile
    with tempfile.TemporaryDirectory() as tmpdir:
        tract_shp_path = unzip_shapefile(TRACT_SHAPE_ZIP, Path(tmpdir))
        tract_shp = gpd.read_file(tract_shp_path)

    tract_shp_sd = filter_san_diego(tract_shp, "GISJOIN")
    print(f"  Loaded {len(tract_shp_sd)} San Diego County tract shapes")

    # Join CSV to shapefile
    tract_geo = tract_shp_sd.merge(tract_csv_sd, on='GISJOIN', how='inner')
    print(f"  Joined to {len(tract_geo)} tracts with data")

    # Reproject to match streets
    tract_geo = tract_geo.to_crs(streets.crs)

    # ⚠️  IMPORTANT: Clip to City extent (same as Block Groups)
    tract_geo_clipped = tract_geo[tract_geo.geometry.intersects(streets_extent)].copy()
    print(f"  → Clipped to City extent: {len(tract_geo_clipped)} tracts")
    tract_geo = tract_geo_clipped

    # ---- 4. Spatial Joins ----
    print("\n[4/6] Performing spatial joins...")

    # Join to Block Groups
    streets_bg = spatial_join_largest_overlap(streets, bg_geo, 'GISJOIN')
    streets_bg = streets_bg.rename(columns={'GISJOIN': 'GISJOIN_BG'})

    # Join to Tracts
    streets_full = spatial_join_largest_overlap(streets_bg, tract_geo, 'GISJOIN')
    streets_full = streets_full.rename(columns={'GISJOIN': 'GISJOIN_TRACT'})

    # ---- 5. Attach Demographics ----
    print("\n[5/6] Calculating demographics...")

    bg_vars, tract_vars = calculate_demographics(bg_csv_sd, tract_csv_sd)

    # Merge to streets
    streets_full = streets_full.merge(
        bg_vars, left_on='GISJOIN_BG', right_on='GISJOIN', how='left'
    ).drop(columns=['GISJOIN'])

    streets_full = streets_full.merge(
        tract_vars, left_on='GISJOIN_TRACT', right_on='GISJOIN', how='left'
    ).drop(columns=['GISJOIN'])

    print(f"  Attached demographics to {len(streets_full)} segments")

    # ---- 6. Define Sample Sets ----
    print("\n[6/6] Defining sample sets...")

    # Sample 1: City overall (all segments)
    city_all = streets_full.copy()
    print(f"  City overall: {len(city_all)} segments")

    # Sample 2: SFH-eligible segments
    # Use fieldprep's sweep logic to identify eligible segments
    try:
        # Try to load from existing sweep output
        root, cfg, out_root = paths()
        sweep_tag = "locked"  # or read from params
        sweep_dir = out_root / "sweep" / sweep_tag

        # Find most recent eligible parquet
        eligible_files = list(sweep_dir.glob("eligible_*.parquet"))
        if eligible_files:
            latest = max(eligible_files, key=lambda p: p.stat().st_mtime)
            print(f"  Loading eligible segments from: {latest.name}")

            eligible = gpd.read_parquet(latest)
            eligible_ids = set(eligible['iamfloc'].astype(str) if 'iamfloc' in eligible.columns
                              else eligible[seg_id_col].astype(str))

            sfh_eligible = streets_full[streets_full['segment_id'].isin(eligible_ids)].copy()
            print(f"  SFH-eligible: {len(sfh_eligible)} segments")
        else:
            print("  Warning: No sweep output found, using all segments as SFH-eligible")
            sfh_eligible = city_all.copy()
    except Exception as e:
        print(f"  Warning: Could not load sweep data ({e}), using all segments")
        sfh_eligible = city_all.copy()

    # Sample 3: Bundle segments
    try:
        # Load from bundle parquet
        bundle_dir = out_root / "bundles" / "DH"
        bundle_file = bundle_dir / "bundles_multibfs_regroup_filtered.parquet"

        if bundle_file.exists():
            print(f"  Loading bundle segments from: {bundle_file.name}")
            bundles = gpd.read_parquet(bundle_file)

            bundle_seg_col = None
            for col in ['iamfloc', 'segment_id', seg_id_col]:
                if col in bundles.columns:
                    bundle_seg_col = col
                    break

            if bundle_seg_col:
                bundle_ids = set(bundles[bundle_seg_col].astype(str))
                bundle_segs = streets_full[streets_full['segment_id'].isin(bundle_ids)].copy()
                print(f"  Bundle segments: {len(bundle_segs)} segments")
            else:
                print("  Warning: No segment ID found in bundles, using SFH-eligible")
                bundle_segs = sfh_eligible.copy()
        else:
            print(f"  Warning: Bundle file not found at {bundle_file}")
            bundle_segs = sfh_eligible.copy()
    except Exception as e:
        print(f"  Warning: Could not load bundle data ({e}), using SFH-eligible")
        bundle_segs = sfh_eligible.copy()

    # ---- 7. Calculate Summary Statistics ----
    print("\n" + "="*70)
    print("CALCULATING SUMMARY STATISTICS")
    print("="*70)

    def summarize_sample(gdf, name):
        """Calculate means for a sample"""
        return pd.Series({
            'sample': name,
            'n_segments': len(gdf),
            'avg_pci': gdf['PCI'].mean(),
            'avg_income': gdf['per_capita_income'].mean(),
            'share_college': gdf['share_college'].mean(),
            'share_white': gdf['share_white_nh'].mean(),
            'share_hispanic': gdf['share_hispanic'].mean(),
            'share_asian': gdf['share_asian_nh'].mean(),
            'share_black': gdf['share_black_nh'].mean(),
        })

    results = pd.DataFrame([
        summarize_sample(city_all, 'City Overall'),
        summarize_sample(sfh_eligible, 'SFH-Eligible Segments'),
        summarize_sample(bundle_segs, 'Bundle Segments'),
    ])

    # ---- 8. Format and Display Table ----
    print("\n" + "="*70)
    print("DEMOGRAPHICS & PCI COMPARISON TABLE")
    print("="*70)
    print()

    # Format percentages and currency
    results_formatted = results.copy()
    results_formatted['avg_income'] = results_formatted['avg_income'].apply(lambda x: f"${x:,.0f}")
    results_formatted['avg_pci'] = results_formatted['avg_pci'].apply(lambda x: f"{x:.1f}")

    for col in ['share_college', 'share_white', 'share_hispanic', 'share_asian', 'share_black']:
        results_formatted[col] = results_formatted[col].apply(lambda x: f"{x*100:.1f}%")

    # Rename columns for display
    results_formatted = results_formatted.rename(columns={
        'sample': 'Sample',
        'n_segments': 'N Segments',
        'avg_pci': 'Avg PCI',
        'avg_income': 'Per Capita Income',
        'share_college': 'College+ %',
        'share_white': 'White (NH) %',
        'share_hispanic': 'Hispanic %',
        'share_asian': 'Asian (NH) %',
        'share_black': 'Black (NH) %',
    })

    print(results_formatted.to_string(index=False))
    print()

    # ---- 9. Save Results ----
    root, cfg, out_root = paths()
    output_file = out_root / "demographics_pci_table.csv"
    results.to_csv(output_file, index=False)
    print(f"Saved raw results to: {output_file}")

    # Also save formatted version
    output_file_fmt = out_root / "demographics_pci_table_formatted.csv"
    results_formatted.to_csv(output_file_fmt, index=False)
    print(f"Saved formatted table to: {output_file_fmt}")

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
