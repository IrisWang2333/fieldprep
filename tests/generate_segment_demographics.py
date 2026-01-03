#!/usr/bin/env python3
"""
Generate segment-level demographics file for balance checks.

This script runs the demographics processing and saves segment-level data.
"""

import sys
from pathlib import Path

# Add src to path
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import geopandas as gpd
import tempfile
import zipfile
import numpy as np

from sd311_fieldprep.utils import paths

# Configuration - same as demographics_table.py
DATA_ROOT = Path("/Users/iris/Dropbox/SanDiego311/data/raw")
BG_CSV = DATA_ROOT / "nhgis/nhgis_blockgroup_census/nhgis0008_ds258_2020_blck_grp.csv"
BG_SHAPE_ZIP = DATA_ROOT / "nhgis/nhgis0002_shape/nhgis0002_shapefile_tl2023_060_blck_grp_2023.zip"
TRACT_CSV = DATA_ROOT / "nhgis/nhgis0001_csv_popuraceedu/nhgis0001_ds267_20235_tract.csv"
TRACT_SHAPE_ZIP = DATA_ROOT / "nhgis/nhgis0002_shape/nhgis0002_shapefile_tl2023_us_tract_2023.zip"
STREETS_SHP = DATA_ROOT / "DataSD/sd_paving_segs_datasd/sd_paving_segs_datasd.shp"


# Helper functions - same as demographics_table.py
def unzip_shapefile(zip_path: Path, temp_dir: Path):
    """Extract shapefile from zip to temp directory"""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(temp_dir)
    shp_files = list(temp_dir.glob("**/*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {zip_path}")
    for shp in shp_files:
        if "073" in shp.stem or "san_diego" in shp.stem.lower():
            return shp
    return shp_files[0]


def filter_san_diego(gdf, geoid_col="GISJOIN"):
    """Filter to San Diego County"""
    if "COUNTYA" in gdf.columns:
        sd_mask = (gdf["COUNTYA"] == 73) | (gdf["COUNTYA"] == "73")
        if sd_mask.any():
            return gdf[sd_mask].copy()
    for col in ["COUNTYFP", "COUNTY"]:
        if col in gdf.columns:
            sd_mask = gdf[col].astype(str).isin(["073", "73"])
            if sd_mask.any():
                return gdf[sd_mask].copy()
    if geoid_col in gdf.columns:
        sd_mask = gdf[geoid_col].str.contains("G06073", na=False)
        if sd_mask.any():
            return gdf[sd_mask].copy()
    return gdf.copy()


def spatial_join_largest_overlap(segments_gdf, census_gdf, census_id_col):
    """Assign each segment to census unit with largest area overlap"""
    if segments_gdf.crs != census_gdf.crs:
        census_gdf = census_gdf.to_crs(segments_gdf.crs)
    overlay = gpd.overlay(
        segments_gdf[[segments_gdf.geometry.name]].reset_index(),
        census_gdf[[census_id_col, census_gdf.geometry.name]],
        how='intersection'
    )
    if overlay.empty:
        raise ValueError("No overlaps found")
    overlay['overlap_area'] = overlay.geometry.area
    idx_max = overlay.groupby('index')['overlap_area'].idxmax()
    largest = overlay.loc[idx_max, ['index', census_id_col]].set_index('index')
    result = segments_gdf.copy()
    result[census_id_col] = result.index.map(largest[census_id_col])
    matched = result[census_id_col].notna().sum()
    print(f"  → Matched {matched}/{len(result)} segments")
    return result


def calculate_demographics(bg_data, tract_data):
    """Calculate demographic variables from Census data"""
    # Block Group: Race/Ethnicity
    bg = bg_data.copy()
    bg['population'] = bg['U7P001'].fillna(0)
    total = bg['U7P001'].replace(0, np.nan)
    bg['share_hispanic'] = bg['U7P002'] / total
    bg['share_white_nh'] = bg['U7P005'] / total
    bg['share_black_nh'] = bg['U7P006'] / total
    bg['share_asian_nh'] = bg['U7P008'] / total
    bg_vars = bg[['GISJOIN', 'population', 'share_hispanic', 'share_white_nh',
                  'share_black_nh', 'share_asian_nh']].copy()

    # Tract: Income/Education
    tract = tract_data.copy()
    tract['per_capita_income'] = tract['ASRTE001'].replace(-666666666, np.nan)
    tract.loc[tract['per_capita_income'] < 0, 'per_capita_income'] = np.nan
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


def main():
    print("\n" + "="*70)
    print("GENERATE SEGMENT-LEVEL DEMOGRAPHICS")
    print("="*70)

    # ---- 1. Load Street Segments ----
    print("\n[1/5] Loading street segments...")
    streets = gpd.read_file(STREETS_SHP)
    print(f"  Loaded {len(streets)} street segments")

    # PCI
    pci_col = None
    for col in ['pci23', 'PCI23', 'PCI', 'pci']:
        if col in streets.columns:
            pci_col = col
            break
    if pci_col:
        streets['pci'] = pd.to_numeric(streets[pci_col], errors='coerce')

    # Segment ID
    seg_id_col = None
    for col in ['iamfloc', 'segment_id', 'SEGMENT_ID']:
        if col in streets.columns:
            seg_id_col = col
            break
    if seg_id_col:
        streets['segment_id'] = streets[seg_id_col].astype(str)
    else:
        streets['segment_id'] = streets.index.astype(str)

    # Reproject to meters
    if streets.crs.to_epsg() != 26911:
        streets = streets.to_crs("EPSG:26911")

    # ---- 2. Load Census Block Group Data ----
    print("\n[2/5] Loading Census Block Group data...")
    bg_csv = pd.read_csv(BG_CSV, skiprows=[1], low_memory=False)
    bg_csv_sd = bg_csv[bg_csv['COUNTYA'] == 73].copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        bg_shp_path = unzip_shapefile(BG_SHAPE_ZIP, Path(tmpdir))
        bg_shp = gpd.read_file(bg_shp_path)

    bg_shp_sd = filter_san_diego(bg_shp, "GISJOIN")
    bg_geo = bg_shp_sd.merge(bg_csv_sd, on='GISJOIN', how='inner')
    bg_geo = bg_geo.to_crs(streets.crs)

    # Clip to city extent
    streets_extent = streets.geometry.union_all().envelope.buffer(1000)
    bg_geo = bg_geo[bg_geo.geometry.intersects(streets_extent)].copy()

    print(f"  Loaded {len(bg_geo)} block groups (clipped to city)")

    # ---- 3. Load Census Tract Data ----
    print("\n[3/5] Loading Census Tract data...")
    tract_csv = pd.read_csv(TRACT_CSV, skiprows=[1], low_memory=False)
    tract_csv_sd = tract_csv[tract_csv['COUNTYA'] == 73].copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        tract_shp_path = unzip_shapefile(TRACT_SHAPE_ZIP, Path(tmpdir))
        tract_shp = gpd.read_file(tract_shp_path)

    tract_shp_sd = filter_san_diego(tract_shp, "GISJOIN")
    tract_geo = tract_shp_sd.merge(tract_csv_sd, on='GISJOIN', how='inner')
    tract_geo = tract_geo.to_crs(streets.crs)

    # Clip to city extent
    tract_geo = tract_geo[tract_geo.geometry.intersects(streets_extent)].copy()

    print(f"  Loaded {len(tract_geo)} tracts (clipped to city)")

    # ---- 4. Spatial Joins ----
    print("\n[4/5] Performing spatial joins...")
    streets_bg = spatial_join_largest_overlap(streets, bg_geo, 'GISJOIN')
    streets_bg = streets_bg.rename(columns={'GISJOIN': 'GISJOIN_BG'})

    streets_full = spatial_join_largest_overlap(streets_bg, tract_geo, 'GISJOIN')
    streets_full = streets_full.rename(columns={'GISJOIN': 'GISJOIN_TRACT'})

    # ---- 5. Attach Demographics ----
    print("\n[5/5] Calculating and attaching demographics...")
    bg_vars, tract_vars = calculate_demographics(bg_csv_sd, tract_csv_sd)

    streets_full = streets_full.merge(
        bg_vars, left_on='GISJOIN_BG', right_on='GISJOIN', how='left'
    ).drop(columns=['GISJOIN'])

    streets_full = streets_full.merge(
        tract_vars, left_on='GISJOIN_TRACT', right_on='GISJOIN', how='left'
    ).drop(columns=['GISJOIN'])

    print(f"  Attached demographics to {len(streets_full)} segments")

    # ---- 6. Save Output ----
    root, cfg, out_root = paths()
    output_file = out_root / "segments_with_demographics.parquet"

    # Select relevant columns only
    keep_cols = [
        'segment_id', 'pci',
        'per_capita_income', 'share_college',
        'share_hispanic', 'share_white_nh', 'share_black_nh', 'share_asian_nh',
        'GISJOIN_BG', 'GISJOIN_TRACT'
    ]

    # Keep only columns that exist
    keep_cols = [c for c in keep_cols if c in streets_full.columns]

    output_df = streets_full[keep_cols].copy()
    output_df.to_parquet(output_file, index=False)

    print(f"\n✓ Saved segment-level demographics to: {output_file}")
    print(f"  {len(output_df):,} segments")
    print(f"  Columns: {', '.join(keep_cols)}")

    # Print sample
    print(f"\nSample (first 3 rows):")
    print(output_df.head(3))

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
