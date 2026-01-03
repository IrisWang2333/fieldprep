#!/usr/bin/env python
"""
Download required data files for GitHub Actions

Downloads geospatial data from San Diego Open Data Portal
"""
import requests
from pathlib import Path
import zipfile
import io
import geopandas as gpd

# Data directory
DATA_DIR = Path("data/raw/DataSD")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_and_extract_geojson(url, output_dir, output_name):
    """Download GeoJSON from ArcGIS REST API and save as shapefile."""
    print(f"Downloading {output_name}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download GeoJSON
    response = requests.get(url)
    response.raise_for_status()

    # Read with geopandas
    gdf = gpd.read_file(io.BytesIO(response.content))

    # Save as shapefile
    output_file = output_dir / f"{output_name}.shp"
    gdf.to_file(output_file)

    print(f"  ✓ Saved to {output_file}")
    return output_file


def download_csv(url, output_path):
    """Download CSV file."""
    print(f"Downloading {output_path}...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    print(f"  ✓ Saved to {output_path}")
    return output_path


def main():
    """Download all required data files."""

    print("="*70)
    print("DOWNLOADING DATA FILES FOR GITHUB ACTIONS")
    print("="*70)

    # 1. Streets segments (from San Diego Open Data)
    # This is the most critical file
    streets_url = "https://seshat.datasd.org/sde/street_segments/street_segments_datasd.zip"
    print("\n1. Downloading street segments...")
    try:
        response = requests.get(streets_url, timeout=300)
        response.raise_for_status()

        # Extract zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            streets_dir = DATA_DIR / "sd_paving_segs_datasd"
            streets_dir.mkdir(parents=True, exist_ok=True)
            z.extractall(streets_dir)

        print(f"  ✓ Extracted to {streets_dir}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  Using alternative: ArcGIS REST API")
        download_and_extract_geojson(
            "https://services1.arcgis.com/1vIhDJwtG5eNmiqX/arcgis/rest/services/Street_Segments/FeatureServer/0/query?where=1=1&outFields=*&f=geojson",
            DATA_DIR / "sd_paving_segs_datasd",
            "sd_paving_segs_datasd"
        )

    # 2. Address points (from config - already has URL)
    print("\n2. Downloading address points...")
    download_and_extract_geojson(
        "https://webmaps.sandiego.gov/arcgis/rest/services/GeocoderMerged/MapServer/2/query?where=1=1&outFields=*&f=geojson",
        DATA_DIR / "addrapn_datasd",
        "addrapn_datasd"
    )

    # 3. Council districts (from config)
    print("\n3. Downloading council districts...")
    download_and_extract_geojson(
        "https://webmaps.sandiego.gov/arcgis/rest/services/DSD/Basemap/MapServer/7/query?where=1=1&outFields=*&f=geojson",
        DATA_DIR / "council_districts_datasd",
        "council_districts_datasd"
    )

    # 4. Zoning
    print("\n4. Downloading zoning...")
    try:
        download_and_extract_geojson(
            "https://services1.arcgis.com/1vIhDJwtG5eNmiqX/arcgis/rest/services/Zoning/FeatureServer/0/query?where=1=1&outFields=*&f=geojson",
            DATA_DIR / "zoning_datasd",
            "zoning_datasd"
        )
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  Zoning data optional - continuing...")

    # 5. Community plan areas
    print("\n5. Downloading community plan areas...")
    try:
        download_and_extract_geojson(
            "https://services1.arcgis.com/1vIhDJwtG5eNmiqX/arcgis/rest/services/Community_Plan_Areas/FeatureServer/0/query?where=1=1&outFields=*&f=geojson",
            DATA_DIR / "cmty_plan_datasd",
            "cmty_plan_datasd"
        )
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  Community plan data optional - continuing...")

    # 6. City boundary
    print("\n6. Downloading city boundary...")
    try:
        download_and_extract_geojson(
            "https://services1.arcgis.com/1vIhDJwtG5eNmiqX/arcgis/rest/services/City_Boundary/FeatureServer/0/query?where=1=1&outFields=*&f=geojson",
            DATA_DIR / "san_diego_boundary_datasd",
            "san_diego_boundary_datasd"
        )
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  City boundary optional - continuing...")

    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)

    # List downloaded files
    print("\nDownloaded files:")
    for item in DATA_DIR.rglob("*.shp"):
        size = item.stat().st_size / (1024*1024)  # MB
        print(f"  - {item.relative_to(DATA_DIR.parent)} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
