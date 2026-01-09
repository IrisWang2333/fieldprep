#!/usr/bin/env python3
"""
Load and geocode interviewer addresses from Google Sheet.

This module reads interviewer information from the shared Google Sheet
(tab: "Interviewers") and geocodes their home addresses.
"""

import pandas as pd
from pathlib import Path
import requests
from typing import Dict, List
import time
import json


def load_interviewers_from_sheet(
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0',
    cache_file: str = 'data/interviewers_geocoded_cache.json'
) -> pd.DataFrame:
    """
    Load interviewer addresses from Google Sheet and geocode them.

    Uses a local cache to avoid re-geocoding on every run.

    Args:
        sheet_id: Google Sheets ID
        cache_file: Path to cache file for geocoded results

    Returns:
        DataFrame with columns: name, email, address, lat, lon
    """
    # Sheet structure:
    # - Tab 1 (gid=0): Daily Assignments (Date, A, B, C, D, E, F)
    # - Tab 2: Interviewers (Name, Email, Home Address)

    # Try common gids for the second tab
    interviewers_df = None
    for gid in [1, 2, '1', '2', 'Interviewers']:
        try:
            # Try with gid as parameter
            if isinstance(gid, int) or gid.isdigit():
                url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
            else:
                # Try with sheet name
                url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={gid}'

            df = pd.read_csv(url)

            # Check if this looks like the interviewers sheet
            if 'Name' in df.columns and 'Home Address' in df.columns:
                interviewers_df = df
                print(f"[Geocoding] Found Interviewers sheet at gid={gid}")
                break
        except Exception as e:
            continue

    if interviewers_df is None:
        raise ValueError("Could not find Interviewers sheet in Google Sheets")

    # Clean up the dataframe
    interviewers_df = interviewers_df[['Name', 'Email', 'Home Address']].copy()
    interviewers_df.columns = ['name', 'email', 'address']

    # Remove empty rows
    interviewers_df = interviewers_df.dropna(subset=['name', 'address'])
    interviewers_df = interviewers_df[interviewers_df['name'].str.strip() != '']
    interviewers_df = interviewers_df[interviewers_df['address'].str.strip() != '']

    print(f"[Geocoding] Loaded {len(interviewers_df)} interviewers from sheet")

    # Load cache if exists
    cache_path = Path(cache_file)
    cache = {}
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        print(f"[Geocoding] Loaded cache with {len(cache)} entries")

    # Geocode addresses
    results = []
    for _, row in interviewers_df.iterrows():
        name = row['name'].strip()
        email = row['email'].strip() if pd.notna(row['email']) else ''
        address = row['address'].strip()

        # Check cache first
        cache_key = address.lower()
        if cache_key in cache:
            lat, lon = cache[cache_key]['lat'], cache[cache_key]['lon']
            print(f"[Geocoding] {name}: Using cached location ({lat:.4f}, {lon:.4f})")
        else:
            # Geocode using Nominatim (OpenStreetMap)
            lat, lon = geocode_address(address)

            if lat is not None and lon is not None:
                cache[cache_key] = {'lat': lat, 'lon': lon, 'address': address}
                print(f"[Geocoding] {name}: Geocoded to ({lat:.4f}, {lon:.4f})")

                # Save cache after each successful geocode
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(cache, f, indent=2)

                # Rate limit: wait 1 second between requests
                time.sleep(1)
            else:
                print(f"[Geocoding] {name}: Failed to geocode address: {address}")

        results.append({
            'name': name,
            'email': email,
            'address': address,
            'lat': lat,
            'lon': lon
        })

    result_df = pd.DataFrame(results)
    return result_df


def geocode_address(address: str) -> tuple[float | None, float | None]:
    """
    Geocode an address using Nominatim (OpenStreetMap).

    Args:
        address: Address string

    Returns:
        (latitude, longitude) or (None, None) if geocoding fails
    """
    try:
        # Use Nominatim API
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': address,
            'format': 'json',
            'limit': 1
        }
        headers = {
            'User-Agent': 'SD311-FieldPrep/1.0'
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data and len(data) > 0:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
        else:
            return None, None

    except Exception as e:
        print(f"[Geocoding] Error geocoding '{address}': {e}")
        return None, None


def get_interviewers_for_date_with_locations(
    date: str,
    sheet_id: str = '1IFb5AF2VEd9iMK69B4GFlYovVOM-7_TxIo6MrsJ-6X0',
    cache_file: str = 'data/interviewers_geocoded_cache.json'
) -> List[Dict]:
    """
    Get interviewer assignments for a date with their geocoded locations.

    Args:
        date: Date string (YYYY-MM-DD)
        sheet_id: Google Sheets ID
        cache_file: Path to geocoding cache

    Returns:
        List of dicts with keys: name, email, lat, lon
    """
    # Load assignments from tab 1
    assignments_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0'
    assignments_df = pd.read_csv(assignments_url)

    matching_rows = assignments_df[assignments_df['Date'] == date]

    if len(matching_rows) == 0:
        raise ValueError(f"No assignment found for date {date}")

    row = matching_rows.iloc[0]
    assigned_names = [row['A'], row['B'], row['C'], row['D'], row['E'], row['F']]

    # Load geocoded interviewers from tab 2
    all_interviewers_df = load_interviewers_from_sheet(sheet_id, cache_file)

    # Filter to assigned interviewers
    assigned_interviewers = all_interviewers_df[
        all_interviewers_df['name'].isin(assigned_names)
    ].copy()

    # Drop any without valid geocoding
    assigned_interviewers = assigned_interviewers.dropna(subset=['lat', 'lon'])

    if len(assigned_interviewers) != len(assigned_names):
        missing = set(assigned_names) - set(assigned_interviewers['name'])
        print(f"[Warning] Some assigned interviewers lack geocoding: {missing}")

    # Convert to list of dicts
    result = assigned_interviewers[['name', 'email', 'lat', 'lon']].to_dict('records')

    return result


if __name__ == '__main__':
    # Test the geocoding
    print("=== Testing Interviewer Geocoding ===\n")

    df = load_interviewers_from_sheet()

    print("\n=== Geocoded Results ===")
    for _, row in df.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            print(f"{row['name']}: ({row['lat']:.6f}, {row['lon']:.6f})")
        else:
            print(f"{row['name']}: FAILED TO GEOCODE")

    print("\n=== Testing Date Assignment ===")
    test_date = "2025-12-27"
    assigned = get_interviewers_for_date_with_locations(test_date)

    print(f"\nInterviewers assigned for {test_date}:")
    for interviewer in assigned:
        print(f"  {interviewer['name']}: ({interviewer['lat']:.6f}, {interviewer['lon']:.6f})")
