#!/usr/bin/env python
"""
Data Fetcher for Pothole Data

Handles loading and filtering of notification_activities data for:
1. Determining which bundles have potholes (eligibility)
2. Determining fix status (R_it)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import requests

# Data paths - use relative to project root
# Try to find project root (where .git or src exists)
def get_project_root():
    """Find project root directory."""
    current = Path(__file__).resolve()
    # Go up to find project root (where src/ exists)
    for parent in [current.parent.parent, current.parent.parent.parent]:
        if (parent / 'src').exists() or (parent / '.git').exists():
            return parent
    # Fallback to hardcoded path for local development
    return Path("/Users/iris/Dropbox/sandiego code/code/fieldprep")

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
NOTIFICATION_ACTIVITIES_LOCAL = DATA_DIR / "notification_activities.csv"
NOTIFICATION_ACTIVITIES_URL = "https://seshat.datasd.org/td_optimizations/notification_activities.csv"

def fetch_latest_notification_activities(use_local=True, download_if_missing=True):
    """
    Fetch notification activities data.

    Parameters
    ----------
    use_local : bool
        If True, use local file if available
    download_if_missing : bool
        If True and local file doesn't exist, download from URL

    Returns
    -------
    pd.DataFrame
        Filtered notification activities (pothole patches only)
    """
    # Try local file first
    if use_local and NOTIFICATION_ACTIVITIES_LOCAL.exists():
        print(f"Loading local file: {NOTIFICATION_ACTIVITIES_LOCAL}")
        df = pd.read_csv(NOTIFICATION_ACTIVITIES_LOCAL)
    else:
        if download_if_missing:
            print(f"Downloading from: {NOTIFICATION_ACTIVITIES_URL}")
            try:
                df = pd.read_csv(NOTIFICATION_ACTIVITIES_URL)
                # Create data directory if it doesn't exist
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                # Save locally for future use
                df.to_csv(NOTIFICATION_ACTIVITIES_LOCAL, index=False)
                print(f"Saved to: {NOTIFICATION_ACTIVITIES_LOCAL}")
            except Exception as e:
                print(f"Error downloading: {e}")
                if NOTIFICATION_ACTIVITIES_LOCAL.exists():
                    print(f"Falling back to local file")
                    df = pd.read_csv(NOTIFICATION_ACTIVITIES_LOCAL)
                else:
                    raise
        else:
            raise FileNotFoundError(f"Local file not found and download disabled")

    # Filter to pothole patches
    df_filtered = filter_to_pothole_patches(df)

    return df_filtered


def filter_to_pothole_patches(df):
    """
    Filter notification activities to pothole patches only.

    Filters:
    - ACTIVITY_CODE_GROUP_TEXT = "ASPHALT"
    - ACTIVITY_CODE_TEXT = "POTHOLE PATCHED (EA)"

    Parameters
    ----------
    df : pd.DataFrame
        Raw notification activities data

    Returns
    -------
    pd.DataFrame
        Filtered data with parsed dates
    """
    print(f"\n[Filtering] Total records: {len(df):,}")

    # Filter to pothole patches
    mask = (
        (df['ACTIVITY_CODE_GROUP_TEXT'] == 'ASPHALT') &
        (df['ACTIVITY_CODE_TEXT'] == 'POTHOLE PATCHED (EA)')
    )

    df_pothole = df[mask].copy()
    print(f"[Filtering] Pothole patch records: {len(df_pothole):,}")

    # Parse dates
    print(f"\n[Parsing dates]")
    # Use NOTIFICATION_DATE for pothole occurrence (Y_it)
    df_pothole['date_reported'] = pd.to_datetime(
        df_pothole['NOTIFICATION_DATE'],
        errors='coerce'
    )
    # Use COMPLETION_DATE for fix status (R_it)
    df_pothole['date_closed'] = pd.to_datetime(
        df_pothole['COMPLETION_DATE'],
        errors='coerce'
    )

    # Also keep activity dates for reference
    df_pothole['activity_start'] = pd.to_datetime(
        df_pothole['CREATED_ON_DATE_FOR_ACTIVITY'],
        errors='coerce'
    )
    df_pothole['activity_end'] = pd.to_datetime(
        df_pothole['END_DATE_FOR_ACTIVITY'],
        errors='coerce'
    )

    # Clean segment ID
    df_pothole['segment_id'] = df_pothole['FUNCTIONAL_LOCATION'].astype(str).str.strip()

    # Filter to records with segment ID
    df_pothole = df_pothole[df_pothole['segment_id'].notna()].copy()
    print(f"[Filtering] Records with segment ID: {len(df_pothole):,}")

    # Add week calculations (Saturday-based, based on notification date)
    df_pothole['week_start'] = df_pothole['date_reported'].apply(get_week_start)

    print(f"\n[Summary]")
    print(f"  Unique segments: {df_pothole['segment_id'].nunique():,}")
    print(f"  Date range (notification): {df_pothole['date_reported'].min()} to {df_pothole['date_reported'].max()}")
    print(f"  Records with close date: {df_pothole['date_closed'].notna().sum():,} ({df_pothole['date_closed'].notna().mean()*100:.1f}%)")

    return df_pothole


def get_week_start(date):
    """
    Get the Saturday start of the week for a given date.

    Week definition: Saturday (day 0) to Friday (day 6)

    Parameters
    ----------
    date : pd.Timestamp
        Any date

    Returns
    -------
    pd.Timestamp
        The Saturday that starts that week
    """
    if pd.isna(date):
        return pd.NaT

    # Python weekday: Monday=0, Sunday=6
    # We want Saturday as week start
    days_since_saturday = (date.weekday() + 2) % 7
    week_start = date - timedelta(days=days_since_saturday)

    return pd.Timestamp(week_start.date())


def get_bundles_with_potholes_in_period(
    activities_df,
    bundles_df,
    start_date,
    end_date,
    segment_col='segment_id'
):
    """
    Get bundles that had at least one pothole reported in a date range.

    Uses NOTIFICATION_DATE (date_reported) for determining when pothole occurred.

    Parameters
    ----------
    activities_df : pd.DataFrame
        Pothole activities (filtered)
    bundles_df : pd.DataFrame
        Bundle-segment mapping
    start_date : datetime-like
        Start of period (inclusive)
    end_date : datetime-like
        End of period (inclusive)
    segment_col : str
        Column name for segment ID in bundles_df

    Returns
    -------
    set
        Set of bundle_ids that had potholes reported in the period
    """
    # Filter to potholes reported in the period (using NOTIFICATION_DATE)
    mask = (
        (activities_df['date_reported'] >= start_date) &
        (activities_df['date_reported'] <= end_date)
    )

    potholes_in_period = activities_df[mask]

    # Get unique segments with potholes
    segments_with_potholes = set(potholes_in_period['segment_id'].unique())

    if len(segments_with_potholes) == 0:
        return set()

    # Map to bundles
    bundles_with_potholes = set(
        bundles_df[
            bundles_df[segment_col].isin(segments_with_potholes)
        ]['bundle_id'].unique()
    )

    return bundles_with_potholes


def get_eligible_bundles_for_date(
    current_date,
    activities_df,
    bundles_df,
    segment_col='segment_id'
):
    """
    Get eligible bundles for a given date.

    Eligible = had at least one pothole in the preceding week

    Parameters
    ----------
    current_date : datetime-like
        The date to determine eligibility for
    activities_df : pd.DataFrame
        Pothole activities (filtered)
    bundles_df : pd.DataFrame
        Bundle-segment mapping
    segment_col : str
        Column name for segment ID in bundles_df

    Returns
    -------
    set
        Set of eligible bundle_ids
    """
    # Get preceding week's date range
    current_week_start = get_week_start(pd.Timestamp(current_date))
    preceding_week_start = current_week_start - timedelta(days=7)
    preceding_week_end = current_week_start - timedelta(days=1)  # Friday

    # Get bundles with potholes in preceding week
    eligible_bundles = get_bundles_with_potholes_in_period(
        activities_df=activities_df,
        bundles_df=bundles_df,
        start_date=preceding_week_start,
        end_date=preceding_week_end,
        segment_col=segment_col
    )

    return eligible_bundles


def build_panel_from_activities(activities_df, start_year=2021, end_year=2025):
    """
    Build segment-week panel from activities data.

    Creates Y_it (pothole occurrence) and R_it (fix status) variables.

    Parameters
    ----------
    activities_df : pd.DataFrame
        Filtered pothole activities
    start_year : int
        First year to include
    end_year : int
        Last year to include

    Returns
    -------
    pd.DataFrame
        Panel data with Y_it and R_it
    """
    print(f"\n{'='*70}")
    print(f"Building Panel Data ({start_year}-{end_year})")
    print(f"{'='*70}")

    # Filter to date range (based on notification date)
    activities_filtered = activities_df[
        (activities_df['date_reported'].dt.year >= start_year) &
        (activities_df['date_reported'].dt.year <= end_year)
    ].copy()

    print(f"\nPotholes reported in {start_year}-{end_year}: {len(activities_filtered):,}")

    # Get all unique segments and weeks
    segments = activities_filtered['segment_id'].unique()

    # Generate all weeks in range
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")

    # Get Saturday starts for all weeks
    all_weeks = pd.date_range(
        start=get_week_start(start_date),
        end=get_week_start(end_date),
        freq='W-SAT'
    )

    print(f"Segments: {len(segments):,}")
    print(f"Weeks: {len(all_weeks)}")

    # Create full panel (all segment-week combinations)
    print(f"\nCreating full panel...")
    panel_index = pd.MultiIndex.from_product(
        [segments, all_weeks],
        names=['segment_id', 'week_start']
    )

    panel = pd.DataFrame(index=panel_index).reset_index()
    print(f"Panel size: {len(panel):,} segment-weeks")

    # Calculate Y_it (pothole occurrence)
    print(f"\nCalculating Y_it (pothole occurrence)...")
    activities_by_seg_week = (
        activities_filtered
        .groupby(['segment_id', 'week_start'])
        .size()
        .reset_index(name='pothole_count')
    )

    panel = panel.merge(
        activities_by_seg_week,
        on=['segment_id', 'week_start'],
        how='left'
    )

    panel['Y_it'] = (panel['pothole_count'] > 0).astype(int)
    panel['pothole_count'] = panel['pothole_count'].fillna(0).astype(int)

    print(f"  Segment-weeks with potholes: {panel['Y_it'].sum():,} ({panel['Y_it'].mean()*100:.2f}%)")

    # Calculate R_it (fix status)
    print(f"\nCalculating R_it (fix status)...")

    # For each segment-week with pothole, check if ALL potholes fixed by Friday
    def calculate_fix_status(group):
        """
        Check if all potholes in this segment-week were closed by Friday.

        Uses COMPLETION_DATE (date_closed) for determining fix status.
        """
        week_start = group['week_start'].iloc[0]
        week_friday = week_start + timedelta(days=6)

        # All potholes must have close date <= Friday
        all_fixed = (
            group['date_closed'].notna() &
            (group['date_closed'] <= week_friday)
        ).all()

        return 1 if all_fixed else 0

    fix_status = (
        activities_filtered[activities_filtered['week_start'].notna()]
        .groupby(['segment_id', 'week_start'])
        .apply(calculate_fix_status)
        .reset_index(name='R_it')
    )

    panel = panel.merge(
        fix_status,
        on=['segment_id', 'week_start'],
        how='left'
    )

    # R_it is only defined for Y_it=1
    panel.loc[panel['Y_it'] == 0, 'R_it'] = np.nan

    print(f"  Pothole-weeks fixed by Friday: {panel['R_it'].sum():.0f} ({panel['R_it'].sum() / panel['Y_it'].sum() * 100:.1f}% of potholes)")

    return panel


if __name__ == "__main__":
    # Test data loading
    print("Testing data fetcher...")

    activities = fetch_latest_notification_activities(use_local=True)

    print(f"\n{'='*70}")
    print("Sample data:")
    print(activities[['segment_id', 'date_reported', 'date_closed', 'week_start']].head(10))

    # Build panel
    panel = build_panel_from_activities(activities, start_year=2021, end_year=2025)

    print(f"\n{'='*70}")
    print("Panel summary:")
    print(panel.head(20))
    print(f"\nY_it distribution:")
    print(panel['Y_it'].value_counts())
    print(f"\nR_it distribution (among Y_it=1):")
    print(panel[panel['Y_it']==1]['R_it'].value_counts())
