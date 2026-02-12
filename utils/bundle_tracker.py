#!/usr/bin/env python
"""
Bundle Usage Tracker

Tracks which bundles have been used in previous plans to ensure:
1. No bundle is reused across different weeks (without replacement)
2. D2DS conditional bundles come from previous week's DH conditional bundles
3. Historical plan consistency
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Set, List, Tuple


def get_all_historical_plans(plan_dir: Path, min_date: str = None, max_date: str = None) -> List[Tuple[str, pd.DataFrame]]:
    """
    Get all historical plan CSV files.

    Parameters
    ----------
    plan_dir : Path
        Directory containing plan CSV files
    min_date : str, optional
        Minimum date (YYYY-MM-DD) to include. Plans before this date are excluded.
        Useful for separating pilot from official experiment.
    max_date : str, optional
        Maximum date (YYYY-MM-DD) to include. Plans after this date are excluded.
        Useful for limiting bundle tracking to specific phase (pilot or official).

    Returns
    -------
    list of (date_str, dataframe) tuples, sorted by date
    """
    plan_files = sorted(plan_dir.glob("bundles_plan_*.csv"))
    plans = []

    for pfile in plan_files:
        # Extract date from filename: bundles_plan_2026-01-03.csv
        date_str = pfile.stem.replace("bundles_plan_", "")

        try:
            # Validate date format
            plan_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Skip if before min_date
            if min_date:
                min_dt = datetime.strptime(min_date, "%Y-%m-%d")
                if plan_date < min_dt:
                    continue

            # Skip if after max_date
            if max_date:
                max_dt = datetime.strptime(max_date, "%Y-%m-%d")
                if plan_date > max_dt:
                    continue

            df = pd.read_csv(pfile)
            plans.append((date_str, df))
        except (ValueError, Exception) as e:
            print(f"  Skipping invalid plan file {pfile.name}: {e}")
            continue

    # Sort by date
    plans.sort(key=lambda x: x[0])

    return plans


def get_used_bundles(plan_dir: Path, exclude_date: str = None, min_date: str = None, max_date: str = None) -> Dict[str, Set[int]]:
    """
    Get all bundles used in historical plans.

    Parameters
    ----------
    plan_dir : Path
        Directory containing plan CSV files
    exclude_date : str, optional
        Date to exclude (e.g., current date being generated)
    min_date : str, optional
        Minimum date (YYYY-MM-DD) to include. Plans before this date are excluded.
        Useful for separating pilot from official experiment.
    max_date : str, optional
        Maximum date (YYYY-MM-DD) to include. Plans after this date are excluded.
        Useful for limiting bundle tracking to specific phase (pilot or official).

    Returns
    -------
    dict
        {
            'all': set of all used bundle IDs,
            'dh': set of DH bundle IDs,
            'd2ds': set of D2DS bundle IDs,
            'by_date': {date_str: {
                'dh': [...],
                'dh_conditional': [...],
                'dh_random': [...],
                'd2ds': [...],
                'd2ds_conditional': [...],
                'd2ds_random': [...]
            }}
        }
    """
    plans = get_all_historical_plans(plan_dir, min_date=min_date, max_date=max_date)

    used = {
        'all': set(),
        'dh': set(),
        'd2ds': set(),
        'by_date': {}
    }

    for date_str, df in plans:
        # Skip if this is the date we're excluding
        if exclude_date and date_str == exclude_date:
            continue

        # Get bundles for this date
        dh_bundles = df[df['task'].str.upper() == 'DH']['bundle_id'].tolist()
        d2ds_bundles = df[df['task'].str.upper() == 'D2DS']['bundle_id'].tolist()

        # Update global sets
        used['all'].update(dh_bundles)
        used['all'].update(d2ds_bundles)
        used['dh'].update(dh_bundles)
        used['d2ds'].update(d2ds_bundles)

        # Store by date (we'll need this to get previous week's conditional bundles)
        used['by_date'][date_str] = {
            'dh': dh_bundles,
            'd2ds': d2ds_bundles
        }

    return used


def get_previous_week_conditional_bundles(
    plan_dir: Path,
    current_date: str,
    activities_df: pd.DataFrame,
    bundles_df: pd.DataFrame,
    min_date: str = None,
    max_date: str = None
) -> Set[int]:
    """
    Get conditional DH bundles from previous week's plan.

    For Week 2+, D2DS conditional bundles should come from
    the previous week's DH conditional bundles.

    Parameters
    ----------
    plan_dir : Path
        Directory containing plan CSV files
    current_date : str
        Current date (YYYY-MM-DD)
    activities_df : pd.DataFrame
        Pothole activities data (for fallback)
    bundles_df : pd.DataFrame
        Bundle data (for fallback)
    min_date : str, optional
        Minimum date (YYYY-MM-DD) to include. Plans before this date are excluded.
    max_date : str, optional
        Maximum date (YYYY-MM-DD) to include. Plans after this date are excluded.
        Useful for limiting to same experiment phase (pilot or official).

    Returns
    -------
    set of bundle IDs that were conditional DH in previous week
    """
    plans = get_all_historical_plans(plan_dir, min_date=min_date, max_date=max_date)

    if not plans:
        print("  No historical plans found, cannot determine previous week's conditional bundles")
        return set()

    # Find the most recent plan before current_date
    current_dt = datetime.strptime(current_date, "%Y-%m-%d")
    previous_plans = [(date_str, df) for date_str, df in plans
                      if datetime.strptime(date_str, "%Y-%m-%d") < current_dt]

    if not previous_plans:
        print("  No previous plans found, cannot determine previous week's conditional bundles")
        return set()

    # Get the most recent previous plan
    prev_date_str, prev_df = previous_plans[-1]

    # Get DH bundles from previous week
    prev_dh_bundles = prev_df[prev_df['task'].str.upper() == 'DH']['bundle_id'].tolist()

    # Determine which were conditional (had potholes in their preceding week)
    # This requires checking pothole data for the week before prev_date
    prev_dt = datetime.strptime(prev_date_str, "%Y-%m-%d")

    # Import here to avoid circular dependency
    from utils.data_fetcher import get_eligible_bundles_for_date

    eligible_for_prev = get_eligible_bundles_for_date(
        current_date=prev_dt,
        activities_df=activities_df,
        bundles_df=bundles_df,
        segment_col='segment_id'
    )

    # Debug: Print eligible bundles info
    print(f"  DEBUG: Found {len(eligible_for_prev)} eligible bundles for {prev_date_str}")
    print(f"  DEBUG: prev_dh_bundles types: {[type(b).__name__ for b in list(prev_dh_bundles)[:3]]}")
    print(f"  DEBUG: eligible_for_prev types: {[type(b).__name__ for b in list(eligible_for_prev)[:3]] if eligible_for_prev else 'empty'}")
    print(f"  DEBUG: prev_dh_bundles values: {sorted(list(prev_dh_bundles))}")
    print(f"  DEBUG: eligible_for_prev sample (first 10): {sorted(list(eligible_for_prev))[:10]}")

    # Conditional bundles = prev week's DH bundles that were also eligible
    # Explicitly convert both to int to avoid type mismatch (CSV→int, parquet→float64)
    prev_dh_set = set(int(b) for b in prev_dh_bundles)
    eligible_set = set(int(b) for b in eligible_for_prev)

    print(f"  DEBUG: After int conversion - prev_dh_set: {sorted(prev_dh_set)}")
    print(f"  DEBUG: After int conversion - eligible_set sample: {sorted(list(eligible_set))[:10]}")

    conditional_from_prev = prev_dh_set & eligible_set

    print(f"  Previous week ({prev_date_str}): {len(prev_dh_bundles)} DH bundles")
    print(f"  Of those, {len(conditional_from_prev)} were conditional (had potholes)")
    if conditional_from_prev:
        print(f"  Conditional bundle IDs: {sorted(conditional_from_prev)}")

    return conditional_from_prev


def filter_available_bundles(
    all_bundles: Set[int],
    used_tracker: Dict[str, Set[int]],
    exempt_bundles: Set[int] = None
) -> Set[int]:
    """
    Filter out bundles that have been used in previous plans.

    Parameters
    ----------
    all_bundles : set
        All possible bundle IDs
    used_tracker : dict
        Output from get_used_bundles()
    exempt_bundles : set, optional
        Bundle IDs that are exempt from the without-replacement rule.
        These bundles will be available even if they were used before.
        Used for D2DS conditional reuse of previous week's DH conditional bundles.

    Returns
    -------
    set of bundle IDs that haven't been used yet (plus any exempt bundles)
    """
    exempt_bundles = exempt_bundles or set()

    # Allow bundles that are either unused OR exempted
    # This supports D2DS conditional reuse while maintaining without-replacement for others
    available = (all_bundles - used_tracker['all']) | (exempt_bundles & all_bundles)

    n_total = len(all_bundles)
    n_used = len(used_tracker['all'])
    n_exempt = len(exempt_bundles & all_bundles)
    n_available = len(available)

    print(f"\n[Bundle Availability]")
    print(f"  Total bundles: {n_total}")
    print(f"  Used in previous plans: {n_used}")
    if n_exempt > 0:
        print(f"  Exempt from without-replacement: {n_exempt} (for D2DS conditional reuse)")
    print(f"  Available for sampling: {n_available}")

    if n_available < 30:  # Less than 2 weeks worth
        print(f"  ⚠️  WARNING: Only {n_available} bundles remain! Consider expanding bundle pool.")

    return available


def print_usage_summary(used_tracker: Dict[str, Set[int]]):
    """Print a summary of bundle usage across all plans."""
    print(f"\n{'='*70}")
    print("HISTORICAL BUNDLE USAGE SUMMARY")
    print(f"{'='*70}")
    print(f"Total unique bundles used: {len(used_tracker['all'])}")
    print(f"  DH bundles: {len(used_tracker['dh'])}")
    print(f"  D2DS bundles: {len(used_tracker['d2ds'])}")
    print(f"\nBreakdown by date:")

    for date_str in sorted(used_tracker['by_date'].keys()):
        date_data = used_tracker['by_date'][date_str]
        dh_count = len(date_data['dh'])
        d2ds_count = len(date_data['d2ds'])
        print(f"  {date_str}: {dh_count} DH, {d2ds_count} D2DS")


if __name__ == "__main__":
    # Test the bundle tracker
    from pathlib import Path

    plan_dir = Path("outputs/plans")

    if plan_dir.exists():
        print("Testing bundle tracker...")

        # Get all used bundles
        used = get_used_bundles(plan_dir)
        print_usage_summary(used)

        # Test filtering
        all_bundles = set(range(1, 1628))  # Assuming 1627 total bundles
        available = filter_available_bundles(all_bundles, used)

        print(f"\n{'='*70}")
        print(f"Available bundles: {len(available)}")
        print(f"Sample of available: {sorted(list(available))[:10]}")
    else:
        print(f"Plan directory not found: {plan_dir}")
