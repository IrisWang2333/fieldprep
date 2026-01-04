#!/usr/bin/env python
"""
Multi-Day Experiment Simulation Using 2025 Historical Data

Simulate a 30-week field experiment using 2025 historical pothole data.

New Design (Single Layer Randomization):
- Week 1: 30 DH bundles (4 conditional + 26 random), NO D2DS
- Week 2+: Each week:
  - DH: 6 bundles (4 conditional + 2 random)
  - D2DS: 6 bundles (4 from PREVIOUS week's DH conditional + 2 random)

IMPORTANT: D2DS conditional bundles come from the PREVIOUS week's DH conditional bundles,
not the current week's. This ensures D2DS revisits areas that were identified as problematic
in the previous week.

Conditional = bundle had at least one pothole in preceding week (based on 2025 data)
Sampling: Without replacement across weeks
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import sys
from datetime import datetime, timedelta

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from utils.data_fetcher import (
    fetch_latest_notification_activities,
    get_eligible_bundles_for_date
)
from utils.sampling import (
    sample_dh_bundles,
    select_d2ds_bundles,
    get_week_start
)


def simulate_multiday_experiment(
    n_weeks: int = 30,
    n_interviewers: int = 6,
    bundle_file: str = "outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet",
    addr_assignment_file: str = "outputs/sweep/locked/segment_addresses_b40_m2.parquet",
    pothole_file: str = None,
    output_dir: str = None,
    start_date: str = "2025-01-06",
    list_code: int = 30,
    seed: int = 42,
):
    """
    Simulate multi-week experiment using 2025 historical data.

    New Design:
    - Week 1: 30 DH bundles (4 conditional + 26 random), NO D2DS
    - Week 2+: 6 DH (4 conditional + 2 random) + 6 D2DS (4 from PREVIOUS week's DH conditional + 2 random)

    IMPORTANT: D2DS conditional bundles come from the PREVIOUS week's DH conditional bundles.

    Args:
        n_weeks: Total number of weeks (default: 30)
        n_interviewers: Number of interviewers (default: 6)
        bundle_file: Path to bundle parquet
        addr_assignment_file: Path to address assignment parquet
        pothole_file: Not used (loads from notification_activities.csv)
        output_dir: Output directory (default: outputs/simulation/)
        start_date: Start date - Monday (YYYY-MM-DD, default: 2025-01-06)
        list_code: List code for all assignments
        seed: Random seed for reproducibility
    """
    from sd311_fieldprep.utils import paths

    root, cfg, out_root = paths()

    print("="*70)
    print("EXPERIMENT SIMULATION USING 2025 HISTORICAL DATA")
    print("="*70)

    # Load bundles
    bundle_path = Path(bundle_file)
    if not bundle_path.is_absolute():
        bundle_path = root / bundle_file
    bundles = gpd.read_parquet(bundle_path)

    # Load address assignments to count addresses per bundle
    addr_path = Path(addr_assignment_file)
    if not addr_path.is_absolute():
        addr_path = root / addr_assignment_file
    addr_assign = pd.read_parquet(addr_path)

    # Detect segment ID column and standardize
    seg_col = 'iamfloc' if 'iamfloc' in bundles.columns else 'segment_id'
    bundles['segment_id'] = bundles[seg_col].astype(str)

    # Count addresses per bundle (via segment)
    bundle_segments = bundles.groupby('bundle_id')['segment_id'].apply(set).to_dict()
    bundle_addr_counts = {}
    for bid, segs in bundle_segments.items():
        seg_strs = {str(s) for s in segs}
        n_addrs = addr_assign[addr_assign['segment_id'].astype(str).isin(seg_strs)].shape[0]
        bundle_addr_counts[bid] = n_addrs

    print(f"\n[Data Loading] Loaded {len(bundles['bundle_id'].unique())} bundles")
    print(f"[Data Loading] Address counts calculated for {len(bundle_addr_counts)} bundles")

    # Load 2025 pothole activities
    print(f"\n[Data Loading] Loading 2025 pothole activities...")
    activities = fetch_latest_notification_activities(use_local=True)

    # Filter to 2025 only
    activities_2025 = activities[
        (activities['date_reported'].dt.year == 2025)
    ].copy()

    print(f"[Data Loading] Loaded {len(activities_2025):,} pothole records from 2025")
    print(f"[Data Loading] Date range: {activities_2025['date_reported'].min()} to {activities_2025['date_reported'].max()}")

    # Get all bundle IDs
    all_bundles = set(bundles['bundle_id'].unique())

    # Pre-compute eligible bundles for each week
    print(f"\n{'='*70}")
    print("PRE-COMPUTING ELIGIBLE BUNDLES (Conditional on Preceding Week)")
    print(f"{'='*70}")

    # Generate experiment dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    experiment_weeks = []
    for i in range(n_weeks):
        week_date = start_dt + timedelta(weeks=i)
        experiment_weeks.append(week_date)

    print(f"\nExperiment period: {experiment_weeks[0].date()} to {experiment_weeks[-1].date()}")

    # For each week, determine which bundles are eligible
    print(f"\nCalculating eligible bundles for each week...")
    eligible_by_week = {}
    for i, current_date in enumerate(experiment_weeks):
        week_start = get_week_start(pd.Timestamp(current_date))

        # Get eligible bundles for this date
        eligible = get_eligible_bundles_for_date(
            current_date=current_date,
            activities_df=activities_2025,
            bundles_df=bundles,
            segment_col='segment_id'
        )

        eligible_by_week[week_start] = eligible

        if i < 5:
            print(f"  Week {i+1} ({current_date.date()}): {len(eligible)} eligible bundles")

    avg_eligible = np.mean([len(e) for e in eligible_by_week.values()])
    print(f"\n  Average eligible bundles per week: {avg_eligible:.1f}")
    print(f"  Min: {min(len(e) for e in eligible_by_week.values())}")
    print(f"  Max: {max(len(e) for e in eligible_by_week.values())}")

    # Check if we have enough eligible bundles
    min_needed_conditional = 4
    weeks_insufficient = sum(1 for e in eligible_by_week.values() if len(e) < min_needed_conditional)
    if weeks_insufficient > 0:
        print(f"\n  WARNING: {weeks_insufficient} weeks have < {min_needed_conditional} eligible bundles!")
    else:
        print(f"\n  ✓ All weeks have sufficient eligible bundles")

    # Calculate total bundles needed
    week1_dh = 30  # 4 conditional + 26 random
    weekly_dh = 6  # 4 conditional + 2 random
    total_dh = week1_dh + (n_weeks - 1) * weekly_dh

    # D2DS: Week 2+ only, 2 new bundles per week (4 are reused from DH conditional)
    total_d2ds_new = (n_weeks - 1) * 2

    total_unique_bundles = total_dh + total_d2ds_new

    print(f"\n{'='*70}")
    print(f"Bundle Requirements")
    print(f"{'='*70}")
    print(f"  Week 1 DH: {week1_dh}")
    print(f"  Week 2-{n_weeks} DH: {(n_weeks - 1) * weekly_dh}")
    print(f"  Total DH bundles: {total_dh}")
    print(f"  Week 2-{n_weeks} D2DS (new): {total_d2ds_new}")
    print(f"  Total unique bundles needed: {total_unique_bundles}")

    if len(all_bundles) < total_unique_bundles:
        raise ValueError(
            f"Not enough bundles! Need {total_unique_bundles} but only have {len(all_bundles)}"
        )

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Track used bundles (no reuse across weeks)
    used_bundles = set()
    plan_records = []
    stats = []

    # Generate interviewer labels
    interviewer_labels = [chr(ord('A') + i) for i in range(n_interviewers)]

    # Generate dates (one date per week)
    dates = [w.strftime("%Y-%m-%d") for w in experiment_weeks]

    print(f"\n{'='*70}")
    print("RUNNING SIMULATION")
    print(f"{'='*70}")

    # Week 1: 30 DH bundles (4 conditional + 26 random), NO D2DS
    week_num = 1
    current_date = experiment_weeks[0]
    week_start = get_week_start(pd.Timestamp(current_date))

    eligible = eligible_by_week.get(week_start, set())
    eligible_available = eligible - used_bundles
    all_available = all_bundles - used_bundles

    print(f"\n{'='*70}")
    print(f"Week {week_num}: {current_date.date()}")
    print(f"{'='*70}")
    print(f"  Eligible bundles available: {len(eligible_available)}")
    print(f"  Total bundles available: {len(all_available)}")

    # Sample DH bundles for Week 1
    dh_sample = sample_dh_bundles(
        current_date=current_date,
        eligible_bundles=eligible_available,
        all_bundles=all_available,
        is_day_1=True,
        seed=rng.integers(0, 1e9)
    )

    # Add DH bundles to plan
    day1_rows = []
    for bundle_id in dh_sample['conditional']:
        plan_records.append({
            'week': week_num,
            'date': dates[0],
            'bundle_id': bundle_id,
            'bundle_type': 'dh_conditional',
            'task': 'DH',
            'is_d2ds': False,
            'interviewer': None,  # Will assign later
            'list_code': list_code,
        })
        used_bundles.add(bundle_id)

    for bundle_id in dh_sample['random']:
        plan_records.append({
            'week': week_num,
            'date': dates[0],
            'bundle_id': bundle_id,
            'bundle_type': 'dh_random',
            'task': 'DH',
            'is_d2ds': False,
            'interviewer': None,
            'list_code': list_code,
        })
        used_bundles.add(bundle_id)

    # Count week 1 addresses
    week1_bundles = dh_sample['all_sampled']
    week1_dh_addrs = sum(bundle_addr_counts.get(b, 0) for b in week1_bundles)
    stats.append({
        "week": 1,
        "date": dates[0],
        "dh_bundles": len(week1_bundles),
        "d2ds_bundles": 0,
        "dh_addresses": week1_dh_addrs,
        "d2ds_addresses": 0,
        "total_addresses": week1_dh_addrs,
        "cumulative_dh": week1_dh_addrs,
        "cumulative_d2ds": 0,
        "cumulative_total": week1_dh_addrs,
    })

    # Week 2+: 6 DH (4 conditional + 2 random) + 6 D2DS (4 from DH conditional + 2 random)
    for week_num in range(2, n_weeks + 1):
        current_date = experiment_weeks[week_num - 1]
        date = dates[week_num - 1]
        week_start = get_week_start(pd.Timestamp(current_date))

        eligible = eligible_by_week.get(week_start, set())
        eligible_available = eligible - used_bundles
        all_available = all_bundles - used_bundles

        print(f"\n{'='*70}")
        print(f"Week {week_num}: {current_date.date()}")
        print(f"{'='*70}")
        print(f"  Eligible bundles available: {len(eligible_available)}")
        print(f"  Total bundles available: {len(all_available)}")

        # Sample DH bundles
        dh_sample = sample_dh_bundles(
            current_date=current_date,
            eligible_bundles=eligible_available,
            all_bundles=all_available,
            is_day_1=False,
            seed=rng.integers(0, 1e9)
        )

        # Add DH bundles to plan
        for bundle_id in dh_sample['conditional']:
            plan_records.append({
                'week': week_num,
                'date': date,
                'bundle_id': bundle_id,
                'bundle_type': 'dh_conditional',
                'task': 'DH',
                'is_d2ds': False,  # Will update below if used for D2DS
                'interviewer': None,
                'list_code': list_code,
            })
            used_bundles.add(bundle_id)

        for bundle_id in dh_sample['random']:
            plan_records.append({
                'week': week_num,
                'date': date,
                'bundle_id': bundle_id,
                'bundle_type': 'dh_random',
                'task': 'DH',
                'is_d2ds': False,
                'interviewer': None,
                'list_code': list_code,
            })
            used_bundles.add(bundle_id)

        # CRITICAL: D2DS conditional bundles should come from PREVIOUS week's DH conditional
        # NOT from current week's DH conditional!
        prev_week_conditional = [
            r['bundle_id'] for r in plan_records
            if r['week'] == week_num - 1 and r['bundle_type'] == 'dh_conditional'
        ]

        print(f"  Previous week (Week {week_num-1}): {len(prev_week_conditional)} DH conditional bundles")

        # Select D2DS bundles
        # Remove DH bundles just sampled from available pool for D2DS random selection
        all_available_for_d2ds = all_bundles - used_bundles

        d2ds_selection = select_d2ds_bundles(
            conditional_bundles=prev_week_conditional,  # ← Use PREVIOUS week's conditional
            all_bundles=all_available_for_d2ds,
            bundles_df=bundles,
            n_from_conditional=4,
            n_random=2,
            seed=rng.integers(0, 1e9),
            segment_col='segment_id'
        )

        # Add D2DS conditional bundles (from PREVIOUS week's DH conditional)
        # These are separate D2DS task records, not combined with current week's DH
        for bundle_id in d2ds_selection['d2ds_conditional']:
            plan_records.append({
                'week': week_num,
                'date': date,
                'bundle_id': bundle_id,
                'bundle_type': 'd2ds_conditional',  # From previous week's DH conditional
                'task': 'D2DS',
                'is_d2ds': True,
                'interviewer': None,
                'list_code': list_code,
            })
            # Note: These bundles are NOT marked as used_bundles again,
            # as they were already used in the previous week's DH

        # Add D2DS random bundles (these are NEW, not in DH)
        for bundle_id in d2ds_selection['d2ds_random']:
            plan_records.append({
                'week': week_num,
                'date': date,
                'bundle_id': bundle_id,
                'bundle_type': 'd2ds_random',
                'task': 'D2DS',
                'is_d2ds': True,
                'interviewer': None,
                'list_code': list_code,
            })
            used_bundles.add(bundle_id)

        # Count addresses for this week
        week_dh_bundles = dh_sample['all_sampled']
        week_d2ds_bundles = d2ds_selection['d2ds_all']
        week_dh_addrs = sum(bundle_addr_counts.get(b, 0) for b in week_dh_bundles)
        week_d2ds_addrs = sum(bundle_addr_counts.get(b, 0) for b in week_d2ds_bundles)

        prev_stats = stats[-1]
        stats.append({
            "week": week_num,
            "date": date,
            "dh_bundles": len(week_dh_bundles),
            "d2ds_bundles": len(week_d2ds_bundles),
            "dh_addresses": week_dh_addrs,
            "d2ds_addresses": week_d2ds_addrs,
            "total_addresses": week_dh_addrs + week_d2ds_addrs,
            "cumulative_dh": prev_stats["cumulative_dh"] + week_dh_addrs,
            "cumulative_d2ds": prev_stats["cumulative_d2ds"] + week_d2ds_addrs,
            "cumulative_total": prev_stats["cumulative_total"] + week_dh_addrs + week_d2ds_addrs,
        })

    # Create DataFrames
    plan_df = pd.DataFrame(plan_records)
    stats_df = pd.DataFrame(stats)

    # Set output directory
    if output_dir is None:
        output_dir = out_root / "simulation"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write outputs
    plan_file = output_dir / f"plan_{n_weeks}weeks.csv"
    stats_file = output_dir / f"stats_{n_weeks}weeks.csv"
    overlap_file = output_dir / f"overlap_{n_weeks}weeks.csv"

    plan_df.to_csv(plan_file, index=False)
    stats_df.to_csv(stats_file, index=False)

    print(f"\n{'='*70}")
    print("SAVING OUTPUTS")
    print(f"{'='*70}")
    print(f"Wrote plan to {plan_file}")
    print(f"Wrote stats to {stats_file}")

    # Analyze overlap between DH and D2DS
    dh_bundles = set(plan_df[plan_df['task'] == 'DH']['bundle_id'])
    d2ds_bundles = set(plan_df[plan_df['task'] == 'D2DS']['bundle_id'])
    both_bundles = dh_bundles & d2ds_bundles
    dh_only_bundles = dh_bundles - d2ds_bundles
    d2ds_only_bundles = d2ds_bundles - dh_bundles

    # Simulate DH treatment allocation for all DH bundles
    print(f"\n[simulation] Simulating DH treatment allocation...")

    dh_treatment_results = {
        'control': {'addresses': [], 'segments': set()},
        'full': {'addresses': [], 'segments': set()},
        'partial': {'addresses': [], 'segments': set()},
    }

    for bundle_id in dh_bundles:
        if bundle_id not in bundle_segments:
            continue

        # Get all segments in this bundle
        bundle_segs = list(bundle_segments[bundle_id])
        n_segs = len(bundle_segs)

        if n_segs == 0:
            continue

        # Round down to multiple of 4 for clean 50%/25%/25% split at SEGMENT level
        n_usable_segs = (n_segs // 4) * 4
        n_control_segs = n_usable_segs // 2  # 50% of segments
        n_full_segs = n_usable_segs // 4     # 25% of segments
        n_partial_segs = n_usable_segs // 4  # 25% of segments

        # Create deterministic random seed for this bundle
        bundle_seed = hash((seed, bundle_id)) % (2**32)
        bundle_rng = np.random.default_rng(bundle_seed)

        # Shuffle segment IDs
        seg_array = np.array(bundle_segs)
        seg_indices = bundle_rng.permutation(len(seg_array))

        # Assign segments to treatment groups
        control_seg_idx = seg_indices[:n_control_segs]
        full_seg_idx = seg_indices[n_control_segs:n_control_segs + n_full_segs]
        partial_seg_idx = seg_indices[n_control_segs + n_full_segs:n_usable_segs]

        # Handle remainder segments: randomly assign each to control (50%), full (25%), or partial (25%)
        if n_segs > n_usable_segs:
            remainder_seg_idx = seg_indices[n_usable_segs:]

            for idx in remainder_seg_idx:
                rand_val = bundle_rng.random()
                if rand_val < 0.5:  # 50% probability
                    control_seg_idx = np.append(control_seg_idx, idx)
                elif rand_val < 0.75:  # 25% probability (0.5 to 0.75)
                    full_seg_idx = np.append(full_seg_idx, idx)
                else:  # 25% probability (0.75 to 1.0)
                    partial_seg_idx = np.append(partial_seg_idx, idx)

        # Get actual segment IDs
        control_segs = [str(seg_array[i]) for i in control_seg_idx]
        full_segs = [str(seg_array[i]) for i in full_seg_idx]
        partial_segs = [str(seg_array[i]) for i in partial_seg_idx]

        # Get all addresses for each segment group
        control_addrs = addr_assign[addr_assign['segment_id'].astype(str).isin(control_segs)]
        full_addrs = addr_assign[addr_assign['segment_id'].astype(str).isin(full_segs)]
        partial_addrs = addr_assign[addr_assign['segment_id'].astype(str).isin(partial_segs)]

        # Store addresses and segments
        dh_treatment_results['control']['addresses'].extend(control_addrs.index.tolist())
        dh_treatment_results['full']['addresses'].extend(full_addrs.index.tolist())
        dh_treatment_results['partial']['addresses'].extend(partial_addrs.index.tolist())

        # Add segments
        dh_treatment_results['control']['segments'].update(control_segs)
        dh_treatment_results['full']['segments'].update(full_segs)
        dh_treatment_results['partial']['segments'].update(partial_segs)

    # Count results
    control_n_addrs = len(dh_treatment_results['control']['addresses'])
    full_n_addrs = len(dh_treatment_results['full']['addresses'])
    partial_n_addrs = len(dh_treatment_results['partial']['addresses'])

    control_n_segs = len(dh_treatment_results['control']['segments'])
    full_n_segs = len(dh_treatment_results['full']['segments'])
    partial_n_segs = len(dh_treatment_results['partial']['segments'])

    print(f"[simulation] DH treatment allocation complete:")
    print(f"  Control: {control_n_addrs:,} addresses, {control_n_segs:,} segments")
    print(f"  Full: {full_n_addrs:,} addresses, {full_n_segs:,} segments")
    print(f"  Partial: {partial_n_addrs:,} addresses, {partial_n_segs:,} segments")

    # Count addresses and segments for each category
    def count_addrs_segs(bundle_set):
        """Count addresses and segments for a set of bundles"""
        if not bundle_set:
            return 0, 0
        # Get segments from bundles
        segs = set()
        for bid in bundle_set:
            if bid in bundle_segments:
                segs.update(bundle_segments[bid])
        # Count addresses
        n_addrs = sum(bundle_addr_counts.get(bid, 0) for bid in bundle_set)
        return n_addrs, len(segs)

    dh_only_addrs, dh_only_segs = count_addrs_segs(dh_only_bundles)
    d2ds_only_addrs, d2ds_only_segs = count_addrs_segs(d2ds_only_bundles)
    both_addrs, both_segs = count_addrs_segs(both_bundles)

    # Calculate total segments and addresses across all used bundles
    total_segs_used = dh_only_segs + d2ds_only_segs + both_segs
    total_addrs_used = dh_only_addrs + d2ds_only_addrs + both_addrs

    # Calculate total segments and addresses in ALL bundles (after bundling)
    # This is the correct denominator for percentages
    all_bundle_segs = set()
    all_bundle_addrs = 0
    for bid in all_bundles:  # All bundles, not just used ones
        if bid in bundle_segments:
            all_bundle_segs.update(bundle_segments[bid])
        all_bundle_addrs += bundle_addr_counts.get(bid, 0)

    total_segs_all = len(all_bundle_segs)
    total_addrs_all = all_bundle_addrs
    total_bundles_all = len(all_bundles)

    print(f"[simulation] Total available after bundling: {total_bundles_all:,} bundles, {total_segs_all:,} segments, {total_addrs_all:,} addresses")

    # Create overlap summary DataFrame
    overlap_df = pd.DataFrame([
        {
            "category": "DH only",
            "bundles": len(dh_only_bundles),
            "segments": dh_only_segs,
            "addresses": dh_only_addrs,
            "pct_of_bundles": round(len(dh_only_bundles) / total_bundles_all * 100, 1),
            "pct_of_segments": round(dh_only_segs / total_segs_all * 100, 1),
            "pct_of_addresses": round(dh_only_addrs / total_addrs_all * 100, 1),
        },
        {
            "category": "D2DS only",
            "bundles": len(d2ds_only_bundles),
            "segments": d2ds_only_segs,
            "addresses": d2ds_only_addrs,
            "pct_of_bundles": round(len(d2ds_only_bundles) / total_bundles_all * 100, 1),
            "pct_of_segments": round(d2ds_only_segs / total_segs_all * 100, 1),
            "pct_of_addresses": round(d2ds_only_addrs / total_addrs_all * 100, 1),
        },
        {
            "category": "Both DH & D2DS",
            "bundles": len(both_bundles),
            "segments": both_segs,
            "addresses": both_addrs,
            "pct_of_bundles": round(len(both_bundles) / total_bundles_all * 100, 1),
            "pct_of_segments": round(both_segs / total_segs_all * 100, 1),
            "pct_of_addresses": round(both_addrs / total_addrs_all * 100, 1),
        },
        {
            "category": "DH full",
            "bundles": "",
            "segments": full_n_segs,
            "addresses": full_n_addrs,
            "pct_of_bundles": "",
            "pct_of_segments": round(full_n_segs / total_segs_all * 100, 1),
            "pct_of_addresses": round(full_n_addrs / total_addrs_all * 100, 1),
        },
        {
            "category": "DH partial",
            "bundles": "",
            "segments": partial_n_segs,
            "addresses": partial_n_addrs,
            "pct_of_bundles": "",
            "pct_of_segments": round(partial_n_segs / total_segs_all * 100, 1),
            "pct_of_addresses": round(partial_n_addrs / total_addrs_all * 100, 1),
        },
        {
            "category": "Total (used)",
            "bundles": len(dh_bundles | d2ds_bundles),
            "segments": total_segs_used,
            "addresses": total_addrs_used,
            "pct_of_bundles": round(len(dh_bundles | d2ds_bundles) / total_bundles_all * 100, 1),
            "pct_of_segments": round(total_segs_used / total_segs_all * 100, 1),
            "pct_of_addresses": round(total_addrs_used / total_addrs_all * 100, 1),
        },
    ])

    # Save overlap summary
    overlap_df.to_csv(overlap_file, index=False)
    print(f"[simulation] Wrote overlap analysis to {overlap_file}")

    # Run balance checks
    print(f"\n[simulation] Running balance checks...")
    try:
        from sd311_fieldprep.simulation.balance_check import run_all_balance_checks

        all_used_bundles = dh_bundles | d2ds_bundles

        balance_df = run_all_balance_checks(
            dh_bundles=dh_bundles,
            d2ds_bundles=d2ds_bundles,
            all_bundles=all_used_bundles,
            treatment_results=dh_treatment_results,
            bundle_segments=bundle_segments,
            demographics_file=None  # Use default location
        )

        # Save detailed balance check results
        balance_file = output_dir / f"balance_check_{n_weeks}weeks.csv"
        balance_df.to_csv(balance_file, index=False)
        print(f"[simulation] Wrote detailed balance check to {balance_file}")

        # Create simplified summary tables (means only)
        for comparison in balance_df['comparison'].unique():
            comp_df = balance_df[balance_df['comparison'] == comparison].copy()

            # Create simplified table with only means
            summary = pd.DataFrame({
                'Variable': comp_df['variable'],
                comp_df['group1_label'].iloc[0]: comp_df['group1_mean'],
                comp_df['group2_label'].iloc[0]: comp_df['group2_mean'],
                'Difference': comp_df['difference'],
                'Std Diff': comp_df['std_diff'],
                'Balanced': comp_df['std_diff'].abs() < 0.1
            })

            # Generate filename from comparison name
            if 'All DH bundles' in comparison:
                filename = f"balance_level1_treatment_vs_control_{n_weeks}weeks.csv"
            elif 'survey sample' in comparison:
                filename = f"balance_level2_survey_sample_{n_weeks}weeks.csv"
            elif 'Representativeness' in comparison:
                filename = f"balance_level3_representativeness_{n_weeks}weeks.csv"
            else:
                continue

            summary_file = output_dir / filename
            summary.to_csv(summary_file, index=False)
            print(f"[simulation] Wrote {filename}")

    except FileNotFoundError as e:
        print(f"[simulation] Warning: Could not run balance checks - {e}")
    except Exception as e:
        print(f"[simulation] Warning: Balance check failed - {e}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"Simulation Summary ({n_weeks} weeks)")
    print(f"{'='*70}")
    print(f"\nBundles:")
    print(f"  Total unique bundles used: {len(set(plan_df['bundle_id']))}")
    print(f"  Total DH assignments: {len(plan_df[plan_df['task'] == 'DH'])}")
    print(f"  Total D2DS assignments: {len(plan_df[plan_df['task'] == 'D2DS'])}")
    print(f"  Bundles remaining: {len(all_bundles) - len(used_bundles)}")

    print(f"\nBundle Usage by Task:")
    print(f"  DH only: {len(dh_only_bundles)} bundles ({len(dh_only_bundles)/len(dh_bundles)*100:.1f}% of DH bundles)")
    print(f"  D2DS only: {len(d2ds_only_bundles)} bundles ({len(d2ds_only_bundles)/len(d2ds_bundles)*100:.1f}% of D2DS bundles)")
    print(f"  Both DH & D2DS: {len(both_bundles)} bundles ({len(both_bundles)/len(dh_bundles)*100:.1f}% of DH bundles)")

    print(f"\nSegment Coverage:")
    total_segs = dh_only_segs + d2ds_only_segs + both_segs
    print(f"  DH only: {dh_only_segs:,} segments ({dh_only_segs/total_segs*100:.1f}%)")
    print(f"  D2DS only: {d2ds_only_segs:,} segments ({d2ds_only_segs/total_segs*100:.1f}%)")
    print(f"  Both DH & D2DS: {both_segs:,} segments ({both_segs/total_segs*100:.1f}%)")
    print(f"  Total segments: {total_segs:,}")

    print(f"\nAddress Coverage (before DH treatment allocation):")
    total_addrs = dh_only_addrs + d2ds_only_addrs + both_addrs
    print(f"  DH only: {dh_only_addrs:,} addresses ({dh_only_addrs/total_addrs*100:.1f}%)")
    print(f"  D2DS only: {d2ds_only_addrs:,} addresses ({d2ds_only_addrs/total_addrs*100:.1f}%)")
    print(f"  Both DH & D2DS: {both_addrs:,} addresses ({both_addrs/total_addrs*100:.1f}%)")
    print(f"  Total addresses: {total_addrs:,}")

    print(f"\nAddresses (estimated, before DH treatment allocation):")
    final = stats_df.iloc[-1]
    print(f"  Total DH addresses: {int(final['cumulative_dh']):,}")
    print(f"  Total D2DS addresses: {int(final['cumulative_d2ds']):,}")
    print(f"  Grand total: {int(final['cumulative_total']):,}")

    print(f"\nDaily averages:")
    print(f"  DH addresses/day: {stats_df['dh_addresses'].mean():.1f}")
    print(f"  D2DS addresses/day: {stats_df['d2ds_addresses'].mean():.1f}")
    print(f"  Total addresses/day: {stats_df['total_addresses'].mean():.1f}")

    print(f"\nDH Treatment Allocation (actual):")
    print(f"  - Control (not visited): {control_n_addrs:,} addresses ({control_n_addrs/int(final['cumulative_dh'])*100:.1f}%)")
    print(f"  - Full (100% visited): {full_n_addrs:,} addresses ({full_n_addrs/int(final['cumulative_dh'])*100:.1f}%)")
    print(f"  - Partial (50% visited): {partial_n_addrs:,} addresses ({partial_n_addrs/int(final['cumulative_dh'])*100:.1f}%)")
    print(f"\nDH visited addresses: {full_n_addrs + partial_n_addrs:,} (full + partial assigned)")
    print(f"DH control addresses: {control_n_addrs:,} (not visited)")
    print(f"DH total experimental sample: {control_n_addrs + full_n_addrs + partial_n_addrs:,}")
    print(f"\nTotal addresses assigned: {control_n_addrs + full_n_addrs + partial_n_addrs + int(final['cumulative_d2ds']):,} (DH + D2DS)")

    return plan_df, stats_df, overlap_df


if __name__ == "__main__":
    plan, stats, overlap = simulate_multiday_experiment(
        n_weeks=30,
        n_interviewers=6,
        start_date="2025-01-06",
        seed=42
    )
