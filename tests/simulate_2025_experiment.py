#!/usr/bin/env python
"""
Simulate Experiment Using 2025 Historical Data

New Design:
- Week 1: 24 DH bundles (4 conditional + 20 random), NO D2DS
- Week 2+:
  - DH: 6 bundles (4 conditional + 2 random)
  - D2DS: 6 bundles (4 from PREVIOUS week's DH conditional + 2 random)

Conditional = bundle had pothole in preceding week (based on 2025 historical data)
Timeline: Week t-1 pothole → Week t DH selects → Week t+1 D2DS surveys
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_fetcher import (
    fetch_latest_notification_activities,
    get_eligible_bundles_for_date,
    build_panel_from_activities
)
from utils.sampling import (
    sample_dh_bundles,
    select_d2ds_bundles,
    get_week_start
)

# Paths
data_dir = Path("/Users/iris/Dropbox/sandiego code/data")
output_dir = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/simulation_2025")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("EXPERIMENT SIMULATION USING 2025 HISTORICAL DATA")
print("="*70)

# ============================================================================
# Configuration
# ============================================================================
START_DATE = "2025-01-06"  # Monday, January 6, 2025
N_WEEKS = 30  # 30 weeks of simulation
SEED = 42

print(f"\nConfiguration:")
print(f"  Start date: {START_DATE}")
print(f"  Number of weeks: {N_WEEKS}")
print(f"  Random seed: {SEED}")

# ============================================================================
# Load Data
# ============================================================================
print(f"\n{'='*70}")
print("STEP 1: Load Historical Data (2025)")
print(f"{'='*70}")

# Load pothole activities from 2025
print(f"\n[1.1] Loading pothole activities...")
activities = fetch_latest_notification_activities(use_local=True)

# Filter to 2025 only
activities_2025 = activities[
    (activities['date_reported'].dt.year == 2025)
].copy()

print(f"  Loaded {len(activities_2025):,} pothole records from 2025")
print(f"  Date range: {activities_2025['date_reported'].min()} to {activities_2025['date_reported'].max()}")

# Load bundles
print(f"\n[1.2] Loading bundles...")
bundle_file = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
bundles = gpd.read_parquet(bundle_file)

seg_col = 'iamfloc' if 'iamfloc' in bundles.columns else 'segment_id'
bundles['segment_id'] = bundles[seg_col].astype(str)

# Get all bundle IDs
all_bundles = set(bundles['bundle_id'].unique())
print(f"  Loaded {len(all_bundles)} bundles")
print(f"  Total segments: {bundles['segment_id'].nunique():,}")

# ============================================================================
# Pre-compute Eligible Bundles for Each Week
# ============================================================================
print(f"\n{'='*70}")
print("STEP 2: Pre-compute Eligible Bundles (Conditional on Preceding Week)")
print(f"{'='*70}")

# Generate experiment dates
start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
experiment_weeks = []
for i in range(N_WEEKS):
    week_date = start_dt + timedelta(weeks=i)
    experiment_weeks.append(week_date)

print(f"\nExperiment period: {experiment_weeks[0].date()} to {experiment_weeks[-1].date()}")

# For each week, determine which bundles are eligible
# Eligible = had at least one pothole in preceding week
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

    if i == 0:
        print(f"  Week 1 ({current_date.date()}): {len(eligible)} eligible bundles")
    elif i < 5:
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

# ============================================================================
# Run Simulation
# ============================================================================
print(f"\n{'='*70}")
print("STEP 3: Simulate Experiment")
print(f"{'='*70}")

rng = np.random.default_rng(SEED)
plan_records = []
used_bundles = set()  # Track which bundles have been used
prev_week_dh_conditional = set()  # Track previous week's DH conditional for D2DS

for week_num, current_date in enumerate(experiment_weeks, start=1):
    is_day_1 = (week_num == 1)
    week_start = get_week_start(pd.Timestamp(current_date))

    eligible = eligible_by_week.get(week_start, set())

    # Remove already used bundles from eligible pool
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
        is_day_1=is_day_1,
        seed=rng.integers(0, 1e9)
    )

    # Add DH bundles to plan
    for bundle_id in dh_sample['conditional']:
        plan_records.append({
            'week': week_num,
            'date': current_date.strftime("%Y-%m-%d"),
            'bundle_id': bundle_id,
            'bundle_type': 'dh_conditional',
            'task': 'DH',
            'is_d2ds': False
        })
        used_bundles.add(bundle_id)

    for bundle_id in dh_sample['random']:
        plan_records.append({
            'week': week_num,
            'date': current_date.strftime("%Y-%m-%d"),
            'bundle_id': bundle_id,
            'bundle_type': 'dh_random',
            'task': 'DH',
            'is_d2ds': False
        })
        used_bundles.add(bundle_id)

    # Select D2DS bundles (only for Week 2+)
    if not is_day_1:
        # CRITICAL: D2DS conditional bundles come from PREVIOUS week's DH conditional
        # Timeline: Week t-1 pothole → Week t-1 DH selects → Week t D2DS surveys
        print(f"  D2DS conditional bundles from previous week: {len(prev_week_dh_conditional)}")

        # Remove DH bundles just sampled from available pool for D2DS random selection
        all_available_for_d2ds = all_bundles - used_bundles

        d2ds_selection = select_d2ds_bundles(
            conditional_bundles=prev_week_dh_conditional,  # ← Use PREVIOUS week's DH conditional
            all_bundles=all_available_for_d2ds,
            bundles_df=bundles,
            n_from_conditional=4,
            n_random=2,
            seed=rng.integers(0, 1e9),
            segment_col='segment_id'
        )

        # Add D2DS conditional bundles (from previous week's DH)
        for bundle_id in d2ds_selection['d2ds_conditional']:
            plan_records.append({
                'week': week_num,
                'date': current_date.strftime("%Y-%m-%d"),
                'bundle_id': bundle_id,
                'bundle_type': 'd2ds_conditional',
                'task': 'D2DS',
                'is_d2ds': True
            })
            # Don't add to used_bundles - already used in previous week's DH

        # Add D2DS random bundles (these are NEW, not in DH)
        for bundle_id in d2ds_selection['d2ds_random']:
            plan_records.append({
                'week': week_num,
                'date': current_date.strftime("%Y-%m-%d"),
                'bundle_id': bundle_id,
                'bundle_type': 'd2ds_random',
                'task': 'D2DS',
                'is_d2ds': True
            })
            used_bundles.add(bundle_id)

    # Update previous week's DH conditional for next week's D2DS
    prev_week_dh_conditional = dh_sample['conditional']

# ============================================================================
# Create Plan DataFrame and Summary
# ============================================================================
print(f"\n{'='*70}")
print("STEP 4: Create Plan and Summary")
print(f"{'='*70}")

plan_df = pd.DataFrame(plan_records)

print(f"\nPlan summary:")
print(f"  Total records: {len(plan_df)}")
print(f"  Unique bundles used: {plan_df['bundle_id'].nunique()}")
print(f"  DH assignments: {(plan_df['task']=='DH').sum()}")
print(f"  D2DS assignments: {plan_df['is_d2ds'].sum()}")

print(f"\nBy bundle type:")
for btype in plan_df['bundle_type'].unique():
    count = (plan_df['bundle_type']==btype).sum()
    print(f"  {btype}: {count}")

# Save plan
plan_file = output_dir / f"plan_{N_WEEKS}weeks.csv"
plan_df.to_csv(plan_file, index=False)
print(f"\n✓ Saved plan to {plan_file}")

# Create weekly summary
weekly_summary = plan_df.groupby('week').agg({
    'bundle_id': 'count',
    'is_d2ds': 'sum'
}).reset_index()
weekly_summary.columns = ['week', 'total_assignments', 'd2ds_count']
weekly_summary['dh_count'] = weekly_summary['total_assignments'] - weekly_summary['d2ds_count']

print(f"\nWeekly summary (first 10 weeks):")
print(weekly_summary.head(10).to_string(index=False))

summary_file = output_dir / f"summary_{N_WEEKS}weeks.csv"
weekly_summary.to_csv(summary_file, index=False)
print(f"\n✓ Saved summary to {summary_file}")

# ============================================================================
# Final Summary
# ============================================================================
print(f"\n{'='*70}")
print("SIMULATION COMPLETE")
print(f"{'='*70}")

print(f"\nExperiment Design:")
print(f"  Week 1:")
print(f"    - DH: 24 bundles (4 conditional + 20 random)")
print(f"    - D2DS: 0 bundles")
print(f"  Week 2-{N_WEEKS}:")
print(f"    - DH: 6 bundles/week (4 conditional + 2 random)")
print(f"    - D2DS: 6 bundles/week (4 from PREVIOUS week's DH conditional + 2 random)")
print(f"  Timeline: Week t-1 pothole → Week t DH selects → Week t+1 D2DS surveys")

print(f"\nActual Results:")
week1_data = plan_df[plan_df['week']==1]
print(f"  Week 1:")
print(f"    - DH: {len(week1_data)} bundles")
print(f"      - Conditional: {(week1_data['bundle_type']=='dh_conditional').sum()}")
print(f"      - Random: {(week1_data['bundle_type']=='dh_random').sum()}")
print(f"    - D2DS: {week1_data['is_d2ds'].sum()} bundles")

if len(plan_df) > 30:
    week2_data = plan_df[plan_df['week']==2]
    print(f"  Week 2 (example):")
    print(f"    - DH: {(week2_data['task']=='DH').sum()} bundles")
    print(f"    - D2DS: {week2_data['is_d2ds'].sum()} bundles")

print(f"\nTotal bundles used: {plan_df['bundle_id'].nunique()}/{len(all_bundles)} ({plan_df['bundle_id'].nunique()/len(all_bundles)*100:.1f}%)")
print(f"Bundles remaining: {len(all_bundles) - plan_df['bundle_id'].nunique()}")

# ============================================================================
# Overlap Analysis
# ============================================================================
print(f"\n{'='*70}")
print("OVERLAP ANALYSIS")
print(f"{'='*70}")

# Get DH and D2DS bundles from plan
dh_bundles_set = set(plan_df[plan_df['task'] == 'DH']['bundle_id'])
d2ds_bundles_set = set(plan_df[plan_df['task'] == 'D2DS']['bundle_id'])

# Calculate overlaps
dh_only = dh_bundles_set - d2ds_bundles_set
d2ds_only = d2ds_bundles_set - dh_bundles_set
both = dh_bundles_set & d2ds_bundles_set
never_used = all_bundles - (dh_bundles_set | d2ds_bundles_set)

# Get bundle-segment mapping
bundle_segments = bundles.groupby('bundle_id')['segment_id'].apply(set).to_dict()

# Get segments for each category
dh_only_segments = set()
d2ds_only_segments = set()
both_segments = set()
never_used_segments = set()

for bundle_id in dh_only:
    if bundle_id in bundle_segments:
        dh_only_segments.update(bundle_segments[bundle_id])

for bundle_id in d2ds_only:
    if bundle_id in bundle_segments:
        d2ds_only_segments.update(bundle_segments[bundle_id])

for bundle_id in both:
    if bundle_id in bundle_segments:
        both_segments.update(bundle_segments[bundle_id])

for bundle_id in never_used:
    if bundle_id in bundle_segments:
        never_used_segments.update(bundle_segments[bundle_id])

# Load address assignments
try:
    addr_assign_file = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/sweep/locked/segment_addresses_b40_m2.parquet")
    addr_assign = pd.read_parquet(addr_assign_file)

    # Get addresses for each category
    dh_only_addrs = addr_assign[addr_assign['segment_id'].astype(str).isin([str(s) for s in dh_only_segments])]
    d2ds_only_addrs = addr_assign[addr_assign['segment_id'].astype(str).isin([str(s) for s in d2ds_only_segments])]
    both_addrs = addr_assign[addr_assign['segment_id'].astype(str).isin([str(s) for s in both_segments])]
    never_used_addrs = addr_assign[addr_assign['segment_id'].astype(str).isin([str(s) for s in never_used_segments])]

    n_dh_only_addrs = len(dh_only_addrs)
    n_d2ds_only_addrs = len(d2ds_only_addrs)
    n_both_addrs = len(both_addrs)
    n_never_used_addrs = len(never_used_addrs)

    has_addresses = True
except Exception as e:
    print(f"Warning: Could not load address data - {e}")
    n_dh_only_addrs = 0
    n_d2ds_only_addrs = 0
    n_both_addrs = 0
    n_never_used_addrs = 0
    has_addresses = False

# Calculate totals (ALL eligible bundles, not just used)
total_eligible_bundles = len(all_bundles)
total_eligible_segments = sum(len(bundle_segments.get(bid, set())) for bid in all_bundles)
total_eligible_addrs = len(addr_assign) if has_addresses else 0

overlap_data = []

# DH only
overlap_data.append({
    'Category': 'DH only',
    'Bundles': len(dh_only),
    'Bundles (%)': f"{len(dh_only)/total_eligible_bundles*100:.1f}%",
    'Segments': len(dh_only_segments),
    'Segments (%)': f"{len(dh_only_segments)/total_eligible_segments*100:.1f}%",
    'Addresses': n_dh_only_addrs if has_addresses else 'N/A',
    'Addresses (%)': f"{n_dh_only_addrs/total_eligible_addrs*100:.1f}%" if has_addresses and total_eligible_addrs > 0 else 'N/A'
})

# D2DS only
overlap_data.append({
    'Category': 'D2DS only',
    'Bundles': len(d2ds_only),
    'Bundles (%)': f"{len(d2ds_only)/total_eligible_bundles*100:.1f}%",
    'Segments': len(d2ds_only_segments),
    'Segments (%)': f"{len(d2ds_only_segments)/total_eligible_segments*100:.1f}%",
    'Addresses': n_d2ds_only_addrs if has_addresses else 'N/A',
    'Addresses (%)': f"{n_d2ds_only_addrs/total_eligible_addrs*100:.1f}%" if has_addresses and total_eligible_addrs > 0 else 'N/A'
})

# Both DH & D2DS
overlap_data.append({
    'Category': 'Both DH & D2DS',
    'Bundles': len(both),
    'Bundles (%)': f"{len(both)/total_eligible_bundles*100:.1f}%",
    'Segments': len(both_segments),
    'Segments (%)': f"{len(both_segments)/total_eligible_segments*100:.1f}%",
    'Addresses': n_both_addrs if has_addresses else 'N/A',
    'Addresses (%)': f"{n_both_addrs/total_eligible_addrs*100:.1f}%" if has_addresses and total_eligible_addrs > 0 else 'N/A'
})

# Never used
overlap_data.append({
    'Category': 'Never used',
    'Bundles': len(never_used),
    'Bundles (%)': f"{len(never_used)/total_eligible_bundles*100:.1f}%",
    'Segments': len(never_used_segments),
    'Segments (%)': f"{len(never_used_segments)/total_eligible_segments*100:.1f}%",
    'Addresses': n_never_used_addrs if has_addresses else 'N/A',
    'Addresses (%)': f"{n_never_used_addrs/total_eligible_addrs*100:.1f}%" if has_addresses and total_eligible_addrs > 0 else 'N/A'
})

# Total (all eligible)
overlap_data.append({
    'Category': 'Total Eligible',
    'Bundles': total_eligible_bundles,
    'Bundles (%)': '100.0%',
    'Segments': total_eligible_segments,
    'Segments (%)': '100.0%',
    'Addresses': total_eligible_addrs if has_addresses else 'N/A',
    'Addresses (%)': '100.0%' if has_addresses else 'N/A'
})

overlap_df = pd.DataFrame(overlap_data)

# Save overlap analysis
overlap_file = output_dir / f"overlap_{N_WEEKS}weeks.csv"
overlap_df.to_csv(overlap_file, index=False)
print(f"\n✓ Saved overlap analysis to {overlap_file}")

# Print overlap summary
print(f"\nOverlap Summary (% relative to all eligible bundles):")
print(overlap_df.to_string(index=False))

# ============================================================================
# Balance Checks
# ============================================================================
print(f"\n{'='*70}")
print("BALANCE CHECKS")
print(f"{'='*70}")

try:
    # Need to add parent path for imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from sd311_fieldprep.simulation.balance_check import run_all_balance_checks

    # Get bundle-segment mapping
    bundle_segments = bundles.groupby('bundle_id')['segment_id'].apply(set).to_dict()

    # Get all DH and D2DS bundles
    dh_bundles = set(plan_df[plan_df['task'] == 'DH']['bundle_id'])
    d2ds_bundles = set(plan_df[plan_df['task'] == 'D2DS']['bundle_id'])
    all_used_bundles = dh_bundles | d2ds_bundles

    # Load address assignments for treatment simulation
    addr_assign_file = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/sweep/locked/segment_addresses_b40_m2.parquet")
    addr_assign = pd.read_parquet(addr_assign_file)

    # Simulate DH treatment allocation
    print(f"\n[Balance Check] Simulating DH treatment allocation...")
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
        bundle_seed = hash((SEED, bundle_id)) % (2**32)
        bundle_rng = np.random.default_rng(bundle_seed)

        # Shuffle segment IDs
        seg_array = np.array(bundle_segs)
        seg_indices = bundle_rng.permutation(len(seg_array))

        # Assign segments to treatment groups
        control_seg_idx = seg_indices[:n_control_segs]
        full_seg_idx = seg_indices[n_control_segs:n_control_segs + n_full_segs]
        partial_seg_idx = seg_indices[n_control_segs + n_full_segs:n_usable_segs]

        # Handle remainder segments
        if n_segs > n_usable_segs:
            remainder_seg_idx = seg_indices[n_usable_segs:]
            for idx in remainder_seg_idx:
                rand_val = bundle_rng.random()
                if rand_val < 0.5:
                    control_seg_idx = np.append(control_seg_idx, idx)
                elif rand_val < 0.75:
                    full_seg_idx = np.append(full_seg_idx, idx)
                else:
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

        dh_treatment_results['control']['segments'].update(control_segs)
        dh_treatment_results['full']['segments'].update(full_segs)
        dh_treatment_results['partial']['segments'].update(partial_segs)

    print(f"[Balance Check] Treatment allocation complete")

    # Run balance checks
    print(f"[Balance Check] Running balance checks...")
    balance_df = run_all_balance_checks(
        dh_bundles=dh_bundles,
        d2ds_bundles=d2ds_bundles,
        all_bundles=all_used_bundles,
        treatment_results=dh_treatment_results,
        bundle_segments=bundle_segments,
        demographics_file=None  # Use default location
    )

    # Save detailed balance check results
    balance_file = output_dir / f"balance_check_{N_WEEKS}weeks.csv"
    balance_df.to_csv(balance_file, index=False)
    print(f"[Balance Check] Wrote detailed balance check to {balance_file}")

    # Create simplified summary tables with standard errors
    for comparison in balance_df['comparison'].unique():
        comp_df = balance_df[balance_df['comparison'] == comparison].copy()

        # Create summary table with means, standard errors, and significance
        summary = pd.DataFrame({
            'Variable': comp_df['variable'],
            comp_df['group1_label'].iloc[0]: comp_df['group1_mean'],
            f"N ({comp_df['group1_label'].iloc[0]})": comp_df['group1_n'],
            comp_df['group2_label'].iloc[0]: comp_df['group2_mean'],
            f"N ({comp_df['group2_label'].iloc[0]})": comp_df['group2_n'],
            'Difference': comp_df['difference'],
            'Std Error': comp_df['difference'] / comp_df['t_stat'].replace([np.inf, -np.inf], np.nan),  # SE = diff / t-stat
            'p-value': comp_df['p_value'],
            'Significant': comp_df['p_value'] < 0.05
        })

        # Generate filename from comparison name
        if 'All DH bundles' in comparison:
            filename = f"balance_level1_treatment_vs_control_{N_WEEKS}weeks.csv"
        elif 'survey sample' in comparison:
            filename = f"balance_level2_survey_sample_{N_WEEKS}weeks.csv"
        elif 'Representativeness' in comparison:
            filename = f"balance_level3_representativeness_{N_WEEKS}weeks.csv"
        else:
            continue

        summary_file = output_dir / filename
        summary.to_csv(summary_file, index=False)
        print(f"[Balance Check] Wrote {filename}")

        # Print summary to console
        print(f"\n{comparison}:")
        print(summary.to_string(index=False))

except FileNotFoundError as e:
    print(f"[Balance Check] Warning: Could not run balance checks - {e}")
except Exception as e:
    print(f"[Balance Check] Warning: Balance check failed - {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
print(f"Output files:")
print(f"  - {plan_file}")
print(f"  - {summary_file}")
print(f"  - {overlap_file}")
if 'balance_file' in locals():
    print(f"  - Balance checks: {output_dir}/balance_*.csv")
