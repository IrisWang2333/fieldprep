#!/usr/bin/env python
"""
Simulate Experiment Using 2025 Historical Data

New Design:
- Day 1: 30 DH bundles (4 conditional + 26 random), NO D2DS
- Day 2+:
  - DH: 6 bundles (4 conditional + 2 random)
  - D2DS: 6 bundles (4 from DH conditional + 2 random)

Conditional = bundle had pothole in preceding week (based on 2025 historical data)
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
        # Remove DH bundles just sampled from available pool for D2DS random selection
        all_available_for_d2ds = all_bundles - used_bundles

        d2ds_selection = select_d2ds_bundles(
            conditional_bundles=dh_sample['conditional'],
            all_bundles=all_available_for_d2ds,
            bundles_df=bundles,
            n_from_conditional=4,
            n_random=2,
            seed=rng.integers(0, 1e9),
            segment_col='segment_id'
        )

        # Mark DH conditional bundles that are also used for D2DS
        for bundle_id in d2ds_selection['d2ds_conditional']:
            # Update existing DH record
            for record in plan_records:
                if (record['week'] == week_num and
                    record['bundle_id'] == bundle_id and
                    record['task'] == 'DH'):
                    record['is_d2ds'] = True

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
print(f"    - DH: 30 bundles (4 conditional + 26 random)")
print(f"    - D2DS: 0 bundles")
print(f"  Week 2-{N_WEEKS}:")
print(f"    - DH: 6 bundles/week (4 conditional + 2 random)")
print(f"    - D2DS: 6 bundles/week (4 from DH conditional + 2 random)")

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

print(f"\nOutput files:")
print(f"  - {plan_file}")
print(f"  - {summary_file}")
