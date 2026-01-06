#!/usr/bin/env python
"""
D2DS Feasibility Analysis

Analyzes whether the D2DS design is feasible given:
1. Conditional eligibility (bundles must have pothole in preceding week)
2. Sufficient unfixed segments across weeks
3. Timing of fixes relative to survey dates
4. Historical pothole exposure and fix rates for D2DS segments
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Paths
data_dir = Path("/Users/iris/Dropbox/sandiego code/data")
output_dir = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/pothole_model")
analysis_output_dir = output_dir / "d2ds_feasibility"
analysis_output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("D2DS FEASIBILITY ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1] Loading data...")

# Load historical pothole data with panel
panel = pd.read_parquet(output_dir / "panel_with_outcomes.parquet")
print(f"  Historical panel: {len(panel):,} segment-weeks")

# Load D2DS plan
plan_file = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/simulation/plan_30days.csv")
plan_df = pd.read_csv(plan_file)
plan_df['date'] = pd.to_datetime(plan_df['date'])

# Get D2DS bundles
d2ds_plan = plan_df[plan_df['task'] == 'D2DS'].copy()
d2ds_bundles = set(d2ds_plan['bundle_id'].unique())
print(f"  D2DS bundles: {len(d2ds_bundles)}")

# Load bundle-segment mapping
bundle_file = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
bundles = gpd.read_parquet(bundle_file)
seg_col = 'iamfloc' if 'iamfloc' in bundles.columns else 'segment_id'
bundles['segment_id'] = bundles[seg_col].astype(str)

# Identify DH bundles (assuming bundles with 'DH' in task or bundle_id >= 5000)
dh_bundles = set(bundles[bundles['bundle_id'] >= 5000]['bundle_id'].unique())
print(f"  DH bundles: {len(dh_bundles)}")
print(f"  Non-DH bundles: {len(bundles['bundle_id'].unique()) - len(dh_bundles)}")

# Get segments for each bundle
bundle_segments = bundles.groupby('bundle_id')['segment_id'].apply(list).to_dict()

# ============================================================================
# ANALYSIS 1: Bundle Eligibility (Conditional on Preceding Week Potholes)
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 1: Bundle Eligibility per Week")
print("="*70)
print("\nQuestion: How many bundles are eligible each week if we require")
print("          at least one segment to have a pothole in the PRECEDING week?")

# For each week in historical data, find bundles with potholes
panel['bundle_id'] = panel['segment_id'].map(
    bundles.set_index('segment_id')['bundle_id'].to_dict()
)

# Get bundles with at least one pothole in each week
bundles_with_potholes_by_week = (
    panel[panel['Y_it'] == 1]
    .groupby(['week_start', 'bundle_id'])
    .size()
    .reset_index(name='pothole_count')
)

# For each week t, find bundles eligible (had pothole in week t-1)
all_weeks = sorted(panel['week_start'].unique())
eligibility_results = []

for i, week in enumerate(all_weeks):
    if i == 0:  # Skip first week (no preceding week)
        continue

    preceding_week = all_weeks[i-1]

    # Bundles with potholes in preceding week
    eligible_bundles = set(
        bundles_with_potholes_by_week[
            bundles_with_potholes_by_week['week_start'] == preceding_week
        ]['bundle_id'].unique()
    )

    # Split by DH vs non-DH
    eligible_dh = eligible_bundles & dh_bundles
    eligible_non_dh = eligible_bundles - dh_bundles

    eligibility_results.append({
        'week': week,
        'total_eligible': len(eligible_bundles),
        'eligible_dh': len(eligible_dh),
        'eligible_non_dh': len(eligible_non_dh)
    })

eligibility_df = pd.DataFrame(eligibility_results)

print(f"\n[1.1] Weekly Eligibility Statistics (2021-2025):")
print(f"  Mean eligible bundles per week: {eligibility_df['total_eligible'].mean():.1f}")
print(f"  Mean eligible DH bundles: {eligibility_df['eligible_dh'].mean():.1f}")
print(f"  Mean eligible non-DH bundles: {eligibility_df['eligible_non_dh'].mean():.1f}")
print(f"  Min eligible bundles: {eligibility_df['total_eligible'].min()}")
print(f"  Weeks with <4 DH bundles: {(eligibility_df['eligible_dh'] < 4).sum()} / {len(eligibility_df)}")
print(f"  Weeks with <6 total bundles: {(eligibility_df['total_eligible'] < 6).sum()} / {len(eligibility_df)}")

# Save summary
eligibility_summary_file = analysis_output_dir / "bundle_eligibility_by_week.csv"
eligibility_df.to_csv(eligibility_summary_file, index=False)
print(f"\n[1.2] Saved eligibility data to {eligibility_summary_file}")

# Visualize
print(f"\n[1.3] Creating eligibility visualization...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Top: Time series
ax1.plot(eligibility_df['week'], eligibility_df['total_eligible'],
         label='Total Eligible', linewidth=2, color='steelblue')
ax1.plot(eligibility_df['week'], eligibility_df['eligible_dh'],
         label='DH Eligible', linewidth=2, color='green', alpha=0.7)
ax1.plot(eligibility_df['week'], eligibility_df['eligible_non_dh'],
         label='Non-DH Eligible', linewidth=2, color='orange', alpha=0.7)
ax1.axhline(y=6, color='red', linestyle='--', linewidth=1.5,
           label='Required: 6 bundles/week')
ax1.axhline(y=4, color='darkgreen', linestyle='--', linewidth=1.5,
           label='Required: 4 DH bundles/week')
ax1.set_xlabel('Week', fontsize=12)
ax1.set_ylabel('Number of Eligible Bundles', fontsize=12)
ax1.set_title('Bundle Eligibility Over Time\n(Conditional on Pothole in Preceding Week)',
             fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Bottom: Distribution
bins = np.arange(0, eligibility_df['total_eligible'].max() + 2, 1)
ax2.hist(eligibility_df['total_eligible'], bins=bins, alpha=0.6,
         color='steelblue', edgecolor='black', label='Total Eligible')
ax2.hist(eligibility_df['eligible_dh'], bins=bins, alpha=0.6,
         color='green', edgecolor='black', label='DH Eligible')
ax2.axvline(x=6, color='red', linestyle='--', linewidth=2, label='Required: 6')
ax2.axvline(x=4, color='darkgreen', linestyle='--', linewidth=2, label='Required: 4 (DH)')
ax2.set_xlabel('Number of Eligible Bundles', fontsize=12)
ax2.set_ylabel('Number of Weeks', fontsize=12)
ax2.set_title('Distribution of Eligible Bundles per Week', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_file = analysis_output_dir / "bundle_eligibility_analysis.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved figure to {fig_file}")

# ============================================================================
# ANALYSIS 2: Unfixed Segments Availability
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 2: Unfixed Segments Availability")
print("="*70)
print("\nQuestion: Do we run out of unfixed segments in bundles?")

# For each bundle-week with potholes, count unfixed segments
unfixed_by_bundle_week = []

for (bundle_id, week), group in panel[panel['Y_it'] == 1].groupby(['bundle_id', 'week_start']):
    total_with_potholes = len(group)
    unfixed = (group['R_it'] == 0).sum()

    unfixed_by_bundle_week.append({
        'bundle_id': bundle_id,
        'week_start': week,
        'total_potholes': total_with_potholes,
        'unfixed': unfixed,
        'unfixed_rate': unfixed / total_with_potholes if total_with_potholes > 0 else 0,
        'is_dh': bundle_id in dh_bundles
    })

unfixed_df = pd.DataFrame(unfixed_by_bundle_week)

print(f"\n[2.1] Unfixed Segments Statistics:")
print(f"  Total bundle-weeks with potholes: {len(unfixed_df):,}")
print(f"  Mean unfixed segments per bundle-week: {unfixed_df['unfixed'].mean():.2f}")
print(f"  Median unfixed segments: {unfixed_df['unfixed'].median():.1f}")
print(f"  Mean unfixed rate: {unfixed_df['unfixed_rate'].mean():.1%}")
print(f"  Bundle-weeks with 0 unfixed: {(unfixed_df['unfixed'] == 0).sum()} ({(unfixed_df['unfixed'] == 0).mean():.1%})")
print(f"  Bundle-weeks with ≥1 unfixed: {(unfixed_df['unfixed'] >= 1).sum()} ({(unfixed_df['unfixed'] >= 1).mean():.1%})")

# Save
unfixed_summary_file = analysis_output_dir / "unfixed_segments_by_bundle_week.csv"
unfixed_df.to_csv(unfixed_summary_file, index=False)
print(f"\n[2.2] Saved unfixed segments data to {unfixed_summary_file}")

# Visualize
print(f"\n[2.3] Creating unfixed segments visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Distribution of unfixed segments
ax1.hist(unfixed_df['unfixed'], bins=30, color='coral', edgecolor='black', alpha=0.7)
ax1.axvline(x=unfixed_df['unfixed'].mean(), color='red', linestyle='--',
           linewidth=2, label=f"Mean: {unfixed_df['unfixed'].mean():.1f}")
ax1.axvline(x=unfixed_df['unfixed'].median(), color='blue', linestyle='--',
           linewidth=2, label=f"Median: {unfixed_df['unfixed'].median():.1f}")
ax1.set_xlabel('Number of Unfixed Segments', fontsize=12)
ax1.set_ylabel('Number of Bundle-Weeks', fontsize=12)
ax1.set_title('Distribution of Unfixed Segments per Bundle-Week',
             fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Right: Cumulative availability over experiment period
# Simulate sampling from these bundles
unfixed_df_sorted = unfixed_df.sort_values('week_start')
cumulative_unfixed = []
weeks = sorted(unfixed_df['week_start'].unique())

for week in weeks:
    available = unfixed_df[unfixed_df['week_start'] == week]['unfixed'].sum()
    cumulative_unfixed.append({
        'week': week,
        'available_unfixed': available
    })

cum_df = pd.DataFrame(cumulative_unfixed)
ax2.plot(cum_df['week'], cum_df['available_unfixed'], linewidth=2, color='steelblue')
ax2.fill_between(cum_df['week'], 0, cum_df['available_unfixed'], alpha=0.3, color='steelblue')
ax2.axhline(y=6, color='red', linestyle='--', linewidth=2,
           label='Weekly demand: 6 bundles')
ax2.set_xlabel('Week', fontsize=12)
ax2.set_ylabel('Total Unfixed Segments Available', fontsize=12)
ax2.set_title('Weekly Supply of Unfixed Segments', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_file = analysis_output_dir / "unfixed_segments_analysis.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved figure to {fig_file}")

# Create time series of fixed vs unfixed SEGMENTS (tracking potholes from week t-2 to week t)
print(f"\n[2.4] Creating fixed vs unfixed time series...")

# Load pothole data (POTHOLE PATCHED activities only)
potholes = pd.read_parquet(output_dir / "pothole_data_2021_2025.parquet")

# Create segment to bundle mapping
segment_to_bundle = bundles.set_index('segment_id')['bundle_id'].to_dict()

# For each week t, track segments from week t-2 and check if ALL their potholes are fixed by week t
# Timeline: Week t-2 pothole appears → Week t-1 DH selects → Week t D2DS surveys
# R_it=1 (fixed) means ALL potholes on that segment are fixed by week t
fixed_unfixed_time_series = []

for i, week in enumerate(all_weeks):
    if i < 2:  # Skip first 2 weeks (need t-2)
        continue

    week_t_minus_2 = all_weeks[i-2]
    week_t = week

    # Calculate date ranges
    week_t_minus_2_start = pd.to_datetime(week_t_minus_2)
    week_t_minus_2_end = week_t_minus_2_start + pd.Timedelta(days=7)
    week_t_saturday = pd.to_datetime(week_t) + pd.Timedelta(days=6)  # Saturday of week t

    # Get potholes reported in week t-2
    potholes_t_minus_2 = potholes[
        (potholes['date_requested'] >= week_t_minus_2_start) &
        (potholes['date_requested'] < week_t_minus_2_end)
    ].copy()

    # Filter to segments in DH bundles only
    potholes_t_minus_2['bundle_id'] = potholes_t_minus_2['segment_id'].map(segment_to_bundle)
    potholes_t_minus_2 = potholes_t_minus_2[potholes_t_minus_2['bundle_id'].isin(dh_bundles)]

    if len(potholes_t_minus_2) == 0:
        fixed_unfixed_time_series.append({
            'week_start': week_t,
            'fixed': 0,
            'unfixed': 0,
            'total': 0
        })
        continue

    # Check fix status at week t Saturday
    # Fixed if date_closed exists and is before week t Saturday
    potholes_t_minus_2['is_fixed_by_week_t'] = (
        potholes_t_minus_2['date_closed'].notna() &
        (potholes_t_minus_2['date_closed'] < week_t_saturday)
    )

    # Group by segment: R_it=1 only if ALL potholes on segment are fixed
    segment_fix_status = potholes_t_minus_2.groupby('segment_id').agg({
        'is_fixed_by_week_t': 'all'  # ALL potholes must be fixed
    }).reset_index()

    # Count fixed vs unfixed SEGMENTS
    fixed_count = segment_fix_status['is_fixed_by_week_t'].sum()
    unfixed_count = (~segment_fix_status['is_fixed_by_week_t']).sum()

    fixed_unfixed_time_series.append({
        'week_start': week_t,
        'fixed': fixed_count,
        'unfixed': unfixed_count,
        'total': len(segment_fix_status)
    })

fixed_unfixed_by_week = pd.DataFrame(fixed_unfixed_time_series)

# Create wide figure
fig, ax = plt.subplots(1, 1, figsize=(16, 5))

# Plot lines with fill
ax.plot(fixed_unfixed_by_week['week_start'], fixed_unfixed_by_week['fixed'],
        label='Fixed (R_it=1)', linewidth=2, color='green', alpha=0.8)
ax.plot(fixed_unfixed_by_week['week_start'], fixed_unfixed_by_week['unfixed'],
        label='Unfixed (R_it=0)', linewidth=2, color='red', alpha=0.8)

# Fill between
ax.fill_between(fixed_unfixed_by_week['week_start'], fixed_unfixed_by_week['fixed'],
                alpha=0.3, color='green', label='_nolegend_')
ax.fill_between(fixed_unfixed_by_week['week_start'], fixed_unfixed_by_week['unfixed'],
                alpha=0.3, color='red', label='_nolegend_')

ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('Number of Segments', fontsize=12)
ax.set_title('Fixed vs Unfixed Segments Over Time\n(Tracking Week t-2 Segments from DH Bundles to Week t)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_file = analysis_output_dir / "fixed_vs_unfixed_time_series.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved figure to {fig_file}")

# ============================================================================
# ANALYSIS 3: D2DS Segments - Fixed Before D2DS Survey
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 3: D2DS Segments Fixed Before D2DS Survey")
print("="*70)
print("\nQuestion: Among D2DS segments, how many potholes are fixed before D2DS survey?")
print("Note: D2DS surveys happen 2 weeks after pothole appeared")
print("      Timeline: Pothole Week N-1 → DH Week N → D2DS Survey Week N+1")

# Load simulation results
sim_panel = pd.read_parquet(output_dir / "pothole_simulation_2026.parquet")
d2ds_exposure = pd.read_csv(output_dir / "d2ds_pothole_exposure.csv")

# Map D2DS plan to weeks
d2ds_plan['day_of_week'] = d2ds_plan['date'].dt.dayofweek
d2ds_plan['days_since_saturday'] = (d2ds_plan['day_of_week'] + 2) % 7
d2ds_plan['week_start'] = d2ds_plan['date'] - pd.to_timedelta(d2ds_plan['days_since_saturday'], unit='D')
d2ds_plan['week_start'] = d2ds_plan['week_start'].dt.normalize()

# For D2DS segments with potholes, check if fixed before Saturday
# Load raw pothole data to check fix timing
potholes = pd.read_parquet(output_dir / "pothole_data_2021_2025.parquet")

# Get D2DS segments
d2ds_segments = set(d2ds_exposure['segment_id'].unique())

# Filter to D2DS segments
d2ds_potholes = potholes[potholes['segment_id'].isin(d2ds_segments)].copy()

# CRITICAL: D2DS conditional bundles come from PREVIOUS week's DH conditional
# Timeline: Pothole in Week N-1 → DH selects in Week N → D2DS surveys in Week N+1
# D2DS survey happens 2 weeks after pothole appeared
d2ds_potholes['pothole_week_saturday'] = d2ds_potholes['week_start']  # Week pothole appeared
d2ds_potholes['d2ds_survey_saturday'] = d2ds_potholes['week_start'] + pd.Timedelta(days=14)  # D2DS survey (2 weeks later)

# Check if fixed before D2DS survey Saturday (the critical contamination point)
# If fixed before D2DS survey, we won't observe the pothole's effect
d2ds_potholes['fixed_before_d2ds_survey'] = (
    (d2ds_potholes['date_closed'].notna()) &
    (d2ds_potholes['date_closed'] < d2ds_potholes['d2ds_survey_saturday'])
).astype(int)

# Check fix timing
d2ds_potholes['days_to_fix'] = (
    d2ds_potholes['date_closed'] - d2ds_potholes['date_requested']
).dt.total_seconds() / (24 * 3600)

print(f"\n[3.1] D2DS Segments Pothole Fix Timing (Historical 2021-2025):")
print(f"  Total potholes in D2DS segments: {len(d2ds_potholes):,}")
print(f"  Potholes with fix data: {d2ds_potholes['date_closed'].notna().sum():,}")
print(f"\n  Timeline: Pothole appears Week N-1 → DH selects Week N → D2DS surveys Week N+1")
print(f"  D2DS survey happens 2 weeks after pothole appeared")
print(f"  Fixed before D2DS survey Saturday: {d2ds_potholes['fixed_before_d2ds_survey'].sum():,} "
      f"({d2ds_potholes['fixed_before_d2ds_survey'].mean():.1%})")
print(f"\n  Mean days to fix: {d2ds_potholes['days_to_fix'].mean():.1f}")
print(f"  Median days to fix: {d2ds_potholes['days_to_fix'].median():.1f}")
print(f"\n  Note: If fixed before D2DS survey, we can't observe pothole's effect (contamination)")

# By week of year pattern
d2ds_potholes['week_of_year'] = d2ds_potholes['week_start'].dt.isocalendar().week
fix_timing_by_week = d2ds_potholes.groupby('week_of_year').agg({
    'fixed_before_d2ds_survey': 'mean',
    'days_to_fix': 'mean'
}).reset_index()

# Save
fix_timing_file = analysis_output_dir / "d2ds_fix_timing.csv"
d2ds_potholes[['segment_id', 'week_start', 'pothole_week_saturday', 'd2ds_survey_saturday',
               'date_requested', 'date_closed',
               'days_to_fix', 'fixed_before_d2ds_survey']].to_csv(fix_timing_file, index=False)
print(f"\n[3.2] Saved fix timing data to {fix_timing_file}")

# Create fix timing visualization
print(f"\n[3.3] Creating fix timing visualization...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Distribution of days to fix
d2ds_potholes_with_fix = d2ds_potholes[d2ds_potholes['date_closed'].notna()]
ax1.hist(d2ds_potholes_with_fix['days_to_fix'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(x=14, color='red', linestyle='--', linewidth=2, label='D2DS survey (14 days)')
ax1.axvline(x=d2ds_potholes_with_fix['days_to_fix'].mean(), color='orange', linestyle='--',
           linewidth=2, label=f"Mean: {d2ds_potholes_with_fix['days_to_fix'].mean():.1f} days")
ax1.axvline(x=d2ds_potholes_with_fix['days_to_fix'].median(), color='green', linestyle='--',
           linewidth=2, label=f"Median: {d2ds_potholes_with_fix['days_to_fix'].median():.1f} days")
ax1.set_xlabel('Days to Fix', fontsize=11)
ax1.set_ylabel('Number of Potholes', fontsize=11)
ax1.set_title('Distribution of Days to Fix\n(D2DS Segments, Historical)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Top-right: Fix rate before D2DS survey by week of year
ax2.plot(fix_timing_by_week['week_of_year'], fix_timing_by_week['fixed_before_d2ds_survey'],
        linewidth=2, color='coral', marker='o', markersize=4, alpha=0.7)
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='50% threshold')
ax2.set_xlabel('Week of Year', fontsize=11)
ax2.set_ylabel('Fixed Before D2DS Survey Rate', fontsize=11)
ax2.set_title('Contamination Risk by Week of Year\n(Fixed Before D2DS Survey)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

# Bottom-left: Cumulative distribution of days to fix
sorted_days = np.sort(d2ds_potholes_with_fix['days_to_fix'].dropna())
cumulative_pct = np.arange(1, len(sorted_days) + 1) / len(sorted_days)
ax3.plot(sorted_days, cumulative_pct, linewidth=2, color='steelblue')
ax3.axvline(x=14, color='red', linestyle='--', linewidth=2, label='D2DS survey (14 days)')
ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='50%')
ax3.set_xlabel('Days to Fix', fontsize=11)
ax3.set_ylabel('Cumulative Percentage', fontsize=11)
ax3.set_title('Cumulative Distribution of Days to Fix\n(D2DS Segments)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 60)

# Bottom-right: Fixed vs Not Fixed before D2DS survey
fix_status = pd.DataFrame({
    'Status': ['Not Fixed\nbefore D2DS', 'Fixed\nbefore D2DS'],
    'Count': [
        (d2ds_potholes['fixed_before_d2ds_survey'] == 0).sum(),
        (d2ds_potholes['fixed_before_d2ds_survey'] == 1).sum()
    ]
})
colors = ['lightgreen', 'lightcoral']
bars = ax4.bar(fix_status['Status'], fix_status['Count'], color=colors, edgecolor='black', alpha=0.7)
ax4.set_ylabel('Number of Potholes', fontsize=11)
ax4.set_title('D2DS Contamination Status\n(Historical 2021-2025)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add percentage labels on bars
total = fix_status['Count'].sum()
for i, (bar, count) in enumerate(zip(bars, fix_status['Count'])):
    height = bar.get_height()
    pct = count / total * 100
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig_file = analysis_output_dir / "d2ds_fix_timing_analysis.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved figure to {fig_file}")

# ============================================================================
# ANALYSIS 4: Historical D2DS Segments Pothole Exposure
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 4: Historical D2DS Segments Pothole Exposure")
print("="*70)
print("\nQuestion: What share of D2DS segments had potholes and were fixed in preceding week?")

# Filter panel to D2DS segments only
panel['segment_id'] = panel['segment_id'].astype(str)
d2ds_panel = panel[panel['segment_id'].isin(d2ds_segments)].copy()

print(f"\n[4.1] D2DS Segments Historical Statistics (2021-2025):")
print(f"  Total D2DS segment-weeks: {len(d2ds_panel):,}")
print(f"  Weeks with potholes (Y_it=1): {d2ds_panel['Y_it'].sum():,} ({d2ds_panel['Y_it'].mean():.2%})")
print(f"  Weeks with potholes fixed (R_it=1): {d2ds_panel['R_it'].sum():,} "
      f"({d2ds_panel['R_it'].sum() / d2ds_panel['Y_it'].sum():.2%} of pothole-weeks)")

# For each week, check pothole status and fix status in preceding week
d2ds_panel_sorted = d2ds_panel.sort_values(['segment_id', 'week_start'])
d2ds_panel_sorted['prev_week_pothole'] = d2ds_panel_sorted.groupby('segment_id')['Y_it'].shift(1)
d2ds_panel_sorted['prev_week_fixed'] = d2ds_panel_sorted.groupby('segment_id')['R_it'].shift(1)

# Filter to cases where preceding week had data
d2ds_with_prev = d2ds_panel_sorted[d2ds_panel_sorted['prev_week_pothole'].notna()].copy()

print(f"\n[4.2] Preceding Week Analysis:")
print(f"  Segment-weeks with preceding week data: {len(d2ds_with_prev):,}")
print(f"  Had pothole in preceding week: {d2ds_with_prev['prev_week_pothole'].sum():,} "
      f"({d2ds_with_prev['prev_week_pothole'].mean():.2%})")
print(f"  Among those with pothole in preceding week:")
prev_pothole_cases = d2ds_with_prev[d2ds_with_prev['prev_week_pothole'] == 1]
if len(prev_pothole_cases) > 0:
    print(f"    Fixed in preceding week: {prev_pothole_cases['prev_week_fixed'].sum():,} "
          f"({prev_pothole_cases['prev_week_fixed'].mean():.2%})")
    print(f"    NOT fixed in preceding week: {(prev_pothole_cases['prev_week_fixed'] == 0).sum():,} "
          f"({(prev_pothole_cases['prev_week_fixed'] == 0).mean():.2%})")

# Save
historical_analysis_file = analysis_output_dir / "d2ds_historical_exposure.csv"
d2ds_with_prev.to_csv(historical_analysis_file, index=False)
print(f"\n[4.3] Saved historical analysis to {historical_analysis_file}")

# Create summary visualization
print(f"\n[4.4] Creating historical exposure visualization...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Pothole rate by week of year
d2ds_panel['week_of_year'] = d2ds_panel['week_start'].dt.isocalendar().week
pothole_rate_by_woy = d2ds_panel.groupby('week_of_year')['Y_it'].mean()
ax1.plot(pothole_rate_by_woy.index, pothole_rate_by_woy.values, linewidth=2, color='steelblue')
ax1.set_xlabel('Week of Year', fontsize=11)
ax1.set_ylabel('Pothole Rate', fontsize=11)
ax1.set_title('D2DS Segments: Pothole Occurrence Rate by Week', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Top-right: Fix rate by week of year
fix_rate_by_woy = d2ds_panel[d2ds_panel['Y_it'] == 1].groupby('week_of_year')['R_it'].mean()
ax2.plot(fix_rate_by_woy.index, fix_rate_by_woy.values, linewidth=2, color='green')
ax2.set_xlabel('Week of Year', fontsize=11)
ax2.set_ylabel('Fix Rate (among potholes)', fontsize=11)
ax2.set_title('D2DS Segments: Fix Rate by Week', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Bottom-left: Distribution of potholes per segment
potholes_per_segment = d2ds_panel.groupby('segment_id')['Y_it'].sum()
ax3.hist(potholes_per_segment, bins=30, color='coral', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Total Potholes (2021-2025)', fontsize=11)
ax3.set_ylabel('Number of Segments', fontsize=11)
ax3.set_title('Distribution of Potholes per D2DS Segment', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Bottom-right: Preceding week status
prev_week_summary = pd.DataFrame({
    'Category': ['No pothole\nin prev week', 'Pothole &\nfixed', 'Pothole &\nNOT fixed'],
    'Count': [
        (d2ds_with_prev['prev_week_pothole'] == 0).sum(),
        (prev_pothole_cases['prev_week_fixed'] == 1).sum() if len(prev_pothole_cases) > 0 else 0,
        (prev_pothole_cases['prev_week_fixed'] == 0).sum() if len(prev_pothole_cases) > 0 else 0
    ]
})
ax4.bar(prev_week_summary['Category'], prev_week_summary['Count'],
       color=['lightblue', 'lightgreen', 'lightcoral'], edgecolor='black', alpha=0.7)
ax4.set_ylabel('Number of Segment-Weeks', fontsize=11)
ax4.set_title('Preceding Week Status\n(D2DS Segments, Historical)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
for i, (cat, count) in enumerate(zip(prev_week_summary['Category'], prev_week_summary['Count'])):
    ax4.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig_file = analysis_output_dir / "d2ds_historical_exposure.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved figure to {fig_file}")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

summary_report = f"""
D2DS FEASIBILITY ANALYSIS SUMMARY
{'='*70}

1. BUNDLE ELIGIBILITY (Conditional on Preceding Week Potholes)
   - Mean eligible bundles per week: {eligibility_df['total_eligible'].mean():.1f}
   - Mean eligible DH bundles: {eligibility_df['eligible_dh'].mean():.1f}
   - Mean eligible non-DH bundles: {eligibility_df['eligible_non_dh'].mean():.1f}
   - Weeks with <4 DH bundles: {(eligibility_df['eligible_dh'] < 4).sum()} / {len(eligibility_df)} ({(eligibility_df['eligible_dh'] < 4).mean():.1%})
   - Weeks with <6 total bundles: {(eligibility_df['total_eligible'] < 6).sum()} / {len(eligibility_df)} ({(eligibility_df['total_eligible'] < 6).mean():.1%})

   CONCLUSION: {'FEASIBLE' if (eligibility_df['eligible_dh'] >= 4).mean() > 0.9 and (eligibility_df['total_eligible'] >= 6).mean() > 0.9 else 'CHALLENGING'}
   {'✓ Sufficient eligible bundles in most weeks' if (eligibility_df['total_eligible'] >= 6).mean() > 0.9 else '⚠ May struggle to find enough eligible bundles'}

2. UNFIXED SEGMENTS AVAILABILITY
   - Mean unfixed segments per bundle-week: {unfixed_df['unfixed'].mean():.2f}
   - Median unfixed segments: {unfixed_df['unfixed'].median():.1f}
   - Bundle-weeks with ≥1 unfixed: {(unfixed_df['unfixed'] >= 1).sum()} ({(unfixed_df['unfixed'] >= 1).mean():.1%})

   CONCLUSION: {'SUFFICIENT' if unfixed_df['unfixed'].mean() >= 1 else 'LIMITED'}
   {'✓ Ample unfixed segments available' if unfixed_df['unfixed'].mean() >= 1 else '⚠ Limited unfixed segments'}

3. D2DS SEGMENTS FIX TIMING (Historical)
   Timeline: Pothole Week N-1 → DH Week N → D2DS Survey Week N+1 (2 weeks after pothole)
   - Fixed before D2DS survey Saturday: {d2ds_potholes['fixed_before_d2ds_survey'].sum():,} ({d2ds_potholes['fixed_before_d2ds_survey'].mean():.1%})
   - Mean days to fix: {d2ds_potholes['days_to_fix'].mean():.1f} days

   CONCLUSION: {'LOW CONTAMINATION' if d2ds_potholes['fixed_before_d2ds_survey'].mean() < 0.5 else 'HIGH CONTAMINATION'}
   {'✓ Most potholes remain unfixed at D2DS survey time' if d2ds_potholes['fixed_before_d2ds_survey'].mean() < 0.5 else '⚠ Many potholes fixed before D2DS survey (contamination risk)'}

4. HISTORICAL D2DS SEGMENTS EXPOSURE
   - Pothole rate: {d2ds_panel['Y_it'].mean():.2%}
   - Fix rate (among potholes): {d2ds_panel['R_it'].sum() / d2ds_panel['Y_it'].sum():.2%}
   - Had pothole in preceding week: {d2ds_with_prev['prev_week_pothole'].mean():.2%}
   - Fixed in preceding week (among those with pothole): {prev_pothole_cases['prev_week_fixed'].mean():.2%}

OVERALL FEASIBILITY: {'FEASIBLE ✓' if (eligibility_df['total_eligible'] >= 6).mean() > 0.9 and unfixed_df['unfixed'].mean() >= 1 else 'NEEDS ADJUSTMENT ⚠'}

All outputs saved to: {analysis_output_dir}
"""

print(summary_report)

# Save summary report
report_file = analysis_output_dir / "feasibility_summary.txt"
with open(report_file, 'w') as f:
    f.write(summary_report)
print(f"\nSaved summary report to {report_file}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
