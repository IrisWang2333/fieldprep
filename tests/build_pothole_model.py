#!/usr/bin/env python
"""
Build Pothole Statistical Model

Following the methodology in pothole_model.pdf:
1. Load historical pothole data (2021-2025) from notification_activities.csv
2. Build segment-week panel
3. Estimate occurrence model: Y_it = α_i + γ_t + ε_it
4. Estimate fix time model: R_it = μ_t + u_it
5. Simulate potholes for experiment period
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_fetcher import fetch_latest_notification_activities

# Paths
output_dir = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/pothole_model")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("STEP 1: Load and Prepare Historical Pothole Data (2021-2025)")
print("="*70)
print("\nUsing notification_activities.csv (pothole patch records)")
print("Filters: ACTIVITY_CODE_GROUP_TEXT = 'ASPHALT' AND ACTIVITY_CODE_TEXT = 'POTHOLE PATCHED (EA)'")

# Load pothole activities data using new data fetcher
print(f"\n[1.1] Loading pothole activities...")
potholes = fetch_latest_notification_activities(use_local=True)
print(f"  Loaded {len(potholes):,} pothole patch records")

# The data fetcher already filters and parses dates:
# - date_reported: NOTIFICATION_DATE (for Y_it)
# - date_closed: COMPLETION_DATE (for R_it)
# - week_start: Saturday-based week
# - segment_id: FUNCTIONAL_LOCATION

# Filter to 2021-2025
print(f"\n[1.2] Filtering to 2021-2025...")
potholes = potholes[
    (potholes['date_reported'] >= '2021-01-01') &
    (potholes['date_reported'] < '2026-01-01')
].copy()
print(f"  Records 2021-2025: {len(potholes):,}")

# Add additional columns for compatibility with rest of script
potholes['date_requested'] = potholes['date_reported']  # For compatibility
potholes['service_request_id'] = potholes.index.astype(str)  # Unique ID
potholes['year'] = potholes['date_reported'].dt.year
potholes['week'] = potholes['date_reported'].dt.isocalendar().week

print(f"\n[1.3] Data summary:")
print(f"  Unique segments: {potholes['segment_id'].nunique():,}")
print(f"  Date range: {potholes['date_reported'].min()} to {potholes['date_reported'].max()}")
print(f"  Total weeks: {potholes['week_start'].nunique():,}")
print(f"  Records with close date: {potholes['date_closed'].notna().sum():,} ({potholes['date_closed'].notna().mean()*100:.1f}%)")

# Save cleaned data
clean_file = output_dir / "pothole_data_2021_2025.parquet"
potholes.to_parquet(clean_file, index=False)
print(f"\n[1.4] Saved cleaned data to {clean_file}")

print(f"\n{'='*70}")
print(f"STEP 1 COMPLETE")
print(f"{'='*70}")

print("\n" + "="*70)
print("STEP 2: Build Segment-Week Panel Dataset")
print("="*70)

# Get all unique segments and ALL weeks in the period (not just weeks with potholes)
print(f"\n[2.1] Creating complete segment-week panel...")
all_segments = sorted(potholes['segment_id'].unique())

# Create ALL weeks from 2021-01-01 to 2025-12-31 (Saturday-based)
start_date = pd.Timestamp('2021-01-01')
end_date = pd.Timestamp('2025-12-31')

# Find first Saturday on or after start_date
days_since_saturday = (start_date.weekday() + 2) % 7
first_saturday = start_date - pd.Timedelta(days=days_since_saturday)

# Generate all Saturdays
all_weeks = []
current_week = first_saturday
while current_week <= end_date:
    all_weeks.append(current_week)
    current_week += pd.Timedelta(days=7)

all_weeks = sorted(all_weeks)

print(f"  Segments: {len(all_segments):,}")
print(f"  Weeks: {len(all_weeks):,} (all weeks 2021-2025, including weeks with no potholes)")
print(f"  Total panel size: {len(all_segments) * len(all_weeks):,}")

# Create panel
from itertools import product
panel = pd.DataFrame(
    list(product(all_segments, all_weeks)),
    columns=['segment_id', 'week_start']
)

print(f"  Panel created: {len(panel):,} rows")

# Save panel
panel_file = output_dir / "segment_week_panel.parquet"
panel.to_parquet(panel_file, index=False)
print(f"\n[2.2] Saved panel to {panel_file}")

print(f"\n{'='*70}")
print(f"STEP 2 COMPLETE")
print(f"{'='*70}")

print("\n" + "="*70)
print("STEP 3: Create Y_it Variable (Pothole Occurrence)")
print("="*70)

print(f"\n[3.1] Aggregating potholes by segment-week...")
# For each segment-week, check if there was at least one pothole
pothole_weeks = potholes.groupby(['segment_id', 'week_start']).agg({
    'service_request_id': 'count',  # Count of potholes
}).reset_index()
pothole_weeks.columns = ['segment_id', 'week_start', 'pothole_count']

# Merge with panel
panel = panel.merge(pothole_weeks, on=['segment_id', 'week_start'], how='left')
panel['pothole_count'] = panel['pothole_count'].fillna(0).astype(int)

# Create Y_it: binary indicator for at least one pothole
panel['Y_it'] = (panel['pothole_count'] > 0).astype(int)

print(f"  Segment-weeks with potholes: {panel['Y_it'].sum():,} ({panel['Y_it'].mean()*100:.2f}%)")
print(f"  Segment-weeks without potholes: {(panel['Y_it']==0).sum():,} ({(panel['Y_it']==0).mean()*100:.2f}%)")

print(f"\n{'='*70}")
print(f"STEP 3 COMPLETE")
print(f"{'='*70}")

print("\n" + "="*70)
print("STEP 4: Create R_it Variable (Pothole Fix)")
print("="*70)

print(f"\n[4.1] Calculating fix status for pothole-weeks...")
# For each pothole, check if it was fixed before end of that week
# Week runs Saturday-Friday, so week_end is Friday (week_start + 6 days)
potholes['week_end'] = potholes['week_start'] + pd.Timedelta(days=6)
potholes['fixed_by_week_end'] = (
    (potholes['date_closed'].notna()) &
    (potholes['date_closed'] <= potholes['week_end'])
).astype(int)

# Aggregate by segment-week: R_it = 1 only if ALL potholes were fixed
fix_weeks = potholes[potholes['date_requested'].notna()].groupby(['segment_id', 'week_start']).agg({
    'fixed_by_week_end': 'min',  # 1 only if ALL potholes were fixed
}).reset_index()
fix_weeks.columns = ['segment_id', 'week_start', 'R_it']

# Merge with panel
panel = panel.merge(fix_weeks, on=['segment_id', 'week_start'], how='left')
panel['R_it'] = panel['R_it'].fillna(0).astype(int)

# R_it only defined for Y_it = 1
panel.loc[panel['Y_it'] == 0, 'R_it'] = np.nan

print(f"  Pothole-weeks (Y_it=1): {panel['Y_it'].sum():,}")
print(f"  Fixed by week end (R_it=1): {panel['R_it'].sum():,} ({panel['R_it'].sum()/panel['Y_it'].sum()*100:.1f}% of potholes)")
print(f"  Not fixed by week end (R_it=0): {(panel['R_it']==0).sum():,} ({(panel['R_it']==0).sum()/panel['Y_it'].sum()*100:.1f}% of potholes)")

# Save panel with Y_it and R_it
panel_full_file = output_dir / "panel_with_outcomes.parquet"
panel.to_parquet(panel_full_file, index=False)
print(f"\n[4.2] Saved panel with Y_it and R_it to {panel_full_file}")

# Visualize repair time distribution
print(f"\n[4.3] Creating repair time distribution visualization...")
import matplotlib.pyplot as plt

# Calculate repair time for all potholes (days from requested to closed)
potholes_with_fix = potholes[
    (potholes['date_requested'].notna()) &
    (potholes['date_closed'].notna())
].copy()

potholes_with_fix['repair_days'] = (
    potholes_with_fix['date_closed'] - potholes_with_fix['date_requested']
).dt.total_seconds() / (24 * 3600)

# Filter to reasonable range (0-90 days)
potholes_with_fix = potholes_with_fix[
    (potholes_with_fix['repair_days'] >= 0) &
    (potholes_with_fix['repair_days'] <= 90)
]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Histogram of repair times
ax1.hist(potholes_with_fix['repair_days'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(x=7, color='red', linestyle='--', linewidth=2, label='R_it cutoff (7 days)')
ax1.set_xlabel('Days to Repair', fontsize=12)
ax1.set_ylabel('Number of Potholes', fontsize=12)
ax1.set_title('Distribution of Pothole Repair Time (2021-2025)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Cumulative distribution
repair_sorted = np.sort(potholes_with_fix['repair_days'])
cumulative = np.arange(1, len(repair_sorted) + 1) / len(repair_sorted) * 100
ax2.plot(repair_sorted, cumulative, linewidth=2, color='#1f77b4')
ax2.axvline(x=7, color='red', linestyle='--', linewidth=2, label='R_it cutoff (7 days)')

# Find percentage repaired within 7 days
pct_within_7days = (potholes_with_fix['repair_days'] <= 7).mean() * 100
ax2.plot(7, pct_within_7days, 'ro', markersize=10)
ax2.annotate(f'{pct_within_7days:.1f}% within 7 days',
            xy=(7, pct_within_7days),
            xytext=(20, pct_within_7days - 10),
            fontsize=11,
            color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red'),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax2.set_xlabel('Days to Repair', fontsize=12)
ax2.set_ylabel('Cumulative % Repaired', fontsize=12)
ax2.set_title('Cumulative Repair Rate Over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 90)
ax2.set_ylim(0, 100)

plt.tight_layout()
repair_dist_file = output_dir / "repair_time_distribution.png"
plt.savefig(repair_dist_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"  Saved repair time distribution to {repair_dist_file}")
print(f"  Median repair time: {potholes_with_fix['repair_days'].median():.1f} days")
print(f"  Repaired within 7 days: {pct_within_7days:.1f}%")

print(f"\n{'='*70}")
print(f"STEP 4 COMPLETE")
print(f"{'='*70}")

print("\n" + "="*70)
print("STEP 5-7: Pothole Occurrence Model (Y_it = α_i + γ_t + ε_it)")
print("="*70)

print(f"\n[5.1] Running fixed effects regression...")

# Create week number for easy indexing
week_to_num = {w: i for i, w in enumerate(sorted(panel['week_start'].unique()))}
panel['week_num'] = panel['week_start'].map(week_to_num)

# Prepare data for panel regression
from linearmodels.panel import PanelOLS

# Set multi-index for panel data
panel_reg = panel.set_index(['segment_id', 'week_num'])

# Run two-way fixed effects regression
# Y_it = α_i + γ_t + ε_it
print(f"  Estimating two-way fixed effects model...")
print(f"  Panel dimensions: {panel['segment_id'].nunique():,} segments × {panel['week_num'].nunique():,} weeks")

# Add constant for regression (required by PanelOLS)
panel_reg['const'] = 1

mod = PanelOLS(
    dependent=panel_reg['Y_it'],
    exog=panel_reg[['const']],  # Constant term
    entity_effects=True,  # Segment fixed effects (α_i)
    time_effects=True,    # Week fixed effects (γ_t)
)

res = mod.fit(cov_type='clustered', cluster_entity=True)

print(f"\n[5.2] Regression results:")
print(f"  R-squared: {res.rsquared:.4f}")
print(f"  Entity effects (segments): {res.included_effects[0]}")
print(f"  Time effects (weeks): {res.included_effects[1]}")

# Extract fixed effects using alternating projections (Gauss-Seidel)
# This properly controls for both effects simultaneously
print(f"  Extracting fixed effects using alternating projections...")

# Overall mean
overall_mean = panel['Y_it'].mean()

# Prepare data with multi-index
panel_indexed = panel.set_index(['segment_id', 'week_num'])
y = panel_indexed['Y_it']

# Initialize FE
segment_fe = pd.Series(0.0, index=y.index.get_level_values('segment_id').unique())
week_fe = pd.Series(0.0, index=y.index.get_level_values('week_num').unique())

# Alternating projections to extract α_i and γ_t
max_iter = 100
tol = 1e-8

for iteration in range(max_iter):
    # Update week FE: mean of (Y - segment_fe) by week
    y_minus_segment = y - y.index.get_level_values('segment_id').map(segment_fe)
    week_fe_new = y_minus_segment.groupby(level='week_num').mean()
    week_fe_new = week_fe_new - week_fe_new.mean()  # Normalize to sum to zero

    # Update segment FE: mean of (Y - week_fe) by segment
    y_minus_week = y - y.index.get_level_values('week_num').map(week_fe_new)
    segment_fe_new = y_minus_week.groupby(level='segment_id').mean()
    segment_fe_new = segment_fe_new - segment_fe_new.mean()  # Normalize to sum to zero

    # Check convergence
    if (np.abs(segment_fe_new - segment_fe).max() < tol and
        np.abs(week_fe_new - week_fe).max() < tol):
        print(f"    Converged after {iteration+1} iterations")
        break

    segment_fe = segment_fe_new
    week_fe = week_fe_new

segment_fe.name = 'segment_fe'
week_fe.name = 'week_fe'

print(f"\n[5.3] Fixed effects summary:")
print(f"  Overall mean: {overall_mean:.6f}")
print(f"  Segment FE (α_i): {len(segment_fe):,} coefficients")
print(f"    Range: [{segment_fe.min():.4f}, {segment_fe.max():.4f}]")
print(f"    Mean: {segment_fe.mean():.6f}")
print(f"    Std: {segment_fe.std():.6f}")
print(f"  Week FE (γ_t): {len(week_fe):,} coefficients")
print(f"    Range: [{week_fe.min():.4f}, {week_fe.max():.4f}]")
print(f"    Mean: {week_fe.mean():.6f}")
print(f"    Std: {week_fe.std():.6f}")

# Calculate fitted values
panel['segment_fe'] = panel['segment_id'].map(segment_fe.to_dict())
panel['week_fe'] = panel['week_num'].map(week_fe.to_dict())

# Manually calculate fitted values: Y_hat = overall_mean + α_i + γ_t
# Note: PanelOLS fitted_values only include exog (constant), not FE
panel['Y_hat'] = overall_mean + panel['segment_fe'] + panel['week_fe']
panel['Y_hat'] = panel['Y_hat'].clip(0, 1)  # Clip to [0, 1] for probability

print(f"\n[7.1] Fitted values (Ŷ_it):")
print(f"  Range: [{panel['Y_hat'].min():.4f}, {panel['Y_hat'].max():.4f}]")
print(f"  Mean: {panel['Y_hat'].mean():.4f}")
print(f"  Std: {panel['Y_hat'].std():.4f}")

# Save coefficients
import pickle

fe_dict = {
    'overall_mean': overall_mean,
    'segment_fe': segment_fe.to_dict(),
    'week_fe': week_fe.to_dict(),
    'week_to_num': week_to_num,
}

fe_file = output_dir / "pothole_occurrence_fe.pkl"
with open(fe_file, 'wb') as f:
    pickle.dump(fe_dict, f)
print(f"\n[5.2] Saved fixed effects to {fe_file}")

# Save panel with fitted values
panel_fitted_file = output_dir / "panel_with_fitted.parquet"
panel.to_parquet(panel_fitted_file, index=False)
print(f"[7.2] Saved panel with fitted values to {panel_fitted_file}")

# Visualize seasonal pattern for occurrence
print(f"\n[7.3] Creating visualization of pothole occurrence probability by week...")

import matplotlib.pyplot as plt

# Map week_num to actual dates for better visualization
week_num_to_start = {v: k for k, v in week_to_num.items()}
week_occur_df = pd.DataFrame({
    'week_num': week_fe.index,
    'occur_fe': week_fe.values
})
week_occur_df['week_start'] = week_occur_df['week_num'].map(week_num_to_start)
week_occur_df['week_of_year'] = week_occur_df['week_start'].dt.isocalendar().week
week_occur_df['year'] = week_occur_df['week_start'].dt.year

# Average occurrence FE by week-of-year (across all years)
# Note: week FE are deviations from mean, so add back overall_mean
week_occur_df['occur_prob'] = overall_mean + week_occur_df['occur_fe']
weekly_occur_pattern = week_occur_df.groupby('week_of_year')['occur_prob'].mean().sort_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(weekly_occur_pattern.index, weekly_occur_pattern.values, 'o-',
        linewidth=2, markersize=4, color='#1f77b4')
ax.axhline(y=overall_mean, color='r', linestyle='--', alpha=0.5,
           label=f'Overall mean ({overall_mean:.2%})')
ax.set_xlabel('Week of Year', fontsize=12)
ax.set_ylabel('Probability of Pothole Occurrence', fontsize=12)
ax.set_title('Seasonal Pattern in Pothole Occurrence (Week = Sat-Fri)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Add month labels
month_weeks = [1, 5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48]
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(month_weeks)
ax.set_xticklabels(month_names)

fig_file = output_dir / "occurrence_probability_by_week.png"
plt.tight_layout()
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"  Saved visualization to {fig_file}")
print(f"  Peak occurrence probability: Week {weekly_occur_pattern.idxmax()} ({weekly_occur_pattern.max():.2%})")
print(f"  Lowest occurrence probability: Week {weekly_occur_pattern.idxmin()} ({weekly_occur_pattern.min():.2%})")

print(f"\n{'='*70}")
print(f"STEPS 5-7 COMPLETE")
print(f"{'='*70}")

print("\n" + "="*70)
print("STEP 8-10: Pothole Fix Time Model (R_it = μ_t + u_it)")
print("="*70)

print(f"\n[8.1] Running fixed effects regression for fix probability...")

# Filter to Y_it = 1 (only segment-weeks with potholes)
panel_with_potholes = panel[panel['Y_it'] == 1].copy()

print(f"  Observations with potholes (Y_it=1): {len(panel_with_potholes):,}")

# Prepare for panel regression (only week FE, no segment FE for fix model)
panel_fix_reg = panel_with_potholes.set_index(['segment_id', 'week_num'])

# Run regression with only time fixed effects
# R_it = μ_t + u_it
print(f"  Estimating time fixed effects only...")

# Add constant for regression (required by PanelOLS)
panel_fix_reg['const'] = 1

mod_fix = PanelOLS(
    dependent=panel_fix_reg['R_it'],
    exog=panel_fix_reg[['const']],  # Constant term
    entity_effects=False,  # No segment FE for fix model
    time_effects=True,      # Week fixed effects (μ_t)
)

res_fix = mod_fix.fit(cov_type='clustered', cluster_entity=True)

print(f"\n[8.2] Regression results:")
print(f"  R-squared: {res_fix.rsquared:.4f}")
print(f"  Time effects included: {res_fix.included_effects[0] if len(res_fix.included_effects) > 0 else 'Time'}")

# Extract week fixed effects using group means (same as occurrence model)
# Week FE = mean(R_it) for each week
week_fix_means = panel_with_potholes.groupby('week_num')['R_it'].mean()
week_fix_means.name = 'week_fix_fe'

print(f"\n[8.3] Week fixed effects (μ_t):")
print(f"  Coefficients: {len(week_fix_means):,}")
print(f"    Range: [{week_fix_means.min():.4f}, {week_fix_means.max():.4f}]")
print(f"    Mean: {week_fix_means.mean():.4f}")
print(f"    Std: {week_fix_means.std():.4f}")

# Map back to panel
panel_with_potholes['mu_t'] = panel_with_potholes['week_num'].map(week_fix_means.to_dict())
panel_with_potholes['R_hat'] = panel_with_potholes['mu_t'].clip(0, 1)

print(f"\n[9.1] Fitted values (R̂_it) for fix probability:")
print(f"  Range: [{panel_with_potholes['R_hat'].min():.4f}, {panel_with_potholes['R_hat'].max():.4f}]")
print(f"  Mean: {panel_with_potholes['R_hat'].mean():.4f}")

# Save fix model coefficients
fix_fe_dict = {
    'week_fix_fe': week_fix_means.to_dict(),
}

fix_fe_file = output_dir / "pothole_fix_fe.pkl"
with open(fix_fe_file, 'wb') as f:
    pickle.dump(fix_fe_dict, f)
print(f"\n[8.2] Saved fix model coefficients to {fix_fe_file}")

# Visualize seasonal pattern
print(f"\n[10.1] Creating visualization of fix probability by week...")

import matplotlib.pyplot as plt

# Map week_num to actual dates for better visualization
week_num_to_start = {v: k for k, v in week_to_num.items()}
week_fix_df = pd.DataFrame({
    'week_num': week_fix_means.index,
    'fix_prob': week_fix_means.values
})
week_fix_df['week_start'] = week_fix_df['week_num'].map(week_num_to_start)
week_fix_df['week_of_year'] = week_fix_df['week_start'].dt.isocalendar().week
week_fix_df['year'] = week_fix_df['week_start'].dt.year

# Average fix probability by week-of-year (across all years)
weekly_pattern = week_fix_df.groupby('week_of_year')['fix_prob'].mean().sort_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(weekly_pattern.index, weekly_pattern.values, 'o-', linewidth=2, markersize=4)
# Calculate actual overall mean from the data
overall_fix_rate = panel_with_potholes['R_it'].mean()
ax.axhline(y=overall_fix_rate, color='r', linestyle='--', alpha=0.5, label=f'Overall mean ({overall_fix_rate:.1%})')
ax.set_xlabel('Week of Year', fontsize=12)
ax.set_ylabel('Probability of Fix by Week End (Friday)', fontsize=12)
ax.set_title('Seasonal Pattern in Pothole Fix Probability (Week = Sat-Fri)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Add month labels
month_weeks = [1, 5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48]
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(month_weeks)
ax.set_xticklabels(month_names)

fig_file = output_dir / "fix_probability_by_week.png"
plt.tight_layout()
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"  Saved visualization to {fig_file}")
print(f"  Peak fix probability: Week {weekly_pattern.idxmax()} ({weekly_pattern.max():.1%})")
print(f"  Lowest fix probability: Week {weekly_pattern.idxmin()} ({weekly_pattern.min():.1%})")

# Visualize seasonal patterns by year
print(f"\n[10.2] Creating seasonal patterns by year visualization...")

# Prepare data for occurrence pattern by year
panel['year'] = panel['week_start'].dt.year
panel['week_of_year'] = panel['week_start'].dt.isocalendar().week

# Occurrence patterns by year
occur_by_year = panel.groupby(['year', 'week_of_year'])['Y_it'].mean().reset_index()
occur_by_year_pivot = occur_by_year.pivot(index='week_of_year', columns='year', values='Y_it')

# Fix patterns by year (only for weeks with potholes)
panel_with_potholes_year = panel[panel['Y_it'] == 1].copy()
panel_with_potholes_year['year'] = panel_with_potholes_year['week_start'].dt.year
panel_with_potholes_year['week_of_year'] = panel_with_potholes_year['week_start'].dt.isocalendar().week

fix_by_year = panel_with_potholes_year.groupby(['year', 'week_of_year'])['R_it'].mean().reset_index()
fix_by_year_pivot = fix_by_year.pivot(index='week_of_year', columns='year', values='R_it')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top: Occurrence patterns
colors = plt.cm.tab10(np.linspace(0, 1, len(occur_by_year_pivot.columns)))
for i, year in enumerate(occur_by_year_pivot.columns):
    ax1.plot(occur_by_year_pivot.index, occur_by_year_pivot[year],
            linewidth=1.5, alpha=0.7, label=str(year), color=colors[i])

ax1.axhline(y=overall_mean, color='r', linestyle='--', alpha=0.5, linewidth=2,
           label=f'Overall mean ({overall_mean:.2%})')
ax1.set_xlabel('Week of Year', fontsize=12)
ax1.set_ylabel('Pothole Occurrence Rate', fontsize=12)
ax1.set_title('Seasonal Patterns in Pothole Occurrence by Year (2021-2025)',
             fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', ncol=3, fontsize=9)

# Add month labels
month_weeks = [1, 5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48]
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax1.set_xticks(month_weeks)
ax1.set_xticklabels(month_names)

# Bottom: Fix probability patterns
colors = plt.cm.tab10(np.linspace(0, 1, len(fix_by_year_pivot.columns)))
for i, year in enumerate(fix_by_year_pivot.columns):
    ax2.plot(fix_by_year_pivot.index, fix_by_year_pivot[year],
            linewidth=1.5, alpha=0.7, label=str(year), color=colors[i])

ax2.axhline(y=overall_fix_rate, color='r', linestyle='--', alpha=0.5, linewidth=2,
           label=f'Overall mean ({overall_fix_rate:.1%})')
ax2.set_xlabel('Week of Year', fontsize=12)
ax2.set_ylabel('Fix Probability', fontsize=12)
ax2.set_title('Seasonal Patterns in Pothole Fix Probability by Year (2021-2025)',
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', ncol=3, fontsize=9)

# Add month labels
ax2.set_xticks(month_weeks)
ax2.set_xticklabels(month_names)

plt.tight_layout()
seasonal_fig_file = output_dir / "seasonal_patterns_by_year.png"
plt.savefig(seasonal_fig_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"  Saved seasonal patterns by year to {seasonal_fig_file}")

print(f"\n{'='*70}")
print(f"STEPS 8-10 COMPLETE")
print(f"{'='*70}")

print("\n" + "="*70)
print("STEP 11-12: Simulate Potholes for Experiment Period (Jan-Aug 2026)")
print("="*70)

# Define experiment period: Jan 10, 2026 - Aug 1, 2026 (30 weeks)
# Jan 10, 2026 is Saturday (perfect for week start)
experiment_start = pd.Timestamp('2026-01-10')
experiment_end = pd.Timestamp('2026-08-01')

print(f"\n[11.1] Creating experiment period calendar...")
print(f"  Start date: {experiment_start.date()}")
print(f"  End date: {experiment_end.date()}")

# Generate week starts for experiment (Saturdays)
# Calculate days since last Saturday
days_since_saturday = (experiment_start.weekday() + 2) % 7
first_saturday = experiment_start - pd.Timedelta(days=days_since_saturday)

experiment_weeks = []
current_week = first_saturday
while current_week < experiment_end:
    experiment_weeks.append(current_week)
    current_week += pd.Timedelta(days=7)

print(f"  Number of weeks: {len(experiment_weeks)}")
print(f"  First week starts: {experiment_weeks[0].date()} (Saturday)")

# Map experiment weeks to week-of-year for matching with historical patterns
exp_week_to_woy = {}
for w in experiment_weeks:
    woy = w.isocalendar().week
    exp_week_to_woy[w] = woy

print(f"\n[11.2] Simulating pothole occurrence...")

# For each segment and experiment week, predict occurrence probability
# Get segment fixed effects
seg_fe_dict = segment_fe.to_dict()

# Get week-of-year fixed effects (average across years)
# Map historical week_num to week-of-year
panel_sample = panel[['week_num', 'week_start']].drop_duplicates()
panel_sample['week_of_year'] = panel_sample['week_start'].dt.isocalendar().week

# Average week FE by week-of-year
week_num_to_woy = panel_sample.set_index('week_num')['week_of_year'].to_dict()
week_fe_by_woy = {}
for week_num, woy in week_num_to_woy.items():
    if woy not in week_fe_by_woy:
        week_fe_by_woy[woy] = []
    week_fe_by_woy[woy].append(week_fe[week_num])

week_fe_avg = {woy: np.mean(vals) for woy, vals in week_fe_by_woy.items()}

# Create simulation panel
sim_data = []
for seg_id in all_segments:
    seg_fe_val = seg_fe_dict.get(seg_id, 0)

    for exp_week in experiment_weeks:
        woy = exp_week_to_woy[exp_week]
        week_fe_val = week_fe_avg.get(woy, 0)

        # Calculate pothole occurrence probability
        y_hat = overall_mean + seg_fe_val + week_fe_val
        y_hat = np.clip(y_hat, 0, 1)

        sim_data.append({
            'segment_id': seg_id,
            'week_start': exp_week,
            'week_of_year': woy,
            'Y_hat': y_hat,
        })

sim_panel = pd.DataFrame(sim_data)

print(f"  Simulation panel: {len(sim_panel):,} segment-weeks")
print(f"  Mean pothole probability: {sim_panel['Y_hat'].mean():.4f}")

print(f"\n[11.3] Calculating fix probabilities (R_hat) for all segments...")

# Calculate R_hat for ALL segments (before simulation)
# Get fix probabilities by week-of-year from historical data
fix_fe_by_woy = {}
for week_num, fix_prob in week_fix_means.items():
    woy = week_num_to_woy.get(week_num)
    if woy is not None:
        if woy not in fix_fe_by_woy:
            fix_fe_by_woy[woy] = []
        fix_fe_by_woy[woy].append(fix_prob)

fix_prob_avg = {woy: np.mean(vals) for woy, vals in fix_fe_by_woy.items()}

# Map to ALL segments in simulation panel
sim_panel['R_hat'] = sim_panel['week_of_year'].map(fix_prob_avg)
sim_panel['R_hat'] = sim_panel['R_hat'].fillna(0.35).clip(0, 1)

print(f"  R_hat calculated for {sim_panel['R_hat'].notna().sum():,} segment-weeks")
print(f"  Mean fix probability: {sim_panel['R_hat'].mean():.4f}")

# Now simulate pothole occurrence (Bernoulli draws)
print(f"\n[11.4] Simulating pothole occurrence (y_sim)...")
rng = np.random.default_rng(42)
sim_panel['y_sim'] = rng.binomial(1, sim_panel['Y_hat'])

n_simulated_potholes = sim_panel['y_sim'].sum()
print(f"  Simulated potholes: {n_simulated_potholes:,} ({n_simulated_potholes/len(sim_panel)*100:.2f}%)")

print(f"\n[12.1] Simulating pothole fix times (r_sim)...")

# For segments with simulated potholes, simulate fix status using pre-calculated R_hat
sim_panel_with_potholes = sim_panel[sim_panel['y_sim'] == 1].copy()
sim_panel_with_potholes['r_sim'] = rng.binomial(1, sim_panel_with_potholes['R_hat'])

n_fixed = sim_panel_with_potholes['r_sim'].sum()
print(f"  Potholes fixed by week end: {n_fixed:,} ({n_fixed/len(sim_panel_with_potholes)*100:.1f}%)")
print(f"  Potholes not fixed by week end: {len(sim_panel_with_potholes)-n_fixed:,} ({(len(sim_panel_with_potholes)-n_fixed)/len(sim_panel_with_potholes)*100:.1f}%)")

# Merge r_sim back to full simulation panel (r_sim is NaN for y_sim=0)
sim_panel = sim_panel.merge(
    sim_panel_with_potholes[['segment_id', 'week_start', 'r_sim']],
    on=['segment_id', 'week_start'],
    how='left'
)

# Save simulation results
sim_file = output_dir / "pothole_simulation_2026.parquet"
sim_panel.to_parquet(sim_file, index=False)
print(f"\n[12.2] Saved simulation results to {sim_file}")

# Create summary by week
sim_summary = sim_panel.groupby('week_start').agg({
    'y_sim': 'sum',
    'segment_id': 'count'
}).reset_index()
sim_summary.columns = ['week_start', 'potholes', 'total_segments']
sim_summary['pothole_rate'] = sim_summary['potholes'] / sim_summary['total_segments']

# Add fix counts
fix_by_week = sim_panel[sim_panel['y_sim']==1].groupby('week_start')['r_sim'].sum().reset_index()
fix_by_week.columns = ['week_start', 'fixed']
sim_summary = sim_summary.merge(fix_by_week, on='week_start', how='left')
sim_summary['fixed'] = sim_summary['fixed'].fillna(0).astype(int)
# Fix rate: avoid division by zero (set to NaN when potholes=0)
sim_summary['fix_rate'] = (sim_summary['fixed'] / sim_summary['potholes']).replace([np.inf, -np.inf], np.nan)

sim_summary_file = output_dir / "simulation_summary_by_week.csv"
sim_summary.to_csv(sim_summary_file, index=False)
print(f"[12.3] Saved weekly summary to {sim_summary_file}")

print(f"\n[12.4] Simulation summary:")
print(sim_summary[['week_start', 'potholes', 'fixed', 'pothole_rate', 'fix_rate']].to_string(index=False))

print(f"\n{'='*70}")
print(f"STEPS 11-12 COMPLETE")
print(f"{'='*70}")

print("\n" + "="*70)
print("STEP 13-15: Analyze D2DS Segment Pothole Exposure")
print("="*70)

# Load 30-day simulation plan to get D2DS segments
print(f"\n[13.1] Loading 30-day simulation plan...")
plan_file = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/simulation/plan_30days.csv")

if not plan_file.exists():
    print(f"  WARNING: Plan file not found at {plan_file}")
    print(f"  Skipping D2DS analysis. Run quick_simulate_30days.py first.")
else:
    plan_df = pd.read_csv(plan_file)
    plan_df['date'] = pd.to_datetime(plan_df['date'])

    # Get D2DS bundles
    d2ds_plan = plan_df[plan_df['task'] == 'D2DS'].copy()
    d2ds_bundles = set(d2ds_plan['bundle_id'].unique())

    print(f"  Total D2DS assignments: {len(d2ds_plan)}")
    print(f"  Unique D2DS bundles: {len(d2ds_bundles)}")

    # Load bundle-segment mapping
    print(f"\n[13.2] Loading bundle-segment mapping...")
    bundle_file = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
    bundles = gpd.read_parquet(bundle_file)

    seg_col = 'iamfloc' if 'iamfloc' in bundles.columns else 'segment_id'
    bundles['segment_id'] = bundles[seg_col].astype(str)

    # Get segments in D2DS bundles
    d2ds_segments = set(
        bundles[bundles['bundle_id'].isin(d2ds_bundles)]['segment_id'].unique()
    )

    print(f"  D2DS segments: {len(d2ds_segments):,}")

    # Map D2DS dates to survey weeks (Saturday-based)
    d2ds_plan['day_of_week'] = d2ds_plan['date'].dt.dayofweek
    d2ds_plan['days_since_saturday'] = (d2ds_plan['day_of_week'] + 2) % 7
    d2ds_plan['week_start'] = d2ds_plan['date'] - pd.to_timedelta(d2ds_plan['days_since_saturday'], unit='D')
    d2ds_plan['week_start'] = d2ds_plan['week_start'].dt.normalize()

    # Create segment-week mapping for D2DS
    bundle_to_weeks = d2ds_plan.groupby('bundle_id')['week_start'].apply(set).to_dict()

    # Get segments and their survey weeks
    d2ds_seg_weeks = []
    for bundle_id in d2ds_bundles:
        segs = bundles[bundles['bundle_id'] == bundle_id]['segment_id'].unique()
        weeks = bundle_to_weeks.get(bundle_id, set())
        for seg in segs:
            for week in weeks:
                d2ds_seg_weeks.append({
                    'segment_id': seg,
                    'survey_week': week,
                    'bundle_id': bundle_id
                })

    d2ds_survey_df = pd.DataFrame(d2ds_seg_weeks)

    print(f"\n[13.3] Analyzing pothole exposure during D2DS survey weeks...")

    # Merge with simulation results
    sim_panel['segment_id'] = sim_panel['segment_id'].astype(str)
    d2ds_survey_df['segment_id'] = d2ds_survey_df['segment_id'].astype(str)

    d2ds_with_potholes = d2ds_survey_df.merge(
        sim_panel[['segment_id', 'week_start', 'y_sim', 'r_sim', 'Y_hat', 'R_hat']],
        left_on=['segment_id', 'survey_week'],
        right_on=['segment_id', 'week_start'],
        how='left'
    )

    # Calculate summary statistics
    total_d2ds_obs = len(d2ds_with_potholes)
    exposed_to_pothole = d2ds_with_potholes['y_sim'].sum()
    pothole_fixed = d2ds_with_potholes['r_sim'].sum()

    print(f"  Total D2DS segment-weeks: {total_d2ds_obs:,}")
    print(f"  Exposed to pothole: {exposed_to_pothole:,} ({exposed_to_pothole/total_d2ds_obs*100:.1f}%)")
    print(f"  Pothole fixed by week end: {pothole_fixed:,} ({pothole_fixed/exposed_to_pothole*100:.1f}% of exposed)")
    print(f"  Pothole not fixed: {exposed_to_pothole-pothole_fixed:,} ({(exposed_to_pothole-pothole_fixed)/exposed_to_pothole*100:.1f}% of exposed)")

    # Create PotholeFix treatment variable
    print(f"\n[15.1] Creating PotholeFix treatment variable...")

    # PotholeFix = 1 if segment had pothole AND it was fixed by week end
    d2ds_with_potholes['PotholeFix'] = (
        (d2ds_with_potholes['y_sim'] == 1) &
        (d2ds_with_potholes['r_sim'] == 1)
    ).astype(int)

    # Control group: had pothole but NOT fixed by week end
    d2ds_with_potholes['PotholeControl'] = (
        (d2ds_with_potholes['y_sim'] == 1) &
        (d2ds_with_potholes['r_sim'] == 0)
    ).astype(int)

    # No pothole exposure
    d2ds_with_potholes['NoPothole'] = (d2ds_with_potholes['y_sim'] == 0).astype(int)

    print(f"  Treatment group (PotholeFix=1): {d2ds_with_potholes['PotholeFix'].sum():,}")
    print(f"  Control group (pothole not fixed): {d2ds_with_potholes['PotholeControl'].sum():,}")
    print(f"  No pothole: {d2ds_with_potholes['NoPothole'].sum():,}")

    # Save D2DS pothole exposure analysis
    d2ds_exposure_file = output_dir / "d2ds_pothole_exposure.csv"
    d2ds_with_potholes.to_csv(d2ds_exposure_file, index=False)
    print(f"\n[15.2] Saved D2DS pothole exposure to {d2ds_exposure_file}")

    # Create summary by bundle
    bundle_summary = d2ds_with_potholes.groupby('bundle_id').agg({
        'segment_id': 'count',
        'y_sim': 'sum',
        'r_sim': 'sum',
        'PotholeFix': 'sum',
        'PotholeControl': 'sum'
    }).reset_index()
    bundle_summary.columns = [
        'bundle_id', 'segments', 'potholes',
        'fixed', 'treatment', 'control'
    ]

    bundle_summary_file = output_dir / "d2ds_pothole_by_bundle.csv"
    bundle_summary.to_csv(bundle_summary_file, index=False)
    print(f"[15.3] Saved bundle-level summary to {bundle_summary_file}")

    print(f"\n[15.4] Top 10 bundles with most pothole fixes:")
    top_bundles = bundle_summary.sort_values('fixed', ascending=False).head(10)
    print(top_bundles.to_string(index=False))

print(f"\n{'='*70}")
print(f"STEPS 13-15 COMPLETE")
print(f"{'='*70}")

print(f"\n{'='*70}")
print(f"ALL STEPS COMPLETE!")
print(f"{'='*70}")
print(f"\nOutput files saved to: {output_dir}")
print(f"\nKey outputs:")
print(f"  1. pothole_occurrence_fe.pkl - Fixed effects for occurrence model")
print(f"  2. pothole_fix_fe.pkl - Fixed effects for fix time model")
print(f"  3. pothole_simulation_2026.parquet - Simulated potholes for experiment")
print(f"  4. d2ds_pothole_exposure.csv - D2DS segment-week pothole exposure")
print(f"  5. fix_probability_by_week.png - Seasonal fix probability visualization")
