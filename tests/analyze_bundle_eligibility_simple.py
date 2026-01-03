#!/usr/bin/env python
"""
Simple Bundle Eligibility Analysis

Question: Conditional on at least one segment having at least one reported
pothole in the preceding week, can we have enough bundles in a week?

Basic statistics only - no distinction between DH/D2DS.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
data_dir = Path("/Users/iris/Dropbox/sandiego code/data")
output_dir = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/pothole_model")
analysis_output_dir = output_dir / "bundle_eligibility_simple"
analysis_output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("SIMPLE BUNDLE ELIGIBILITY ANALYSIS")
print("="*70)
print("\nQuestion: Conditional on having potholes in preceding week,")
print("          do we have enough bundles each week?")
print("="*70)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1] Loading data...")

# Load historical panel data (2021-2025)
panel = pd.read_parquet(output_dir / "panel_with_outcomes.parquet")
print(f"  Historical panel: {len(panel):,} segment-weeks")
print(f"  Date range: {panel['week_start'].min()} to {panel['week_start'].max()}")

# Load bundle-segment mapping
bundle_file = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
bundles = gpd.read_parquet(bundle_file)
seg_col = 'iamfloc' if 'iamfloc' in bundles.columns else 'segment_id'
bundles['segment_id'] = bundles[seg_col].astype(str)

total_bundles = bundles['bundle_id'].nunique()
total_segments = bundles['segment_id'].nunique()
print(f"  Total bundles: {total_bundles:,}")
print(f"  Total segments: {total_segments:,}")

# ============================================================================
# Analysis: Bundle Eligibility per Week
# ============================================================================
print("\n" + "="*70)
print("ELIGIBILITY ANALYSIS")
print("="*70)

# Add bundle_id to panel
panel['bundle_id'] = panel['segment_id'].map(
    bundles.set_index('segment_id')['bundle_id'].to_dict()
)

# Remove segments not in any bundle
panel_with_bundles = panel[panel['bundle_id'].notna()].copy()
print(f"\n[2] Panel with bundle mapping: {len(panel_with_bundles):,} segment-weeks")

# For each week, find bundles with at least one pothole
bundles_with_potholes_by_week = (
    panel_with_bundles[panel_with_bundles['Y_it'] == 1]
    .groupby(['week_start', 'bundle_id'])
    .size()
    .reset_index(name='pothole_count')
)

print(f"  Bundle-weeks with potholes: {len(bundles_with_potholes_by_week):,}")

# For each week t, find bundles eligible (had pothole in week t-1)
all_weeks = sorted(panel_with_bundles['week_start'].unique())
eligibility_results = []

for i, week in enumerate(all_weeks):
    if i == 0:  # Skip first week (no preceding week)
        continue

    preceding_week = all_weeks[i-1]

    # Bundles with potholes in preceding week = eligible bundles for current week
    eligible_bundles = set(
        bundles_with_potholes_by_week[
            bundles_with_potholes_by_week['week_start'] == preceding_week
        ]['bundle_id'].unique()
    )

    eligibility_results.append({
        'week': week,
        'eligible_bundles': len(eligible_bundles)
    })

eligibility_df = pd.DataFrame(eligibility_results)

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nTotal weeks analyzed: {len(eligibility_df)}")
print(f"Total bundles in pool: {total_bundles:,}")
print(f"\nEligible bundles per week (conditional on preceding week pothole):")
print(f"  Mean:   {eligibility_df['eligible_bundles'].mean():.1f}")
print(f"  Median: {eligibility_df['eligible_bundles'].median():.1f}")
print(f"  Min:    {eligibility_df['eligible_bundles'].min()}")
print(f"  Max:    {eligibility_df['eligible_bundles'].max()}")
print(f"  Std:    {eligibility_df['eligible_bundles'].std():.1f}")

print(f"\nPercentage of total bundle pool:")
print(f"  Mean eligible: {eligibility_df['eligible_bundles'].mean() / total_bundles * 100:.2f}%")
print(f"  Min eligible:  {eligibility_df['eligible_bundles'].min() / total_bundles * 100:.2f}%")
print(f"  Max eligible:  {eligibility_df['eligible_bundles'].max() / total_bundles * 100:.2f}%")

print(f"\nWeeks with sufficient bundles (assuming different thresholds):")
for threshold in [6, 10, 20, 30, 50]:
    n_sufficient = (eligibility_df['eligible_bundles'] >= threshold).sum()
    pct = n_sufficient / len(eligibility_df) * 100
    print(f"  ≥{threshold:2d} bundles: {n_sufficient:3d}/{len(eligibility_df)} weeks ({pct:5.1f}%)")

# Distribution percentiles
print(f"\nDistribution percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(eligibility_df['eligible_bundles'], p)
    print(f"  {p:2d}th percentile: {val:.1f} bundles")

# Save results
output_file = analysis_output_dir / "weekly_eligibility.csv"
eligibility_df.to_csv(output_file, index=False)
print(f"\n[3] Saved results to {output_file}")

# ============================================================================
# Visualization
# ============================================================================
print("\n[4] Creating visualization...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Time series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(eligibility_df['week'], eligibility_df['eligible_bundles'],
         linewidth=2, color='steelblue', marker='o', markersize=3, alpha=0.7)
ax1.axhline(y=eligibility_df['eligible_bundles'].mean(),
           color='red', linestyle='--', linewidth=2,
           label=f"Mean: {eligibility_df['eligible_bundles'].mean():.1f}")
ax1.fill_between(eligibility_df['week'], 0, eligibility_df['eligible_bundles'],
                 alpha=0.2, color='steelblue')
ax1.set_xlabel('Week', fontsize=12)
ax1.set_ylabel('Number of Eligible Bundles', fontsize=12)
ax1.set_title('Weekly Eligible Bundles Over Time (2021-2025)\n' +
             'Conditional on Having Pothole in Preceding Week',
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 2. Distribution histogram
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(eligibility_df['eligible_bundles'], bins=30,
        color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=eligibility_df['eligible_bundles'].mean(),
           color='red', linestyle='--', linewidth=2,
           label=f"Mean: {eligibility_df['eligible_bundles'].mean():.1f}")
ax2.axvline(x=eligibility_df['eligible_bundles'].median(),
           color='green', linestyle='--', linewidth=2,
           label=f"Median: {eligibility_df['eligible_bundles'].median():.1f}")
ax2.set_xlabel('Number of Eligible Bundles', fontsize=12)
ax2.set_ylabel('Number of Weeks', fontsize=12)
ax2.set_title('Distribution of Eligible Bundles per Week',
             fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Cumulative distribution
ax3 = fig.add_subplot(gs[1, 1])
sorted_eligible = np.sort(eligibility_df['eligible_bundles'])
cumulative = np.arange(1, len(sorted_eligible) + 1) / len(sorted_eligible) * 100
ax3.plot(sorted_eligible, cumulative, linewidth=2.5, color='darkgreen')
ax3.fill_between(sorted_eligible, 0, cumulative, alpha=0.2, color='darkgreen')
ax3.set_xlabel('Number of Eligible Bundles', fontsize=12)
ax3.set_ylabel('Cumulative Percentage of Weeks (%)', fontsize=12)
ax3.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(left=0)

# 4. Weekly pattern (by week of year)
ax4 = fig.add_subplot(gs[2, 0])
eligibility_df['week_of_year'] = pd.to_datetime(eligibility_df['week']).dt.isocalendar().week
weekly_pattern = eligibility_df.groupby('week_of_year')['eligible_bundles'].agg(['mean', 'std'])
ax4.plot(weekly_pattern.index, weekly_pattern['mean'],
        linewidth=2, color='steelblue', marker='o', markersize=4)
ax4.fill_between(weekly_pattern.index,
                weekly_pattern['mean'] - weekly_pattern['std'],
                weekly_pattern['mean'] + weekly_pattern['std'],
                alpha=0.3, color='steelblue')
ax4.set_xlabel('Week of Year', fontsize=12)
ax4.set_ylabel('Eligible Bundles', fontsize=12)
ax4.set_title('Seasonal Pattern (Mean ± Std)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(1, 53)

# 5. Summary statistics table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

summary_data = [
    ['Metric', 'Value'],
    ['─'*25, '─'*15],
    ['Total weeks analyzed', f'{len(eligibility_df)}'],
    ['Total bundles in pool', f'{total_bundles:,}'],
    ['', ''],
    ['Mean eligible/week', f'{eligibility_df["eligible_bundles"].mean():.1f}'],
    ['Median eligible/week', f'{eligibility_df["eligible_bundles"].median():.1f}'],
    ['Min eligible', f'{eligibility_df["eligible_bundles"].min()}'],
    ['Max eligible', f'{eligibility_df["eligible_bundles"].max()}'],
    ['Std deviation', f'{eligibility_df["eligible_bundles"].std():.1f}'],
    ['', ''],
    ['% of total pool (mean)', f'{eligibility_df["eligible_bundles"].mean() / total_bundles * 100:.2f}%'],
    ['', ''],
    ['Weeks with ≥6 bundles', f'{(eligibility_df["eligible_bundles"] >= 6).sum()}/{len(eligibility_df)} ({(eligibility_df["eligible_bundles"] >= 6).mean()*100:.1f}%)'],
    ['Weeks with ≥10 bundles', f'{(eligibility_df["eligible_bundles"] >= 10).sum()}/{len(eligibility_df)} ({(eligibility_df["eligible_bundles"] >= 10).mean()*100:.1f}%)'],
    ['Weeks with ≥20 bundles', f'{(eligibility_df["eligible_bundles"] >= 20).sum()}/{len(eligibility_df)} ({(eligibility_df["eligible_bundles"] >= 20).mean()*100:.1f}%)'],
]

table = ax5.table(cellText=summary_data, cellLoc='left', loc='center',
                 colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style alternating rows
for i in range(2, len(summary_data)):
    if summary_data[i][0] == '':
        continue
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

ax5.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)

plt.suptitle('Bundle Eligibility Analysis (2021-2025)\nConditional on Pothole in Preceding Week',
            fontsize=16, fontweight='bold', y=0.995)

fig_file = analysis_output_dir / "bundle_eligibility_summary.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved figure to {fig_file}")

# ============================================================================
# Create summary report
# ============================================================================
summary_report = f"""
{'='*70}
BUNDLE ELIGIBILITY ANALYSIS - SUMMARY REPORT
{'='*70}

QUESTION:
Conditional on at least one segment having at least one reported pothole
in the preceding week, can we have enough bundles in a week?

DATA:
- Historical period: 2021-2025
- Total weeks analyzed: {len(eligibility_df)}
- Total bundles in pool: {total_bundles:,}

RESULTS:
Eligible bundles per week (with pothole in preceding week):
  Mean:   {eligibility_df['eligible_bundles'].mean():.1f} bundles
  Median: {eligibility_df['eligible_bundles'].median():.1f} bundles
  Range:  [{eligibility_df['eligible_bundles'].min()}, {eligibility_df['eligible_bundles'].max()}] bundles
  Std:    {eligibility_df['eligible_bundles'].std():.1f} bundles

Percentage of total pool:
  Mean: {eligibility_df['eligible_bundles'].mean() / total_bundles * 100:.2f}%
  This means approximately {eligibility_df['eligible_bundles'].mean() / total_bundles * 100:.1f}% of bundles are "active"
  (had pothole in preceding week) in any given week.

Sufficiency Analysis (weeks meeting threshold):
  ≥ 6 bundles: {(eligibility_df['eligible_bundles'] >= 6).sum():3d}/{len(eligibility_df)} weeks ({(eligibility_df['eligible_bundles'] >= 6).mean()*100:5.1f}%)
  ≥10 bundles: {(eligibility_df['eligible_bundles'] >= 10).sum():3d}/{len(eligibility_df)} weeks ({(eligibility_df['eligible_bundles'] >= 10).mean()*100:5.1f}%)
  ≥20 bundles: {(eligibility_df['eligible_bundles'] >= 20).sum():3d}/{len(eligibility_df)} weeks ({(eligibility_df['eligible_bundles'] >= 20).mean()*100:5.1f}%)
  ≥30 bundles: {(eligibility_df['eligible_bundles'] >= 30).sum():3d}/{len(eligibility_df)} weeks ({(eligibility_df['eligible_bundles'] >= 30).mean()*100:5.1f}%)
  ≥50 bundles: {(eligibility_df['eligible_bundles'] >= 50).sum():3d}/{len(eligibility_df)} weeks ({(eligibility_df['eligible_bundles'] >= 50).mean()*100:5.1f}%)

CONCLUSION:
{'✓ SUFFICIENT' if eligibility_df['eligible_bundles'].mean() >= 6 else '⚠ INSUFFICIENT'}
On average, {eligibility_df['eligible_bundles'].mean():.1f} bundles per week are eligible
(had pothole in preceding week).

{'✓ ' if (eligibility_df['eligible_bundles'] >= 6).mean() > 0.95 else '⚠ '}{(eligibility_df['eligible_bundles'] >= 6).mean()*100:.1f}% of weeks have ≥6 eligible bundles
{'✓ ' if (eligibility_df['eligible_bundles'] >= 10).mean() > 0.90 else '⚠ '}{(eligibility_df['eligible_bundles'] >= 10).mean()*100:.1f}% of weeks have ≥10 eligible bundles
{'✓ ' if (eligibility_df['eligible_bundles'] >= 20).mean() > 0.75 else '⚠ '}{(eligibility_df['eligible_bundles'] >= 20).mean()*100:.1f}% of weeks have ≥20 eligible bundles

{'='*70}
OUTPUT FILES:
- {output_file}
- {fig_file}
- {analysis_output_dir / "summary_report.txt"}
{'='*70}
"""

print("\n" + summary_report)

# Save report
report_file = analysis_output_dir / "summary_report.txt"
with open(report_file, 'w') as f:
    f.write(summary_report)

print(f"\n✓ Analysis complete. All files saved to {analysis_output_dir}/")
