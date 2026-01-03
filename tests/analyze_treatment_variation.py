#!/usr/bin/env python
"""
Treatment Variation Analysis

Question: Among eligible bundles (with potholes in preceding week),
do we have both FIXED and UNFIXED segments each week?

Goal: Ensure treatment variation exists - avoid weeks where all segments
      are fixed (treatment=1) or all unfixed (treatment=0)
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
output_dir = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/pothole_model")
analysis_output_dir = output_dir / "treatment_variation"
analysis_output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("TREATMENT VARIATION ANALYSIS")
print("="*70)
print("\nQuestion: In eligible bundles, do we have both fixed and unfixed")
print("          segments each week to ensure treatment variation?")
print("="*70)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1] Loading data...")

# Load historical panel data
panel = pd.read_parquet(output_dir / "panel_with_outcomes.parquet")
print(f"  Historical panel: {len(panel):,} segment-weeks")

# Load bundle-segment mapping
bundle_file = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
bundles = gpd.read_parquet(bundle_file)
seg_col = 'iamfloc' if 'iamfloc' in bundles.columns else 'segment_id'
bundles['segment_id'] = bundles[seg_col].astype(str)

# Add bundle_id to panel
panel['bundle_id'] = panel['segment_id'].map(
    bundles.set_index('segment_id')['bundle_id'].to_dict()
)
panel_with_bundles = panel[panel['bundle_id'].notna()].copy()

print(f"  Panel with bundles: {len(panel_with_bundles):,} segment-weeks")

# ============================================================================
# Step 1: Identify Eligible Bundles (with pothole in preceding week)
# ============================================================================
print("\n[2] Identifying eligible bundles per week...")

# Bundles with potholes by week
bundles_with_potholes_by_week = (
    panel_with_bundles[panel_with_bundles['Y_it'] == 1]
    .groupby(['week_start', 'bundle_id'])
    .size()
    .reset_index(name='pothole_count')
)

# For each week t, find bundles eligible (had pothole in week t-1)
all_weeks = sorted(panel_with_bundles['week_start'].unique())
weekly_analysis = []

for i, week in enumerate(all_weeks):
    if i == 0:  # Skip first week
        continue

    preceding_week = all_weeks[i-1]

    # Eligible bundles = bundles with potholes in preceding week
    eligible_bundles = set(
        bundles_with_potholes_by_week[
            bundles_with_potholes_by_week['week_start'] == preceding_week
        ]['bundle_id'].unique()
    )

    if len(eligible_bundles) == 0:
        continue

    # Get all segments in eligible bundles for CURRENT week
    eligible_segments_this_week = panel_with_bundles[
        (panel_with_bundles['week_start'] == week) &
        (panel_with_bundles['bundle_id'].isin(eligible_bundles))
    ].copy()

    # Among these segments, filter to those with potholes THIS week
    segments_with_potholes = eligible_segments_this_week[
        eligible_segments_this_week['Y_it'] == 1
    ].copy()

    if len(segments_with_potholes) == 0:
        # No potholes this week in eligible bundles
        weekly_analysis.append({
            'week': week,
            'eligible_bundles': len(eligible_bundles),
            'total_segments_in_eligible_bundles': len(eligible_segments_this_week),
            'segments_with_potholes': 0,
            'segments_fixed': 0,
            'segments_unfixed': 0,
            'pct_fixed': np.nan,
            'pct_unfixed': np.nan,
            'has_variation': False,
            'issue': 'no_potholes'
        })
        continue

    # Count fixed vs unfixed
    # R_it = 1: all potholes in segment fixed by Friday
    # R_it = 0: not all potholes fixed
    n_fixed = (segments_with_potholes['R_it'] == 1).sum()
    n_unfixed = (segments_with_potholes['R_it'] == 0).sum()
    n_total = len(segments_with_potholes)

    pct_fixed = n_fixed / n_total if n_total > 0 else 0
    pct_unfixed = n_unfixed / n_total if n_total > 0 else 0

    # Check if we have variation (both fixed and unfixed)
    has_variation = (n_fixed > 0) and (n_unfixed > 0)

    # Identify issue type
    if n_total == 0:
        issue = 'no_potholes'
    elif n_fixed == 0:
        issue = 'all_unfixed'
    elif n_unfixed == 0:
        issue = 'all_fixed'
    else:
        issue = 'good_variation'

    weekly_analysis.append({
        'week': week,
        'eligible_bundles': len(eligible_bundles),
        'total_segments_in_eligible_bundles': len(eligible_segments_this_week),
        'segments_with_potholes': n_total,
        'segments_fixed': n_fixed,
        'segments_unfixed': n_unfixed,
        'pct_fixed': pct_fixed,
        'pct_unfixed': pct_unfixed,
        'has_variation': has_variation,
        'issue': issue
    })

df = pd.DataFrame(weekly_analysis)

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nTotal weeks analyzed: {len(df)}")
print(f"\nOverall segments in eligible bundles:")
print(f"  Mean segments with potholes per week: {df['segments_with_potholes'].mean():.1f}")
print(f"  Mean fixed segments: {df['segments_fixed'].mean():.1f}")
print(f"  Mean unfixed segments: {df['segments_unfixed'].mean():.1f}")

print(f"\nProportion breakdown (among segments with potholes):")
print(f"  Mean % fixed: {df['pct_fixed'].mean()*100:.1f}%")
print(f"  Mean % unfixed: {df['pct_unfixed'].mean()*100:.1f}%")

print(f"\nTreatment Variation Analysis:")
print(f"  Weeks with BOTH fixed and unfixed: {df['has_variation'].sum()} / {len(df)} ({df['has_variation'].mean()*100:.1f}%)")
print(f"  Weeks with NO variation: {(~df['has_variation']).sum()} / {len(df)} ({(~df['has_variation']).mean()*100:.1f}%)")

print(f"\nBreakdown of issues:")
issue_counts = df['issue'].value_counts()
for issue, count in issue_counts.items():
    pct = count / len(df) * 100
    print(f"  {issue:20s}: {count:3d} weeks ({pct:5.1f}%)")

# Identify problematic weeks
problematic_weeks = df[~df['has_variation']].copy()
if len(problematic_weeks) > 0:
    print(f"\n⚠ PROBLEMATIC WEEKS (no treatment variation):")
    for _, row in problematic_weeks.head(10).iterrows():
        print(f"  {row['week'].date()}: {row['issue']:15s} "
              f"(eligible bundles: {row['eligible_bundles']:3.0f}, "
              f"segments w/ potholes: {row['segments_with_potholes']:3.0f}, "
              f"fixed: {row['segments_fixed']:.0f}, unfixed: {row['segments_unfixed']:.0f})")
    if len(problematic_weeks) > 10:
        print(f"  ... and {len(problematic_weeks) - 10} more weeks")
else:
    print(f"\n✓ NO PROBLEMATIC WEEKS - All weeks have treatment variation!")

# Distribution stats
print(f"\nDistribution of fixed/unfixed ratio:")
df_with_potholes = df[df['segments_with_potholes'] > 0].copy()
print(f"  Weeks with potholes: {len(df_with_potholes)}")
print(f"  Mean % fixed: {df_with_potholes['pct_fixed'].mean()*100:.1f}%")
print(f"  Median % fixed: {df_with_potholes['pct_fixed'].median()*100:.1f}%")
print(f"  Std % fixed: {df_with_potholes['pct_fixed'].std()*100:.1f}%")

for pct_threshold in [10, 20, 30, 40]:
    n_balanced = ((df_with_potholes['pct_fixed'] >= pct_threshold/100) &
                  (df_with_potholes['pct_fixed'] <= 1 - pct_threshold/100)).sum()
    print(f"  Weeks with {pct_threshold}-{100-pct_threshold}% split: "
          f"{n_balanced}/{len(df_with_potholes)} ({n_balanced/len(df_with_potholes)*100:.1f}%)")

# Save results
output_file = analysis_output_dir / "treatment_variation_by_week.csv"
df.to_csv(output_file, index=False)
print(f"\n[3] Saved results to {output_file}")

# ============================================================================
# Visualization
# ============================================================================
print("\n[4] Creating visualizations...")

# 1. Time series of fixed vs unfixed counts
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['week'], df['segments_fixed'],
         label='Fixed (R_it=1)', linewidth=2, color='green', marker='o', markersize=3)
ax.plot(df['week'], df['segments_unfixed'],
         label='Unfixed (R_it=0)', linewidth=2, color='red', marker='o', markersize=3)
ax.fill_between(df['week'], 0, df['segments_fixed'], alpha=0.2, color='green')
ax.fill_between(df['week'], 0, df['segments_unfixed'], alpha=0.2, color='red')
ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('Number of Segments', fontsize=12)
ax.set_title('Fixed vs Unfixed Segments Over Time\n(In Eligible Bundles with Potholes)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_file = analysis_output_dir / "1_timeseries_counts.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [1/5] Saved {fig_file.name}")

# 2. Time series of percentages
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['week'], df['pct_fixed']*100,
         label='% Fixed', linewidth=2, color='green', marker='o', markersize=3)
ax.plot(df['week'], df['pct_unfixed']*100,
         label='% Unfixed', linewidth=2, color='red', marker='o', markersize=3)
ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='50%')
ax.fill_between(df['week'], 30, 70, alpha=0.1, color='blue', label='30-70% range')
ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Percentage Fixed vs Unfixed Over Time',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 105)
plt.tight_layout()
fig_file = analysis_output_dir / "2_timeseries_percentages.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [2/5] Saved {fig_file.name}")

# 3. Distribution of % fixed
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_with_potholes['pct_fixed']*100, bins=30,
        color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(x=df_with_potholes['pct_fixed'].mean()*100,
           color='red', linestyle='--', linewidth=2,
           label=f"Mean: {df_with_potholes['pct_fixed'].mean()*100:.1f}%")
ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='50%')
ax.set_xlabel('Percentage Fixed (%)', fontsize=12)
ax.set_ylabel('Number of Weeks', fontsize=12)
ax.set_title('Distribution of % Fixed\n(Among Weeks with Potholes)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig_file = analysis_output_dir / "3_distribution_pct_fixed.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [3/5] Saved {fig_file.name}")

# 4. Variation status pie chart
fig, ax = plt.subplots(figsize=(8, 6))
variation_counts = df['has_variation'].value_counts()
colors = ['lightgreen' if x else 'lightcoral' for x in variation_counts.index]
labels = ['Has Variation' if x else 'No Variation' for x in variation_counts.index]
wedges, texts, autotexts = ax.pie(variation_counts, labels=labels, autopct='%1.1f%%',
                                    colors=colors, startangle=90, textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(14)
ax.set_title('Treatment Variation Status', fontsize=14, fontweight='bold')
plt.tight_layout()
fig_file = analysis_output_dir / "4_variation_status_pie.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [4/5] Saved {fig_file.name}")

# 5. Issue breakdown
fig, ax = plt.subplots(figsize=(10, 6))
issue_counts_sorted = df['issue'].value_counts().sort_values(ascending=True)
colors_map = {
    'good_variation': 'lightgreen',
    'all_unfixed': 'lightcoral',
    'all_fixed': 'lightyellow',
    'no_potholes': 'lightgray'
}
colors = [colors_map.get(issue, 'lightblue') for issue in issue_counts_sorted.index]
ax.barh(issue_counts_sorted.index, issue_counts_sorted.values,
        color=colors, edgecolor='black', alpha=0.8)
ax.set_xlabel('Number of Weeks', fontsize=12)
ax.set_title('Issue Breakdown\n(Why Some Weeks Lack Treatment Variation)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for i, (issue, count) in enumerate(issue_counts_sorted.items()):
    ax.text(count, i, f' {count} ({count/len(df)*100:.1f}%)',
            va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
fig_file = analysis_output_dir / "5_issue_breakdown.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [5/5] Saved {fig_file.name}")

print(f"\n  All figures saved to {analysis_output_dir}/")

# ============================================================================
# Summary Report
# ============================================================================
summary_report = f"""
{'='*70}
TREATMENT VARIATION ANALYSIS - SUMMARY REPORT
{'='*70}

QUESTION:
Among eligible bundles (with potholes in preceding week), do we have
both FIXED and UNFIXED segments each week to ensure treatment variation?

GOAL:
Avoid weeks where treatment is all 0 (all unfixed) or all 1 (all fixed),
which would eliminate treatment variation.

DATA:
- Historical period: 2021-2025
- Total weeks analyzed: {len(df)}
- Weeks with potholes in eligible bundles: {len(df_with_potholes)}

RESULTS:

1. SEGMENT COUNTS (Mean per week):
   - Segments with potholes: {df['segments_with_potholes'].mean():.1f}
   - Segments fixed (R_it=1): {df['segments_fixed'].mean():.1f}
   - Segments unfixed (R_it=0): {df['segments_unfixed'].mean():.1f}

2. PROPORTION (Among segments with potholes):
   - Mean % fixed: {df_with_potholes['pct_fixed'].mean()*100:.1f}%
   - Mean % unfixed: {df_with_potholes['pct_unfixed'].mean()*100:.1f}%
   - Median % fixed: {df_with_potholes['pct_fixed'].median()*100:.1f}%
   - Std % fixed: {df_with_potholes['pct_fixed'].std()*100:.1f}%

3. TREATMENT VARIATION:
   - Weeks with BOTH fixed and unfixed: {df['has_variation'].sum()}/{len(df)} ({df['has_variation'].mean()*100:.1f}%)
   - Weeks WITHOUT variation: {(~df['has_variation']).sum()}/{len(df)} ({(~df['has_variation']).mean()*100:.1f}%)

4. ISSUE BREAKDOWN:
"""

for issue, count in issue_counts.items():
    pct = count / len(df) * 100
    summary_report += f"   - {issue:20s}: {count:3d} weeks ({pct:5.1f}%)\n"

summary_report += f"""
5. BALANCE ANALYSIS (Among weeks with potholes):
"""
for pct_threshold in [10, 20, 30, 40]:
    n_balanced = ((df_with_potholes['pct_fixed'] >= pct_threshold/100) &
                  (df_with_potholes['pct_fixed'] <= 1 - pct_threshold/100)).sum()
    pct = n_balanced/len(df_with_potholes)*100
    summary_report += f"   - {pct_threshold:2d}-{100-pct_threshold:2d}% split: {n_balanced:3d}/{len(df_with_potholes)} ({pct:5.1f}%)\n"

conclusion = "✓ SUFFICIENT VARIATION" if df['has_variation'].mean() > 0.80 else "⚠ LIMITED VARIATION"
summary_report += f"""
CONCLUSION: {conclusion}

{df['has_variation'].mean()*100:.1f}% of weeks have both fixed and unfixed segments.
Mean fix rate is {df_with_potholes['pct_fixed'].mean()*100:.1f}%, indicating substantial treatment variation.

{'✓ Treatment variation is sufficient for causal inference.' if df['has_variation'].mean() > 0.80 else '⚠ Some weeks lack treatment variation - may need to adjust design.'}

{'='*70}
OUTPUT FILES:
- {output_file}
- {analysis_output_dir / "1_timeseries_counts.png"}
- {analysis_output_dir / "2_timeseries_percentages.png"}
- {analysis_output_dir / "3_distribution_pct_fixed.png"}
- {analysis_output_dir / "4_variation_status_pie.png"}
- {analysis_output_dir / "5_issue_breakdown.png"}
- {analysis_output_dir / "summary_report.txt"}
{'='*70}
"""

print("\n" + summary_report)

# Save report
report_file = analysis_output_dir / "summary_report.txt"
with open(report_file, 'w') as f:
    f.write(summary_report)

print(f"\n✓ Analysis complete. All files saved to {analysis_output_dir}/")
