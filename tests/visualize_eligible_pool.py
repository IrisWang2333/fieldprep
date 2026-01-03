#!/usr/bin/env python
"""
Visualize Eligible Pool vs Sample Size
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
output_dir = Path("/Users/iris/Dropbox/sandiego code/code/fieldprep/outputs/pothole_model/d2ds_feasibility")
eligibility_file = output_dir / "bundle_eligibility_by_week.csv"

# Load data
df = pd.read_csv(eligibility_file)
df['week'] = pd.to_datetime(df['week'])
df['year'] = df['week'].dt.year
df['week_of_year'] = df['week'].dt.isocalendar().week

print("Creating visualizations...")

# Create comprehensive figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# Plot 1: Pool Size vs Sample Size Comparison (Top Left)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

categories = ['DH\nBundles', 'Non-DH\nBundles', 'Total']
pool_sizes = [df['eligible_dh'].mean(), df['eligible_non_dh'].mean(), df['total_eligible'].mean()]
sample_sizes = [4, 2, 6]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, pool_sizes, width, label='符合条件的Pool',
               color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, sample_sizes, width, label='需要选择的Sample',
               color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkred')

# Add ratio annotations
ratios = [pool_sizes[i] / sample_sizes[i] for i in range(len(categories))]
for i, ratio in enumerate(ratios):
    ax1.text(i, max(pool_sizes[i], sample_sizes[i]) + 3,
            f'比例: {ratio:.1f}×',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax1.set_ylabel('Bundle数量', fontsize=12, fontweight='bold')
ax1.set_title('符合条件的Pool vs 需要选择的Sample\n(每周平均)',
             fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, max(pool_sizes) * 1.2)

# ============================================================================
# Plot 2: Time Series - Eligible Pool Size (Top Right)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

ax2.fill_between(df['week'], 0, df['total_eligible'], alpha=0.3, color='steelblue', label='Total Eligible')
ax2.plot(df['week'], df['total_eligible'], linewidth=2, color='steelblue', alpha=0.8)
ax2.plot(df['week'], df['eligible_dh'], linewidth=2, color='green', alpha=0.8, label='DH Eligible')
ax2.plot(df['week'], df['eligible_non_dh'], linewidth=2, color='orange', alpha=0.8, label='Non-DH Eligible')

# Add requirement lines
ax2.axhline(y=6, color='red', linestyle='--', linewidth=2, label='需要: 6 bundles', alpha=0.7)
ax2.axhline(y=4, color='darkgreen', linestyle='--', linewidth=2, label='需要: 4 DH', alpha=0.7)

ax2.set_xlabel('时间', fontsize=12, fontweight='bold')
ax2.set_ylabel('符合条件的Bundle数量', fontsize=12, fontweight='bold')
ax2.set_title('每周符合条件的Bundles数量\n(2021-2025历史数据)',
             fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# ============================================================================
# Plot 3: Distribution Histogram (Middle Left)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])

# DH bundles distribution
ax3.hist(df['eligible_dh'], bins=30, alpha=0.6, color='green',
        edgecolor='black', label='DH Bundles')
ax3.axvline(x=df['eligible_dh'].mean(), color='green', linestyle='--',
           linewidth=2, label=f"均值: {df['eligible_dh'].mean():.1f}")
ax3.axvline(x=df['eligible_dh'].median(), color='darkgreen', linestyle=':',
           linewidth=2, label=f"中位数: {df['eligible_dh'].median():.1f}")
ax3.axvline(x=4, color='red', linestyle='--', linewidth=3,
           label='需要: 4个', alpha=0.8)

ax3.set_xlabel('DH Bundles数量', fontsize=12, fontweight='bold')
ax3.set_ylabel('周数', fontsize=12, fontweight='bold')
ax3.set_title('DH Bundles符合条件数量的分布\n(261周历史数据)',
             fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Add text annotation
insufficent_weeks = (df['eligible_dh'] < 4).sum()
ax3.text(0.95, 0.95, f'不足4个的周数:\n{insufficent_weeks} / 261 ({insufficent_weeks/261*100:.1f}%)',
        transform=ax3.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.8),
        fontsize=10, fontweight='bold')

# ============================================================================
# Plot 4: Sampling Illustration (Middle Right)
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])

# Create visual representation
np.random.seed(42)

# Pool of bundles
dh_pool = 18  # Example week
non_dh_pool = 35

# Sample needed
dh_sample = 4
non_dh_sample = 2

# Create scatter plot to visualize
y_positions = []
colors = []
sizes = []
labels_list = []

# DH Pool (not selected)
for i in range(dh_pool - dh_sample):
    y_positions.append(2)
    colors.append('lightgreen')
    sizes.append(200)
    labels_list.append('DH Pool')

# DH Selected
for i in range(dh_sample):
    y_positions.append(2)
    colors.append('darkgreen')
    sizes.append(400)
    labels_list.append('DH Selected')

# Non-DH Pool (not selected)
for i in range(non_dh_pool - non_dh_sample):
    y_positions.append(1)
    colors.append('lightsalmon')
    sizes.append(200)
    labels_list.append('Non-DH Pool')

# Non-DH Selected
for i in range(non_dh_sample):
    y_positions.append(1)
    colors.append('orangered')
    sizes.append(400)
    labels_list.append('Non-DH Selected')

x_positions = []
current_x = 0
for y in y_positions:
    x_positions.append(current_x)
    current_x += 1
    if y == 2 and current_x >= dh_pool:
        current_x = 0

# Reset x positions to make it centered
x_dh = np.linspace(0, 10, dh_pool)
x_non_dh = np.linspace(0, 10, non_dh_pool)

# Plot DH
dh_colors = ['darkgreen'] * dh_sample + ['lightgreen'] * (dh_pool - dh_sample)
dh_sizes = [500] * dh_sample + [200] * (dh_pool - dh_sample)
ax4.scatter(x_dh, [2]*dh_pool, c=dh_colors, s=dh_sizes, alpha=0.8, edgecolors='black', linewidth=1.5)

# Plot Non-DH
non_dh_colors = ['orangered'] * non_dh_sample + ['lightsalmon'] * (non_dh_pool - non_dh_sample)
non_dh_sizes = [500] * non_dh_sample + [200] * (non_dh_pool - non_dh_sample)
ax4.scatter(x_non_dh, [1]*non_dh_pool, c=non_dh_colors, s=non_dh_sizes, alpha=0.8, edgecolors='black', linewidth=1.5)

# Add labels
ax4.text(-0.5, 2, f'DH Bundles\n(Pool: {dh_pool}个)', ha='right', va='center', fontsize=11, fontweight='bold')
ax4.text(-0.5, 1, f'Non-DH Bundles\n(Pool: {non_dh_pool}个)', ha='right', va='center', fontsize=11, fontweight='bold')

ax4.text(5, 2.5, f'选中{dh_sample}个（深色）', ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='darkgreen', alpha=0.3))
ax4.text(5, 0.5, f'选中{non_dh_sample}个（深色）', ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='orangered', alpha=0.3))

ax4.set_ylim(0.3, 2.7)
ax4.set_xlim(-2, 12)
ax4.set_yticks([])
ax4.set_xticks([])
ax4.set_title('采样示意图（某一周的例子）\n深色 = 被选中, 浅色 = Pool中但未被选中',
             fontsize=13, fontweight='bold')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['bottom'].set_visible(False)

# ============================================================================
# Plot 5: Cumulative Distribution (Bottom Left)
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])

# Calculate cumulative distributions
dh_sorted = np.sort(df['eligible_dh'])
dh_cumulative = np.arange(1, len(dh_sorted) + 1) / len(dh_sorted) * 100

total_sorted = np.sort(df['total_eligible'])
total_cumulative = np.arange(1, len(total_sorted) + 1) / len(total_sorted) * 100

ax5.plot(dh_sorted, dh_cumulative, linewidth=2.5, color='green', label='DH Bundles')
ax5.plot(total_sorted, total_cumulative, linewidth=2.5, color='steelblue', label='Total Bundles')

# Mark critical thresholds
ax5.axvline(x=4, color='red', linestyle='--', linewidth=2, alpha=0.7, label='需要: 4个DH')
ax5.axvline(x=6, color='darkred', linestyle='--', linewidth=2, alpha=0.7, label='需要: 6个Total')

# Find percentiles
pct_below_4_dh = (df['eligible_dh'] < 4).mean() * 100
pct_below_6_total = (df['total_eligible'] < 6).mean() * 100

ax5.plot(4, pct_below_4_dh, 'ro', markersize=12)
ax5.text(4, pct_below_4_dh + 5, f'{pct_below_4_dh:.1f}%的周\n不足4个DH',
        ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red'))

ax5.set_xlabel('符合条件的Bundle数量', fontsize=12, fontweight='bold')
ax5.set_ylabel('累积百分比 (%)', fontsize=12, fontweight='bold')
ax5.set_title('累积分布函数 (CDF)\n多少周有至少N个符合条件的bundles?',
             fontsize=13, fontweight='bold')
ax5.legend(loc='lower right', fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 70)
ax5.set_ylim(0, 105)

# ============================================================================
# Plot 6: Seasonal Pattern (Bottom Right)
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1])

# Calculate by week of year
weekly_avg = df.groupby('week_of_year').agg({
    'eligible_dh': 'mean',
    'eligible_non_dh': 'mean',
    'total_eligible': 'mean'
}).reset_index()

ax6.plot(weekly_avg['week_of_year'], weekly_avg['total_eligible'],
        linewidth=2.5, color='steelblue', label='Total', marker='o', markersize=3)
ax6.plot(weekly_avg['week_of_year'], weekly_avg['eligible_dh'],
        linewidth=2.5, color='green', label='DH', marker='s', markersize=3)
ax6.plot(weekly_avg['week_of_year'], weekly_avg['eligible_non_dh'],
        linewidth=2.5, color='orange', label='Non-DH', marker='^', markersize=3)

ax6.axhline(y=6, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
ax6.axhline(y=4, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.6)

ax6.set_xlabel('Week of Year', fontsize=12, fontweight='bold')
ax6.set_ylabel('平均符合条件的Bundle数量', fontsize=12, fontweight='bold')
ax6.set_title('季节性模式\n(按周统计2021-2025平均)', fontsize=13, fontweight='bold')
ax6.legend(loc='upper right', fontsize=10)
ax6.grid(True, alpha=0.3)

# Add month labels
month_weeks = [1, 5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48]
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax6.set_xticks(month_weeks)
ax6.set_xticklabels(month_names)

# ============================================================================
# Overall title
# ============================================================================
fig.suptitle('D2DS可行性分析：符合条件的Bundle Pool vs 需要选择的Sample Size',
            fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()

# Save
output_file = output_dir / "eligible_pool_vs_sample_visualization.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to {output_file}")

plt.close()

# ============================================================================
# Create a simple infographic-style figure
# ============================================================================
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Simple bar comparison
categories = ['DH\nBundles', 'Non-DH\nBundles', 'Total']
pool = [17.9, 32.8, 50.7]
sample = [4, 2, 6]

x = np.arange(len(categories))
width = 0.35

ax1.bar(x - width/2, pool, width, label='Pool (符合条件)', color='#3498db', alpha=0.9)
ax1.bar(x + width/2, sample, width, label='Sample (需要选择)', color='#e74c3c', alpha=0.9)

for i in range(len(categories)):
    ax1.text(i, max(pool[i], sample[i]) + 2, f'{pool[i]/sample[i]:.1f}×',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

ax1.set_ylabel('Bundle数量', fontsize=13, fontweight='bold')
ax1.set_title('每周平均情况', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=12)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Pie chart showing sufficiency
labels = ['≥4个DH\n(充足)', '<4个DH\n(不足)']
sizes = [(df['eligible_dh'] >= 4).sum(), (df['eligible_dh'] < 4).sum()]
colors = ['#2ecc71', '#e74c3c']
explode = (0.1, 0)

ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('DH Bundles充足性\n(261周历史数据)', fontsize=14, fontweight='bold')

# Plot 3: Box plot
data_to_plot = [df['eligible_dh'], df['eligible_non_dh']]
bp = ax3.boxplot(data_to_plot, labels=['DH\nBundles', 'Non-DH\nBundles'],
                patch_artist=True, showmeans=True)

for patch, color in zip(bp['boxes'], ['lightgreen', 'lightsalmon']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.axhline(y=4, color='red', linestyle='--', linewidth=2, label='需要: 4个DH')
ax3.axhline(y=2, color='darkred', linestyle='--', linewidth=2, label='需要: 2个Non-DH')
ax3.set_ylabel('符合条件的Bundle数量', fontsize=13, fontweight='bold')
ax3.set_title('分布统计\n(箱线图)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Summary statistics table
summary_data = [
    ['DH Bundles', f"{df['eligible_dh'].mean():.1f}", f"{df['eligible_dh'].median():.0f}",
     f"{df['eligible_dh'].min()}", f"{df['eligible_dh'].max()}"],
    ['Non-DH Bundles', f"{df['eligible_non_dh'].mean():.1f}", f"{df['eligible_non_dh'].median():.0f}",
     f"{df['eligible_non_dh'].min()}", f"{df['eligible_non_dh'].max()}"],
    ['Total', f"{df['total_eligible'].mean():.1f}", f"{df['total_eligible'].median():.0f}",
     f"{df['total_eligible'].min()}", f"{df['total_eligible'].max()}"]
]

ax4.axis('tight')
ax4.axis('off')

table = ax4.table(cellText=summary_data,
                 colLabels=['类型', '平均值', '中位数', '最小值', '最大值'],
                 cellLoc='center',
                 loc='center',
                 colColours=['lightgray']*5)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
colors_row = ['#abebc6', '#fadbd8', '#d6eaf8']
for i in range(1, 4):
    for j in range(5):
        table[(i, j)].set_facecolor(colors_row[i-1])

ax4.set_title('统计摘要\n(2021-2025, 261周)', fontsize=14, fontweight='bold', pad=20)

fig2.suptitle('D2DS采样可行性总结', fontsize=16, fontweight='bold')
plt.tight_layout()

output_file2 = output_dir / "eligible_pool_summary.png"
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✓ Saved summary to {output_file2}")

print("\n" + "="*70)
print("All visualizations created successfully!")
print("="*70)
