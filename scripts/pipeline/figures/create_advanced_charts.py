#!/usr/bin/env python3
"""
Create advanced chart figures for pipeline documentation
- Figure 6.1: Target Event Filtering Funnel
- Figure 7.3: Attack Lifecycle Expansion Waterfall
- Figure 7.4: Timeline Visualization Example
- Figure 4.2: Attribution Rate by APT Campaign
- Figure 3.3: Performance Scaling Chart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")

# ============================================================================
# FIGURE 6.1: Target Event Filtering Logic Funnel
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 10))

# Funnel stages
stages = [
    ('Raw Sysmon Events\n(All EventIDs)', 145832, '#B0B0B0'),
    ('Filter EventID 1, 11, 23', 47899, '#FDB462'),
    ('Manual Analyst Review', 47899, '#FFED6F'),
    ('Marked Seed Events', 215, '#DC143C')
]

y_start = 9
stage_height = 1.5
max_width = 8

for i, (label, count, color) in enumerate(stages):
    y_pos = y_start - i * stage_height

    # Calculate width
    if i == 0:
        width = max_width
    else:
        width = max_width * (np.log10(count + 1) / np.log10(stages[0][1] + 1))
        width = max(width, max_width * 0.2)  # Minimum width

    # Next width
    if i < len(stages) - 1:
        next_count = stages[i + 1][1]
        next_width = max_width * (np.log10(next_count + 1) / np.log10(stages[0][1] + 1))
        next_width = max(next_width, max_width * 0.2)
    else:
        next_width = width

    # Draw trapezoid
    trap = Polygon([
        [5 - width/2, y_pos],
        [5 + width/2, y_pos],
        [5 + next_width/2, y_pos - stage_height + 0.2],
        [5 - next_width/2, y_pos - stage_height + 0.2],
    ], facecolor=color, edgecolor='black', linewidth=2.5, alpha=0.8)
    ax.add_patch(trap)

    # Stage label
    ax.text(5, y_pos - stage_height/2 + 0.1, label,
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='black' if i < 2 else 'white')

    # Count and percentage
    pct = (count / stages[0][1]) * 100
    count_label = f'{count:,} events ({pct:.1f}%)'
    ax.text(5 + max_width/2 + 0.8, y_pos - stage_height/2 + 0.1, count_label,
            ha='left', va='center', fontsize=10, fontweight='bold',
            color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=2))

# Arrow annotations
arrow1 = FancyArrowPatch(
    (5 + max_width/2 + 0.3, y_start - stage_height + 0.1),
    (5 + max_width/2 + 0.3, y_start - 2*stage_height + 0.1),
    arrowstyle='->,head_width=0.4,head_length=0.3',
    linewidth=2, color='#333333', linestyle='--'
)
ax.add_patch(arrow1)
ax.text(5 + max_width/2 + 0.6, y_start - 1.5*stage_height, '32.8%\nretained',
        ha='left', va='center', fontsize=8, style='italic',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9E6'))

arrow2 = FancyArrowPatch(
    (5 + max_width/2 + 0.3, y_start - 3*stage_height + 0.1),
    (5 + max_width/2 + 0.3, y_start - 4*stage_height + 0.3),
    arrowstyle='->,head_width=0.4,head_length=0.3',
    linewidth=2, color='#DC143C', linestyle='--'
)
ax.add_patch(arrow2)
ax.text(5 + max_width/2 + 0.6, y_start - 3.5*stage_height, '0.14%\nmarked',
        ha='left', va='center', fontsize=8, style='italic',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFE6E6'))

# Title
ax.text(5, 10.5, 'Figure 6.1: Target Event Filtering Logic',
        ha='center', fontsize=14, fontweight='bold', color='#005571')
ax.text(5, 10.1, 'Sysmon Seed Event Extraction Funnel',
        ha='center', fontsize=10, style='italic', color='#333333')

# Note
note = 'Human analyst reviews extracted events and marks significant attack operations as "seed events"\nwith MITRE ATT&CK Tactic and Technique labels for downstream lifecycle tracing.'
ax.text(5, 1.5, note, ha='center', fontsize=8, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9F9F9', edgecolor='#CCCCCC'),
        color='#666666')

ax.set_xlim(0, 10)
ax.set_ylim(0.5, 11)
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_6_1_filtering_funnel.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_6_1_filtering_funnel.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 6.1: Filtering Funnel created")

# ============================================================================
# FIGURE 7.3: Attack Lifecycle Expansion Waterfall
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Data
categories = ['Seed Events', '+ ProcessGuid\nTraced (ID 1)', '+ File Ops\n(ID 11)', '+ File Deletes\n(ID 23)', 'Total Traced']
values = [15, 198, 35, 14, 0]  # Last is calculated
colors_waterfall = ['#DC143C', '#4169E1', '#228B22', '#8B4513', '#9932CC']

# Calculate cumulative
cumulative = [values[0]]
for i in range(1, len(values)-1):
    cumulative.append(cumulative[-1] + values[i])
cumulative.append(cumulative[-1])  # Total

# Plot waterfall
x_positions = np.arange(len(categories))
bar_width = 0.6

for i in range(len(categories)):
    if i == 0:
        # Starting bar (seeds)
        ax.bar(x_positions[i], values[i], bar_width,
               color=colors_waterfall[i], edgecolor='black', linewidth=2,
               label='Seed Events')
        ax.text(x_positions[i], values[i]/2, f'{values[i]}',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    elif i < len(categories) - 1:
        # Incremental bars
        bottom = cumulative[i-1]
        height = values[i]
        ax.bar(x_positions[i], height, bar_width, bottom=bottom,
               color=colors_waterfall[i], edgecolor='black', linewidth=2,
               alpha=0.9)
        ax.text(x_positions[i], bottom + height/2, f'+{height}',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # Connection line
        ax.plot([x_positions[i-1] + bar_width/2, x_positions[i] - bar_width/2],
                [cumulative[i-1], cumulative[i-1]],
                'k--', linewidth=1.5, alpha=0.5)
    else:
        # Total bar
        total_val = cumulative[-1]
        ax.bar(x_positions[i], total_val, bar_width,
               color=colors_waterfall[i], edgecolor='black', linewidth=2.5,
               alpha=0.9)
        ax.text(x_positions[i], total_val/2, f'{int(total_val)}',
                ha='center', va='center', fontsize=13, fontweight='bold', color='white')

        # Connection line
        ax.plot([x_positions[i-1] + bar_width/2, x_positions[i] - bar_width/2],
                [cumulative[i-1], cumulative[i-1]],
                'k--', linewidth=1.5, alpha=0.5)

# Labels
ax.set_xticks(x_positions)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylabel('Event Count', fontsize=12, fontweight='bold')
ax.set_title('Figure 7.3: Attack Lifecycle Expansion Metrics\nFrom Seed Events to Complete Attack Traces',
             fontsize=14, fontweight='bold', pad=20)

ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, cumulative[-1] * 1.1)

# Add expansion rate annotation
expansion_rate = cumulative[-1] / values[0]
ax.text(2, cumulative[-1] * 1.05, f'Expansion Rate: {expansion_rate:.1f}x',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6', edgecolor='#FDB462', linewidth=2))

plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_7_3_expansion_waterfall.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_7_3_expansion_waterfall.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 7.3: Expansion Waterfall created")

# ============================================================================
# FIGURE 7.4: Timeline Visualization Example
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Simulate attack timeline data
np.random.seed(42)
n_events = 50
timestamps = np.sort(np.random.uniform(0, 3600, n_events))  # 1 hour in seconds

# Hostnames
hosts = ['victim-workstation.local', 'dc-server.local', 'file-server.local']
host_assignments = np.random.choice([0, 1, 2], n_events, p=[0.6, 0.25, 0.15])

# Tactics
tactics = ['initial-access', 'execution', 'persistence', 'discovery', 'collection', 'exfiltration']
tactic_assignments = []
for i, ts in enumerate(timestamps):
    if ts < 600:
        tactic_assignments.append(0)  # initial-access
    elif ts < 1200:
        tactic_assignments.append(1)  # execution
    elif ts < 1800:
        tactic_assignments.append(np.random.choice([2, 3]))  # persistence/discovery
    elif ts < 2400:
        tactic_assignments.append(np.random.choice([3, 4]))  # discovery/collection
    else:
        tactic_assignments.append(np.random.choice([4, 5]))  # collection/exfiltration

# Tactic colors (MITRE)
tactic_colors_map = {
    'initial-access': '#000000',
    'execution': '#4169E1',
    'persistence': '#228B22',
    'discovery': '#8B4513',
    'collection': '#9932CC',
    'exfiltration': '#32CD32'
}

# Mark first 3 as seed events
is_seed = [True] * 3 + [False] * (n_events - 3)

# Plot events
for i in range(n_events):
    ts = timestamps[i]
    host_idx = host_assignments[i]
    tactic = tactics[tactic_assignments[i]]
    color = tactic_colors_map[tactic]

    if is_seed[i]:
        # Seed event - red star
        ax.scatter(ts, host_idx, marker='*', s=400, color='#DC143C',
                   edgecolors='black', linewidths=2, zorder=10, label='Seed Event' if i == 0 else '')
    else:
        # Traced event - colored circle
        ax.scatter(ts, host_idx, marker='o', s=150, color=color,
                   edgecolors='black', linewidths=1, alpha=0.8, zorder=5,
                   label=tactic.title() if i == 3 else '')

# Format
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(hosts, fontsize=10)
ax.set_xlabel('Time (seconds since attack start)', fontsize=11, fontweight='bold')
ax.set_ylabel('Computer / Host', fontsize=11, fontweight='bold')
ax.set_title('Figure 7.4: Attack Lifecycle Timeline Visualization\nSeed Events (★) and Traced Events (●) Color-Coded by MITRE ATT&CK Tactic',
             fontsize=13, fontweight='bold', pad=20)

# Add phase annotations
phase_boundaries = [0, 600, 1200, 1800, 2400, 3000, 3600]
phase_labels = ['Initial\nAccess', 'Execution', 'Persistence\n& Discovery', 'Collection', 'Exfiltration']
for i in range(len(phase_labels)):
    mid_point = (phase_boundaries[i] + phase_boundaries[i+1]) / 2
    ax.text(mid_point, 2.7, phase_labels[i], ha='center', va='center',
            fontsize=8, style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', alpha=0.7))

ax.set_ylim(-0.5, 3)
ax.grid(axis='x', alpha=0.3)

# Legend
handles, labels = ax.get_legend_handles_labels()
# Remove duplicates
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_7_4_timeline_example.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_7_4_timeline_example.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 7.4: Timeline Visualization created")

# ============================================================================
# FIGURE 4.2: Attribution Rate by APT Campaign
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Data (sample attribution rates)
apt_types = ['APT-1', 'APT-2', 'APT-3', 'APT-4', 'APT-5', 'APT-6']
mean_rates = [72.5, 68.3, 71.8, 85.4, 82.1, 65.7]
std_devs = [8.2, 12.1, 9.5, 6.3, 7.8, 14.2]

# Colors based on performance
colors_apt = []
for rate in mean_rates:
    if rate >= 80:
        colors_apt.append('#228B22')  # Green - high performing
    elif rate >= 70:
        colors_apt.append('#FDB462')  # Yellow - medium
    else:
        colors_apt.append('#DC143C')  # Red - low

# Create bar chart
x_pos = np.arange(len(apt_types))
bars = ax.bar(x_pos, mean_rates, yerr=std_devs, capsize=5,
              color=colors_apt, edgecolor='black', linewidth=2, alpha=0.8)

# Add horizontal line at 90% threshold
ax.axhline(y=90, color='#228B22', linestyle='--', linewidth=2, label='High-Performing Threshold (90%)')

# Add value labels
for i, (bar, rate, std) in enumerate(zip(bars, mean_rates, std_devs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + std + 2,
            f'{rate:.1f}%\n±{std:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Labels
ax.set_xticks(x_pos)
ax.set_xticklabels(apt_types, fontsize=11, fontweight='bold')
ax.set_ylabel('Attribution Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 4.2: NetFlow Attribution Rate by APT Campaign\nDual-Domain Temporal Causation Correlation Performance',
             fontsize=13, fontweight='bold', pad=20)

ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

# Add performance legend
legend_elements = [
    mpatches.Patch(facecolor='#228B22', edgecolor='black', label='High (≥80%)'),
    mpatches.Patch(facecolor='#FDB462', edgecolor='black', label='Medium (70-80%)'),
    mpatches.Patch(facecolor='#DC143C', edgecolor='black', label='Low (<70%)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, title='Performance')

plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_4_2_attribution_by_campaign.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_4_2_attribution_by_campaign.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 4.2: Attribution by Campaign created")

# ============================================================================
# FIGURE 3.3: Performance Scaling Chart
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Data
workers = [1, 2, 4, 8, 16, 32]
events_per_sec = [8500, 16200, 30800, 55300, 89200, 102400]

# Calculate speedup
speedup = [eps / events_per_sec[0] for eps in events_per_sec]
ideal_speedup = [1, 2, 4, 8, 16, 32]

# Create plot with dual y-axes
ax1 = ax
ax2 = ax1.twinx()

# Plot throughput
line1 = ax1.plot(workers, events_per_sec, 'o-', linewidth=3, markersize=10,
                 color='#4169E1', label='Events/Second')
ax1.fill_between(workers, events_per_sec, alpha=0.3, color='#4169E1')

# Plot speedup
line2 = ax2.plot(workers, speedup, 's--', linewidth=2, markersize=8,
                 color='#228B22', label='Actual Speedup')
line3 = ax2.plot(workers, ideal_speedup, '^:', linewidth=1.5, markersize=6,
                 color='#DC143C', label='Ideal (Linear) Speedup')

# Labels
ax1.set_xlabel('Number of Workers (CPU Cores)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Processing Rate (Events/Second)', fontsize=11, fontweight='bold', color='#4169E1')
ax2.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold', color='#228B22')

ax1.tick_params(axis='y', labelcolor='#4169E1')
ax2.tick_params(axis='y', labelcolor='#228B22')

ax.set_title('Figure 3.3: Multi-Threading Performance Scaling\nNetFlow CSV Creator Processing Rate vs. Worker Count',
             fontsize=13, fontweight='bold', pad=20)

# Grid
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.set_xticks(workers)
ax1.set_xticklabels(workers)

# Combined legend
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10)

# Add efficiency annotation
efficiency_16 = (speedup[4] / 16) * 100
ax2.text(16, speedup[4] + 2, f'Efficiency @ 16 cores:\n{efficiency_16:.1f}%',
         ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6', edgecolor='#FDB462'))

plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_3_3_performance_scaling.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_3_3_performance_scaling.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 3.3: Performance Scaling created")

print("\n" + "="*60)
print("✅ ALL ADVANCED CHART FIGURES CREATED SUCCESSFULLY!")
print("="*60)
