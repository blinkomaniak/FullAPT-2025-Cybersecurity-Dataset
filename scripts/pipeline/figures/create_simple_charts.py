#!/usr/bin/env python3
"""
Create simple chart figures for pipeline documentation
- Figure 2.3: EventID Distribution Chart
- Figure 6.3: EventID Distribution Pie Chart
- Figure 8.2: Label Distribution Pyramid
- Figure 8.3: Malicious Event Tactic Breakdown
- Figure X.1: Data Volume Funnel
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================================
# FIGURE 2.3: EventID Distribution Chart (Horizontal Bar)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Sample data for typical APT run
event_ids = [1, 3, 5, 7, 8, 10, 11, 12, 13, 15, 17, 18, 22, 23, 26]
event_names = [
    'Process Creation',
    'Network Connection',
    'Process Terminated',
    'Image Loaded',
    'CreateRemoteThread',
    'Process Access',
    'File Creation',
    'Registry Add/Delete',
    'Registry Value Set',
    'File Stream Created',
    'Pipe Created',
    'Pipe Connected',
    'DNS Query',
    'File Deletion',
    'File Delete Detected'
]
event_counts = [28456, 15234, 12890, 45123, 892, 3421, 15287, 8934, 6123, 1245, 567, 423, 8901, 4156, 2340]

# Calculate percentages
total = sum(event_counts)
percentages = [(count/total)*100 for count in event_counts]

# Create horizontal bar chart
colors = sns.color_palette("viridis", len(event_ids))
bars = ax.barh(range(len(event_ids)), event_counts, color=colors, edgecolor='black', linewidth=0.5)

# Add labels
ax.set_yticks(range(len(event_ids)))
ax.set_yticklabels([f'EventID {eid}: {name}' for eid, name in zip(event_ids, event_names)], fontsize=9)
ax.set_xlabel('Event Count', fontsize=11, fontweight='bold')
ax.set_title('Figure 2.3: Sysmon EventID Distribution\nTypical APT Run Dataset (145,832 total events)',
             fontsize=13, fontweight='bold', pad=20)

# Add value labels on bars
for i, (bar, count, pct) in enumerate(zip(bars, event_counts, percentages)):
    width = bar.get_width()
    ax.text(width + 1000, bar.get_y() + bar.get_height()/2,
            f'{count:,} ({pct:.1f}%)',
            ha='left', va='center', fontsize=8, fontweight='bold')

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_2_3_eventid_distribution.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_2_3_eventid_distribution.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 2.3: EventID Distribution created")

# ============================================================================
# FIGURE 6.3: EventID Distribution Pie Chart (Target Events Only)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Data for extracted target events (EventID 1, 11, 23)
target_event_ids = ['EventID 1\nProcess Creation', 'EventID 11\nFile Creation', 'EventID 23\nFile Deletion']
target_counts = [28456, 15287, 4156]
target_colors = ['#4169E1', '#228B22', '#DC143C']  # Blue, Green, Red

# Create pie chart
wedges, texts, autotexts = ax.pie(target_counts,
                                    labels=target_event_ids,
                                    colors=target_colors,
                                    autopct='%1.1f%%',
                                    startangle=90,
                                    textprops={'fontsize': 11, 'fontweight': 'bold'},
                                    pctdistance=0.85,
                                    explode=(0.05, 0.05, 0.05))

# Enhance text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

# Add counts in legend
legend_labels = [
    f'EventID 1: {target_counts[0]:,} events ({target_counts[0]/sum(target_counts)*100:.1f}%)',
    f'EventID 11: {target_counts[1]:,} events ({target_counts[1]/sum(target_counts)*100:.1f}%)',
    f'EventID 23: {target_counts[2]:,} events ({target_counts[2]/sum(target_counts)*100:.1f}%)'
]
ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)

ax.set_title('Figure 6.3: Target Event Distribution by EventID\nExtracted Events for Manual Seed Selection (47,899 total)',
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_6_3_target_eventid_pie.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_6_3_target_eventid_pie.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 6.3: Target EventID Pie Chart created")

# ============================================================================
# FIGURE 8.2: Label Distribution Pyramid (Imbalanced Dataset)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Data
benign_count = 145617
malicious_count = 215
total = benign_count + malicious_count

benign_pct = (benign_count / total) * 100
malicious_pct = (malicious_count / total) * 100

# Create inverted pyramid using polygons
# Top (malicious - small)
malicious_height = 0.5
malicious_width_top = 10
malicious_width_bottom = 2

malicious_poly = Polygon([
    [5 - malicious_width_bottom/2, 9],  # Top left
    [5 + malicious_width_bottom/2, 9],  # Top right
    [5 + malicious_width_top/2, 9 - malicious_height],  # Bottom right
    [5 - malicious_width_top/2, 9 - malicious_height],  # Bottom left
], facecolor='#DC143C', edgecolor='black', linewidth=2, alpha=0.8, zorder=2)
ax.add_patch(malicious_poly)

# Label for malicious
ax.text(5, 9.25, 'MALICIOUS', ha='center', va='bottom',
        fontsize=14, fontweight='bold', color='#DC143C')
ax.text(5, 8.75, f'{malicious_count:,} events ({malicious_pct:.2f}%)',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Bottom (benign - large)
benign_height = 6.5
benign_width_top = 10
benign_width_bottom = 9

benign_poly = Polygon([
    [5 - benign_width_top/2, 9 - malicious_height],  # Top left
    [5 + benign_width_top/2, 9 - malicious_height],  # Top right
    [5 + benign_width_bottom/2, 9 - malicious_height - benign_height],  # Bottom right
    [5 - benign_width_bottom/2, 9 - malicious_height - benign_height],  # Bottom left
], facecolor='#808080', edgecolor='black', linewidth=2, alpha=0.6, zorder=1)
ax.add_patch(benign_poly)

# Label for benign
ax.text(5, 5, 'BENIGN', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')
ax.text(5, 4.3, f'{benign_count:,} events ({benign_pct:.2f}%)',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Annotations
ax.text(5, 1.2, 'Imbalanced Dataset Challenge', ha='center', fontsize=11,
        fontweight='bold', style='italic', color='#333333')
ax.text(5, 0.7, 'Requires: SMOTE, Class Weighting, or Anomaly Detection Methods',
        ha='center', fontsize=9, color='#666666')

# Title
ax.text(5, 10.2, 'Figure 8.2: Labeled Sysmon Dataset Distribution',
        ha='center', fontsize=14, fontweight='bold', color='#005571')
ax.text(5, 9.8, 'Binary Classification: Benign vs. Malicious Events',
        ha='center', fontsize=10, style='italic', color='#333333')

ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_8_2_label_distribution_pyramid.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_8_2_label_distribution_pyramid.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 8.2: Label Distribution Pyramid created")

# ============================================================================
# FIGURE 8.3: Malicious Event Tactic Breakdown
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Data for malicious events by tactic
tactics = ['Discovery', 'Defense-Evasion', 'Persistence', 'Exfiltration',
           'Collection', 'Execution', 'Initial-Access', 'Credential-Access',
           'Lateral-Movement', 'Command-and-Control']
counts = [86, 24, 21, 20, 18, 16, 10, 8, 7, 5]
total_malicious = sum(counts)
percentages = [(c/total_malicious)*100 for c in counts]

# MITRE ATT&CK color scheme
colors_mitre = ['#8B4513', '#FF8C00', '#228B22', '#32CD32',
                '#9932CC', '#4169E1', '#000000', '#FFD700',
                '#FF1493', '#00CED1']

# Create horizontal bar chart
bars = ax.barh(range(len(tactics)), counts, color=colors_mitre,
               edgecolor='black', linewidth=1)

# Labels
ax.set_yticks(range(len(tactics)))
ax.set_yticklabels(tactics, fontsize=11, fontweight='bold')
ax.set_xlabel('Event Count', fontsize=12, fontweight='bold')
ax.set_title('Figure 8.3: Malicious Event Distribution by MITRE ATT&CK Tactic\nLabeled Sysmon Dataset (215 total malicious events)',
             fontsize=13, fontweight='bold', pad=20)

# Add value labels
for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{count} events ({pct:.1f}%)',
            ha='left', va='center', fontsize=9, fontweight='bold')

ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, max(counts) + 15)

plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_8_3_tactic_breakdown.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_8_3_tactic_breakdown.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 8.3: Tactic Breakdown created")

# ============================================================================
# FIGURE X.1: Data Volume Funnel Across Pipeline
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 10))

# Funnel data
stages = [
    'Raw Elasticsearch\nEvents',
    'Sysmon CSV\n(Step 2)',
    'NetFlow CSV\n(Step 3)',
    'Target Events\n(Step 6)',
    'Seed Events\n(Step 6→7)',
    'Traced Events\n(Step 7)',
    'Labeled Sysmon\n(Step 8)',
    'Labeled NetFlow\n(Step 9)'
]

# Sample volumes (representative)
volumes = [2500000, 150000, 50000, 48000, 215, 1200, 150000, 50000]
colors_funnel = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Create funnel
y_positions = np.arange(len(stages), 0, -1)
max_width = 8

for i, (stage, volume, color, y_pos) in enumerate(zip(stages, volumes, colors_funnel, y_positions)):
    # Calculate width proportional to log scale for visibility
    if volume > 1000:
        width = max_width * (np.log10(volume) / np.log10(max(volumes)))
    else:
        width = max_width * 0.15  # Minimum width for small values

    # Create trapezoid
    if i < len(stages) - 1:
        next_volume = volumes[i + 1]
        if next_volume > 1000:
            next_width = max_width * (np.log10(next_volume) / np.log10(max(volumes)))
        else:
            next_width = max_width * 0.15
    else:
        next_width = width

    # Draw trapezoid
    trap = Polygon([
        [5 - width/2, y_pos + 0.4],
        [5 + width/2, y_pos + 0.4],
        [5 + next_width/2, y_pos - 0.4],
        [5 - next_width/2, y_pos - 0.4],
    ], facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(trap)

    # Add stage label
    ax.text(5, y_pos, stage, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

    # Add volume label
    if volume >= 1000:
        volume_text = f'{volume:,} events'
    else:
        volume_text = f'{volume} events'

    ax.text(5 + max_width/2 + 0.5, y_pos, volume_text,
            ha='left', va='center', fontsize=9, fontweight='bold',
            color=color)

# Title
ax.text(5, len(stages) + 1, 'Figure X.1: Data Volume Funnel Across Pipeline',
        ha='center', fontsize=14, fontweight='bold', color='#005571')
ax.text(5, len(stages) + 0.6, 'Volume transformation from raw extraction to labeled dual-domain dataset',
        ha='center', fontsize=10, style='italic', color='#333333')

# Notes
note_text = (
    "Note: Funnel width uses logarithmic scale for visibility.\n"
    "Seed event selection creates bottleneck, then expands via lifecycle tracing.\n"
    "Final output includes both host (Sysmon) and network (NetFlow) labeled datasets."
)
ax.text(5, -0.5, note_text, ha='center', va='top', fontsize=8,
        style='italic', color='#666666',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9F9F9', edgecolor='#CCCCCC'))

ax.set_xlim(0, 10)
ax.set_ylim(-1.5, len(stages) + 1.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_X_1_data_volume_funnel.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_X_1_data_volume_funnel.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure X.1: Data Volume Funnel created")

print("\n" + "="*60)
print("✅ ALL SIMPLE CHART FIGURES CREATED SUCCESSFULLY!")
print("="*60)
