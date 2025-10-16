#!/usr/bin/env python3
"""
Figure 1.1: Elasticsearch Data Extraction Architecture
System architecture diagram showing data extraction pipeline from Elasticsearch cluster
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Create figure with specific size for clarity
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
color_elasticsearch = '#005571'  # Elasticsearch teal
color_process = '#FEC514'  # Yellow for processing
color_output = '#00BFB3'  # Teal for output
color_arrow = '#666666'  # Gray for arrows
color_text = '#333333'  # Dark gray for text

# ============================================================================
# 1. ELASTICSEARCH CLUSTER (Left side)
# ============================================================================
# Main cluster box
es_box = FancyBboxPatch(
    (0.5, 3.5), 1.8, 3.0,
    boxstyle="round,pad=0.1",
    linewidth=2,
    edgecolor=color_elasticsearch,
    facecolor='#E8F4F8',
    zorder=2
)
ax.add_patch(es_box)

# Cloud icon simulation (circles)
cloud_circles = [
    Circle((1.4, 6.2), 0.25, color=color_elasticsearch, alpha=0.3, zorder=1),
    Circle((1.15, 6.0), 0.3, color=color_elasticsearch, alpha=0.3, zorder=1),
    Circle((1.65, 6.0), 0.3, color=color_elasticsearch, alpha=0.3, zorder=1),
]
for circle in cloud_circles:
    ax.add_patch(circle)

# Elasticsearch label
ax.text(1.4, 5.5, 'Elasticsearch\nCluster',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color=color_elasticsearch)

# IP Address
ax.text(1.4, 4.8, '10.2.0.20:9200',
        ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_elasticsearch),
        color=color_text)

# Index types
ax.text(1.4, 4.2, '• sysmon indices',
        ha='center', va='center', fontsize=8, style='italic',
        color=color_text)
ax.text(1.4, 3.9, '• network_traffic indices',
        ha='center', va='center', fontsize=8, style='italic',
        color=color_text)

# ============================================================================
# 2. SCROLL API MECHANISM (Center)
# ============================================================================
scroll_box = FancyBboxPatch(
    (3.5, 4.2), 2.0, 2.0,
    boxstyle="round,pad=0.1",
    linewidth=2,
    edgecolor=color_process,
    facecolor='#FFF9E6',
    zorder=2
)
ax.add_patch(scroll_box)

ax.text(4.5, 5.7, 'Scroll API',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color=color_process)

ax.text(4.5, 5.3, 'Bulk Download',
        ha='center', va='center', fontsize=9,
        color=color_text)

# Pagination visualization (small boxes representing chunks)
pagination_y = 4.8
chunk_width = 0.15
chunk_spacing = 0.05
for i in range(5):
    chunk_x = 3.9 + i * (chunk_width + chunk_spacing)
    chunk = mpatches.Rectangle(
        (chunk_x, pagination_y), chunk_width, 0.2,
        linewidth=1,
        edgecolor=color_process,
        facecolor=color_process,
        alpha=0.6
    )
    ax.add_patch(chunk)

ax.text(4.5, 4.5, 'Paginated Batches',
        ha='center', va='center', fontsize=7, style='italic',
        color=color_text)

# ============================================================================
# 3. OUTPUT FILES (Right side)
# ============================================================================

# Sysmon output (upper)
sysmon_box = FancyBboxPatch(
    (6.5, 5.5), 3.0, 1.2,
    boxstyle="round,pad=0.08",
    linewidth=2,
    edgecolor=color_output,
    facecolor='#E6F9F7',
    zorder=2
)
ax.add_patch(sysmon_box)

# File icon simulation (document shape)
file_icon_sysmon = mpatches.FancyBboxPatch(
    (6.7, 5.7), 0.3, 0.8,
    boxstyle="round,pad=0.02",
    linewidth=1.5,
    edgecolor=color_output,
    facecolor='white',
    zorder=3
)
ax.add_patch(file_icon_sysmon)
# File lines
for i in range(3):
    ax.plot([6.75, 6.95], [6.3 - i*0.15, 6.3 - i*0.15],
            color=color_output, linewidth=1, zorder=4)

ax.text(7.3, 6.35, 'ds-logs-windows-sysmon',
        ha='left', va='center', fontsize=8, fontweight='bold',
        color=color_text)
ax.text(7.3, 6.05, '_operational-default',
        ha='left', va='center', fontsize=8, fontweight='bold',
        color=color_text)
ax.text(7.3, 5.75, '-run-XX.jsonl.gz',
        ha='left', va='center', fontsize=8, fontweight='bold',
        color=color_output)

# Compression indicator
ax.text(9.2, 5.65, '📦 GZIP',
        ha='center', va='center', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFE6E6', edgecolor='none'),
        color=color_text)

# NetFlow output (lower)
netflow_box = FancyBboxPatch(
    (6.5, 3.8), 3.0, 1.2,
    boxstyle="round,pad=0.08",
    linewidth=2,
    edgecolor=color_output,
    facecolor='#E6F9F7',
    zorder=2
)
ax.add_patch(netflow_box)

# File icon simulation
file_icon_netflow = mpatches.FancyBboxPatch(
    (6.7, 4.0), 0.3, 0.8,
    boxstyle="round,pad=0.02",
    linewidth=1.5,
    edgecolor=color_output,
    facecolor='white',
    zorder=3
)
ax.add_patch(file_icon_netflow)
# File lines
for i in range(3):
    ax.plot([6.75, 6.95], [4.6 - i*0.15, 4.6 - i*0.15],
            color=color_output, linewidth=1, zorder=4)

ax.text(7.3, 4.65, 'ds-logs-network_traffic',
        ha='left', va='center', fontsize=8, fontweight='bold',
        color=color_text)
ax.text(7.3, 4.35, '-flow-default',
        ha='left', va='center', fontsize=8, fontweight='bold',
        color=color_text)
ax.text(7.3, 4.05, '-run-XX.jsonl.gz',
        ha='left', va='center', fontsize=8, fontweight='bold',
        color=color_output)

# Compression indicator
ax.text(9.2, 3.95, '📦 GZIP',
        ha='center', va='center', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFE6E6', edgecolor='none'),
        color=color_text)

# ============================================================================
# 4. ARROWS (Data flow)
# ============================================================================

# ES to Scroll API
arrow1 = FancyArrowPatch(
    (2.3, 5.5), (3.5, 5.5),
    arrowstyle='->,head_width=0.4,head_length=0.3',
    linewidth=2.5,
    color=color_arrow,
    zorder=1
)
ax.add_patch(arrow1)
ax.text(2.9, 5.8, 'HTTPS', ha='center', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'),
        color=color_text)

# Scroll API to Sysmon output
arrow2 = FancyArrowPatch(
    (5.5, 5.6), (6.5, 6.1),
    arrowstyle='->,head_width=0.4,head_length=0.3',
    linewidth=2.5,
    color=color_arrow,
    connectionstyle="arc3,rad=0.2",
    zorder=1
)
ax.add_patch(arrow2)
ax.text(5.8, 6.3, 'Sysmon\nEvents', ha='center', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F4F8', edgecolor='none'),
        color=color_text)

# Scroll API to NetFlow output
arrow3 = FancyArrowPatch(
    (5.5, 5.0), (6.5, 4.4),
    arrowstyle='->,head_width=0.4,head_length=0.3',
    linewidth=2.5,
    color=color_arrow,
    connectionstyle="arc3,rad=-0.2",
    zorder=1
)
ax.add_patch(arrow3)
ax.text(5.8, 4.1, 'Network\nFlows', ha='center', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F4F8', edgecolor='none'),
        color=color_text)

# ============================================================================
# 5. TITLE AND ANNOTATIONS
# ============================================================================

# Main title
ax.text(5.0, 8.5, 'Figure 1.1: Elasticsearch Data Extraction Architecture',
        ha='center', va='center', fontsize=14, fontweight='bold',
        color=color_elasticsearch)

# Subtitle
ax.text(5.0, 8.0, 'Pipeline Step 1: Raw Data Extraction from Monitoring Infrastructure',
        ha='center', va='center', fontsize=10, style='italic',
        color=color_text)

# Statistics box (bottom)
stats_box = FancyBboxPatch(
    (0.5, 0.5), 9.0, 1.5,
    boxstyle="round,pad=0.1",
    linewidth=1,
    edgecolor='#CCCCCC',
    facecolor='#F9F9F9',
    zorder=1
)
ax.add_patch(stats_box)

ax.text(5.0, 1.6, 'Typical Dataset Characteristics',
        ha='center', va='center', fontsize=9, fontweight='bold',
        color=color_text)

stats_text = (
    "• Volume: 100K-1M+ events per APT run   "
    "• Compression: ~60-80% size reduction (GZIP)   "
    "• Format: JSON Lines (JSONL)\n"
    "• Sources: Windows Sysmon (host events) + Network Traffic Flow (network events)   "
    "• Authentication: Username/password   • SSL: Disabled (lab environment)"
)
ax.text(5.0, 0.9, stats_text,
        ha='center', va='center', fontsize=7,
        color=color_text)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#E8F4F8', edgecolor=color_elasticsearch, linewidth=2, label='Data Source'),
    mpatches.Patch(facecolor='#FFF9E6', edgecolor=color_process, linewidth=2, label='Processing'),
    mpatches.Patch(facecolor='#E6F9F7', edgecolor=color_output, linewidth=2, label='Output Files'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

# ============================================================================
# 6. SAVE FIGURE
# ============================================================================

plt.tight_layout()
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_1_1_elasticsearch_architecture.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_1_1_elasticsearch_architecture.pdf',
            bbox_inches='tight', facecolor='white')

print("✅ Figure 1.1 created successfully!")
print("📁 Saved to: /home/researcher/Downloads/research/scripts/pipeline/figures/")
print("   - figure_1_1_elasticsearch_architecture.png (300 DPI)")
print("   - figure_1_1_elasticsearch_architecture.pdf (vector)")
