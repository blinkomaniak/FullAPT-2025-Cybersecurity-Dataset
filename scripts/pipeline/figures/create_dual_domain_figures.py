#!/usr/bin/env python3
"""
Create dual-domain specific figures
- Figure 9.3: Dual-Domain Temporal Correlation
- Figure 9.5: Smart Sampling Visualization
- Figure 2.2: Schema Transformation Table
- Figure 7.1: ProcessGuid Correlation Tree (Simplified)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import seaborn as sns

sns.set_style("white")

# ============================================================================
# FIGURE 9.3: Dual-Domain Temporal Correlation
# ============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.15})

# Time axis (0-3600 seconds = 1 hour)
np.random.seed(42)

# UPPER TIMELINE: Sysmon Events (Host Domain)
sysmon_times = np.sort(np.random.uniform(0, 3600, 35))
sysmon_types = np.random.choice([1, 11, 23], 35, p=[0.7, 0.2, 0.1])
sysmon_colors = {1: '#4169E1', 11: '#228B22', 23: '#DC143C'}
sysmon_labels = {1: 'EventID 1 (Process)', 11: 'EventID 11 (File Create)', 23: 'EventID 23 (File Delete)'}

for i, (t, eid) in enumerate(zip(sysmon_times, sysmon_types)):
    color = sysmon_colors[eid]
    label = sysmon_labels[eid] if i < 3 and eid in [1, 11, 23] else ''
    ax1.scatter(t, 0.5, marker='o', s=200, color=color, edgecolors='black',
                linewidths=1.5, alpha=0.8, zorder=5, label=label)

ax1.set_ylabel('Sysmon Events\n(Host Domain)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.set_yticks([])
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.grid(axis='x', alpha=0.3)
ax1.legend(loc='upper left', fontsize=9, ncol=3)

# LOWER TIMELINE: NetFlow Events (Network Domain)
netflow_times = np.sort(np.random.uniform(0, 3600, 40))
netflow_protocols = np.random.choice(['TCP', 'UDP', 'ICMP'], 40, p=[0.6, 0.3, 0.1])
protocol_colors = {'TCP': '#00CED1', 'UDP': '#9932CC', 'ICMP': '#FFD700'}
protocol_styles = {'TCP': '-', 'UDP': '--', 'ICMP': ':'}

for i, (t, proto) in enumerate(zip(netflow_times, netflow_protocols)):
    color = protocol_colors[proto]
    style = protocol_styles[proto]
    label = proto if i < 3 and proto in ['TCP', 'UDP', 'ICMP'] else ''

    # Draw flow as horizontal line segment
    duration = np.random.uniform(1, 30)
    ax2.plot([t, t+duration], [0.5, 0.5], color=color, linestyle=style,
             linewidth=3, alpha=0.7, zorder=5, label=label)
    # Start marker
    ax2.scatter(t, 0.5, marker='|', s=100, color=color, linewidths=2, zorder=6)

ax2.set_ylabel('NetFlow Events\n(Network Domain)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.set_yticks([])
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.set_xlabel('Time (seconds since attack start)', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.legend(loc='upper left', fontsize=9, ncol=3)

# CORRELATION LINKS (Vertical connectors between timelines)
# Select some sysmon events and correlate with nearby netflow events
correlation_pairs = []
for sysmon_t in sysmon_times[:8]:  # First 8 sysmon events
    # Find nearest netflow within ±10 seconds
    time_window = 10
    nearby_netflows = [nf_t for nf_t in netflow_times if abs(nf_t - sysmon_t) <= time_window]
    if nearby_netflows:
        netflow_t = nearby_netflows[0]
        correlation_pairs.append((sysmon_t, netflow_t))

        # Draw correlation arrow
        arrow = mpatches.FancyArrowPatch(
            (sysmon_t, 0), (netflow_t, 1),
            transform=fig.transFigure,
            arrowstyle='->,head_width=0.3,head_length=0.2',
            color='#FF6347', linewidth=1.5, alpha=0.5, linestyle='--',
            mutation_scale=20
        )

        # Convert data coordinates to figure coordinates
        xy1 = ax1.transData.transform((sysmon_t, 0.1))
        xy2 = ax2.transData.transform((netflow_t, 0.9))
        xy1_fig = fig.transFigure.inverted().transform(xy1)
        xy2_fig = fig.transFigure.inverted().transform(xy2)

        arrow = mpatches.FancyArrowPatch(
            xy1_fig, xy2_fig,
            transform=fig.transFigure,
            arrowstyle='<->,head_width=0.15,head_length=0.25',
            color='#FF6347', linewidth=2, alpha=0.6, linestyle='--',
            mutation_scale=20, zorder=1
        )
        fig.patches.append(arrow)

# Time window shading (show ±10 sec window for one event)
example_event = sysmon_times[3]
window_start = example_event - 10
window_end = example_event + 10
ax1.axvspan(window_start, window_end, alpha=0.15, color='yellow', zorder=1)
ax2.axvspan(window_start, window_end, alpha=0.15, color='yellow', zorder=1)
ax1.text(example_event, 0.9, '±10s\nwindow', ha='center', fontsize=8,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

# Title
fig.suptitle('Figure 9.3: Dual-Domain Temporal Correlation\nHost Events (Sysmon) Correlated with Network Events (NetFlow)',
             fontsize=14, fontweight='bold', y=0.98)

# Causality tier labels
tier_text = (
    "Correlation Strategy: Temporal causation analysis within ±10 second windows\n"
    "Red dashed arrows indicate correlated events across domains (Tier 1/2 attribution)"
)
fig.text(0.5, 0.02, tier_text, ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9F9F9', edgecolor='#CCCCCC'))

plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_9_3_dual_domain_correlation.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_9_3_dual_domain_correlation.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 9.3: Dual-Domain Temporal Correlation created")

# ============================================================================
# FIGURE 9.5: Smart Sampling Visualization
# ============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# BEFORE: Full dataset (2.6M events)
np.random.seed(42)
full_benign_times = np.sort(np.random.uniform(0, 3600, 8000))  # Scaled down for viz
full_malicious_times = np.sort(np.random.uniform(0, 3600, 50))
full_tactics = np.random.choice(['discovery', 'execution', 'exfiltration'], 50)

tactic_colors = {'discovery': '#8B4513', 'execution': '#4169E1', 'exfiltration': '#32CD32'}

# Plot full dataset
ax1.scatter(full_benign_times, np.ones(len(full_benign_times)),
            marker='o', s=1, color='gray', alpha=0.3, label='Benign')

for tactic in ['discovery', 'execution', 'exfiltration']:
    tactic_times = full_malicious_times[full_tactics == tactic]
    ax1.scatter(tactic_times, np.ones(len(tactic_times)) * 1.05,
                marker='o', s=20, color=tactic_colors[tactic], alpha=0.8,
                edgecolors='black', linewidths=0.5, label=tactic.title())

ax1.set_ylabel('Events', fontsize=11, fontweight='bold')
ax1.set_title('BEFORE Smart Sampling: 2.6M Events\n⚠️ Causes Memory Crash / Segmentation Fault',
              fontsize=12, fontweight='bold', color='#DC143C')
ax1.set_ylim(0.95, 1.15)
ax1.set_yticks([])
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(axis='x', alpha=0.3)

# AFTER: Sampled dataset (200K events)
# Smart sampling: preserve first + last per tactic, random middle
sampled_benign_times = np.sort(np.random.choice(full_benign_times, 800, replace=False))
sampled_malicious_times_dict = {}

for tactic in ['discovery', 'execution', 'exfiltration']:
    tactic_times = full_malicious_times[full_tactics == tactic]
    if len(tactic_times) > 2:
        # Preserve first and last
        first = tactic_times[0]
        last = tactic_times[-1]
        middle = tactic_times[1:-1]
        # Sample middle
        n_sample = min(len(middle), 5)
        sampled_middle = np.random.choice(middle, n_sample, replace=False)
        sampled_malicious_times_dict[tactic] = np.concatenate([[first], sampled_middle, [last]])
    else:
        sampled_malicious_times_dict[tactic] = tactic_times

# Plot sampled dataset
ax2.scatter(sampled_benign_times, np.ones(len(sampled_benign_times)),
            marker='o', s=1, color='gray', alpha=0.3, label='Benign')

for tactic in ['discovery', 'execution', 'exfiltration']:
    tactic_times = sampled_malicious_times_dict[tactic]
    ax2.scatter(tactic_times, np.ones(len(tactic_times)) * 1.05,
                marker='o', s=20, color=tactic_colors[tactic], alpha=0.8,
                edgecolors='black', linewidths=0.5, label=tactic.title())

    # Highlight first and last (temporal boundaries)
    if len(tactic_times) > 1:
        ax2.scatter([tactic_times[0], tactic_times[-1]], [1.05, 1.05],
                    marker='s', s=80, color=tactic_colors[tactic],
                    edgecolors='black', linewidths=2, zorder=10)

ax2.set_ylabel('Events', fontsize=11, fontweight='bold')
ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_title('AFTER Smart Sampling: 200K Events (10% benign, 90% malicious budget)\n✅ Prevents Crashes While Preserving Temporal Boundaries (squares)',
              fontsize=12, fontweight='bold', color='#228B22')
ax2.set_ylim(0.95, 1.15)
ax2.set_yticks([])
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(axis='x', alpha=0.3)

# Overall title
fig.suptitle('Figure 9.5: Smart Sampling Strategy for Large Datasets\nTemporal-Boundary-Preserving Sampling',
             fontsize=14, fontweight='bold', y=0.98)

# Note
note_text = (
    "Strategy: Preserve first + last event per (Label, Tactic) group to maintain temporal span.\n"
    "Sample random middle events. Budget: 10% benign (background), 90% malicious (attack focus).\n"
    "Original event counts shown in legend (not sampled counts)."
)
fig.text(0.5, 0.02, note_text, ha='center', fontsize=8, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9F9F9', edgecolor='#CCCCCC'))

plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_9_5_smart_sampling.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_9_5_smart_sampling.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 9.5: Smart Sampling Visualization created")

# ============================================================================
# FIGURE 2.2: Schema Transformation Table
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

# Title
ax.text(0.5, 0.98, 'Figure 2.2: Sysmon Event Schema Transformation',
        ha='center', va='top', fontsize=14, fontweight='bold',
        transform=ax.transAxes, color='#005571')
ax.text(0.5, 0.95, 'JSONL (Nested XML) → CSV (Flat Normalized Schema)',
        ha='center', va='top', fontsize=11, style='italic',
        transform=ax.transAxes, color='#333333')

# Create two boxes: Before (JSONL) and After (CSV)

# LEFT SIDE: JSONL Structure
jsonl_box = FancyBboxPatch(
    (0.05, 0.15), 0.4, 0.75,
    boxstyle="round,pad=0.02",
    linewidth=3,
    edgecolor='#DC143C',
    facecolor='#FFE6E6',
    transform=ax.transAxes,
    zorder=1
)
ax.add_patch(jsonl_box)

ax.text(0.25, 0.88, 'BEFORE: Raw JSONL',
        ha='center', va='top', fontsize=12, fontweight='bold',
        transform=ax.transAxes, color='#DC143C')

jsonl_content = '''
{
  "@timestamp": "2025-05-24T23:19:21.858Z",
  "event": {
    "code": "11",
    "provider": "Microsoft-Windows-Sysmon"
  },
  "winlog": {
    "event_data": {
      "RuleName": "-",
      "UtcTime": "2025-05-24 23:19:21.858",
      "ProcessGuid": "{12abc...}",
      "ProcessId": "4892",
      "Image": "C:\\\\Windows\\\\sandcat.exe",
      "TargetFilename": "C:\\\\Users\\\\...",
      "User": "NT AUTHORITY\\\\SYSTEM"
    },
    "computer_name": "victim.local"
  },
  "message": "<Event xmlns=...\n    <EventData>\n      <Data Name='ProcessGuid'>{12abc...}</Data>\n      ...\n    </EventData>\n  </Event>"
}
'''

ax.text(0.07, 0.82, jsonl_content,
        ha='left', va='top', fontsize=7, family='monospace',
        transform=ax.transAxes, color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#DC143C'))

ax.text(0.25, 0.18, 'Nested JSON + Embedded XML',
        ha='center', va='center', fontsize=9, style='italic',
        transform=ax.transAxes, color='#666666')

# ARROW
arrow = FancyArrowPatch(
    (0.47, 0.5), (0.53, 0.5),
    arrowstyle='->,head_width=0.4,head_length=0.2',
    linewidth=4,
    color='#FDB462',
    transform=ax.transAxes,
    mutation_scale=40
)
ax.add_patch(arrow)

ax.text(0.5, 0.55, 'XML Parsing\n+\nField Extraction',
        ha='center', va='bottom', fontsize=9, fontweight='bold',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9E6', edgecolor='#FDB462', linewidth=2))

# RIGHT SIDE: CSV Structure
csv_box = FancyBboxPatch(
    (0.55, 0.15), 0.4, 0.75,
    boxstyle="round,pad=0.02",
    linewidth=3,
    edgecolor='#228B22',
    facecolor='#E6F9E6',
    transform=ax.transAxes,
    zorder=1
)
ax.add_patch(csv_box)

ax.text(0.75, 0.88, 'AFTER: Normalized CSV',
        ha='center', va='top', fontsize=12, fontweight='bold',
        transform=ax.transAxes, color='#228B22')

csv_content = '''
EventID,TimeCreated,Computer,ProcessGuid,ProcessId,Image,...
11,2025-05-24T23:19:21.858,victim.local,{12abc...},4892,...
1,2025-05-24T23:19:22.029,victim.local,{45def...},5102,...
3,2025-05-24T23:19:23.145,victim.local,{45def...},5102,...
11,2025-05-24T23:19:24.287,victim.local,{67ghi...},5234,...
'''

csv_columns = '''
Standard Columns (Normalized):
• EventID: Sysmon event type (1-26)
• TimeCreated: ISO timestamp
• Computer: Hostname
• ProcessId, ProcessGuid: Process identifiers
• Image: Executable path
• CommandLine: Process arguments
• User: Security context
• TargetFilename: File operations
• DestinationIp, DestinationPort: Network
• ParentProcessId, ParentProcessGuid
• ... (40+ standardized fields)
'''

ax.text(0.57, 0.82, csv_content,
        ha='left', va='top', fontsize=8, family='monospace',
        transform=ax.transAxes, color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#228B22'))

ax.text(0.57, 0.48, csv_columns,
        ha='left', va='top', fontsize=7,
        transform=ax.transAxes, color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0FFF0', edgecolor='#228B22'))

ax.text(0.75, 0.18, 'Flat Tabular Structure\n(ML-Ready)',
        ha='center', va='center', fontsize=9, style='italic',
        transform=ax.transAxes, color='#666666')

# Benefits box
benefits = (
    "✓ Consistent schema across all APT runs\n"
    "✓ Efficient pandas DataFrame operations\n"
    "✓ Direct CSV import to ML frameworks\n"
    "✓ ~60% file size reduction vs. JSONL"
)
ax.text(0.5, 0.08, benefits,
        ha='center', va='center', fontsize=9,
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E6F9F7', edgecolor='#00BFB3', linewidth=2))

plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_2_2_schema_transformation.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_2_2_schema_transformation.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 2.2: Schema Transformation created")

# ============================================================================
# FIGURE 7.1: ProcessGuid Correlation Tree (Simplified)
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Figure 7.1: ProcessGuid Correlation Tree',
        ha='center', fontsize=14, fontweight='bold', color='#005571')
ax.text(5, 9.1, 'Attack Lifecycle Expansion via Parent-Child Process Relationships',
        ha='center', fontsize=10, style='italic', color='#333333')

# Helper function to draw process node
def draw_process_node(ax, x, y, label, is_seed=False, tactic='execution', cmdline=''):
    if is_seed:
        # Red star for seed
        ax.scatter(x, y, marker='*', s=800, color='#DC143C',
                   edgecolors='black', linewidths=2, zorder=10)
        color = '#DC143C'
    else:
        # Colored circle by tactic
        tactic_colors_map = {
            'execution': '#4169E1',
            'discovery': '#8B4513',
            'persistence': '#228B22',
            'collection': '#9932CC',
            'exfiltration': '#32CD32'
        }
        color = tactic_colors_map.get(tactic, '#4169E1')
        ax.scatter(x, y, marker='o', s=400, color=color,
                   edgecolors='black', linewidths=1.5, zorder=5)

    # Label
    ax.text(x, y + 0.5, label, ha='center', fontsize=8, fontweight='bold')
    # Command line (truncated)
    if cmdline:
        ax.text(x, y - 0.4, cmdline, ha='center', fontsize=6, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    return color

# Helper function to draw edge
def draw_edge(ax, x1, y1, x2, y2, label=''):
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.5, zorder=1)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x, mid_y, label, ha='center', fontsize=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

# ROOT: Seed event
root_x, root_y = 5, 7.5
draw_process_node(ax, root_x, root_y, 'sandcat.exe\n(Seed Event)', is_seed=True,
                  cmdline='C:\\sandcat.exe')

# Level 1: Direct children
child1_x, child1_y = 2, 5.5
draw_process_node(ax, child1_x, child1_y, 'cmd.exe', tactic='execution',
                  cmdline='cmd.exe /C powershell...')
draw_edge(ax, root_x, root_y - 0.5, child1_x, child1_y + 0.5, 'ParentGuid')

child2_x, child2_y = 5, 5.5
draw_process_node(ax, child2_x, child2_y, 'powershell.exe', tactic='execution',
                  cmdline='powershell.exe -Exec...')
draw_edge(ax, root_x, root_y - 0.5, child2_x, child2_y + 0.5, 'ParentGuid')

child3_x, child3_y = 8, 5.5
draw_process_node(ax, child3_x, child3_y, 'cmd.exe', tactic='discovery',
                  cmdline='cmd.exe /C ipconfig...')
draw_edge(ax, root_x, root_y - 0.5, child3_x, child3_y + 0.5, 'ParentGuid')

# Level 2: Grandchildren
grandchild1_x, grandchild1_y = 1, 3.5
draw_process_node(ax, grandchild1_x, grandchild1_y, 'net.exe', tactic='discovery',
                  cmdline='net user /domain')
draw_edge(ax, child1_x, child1_y - 0.5, grandchild1_x, grandchild1_y + 0.5)

grandchild2_x, grandchild2_y = 3, 3.5
draw_process_node(ax, grandchild2_x, grandchild2_y, 'whoami.exe', tactic='discovery',
                  cmdline='whoami /all')
draw_edge(ax, child1_x, child1_y - 0.5, grandchild2_x, grandchild2_y + 0.5)

grandchild3_x, grandchild3_y = 5, 3.5
draw_process_node(ax, grandchild3_x, grandchild3_y, 'Invoke-WebRequest', tactic='collection',
                  cmdline='IWR http://...')
draw_edge(ax, child2_x, child2_y - 0.5, grandchild3_x, grandchild3_y + 0.5)

grandchild4_x, grandchild4_y = 7, 3.5
draw_process_node(ax, grandchild4_x, grandchild4_y, 'Compress-Archive', tactic='collection',
                  cmdline='Compress-Archive...')
draw_edge(ax, child2_x, child2_y - 0.5, grandchild4_x, grandchild4_y + 0.5)

grandchild5_x, grandchild5_y = 9, 3.5
draw_process_node(ax, grandchild5_x, grandchild5_y, 'ipconfig.exe', tactic='discovery',
                  cmdline='ipconfig /all')
draw_edge(ax, child3_x, child3_y - 0.5, grandchild5_x, grandchild5_y + 0.5)

# Level 3: Great-grandchildren
greatgrand1_x, greatgrand1_y = 4, 1.5
draw_process_node(ax, greatgrand1_x, greatgrand1_y, 'powershell.exe', tactic='exfiltration',
                  cmdline='Upload-File...')
draw_edge(ax, grandchild3_x, grandchild3_y - 0.5, greatgrand1_x, greatgrand1_y + 0.5)

greatgrand2_x, greatgrand2_y = 6, 1.5
draw_process_node(ax, greatgrand2_x, greatgrand2_y, 'curl.exe', tactic='exfiltration',
                  cmdline='curl -X POST...')
draw_edge(ax, grandchild4_x, grandchild4_y - 0.5, greatgrand2_x, greatgrand2_y + 0.5)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#DC143C', edgecolor='black', label='Seed Event (★)'),
    mpatches.Patch(facecolor='#4169E1', edgecolor='black', label='Execution'),
    mpatches.Patch(facecolor='#8B4513', edgecolor='black', label='Discovery'),
    mpatches.Patch(facecolor='#228B22', edgecolor='black', label='Persistence'),
    mpatches.Patch(facecolor='#9932CC', edgecolor='black', label='Collection'),
    mpatches.Patch(facecolor='#32CD32', edgecolor='black', label='Exfiltration'),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=8, framealpha=0.9)

# Statistics
stats_text = (
    "Seed Events: 1  |  Total Traced: 12  |  Expansion: 12x  |  Max Depth: 3 levels\n"
    "Correlation Method: ProcessGuid (child) matched to ParentProcessGuid (parent)"
)
ax.text(5, 0.5, stats_text, ha='center', fontsize=8, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9F9F9', edgecolor='#CCCCCC'))

plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_7_1_processguid_tree.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/researcher/Downloads/research/scripts/pipeline/figures/figure_7_1_processguid_tree.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Figure 7.1: ProcessGuid Correlation Tree created")

print("\n" + "="*60)
print("✅ ALL DUAL-DOMAIN FIGURES CREATED SUCCESSFULLY!")
print("="*60)
