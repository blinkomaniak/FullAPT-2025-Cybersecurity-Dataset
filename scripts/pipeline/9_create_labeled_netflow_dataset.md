# Pipeline Step 9: Create Labeled NetFlow Dataset

## Overview
**Purpose**: Generate comprehensive labeled NetFlow dataset by correlating network traffic events with labeled Sysmon seed events using temporal causation logic, IP scope filtering, and refined attribution strategies. Produces verification matrices for dual-domain attack behavior analysis and timeline visualizations for validation.

**Position in Pipeline**: Ninth and final step - Network-level dataset labeling and dual-domain correlation

## Functionality

### Core Capabilities
- **Interactive Configuration**: Wizard-based setup for IPs, network scope, and correlation parameters
- **Dual-Domain Correlation**: Links network flows with host-based Sysmon events
- **Temporal Causation**: Configurable correlation time window (default: 10 seconds)
- **Three-Tier Labeling System**: NetFlow attribution, Sub-NetFlow attribution, Seed Event attribution
- **IP Scope Filtering**: Restricted/Unrestricted modes with whitelist/blacklist support
- **Community ID Aggregation**: Groups related flows using network community IDs
- **Multi-Track Timeline Visualization**: Network events by MITRE ATT&CK tactic
- **Dual-Domain Timeline**: Combined Sysmon + NetFlow attack visualization
- **Smart Sampling**: Automatic downsampling for large datasets (>200K events) to prevent memory issues

### Correlation Strategy

**Temporal Window**:
- Default: ±10 seconds correlation window
- Configurable: User-defined via interactive wizard
- Causation types: `possible_cause`, `possible_effect`, `simultaneous`

![Figure 9.2: Temporal Causation Window](figures/figure_9_2_temporal_window.png)
**Figure 9.2**: Timeline diagram illustrating the temporal causation correlation window. A Sysmon seed event (red vertical line) establishes a ±10 second correlation window (shaded region). NetFlow events are classified as 'possible_cause' (blue, ending before seed event), 'possible_effect' (green, starting after seed event), or 'simultaneous' (purple, overlapping with seed event). Events outside the window (gray) are not correlated. The configurable window size balances precision (tight window) versus recall (loose window).

**Computer Matching**:
- Requires seed_event.Computer ∈ netflow.host_hostname
- Case-insensitive hostname matching
- Supports aggregated NetFlow hostname arrays

**Three-Tier Attribution**:
1. **Tier 0 - Attacker IP Baseline**: Flow involves attacker IP (always x-marked, bypasses all filtering)
2. **Tier 1 - Direct NetFlow Attribution**: Temporal + computer correlation with seed events
3. **Tier 2 - Sub-NetFlow Attribution**: Sub-flow correlation using community IDs

![Figure 9.1: Three-Tier NetFlow Labeling System](figures/figure_9_1_three_tier_labeling.png)
**Figure 9.1**: Hierarchical diagram showing the three-tier NetFlow labeling system. Tier 0 (top, red) automatically marks all flows involving the attacker IP, bypassing all filters. Tier 1 (middle, orange) uses temporal causation and computer matching to correlate flows with Sysmon seed events within the correlation window. Tier 2 (bottom, yellow) propagates labels to related sub-flows using network community IDs. The pyramid structure indicates decreasing confidence from Tier 0 (highest) to Tier 2 (lowest), with flow counts increasing at each tier.

## Usage

### Prerequisites
**Required Dependencies**:
- Python 3.7+
- pandas, numpy, matplotlib, seaborn, pyyaml libraries
- Completed Step 8 output (sysmon-run-XX-labeled.csv)
- Aggregated NetFlow data (from Step 4)
- APT YAML attack plan (for technique mapping)

### Execution Location
```bash
# From pipeline directory
cd /home/researcher/Downloads/research/scripts/pipeline/
python3 9_create_labeled_netflow_dataset.py --apt-type apt-1 --run-id 04

# Using wizard mode for all configurations
python3 9_create_labeled_netflow_dataset.py --apt-type apt-4 --run-id 43
```

### Command Line Options
```bash
# APT dataset mode with interactive wizard (recommended)
python3 9_create_labeled_netflow_dataset.py --apt-type apt-1 --run-id 04

# Pre-configure IPs (skip wizard)
python3 9_create_labeled_netflow_dataset.py --apt-type apt-1 --run-id 04 \
    --attacker-ip 192.168.0.4 \
    --internal-network 10.1.0.0/24 \
    --scope-mode restricted

# Custom correlation window
python3 9_create_labeled_netflow_dataset.py --apt-type apt-1 --run-id 04 \
    --correlation-window 15

# Enable TCP filtering for domain controller persistence
python3 9_create_labeled_netflow_dataset.py --apt-type apt-1 --run-id 04 \
    --filter-dc-tcp

# Enable debug logging
python3 9_create_labeled_netflow_dataset.py --apt-type apt-1 --run-id 04 --debug
```

## Interactive Configuration Wizard

### Step 1: Correlation Window Configuration
```
📊 STEP 1: Correlation Time Window
Set the temporal correlation window (in seconds).

Examples:
  - Tight correlation (5s): High precision, may miss delayed effects
  - Standard correlation (10s): Balanced precision/recall (recommended)
  - Loose correlation (20s): High recall, may include unrelated events

Enter correlation window in seconds [default: 10]: 10
✅ Correlation window set to: 10 seconds
```

### Step 2: Attacker IP Configuration
```
📊 STEP 2: Attacker IP Configuration
Enter the external attacker IP address.

Examples:
  - APT-1 to APT-6: 192.168.0.4
  - Custom: User-specified IP

Enter attacker IP address [default: 192.168.0.4]: 192.168.0.4
✅ Attacker IP set to: 192.168.0.4
```

### Step 3: IP Scope Mode Selection
```
📊 STEP 3: IP Scope Mode

[1] RESTRICTED MODE (Recommended)
    → Only internal network + attacker IP + whitelisted IPs appear in CSV
    → Focuses analysis on in-scope attack infrastructure
    → Reduces false positives from external services

[2] UNRESTRICTED MODE
    → All IPs appear in CSV except excluded IPs
    → Includes external infrastructure (DNS, CDNs, cloud services)
    → Useful for discovering unknown external command & control servers

Select mode [1-2, default: 1]: 1
✅ Scope mode set to: RESTRICTED
```

### Step 4: Internal Network Definition (Restricted Mode)
```
📊 STEP 4: Internal Network Range

Define the internal network range (CIDR notation).

Examples:
  - Single subnet: 10.1.0.0/24 (hosts: 10.1.0.1 - 10.1.0.254)
  - Larger network: 10.0.0.0/16 (hosts: 10.0.0.1 - 10.0.255.254)

Enter internal network CIDR [default: 10.1.0.0/24]: 10.1.0.0/24
✅ Internal network set to: 10.1.0.0/24 (256 hosts)
```

### Step 5: External IP Whitelist (Restricted Mode)
```
📊 STEP 5: External IP Whitelist

Whitelist specific external IPs to include in restricted mode.

Examples:
  - Known C2 server: 203.0.113.45
  - Cloud infrastructure: 198.51.100.10
  - Partner network: 192.0.2.5

Enter external IPs (comma-separated) [default: none]: 203.0.113.45, 198.51.100.10
✅ External whitelist: ['203.0.113.45', '198.51.100.10']
```

### Step 6: TCP Persistence Filtering
```
📊 STEP 6: TCP Persistence Filtering

Configure filtering for persistent TCP connections involving Domain Controller:

⚡ NOTE: Flows involving the attacker IP (192.168.0.4) are ALWAYS x-marked
         regardless of protocol, duration, or DC involvement.

[1] FILTER DC-INVOLVED TCP (Default - Recommended)
    → Filter persistent TCP (>20s) involving DC (10.1.0.4) as source/destination
    → Includes bidirectional flows (e.g., [10.1.0.4, 10.1.0.5] ↔ [10.1.0.4, 10.1.0.5])
    → Preserves flows with attacker IP involvement (auto-whitelisted)
    → Reduces false positives from domain authentication traffic

[2] NO TCP FILTERING
    → All TCP flows eligible for x-marking (no filtering)
    → May include persistent DC authentication traffic
    → Use if DC traffic is relevant to attack analysis

Select option [1-2, default: 1]: 1
✅ TCP filtering configured: FILTER DC-INVOLVED TCP
```

## Input Requirements

### Directory Structure
```
dataset/
├── apt-1/
│   ├── apt-1-run-04/
│   │   ├── sysmon-run-04-labeled.csv                # Labeled Sysmon (Step 8)
│   │   ├── netflow-run-04.csv                        # Raw NetFlow (Step 3)
│   │   └── netflow_event_tracing_analysis_results/  # Aggregated flows (Step 4)
│   │       ├── subnetflow_assignment_template_run-04.csv
│   │       ├── [IP_pair]__[protocol]/
│   │       │   ├── community_X_[hash]_events.csv
│   │       │   └── community_X_[hash]_metadata.csv
│   │       └── subnetflow_analysis/
│   │           ├── community_X_[hash]_subnetflows.csv
│   │           └── community_X_[hash]_metadata.csv
│   └── ...
└── [apt-2 through apt-7]/

scripts/exploratory/
└── apt-yaml/
    └── [APT_name].yaml                               # Attack plan for technique mapping
```

### Input Format
**Labeled Sysmon CSV** (sysmon-run-XX-labeled.csv):
- Binary labels: benign/malicious
- MITRE ATT&CK tactic and technique annotations
- Seed_RowNumber for attribution tracking
- timestamp, Computer, EventID columns required

**Aggregated NetFlow** (community events and metadata):
- Community ID groupings
- Sub-flow analysis results
- Temporal flow metadata
- Process attribution (PID, ProcessGuid)

**APT YAML** (attack plan):
- Command patterns
- Tactic and technique mappings
- Used for filling missing technique IDs

## Output Generated

### Primary Outputs
```
📊 VERIFICATION MATRIX OUTPUTS:
├── verification_matrix_run-XX.csv            # Original matrix (43 columns)
├── verification_matrix_v2_run-XX.csv         # Reordered matrix (43 columns)
├── netflow-run-XX-labeled.csv                # Labeled NetFlow dataset
├── netflow_event_tracing_analysis_results/
│   ├── multi_track_timeline_run-XX.png       # Network events by tactic
│   └── dual_domain_attack_timeline.png       # Combined Sysmon + NetFlow
└── config_summary.yaml                        # Configuration record
```

### Verification Matrix Schema (43 columns)
```csv
# Column Order (v2):
nci,                       # NetFlow community ID
sern,                      # Seed event row number
netflow_attribution,       # NetFlow attribution result (x or none)
subnetflow_attribution,    # Sub-NetFlow attribution result (x or none)
setac,                     # Seed event tactic
setech,                    # Seed event technique
nfw_se_attrib,             # Unified NetFlow-SeedEvent attribution (x or none)
causality_type,            # Temporal causation type (cause/effect/simultaneous/none)
diff_set_nst,              # Time difference: seed_timestamp - netflow_start
diff_set_net,              # Time difference: seed_timestamp - netflow_end
netflow_duration,          # NetFlow duration: end - start
nst, net, nts,             # NetFlow start, end, timestamps
nsip, nsp, ndip, ndp,      # NetFlow source/dest IPs and ports
ntr, nby, npack,           # NetFlow transport, bytes, packets
nhhost,                    # NetFlow hostname (computer)
npe, npid, npn, nparg,     # NetFlow process executable, PID, name, arguments
ndpe, ndpid, ndpn, ndparg, # NetFlow dest process info
set, seid, secomp,         # Seed event timestamp, EventID, computer
secline, setf, sepguid,    # Seed event command line, target filename, ProcessGuid
sepid, seppguid, seppid,   # Seed event ProcessId, ParentProcessGuid, ParentProcessId
seim, sepim,               # Seed event Image, ParentImage
diff_set_nst_max,          # Max time difference (multi-seed)
diff_set_net_max           # Max time difference (multi-seed)
```

### Labeled NetFlow Dataset
```csv
Label,Tactic,Technique,community_id,timestamp,source_ip,dest_ip,protocol,...
malicious,discovery,T1016,abc123...,1748128969443,10.1.0.5,8.8.8.8,udp,...
benign,,,def456...,1748128970123,10.1.0.4,10.1.0.5,tcp,...
malicious,exfiltration,T1041,ghi789...,1748129089316,10.1.0.5,192.168.0.4,tcp,...
```

## Filtering Logic

The filtering system applies a hierarchical priority scheme to reduce false positives while preserving all attacker-related network activity.

![Figure 9.4: IP Scope Filtering Logic](figures/figure_9_4_ip_filtering.png)
**Figure 9.4**: Flowchart showing the priority-based filtering logic for NetFlow events. The algorithm first checks for attacker IP involvement (Priority 1 - auto-whitelist, bypasses all filters). If not present, it sequentially applies ICMP filtering (Priority 2), UDP persistence filtering (Priority 3), and optional TCP/DC filtering (Priority 4). Each filter can mark events as 'none' (filtered out) or allow them to proceed to x-marking. This hierarchical approach eliminates benign persistent connections while preserving attack-related flows.

### Priority 1: Attacker IP Whitelist
**Highest Priority** - Bypasses ALL other filtering rules
```python
if attacker_ip in (source_ips or dest_ips):
    nfw_se_attrib = 'x'        # ALWAYS x-marked
    causality_type = 'x'       # ALWAYS x-marked
    # Skip all filtering below
```

### Priority 2: ICMP Filtering
```python
if protocol == 'icmp' and attacker_ip NOT in (source_ips or dest_ips):
    nfw_se_attrib = 'none'
    causality_type = 'none'
```

### Priority 3: UDP Persistence Filtering
```python
if protocol == 'udp' and attacker_ip NOT in (source_ips or dest_ips):
    if duration > (2 * correlation_window_seconds * 1000):
        nfw_se_attrib = 'none'
        causality_type = 'none'
```

### Priority 4: TCP Persistence Filtering (Optional)
```python
if protocol == 'tcp' and filter_dc_tcp and attacker_ip NOT in (source_ips or dest_ips):
    if duration > (2 * correlation_window_seconds * 1000):
        if dc_ip in (source_ips or dest_ips):
            nfw_se_attrib = 'none'
            causality_type = 'none'
```

## Causality Types

### Temporal Relationship Classification

The correlation engine classifies temporal relationships between Sysmon seed events and NetFlow events to distinguish causation direction and identify simultaneous activities.

![Figure 9.6: Causality Types Diagram](figures/figure_9_6_causality_types.png)
**Figure 9.6**: Temporal relationship classification diagram showing the three causality types. The diagram uses overlapping timeline bars to illustrate: 'possible_cause' (NetFlow ends before seed event starts, suggesting network activity triggered host event), 'possible_effect' (NetFlow starts after seed event ends, suggesting host event triggered network activity), and 'simultaneous' (timelines overlap, indicating concurrent activity). Each type has different analytical implications for understanding attack execution sequences.
**possible_cause** (NetFlow → Seed Event):
- NetFlow END timestamp < Seed Event timestamp
- Network activity likely caused or triggered the host event
- Example: C2 callback triggers command execution

**possible_effect** (Seed Event → NetFlow):
- Seed Event timestamp < NetFlow START timestamp
- Host event likely caused the network activity
- Example: Command execution triggers data exfiltration

**simultaneous** (Overlapping):
- Events overlap in time
- Concurrent host and network activity
- Example: Process execution during active network connection

## Timeline Visualizations

### Multi-Track Timeline (NetFlow by Tactic)

The multi-track timeline separates NetFlow events by MITRE ATT&CK tactic, providing a clear view of attack phase progression over time.

![Figure 9.7: Multi-Track Timeline Visualization](figures/figure_9_7_multitrack_timeline.png)
**Figure 9.7**: Multi-track timeline showing NetFlow events separated by MITRE ATT&CK tactic. Each horizontal track represents a different tactic (initial-access, execution, discovery, collection, exfiltration, etc.) with events plotted chronologically. Benign background traffic appears in the bottom track. Colors match the MITRE ATT&CK framework palette for consistency. The legend shows original event counts before sampling. This visualization reveals temporal attack patterns, such as discovery preceding collection, and collection preceding exfiltration.

**Features**:
- Individual tracks for each MITRE ATT&CK tactic
- Benign background traffic visualization
- Color-coded by tactic
- Temporal attack phase progression
- Legend with event counts (original dataset counts, not sampled)

**Smart Sampling** (for datasets >200K events):
- Automatically reduces to 200K events for plotting
- Preserves first and last event per (Label, Tactic) group
- Allocates 10% to benign, 90% to malicious
- Title indicates sampling: "(Sampled: 200,000 events shown from 2,641,388 total)"
- Legend shows ORIGINAL counts (not sampled counts)

![Figure 9.5: Smart Sampling Strategy](figures/figure_9_5_smart_sampling.png)
**Figure 9.5**: Flowchart illustrating the smart sampling algorithm for large NetFlow datasets (>200K events). The process groups events by (Label, Tactic), preserves temporal boundaries (first and last events in each group to maintain timeline span), allocates 90% of sample budget to malicious events and 10% to benign, randomly samples middle events proportionally, and combines all samples into a 200K-event visualization dataset. This strategy prevents memory issues while preserving attack patterns and temporal characteristics.

### Dual-Domain Timeline (Sysmon + NetFlow)

The dual-domain timeline combines Sysmon host events and NetFlow network events in a synchronized visualization, enabling comprehensive attack behavior analysis across both domains.

![Figure 9.3: Dual-Domain Attack Timeline](figures/figure_9_3_dual_domain_timeline.png)
**Figure 9.3**: Dual-domain timeline with Sysmon events (top panel, grouped by computer hostname) and NetFlow events (bottom panel, separated by tactic) sharing a unified time axis. Vertical alignment reveals temporal correlations between host and network activities. For example, a Sysmon discovery event (top) temporally aligns with NetFlow discovery traffic (bottom), validating the correlation logic. This visualization is essential for understanding how host-level attack operations manifest as network communications.

**Features**:
- **Top Panel**: Sysmon malicious events grouped by computer
- **Bottom Panel**: NetFlow events by tactic
- Unified time axis for temporal correlation analysis
- Validates dual-domain correlation quality
- Shows relationship between host and network activity

## Integration with Pipeline

### Input Dependencies
**Step 8 Output**: Labeled Sysmon dataset (sysmon-run-XX-labeled.csv)
**Step 3 Output**: Raw NetFlow (netflow-run-XX.csv)
**Step 4 Output**: Aggregated community flows

### Output Integration
**Research Applications**: Publication-ready verification matrices and timelines
**Machine Learning**: Labeled NetFlow dataset for network-based detection
**Further Analysis**: CSV exports for statistical analysis

**Data Flow**:
```
Labeled Sysmon + Aggregated NetFlow → Correlation → Verification Matrix + Labeled NetFlow
   (Step 8)           (Step 4)           (Step 9)            ↓
                                                       ML Training
                                                       Threat Intelligence
                                                       Detection Engineering
```

## Performance Characteristics

### Processing Metrics
- **Runtime**: 5-20 minutes per APT run (depends on flow count and correlation window)
- **Memory Usage**: 2-8GB (scales with NetFlow event count)
- **Correlation Pairs**: 1M-10M potential pairs evaluated
- **Smart Sampling Trigger**: Automatically activates for >200K NetFlow events
- **Visualization Generation**: 30-90 seconds per timeline plot

### Scalability
- **Small Datasets**: 10K-50K flows → ~100-500 correlated events
- **Medium Datasets**: 50K-200K flows → ~500-2000 correlated events
- **Large Datasets**: 200K-3M flows → ~2K-20K correlated events (with smart sampling)
- **Correlation Window Impact**: Larger windows = more pairs = longer runtime

## Quality Assurance

### Validation Features
- **IP Range Validation**: Validates CIDR notation and IP format
- **Timestamp Consistency**: Ensures causality logic temporal coherence
- **Column Reordering**: Verification matrix v2 has analyst-friendly column order
- **Attribution Consistency**: Validates three-tier labeling logic
- **Sampling Transparency**: Clear indication when sampling is applied

### Error Handling
- **Missing Seed Events**: Gracefully handles datasets with no malicious events
- **Empty Community IDs**: Skips invalid or empty community groupings
- **Invalid Timestamps**: Filters out negative or corrupt timestamps
- **Memory Overflow**: Smart sampling prevents segmentation faults on large datasets

## Research Applications

### Dual-Domain Analysis
- **Host-Network Correlation**: Quantify relationship between Sysmon and NetFlow events
- **Attack Kill Chain Mapping**: Map network indicators to attack phases
- **C2 Channel Identification**: Discover command and control communication patterns
- **Data Exfiltration Analysis**: Correlate file collection with network transfers

### Network Threat Intelligence
- **Behavioral Signatures**: Network patterns per ATT&CK tactic
- **Port Usage Analysis**: Protocol and port preferences by adversary
- **Traffic Volume Patterns**: Bytes/packets distribution for malicious flows
- **Temporal Patterns**: Attack timing and duration characteristics

### Detection Engineering
- **Network Signature Development**: Create IDS/IPS rules from labeled flows
- **Baseline Comparison**: Compare attack traffic vs benign baselines
- **False Positive Reduction**: Use IP scope filtering to reduce noise
- **Rule Validation**: Test detection rules against ground truth labeled data

## Troubleshooting

### Common Issues
**Low Correlation Rate**:
- Increase correlation window (e.g., from 10s to 20s)
- Review IP scope configuration (ensure internal network includes all hosts)
- Check that attacker IP is correctly identified
- Verify Sysmon and NetFlow timestamps are in same time zone

**High False Positive Rate**:
- Enable TCP filtering for domain controller traffic
- Use RESTRICTED scope mode to focus on in-scope IPs
- Reduce correlation window for tighter temporal matching
- Review ICMP and UDP filtering rules

**Memory Issues on Large Datasets**:
- Smart sampling automatically activates for >200K events
- If still encountering issues, process smaller time windows
- Close other applications to free memory
- Consider reducing correlation window to decrease pair count

**Visualization Segmentation Faults**:
- Smart sampling now prevents this (reduced from 1M to 200K threshold)
- If issues persist, manually set MAX_EVENTS_FOR_VISUALIZATION lower in code
- Check matplotlib backend configuration

### Debug Mode
Enable detailed logging to troubleshoot issues:
```bash
python3 9_create_labeled_netflow_dataset.py --apt-type apt-1 --run-id 04 --debug
```

---
*This NetFlow labeling pipeline provides comprehensive dual-domain attack analysis, correlating network and host events to produce ML-ready labeled datasets with refined causality logic, IP scope filtering, and robust visualization for attack campaign characterization and detection engineering.*
