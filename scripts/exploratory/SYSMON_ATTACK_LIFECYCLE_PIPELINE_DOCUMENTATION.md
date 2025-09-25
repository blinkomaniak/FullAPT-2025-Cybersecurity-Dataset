# Sysmon Attack Lifecycle Analysis Pipeline Documentation

## Overview

This document provides comprehensive documentation for the two-script pipeline that extracts seed events from Sysmon data and performs complete attack lifecycle analysis. The pipeline consists of:

1. **`5_sysmon_seed_event_extractor.py`** - Extracts and labels attack seed events from Sysmon datasets
2. **`6_sysmon_attack_lifecycle_tracer.py`** - Traces complete attack lifecycles from seed events across multiple EventIDs

## Pipeline Architecture

```
Raw Sysmon Dataset (sysmon-run-XX.csv)
                ↓
    [5_sysmon_seed_event_extractor.py]
                ↓
    Attack Seed Events (all_target_events_run-XX.csv)
                ↓
    [6_sysmon_attack_lifecycle_tracer.py]
                ↓
    Complete Attack Analysis Results
    ├── Timeline Visualizations (.png)
    ├── Labeled Sysmon Dataset (sysmon-run-XX-labeled.csv)
    ├── Traced Events (traced_sysmon_events_with_tactics.csv)
    └── Analysis Results (.json)
```

## Script 1: 5_sysmon_seed_event_extractor.py

### Purpose
Extracts and labels attack seed events (originators) from raw Sysmon datasets using configurable detection rules and MITRE ATT&CK framework integration.

### Key Capabilities
- **Multi-EventID Support**: Processes EventID 1 (Process Creation), EventID 11 (File Create), EventID 23 (File Delete)
- **MITRE ATT&CK Integration**: Maps detected events to tactics and techniques
- **Flexible Detection Rules**: Uses configurable patterns for command-line analysis, file operations, and process behaviors
- **IP-based Filtering**: Distinguishes local vs. remote-initiated attacks
- **Comprehensive Reporting**: Provides detailed statistics and event categorization

### Input Requirements
- **Primary**: `sysmon-run-XX.csv` - Raw Sysmon dataset
- **Optional**: Custom detection rules configuration

### Output Files
- **`all_target_events_run-XX.csv`** - Extracted seed events with MITRE ATT&CK labels
- **Analysis reports and statistics** (console output)

### Usage Example
```bash
python3 5_sysmon_seed_event_extractor.py \
    --apt-type apt-1 \
    --run-id 04 \
    --sysmon-csv ../../apt-1/apt-1-run-04/sysmon-run-04.csv \
    --output-dir ../../apt-1/apt-1-run-04/
```

### Detection Categories
1. **Process Creation Events (EventID 1)**:
   - Command-line pattern analysis
   - Suspicious process execution detection
   - Lateral movement indicators

2. **File Creation Events (EventID 11)**:
   - Malicious file drops
   - Payload staging detection
   - Configuration file creation

3. **File Deletion Events (EventID 23)**:
   - Evidence removal detection
   - Cleanup activity identification
   - Anti-forensics indicators

## Script 2: 6_sysmon_attack_lifecycle_tracer.py

### Purpose
Performs comprehensive attack lifecycle analysis by tracing all related events from seed events across multiple Sysmon EventIDs, providing complete attack progression visualization and analysis. **Now includes integrated labeling functionality for complete dataset preparation.**

### Key Capabilities
- **Multi-EventID Tracing**: Traces EventID 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 17, 18, 23
- **Process Tree Reconstruction**: Builds complete parent-child process relationships
- **Cross-Computer Analysis**: Traces lateral movement across multiple hosts
- **Advanced Correlation**: Links EventID 8/10 (ProcessAccess) to specific child processes
- **Integrated Labeling**: Automatically creates labeled datasets with MITRE Tactic/Technique columns
- **Comprehensive Visualization**: Generates multiple timeline plots and analysis views
- **Self-Contained Processing**: No external dependencies or re-runs required

### Input Requirements
- **Primary**: `sysmon-run-XX.csv` - Raw Sysmon dataset
- **Seed Events**: `all_target_events_run-XX.csv` - Output from seed extractor
- **Optional**: NetFlow dataset for enhanced correlation

### Output Files

#### 1. Timeline Visualizations
- **`timeline_all_malicious_events.png`** - Computer-grouped attack progression
- **`timeline_all_malicious_events_with_tactics.png`** - Complete Sysmon timeline with MITRE tactics highlighting
- **`eventidX_timeline_row_XXXXX.png`** - Individual attack event timelines (62 individual plots)

#### 2. Data Exports
- **`traced_sysmon_events_with_tactics.csv`** - All traced malicious events with MITRE labels
- **`sysmon-run-XX-labeled.csv`** - Complete Sysmon dataset with Tactic/Technique columns
- **`multi_eventid_analysis_results.json`** - Comprehensive analysis statistics

#### 3. Analysis Results
- **Attack progression statistics**
- **Event correlation metrics** 
- **Cross-computer activity mapping**
- **MITRE ATT&CK coverage analysis**

### Usage Example
```bash
python3 6_sysmon_attack_lifecycle_tracer.py \
    --apt-type apt-1 \
    --run-id 04 \
    --sysmon-csv ../../apt-1/apt-1-run-04/sysmon-run-04.csv \
    --originators-csv ../../apt-1/apt-1-run-04/all_target_events_run-04.csv \
    --output-dir ../../apt-1/apt-1-run-04/eventid1_analysis_results
```

### Core Analysis Features

#### 1. Process Tree Reconstruction
- **Parent-Child Mapping**: Links processes via ProcessGuid relationships
- **Recursive Tracing**: Follows complete process spawning chains
- **Cross-EventID Correlation**: Associates file operations with originating processes

#### 2. Event Correlation Logic
- **EventID 8/10 Attribution**: Advanced logic for ProcessAccess event attribution to correct child processes
- **Deduplication**: "Latest/most specific wins" approach for duplicate events
- **Timeline Reconstruction**: Chronological ordering of attack progression

#### 3. Multi-Computer Analysis
- **Lateral Movement Detection**: Tracks attack progression across hosts
- **Computer-based Grouping**: Organizes events by target computer
- **Network Activity Correlation**: Links host events with network flows (when NetFlow data available)

#### 4. MITRE ATT&CK Integration
- **Tactic Mapping**: Categorizes events by MITRE tactics (Initial Access, Execution, Discovery, etc.)
- **Technique Attribution**: Maps specific techniques (T1659, T1083, etc.)
- **Attack Phase Analysis**: Temporal analysis of tactic progression

## Pipeline Results Analysis

### Example Output Statistics (APT-1 Run-04)
- **Total Sysmon Events**: 363,657
- **Malicious Events Identified**: 3,939 (1.08%)
- **Attack Seed Events**: 62
- **Individual Timeline Plots**: 62
- **Computers Affected**: Multiple (theblock.boombox.local, etc.)

### MITRE Tactic Distribution
- **Discovery**: 1,829 events (largest category)
- **Initial-access**: 857 events  
- **Exfiltration**: 426 events
- **Credential-access**: 293 events
- **Execution**: 278 events
- **Command-and-control**: 125 events
- **Defense-evasion**: 129 events
- **Persistence**: 2 events

## Visualization Features

### 1. Computer-Grouped Timeline (`timeline_all_malicious_events.png`)
- Shows attack progression per computer
- Events sorted by activity level (top-to-bottom)
- Color-coded by EventID type
- Temporal analysis of attack phases

### 2. Complete Context Timeline (`timeline_all_malicious_events_with_tactics.png`)
- **Background**: All 360K+ benign events (pale gray)
- **Foreground**: Malicious events color-coded by MITRE tactics
- **Complete situational awareness** of attack vs. normal activity
- **Proper temporal distribution** across attack timeframe

### 3. Individual Attack Timelines
- Detailed view of each seed event's progression
- Process tree visualization
- Cross-EventID correlation display
- Computer-specific attack analysis

## Key Technical Innovations

### 1. Advanced EventID 8/10 Correlation
Fixed critical attribution bug where ProcessAccess events were incorrectly attributed to parent processes instead of specific child processes that spawned them.

### 2. Comprehensive Event Deduplication
Implements sophisticated deduplication logic ensuring accurate event counting and avoiding false inflation of attack metrics.

### 3. Multi-Domain Analysis Ready
Pipeline designed to integrate with NetFlow correlation for complete dual-domain (host + network) attack analysis.

### 4. Production-Scale Processing
Handles large Sysmon datasets (300K+ events) efficiently with robust error handling and progress tracking.

## Integration with Broader Research

### Dual-Domain Dataset Development
This pipeline contributes to the development of a state-of-the-art cybersecurity dataset emphasizing:
- **Host-level events** (Sysmon) combined with **network flow events**
- **Complete attack patterns** across both domains
- **MITRE ATT&CK framework** integration for standardized analysis

### Machine Learning Pipeline Ready
Outputs structured, labeled datasets suitable for:
- **Anomaly detection** model training
- **Attack classification** algorithms
- **Behavioral analysis** research
- **Temporal pattern recognition** studies

## Future Enhancements

### Planned Improvements
1. **NetFlow Integration**: Enhanced dual-domain correlation with network flow data
2. **Automated Tactic Detection**: ML-based automatic MITRE tactic assignment
3. **Real-time Processing**: Streaming analysis capabilities for live monitoring
4. **Advanced Visualization**: Interactive timeline plots and network graphs

### Research Applications
- **APT Behavior Modeling**: Comprehensive attack pattern analysis
- **Defense Evaluation**: Testing detection system effectiveness
- **Threat Intelligence**: Attack technique frequency and progression analysis
- **Incident Response**: Rapid attack reconstruction and impact assessment

## Troubleshooting

### Common Issues
1. **Timestamp Parsing**: Ensures Unix millisecond timestamps are correctly converted
2. **Memory Management**: Handles large datasets efficiently
3. **File Path Dependencies**: Robust handling of missing input files
4. **Cross-Platform Compatibility**: Windows path handling in attack data

### Performance Considerations
- **Processing Time**: ~2-3 minutes per dataset (APT run)
- **Memory Usage**: Optimized for datasets up to 400K events
- **Output Size**: Complete analysis generates ~50-100MB results per run

## Conclusion

This two-script pipeline provides comprehensive, production-ready attack lifecycle analysis from raw Sysmon data. It combines sophisticated event correlation, MITRE ATT&CK integration, and advanced visualization capabilities to deliver complete attack progression analysis suitable for cybersecurity research, threat hunting, and incident response activities.

The pipeline represents a significant advancement in host-based attack analysis, providing the foundation for dual-domain cybersecurity dataset development and advanced threat detection research.