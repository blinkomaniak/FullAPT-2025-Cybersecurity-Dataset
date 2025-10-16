# Sysmon Attack Lifecycle Analysis Pipeline Documentation

## Overview

This document provides comprehensive documentation for the three-script pipeline that extracts seed events from Sysmon data and performs complete attack lifecycle analysis with manual curation capabilities. The pipeline consists of:

1. **`5_sysmon_seed_event_extractor.py`** - Simple extractor for manual seed event selection from Sysmon datasets
2. **`6_sysmon_attack_lifecycle_tracer.py`** - Traces complete attack lifecycles from seed events across multiple EventIDs
3. **`7_create_labeled_sysmon_dataset.py`** - Creates final labeled datasets with manual correction capabilities

## Pipeline Architecture

```
Raw Sysmon Dataset (sysmon-run-XX.csv)
                ↓
    [5_sysmon_seed_event_extractor.py]
                ↓
    Attack Seed Events (all_target_events_run-XX.csv)
    │   • Manual selection via Seed_Event, Tactic, Technique columns
    │   • Human-readable timestamps (timestamp_h) + original timestamps
                ↓
    [6_sysmon_attack_lifecycle_tracer.py]
                ↓
    Traced Attack Events (traced_sysmon_events_with_tactics.csv)
                ↓
    [Manual Copy & Edit Step]
    │   • Copy to traced_sysmon_events_with_tactics_v2.csv
    │   • Manual corrections in Correct_SeedRowNumber column
                ↓
    [7_create_labeled_sysmon_dataset.py]
                ↓
    Final Labeled Dataset
    ├── Timeline Visualizations (.png)
    ├── Complete Labeled Sysmon Dataset (sysmon-run-XX-labeled.csv)
    ├── Corrected Events (traced_sysmon_events_with_tactics_v2.csv)
    └── Analysis Results (.json)
```

## Script 1: 5_sysmon_seed_event_extractor.py

### Purpose
Simple extractor that filters target EventIDs from raw Sysmon datasets and provides manual selection framework for identifying attack seed events. **No automatic detection** - relies on human expertise for accurate attack identification.

### Key Capabilities
- **Multi-EventID Support**: Extracts EventID 1 (Process Creation), EventID 11 (File Create), EventID 23 (File Delete)
- **Manual Selection Framework**: Provides structured columns for manual attack event identification
- **Dual Timestamp Format**: Human-readable timestamps for analysis + preserved Unix milliseconds for processing
- **Row Traceability**: Maintains `RawDatasetRowNumber` for complete audit trail
- **Selection Preservation**: Preserves existing manual selections when re-running
- **Chronological Ordering**: Events sorted by timestamp for attack timeline analysis

### Input Requirements
- **Primary**: `sysmon-run-XX.csv` - Raw Sysmon dataset

### Output Files
- **`all_target_events_run-XX.csv`** - Filtered events with manual selection columns

### Column Structure (Output)
```
Seed_Event, Tactic, Technique, RawDatasetRowNumber, timestamp_h, EventID, Computer,
CommandLine, TargetFilename, ProcessGuid, ProcessId, ParentProcessGuid, ParentProcessId,
Image, ParentImage, timestamp [remaining columns...]
```

### Usage Example
```bash
python3 5_sysmon_seed_event_extractor.py \
    --apt-type apt-1 \
    --run-id 04
```

### Manual Selection Workflow
1. **Run Script**: Extracts all EventID 1, 11, 23 events to CSV
2. **Manual Review**: Open CSV in Excel/LibreOffice for human analysis
3. **Mark Events**:
   - `Seed_Event` column: Mark 'X' for significant attack events
   - `Tactic` column: Enter MITRE ATT&CK tactic (e.g., 'Discovery', 'Execution')
   - `Technique` column: Enter MITRE ATT&CK technique ID (e.g., 'T1083', 'T1059')
4. **Save & Re-run**: Script preserves selections when re-executed

### Key Features
- **Human-Readable Timestamps**: `timestamp_h` column shows `YYYY-MM-DD HH:MM:SS.mmm` format
- **Original Timestamps Preserved**: `timestamp` column maintains Unix milliseconds for downstream processing
- **Invalid Timestamp Handling**: Negative/invalid timestamps marked as `INVALID_TIMESTAMP`
- **Selection Statistics**: Reports count of marked events by type

## Script 2: 6_sysmon_attack_lifecycle_tracer.py

### Purpose
Performs comprehensive attack lifecycle analysis by tracing all related events from manually-selected seed events across multiple Sysmon EventIDs. Produces intermediate traced events file that requires manual correction before final dataset creation.

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
- **Seed Events**: `all_target_events_run-XX.csv` - Output from seed extractor with manual selections

### Output Files

#### Primary Output
- **`traced_sysmon_events_with_tactics.csv`** - All traced events ready for manual correction

#### Column Structure (traced_sysmon_events_with_tactics.csv)
```
Tactic, Technique, OriginatorRow, Correct_SeedRowNumber, EventID, Computer, timestamp_h,
CommandLine, TargetFilename, ParentCommandLine, ProcessGuid, ParentProcessGuid,
ProcessId, ParentProcessId, timestamp [remaining columns...]
```

#### Key Column Descriptions
- **`OriginatorRow`**: References `RawDatasetRowNumber` from original seed selection
- **`Correct_SeedRowNumber`**: Empty column for manual corrections in v2.csv workflow
- **`timestamp_h`**: Human-readable timestamps for analysis
- **`timestamp`**: Original Unix milliseconds (preserved for Script #7)

#### Timeline Visualizations
- **`timeline_all_malicious_events.png`** - Computer-grouped attack progression
- **`timeline_all_malicious_events_with_tactics.png`** - Complete Sysmon timeline with MITRE tactics highlighting
- **Individual timeline plots** - Per-originator attack progression analysis

#### Analysis Files
- **`multi_eventid_analysis_results.json`** - Comprehensive tracing statistics

### Manual Correction Workflow
1. **Copy Output**: `cp traced_sysmon_events_with_tactics.csv traced_sysmon_events_with_tactics_v2.csv`
2. **Manual Review**: Open v2.csv for human verification of event attribution
3. **Correct Attribution**: Fill `Correct_SeedRowNumber` column where automatic tracing attribution is incorrect
4. **Proceed to Script #7**: Use corrected v2.csv as input for final dataset creation

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

## Script 3: 7_create_labeled_sysmon_dataset.py

### Purpose
Creates the final labeled Sysmon dataset by applying manual corrections from the v2.csv file and generating a complete labeled dataset suitable for machine learning and analysis.

### Key Capabilities
- **Manual Correction Integration**: Applies human-verified corrections from `Correct_SeedRowNumber` column
- **Master Tactics Lookup**: Cross-references with original seed selection file for accurate MITRE labeling
- **Complete Dataset Labeling**: Labels entire Sysmon dataset with attack/benign classifications
- **Timeline Visualization**: Creates final timeline plots with corrected attributions
- **Audit Trail Preservation**: Maintains complete traceability from raw data to final labels

### Input Requirements
- **Primary**: `sysmon-run-XX.csv` - Raw Sysmon dataset
- **Traced Events**: `traced_sysmon_events_with_tactics_v2.csv` - Manually corrected traced events
- **Master File**: `all_target_events_run-XX.csv` - Original seed selections for tactic lookup

### Output Files

#### Primary Output
- **`sysmon-run-XX-labeled.csv`** - Complete labeled Sysmon dataset

#### Column Structure (Final Labeled Dataset)
```
[All original Sysmon columns...], Tactic, Technique, Attack_Label
```

#### Key Output Descriptions
- **`Tactic`**: MITRE ATT&CK tactic for malicious events, empty for benign
- **`Technique`**: MITRE ATT&CK technique for malicious events, empty for benign
- **`Attack_Label`**: Binary classification (malicious/benign) for ML applications

### Usage Example
```bash
python3 7_create_labeled_sysmon_dataset.py \
    --apt-type apt-1 \
    --run-id 04
```

### Processing Logic
1. **Load Inputs**: Reads raw Sysmon data, corrected traced events, and master tactics file
2. **Apply Corrections**: Uses `Correct_SeedRowNumber` values when provided, falls back to `OriginatorRow`
3. **Lookup Tactics**: Cross-references with master file to get accurate MITRE labels
4. **Label Dataset**: Marks all traced events as malicious with appropriate tactics/techniques
5. **Generate Output**: Creates complete labeled dataset with audit information

### Manual Correction Workflow Integration
- **Reads v2.csv**: Processes manually-corrected traced events file
- **Respects Human Judgment**: Prioritizes manual corrections over automatic attributions
- **Maintains Attribution**: Preserves original automatic attributions when no manual correction provided
- **Audit Trail**: Reports correction statistics and attribution sources

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

This three-script pipeline provides comprehensive, production-ready attack lifecycle analysis from raw Sysmon data with integrated human expertise validation. It combines:

- **Manual Curation**: Human-guided seed event selection and correction workflows
- **Sophisticated Event Correlation**: ProcessGuid-based tracing across multiple EventIDs
- **MITRE ATT&CK Integration**: Standardized tactic and technique labeling
- **Advanced Visualization**: Timeline analysis and attack progression mapping
- **Quality Assurance**: Manual correction capabilities for accurate final datasets

### Key Innovations
1. **Human-in-the-Loop Design**: Balances automation with human expertise for accurate attack identification
2. **Dual Timestamp Architecture**: Human-readable analysis + preserved Unix milliseconds for processing compatibility
3. **Complete Audit Trail**: Full traceability from raw data through manual corrections to final labels
4. **Production-Scale Processing**: Handles large Sysmon datasets efficiently with robust error handling

The pipeline represents a significant advancement in host-based attack analysis, providing the foundation for high-quality labeled cybersecurity datasets suitable for machine learning research, threat hunting, and incident response activities.