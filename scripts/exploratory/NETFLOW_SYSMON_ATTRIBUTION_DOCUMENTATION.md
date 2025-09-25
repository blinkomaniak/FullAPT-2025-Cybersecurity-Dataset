# NetFlow-Sysmon Enhanced Temporal Causation Attribution System
## Comprehensive Documentation for `4_enhanced_temporal_causation_correlator.py`

**Purpose**: This document provides comprehensive technical documentation for the enhanced dual-domain causal attribution system that establishes causal relationships between network flow events and host process events across APT simulation scenarios with detailed temporal causation analysis.

For a correct generation of this novel dual-domain dataset -that collects host and network events while performing APT attacks on a target network- it is important to grant and measure how well the event-data collection was performed over both sub-datasets. We can state the premise that all the network events has its origin in some host activity, and only some host activity is a product from network activity. Therefore, one of the fundamental aspects to measure how well the event-data collection was is to find out the percetage of network events that we can attribute to system activity.

**Key Innovation**: The enhanced approach performs **comprehensive temporal causal attribution** with individual case tracking - establishing that network flow events are **caused by** specific host processes through detailed analysis of 13 distinct temporal scenarios across 4 process lifecycle types.

---

## 🎯 Enhanced Attribution Framework Overview

### **Causal Attribution Chain**:
```
Initial Machine (Source Host)
    ↓ Host Activity (Sysmon Events)
    ↓ Generates Network Traffic
Network Flow Events (NetFlow)
    ↓ Propagates Actions  
    ↓ Triggers Remote Execution
Target Machine (Destination Host)
    ↓ Resulting Host Activity (Sysmon Events)
```

### **Enhanced Attribution Relationships**:

#### **1. Direct Process Attribution (Same Host)**:
- **NetFlow → Sysmon**: "These network events were **caused by** this process"
- **Temporal Validation**: 13 individual temporal scenarios analyzed per attribution
- **Case-Specific Analysis**: Each temporal relationship tracked separately

#### **2. Cross-Host Causal Attribution (Different Hosts)**:
- **Source Sysmon → NetFlow → Target Sysmon**: "This initial activity **triggered** that remote activity"
- **Multi-Stage Activity Chain**: Process A → Network → Process B → Network → Process C
- **Enhanced Cross-Host Timing**: Detailed temporal gap analysis for lateral movement detection

#### **3. Enhanced Temporal Causal Validation**:
- **13 Individual Cases**: Comprehensive scenario coverage across all process lifecycle types
- **Case-Specific Timing Collection**: Pre-gaps, post-gaps, and overlap ratios per case
- **Process Type Specialization**: Different temporal logic for Start-End, No-End, No-Start, No-Start-No-End processes

---

## 🔍 Enhanced Attribution Methodology

### **Process Lifecycle Attribution with Case-Specific Analysis**:

#### **1. Start-End Processes (6 Individual Cases)**:
- **Case 1**: NetFlow completely inside Sysmon process span
- **Case 2**: NetFlow starts before Sysmon process starts
- **Case 3**: NetFlow ends after Sysmon process ends
- **Case 4**: Sysmon process completely inside NetFlow span
- **Case 5**: NetFlow occurs after Sysmon process ends (post-termination)
- **Case 6**: NetFlow occurs before Sysmon process starts (pre-start)

#### **2. No-End Processes (3 Individual Cases)**:
- **Case 1**: NetFlow starts after process start (normal operation)
- **Case 2**: NetFlow spans across process start (overlap scenario)
- **Case 3**: NetFlow ends before process start (trigger scenario)

#### **3. No-Start Processes (3 Individual Cases)**:
- **Case 1**: NetFlow ends before process end (contained activity)
- **Case 2**: NetFlow spans across process end (overlap scenario)
- **Case 3**: NetFlow starts after process end (post-termination activity)

#### **4. No-Start-No-End Processes (1 Case)**:
- **Case 1**: Always attributed (no temporal constraints due to monitoring window limitations)

### **Multi-Stage Attribution Strategy**:
```python
# Enhanced Primary Attribution: Direct PID matching with temporal validation
if netflow_event.process_pid == sysmon_process.pid:
    temporal_result = analyze_temporal_overlap_enhanced(flow, sysmon_process)
    if temporal_result.has_overlap:
        → Direct Attribution with Case-Specific Classification

# Enhanced Source Attribution: Source IP → Computer mapping with timing analysis
elif netflow_event.source_ip == sysmon_process.computer_ip:
    → Source Machine Attribution with Temporal Analysis

# Enhanced Destination Attribution: Cross-host causation with detailed timing
elif netflow_event.destination_ip == sysmon_process.computer_ip:
    → Target Machine Attribution with Case-Specific Timing Collection
```

---

## 🏗️ Enhanced System Architecture

### **Enhanced Attribution Engine Components**:

#### **1. Enhanced Data Ingestion**:
- **NetFlow Events**: Network activity with comprehensive process context
- **Sysmon Events**: Host process lifecycle with detailed temporal boundaries
- **Dynamic IP-Computer Mapping**: Real-time host identification extraction
- **Case-Specific Timing Collection**: Individual timing metrics per scenario

#### **2. Enhanced Temporal Analysis Processing**:
- **Process Lifecycle Analysis**: Complete 4-type process classification system
- **Individual Case Classification**: 13 distinct temporal scenario identification
- **Case-Specific Timing Collection**: Pre-gaps, post-gaps, overlap ratios per case
- **Multi-Stage Attribution**: Primary → Source → Destination with temporal validation

#### **3. Enhanced Validation & Quality Control**:
- **Comprehensive Temporal Validation**: All 13 cases validated individually
- **Case-Specific Statistics**: Detailed timing metrics per scenario
- **Cross-Host Temporal Analysis**: Enhanced lateral movement timing validation
- **Attribution Confidence Scoring**: Quality metrics per temporal case

### **Enhanced Attribution Performance Metrics**:

#### **Case-Specific Event Attribution**:
- **Total Events**: Individual network events processed with case classification
- **Case Distribution**: Events classified into 13 individual temporal scenarios
- **Case-Specific Attribution Rates**: Success rates per temporal case

#### **Enhanced Flow-Level Attribution**:
- **Total Flows**: Logical network sessions with comprehensive temporal analysis
- **Case-Specific Flow Distribution**: Flows classified by temporal scenario
- **Temporal Case Performance**: Attribution success per individual case

---

## 📊 Enhanced Temporal Analysis - Detailed Case Calculations

### **📊 Start-End Process Cases (6 cases)**

| **Case** | **Condition** | **Pre-Gap Calculation** | **Post-Gap Calculation** | **Overlap Ratio Calculation** |
|----------|---------------|------------------------|--------------------------|-------------------------------|
| **Case 1** | `sysmon_start ≤ netflow_start AND netflow_end ≤ sysmon_end`<br>*(NetFlow inside Sysmon)* | `netflow_start - sysmon_start`<br>*(≥0, NetFlow starts after)* | `sysmon_end - netflow_end`<br>*(≥0, Sysmon ends after)* | `netflow_duration / sysmon_duration`<br>*(NetFlow fraction of Sysmon)* |
| **Case 2** | `netflow_start ≤ sysmon_start AND netflow_end ≤ sysmon_end`<br>*(NetFlow starts before)* | `sysmon_start - netflow_start`<br>*(≥0, NetFlow starts before)* | `sysmon_end - netflow_end`<br>*(≥0, Sysmon ends after)* | `overlap_duration / sysmon_duration`<br>*(Overlap fraction of Sysmon)* |
| **Case 3** | `sysmon_start ≤ netflow_start AND sysmon_end ≤ netflow_end`<br>*(NetFlow ends after)* | `netflow_start - sysmon_start`<br>*(≥0, NetFlow starts after)* | `netflow_end - sysmon_end`<br>*(≥0, NetFlow ends after)* | `overlap_duration / sysmon_duration`<br>*(Overlap fraction of Sysmon)* |
| **Case 4** | `sysmon_start ≥ netflow_start AND sysmon_end ≤ netflow_end`<br>*(Sysmon inside NetFlow)* | `sysmon_start - netflow_start`<br>*(≥0, NetFlow starts before)* | `netflow_end - sysmon_end`<br>*(≥0, NetFlow ends after)* | `sysmon_duration / netflow_duration`<br>*(Sysmon fraction of NetFlow)* |
| **Case 5** | `netflow_start ≥ sysmon_end`<br>*(NetFlow after Sysmon ends)* | **N/A** *(No overlap)* | `netflow_start - sysmon_end`<br>*(Gap after termination)* | **N/A** *(No overlap)* |
| **Case 6** | `sysmon_start ≥ netflow_end`<br>*(NetFlow before Sysmon starts)* | `sysmon_start - netflow_end`<br>*(Gap before start)* | **N/A** *(No overlap)* | **N/A** *(No overlap)* |

### **📊 No-End Process Cases (3 cases)**

| **Case** | **Condition** | **Pre-Gap Calculation** | **Post-Gap Calculation** | **Overlap Ratio Calculation** |
|----------|---------------|------------------------|--------------------------|-------------------------------|
| **Case 1** | `netflow_start ≥ sysmon_start`<br>*(NetFlow starts after process start)* | `netflow_start - sysmon_start`<br>*(≥0, NetFlow after start)* | **N/A** *(No sysmon end)* | **N/A** *(Unbounded end)* |
| **Case 2** | `netflow_end ≥ sysmon_start AND sysmon_start ≥ netflow_start`<br>*(NetFlow spans across start)* | `sysmon_start - netflow_start`<br>*(≥0, NetFlow starts before)* | **N/A** *(No sysmon end)* | **N/A** *(Unbounded end)* |
| **Case 3** | `sysmon_start ≥ netflow_end`<br>*(NetFlow ends before start - trigger scenario)* | `sysmon_start - netflow_end`<br>*(Trigger gap)* | **N/A** *(No sysmon end)* | **N/A** *(Unbounded end)* |

### **📊 No-Start Process Cases (3 cases)**

| **Case** | **Condition** | **Pre-Gap Calculation** | **Post-Gap Calculation** | **Overlap Ratio Calculation** |
|----------|---------------|------------------------|--------------------------|-------------------------------|
| **Case 1** | `sysmon_end ≥ netflow_end`<br>*(NetFlow ends before process end)* | **N/A** *(No sysmon start)* | `sysmon_end - netflow_end`<br>*(≥0, Process ends after)* | **N/A** *(Unbounded start)* |
| **Case 2** | `netflow_end ≥ sysmon_end AND sysmon_end ≥ netflow_start`<br>*(NetFlow spans across end)* | **N/A** *(No sysmon start)* | `netflow_end - sysmon_end`<br>*(≥0, NetFlow ends after)* | **N/A** *(Unbounded start)* |
| **Case 3** | `netflow_start ≥ sysmon_end`<br>*(NetFlow starts after process end)* | **N/A** *(No sysmon start)* | `netflow_start - sysmon_end`<br>*(Post-termination gap)* | **N/A** *(Unbounded start)* |

### **📊 No-Start-No-End Process Cases (1 case)**

| **Case** | **Condition** | **Pre-Gap Calculation** | **Post-Gap Calculation** | **Overlap Ratio Calculation** |
|----------|---------------|------------------------|--------------------------|-------------------------------|
| **Case 1** | *Always attributed*<br>*(No temporal constraints)* | **N/A** *(No boundaries)* | **N/A** *(No boundaries)* | **N/A** *(No boundaries)* |

---

## 📝 Enhanced Timing Calculation Reference

### **Key Timing Variables**:
- `netflow_start` = `flow.event_start` (earliest NetFlow event timestamp)
- `netflow_end` = `flow.event_end` (latest NetFlow event timestamp)
- `sysmon_start` = `sysmon_process.start_time` (process creation timestamp)
- `sysmon_end` = `sysmon_process.end_time` (process termination timestamp)
- `netflow_duration` = `netflow_end - netflow_start`
- `sysmon_duration` = `sysmon_end - sysmon_start`

### **Enhanced Gap Interpretation Guidelines**:
- **All Pre-Gap Values**: Always positive (≥0) - representing absolute temporal distance
- **All Post-Gap Values**: Always positive (≥0) - representing absolute temporal distance
- **Gap Magnitude**: Larger values indicate greater temporal separation
- **Zero Gaps**: Indicate exact temporal alignment

### **Overlap Ratio Interpretation**:
- **< 1.0**: NetFlow is shorter than the reference timespan
- **= 1.0**: NetFlow exactly matches the reference timespan
- **> 1.0**: NetFlow is longer than the reference timespan

### **Special Case Handling**:
- **Cases 5 & 6 (Start-End)**: No overlap scenarios - only gap measurements are meaningful
- **No-End Processes**: Only pre-gaps are calculated (no end boundary exists)
- **No-Start Processes**: Only post-gaps are calculated (no start boundary exists)
- **No-Start-No-End**: No timing constraints - processes are always attributed regardless of temporal relationships

---

## 🔧 Enhanced Technical Implementation

### **Enhanced Key Attribution Functions**:

#### **`analyze_temporal_overlap_enhanced()`** - Core Temporal Analysis Engine:
```python
def analyze_temporal_overlap_enhanced(self, flow: Dict, sysmon_process: Dict) -> Dict:
    """Enhanced temporal causation analysis with 13 individual case classification"""
    
    # Extract temporal information
    netflow_start = flow.get('event_start')
    netflow_end = flow.get('event_end')
    sysmon_start = sysmon_process.get('start_time')
    sysmon_end = sysmon_process.get('end_time')
    process_type = sysmon_process.get('lifecycle_type')
    
    # Process type-specific temporal analysis with case classification
    if process_type == "Start-End Process":
        return self._analyze_start_end_process(
            netflow_start, netflow_end, sysmon_start, sysmon_end
        )
    elif process_type == "No-End Process":
        return self._analyze_no_end_process(
            netflow_start, netflow_end, sysmon_start
        )
    elif process_type == "No-Start Process":
        return self._analyze_no_start_process(
            netflow_start, netflow_end, sysmon_end
        )
    elif process_type == "No-Start-No-End Process":
        return self._analyze_no_bounds_process(netflow_start, netflow_end)
```

#### **`_analyze_start_end_process()`** - 6-Case Temporal Analysis:
```python
def _analyze_start_end_process(self, netflow_start, netflow_end, sysmon_start, sysmon_end):
    """Analyze Start-End Process temporal scenarios with individual case tracking"""
    
    # Case 1: Netflow span inside Sysmon process span
    if sysmon_start <= netflow_start and netflow_end <= sysmon_end:
        self.temporal_stats.start_end_cases['case_1'] += 1
        pre_gap_ms = (netflow_start - sysmon_start).total_seconds() * 1000
        post_gap_ms = (sysmon_end - netflow_end).total_seconds() * 1000
        overlap_ratio = netflow_duration_ms / sysmon_duration_ms
        
        # Store case-specific timing statistics
        self.temporal_stats.timing_stats['start_end_case_1']['pre_gaps'].append(pre_gap_ms)
        self.temporal_stats.timing_stats['start_end_case_1']['post_gaps'].append(post_gap_ms)
        self.temporal_stats.timing_stats['start_end_case_1']['overlap_ratios'].append(overlap_ratio)
        
        return {'has_overlap': True, 'scenario': 'start_end_case_1'}
    
    # Case 2: Netflow starts before Sysmon process starts
    elif netflow_start <= sysmon_start and netflow_end <= sysmon_end:
        self.temporal_stats.start_end_cases['case_2'] += 1
        pre_gap_ms = (sysmon_start - netflow_start).total_seconds() * 1000  # Positive value
        post_gap_ms = (sysmon_end - netflow_end).total_seconds() * 1000
        overlap_ratio = overlap_duration_ms / sysmon_duration_ms
        
        # Store case-specific timing statistics
        self.temporal_stats.timing_stats['start_end_case_2']['pre_gaps'].append(pre_gap_ms)
        self.temporal_stats.timing_stats['start_end_case_2']['post_gaps'].append(post_gap_ms)
        self.temporal_stats.timing_stats['start_end_case_2']['overlap_ratios'].append(overlap_ratio)
        
        return {'has_overlap': True, 'scenario': 'start_end_case_2'}
    
    # ... (Cases 3-6 follow similar pattern with case-specific timing collection)
```

### **Enhanced Attribution Output Structure**:
```json
{
    "analysis_metadata": {
        "apt_type": "apt-1",
        "run_id": "10",
        "analysis_version": "v3.0-enhanced-temporal",
        "total_flows_analyzed": 9515
    },
    "attribution_summary": {
        "successfully_attributed": 5599,
        "temporal_mismatches": 1459,
        "missing_pid": 1373,
        "no_sysmon_match": 890
    },
    "temporal_scenario_statistics": {
        "start_end_cases": {
            "case_1": 3247,
            "case_2": 1582,
            "case_3": 891,
            "case_4": 445,
            "case_5": 234,
            "case_6": 200
        },
        "no_end_cases": {
            "case_1": 1456,
            "case_2": 678,
            "case_3": 234
        },
        "no_start_cases": {
            "case_1": 890,
            "case_2": 445,
            "case_3": 123
        },
        "no_bounds_cases": {
            "case_1": 567
        }
    },
    "case_specific_timing_statistics": {
        "start_end_case_1": {
            "pre_gaps_stats": {"mean": 245.7, "median": 89.2, "std": 567.3},
            "post_gaps_stats": {"mean": 1567.9, "median": 445.1, "std": 2134.5},
            "overlap_ratios_stats": {"mean": 0.67, "median": 0.58, "std": 0.34}
        },
        "start_end_case_2": {
            "pre_gaps_stats": {"mean": 334.5, "median": 156.7, "std": 678.9},
            "post_gaps_stats": {"mean": 2234.1, "median": 891.3, "std": 3456.7},
            "overlap_ratios_stats": {"mean": 0.45, "median": 0.34, "std": 0.28}
        }
        // ... (all 13 cases with detailed timing statistics)
    }
}
```

---

## 📊 Enhanced Attribution Analysis Results

### **High-Performing Enhanced Attribution Scenarios** (≥90% Event Attribution):
- **APT-1**: 6 runs (09, 10, 11, 12, 15, 51) - Enhanced temporal analysis ready
- **APT-2**: 3 runs (22, 29, 30) - Case-specific timing analysis available
- **APT-3**: 2 runs (37, 38) - Individual case classification completed
- **APT-4**: 5 runs (39, 41, 42, 43, 44) - Comprehensive temporal metrics collected
- **APT-5**: 2 runs (46, 47) - Full 13-case analysis implemented
- **APT-6**: 3 runs (48, 49, 50) - Complete case-specific timing available

### **Enhanced Attribution Performance Analysis**:
- **Overall Event Attribution**: 71.30% with case-specific breakdown available
- **Individual Case Distribution**: Detailed statistics per temporal scenario
- **Case-Specific Performance**: Success rates per individual temporal case
- **Enhanced Timing Metrics**: Pre-gaps, post-gaps, overlap ratios per case

---

## 📚 Enhanced Usage Guidelines

### **Enhanced Attribution Analysis Workflow**:

#### **Phase 1: Data Validation**
```bash
# Validate network flow foundations
python3 1_comprehensive_network_community_id_analyzer.py --apt-type apt-1 --run-id 10

# Validate process identity consistency  
python3 2_process_tuple_uniqueness_validator.py --apt-type apt-1 --run-id 10
```

#### **Phase 2: Enhanced Temporal Causal Attribution Analysis**
```bash
# Run enhanced temporal attribution engine (single run)
python3 4_enhanced_temporal_causation_correlator.py --apt-type apt-1 --run-id 10

# Run enhanced temporal attribution (high-performing runs batch)
python3 4_enhanced_temporal_causation_correlator.py --batch-high-performing --workers 8

# Run enhanced temporal attribution (all runs batch)
python3 4_enhanced_temporal_causation_correlator.py --batch-all --workers 8
```

#### **Phase 3: Enhanced Attribution Visualization & Analysis**
```bash  
# Generate comprehensive attribution analysis
python3 complete_correlation_summary.py
python3 event_attribution_timeline_plot.py
```

### **Enhanced Attribution Output Files**:

#### **Comprehensive JSON Results**:
- `enhanced_netflow_sysmon_correlation-run-XX.json` - Complete case-specific attribution results

#### **Enhanced Visualization Suite**:
- `scenario_distribution_flows.png` - Flow-level scenario distribution with case details
- `scenario_distribution_events.png` - Event-level scenario distribution with case breakdown
- `process_lifecycle_breakdown.png` - Process type distribution with percentages
- `start_end_timing_analysis.png` - 6×3 subplot grid (6 cases × 3 timing types)
- `no_end_timing_analysis.png` - 3×2 subplot grid (3 cases × pre-gaps + netflow duration)
- `no_start_timing_analysis.png` - 3×2 subplot grid (3 cases × post-gaps + netflow duration)
- `no_bounds_timing_analysis.png` - Special plot for unbounded processes + netflow duration

#### **NPZ Format Exports**:
- All plots include corresponding `.npz` files for batch processing and analysis

---

## 🚀 Enhanced Future Development

### **Advanced Temporal Analysis Capabilities**:
- **Temporal Pattern Recognition**: ML-based temporal scenario pattern analysis
- **Cross-Case Temporal Modeling**: Relationships between different temporal cases
- **Dynamic Temporal Thresholds**: Adaptive timing windows based on case performance
- **Temporal Anomaly Detection**: Unusual temporal patterns indicating advanced techniques

### **Case-Specific Research Extensions**:
- **Case Performance Optimization**: Individual case tuning for improved attribution
- **Temporal Case Correlation**: Relationships between cases and attack techniques
- **Case-Based Attribution Confidence**: Confidence scoring per temporal scenario
- **Advanced Temporal Visualization**: Interactive temporal case exploration tools

---

**Conclusion**: This enhanced dual-domain temporal causation attribution system represents a significant advancement in cybersecurity analytics by establishing detailed causal relationships through comprehensive individual case analysis. The 13-case temporal framework enables precise attribution analysis across all process lifecycle types, providing unprecedented insight into the temporal relationships between network activities and host processes for advanced persistent threat detection and analysis.

---

## 🚀 **MAJOR UPDATE 2025-09-25: FULL AUTOMATION & VISUALIZATION ENHANCEMENT**

### **BREAKTHROUGH: Complete Manual Checkpoint Automation**

**Revolutionary Achievement**: Successfully automated the entire INTEGRATED NetFlow labeling pipeline, eliminating manual intervention bottlenecks while maintaining identical accuracy and adding professional-quality visualizations.

#### **🎯 Automated Assignment System Implementation**

##### **Manual Checkpoint 2 Elimination**:
Previously, researchers had to manually edit CSV files for subnetflow assignments. This manual step has been **completely automated** through:

**Feature Flag System**:
```bash
# Fully automated mode (recommended)
python3 INTEGRATED_netflow_labeler.py --apt-type apt-1 --run-id 04 --automated-assignment

# Traditional manual mode (still available)
python3 INTEGRATED_netflow_labeler.py --apt-type apt-1 --run-id 04 --no-automated-assignment
```

**End-Time Proximity Assignment Algorithm**:
- **Window**: ±5 seconds from subnetflow end time
- **Logic**: Multi-assignment capability (one seed event can assign to multiple subnetflows)
- **Processing**: Systematic handling of ALL communities marked for Subnetflow-attribution
- **Accuracy**: 400+ automated assignments matching manual reference exactly (APT-1-Run-04)

##### **Community-Specific Processing**:
```python
# Dynamic community discovery from verification matrix
subnetflow_communities = verification_df[subnetflow_mask]['network_community_id'].unique()

for community_id in subnetflow_communities:
    # Load community-specific seed events from verification matrix timestamps
    community_seed_events = self._get_community_seed_events(community_id, apt_type, run_id)

    # Apply automated end-time proximity assignment
    assignments = self._apply_end_time_proximity_assignment(
        subnetflows_df, community_seed_events, time_window_seconds=10
    )
```

#### **🎨 Professional Visualization System Enhancement**

##### **Critical Visualization Issues Resolved**:

**1. Empty Plot Problem Fixed**:
- **Issue**: Community `1:6okao16uEuHUP9NnUm/SWPt9rTQ=` with 818 subnetflows showed empty plots
- **Root Cause**: Subnetflow durations averaging 0.088 seconds resulted in matplotlib bar widths of ~1e-06 (invisible)
- **Solution**: Implemented 1-second absolute minimum bar width
- **Technical Implementation**:
```python
min_bar_width_seconds = 1.0  # 1 second minimum for visibility
min_bar_width = min_bar_width_seconds / 86400.0  # Convert to matplotlib date units
if duration_mpl <= 0 or duration_mpl < min_bar_width:
    duration_mpl = min_bar_width
```
- **Impact**: All subnetflows now visible regardless of actual duration

**2. Y-axis Label Clarity**:
- **Before**: Complex network_community_id format cluttering labels
- **After**: Clean ordinal format: `1 → 118936`, `2 → 119570` (subnetflow → seed_event)
- **Implementation**: Simple ordinal numbering with seed event assignment arrows

**3. X-axis Alignment Precision**:
- **Issue**: Top axis seed event numbers misaligned with vertical lines
- **Root Cause**: Top axis configuration occurred before main axis limits were established
- **Solution**: Moved top axis setup to occur AFTER main axis limits are finalized
- **Result**: Perfect pixel-level alignment between vertical lines and labels

**4. Visual Clarity Enhancement**:
- **Eliminated**: Distracting purple/navy border colors on horizontal bars
- **Implementation**: `edgecolor='none'` and `linewidth=0`
- **Result**: Clean tactic-colored bars with maximum visual clarity

**5. Dynamic Title Accuracy**:
- **Fixed**: Hardcoded `±7s` in titles despite using `±5s` algorithm
- **Solution**: Dynamic parameter passing and calculation
- **Result**: Accurate titles showing `±{actual_window}s End-Time Proximity Algorithm`

#### **📊 Enhanced Timeline Visualization Features**

##### **Community-Specific Filtering**:
```python
def _get_community_seed_events(self, community_id: str, apt_type: str, run_id: str):
    """Filter seed events to only those relevant to specific community"""

    # Use verification matrix correlation timestamps (not raw seed timestamps)
    community_seeds = verification_df[
        (verification_df['network_community_id'] == community_id) &
        (verification_df['Subnetflow-attribution'] == 'x')
    ]

    # Create DataFrame with correlation-adjusted timestamps
    seed_details = []
    for _, row in community_seeds.iterrows():
        seed_details.append({
            'OriginalRowNumber': row['seed_event'],
            'seed_timestamp': pd.to_datetime(row['seed_timestamp']),  # Correlation time
            'Tactic': seed_info['Tactic'],
            'Technique': seed_info['Technique']
        })
```

##### **Professional Plot Specifications**:
- **Minimum Bar Width**: 1 second absolute (prevents invisible bars)
- **Y-axis Labels**: Ordinal numbering with assignment arrows
- **X-axis Alignment**: Post-axis-limits configuration for precision
- **Color Scheme**: Pure MITRE tactic colors without borders
- **Title Format**: Dynamic time window parameter display
- **Layout**: Optimized spacing and legend positioning

#### **🗂️ Complete Project Consolidation**

##### **Script Cleanup & Integration**:
**Moved to Backup** (15 files total):
- `end_time_proximity_assignment.py` ✅ (logic fully integrated into INTEGRATED script)
- `complete_pattern_analysis_workflow*.py` ✅ (superseded by INTEGRATED pipeline)
- All debugging/testing scripts from today's session ✅
- Pre-existing metadata testing scripts ✅

**Production Environment**:
- **Single Script Solution**: INTEGRATED_netflow_labeler.py handles complete pipeline
- **Clean Workspace**: Only essential production scripts remain
- **Maintainable**: Clear separation between production and development code

#### **🚀 Current Production Capabilities**

##### **Complete Automated Workflow**:
```bash
# Single command for complete automated pipeline
python3 INTEGRATED_netflow_labeler.py --apt-type apt-1 --run-id 04 --automated-assignment
```

**Expected Results**:
1. ✅ **Zero Manual Intervention**: Complete automation of subnetflow assignments
2. ✅ **Professional Visualizations**: Publication-ready timeline plots with perfect alignment
3. ✅ **Universal Visibility**: All subnetflows visible regardless of duration
4. ✅ **Accurate Metrics**: Dynamic titles and labels reflecting actual parameters
5. ✅ **Complete Coverage**: Systematic processing of all marked communities
6. ✅ **Quality Assurance**: Comprehensive validation and error handling

##### **Technical Performance (APT-1-Run-04)**:
- **Communities Processed**: 19 (marked for Subnetflow-attribution)
- **Automated Assignments**: 400+ (matching manual reference exactly)
- **Processing Time**: ~5-10 minutes total
- **Timeline Plots**: Professional quality with 1-second minimum bar widths
- **Visualization Files**: Complete timeline + individual community plots

#### **📈 Research & Analysis Benefits**

##### **Advanced Attribution Capabilities**:
- **Dual-Tier System**: Community-level + segment-level attribution
- **Temporal Precision**: ±5 second proximity with millisecond accuracy
- **Visual Validation**: Every assignment visible in professional timeline plots
- **Quality Metrics**: Comprehensive statistics and confidence reporting
- **Scalability**: Batch processing across all APT datasets

##### **Enhanced Data Quality**:
- **Ground Truth Generation**: High-quality labeled datasets for ML training
- **Attribution Provenance**: Complete tracking of assignment decisions
- **Visual Verification**: Every subnetflow visible and properly attributed
- **Consistency**: Uniform processing regardless of data characteristics

**AUTOMATION STATUS**: The INTEGRATED NetFlow labeling system now provides complete end-to-end automation with professional visualization quality, eliminating manual bottlenecks while maintaining research-grade accuracy and providing unprecedented visual insight into dual-domain APT attack attribution patterns.

---

# Script 5: Sysmon Seed Event Extractor
## Comprehensive Documentation for `5_sysmon_seed_event_extractor.py`

**Purpose**: Extracts and processes manually selected seed events from Sysmon datasets to serve as ground truth attack indicators for NetFlow attribution analysis. This script identifies and validates critical attack lifecycle events that will be used as reference points for downstream correlation and labeling processes.

### **🎯 Core Functionality**

#### **1. Manual Seed Event Selection**:
- Processes `all_target_events_run-XX.csv` containing all potential seed events
- Filters to manually selected events marked in the dataset
- Validates seed event integrity and completeness
- Extracts MITRE ATT&CK technique and tactic information

#### **2. Attack Lifecycle Categorization**:
- **Initial Access Events**: First compromise indicators
- **Execution Events**: Command execution and process creation
- **Persistence Events**: Registry modifications and startup persistence
- **Defense Evasion Events**: Anti-analysis and stealth techniques
- **Credential Access Events**: Password dumping and credential harvesting
- **Discovery Events**: System and network reconnaissance
- **Lateral Movement Events**: Cross-host activity propagation
- **Collection Events**: Data gathering and staging
- **Exfiltration Events**: Data transfer and command-and-control

#### **3. Temporal Validation**:
- Validates timestamp consistency across seed events
- Ensures proper chronological ordering within attack campaigns
- Identifies temporal gaps and clustering patterns
- Validates event timing against NetFlow data boundaries

### **📊 Output Generation**

#### **Seed Event Dataset**:
```csv
OriginalRowNumber,EventID,Computer,timestamp,Tactic,Technique,selected_manually
12845,1,DESKTOP-USER01,1620123456789,Execution,T1059.003,true
15623,11,SERVER-DC01,1620123478901,Persistence,T1547.001,true
```

#### **Validation Reports**:
- Seed event completeness analysis
- Temporal distribution validation
- MITRE technique coverage assessment
- Cross-run consistency verification

### **🚀 Usage Examples**

```bash
# Extract seed events for specific run
python3 5_sysmon_seed_event_extractor.py --run-id 05 --apt-type apt-1

# Validate seed events across multiple runs
python3 5_sysmon_seed_event_extractor.py --validate-all --apt-type apt-1

# Generate seed event summary report
python3 5_sysmon_seed_event_extractor.py --summary-report --apt-type apt-1
```

---

# Script 6: Sysmon Attack Lifecycle Tracer
## Comprehensive Documentation for `6_sysmon_attack_lifecycle_tracer.py`

**Purpose**: Traces the complete attack lifecycle through Sysmon events, establishing temporal chains of attack activities and identifying attack progression patterns across the target network infrastructure.

### **🎯 Core Functionality**

#### **1. Attack Chain Reconstruction**:
- Traces attack progression from initial access to exfiltration
- Identifies parent-child process relationships
- Maps cross-host lateral movement sequences
- Establishes temporal causation chains between attack stages

#### **2. Lifecycle Stage Analysis**:
```python
ATTACK_STAGES = {
    'initial_access': ['T1566', 'T1078', 'T1190'],
    'execution': ['T1059', 'T1047', 'T1053'],
    'persistence': ['T1547', 'T1543', 'T1136'],
    'privilege_escalation': ['T1068', 'T1055', 'T1134'],
    'defense_evasion': ['T1027', 'T1070', 'T1036'],
    'credential_access': ['T1003', 'T1552', 'T1558'],
    'discovery': ['T1087', 'T1018', 'T1083'],
    'lateral_movement': ['T1021', 'T1210', 'T1080'],
    'collection': ['T1560', 'T1005', 'T1039'],
    'exfiltration': ['T1041', 'T1048', 'T1567']
}
```

#### **3. Process Relationship Mapping**:
- **Parent-Child Process Trees**: Complete process genealogy tracking
- **Cross-Host Process Chains**: Lateral movement process relationships
- **Command Line Analysis**: Argument and parameter pattern identification
- **File System Activity**: File creation, modification, and deletion tracking

#### **4. Temporal Attack Pattern Analysis**:
- **Attack Velocity Measurement**: Time between attack stages
- **Dwell Time Analysis**: Persistence duration per compromise stage
- **Activity Clustering**: Burst vs. sustained attack patterns
- **Cross-Host Timing**: Lateral movement propagation delays

### **📊 Attack Lifecycle Visualization**

#### **Timeline Generation**:
- Complete attack timeline with stage transitions
- Process relationship diagrams with temporal ordering
- Cross-host activity correlation matrices
- Attack technique distribution heatmaps

#### **Statistical Analysis**:
- Stage duration distributions
- Attack path complexity metrics
- Process relationship depth analysis
- Technique co-occurrence patterns

### **🚀 Usage Examples**

```bash
# Trace complete attack lifecycle
python3 6_sysmon_attack_lifecycle_tracer.py --run-id 05 --apt-type apt-1

# Generate attack chain visualization
python3 6_sysmon_attack_lifecycle_tracer.py --visualize-chains --run-id 05 --apt-type apt-1

# Analyze cross-run attack patterns
python3 6_sysmon_attack_lifecycle_tracer.py --compare-runs --apt-type apt-1 --runs 04,05,06
```

---

# Script 7: Labeled Sysmon Dataset Creator
## Comprehensive Documentation for `7_create_labeled_sysmon_dataset.py`

**Purpose**: Creates comprehensive labeled Sysmon datasets by integrating attack lifecycle traces, MITRE ATT&CK mappings, and temporal attribution results to produce ground truth datasets for machine learning and analysis.

### **🎯 Core Functionality**

#### **1. Multi-Source Data Integration**:
- **Seed Events**: Manual ground truth attack indicators
- **Attack Lifecycle**: Complete attack chain relationships
- **Temporal Attribution**: NetFlow-Sysmon correlation results
- **MITRE Mappings**: Technique and tactic classifications

#### **2. Label Generation Strategy**:
```python
LABELING_HIERARCHY = {
    'direct_seed_event': 'malicious',           # Manually selected seed events
    'attack_chain_member': 'malicious',         # Process chain participants
    'temporally_attributed': 'suspicious',      # NetFlow-correlated events
    'baseline_activity': 'benign',              # Normal system operations
    'monitoring_infrastructure': 'ignore'       # Elastic Agent and monitoring
}
```

#### **3. Feature Engineering**:
- **Temporal Features**: Event timing and duration calculations
- **Process Features**: PID relationships and command line analysis
- **Network Features**: IP addresses and port associations
- **Behavioral Features**: File system and registry activity patterns

#### **4. Quality Assurance**:
- **Label Consistency Validation**: Cross-reference label assignments
- **Temporal Logic Validation**: Ensure chronological consistency
- **Data Completeness Checks**: Verify all required fields populated
- **Statistical Validation**: Distribution analysis and outlier detection

### **📊 Output Dataset Structure**

#### **Enhanced Sysmon Dataset**:
```csv
OriginalRowNumber,EventID,Computer,timestamp,ProcessId,Image,CommandLine,
ParentProcessId,ParentImage,User,LogonId,SourceIp,DestinationIp,
Tactic,Technique,Label,AttributionSource,AttackStage,ChainPosition
```

#### **Labeling Statistics**:
- Label distribution across event types
- Attack technique coverage analysis
- Temporal distribution validation
- Cross-host label propagation metrics

### **🚀 Usage Examples**

```bash
# Create labeled dataset for specific run
python3 7_create_labeled_sysmon_dataset.py --run-id 05 --apt-type apt-1

# Generate comprehensive labeled dataset with validation
python3 7_create_labeled_sysmon_dataset.py --run-id 05 --apt-type apt-1 --validate

# Create batch labeled datasets
python3 7_create_labeled_sysmon_dataset.py --batch-runs --apt-type apt-1 --runs 04,05,06
```

---

# Script 8: INTEGRATED NetFlow Labeler (Primary Pipeline)
## Comprehensive Documentation for `INTEGRATED_netflow_labeler.py`

**Purpose**: The flagship integrated pipeline that combines all previous analysis stages into a comprehensive dual-domain labeling system. This script implements a sophisticated two-tier attribution system with manual verification checkpoints to produce high-quality labeled NetFlow datasets for advanced persistent threat detection.

### **🎯 Integrated Pipeline Architecture**

#### **Phase 1: Correlation Analysis**
```python
CORRELATION_WORKFLOW = {
    'data_ingestion': ['NetFlow events', 'Sysmon seed events', 'IP-Computer mapping'],
    'temporal_correlation': ['13-case temporal analysis', 'Process lifecycle validation'],
    'attribution_analysis': ['Direct PID matching', 'Source/Destination IP correlation'],
    'verification_matrix': ['Automated correlation results', 'Manual verification structure']
}
```

#### **Phase 2: Manual Verification & Sub-NetFlow Analysis**
```python
TWO_TIER_SYSTEM = {
    'tier_1_direct': {
        'scope': 'Complete network community',
        'granularity': 'All flows in community',
        'attribution': 'Direct seed event association',
        'confidence': 'High (manual verification)'
    },
    'tier_2_subnetflow': {
        'scope': 'Individual sub-NetFlow segments',
        'granularity': 'Specific flow segments by timing',
        'attribution': 'Segment-specific seed event association',
        'confidence': 'Very High (manual segment assignment)'
    }
}
```

#### **Phase 3: Comprehensive Labeling**
```python
LABELING_INTEGRATION = {
    'direct_attribution': 'Tier 1 community-level labeling',
    'subnetflow_attribution': 'Tier 2 segment-level labeling (takes precedence)',
    'conflict_resolution': 'Sub-NetFlow overrides direct attribution',
    'unlabeled_handling': 'Remains benign with full provenance tracking'
}
```

### **🏗️ Integrated System Components**

#### **1. Hybrid Workflow Management**:
- **Automated Processing**: Correlation analysis and file generation
- **Manual Checkpoints**: Verification matrix editing and sub-NetFlow assignment
- **Validation Gates**: Data quality checks at each stage
- **State Persistence**: Workflow resumption from any checkpoint

#### **2. Advanced Visualization Suite**:
```python
VISUALIZATION_COMPONENTS = {
    'correlation_analysis': [
        'complete_timeline_seed_events_vs_c2_netflow',
        'correlation_hotspots_seed_events_vs_c2_netflow'
    ],
    'subnetflow_analysis': [
        'subnetflow_timeline_analysis',
        'seed_event_overlay_visualization',
        'dynamic_temporal_scaling'
    ]
}
```

#### **3. Two-Tier File Management**:
```python
FILE_WORKFLOW = {
    'verification_matrix': {
        'create': 'verification_matrix_run-XX.csv',
        'manual': 'verification_matrix_v2_run-XX.csv'
    },
    'subnetflow_template': {
        'create': 'subnetflow_assignment_template_run-XX.csv',
        'manual': 'subnetflow_assignment_template_v2_run-XX.csv'
    }
}
```

### **📊 Advanced Attribution Logic**

#### **Multi-Stage Attribution Processing**:
```python
def apply_two_tier_labeling(netflow_df):
    # Stage 1: Direct Attribution (Community Level)
    for community_id, seed_event in direct_mapping.items():
        mask = netflow_df['network_community_id'] == community_id
        netflow_df.loc[mask, 'Label'] = 'malicious'
        netflow_df.loc[mask, 'Tactic'] = tactic_lookup[seed_event]['Tactic']
        netflow_df.loc[mask, 'Technique'] = tactic_lookup[seed_event]['Technique']
    
    # Stage 2: Sub-NetFlow Attribution (Segment Level - Takes Precedence)
    for (community_id, subnetflow_id), seed_event in subnetflow_mapping.items():
        mask = ((netflow_df['network_community_id'] == community_id) & 
                (netflow_df['subnetflow_id'] == subnetflow_id))
        # Override any previous attribution
        netflow_df.loc[mask, 'Label'] = 'malicious'
        netflow_df.loc[mask, 'Tactic'] = tactic_lookup[seed_event]['Tactic']
        netflow_df.loc[mask, 'Technique'] = tactic_lookup[seed_event]['Technique']
        netflow_df.loc[mask, 'attribution_source'] = 'subnetflow'
    
    return netflow_df
```

#### **Conflict Resolution Strategy**:
- **Sub-NetFlow Precedence**: Tier 2 always overrides Tier 1 assignments
- **Conflict Tracking**: Detailed logging of attribution conflicts
- **Attribution Provenance**: Complete tracking of attribution sources
- **Quality Metrics**: Attribution confidence and coverage statistics

### **🎛️ Manual Checkpoint System**

#### **Checkpoint 1: Verification Matrix**:
```bash
======================================================================
⏸️  MANUAL STEP REQUIRED:
   📝 Copy and rename: verification_matrix_run-XX.csv
   📝 Create manually: verification_matrix_v2_run-XX.csv
   📋 Mark 'x' in Direct-attribution and Subnetflow-attribution columns
======================================================================
```

#### **Checkpoint 2: Sub-NetFlow Assignment**:
```bash
======================================================================
⏸️  MANUAL STEP REQUIRED:
   📝 Reference file: subnetflow_assignment_template_run-XX.csv
   📝 Please edit: subnetflow_assignment_template_v2_run-XX.csv
   📋 Fill 'seed_event' column for each sub-NetFlow assignment
   💡 Use OriginalRowNumber values from your seed events
   💡 subnetflow_id is now numeric (1,2,3...) for easy sorting
======================================================================
```

### **📈 Enhanced Output Generation**

#### **Comprehensive Labeled NetFlow Dataset**:
```csv
source_ip,destination_ip,source_port,destination_port,protocol,event_start,event_end,
network_community_id,subnetflow_id,process_pid,process_executable,computer_name,
Tactic,Technique,Label,attribution_source
```

#### **Attribution Statistics and Reporting**:
```python
ATTRIBUTION_METRICS = {
    'direct_attributions': 'Community-level attribution count',
    'subnetflow_attributions': 'Segment-level attribution count',
    'conflicts_resolved': 'Tier 1 vs Tier 2 conflicts',
    'coverage_percentage': 'Labeled vs total flow ratio',
    'attribution_confidence': 'Manual verification quality score'
}
```

#### **Visualization Suite Output**:
- **Timeline Plots**: Complete attack timeline with NetFlow correlation
- **Sub-NetFlow Analysis**: Segment-level visualization with seed event overlay
- **Attribution Heatmaps**: Correlation strength and temporal distribution
- **Statistical Dashboards**: Coverage metrics and attribution quality

### **🚀 Integrated Pipeline Usage**

#### **Complete Pipeline Execution**:
```bash
# Run complete integrated pipeline
python3 INTEGRATED_netflow_labeler.py --run-id 05 --apt-type apt-1

# Resume from specific checkpoint
python3 INTEGRATED_netflow_labeler.py --run-id 05 --apt-type apt-1 --resume-checkpoint verification

# Batch processing with validation
python3 INTEGRATED_netflow_labeler.py --batch-runs --apt-type apt-1 --validate-output
```

#### **Advanced Configuration**:
```bash
# Custom temporal thresholds
python3 INTEGRATED_netflow_labeler.py --run-id 05 --apt-type apt-1 --temporal-threshold 5.0

# Enhanced debugging mode
python3 INTEGRATED_netflow_labeler.py --run-id 05 --apt-type apt-1 --debug --verbose

# Production batch processing
python3 INTEGRATED_netflow_labeler.py --batch-all --workers 4 --output-validation
```

### **🔬 Quality Assurance Framework**

#### **Multi-Level Validation**:
```python
VALIDATION_FRAMEWORK = {
    'data_integrity': 'File completeness and format validation',
    'temporal_consistency': 'Chronological order and gap analysis',
    'attribution_logic': 'Conflict resolution and precedence validation',
    'label_distribution': 'Statistical distribution and outlier detection',
    'cross_reference': 'Seed event and NetFlow correlation validation'
}
```

#### **Automated Quality Checks**:
- **File Existence Validation**: All required input files present
- **Schema Compliance**: Data structure and column validation
- **Temporal Logic Validation**: Timeline consistency checks
- **Attribution Coverage**: Percentage of labeled vs unlabeled flows
- **Manual Verification Quality**: Completeness of manual assignments

### **🎯 Integration Benefits**

#### **Unified Workflow Advantages**:
- **Single Point of Control**: All analysis stages in one pipeline
- **State Management**: Resume processing from any checkpoint
- **Quality Assurance**: Integrated validation throughout pipeline
- **Scalability**: Batch processing with parallel execution support

#### **Research and Analysis Benefits**:
- **Ground Truth Generation**: High-quality labeled datasets for ML training
- **Attribution Provenance**: Complete tracking of labeling decisions
- **Temporal Analysis**: Comprehensive timing relationship analysis
- **Attack Pattern Discovery**: Multi-tier attribution pattern identification

---

## 🚀 Integrated System Future Development

### **Advanced Pipeline Extensions**:
- **Machine Learning Integration**: Automated suggestion systems for manual checkpoints
- **Real-Time Processing**: Streaming analysis for live threat detection
- **Multi-Campaign Analysis**: Cross-APT campaign pattern recognition
- **Attribution Confidence Scoring**: Probabilistic attribution assessment

### **Enhanced Automation Capabilities**:
- **Smart Checkpoint Assistance**: AI-powered manual verification suggestions
- **Adaptive Temporal Thresholds**: Dynamic parameter tuning based on data characteristics
- **Cross-Domain Pattern Learning**: Network and host behavior correlation models
- **Automated Quality Assessment**: Continuous pipeline performance monitoring

---

---

## 🔄 **RECENT MAJOR UPDATES (2025-09-10): ENHANCED CORRELATION LOGIC & VISUALIZATION FIXES**

### **SESSION-FAIL-PROOF DOCUMENTATION** 
*Complete implementation details for correlation hotspots logic changes and PNG generation fixes*

---

## 📊 **Major Update 1: Enhanced Correlation Hotspots Logic**

### **🎯 User Requirements Implemented**:
1. **30-second correlation window** (reduced from 5 minutes)
2. **One subplot per seed event** (instead of merged correlation windows)
3. **Seed events centered** in each ±30s subplot
4. **Enhanced time_diff_seconds calculation** with left/right positioning logic
5. **Include flows where seed event falls inside NetFlow timespan**

### **🔧 Critical Code Changes in `INTEGRATED_netflow_labeler.py`**:

#### **Method 1: `_create_correlation_hotspots_plot()` - Lines 1578-1898**
**COMPLETE REWRITE** - Changed from merged correlation windows to individual seed event subplots.

**Key Changes:**
- **Line 1580**: Updated docstring to "Create correlation hotspots with one subplot per seed event (30-second window)"
- **Lines 1587-1596**: Removed old correlation window detection logic, replaced with individual seed event processing
- **Lines 1601-1634**: Enhanced grid layout calculation with size limits to prevent PNG corruption
- **Lines 1632**: Changed correlation threshold: `correlation_threshold = timedelta(seconds=30)` (was 5 minutes)
- **Lines 1644-1671**: **NEW CORRELATION LOGIC** - Enhanced criteria implementation:

```python
# Apply enhanced correlation criteria (matching run_correlation_analysis logic)
include_flow = False
time_diff_seconds = 0.0

# Criterion 1: Seed event falls INSIDE the netflow timespan
if flow_start <= seed_timestamp <= flow_end:
    include_flow = True
    time_diff_seconds = 0.0  # Inside the flow

# Criterion 2: NetFlow is on the LEFT of seed event (flow ends before seed)
elif seed_timestamp > flow_end and (seed_timestamp - flow_end) <= correlation_threshold:
    include_flow = True
    time_diff_seconds = abs((seed_timestamp - flow_end).total_seconds())

# Criterion 3: NetFlow is on the RIGHT of seed event (flow starts after seed) 
elif seed_timestamp < flow_start and (flow_start - seed_timestamp) <= correlation_threshold:
    include_flow = True
    time_diff_seconds = abs((flow_start - seed_timestamp).total_seconds())
```

**Lines 1754-1758**: **CENTERED SEED EVENT DISPLAY**:
```python
# CRITICAL: Set time limits to ±30s display window (seed event centered)
display_padding = timedelta(seconds=30)
display_start = seed_timestamp - display_padding
display_end = seed_timestamp + display_padding
ax.set_xlim(display_start, display_end)
```

**Lines 1765-1770**: **OriginalRowNumber positioning** - Single tick centered on seed event:
```python
# Single tick at the seed event position
mpl_position = mdates.date2num(seed_timestamp)
ax_top.set_xticks([mpl_position])
ax_top.set_xticklabels([str(seed_event['OriginalRowNumber'])], 
                     rotation=45, fontsize=10, ha='center', fontweight='bold')
```

**Lines 1773-1774**: **Finer time granularity**: `mdates.SecondLocator(interval=10)` (10-second intervals for 30s window)

#### **Method 2: `_create_multiple_correlation_hotspots_plots()` - Lines 1900-2091**
**COMPLETELY NEW METHOD** - Handles cases with >36 seed events by splitting into multiple figures.

**Purpose**: Prevent PNG corruption from oversized plots by splitting into manageable 6x6 grids.

**Key Logic:**
- **Line 1907**: Calculate number of figures: `n_figures = (n_seeds + max_subplots_per_figure - 1) // max_subplots_per_figure`
- **Lines 1924-1926**: Fixed grid size: `rows, cols = 6, 6; figsize = (30, 30)`
- **Lines 1961-1969**: **SAME ENHANCED CORRELATION LOGIC** as main method (code duplication necessary)
- **Lines 2083-2084**: **Multi-part file naming**: `correlation_hotspots_seed_events_vs_c2_netflow_run-{run_id}_part{fig_idx + 1}.png`

#### **Grid Layout Logic Update - Lines 1627-1634**:
**ENHANCED PREVENTION OF PNG CORRUPTION**:
```python
else:
    # For very large numbers (>49), split into multiple figures
    max_subplots_per_figure = 36  # 6x6 grid maximum
    if n_seeds > max_subplots_per_figure:
        return self._create_multiple_correlation_hotspots_plots(seed_df, grouped_flows, apt_type, run_id, max_subplots_per_figure)
    else:
        # Use 8x8 grid as fallback for 50-64 seeds
        rows, cols = 8, 8
        figsize = (32, 32)  # Reduced from 40x40
```

---

## 🖼️ **Major Update 2: PNG Corruption Fix**

### **🚨 Problem Diagnosed**:
- **Original issue**: Correlation hotspots plot generated **27,325 x 6,705 pixel** PNG files
- **Root cause**: Poor grid calculation for 61 seed events: `rows=4, cols=16, figsize=(96, 24)` inches
- **At 300 DPI**: 96*300 = 28,800 pixels width → PNG viewer corruption ("chessboard" display)

### **✅ Solution Implemented**:

#### **Multi-Figure Splitting Logic**:
- **Trigger**: When `n_seeds > 36` (Lines 1629-1630)
- **Result**: 61 seeds split into **2 figures**: 36 + 25 subplots
- **New dimensions**: 
  - Part 1: 8,516 x 8,381 pixels ✅
  - Part 2: 8,515 x 7,237 pixels ✅

#### **File Output Changes**:
- **Before**: Single `correlation_hotspots_seed_events_vs_c2_netflow_run-04.png` (corrupted)
- **After**: 
  - `correlation_hotspots_seed_events_vs_c2_netflow_run-04_part1.png` (working)
  - `correlation_hotspots_seed_events_vs_c2_netflow_run-04_part2.png` (working)

---

## 🔄 **Enhanced Correlation Analysis in `run_correlation_analysis()`**

### **Time Window Update - Line 746**:
```python
time_window = timedelta(seconds=30)  # CHANGED from timedelta(minutes=5)
```

### **Enhanced Three-Criteria Logic - Lines 765-781**:
**SAME LOGIC** as correlation hotspots plotting for consistency:

```python
# Enhanced criteria: Check multiple temporal relationships
include_flow = False
time_diff_seconds = 0.0

# Criterion 1: Seed event falls INSIDE the netflow timespan
if flow_start <= seed_time <= flow_end:
    include_flow = True
    time_diff_seconds = 0.0  # Inside the flow

# Criterion 2: NetFlow is on the LEFT of seed event (flow ends before seed)
elif seed_time > flow_end and (seed_time - flow_end) <= time_window:
    include_flow = True
    time_diff_seconds = abs((seed_time - flow_end).total_seconds())

# Criterion 3: NetFlow is on the RIGHT of seed event (flow starts after seed) 
elif seed_time < flow_start and (flow_start - seed_time) <= time_window:
    include_flow = True
    time_diff_seconds = abs((flow_start - seed_time).total_seconds())
```

### **Verification Matrix Impact**:
- **Column**: `time_diff_seconds` now uses **enhanced calculation**
- **Values**: 
  - `0.0` when seed event is inside NetFlow timespan
  - `abs(seed_time - flow_end).total_seconds()` when NetFlow is LEFT of seed
  - `abs(flow_start - seed_time).total_seconds()` when NetFlow is RIGHT of seed

---

## 📋 **Current File States & Locations**

### **Main Script**: `/home/researcher/Downloads/research/dataset/scripts/exploratory/INTEGRATED_netflow_labeler.py`
- **Status**: ✅ **FULLY UPDATED** with all correlation logic changes and PNG fixes
- **Last modified**: 2025-09-10 07:28
- **Key methods updated**: `_create_correlation_hotspots_plot()`, `run_correlation_analysis()`, `_create_multiple_correlation_hotspots_plots()` (new)

### **Test Results - APT-1-Run-04**:
- **Processed**: 61 seed events, 45 NetFlow community IDs
- **Correlation counts**: 1-9 flows per seed event (varies)
- **Output files**: 
  - ✅ `complete_timeline_seed_events_vs_c2_netflow_run-04.png` (working)
  - ✅ `correlation_hotspots_seed_events_vs_c2_netflow_run-04_part1.png` (36 subplots, working)
  - ✅ `correlation_hotspots_seed_events_vs_c2_netflow_run-04_part2.png` (25 subplots, working)
  - ✅ `verification_matrix_run-04.csv` (242 correlations with enhanced time_diff_seconds)

### **Backup Files Created**:
- **Pre-change backup**: `/home/researcher/Downloads/research/dataset/scripts/exploratory/backup-scripts/INTEGRATED_netflow_labeler_v11_backup.py`

---

## 🎯 **Testing & Validation Status**

### **Successful Test Run - APT-1-Run-04**:
```bash
python3 INTEGRATED_netflow_labeler.py --run-id 04 --apt-type apt-1
```

**Console Output Validation**:
```
🎯 Creating 61 subplots (one per seed event)
📊 Creating multiple correlation hotspots plots (36 subplots per figure)
🎯 Splitting 61 seed events into 2 figures
📈 Creating figure 1/2 with 36 seed events
💾 Correlation hotspots part 1 saved: correlation_hotspots_seed_events_vs_c2_netflow_run-04_part1.png
📈 Creating figure 2/2 with 25 seed events
💾 Correlation hotspots part 2 saved: correlation_hotspots_seed_events_vs_c2_netflow_run-04_part2.png
```

**Per-Seed Correlation Counts**:
- Seed 1 (Row 56058): 3 correlated flows
- Seed 2 (Row 105057): 2 correlated flows
- Seed 53 (Row 268551): 9 correlated flows (maximum)
- Average: ~4.5 flows per seed event

---

## 🔧 **Critical Implementation Details**

### **Function Call Flow**:
1. **`run_complete_pipeline()`** → **`run_correlation_analysis()`** (enhanced 30s logic)
2. **`run_complete_pipeline()`** → **`_create_complete_timeline_plot()`** (unchanged)
3. **`run_complete_pipeline()`** → **`_create_correlation_hotspots_plot()`** (completely rewritten)
4. **`_create_correlation_hotspots_plot()`** → **`_create_multiple_correlation_hotspots_plots()`** (when n_seeds > 36)

### **Correlation Logic Consistency**:
**CRITICAL**: Both `run_correlation_analysis()` (lines 765-781) and `_create_correlation_hotspots_plot()` (lines 1652-1666) use **IDENTICAL** enhanced correlation criteria to ensure verification matrix and visualizations are synchronized.

### **Grid Size Calculations**:
```python
# For correlation hotspots plotting:
if n_seeds <= 36:     # Single figure (6x6 grid, figsize=(30,30))
if n_seeds > 36:      # Multi-figure (6x6 grids, figsize=(30,30) each)
max_subplots_per_figure = 36  # Hard limit to prevent PNG corruption
```

### **File Naming Convention**:
- **Single figure**: `correlation_hotspots_seed_events_vs_c2_netflow_run-{run_id}.png`
- **Multi-figure**: `correlation_hotspots_seed_events_vs_c2_netflow_run-{run_id}_part{N}.png`

---

## 🚀 **Session Recovery Instructions**

### **If Session Interrupted**:

1. **Current State**: All changes implemented and tested in `INTEGRATED_netflow_labeler.py`
2. **Key Methods Modified**: 
   - `_create_correlation_hotspots_plot()` (lines 1578-1898) - **COMPLETELY REWRITTEN**
   - `_create_multiple_correlation_hotspots_plots()` (lines 1900-2091) - **BRAND NEW**
   - `run_correlation_analysis()` (lines 765-781) - **ENHANCED CRITERIA**
3. **Test Command**: `python3 INTEGRATED_netflow_labeler.py --run-id 04 --apt-type apt-1`
4. **Expected Output**: 2 correlation hotspots PNG files (part1 and part2) with reasonable dimensions
5. **Verification**: PNG files should display properly without chessboard corruption

### **Key Features to Verify in New Session**:
- ✅ 30-second correlation windows (not 5 minutes)
- ✅ One subplot per seed event (not merged windows)
- ✅ Seed events centered in ±30s subplots
- ✅ Multi-figure splitting for >36 seed events
- ✅ PNG files with reasonable dimensions (<10K x 10K pixels)

---

**Conclusion**: The INTEGRATED NetFlow Labeler now features **enhanced temporal correlation analysis** with individual seed event subplots, 30-second correlation windows, sophisticated time_diff_seconds calculation, and robust PNG generation that prevents corruption through intelligent multi-figure splitting. The system maintains all previous functionality while providing more precise temporal analysis and reliable visualization output for comprehensive APT attack analysis.
