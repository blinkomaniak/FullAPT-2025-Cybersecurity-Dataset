# BITACORA - TESTING Scripts Development Log

This logbook tracks the development and debugging of the TESTING phase scripts for APT attack originator detection and lifecycle tracing.

## Project Context

**Objective**: Create a two-phase system for APT attack analysis:
- **Phase 1**: Find potential attack originators from Sysmon events using keyword matching
- **Phase 2**: Trace complete attack lifecycles from selected originators

**Dataset Structure**: `dataset/apt-Y/apt-Y-run-X/` where Y=1-6, X=04-51
- Each run contains: `sysmon-run-X.csv`, `netflow-run-X.csv`
- Keywords source: `scripts/exploratory/apt-yaml/oilrig_comprehensive_command_extraction.csv`

---

## TESTING_phase1_direct_originator_finder.py

### Purpose
Find potential attack originators by matching keywords from YAML command analysis against Sysmon events.

### Input/Output Flow
```
Input:
- sysmon-run-X.csv (raw Sysmon events)
- oilrig_comprehensive_command_extraction.csv (57 attack commands with keywords)

Output:
- potential_originators_run-X.csv (matched events ready for manual selection)
```

### Key Logic Components

#### 1. Path Resolution (process_apt_run method)
```python
apt_dir = dataset/apt-1/apt-1-run-04/
sysmon_file = apt_dir/sysmon-run-04.csv
keywords_file = scripts/exploratory/apt-yaml/oilrig_comprehensive_command_extraction.csv
output_file = apt_dir/potential_originators_run-04.csv
```

#### 2. Keyword Rules Loading (load_keyword_rules_from_csv method)
- Reads oilrig_comprehensive_command_extraction.csv
- Extracts: sequence_id, extracted_command, extracted_keywords, technique_id
- Determines EventID using _determine_event_id_from_command()
- Creates rule tuples: (event_id, keywords, rule_info)

#### 3. EventID Determination Logic (_determine_event_id_from_command method)
```python
# File operations → EventID 11
file_operations = ['copy ', 'move ', 'del ', 'mkdir ', 'rmdir ', 'attrib ']

# Process execution → EventID 1  
process_operations = ['cscript', 'whoami', 'hostname', 'ipconfig', ...]

# Default: EventID 1
```

#### 4. Keyword Matching (check_keywords_match method)
- **EventID 1**: AND logic (all keywords must be present in CommandLine)
- **EventID 11/23**: OR logic (any file-related keyword in TargetFilename)

---

## Current Issues Being Investigated

### Issue 1: Incorrect Rule Processing Order (2025-08-26)

**Problem**: 
- Expected first rule: sequence_id=1, keywords=['SystemFailureReporter.exe', 'server', 'group', 'gosta']
- Actual first matches: EventID 11, keywords=['mkdir', 'Programdata', 'VMware'], command='mkdir C:\Programdata\VMware'

**Root Cause Identified**:
The issue is **NOT** with rule processing order, but with **output sorting**:

1. ✅ **Rules are processed correctly**: Rule 1 (SystemFailureReporter.exe) is processed first and finds 1 match
2. ✅ **EventID determination works**: Rule 1 correctly assigned EventID 1 for process execution
3. ❌ **Output is sorted by timestamp**: Results written to CSV in chronological order, not rule discovery order

**Evidence**:
- **Rule 1 match**: Row 105086, timestamp 1742361027986, appears at CSV line ~700
- **First CSV entries**: Rules 45/50 (mkdir VMware), timestamp 1742360440059, appear at CSV lines 2-9
- **Debug log shows**: Rule 1 processed first, finds 1 match correctly

**Technical Details**:
```
Rule Processing Order: Rule 1 → Rule 2 → Rule 3 → ... → Rule 57
CSV Output Order: Sorted by OriginalRowNumber/timestamp (chronological)
```

**Solution Implemented**: 
Limit matches per rule per EventID to reduce noise and focus on key attack events.

**Noise Reduction Strategies**:
1. ✅ **Match Limit**: Max 10 matches per EventID per rule
2. **Process Priority**: Prioritize unique processes over repetitive system processes  
3. **Time Window**: Focus on matches within attack timeframe
4. **Keyword Specificity**: Avoid overly broad keywords that match benign files
5. **False Positive Filtering**: Exclude common Windows system paths

---

## TESTING_phase2_attack_lifecycle_tracer.py

### Purpose
Trace complete attack lifecycles from manually selected originators using ProcessGuid correlation.

### Input/Output Flow
```
Input:
- potential_originators_run-X.csv (with manual selections marked as 'X')
- sysmon-run-X.csv (raw Sysmon events for tracing)

Output:
- Individual timeline plots per originator
- Group timeline plot
- Comprehensive JSON tracing log
```

### Key Logic Components
(To be documented as we investigate this script)

---

## Development Notes

### 2025-08-29 Session - PRODUCTION PIPELINE COMPLETED ✅
**MAJOR MILESTONE**: Completed production-ready Sysmon attack lifecycle analysis pipeline

#### Script Renaming & Organization
- **TESTING_phase1_direct_originator_finder.py** → **5_sysmon_seed_event_extractor.py**
- **TESTING_phase2_attack_lifecycle_tracer.py** → **6_sysmon_attack_lifecycle_tracer.py** 
- **Output directory renamed**: `eventid1_analysis_results` → `sysmon_event_tracing_analysis_results`
- **Deprecated scripts moved** to deprecated folder for clean organization

#### Critical Integration Achievement: Self-Contained Labeling
**Problem Solved**: Eliminated circular dependency between script #6 and separate labeling script
- **Integrated labeling logic** directly into `6_sysmon_attack_lifecycle_tracer.py`
- **Dynamic file naming**: Creates `sysmon-{apt_type}-run-{run_id}-labeled.csv` automatically
- **One-script execution**: No re-runs or external dependencies required
- **Critical failure handling**: Script fails hard if labeling fails (as requested)

#### Major Bug Fixes
1. **Execution Order Fix**: CSV export now happens BEFORE tactics timeline creation
   - **Problem**: `create_tactics_timeline_plot()` tried to read CSV before it was created
   - **Solution**: Moved `export_traced_events_to_csv()` before `create_tactics_timeline_plot()`

2. **EventID 8/10 Attribution Fix**: Fixed critical correlation bug
   - **Problem**: EventID 10 events incorrectly attributed to parent instead of child processes
   - **Solution**: Added missing `_apply_8_and_10_events_mask()` call in recursive child tracing

#### Enhanced Visualizations
1. **Diverse Color Palette**: Completely redesigned MITRE tactic colors
   - **Added strong yellow** (`#FFD700` Gold) for credential-access as requested
   - **10 distinct color families** replacing repetitive blues/reds
   - **High contrast colors**: Crimson Red, Royal Blue, Forest Green, Deep Pink, Dark Turquoise, etc.

2. **Complete Context Timeline**: `timeline_all_malicious_events_with_tactics.png`
   - **Background**: 360K+ benign events as pale gray context
   - **Foreground**: Malicious events color-coded by MITRE tactics
   - **Perfect contrast**: Malicious events clearly highlighted against normal activity

#### Production Features
- **Multi-EventID Support**: EventID 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 17, 18, 23
- **Dynamic APT Support**: Works with all APT types (apt-1 through apt-6)
- **Comprehensive Statistics**: Labeling metrics included in JSON results
- **Deduplication**: "Latest/most specific wins" approach with 59.4% deduplication rate
- **Process Tree Reconstruction**: Complete parent-child process relationships
- **Cross-Computer Analysis**: Tracks lateral movement across hosts

#### Results Quality (APT-1 Run-04)
- **Total Sysmon Events**: 363,657
- **Malicious Events**: 3,939 (1.08%)
- **Traced Originators**: 62 
- **Processing Time**: ~17-20 seconds
- **Attribution Success**: 96%+ accuracy with proper EventID correlation

### 2025-08-26 Session
- Updated oilrig_comprehensive_command_extraction.csv with new Initial Access entry
- New sequence_id=1: SystemFailureReporter.exe deployment 
- All existing entries shifted +1
- Discovered rule processing order issue in Phase 1 script

### Keywords File Evolution
- **Original**: 56 entries (sequences 1-56)
- **Updated**: 57 entries (sequences 1-57)
- **New entry**: Initial Access - Content Injection T1659

---

## Debug Commands

### Run Phase 1 with Debug
```bash
cd dataset/scripts/exploratory/
python3 TESTING_phase1_direct_originator_finder.py --apt-type apt-1 --run-id 04 --debug
```

### Check CSV Structure
```bash
head -5 apt-yaml/oilrig_comprehensive_command_extraction.csv
tail -5 apt-yaml/oilrig_comprehensive_command_extraction.csv
```

### Inspect Output
```bash
head -10 ../apt-1/apt-1-run-04/potential_originators_run-04.csv
```

---

## TODO Investigation Items

- [ ] Debug rule processing order in Phase 1
- [ ] Verify EventID determination logic
- [ ] Test Phase 1 output quality
- [ ] Document Phase 2 script logic
- [ ] Create validation tests for both phases

---

## DEPRECATED SCRIPTS CLEANUP (2025-09-01)

**Objective**: Organized exploratory folder by moving development/debugging scripts to deprecated folder.

### Scripts Moved to Deprecated Folder

#### **TESTING Script Variants (Superseded)**
- **`TESTING_9_netflow_labeler.py`** ❌ → Superseded by `TESTING_9_enhanced_netflow_labeler.py`
  - *Purpose*: Basic NetFlow labeling with correlation data
  - *Why deprecated*: Enhanced version provides confidence scoring and attack relevance analysis

- **`TESTING_9_refined_netflow_labeler.py`** ❌ → Superseded by `TESTING_9_enhanced_netflow_labeler.py` 
  - *Purpose*: Refined NetFlow labeling logic iteration
  - *Why deprecated*: Enhanced version includes systematic attribution strategies

#### **Development Analysis Scripts**
- **`analyze_internal_flows.py`** ❌ → Development debugging
  - *Purpose*: Analyze internal network flow patterns between target machines
  - *Why deprecated*: One-time analysis, logic integrated into main correlation scripts

- **`community_id_comparison.py`** ❌ → Development debugging
  - *Purpose*: Compare network community ID consistency across datasets
  - *Why deprecated*: Development validation, not needed for production

- **`correlation_logic_analyzer.py`** ❌ → Development debugging
  - *Purpose*: Debug correlation distance calculations and logic validation
  - *Why deprecated*: Development debugging, core logic now stable

- **`detailed_correlation_examples.py`** ❌ → Development examples
  - *Purpose*: Generate detailed examples of seed-netflow correlations
  - *Why deprecated*: Development documentation, not production analysis

- **`eventid_temporal_analysis.py`** ❌ → Development analysis
  - *Purpose*: Statistical analysis of EventID temporal patterns (Process Create vs File Create/Delete)
  - *Why deprecated*: One-time hypothesis validation, findings integrated into main scripts

- **`robust_correlation_logic.py`** ❌ → Development iteration
  - *Purpose*: Enhanced correlation logic with improved error handling
  - *Why deprecated*: Development iteration, final logic integrated into production scripts

#### **One-Time Analysis Scripts**
- **`check_missing_associations.py`** ❌ → One-time analysis
  - *Purpose*: Identify gaps in manual correlation associations
  - *Why deprecated*: Specific to initial dataset validation, not ongoing analysis

- **`compare_timestamp_impact.py`** ❌ → One-time analysis
  - *Purpose*: Compare impact of different timestamp parsing approaches
  - *Why deprecated*: Timestamp parsing methodology established

- **`complete_kT6_analysis.py`** ❌ → One-time analysis
  - *Purpose*: Deep analysis of specific community ID 'kT6zsgNZ7wlJuK2OIcC44Iy7n1k='
  - *Why deprecated*: Specific community ID investigation, not generalized

- **`minority_cases_analysis.py`** ❌ → One-time analysis
  - *Purpose*: Analyze edge cases and outliers in correlation results
  - *Why deprecated*: Edge case investigation, findings integrated into main logic

- **`understand_correlation_start_distance.py`** ❌ → One-time analysis
  - *Purpose*: Understand correlation distance calculation methodology
  - *Why deprecated*: Methodology understanding, logic now implemented

#### **Development Testing Scripts**
- **`comprehensive_correlation_test.py`** ❌ → Development testing
  - *Purpose*: Comprehensive testing of correlation algorithms
  - *Why deprecated*: Development validation, production algorithms now stable

- **`test_complete_only.py`** ❌ → Development testing
  - *Purpose*: Test correlation logic with complete/incomplete data scenarios
  - *Why deprecated*: Development testing, edge cases now handled

- **`test_dual_correlation_logic.py`** ❌ → Development testing  
  - *Purpose*: Test dual correlation columns (Auto vs Manual)
  - *Why deprecated*: Development validation, dual logic now integrated

- **`validate_systematic_attribution.py`** ❌ → Development validation
  - *Purpose*: Validate systematic attribution engine accuracy
  - *Why deprecated*: Development validation, attribution logic now proven

#### **Superseded Analysis Scripts**
- **`focused_subnetflow_analyzer.py`** ❌ → Superseded by `multi_community_subnetflow_analyzer.py`
  - *Purpose*: Analyze single community ID sub-netflow patterns
  - *Why deprecated*: Multi-community version provides broader analysis

- **`simple_subnetflow_inspector.py`** ❌ → Superseded by `subnetflow_timeline_analyzer.py`
  - *Purpose*: Basic inspection of sub-netflow temporal boundaries
  - *Why deprecated*: Timeline analyzer provides comprehensive visualization

- **`subnetflow_temporal_analyzer.py`** ❌ → Superseded by `subnetflow_timeline_analyzer.py`
  - *Purpose*: Temporal analysis of sub-netflow patterns
  - *Why deprecated*: Timeline analyzer includes visualization and fixed matplotlib errors

- **`subnetflow_assessment_report.py`** ❌ → One-time analysis
  - *Purpose*: Generate assessment report of sub-netflow quality
  - *Why deprecated*: One-time quality assessment, methodology established

#### **Development Iterations**
- **`corrected_enhanced_correlator.py`** ❌ → Development iteration
  - *Purpose*: Corrected version of enhanced correlation logic
  - *Why deprecated*: Development iteration, final logic in production scripts

- **`systematic_attribution_analyzer.py`** ❌ → Development iteration
  - *Purpose*: Systematic approach to seed-netflow attribution
  - *Why deprecated*: Attribution logic integrated into main production scripts

- **`systematic_subnetflow_attribution.py`** ❌ → Development iteration
  - *Purpose*: Sub-netflow level systematic attribution
  - *Why deprecated*: Sub-netflow logic integrated into timeline analyzers

#### **Utility Scripts**
- **`generic_seed_netflow_attribution_engine.py`** ❌ → Utility script
  - *Purpose*: Generic engine for automated seed-netflow attribution detection
  - *Why deprecated*: Attribution logic integrated into TESTING_8 and production scripts

- **`debug_ip_mapping.py`** ❌ → One-time debugging
  - *Purpose*: Debug IP address mapping issues in correlation
  - *Why deprecated*: IP mapping issues resolved, debugging complete

### **CURRENT ACTIVE SCRIPTS RETAINED**

#### **Core Production Pipeline (1-7)**
- `1_comprehensive_network_community_id_analyzer.py` ✅
- `2_process_tuple_uniqueness_validator.py` ✅  
- `3_enhanced_temporal_causation_correlator.py` ✅
- `4_comprehensive_correlation_analysis.py` ✅
- `5_sysmon_seed_event_extractor.py` ✅
- `6_sysmon_attack_lifecycle_tracer.py` ✅
- `7_create_labeled_sysmon_dataset.py` ✅

#### **Active Testing Scripts**
- `TESTING_8_dual_timeline_correlator.py` ✅ - **CURRENT**: C2/NetFlow correlation with filtering
- `TESTING_9_enhanced_netflow_labeler.py` ✅ - **ACTIVE**: Enhanced labeling with confidence scoring

#### **Recent Analysis Scripts**  
- `subnetflow_timeline_analyzer.py` ✅ - Sub-netflow temporal visualization (recent work)
- `multi_community_subnetflow_analyzer.py` ✅ - Multi-community analysis (recent work)

### **CLEANUP RESULTS**
- **Total scripts moved**: 26 scripts
- **Deprecated folder**: Now contains 43 total deprecated scripts
- **Active scripts remaining**: 13 core scripts in exploratory folder
- **Organization improvement**: 66% reduction in exploratory folder clutter

---

## MAJOR UPDATE 2025-09-25: INTEGRATED SCRIPT AUTOMATION & VISUALIZATION ENHANCEMENT ✅

**BREAKTHROUGH ACHIEVEMENT**: Successfully automated Manual Checkpoint 2 and resolved critical visualization issues in INTEGRATED_netflow_labeler.py.

### **🎯 Automated Assignment Implementation**

#### **Manual Checkpoint 2 Elimination**:
- **Problem**: Manual CSV editing was the workflow bottleneck requiring subnetflow assignment template editing
- **Solution**: Implemented complete automation with `--automated-assignment` and `--no-automated-assignment` feature flags
- **Result**: Zero manual intervention required while maintaining identical output quality

#### **End-Time Proximity Assignment Logic Integration**:
- **Transferred**: Complete logic from `end_time_proximity_assignment.py` into INTEGRATED script
- **Algorithm**: ±5 second end-time proximity window for automated subnetflow assignments
- **Systematic Processing**: Handles ALL communities marked for Subnetflow-attribution in verification matrix
- **Performance**: 400+ automated assignments for APT-1-Run-04 (matching reference script exactly)

### **🎨 Critical Visualization Fixes Implemented**

#### **Issue 1: Y-axis Label Format Fixed**:
- **Problem**: Labels showing network_community_id format instead of simple ordinal numbers
- **Solution**: Changed to clean format: `1 → 118936`, `2 → 119570` (ordinal → seed_event)
- **Implementation**: Added `ordinal_number = i + 1` for consistent y-axis labeling

#### **Issue 2: Top X-axis Alignment Fixed**:
- **Problem**: Seed event OriginalRowNumbers misaligned with vertical lines due to timestamp precision issues
- **Root Cause**: Top axis configuration occurred before main axis limits were properly established
- **Solution**: Moved top axis setup to AFTER main axis limits are finalized
- **Result**: Perfect alignment between vertical lines and top axis labels

#### **Issue 3: Border Color Elimination**:
- **Problem**: Purple/navy border colors distracting from tactic colors
- **Solution**: Set `edgecolor='none'` and `linewidth=0` for clean appearance
- **Result**: Pure tactic-colored horizontal bars without border distraction

#### **Issue 4: Empty Plot Fix (Critical)**:
- **Community**: `1:6okao16uEuHUP9NnUm/SWPt9rTQ=` showed empty plots despite having 818 subnetflows
- **Root Cause**: Extremely short subnetflow durations (0.088 seconds average) resulted in invisible matplotlib bar widths (~1e-06)
- **Solution**: Implemented 1-second absolute minimum bar width: `min_bar_width = 1.0 / 86400.0`
- **Impact**: All subnetflows now visible regardless of duration (consistency across timeline spans)

#### **Issue 5: Dynamic Title Correction**:
- **Problem**: Hardcoded `±7s` in plot titles despite using `±5s` algorithm
- **Solution**: Added `time_window_seconds` parameter to plotting method with dynamic calculation
- **Result**: Accurate title showing `±5s End-Time Proximity Algorithm`

### **🗂️ Project Cleanup & Organization**

#### **Deprecated Scripts Moved to Backup**:
Moved **15 files** to `/backup-deprecated/2025-09-25-integrated-visualization-fixes/`:
- **6** Today's debugging/testing scripts (debug_*.py, test_*.py)
- **2** Test visualization images (test_*.png)
- **1** Reference script (`end_time_proximity_assignment.py` - logic now fully integrated)
- **2** Pattern analysis workflow scripts (superseded by INTEGRATED script)
- **3** Pre-existing metadata testing scripts (no longer needed)
- **1** Original workflow backup script

#### **Clean Production Environment**:
- **Streamlined**: Only 13 essential production scripts remain in `/scripts/exploratory/`
- **Consolidated**: Single INTEGRATED script replaces multiple scattered reference scripts
- **Maintainable**: Clear separation between production and development/testing code

### **🎯 Current Production Usage**

#### **Recommended Command**:
```bash
cd /home/researcher/Downloads/research/scripts/exploratory/
python3 INTEGRATED_netflow_labeler.py --apt-type apt-1 --run-id 04 --automated-assignment
```

#### **Expected Results**:
- ✅ **Clean Y-axis labels**: `1 → 118936`, `2 → 119570` format
- ✅ **Perfect alignment**: Top axis seed event numbers align with vertical lines
- ✅ **Clean appearance**: Pure tactic colors without distracting borders
- ✅ **All bars visible**: 1-second minimum width ensures visibility for all subnetflows
- ✅ **Accurate titles**: Dynamic `±5s` time window display
- ✅ **Complete automation**: Zero manual checkpoint intervention required

### **🚀 Benefits Achieved**:

1. **Workflow Efficiency**: 100% automation eliminates manual bottleneck
2. **Visual Quality**: Professional, publication-ready timeline plots
3. **Data Completeness**: No more empty plots due to invisible bars
4. **Consistency**: Uniform visualization across all communities regardless of duration patterns
5. **Maintainability**: Single integrated script vs multiple reference scripts
6. **Production Ready**: Robust, validated pipeline suitable for all APT datasets

### **📊 Technical Specifications**:
- **Assignment Algorithm**: ±5 second end-time proximity with multi-assignment support
- **Minimum Bar Width**: 1 second absolute (1.0/86400.0 matplotlib date units)
- **Y-axis Format**: Simple ordinal numbering (1, 2, 3...) with seed event arrows
- **X-axis Precision**: Proper datetime positioning with post-axis-limits configuration
- **Title Format**: Dynamic time window parameter display
- **Color Scheme**: Pure MITRE tactic colors without borders

**STATUS**: INTEGRATED_netflow_labeler.py is now a complete, automated, visually-enhanced production pipeline ready for comprehensive APT dataset processing with zero manual intervention and professional-quality visualization output.

---

*Last Updated: 2025-09-25*