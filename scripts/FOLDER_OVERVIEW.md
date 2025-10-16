# SCRIPTS FOLDER OVERVIEW - Processing Engine Documentation

**Folder Mission**: Complete processing engine for data exploration, dataset generation, labeling, and ML analysis

**Project Bible Branch**: Level 2 documentation for /scripts/ folder workflows and interdependencies  
**Parent Document**: `/PROJECT_OVERVIEW.md` (Root-level project bible)

---

## 🎯 **FOLDER PURPOSE & MISSION**

### **Primary Role**: Project Processing Engine
- **Data Flow Direction**: `/dataset/dataset-backup/` (input) → **Processing Engine** → `/dataset/apt-Y/` + `/analysis/` + `/artifacts/` (outputs)
- **Core Function**: Transform primitive JSONL files into labeled dual-domain datasets and ML analysis products
- **Mission Statement**: Complete automation of data exploration → dataset generation → labeling → machine learning analysis

### **Key Responsibilities**:
1. **Data Exploration**: Analyze primitive JSONL data structure and quality
2. **Dataset Generation**: Create structured CSV datasets from raw telemetry  
3. **Dual-Domain Labeling**: NetFlow-Sysmon correlation and attack attribution
4. **ML Analysis**: Pattern recognition, model training, and evaluation
5. **Pipeline Orchestration**: End-to-end workflow automation

---

## 📁 **FOLDER STRUCTURE & ORGANIZATION**

### **Core Processing Subdirectories**:

#### **`/scripts/exploratory/` - Active Development Workspace**
- **Purpose**: Primary development and production environment for dataset labeling
- **Status**: **ACTIVE PRODUCTION ENVIRONMENT**
- **Key Scripts**: Essential labeling pipeline (Scripts 5, 6, 7, INTEGRATED)
- **Current Focus**: Dual-domain correlation analysis and dataset generation

#### **`/scripts/pipeline/` - Automated Processing Workflows**
- **Purpose**: Batch processing and automation scripts
- **Function**: Large-scale dataset processing across all APT runs
- **Integration**: Works with `/scripts/exploratory/` outputs

#### **`/scripts/modeling/` - Machine Learning Analysis**
- **Purpose**: ML pattern recognition and model training workflows
- **Inputs**: Labeled datasets from `/scripts/exploratory/`
- **Outputs**: Model artifacts → `/artifacts/`, analysis results → `/analysis/`

#### **`/scripts/utils/` - Shared Utilities & Libraries**
- **Purpose**: Common functions, configuration management, path utilities
- **Components**: `apt_config.py`, `apt_plotting_utils.py`, `apt_path_utils.py`, `apt_workflow_manager.py`
- **Integration**: Shared across all script subdirectories

---

## 🔄 **ESSENTIAL LABELING WORKFLOW**

### **Primary Processing Pipeline**: `/scripts/exploratory/`

**Core Scripts (Path-Updated for New Structure)**:

#### **Script 5**: `5_sysmon_seed_event_extractor.py`
- **Purpose**: Extract manually verified seed events from Sysmon data
- **Input**: Raw Sysmon CSV files from `/dataset/apt-Y/apt-Y-run-XX/`
- **Output**: Curated seed events with MITRE ATT&CK tactic mapping
- **Dependencies**: Manual expert verification of attack events

#### **Script 6**: `6_sysmon_attack_lifecycle_tracer.py`
- **Purpose**: Trace complete attack lifecycle and generate timeline visualizations
- **Input**: Seed events from Script 5
- **Output**: Attack progression plots, lifecycle analysis
- **Manual Step**: Requires expert review of attack sequence

#### **Script 7**: `7_create_labeled_sysmon_dataset.py`
- **Purpose**: Generate final labeled Sysmon dataset with attack attribution
- **Input**: Verified attack lifecycle from Script 6
- **Output**: Production-ready labeled Sysmon dataset
- **Integration**: Feeds into ML analysis workflows

#### **INTEGRATED**: `INTEGRATED_netflow_labeler.py`
- **Purpose**: Unified NetFlow-Sysmon correlation pipeline with interactive checkpoints
- **Innovation**: Single script replacing fragmented TESTING scripts 8-9-10
- **Features**: 
  - Interactive manual checkpoints with resume capability
  - Enhanced 30-second temporal correlation windows
  - Multi-figure PNG splitting to prevent visualization corruption
  - Workflow state management with JSON progress tracking
- **Performance**: 40% time reduction through data caching and selective processing

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

### **Enhanced Correlation Logic (INTEGRATED Script)**:

**Three-Criteria Temporal Correlation**:
```python
# Criterion 1: Seed event falls INSIDE NetFlow timespan
if flow_start <= seed_timestamp <= flow_end:
    include_flow = True
    time_diff_seconds = 0.0

# Criterion 2: NetFlow is on the LEFT of seed event  
elif seed_timestamp > flow_end and (seed_timestamp - flow_end) <= correlation_threshold:
    include_flow = True
    time_diff_seconds = abs((seed_timestamp - flow_end).total_seconds())

# Criterion 3: NetFlow is on the RIGHT of seed event
elif seed_timestamp < flow_start and (flow_start - seed_timestamp) <= correlation_threshold:
    include_flow = True  
    time_diff_seconds = abs((flow_start - seed_timestamp).total_seconds())
```

**Correlation Window**: 30 seconds (±30s around seed events)
**Individual Seed Event Analysis**: One subplot per seed event (vs merged windows)

### **Multi-Figure PNG Generation**:
- **Problem**: Oversized PNG dimensions (27,325 x 6,705 pixels) causing corruption
- **Solution**: Multi-figure splitting with maximum 36 subplots per figure (6x6 grid)
- **Result**: Manageable PNG files (8,516 x 8,381 and 8,515 x 7,237 pixels)

### **Path Architecture (Post-Reorganization)**:
```python
# Standard path construction pattern for all scripts
scripts_dir = Path(__file__).parent.parent  # Go up to scripts/
project_root = scripts_dir.parent  # Go to research/  
dataset_dir = project_root / "dataset"  # Point to dataset folder
```

---

## 📊 **DATA FLOW ARCHITECTURE**

### **Input Dependencies**:
- **Raw JSONL Data**: `/dataset/dataset-backup/` (primitive Elasticsearch exports)
- **Structured CSV Data**: `/dataset/apt-Y/apt-Y-run-XX/` (processed datasets)
- **Configuration**: Script-specific parameters and feature flags

### **Processing Stages**:
1. **Data Exploration** → Structure analysis, quality assessment
2. **CSV Generation** → JSONL to structured CSV conversion
3. **Seed Event Extraction** → Manual expert verification of attack events
4. **Attack Lifecycle Tracing** → Temporal attack progression analysis
5. **Dual-Domain Correlation** → NetFlow-Sysmon attribution analysis
6. **Dataset Labeling** → Production-ready labeled datasets
7. **ML Analysis** → Pattern recognition and model training

### **Output Destinations**:
- **Labeled Datasets**: `/dataset/apt-Y/apt-Y-run-XX/` (dual-domain datasets)
- **Analysis Results**: `/analysis/` (correlation analysis, pattern discoveries)
- **ML Products**: `/artifacts/` (trained models, evaluation metrics)
- **Visualizations**: PNG plots, timeline analysis, correlation hotspots

---

## 🚀 **WORKFLOW ORCHESTRATION**

### **Interactive Workflow Pattern**:
```bash
# Complete dual-domain labeling workflow
python3 scripts/exploratory/INTEGRATED_netflow_labeler.py --apt-type apt-1 --run-id 04

# Resume from interruption
python3 scripts/exploratory/INTEGRATED_netflow_labeler.py --resume --apt-type apt-1 --run-id 04

# Individual script execution
python3 scripts/exploratory/5_sysmon_seed_event_extractor.py --apt-type apt-1 --run-id 04
python3 scripts/exploratory/6_sysmon_attack_lifecycle_tracer.py --apt-type apt-1 --run-id 04  
python3 scripts/exploratory/7_create_labeled_sysmon_dataset.py --apt-type apt-1 --run-id 04
```

### **Batch Processing Commands**:
```bash
# Server-ready batch processing
bash scripts/pipeline/scripts_batch_process_all_apt_runs.sh

# Enhanced correlation analysis
python3 scripts/pipeline/scripts_batch_correlation_analyzer.py --run-id 05 --apt-type apt-1
```

### **Automated Workflow Management**:
```bash
# Complete automated workflow (Scripts 6 → Manual → Script 7)
python3 scripts/exploratory/apt_workflow_manager.py --apt-type apt-1 --run-id 04

# Skip Script 6 if outputs exist
python3 scripts/exploratory/apt_workflow_manager.py --apt-type apt-1 --run-id 04 --skip-script6
```

---

## 🔗 **INTERDEPENDENCIES & INTEGRATION**

### **Folder Integration Patterns**:

#### **Scripts → Dataset** (Primary Output):
- **Function**: Processing engine generates labeled dual-domain datasets
- **Output Location**: `/dataset/apt-Y/apt-Y-run-XX/`
- **Data Types**: CSV datasets, correlation matrices, timeline data

#### **Dataset → Scripts** (Feedback Loop):
- **Function**: Labeled datasets feed ML analysis workflows  
- **Input Source**: `/dataset/apt-Y/` (processed datasets)
- **Analysis**: Pattern recognition, model training, evaluation

#### **Scripts → Analysis** (Research Output):
- **Function**: Analysis workflows output pattern discoveries and results
- **Output Location**: `/analysis/`
- **Content**: Correlation analysis, statistical results, research findings

#### **Scripts → Artifacts** (ML Output):
- **Function**: ML workflows produce model artifacts and ML products
- **Output Location**: `/artifacts/`
- **Content**: Trained models, evaluation metrics, comparative analysis

#### **Other-datasets → Scripts** (Baseline):
- **Function**: External datasets provide baseline comparisons
- **Integration**: Comparative analysis workflows
- **Purpose**: Benchmarking and validation

---

## 🎯 **CURRENT STATE & RECENT ACHIEVEMENTS**

### **2025-09-10 Status**:
- ✅ **Enhanced Correlation Logic**: Individual seed event analysis with 30-second windows
- ✅ **PNG Generation Fixes**: Multi-figure splitting prevents visualization corruption  
- ✅ **Path Updates Complete**: All 4 essential scripts updated for new folder structure
- ✅ **Production-Ready Pipeline**: Batch processing system validated
- ✅ **Integrated Workflow**: INTEGRATED_netflow_labeler.py replaces fragmented approach

### **Performance Metrics**:
- **Processing Time**: 2-3 minutes per dataset (individual analysis)
- **Correlation Rates**: 56-85% attribution rates (varies by dataset quality)
- **Batch Capacity**: ~50 datasets in 2-4 hours (server deployment ready)
- **PNG Quality**: High-resolution plots without corruption

### **Technical Innovations**:
- **Three-Criteria Correlation**: Inside/Left/Right positioning analysis
- **Workflow State Management**: JSON-based progress tracking with resume capability
- **Interactive Checkpoints**: Seamless pause/resume for manual verification steps
- **Data Caching**: 40% performance improvement through single-load architecture

---

## 📋 **SESSION RECOVERY FOR /scripts/**

### **Essential Commands**:
```bash
# 1. Verify current working environment
python3 scripts/exploratory/INTEGRATED_netflow_labeler.py --help

# 2. Check recent processing results  
ls -la scripts/exploratory/correlation_analysis_results/

# 3. Validate essential script paths
python3 scripts/exploratory/5_sysmon_seed_event_extractor.py --help
python3 scripts/exploratory/6_sysmon_attack_lifecycle_tracer.py --help
python3 scripts/exploratory/7_create_labeled_sysmon_dataset.py --help

# 4. Load technical implementation details
Read /home/researcher/Downloads/research/scripts/exploratory/NETFLOW_SYSMON_ATTRIBUTION_DOCUMENTATION.md --offset 842
```

### **Workflow Validation**:
```bash
# Test complete pipeline on known dataset
python3 scripts/exploratory/INTEGRATED_netflow_labeler.py --apt-type apt-1 --run-id 04

# Verify batch processing capability
ls -la scripts/pipeline/
```

---

## 🔄 **EVOLUTION & TECHNICAL DEBT**

### **Script Generation History**:
1. **Original TESTING Scripts**: Individual scripts 8-9-10 for NetFlow labeling
2. **Fragmented Workflow**: Manual file editing between each script
3. **Integration Phase**: INTEGRATED_netflow_labeler.py consolidation
4. **Enhanced Correlation**: Three-criteria temporal analysis implementation
5. **Path Modernization**: Folder restructuring and path updates
6. **Current State**: Production-ready integrated pipeline

### **Backup Management**:
- **Original Scripts Preserved**: TESTING_*_backup_YYYYMMDD_HHMMSS.py pattern
- **Feature Flags**: Easy rollback capability via configuration
- **Modular Architecture**: Shared utilities support with graceful degradation

## 🔧 **KEY TECHNICAL INNOVATIONS**

### **Critical Algorithm Improvements**:

#### **EventID 8/10 Correlation Fix**:
- **Problem Solved**: ProcessAccess events were incorrectly attributed to parent processes instead of specific child processes
- **Technical Solution**: Advanced logic for ProcessAccess event attribution to correct child processes that spawned them
- **Impact**: Dramatically improved attack lifecycle accuracy and correlation precision
- **Implementation**: Enhanced process tree reconstruction with precise parent-child mapping

#### **Event Deduplication Logic**:
- **Approach**: "Latest/most specific wins" strategy for duplicate events
- **Purpose**: Accurate event counting and prevention of false metric inflation
- **Implementation**: Sophisticated deduplication across multiple EventIDs with confidence scoring
- **Critical for ML**: Ensures labeled datasets maintain accurate event-to-label ratios

#### **ProcessGuid-Based Correlation Technology**:
- **Method**: Complete parent-child process relationship reconstruction via ProcessGuid linking
- **Scope**: Recursive tracing of complete process spawning chains across multiple computers
- **Integration**: Cross-EventID correlation linking file operations to originating processes
- **Innovation**: Multi-computer lateral movement detection with process attribution

#### **Three-Criteria Temporal Correlation**:
- **Enhanced Logic**: Inside/Left/Right positioning analysis for NetFlow-Sysmon correlation
- **Precision**: 30-second correlation windows with millisecond-level accuracy
- **Performance**: +9.7% improvement in correlation rates through comprehensive flow analysis
- **Technical Breakthrough**: Individual seed event analysis vs merged window approach

#### **Multi-Figure PNG Generation**:
- **Problem Solved**: Oversized PNG dimensions (27,325 x 6,705 pixels) causing visualization corruption
- **Solution**: Multi-figure splitting with maximum 36 subplots per figure (6x6 grid)
- **Result**: Manageable PNG files preventing "chessboard" display corruption
- **Implementation**: Smart grid calculation and automatic figure splitting algorithms

### **Advanced Technical Reference**:
For comprehensive implementation details, algorithm specifics, and correlation logic documentation, see:
- **📋 Level 3 Technical Details**: `/scripts/exploratory/SYSMON_ATTACK_LIFECYCLE_PIPELINE_DOCUMENTATION.md`
- **🔍 NetFlow Attribution Documentation**: `/scripts/exploratory/NETFLOW_SYSMON_ATTRIBUTION_DOCUMENTATION.md`

---

### **Future Enhancements**:
- **ML Integration**: Advanced pattern recognition workflows
- **Automated Quality Control**: Statistical validation of correlation accuracy
- **Performance Optimization**: Further batch processing improvements
- **Advanced Visualization**: Interactive correlation analysis tools

## 🔄 **EVOLUTION & TECHNICAL DEBT**

### **Script Generation History**:
1. **Original TESTING Scripts**: Individual scripts 8-9-10 for NetFlow labeling
2. **Fragmented Workflow**: Manual file editing between each script
3. **Integration Phase**: INTEGRATED_netflow_labeler.py consolidation
4. **Enhanced Correlation**: Three-criteria temporal analysis implementation
5. **Path Modernization**: Folder restructuring and path updates
6. **Current State**: Production-ready integrated pipeline

### **Backup Management**:
- **Original Scripts Preserved**: TESTING_*_backup_YYYYMMDD_HHMMSS.py pattern
- **Feature Flags**: Easy rollback capability via configuration
- **Modular Architecture**: Shared utilities support with graceful degradation

---

**Last Updated**: 2025-09-11  
**Status**: Production-ready processing engine with enhanced correlation logic and comprehensive technical innovation documentation  
**Critical Dependencies**: PROJECT_OVERVIEW.md (parent documentation)  
**Next Integration**: ML analysis workflows with labeled dataset inputs

---

**Navigation**: 
- **⬆️ Parent**: `/PROJECT_OVERVIEW.md` (Complete project understanding)
- **🔗 Related**: `/dataset/FOLDER_OVERVIEW.md` (Data storage), `/analysis/FOLDER_OVERVIEW.md` (Results), `/artifacts/FOLDER_OVERVIEW.md` (ML products)
- **📋 Technical Details**: `/scripts/exploratory/SYSMON_ATTACK_LIFECYCLE_PIPELINE_DOCUMENTATION.md` (Level 3 implementation specifics)