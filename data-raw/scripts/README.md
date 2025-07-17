# APT Cybersecurity Analysis Pipeline

A comprehensive, production-ready cybersecurity data processing pipeline for analyzing Advanced Persistent Threat (APT) simulations. This system processes Windows event logs (Sysmon), network traffic flows, and Caldera attack simulation reports to enable detailed security analysis and correlation.

## 📋 Overview

This pipeline is designed to handle large-scale cybersecurity data processing across multiple APT simulation runs. It provides automated data extraction, transformation, analysis, and visualization capabilities with robust error handling and batch processing support.

### Key Capabilities
- **Multi-modal Data Processing**: Sysmon events, network traffic, and attack simulation logs
- **Scalable Architecture**: Handles GB-scale datasets across multiple APT runs
- **Automated Correlation**: Correlates attack events with system telemetry data
- **Batch Processing**: Full automation across multiple simulation runs
- **Professional Quality**: Production-ready with comprehensive error handling and logging

## 🏗️ Architecture

### Directory Structure
```
scripts/
├── pipeline/           # Core processing scripts (#1-#6)
├── batch/              # Batch processing automation
├── analysis/           # Analysis and correlation tools  
├── config/             # Configuration files
├── dev/                # Development notebooks
├── exploratory/        # Data exploration notebooks
└── tools/              # Utility scripts
```

### Data Flow Pipeline
```
┌─ Elasticsearch ─┐    ┌─ Caldera Simulation ─┐
│   Raw Events    │    │    Attack Logs       │
└─────────────────┘    └────────────────────────┘
         │                         │
         ▼                         ▼
    Script #1               Script #4
 (JSONL Download)      (Report Transform)
         │                         │
         ▼                         │
   *.jsonl files                   │
         │                         │
    ┌────┴─────┐                   │
    ▼          ▼                   │
Script #2   Script #3              │
(Sysmon)   (Network)               │
    │          │                   │
    ▼          ▼                   │
 *.csv     *.csv                   │
    │          │                   │
    └────┬─────┘                   │
         ▼                         │
      Script #5 ◄──────────────────┘
   (Event Analysis)
         │
         ▼
    Script #6
  (Visualization)
```

## 🔧 Core Pipeline Scripts

### Script #1: Elasticsearch Data Downloader
**File**: `pipeline/1_elastic_index_downloader.py`

Downloads cybersecurity data from Elasticsearch clusters with interactive index selection.

```bash
python3 1_elastic_index_downloader.py [--output-dir DIR]
```

**Features**:
- Secure authentication to Elasticsearch clusters
- Interactive index and time range selection
- Support for sysmon and network_traffic data types
- Progress tracking for large downloads

**Input**: Elasticsearch connection (interactive)  
**Output**: Raw JSONL files (`*sysmon*.jsonl`, `*network_traffic*.jsonl`)

---

### Script #2: Sysmon CSV Creator
**File**: `pipeline/2_sysmon_csv_creator.py`

Converts Windows Sysmon JSONL data into structured CSV format for analysis.

```bash
# APT directory mode (recommended)
python3 2_sysmon_csv_creator.py --apt-dir ../../apt-X/apt-X-run-XX

# Direct file mode
python3 2_sysmon_csv_creator.py --input file.jsonl --output file.csv

# Batch processing mode
python3 2_sysmon_csv_creator.py --apt-dir DIR --no-validate
```

**Features**:
- Schema-based parsing for all Sysmon EventIDs (1-25)
- Robust XML handling with error recovery
- Auto-detection of input/output files via config.yaml
- Data type optimization for ML pipelines
- Comprehensive validation and backup system

**Input**: Sysmon JSONL files (typically 1-3GB)  
**Output**: Structured CSV with 18+ standardized columns  
**Processing Time**: ~10 minutes for 2.8GB input

---

### Script #3: Network Traffic CSV Creator  
**File**: `pipeline/3_network_traffic_csv_creator.py`

Transforms network traffic flow JSONL data into structured CSV format.

```bash
# APT directory mode (recommended)
python3 3_network_traffic_csv_creator.py --apt-dir ../../apt-X/apt-X-run-XX

# Skip exploratory analysis for speed
python3 3_network_traffic_csv_creator.py --apt-dir DIR --no-analysis
```

**Features**:
- Processes 42 mapped network flow fields
- Port field integer conversion (fixes Elasticsearch float issues)
- Memory-efficient processing for large datasets
- Comprehensive exploratory data analysis
- Temporal, destination, source, and process field extraction

**Input**: Network traffic JSONL files (typically 1-5GB)  
**Output**: Structured CSV with network flow data  
**Processing Time**: ~1.5 minutes for 2.3GB input

---

### Script #4: Universal Caldera Report Transformer
**File**: `pipeline/4_universal_caldera_report_transformer.py`

Transforms original Caldera JSON reports into standardized extracted information format.

```bash
# APT directory mode (uses config.yaml)
python3 4_universal_caldera_report_transformer.py --apt-dir ../../apt-X/apt-X-run-XX

# Legacy file mode
python3 4_universal_caldera_report_transformer.py input_file.json
```

**Features**:
- 7 specialized transformation functions for different command types
- Universal patterns for webshell, xdotool, exec-background commands
- Hardcoded mapping for complex commands (VMware/Chrome variants)
- Automatic input/output detection via config.yaml
- Preserves command interpretation logic for security analysis

**Input**: Original Caldera JSON reports (`*event-logs.json`)  
**Output**: Extracted information JSON (`*extracted_information.json`)  
**Processing Time**: 10-30 seconds per file

---

### Script #5: Sysmon Event Analysis (Main Correlation Engine)
**File**: `pipeline/5_sysmon_event_analysis.py`

Correlates Caldera attack entries with Sysmon events using advanced time-window analysis.

```bash
# APT directory mode
python3 5_sysmon_event_analysis.py --apt-dir ../../apt-X/apt-X-run-XX

# Batch processing mode
python3 5_sysmon_event_analysis.py --apt-dir DIR --batch-mode
```

**Features**:
- Multi-threaded event detection system
- 3 detection masks: child_process, child_processcreate, eventid_8_or_10
- Time window-based correlation (180s primary, 300s fallback)
- Memory optimization for large datasets
- Comprehensive logging and performance monitoring
- Individual plot generation for each detected entry

**Input**: Sysmon CSV + Caldera extracted JSON + entry config CSV  
**Output**: Analysis reports, individual plots, labeled datasets  
**Success Rate**: Typically 90-95% correlation success

---

### Script #6: Timeline Visualization
**File**: `pipeline/6_simple_timeline_plotter.py`

Creates timeline visualizations showing attack progression organized by computer.

```bash
python3 6_simple_timeline_plotter.py
```

**Features**:
- Computer-based event organization
- Professional timeline visualization
- Integration with Script #5 results

**Input**: Labeled Sysmon CSV files  
**Output**: Timeline plots (`timeline_*.png`)

## 🚀 Batch Processing Scripts

### Unified JSONL Processor
**File**: `batch/unified_jsonl_processor.sh`

Comprehensive batch processor for Scripts #2 and #3 across all APT runs.

```bash
cd scripts/batch
./unified_jsonl_processor.sh
```

**Features**:
- Interactive menu with 16 processing options
- Smart skip existing CSV functionality for resumable processing
- Separate processing modes: Sysmon only, Network only, or Both
- Comprehensive progress tracking and error logging
- Environment validation and dependency checking
- Colored output for enhanced user experience

**Processing Options**:
- Single test runs for validation
- Full batch processing with time estimates
- Skip existing mode for interrupted processing recovery

**Estimated Processing Times**:
- Sysmon only: 2-4 hours (full batch)
- Network only: 1-2 hours (full batch)  
- Both types: 3-5 hours (full batch)
- Skip existing: 0.5-3 hours (depending on completion)

---

### Enhanced Caldera Batch Processor
**File**: `batch/enhanced_caldera_batch_processor.sh`

Advanced batch processor for Script #4 across all APT runs.

```bash
cd scripts/batch
./enhanced_caldera_batch_processor.sh
```

**Features**:
- Auto-detection of original vs. processed Caldera files
- Smart skip existing extracted files functionality
- Entry count reporting for processed files
- Interactive menu with processing options
- Comprehensive error handling and logging
- File size and statistics tracking

**Processing Options**:
- Single test runs
- Full batch processing (15-45 minutes estimated)
- Skip existing mode (5-30 minutes estimated)
- Status overview and cleanup utilities

---

### Individual Batch Processors

**Sysmon Batch Processor**: `batch/sysmon_batch_processor.sh`
- Runs Script #5 across multiple APT runs
- Multi-run event analysis processing

**Network Batch Processor**: `batch/network_batch_processor.sh` 
- Runs Script #3 across multiple APT runs
- Batch network traffic processing

**Basic Caldera Batch Processor**: `batch/batch_caldera_report_transformer.sh`
- Simple wrapper for Script #4
- Basic batch Caldera processing

## 📊 Analysis and Utility Scripts

### Caldera Report Analyzer
**File**: `analysis/analyze_caldera_reports.py`

Analyzes structure and content of Caldera reports across multiple runs.

**Features**:
- JSON structure validation
- Entry counting and categorization  
- Cross-run comparison capabilities

### Command Analysis Tool
**File**: `analysis/caldera_command_analysis.py`

Deep analysis of Caldera command patterns and transformations.

**Features**:
- Command classification and pattern matching
- Transformation analysis and validation
- Statistical reporting on command types

### MITRE ATT&CK Tactic Analyzer
**File**: `analysis/tactic_analyzer.py`

Extracts and analyzes MITRE ATT&CK tactics from Caldera attack data.

```bash
python3 tactic_analyzer.py
```

**Features**:
- Tactic extraction from attack_metadata
- Duplicate detection and consistency analysis
- Statistical summary generation

## ⚙️ Configuration

### Main Configuration
**File**: `config/config.yaml`

Central configuration file controlling all pipeline scripts.

**Key Sections**:
- **data_sources**: Input/output file specifications
- **elasticsearch**: Connection settings and credentials
- **sysmon_processor**: Sysmon processing parameters
- **network_traffic_processor**: Network processing parameters  
- **event_analysis**: Correlation analysis settings
- **performance**: Memory and processing optimization
- **visualization**: Plot and output formatting

### Transformer Configuration
**File**: `config/transformer_config.json`

Specialized configuration for Caldera report transformation patterns.

## 📚 Development and Exploration

### Development Notebooks (`dev/`)
Professional development versions of all pipeline scripts:
- `1_elastic-index-downloader.ipynb` - Elasticsearch methodology
- `2_elastic_sysmon-ds_csv_creator.ipynb` - Sysmon conversion development
- `3_elastic_network-traffic-flow-ds_csv_creator.ipynb` - Network processing
- `4_caldera-report-analyzer.ipynb` - Caldera analysis development
- `5_sysmon-tracking-events.ipynb` - Event correlation development (9.9MB)
- `6_attack-timeline-analysis.ipynb` - Timeline visualization development

### Exploratory Notebooks (`exploratory/`)
Data exploration and validation notebooks:
- `2a-exploratory_sysmon-index.ipynb` - Sysmon EventID distribution
- `2b-structure-consistency-analyzer.ipynb` - Sysmon structure validation
- `2c_sysmon-csv-exploratory-analysis.ipynb` - Comprehensive Sysmon EDA
- `3a-exploratory_network-traffic-flow-index.ipynb` - Network flow exploration
- `3b-structure-consistency-analyzer.ipynb` - Network structure validation

## 🛠️ Usage Examples

### Complete Pipeline Execution
```bash
# Navigate to pipeline directory
cd scripts/pipeline

# Step 1: Download data (if needed)
python3 1_elastic_index_downloader.py

# Step 2: Convert to CSV format
python3 2_sysmon_csv_creator.py --apt-dir ../../apt-6/apt-6-run-50
python3 3_network_traffic_csv_creator.py --apt-dir ../../apt-6/apt-6-run-50

# Step 3: Transform Caldera reports  
python3 4_universal_caldera_report_transformer.py --apt-dir ../../apt-6/apt-6-run-50

# Step 4: Perform correlation analysis
python3 5_sysmon_event_analysis.py --apt-dir ../../apt-6/apt-6-run-50

# Step 5: Generate visualizations
python3 6_simple_timeline_plotter.py
```

### Batch Processing All APT Runs
```bash
# Navigate to batch directory
cd scripts/batch

# Process all JSONL files across all APT runs
./unified_jsonl_processor.sh

# Process all Caldera reports across all APT runs  
./enhanced_caldera_batch_processor.sh

# Run correlation analysis across all APT runs
./sysmon_batch_processor.sh
```

### Custom Configuration Usage
```bash
# Use custom configuration file
python3 5_sysmon_event_analysis.py --config custom_config.yaml

# Skip validation for faster processing
python3 2_sysmon_csv_creator.py --apt-dir DIR --no-validate

# Disable exploratory analysis
python3 3_network_traffic_csv_creator.py --apt-dir DIR --no-analysis
```

## 📈 Expected Performance

### Processing Metrics
- **Sysmon Processing**: 500K+ events/run (100% success rate)
- **Network Processing**: 1M+ flows/run (100% success rate)
- **Event Correlation**: Typically 90-95% success rate
- **Total Pipeline Time**: 15-60 minutes per APT run

### Memory Requirements
- **Recommended RAM**: 16GB+ for optimal performance
- **Minimum RAM**: 8GB with swap space
- **Disk Space**: ~2-3x input file size for processing overhead

### File Size Expectations
- **Input JSONL**: 1-5GB per file type per run
- **Output CSV**: 100-800MB per file type per run
- **Caldera Files**: 50-200KB input, 200-800KB extracted

## 🔧 System Requirements

### Dependencies
```bash
# Core Python packages
pip install pandas numpy matplotlib pyyaml beautifulsoup4 elasticsearch

# Optional packages for enhanced features
pip install seaborn jupyter notebook
```

### Environment Setup
```bash
# Activate recommended virtual environment
source dataset-venv/bin/activate

# Verify Python version (3.8+ recommended)
python3 --version
```

## 📝 Best Practices

### File Naming Conventions
- **APT Directories**: `apt-X/apt-X-run-XX` format
- **Input Files**: Elasticsearch export format with timestamps
- **Output Files**: Includes run number and data type
- **Config Files**: One `config.yaml` per APT run directory

### Processing Workflow
1. **Start Small**: Use single test runs before batch processing
2. **Monitor Resources**: Check memory usage during large operations
3. **Use Skip Mode**: Resume interrupted processing with "skip existing" options
4. **Validate Results**: Use built-in validation modes to verify outputs
5. **Backup Data**: Scripts automatically backup original data

### Error Recovery
- All scripts preserve original data during processing
- Comprehensive logging enables issue diagnosis
- Batch processors create detailed operation logs
- Failed operations can be resumed from last successful point

## 📊 Output Validation

### Automatic Validation Features
- **Schema Validation**: Ensures output CSV structure consistency
- **Data Integrity Checks**: Validates record counts and field completeness
- **Statistical Reporting**: Provides processing statistics and success rates
- **Backup Systems**: Preserves original data during all transformations

### Manual Validation Commands
```bash
# Check script help for validation options
python3 2_sysmon_csv_creator.py --help

# Run in validation mode
python3 2_sysmon_csv_creator.py --apt-dir DIR --validate-only

# Check processing logs
cat ../../*_processing_log_*.txt
```

---

## 📄 License and Contributing

This pipeline is designed for cybersecurity research and analysis. When contributing:
- Follow existing code style and documentation standards
- Add comprehensive error handling and logging
- Include validation and testing capabilities
- Update documentation for new features

**Pipeline Status**: ✅ Production Ready  
**Success Rate**: 90-95% correlation accuracy  
**Documentation**: Complete and maintained