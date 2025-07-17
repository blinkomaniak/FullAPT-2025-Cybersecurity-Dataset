# APT-1 Event Tracing Analysis (2025-05-04 Run 05) - Complete Documentation

This directory contains a **professional cybersecurity analysis pipeline** for analyzing Sysmon logs, network traffic, and Caldera attack simulations. This documentation provides a comprehensive guide to all files and their execution order.

## 📋 **Complete File Inventory**

### 🎯 **Production Scripts** (Main Pipeline)
| Script | Purpose | Input | Output | Dependencies |
|--------|---------|-------|--------|--------------|
| **`1_elastic_index_downloader.py`** | Downloads data from Elasticsearch | Interactive (ES connection) | `*.jsonl` files | elasticsearch |
| **`2_sysmon_csv_creator.py`** | Converts Sysmon JSONL to CSV | `-ds-logs-windows-sysmon_*.jsonl` | `sysmon-*.csv` | pandas, beautifulsoup4 |
| **`3_network_traffic_csv_creator.py`** | Converts network JSONL to CSV | `-ds-logs-network_traffic-*.jsonl` | `network_traffic_flow-*.csv` | pandas, numpy |
| **`sysmon_event_analysis.py`** | Correlates Caldera entries with Sysmon | `sysmon-*.csv` + Caldera JSON | Analysis report + Individual plots | pandas, matplotlib |

### 🔧 **Supporting Scripts**
| Script | Purpose | Input | Output | Usage |
|--------|---------|-------|--------|-------|
| **`6_simple_timeline_plotter.py`** | Timeline visualization | `sysmon-*-labeled.csv` | `timeline_*.png` | Standalone |
| **`tactic_analyzer.py`** | Analyzes Caldera tactics | Caldera JSON | Tactic analysis | Standalone |

### 📊 **Input Data Files**
| File | Type | Size | Source | Description |
|------|------|------|--------|-------------|
| **`-ds-logs-windows-sysmon_operational-default-2025-05-04-000001.jsonl`** | Raw Data | 2.8GB | Elasticsearch | Windows Sysmon events in JSONL format |
| **`-ds-logs-network_traffic-flow-default-2025-05-04-000001.jsonl`** | Raw Data | 2.3GB | Elasticsearch | Network traffic flows in JSONL format |
| **`apt34-05-04-test-1_event-logs.json`** | Attack Data | 91KB | Caldera | Original Caldera simulation logs |
| **`apt34-05-04-test-1_event-logs_extracted_information.json`** | Attack Data | 24KB | Processed | Extracted Caldera entries (52 attacks) |

### 📈 **Output Data Files**
| File | Type | Size | Producer | Description |
|------|------|------|----------|-------------|
| **`sysmon-2025-05-04-000001.csv`** | Processed | 162MB | Script #2 | Structured Sysmon events (570K+ records) |
| **`network_traffic_flow-2025-05-04-000001.csv`** | Processed | 814MB | Script #3 | Network flow data (1M+ records) |
| **`sysmon-2025-05-04-000001-labeled.csv`** | Labeled | 174MB | Timeline analysis | Sysmon events with attack labels |

### ⚙️ **Configuration Files**
| File | Purpose | Format | Used By |
|------|---------|--------|---------|
| **`config.yaml`** | Central configuration | YAML | All scripts |
| **`entry_config.csv`** | Entry-to-computer mapping | CSV | Event analysis |

### 📊 **Analysis Results**
| File | Type | Producer | Description |
|------|------|----------|-------------|
| **`tactic_analysis_results.json`** | JSON | Tactic analyzer | Detailed tactic statistics |
| **`tactic_analysis_summary.md`** | Markdown | Tactic analyzer | Human-readable tactic summary |
| **`timeline_by_computer.png`** | Visualization | Timeline analysis | Attack progression by computer |

### 📓 **Documentation Notebooks**
| Notebook | Category | Purpose | Corresponding Script |
|----------|----------|---------|---------------------|
| **`1_elastic-index-downloader.ipynb`** | Core Pipeline | Elasticsearch extraction methodology | `1_elastic_index_downloader.py` |
| **`2_elastic_sysmon-ds_csv_creator.ipynb`** | Core Pipeline | Sysmon JSONL→CSV conversion | `2_sysmon_csv_creator.py` |
| **`3_elastic_network-traffic-flow-ds_csv_creator.ipynb`** | Core Pipeline | Network traffic processing | `3_network_traffic_csv_creator.py` |
| **`5_sysmon-tracking-events.ipynb`** | Core Pipeline | Individual entry analysis + plotting | `sysmon_event_analysis.py` |
| **`2a-exploratory_sysmon-index.ipynb`** | Exploratory | Sysmon EventID distribution | - |
| **`2b-structure-consistency-analyzer.ipynb`** | Exploratory | Sysmon structure validation | - |
| **`2c_sysmon-csv-exploratory-analysis.ipynb`** | Exploratory | Comprehensive Sysmon EDA | - |
| **`3a-exploratory_network-traffic-flow-index.ipynb`** | Exploratory | Network flow exploration | - |
| **`3b-structure-consistency-analyzer.ipynb`** | Exploratory | Network structure validation | - |
| **`4_caldera-report-analyzer.ipynb`** | Analysis | Caldera data analysis | - |
| **`6_attack-timeline-analysis.ipynb`** | Analysis | Timeline visualization | `6_simple_timeline_plotter.py` |

## 🚀 **Execution Pipeline Order**

### **Phase 1: Data Extraction** 
```bash
# Step 1: Download data from Elasticsearch (if needed)
python3 1_elastic_index_downloader.py
```
**Purpose**: Downloads raw cybersecurity data from Elasticsearch clusters  
**Input**: Elasticsearch connection (interactive)  
**Output**: JSONL files containing raw Sysmon and network traffic data  
**When to run**: Only if you need to fetch fresh data from Elasticsearch  

### **Phase 2: Data Conversion** 
```bash
# Step 2A: Convert Sysmon data to structured CSV
python3 2_sysmon_csv_creator.py

# Step 2B: Convert network traffic data to structured CSV  
python3 3_network_traffic_csv_creator.py
```
**Purpose**: Transforms raw JSONL data into structured CSV format for analysis  
**Why this order**: Both scripts can run in parallel as they process different data sources  
**Critical**: These steps are **required** before any analysis can begin  

### **Phase 3: Core Analysis**
```bash
# Step 3: Perform event correlation analysis
python3 sysmon_event_analysis.py
```
**Purpose**: Correlates Caldera attack entries with Sysmon events  
**Dependencies**: Requires outputs from Steps 2A and 2B  
**Result**: 94.2% success rate correlating 52 attack entries  

### **Phase 4: Extended Analysis** (Optional)
```bash
# Optional: Generate timeline visualizations
python3 6_simple_timeline_plotter.py

# Optional: Analyze attack tactics
python3 tactic_analyzer.py
```
**Purpose**: Additional visualizations and tactical intelligence  
**Dependencies**: Can run independently with appropriate input data  

## 📊 **Data Flow Diagram**

```
┌─ Elasticsearch ─┐    ┌─ Caldera Simulation ─┐
│   Raw Events    │    │    Attack Logs       │
└─────────────────┘    └────────────────────────┘
         │                         │
         ▼                         ▼
    Script #1              apt34-*.json
  (Download JSONL)        (Attack Data)
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
   Analysis Report
```

## 🎯 **Key File Relationships**

### **Configuration Dependencies**
- **`config.yaml`**: Controls all script behavior (file paths, processing options)
- **`entry_config.csv`**: Maps Caldera entries to target computers for Script #5

### **Data Dependencies**
1. **Scripts 2 & 3** require JSONL files from Script 1 (or existing JSONL files)
2. **Script 5** requires CSV files from Scripts 2 & 3 + Caldera JSON files
3. **Timeline scripts** require labeled CSV files from previous analysis

### **Documentation Relationships**
- Each **production script** has a corresponding **notebook** showing methodology
- **Exploratory notebooks** provide deep analysis of data structure and patterns
- **Analysis notebooks** demonstrate advanced visualization and correlation techniques

## 🛠️ **Usage Examples**

### **Complete Pipeline Execution**
```bash
# Activate environment
source dataset-venv/bin/activate

# Run complete pipeline
python3 2_sysmon_csv_creator.py          # ~10 minutes for 2.8GB
python3 3_network_traffic_csv_creator.py # ~1.5 minutes for 2.3GB  
python3 sysmon_event_analysis.py        # Event correlation + plotting

# Optional analysis
python3 tactic_analyzer.py              # Tactical intelligence
```

### **Custom Configuration**
```bash
# Use custom settings
python3 sysmon_event_analysis.py --config my_config.yaml

# Skip validation for speed
python3 2_sysmon_csv_creator.py --no-validate

# Disable exploratory analysis  
python3 3_network_traffic_csv_creator.py --no-analysis
```

### **Individual Script Help**
```bash
# Get help for any script
python3 2_sysmon_csv_creator.py --help
python3 sysmon_event_analysis.py --help
```

## 📈 **Expected Results**

### **Performance Metrics**
- **Sysmon Processing**: 570,078 events processed (100% success rate)
- **Network Processing**: 1,090,212 flows processed (100% success rate)  
- **Event Correlation**: 49/52 entries detected (94.2% success rate)
- **Processing Time**: ~15 minutes total for complete pipeline

### **Output Validation**
- All scripts include **validation modes** to verify output correctness
- **Backup functionality** preserves original data during processing
- **Statistical reporting** confirms data integrity throughout pipeline

## 📝 **Notes & Best Practices**

### **File Naming Convention**
- **Raw data**: Prefixed with `-ds-logs-` (Elasticsearch export format)
- **Scripts**: Numbered by execution order (1, 2, 3, 5)
- **Notebooks**: Match script numbers with exploratory variants (2a, 2b, etc.)
- **Outputs**: Include timestamp and data type in filename

### **Memory Considerations**
- Large JSONL files (2.3GB, 2.8GB) require adequate RAM
- CSV conversion optimizes memory usage through streaming processing
- Progress indicators show processing status for long operations

### **Error Handling**
- All scripts include comprehensive error handling and logging
- Failed operations preserve original data and provide diagnostic information
- Validation modes allow verification of output correctness

---
**Pipeline Status**: ✅ Production Ready | **Success Rate**: 94.2% | **Documentation**: Complete