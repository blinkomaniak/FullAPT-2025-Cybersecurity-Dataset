# Dual-Domain Cybersecurity Dataset Pipeline Documentation

## Overview
This documentation suite provides comprehensive guidance for the dual-domain cybersecurity dataset development pipeline, covering both production pipeline scripts and exploratory research tools.

## Directory Structure Discovered

### APT Dataset Coverage
```
📊 APT DATASET MAPPING:
├── apt-1: Runs 04-20, 51 (18 datasets) - OilRig-based attacks
├── apt-2: Runs 21-30 (10 datasets) - OilRig variants  
├── apt-3: Runs 31-38 (8 datasets) - OilRig variants
├── apt-4: Runs 39-44 (6 datasets) - APT-29 based attacks
├── apt-5: Runs 45-47 (3 datasets) - APT-29 variants
├── apt-6: Runs 48-50 (3 datasets) - Wizard Spider based
└── apt-7: Run 52 (1 dataset) - Additional campaign

Total: 49 APT attack datasets across 6 major campaign types
```

## Production Pipeline Scripts

### Pipeline Directory: `/dataset/scripts/pipeline/`

| Script | Documentation | Purpose | Input | Output |
|--------|---------------|---------|-------|---------|
| **1_elastic_index_downloader.py** | [📄 Documentation](pipeline/1_elastic_index_downloader.md) | Downloads cybersecurity data from Elasticsearch clusters | Elasticsearch indices | JSONL files |
| **2_sysmon_csv_creator.py** | [📄 Documentation](pipeline/2_sysmon_csv_creator.md) | Transforms Sysmon JSONL to structured CSV | JSONL (Sysmon) | CSV (Structured) |
| **3_network_traffic_csv_creator.py** | [📄 Documentation](pipeline/3_network_traffic_csv_creator.md) | Transforms NetFlow JSONL to structured CSV | JSONL (NetFlow) | CSV (Structured) |
| **4_enhanced_temporal_causation_correlator.py** | [📄 Documentation](pipeline/4_enhanced_temporal_causation_correlator.md) | Advanced dual-domain temporal correlation | Sysmon + NetFlow CSV | Correlation analysis |
| **5_comprehensive_correlation_analysis.py** | [📄 Documentation](pipeline/5_comprehensive_correlation_analysis.md) | Complete analysis and visualization suite | Correlation results | Executive reports |

## Exploratory Research Scripts

### Exploratory Directory: `/dataset/scripts/exploratory/`

| Script | Documentation | Purpose | Research Focus |
|--------|---------------|---------|----------------|
| **1_comprehensive_network_community_id_analyzer.py** | [📄 Documentation](exploratory/1_comprehensive_network_community_id_analyzer.md) | Network community ID pattern analysis | Network behavior analysis |
| **2_process_tuple_uniqueness_validator.py** | *[Documentation Pending]* | Process tuple validation and uniqueness | Data quality assurance |
| **3_enhanced_temporal_causation_correlator.py** | *[Exploratory version]* | Research version of pipeline script | Temporal correlation research |
| **4_comprehensive_correlation_analysis.py** | *[Exploratory version]* | Research version of pipeline script | Correlation analysis research |
| **5_sysmon_seed_event_extractor.py** | [📄 Documentation](exploratory/5_sysmon_seed_event_extractor.md) | Attack originator event extraction | Manual attack analysis |
| **6_sysmon_attack_lifecycle_tracer.py** | [📄 Documentation](exploratory/6_sysmon_attack_lifecycle_tracer.md) | Multi-EventID attack lifecycle analysis | Attack progression tracking |

## Testing Scripts

### Current Testing Status

| Script | Documentation | Status | Purpose |
|--------|---------------|---------|---------|
| **TESTING_8_dual_timeline_correlator.py** | [📄 Documentation](exploratory/TESTING_8_dual_timeline_correlator.md) | 🟢 **Production Ready** | Dual-domain temporal correlation visualization |
| **test_complete_only.py** | *[Utility Script]* | Helper | Quick testing utility for TESTING_8 |
| **analyze_internal_flows.py** | *[Utility Script]* | Helper | Internal network flow validation |

## Pipeline Execution Workflow

### Data Flow Architecture
```
🔄 COMPLETE PIPELINE FLOW:

1. DATA EXTRACTION
   Elasticsearch → JSONL Files
   [Script 1: elastic_index_downloader.py]

2. DATA TRANSFORMATION  
   JSONL → Structured CSV
   [Script 2: sysmon_csv_creator.py]
   [Script 3: network_traffic_csv_creator.py]

3. MANUAL ANALYSIS
   Raw Events → Selected Events
   [Script 5: sysmon_seed_event_extractor.py]
   [Manual Selection Process]

4. CORRELATION ANALYSIS
   Sysmon + NetFlow → Temporal Correlations
   [Script 4: enhanced_temporal_causation_correlator.py]

5. COMPREHENSIVE ANALYSIS
   Correlation Results → Executive Insights
   [Script 5: comprehensive_correlation_analysis.py]

6. VISUALIZATION & VALIDATION
   Dual-Domain Timeline Analysis
   [TESTING_8: dual_timeline_correlator.py]
```

## Execution Guidelines

### Pipeline Scripts
**Location**: Must run from project root
```bash
cd /home/researcher/Downloads/research/
python3 dataset/scripts/pipeline/[script_name].py [options]
```

### Exploratory Scripts  
**Location**: Can run from exploratory directory
```bash
cd /home/researcher/Downloads/research/dataset/scripts/exploratory/
python3 [script_name].py [options]
```

## Research Applications

### Academic Research
- **Publication-Ready Visualizations**: High-resolution timeline analysis
- **Statistical Evidence**: Comprehensive correlation analysis
- **Methodology Validation**: Dual-domain approach effectiveness
- **Attack Pattern Discovery**: Cross-campaign behavioral analysis

### Industry Applications
- **Threat Detection**: Real-time correlation algorithms
- **Incident Response**: Timeline reconstruction capabilities
- **Security Operations**: Enhanced monitoring methodologies
- **Threat Intelligence**: Attack signature development

## Key Dependencies

### Core Libraries
```bash
pip install pandas numpy matplotlib seaborn
pip install elasticsearch beautifulsoup4 pyyaml
pip install argparse logging pathlib datetime
```

### System Requirements
- **Memory**: 8-32GB recommended for full pipeline
- **CPU**: Multi-core systems recommended (8+ cores optimal)
- **Storage**: 100GB+ for complete APT dataset collection
- **Network**: Access to Elasticsearch cluster (for Script 1)

## Data Protection
⚠️ **CRITICAL**: Pipeline scripts are configured to NEVER delete files from:
- `dataset/dataset-backup/` (backup preservation)
- `dataset/apt-Y/apt-Y-run-X/` (dataset preservation)

All scripts include protection mechanisms to prevent accidental data loss.

## Quality Assurance

### Validation Features
- **Data Integrity**: Comprehensive validation at each pipeline stage
- **Error Handling**: Graceful handling of missing or corrupted data
- **Progress Tracking**: Detailed logging and progress reporting
- **Output Verification**: Automated verification of generated outputs

### Testing Methodology
- **Individual Testing**: Each script tested on sample datasets
- **Integration Testing**: End-to-end pipeline validation
- **Performance Testing**: Resource utilization and timing analysis
- **Quality Control**: Output validation and consistency checking

## Support and Troubleshooting

### Common Issues
1. **Memory Issues**: Adjust chunk sizes and worker counts
2. **Missing Files**: Verify dataset directory structure
3. **Performance**: Optimize threading and processing parameters
4. **Visualization**: Check matplotlib configuration and display settings

### Debug Resources
- **Debug Modes**: Most scripts support --debug flags
- **Sample Testing**: Use --sample parameters for quick validation
- **Log Analysis**: Comprehensive logging for troubleshooting
- **Resource Monitoring**: Built-in performance monitoring

---

## Documentation Status

### ✅ Complete Documentation
- Pipeline Scripts 1-5: ✅ Fully documented
- Exploratory Scripts 1, 5, 6: ✅ Fully documented
- TESTING_8: ✅ Production-ready documentation

### 📋 Future Documentation
- Scripts 2, 3, 4 (exploratory versions): Research-specific documentation
- Utility scripts: Brief usage documentation
- Integration guides: Cross-script workflow documentation

---

*This comprehensive documentation suite provides complete guidance for understanding, executing, and extending the dual-domain cybersecurity dataset development pipeline, supporting both production deployment and continued research development.*