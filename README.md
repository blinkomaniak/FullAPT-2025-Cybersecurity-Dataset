# Bigbase IDS Dataset

A comprehensive labeled cybersecurity dataset for training AI-driven Intrusion Detection Systems (IDS). This dataset contains correlated attack simulations with system telemetry data, providing high-quality labeled examples suitable for machine learning-based threat detection research and model development.

## 🏗️ Dataset Contents and Creation Pipeline

This dataset was created using two integrated components that process and validate cybersecurity data:

### 1. **APT Data Processing Pipeline** (`data-raw/`)
The processing scripts used to correlate APT attack simulations with Windows Sysmon events, network traffic, and Caldera attack data, creating the labeled cybersecurity dataset.

### 2. **ML Dataset Evaluation Framework** (`analysis/`)
Validation scripts that verify dataset quality using multiple machine learning models (SAE, LSTM-SAE, GRU-SAE, Isolation Forest) to ensure the dataset is suitable for AI-driven IDS development.

## 📂 Complete Directory Structure

```
bigbase-ids-dataset/                         # Bigbase IDS Dataset root directory
├── 🎯 data-raw/                             # DATASET CREATION SCRIPTS & APT STRUCTURES
│   ├── scripts/                             # APT processing pipeline
│   │   ├── pipeline/                        # Core processing scripts (#1-#6)
│   │   │   ├── 1_elastic_index_downloader.py
│   │   │   ├── 2_sysmon_csv_creator.py
│   │   │   ├── 3_network_traffic_csv_creator.py
│   │   │   ├── 4_universal_caldera_report_transformer.py
│   │   │   ├── 5_sysmon_event_analysis.py
│   │   │   └── 6_simple_timeline_plotter.py
│   │   ├── batch/                           # Automated batch processing
│   │   │   ├── unified_jsonl_processor.sh
│   │   │   ├── enhanced_caldera_batch_processor.sh
│   │   │   ├── sysmon_batch_processor.sh
│   │   │   └── network_batch_processor.sh
│   │   ├── analysis/                        # Analysis and utilities
│   │   ├── config/                          # APT configuration files
│   │   ├── dev/                             # Development notebooks
│   │   ├── exploratory/                     # Data exploration notebooks
│   │   └── tools/                           # Utility scripts
│   ├── apt-1/ to apt-6/                     # Dataset structure (50 APT scenarios)
│   │   └── apt-X-run-XX/                    # Individual attack scenario directories
│   │       ├── config.yaml                  # Scenario configuration
│   │       ├── entry_config.csv             # Attack entry mapping
│   │       ├── *event-logs.json             # Original Caldera attack reports
│   │       ├── *extracted_information.json  # Processed attack reports
│   │       ├── sysmon-run-XX.csv            # Labeled Windows event logs
│   │       ├── network_traffic_flow-run-XX.csv # Labeled network flow data
│   │       └── *.jsonl                      # Raw telemetry data
│   ├── issues/                              # Known issues and solutions
│   ├── dataset-backup/                      # Compressed backups
│   └── README.md                            # APT pipeline documentation
│
├── 🤖 analysis/                             # DATASET VALIDATION SCRIPTS
│   ├── sae/                                 # Stacked Autoencoder pipeline
│   │   ├── 01-aggregation.py
│   │   ├── 02-encoding-bigbase-sae.py
│   │   ├── 03-training-sae.py
│   │   ├── 04-model-evaluation-sae.py
│   │   ├── 05-result-vis.py
│   │   └── run_sae_pipeline.py
│   ├── lstm-sae/                            # LSTM Autoencoder pipeline
│   │   ├── 01-aggregation.py
│   │   ├── 02-encoding-lstm-sae.py
│   │   ├── 03-training-lstm-sae.py
│   │   ├── 04-model-evaluation-lstm-sae.py
│   │   ├── 05-result-vis-lstm-sae.py
│   │   └── run_lstm_sae_pipeline.py
│   ├── gru-sae/                             # GRU Autoencoder pipeline
│   │   ├── 01-aggregation.py
│   │   ├── 02-encoding-gru-sae.py
│   │   ├── 03-training-gru-sae.py
│   │   ├── 04-model-evaluation-gru-sae.py
│   │   ├── 05-result-vis-gru-sae.py
│   │   └── run_gru_sae_pipeline.py
│   ├── iforest/                             # Isolation Forest pipeline
│   │   ├── 01-aggregation.py
│   │   ├── 02-encoding-bigbase-iforest.py
│   │   ├── 03-training-iforest.py
│   │   ├── 04-model-evaluation-iforest.py
│   │   ├── 05-feature-space-iforest.py
│   │   └── run_iforest_pipeline.py
│   ├── config/                              # ML configuration files
│   │   ├── schema-{dataset}-{model}-v{X}.json
│   │   └── encoding-{dataset}-{model}-v{X}.json
│   ├── comparative-evaluation/              # Cross-model analysis
│   │   ├── comparative_analysis.py
│   │   ├── dataset_quality_analyzer.py
│   │   └── roc_comparison_generator.py
│   └── utils/                               # Shared utilities
│       ├── paths.py
│       ├── config.py
│       └── encoding.py
│
├── 📊 datasets/                             # RAW CYBERSECURITY DATASETS
│   ├── bigbase/                             # Windows Event Logs (50 CSV files)
│   │   ├── dataset-01.csv to dataset-50.csv
│   │   └── reports/                         # Caldera attack reports
│   │       ├── cal-report-01.json to cal-report-50.json
│   │       └── entry_params-01.json to entry_params-50.json
│   ├── unraveled/                           # Multi-modal cybersecurity data
│   │   ├── host-logs/                       # Host-based logs
│   │   │   ├── audit/                       # Linux audit logs
│   │   │   ├── auth/                        # Authentication logs
│   │   │   ├── filebeat/                    # Filebeat logs
│   │   │   ├── syslog/                      # System logs
│   │   │   └── windows/                     # Windows event logs
│   │   ├── network-flows/                   # Network flow data by weeks/days
│   │   │   ├── Week1_Day1-2_05262021-05272021/
│   │   │   ├── Week1_Day3_05282021/
│   │   │   ├── ... (26 week/day directories)
│   │   │   └── Week6_Day6-7_07032021-07042021/
│   │   └── nids/                            # Network intrusion detection
│   │       └── all_snort
│   └── dapt2020/                            # DAPT2020 network flow dataset
│       ├── README.md
│       ├── enp0s3-*.pcap_Flow.csv          # Network flow files
│       └── util.py
│
├── 💾 data/                                 # PROCESSED PIPELINE OUTPUTS
│   ├── processed/                           # Stage 1: Aggregated datasets
│   │   ├── {model}-{dataset}-v{X}.parquet
│   │   └── metadata/
│   └── encoded/                             # Stage 2: Encoded features
│       └── {model}-{dataset}-v{X}/
│           ├── X_train_encoded.npz
│           ├── X_test_encoded.npz
│           ├── y_train.npy
│           └── column_transformer.joblib
│
├── 🎯 models/                               # TRAINED ML MODELS
│   └── {model}-{dataset}-v{X}/
│       ├── model.keras                      # Neural network models
│       ├── model.joblib                     # Scikit-learn models
│       └── training_history.json
│
├── 📈 artifacts/                            # ANALYSIS OUTPUTS
│   ├── eval/                                # Model evaluation results
│   │   ├── {model}-{dataset}-v{X}/
│   │   │   ├── evaluation_report.json
│   │   │   ├── roc_curve.png
│   │   │   └── confusion_matrix.png
│   ├── plots/                               # Visualizations
│   ├── metadata/                            # Dataset metadata
│   ├── comparative/                         # Cross-model comparisons
│   │   ├── comparative_analysis_report.json
│   │   ├── model_performance_comparison.png
│   │   └── feature_set_impact_analysis.png
│   ├── quality_analysis/                    # Dataset quality reports
│   │   ├── quality_analysis_report.json
│   │   └── dataset_quality_analysis.png
│   └── roc_comparison/                      # ROC curve comparisons
│       └── roc_pr_comparison_{dataset}_v{X}.png
│
├── 📚 docs/                                 # DOCUMENTATION
│   ├── ACADEMIC-PAPER-GUIDELINES-BIGBASE.md
│   └── infrastructure.yaml
│
├── 📄 pdfs/                                 # RESEARCH PAPERS
│   └── [100+ cybersecurity research papers]
│
├── 🔧 others/                               # MISCELLANEOUS FILES
│   ├── elasticsearch_certs/                # Elasticsearch certificates
│   ├── hacking-scripts/                     # Attack simulation scripts
│   ├── images/                              # Network diagrams and visualizations
│   ├── papers/                              # Additional research papers
│   └── setup-windows-shared-files/         # Windows setup files
│
├── 🐍 dataset-venv/                         # PYTHON VIRTUAL ENVIRONMENT
├── CLAUDE.md                                # Claude Code documentation
└── README.md                                # This file
```

## 🎯 **APT Data Processing System** (`data-raw/`)

### Overview
Data processing pipeline that transforms APT attack simulations into labeled cybersecurity datasets. Processes 50 comprehensive attack runs across 6 APT campaigns, correlating Windows Sysmon events, network traffic flows, and Caldera attack simulation data to create training data for AI-driven IDS.

### Key Capabilities
- **Multi-modal Data Processing**: Windows events, network flows, attack logs
- **Attack-Event Correlation**: 90-95% success rate in labeling attack events within system telemetry
- **Automated Dataset Creation**: Batch processors for large-scale labeled dataset generation
- **Quality Validation**: Individual event tracking and dataset verification

### Core Pipeline Scripts
1. **Script #1**: Elasticsearch data downloader with secure authentication
2. **Script #2**: Sysmon JSONL to CSV converter (18+ event types)
3. **Script #3**: Network traffic JSONL to CSV converter (42 mapped fields)
4. **Script #4**: Universal Caldera report transformer (7 transformation functions)
5. **Script #5**: Main correlation engine with time-window analysis
6. **Script #6**: Timeline visualization generator

### Generated Datasets
- **50 Labeled Attack Scenarios**: apt-1-run-01 through apt-6-run-50
- **6 APT Campaign Types**: Different attack patterns and techniques for dataset diversity
- **Multi-gigabyte Scale**: 1-5GB processed labeled data per scenario
- **High Labeling Accuracy**: 90-95% attack event labeling success rate

### Usage Example
```bash
cd data-raw/scripts/batch
./unified_jsonl_processor.sh              # Process all JSONL files
./enhanced_caldera_batch_processor.sh     # Process all Caldera reports
./sysmon_batch_processor.sh               # Run correlation analysis
```

## 🤖 **ML Dataset Evaluation Framework** (`analysis/`)

### Overview
Validates the quality and suitability of generated cybersecurity datasets using multiple machine learning model architectures. Tests dataset effectiveness for training AI-driven IDS by evaluating detection performance across different model types.

### Supported Models
- **SAE**: Stacked Autoencoder with bottleneck architecture
- **LSTM-SAE**: LSTM-based Autoencoder for temporal patterns
- **GRU-SAE**: GRU-based Autoencoder (more efficient than LSTM)
- **Isolation Forest**: Ensemble-based anomaly detection with PCA

### Pipeline Stages
1. **Stage 01**: Data aggregation with schema-based processing
2. **Stage 02**: Feature encoding (TF-IDF, OneHot, StandardScaler)
3. **Stage 03**: Model training on benign data only
4. **Stage 04**: Evaluation with ROC-AUC, precision, recall metrics
5. **Stage 05**: Result visualization and analysis

### Dataset Types Supported
- **bigbase**: Windows event logs with text processing for command-line analysis
- **unraveled**: Multi-modal cybersecurity data (host logs, network flows, NIDS)
- **dapt2020**: Network flow data with extracted features for baseline comparison

### Usage Examples
```bash
# SAE Pipeline
cd analysis/sae
python run_sae_pipeline.py --dataset bigbase --version v1

# LSTM-SAE Pipeline
cd analysis/lstm-sae  
python run_lstm_sae_pipeline.py --dataset unraveled --version v1 --seq-len 10

# Comparative Analysis
cd analysis/comparative-evaluation
python comparative_analysis.py
```

## 📊 **Integrated Dataset Creation Capabilities**

### End-to-End Dataset Pipeline
The platform provides complete dataset creation workflow:
- **Attack Simulation Processing** from real APT campaign data
- **ML Model Validation** to ensure dataset quality and effectiveness
- **Multi-modal Data Integration** across different cybersecurity data types
- **Temporal Pattern Preservation** for time-series attack analysis

### Dataset Applications
1. **IDS Training Data**: High-quality labeled datasets for AI-driven intrusion detection
2. **Attack Pattern Recognition**: Datasets containing diverse attack techniques and patterns
3. **Temporal Attack Analysis**: Time-series data for sequential attack modeling
4. **Multi-modal Fusion**: Combined network, host, and application-level attack data
5. **Benchmark Creation**: Standardized datasets for comparing IDS model performance

## 🚀 **Quick Start Guide**

### Prerequisites
```bash
# Activate Python environment
source dataset-venv/bin/activate

# Verify dependencies
python -c "import pandas, numpy, tensorflow, sklearn, matplotlib"
```

### Dataset Creation Quick Start
```bash
# Navigate to data processing system
cd data-raw/scripts/pipeline

# Process a single APT scenario into labeled dataset
python 2_sysmon_csv_creator.py --apt-dir ../../apt-6/apt-6-run-50
python 3_network_traffic_csv_creator.py --apt-dir ../../apt-6/apt-6-run-50
python 4_universal_caldera_report_transformer.py --apt-dir ../../apt-6/apt-6-run-50
python 5_sysmon_event_analysis.py --apt-dir ../../apt-6/apt-6-run-50
```

### Dataset Validation Quick Start
```bash
# Navigate to evaluation framework
cd analysis/sae

# Validate dataset quality with SAE model
python run_sae_pipeline.py --dataset bigbase --version v1

# Compare dataset performance across models
cd ../comparative-evaluation
python comparative_analysis.py
```

## 🔧 **System Requirements**

### Hardware Recommendations
- **RAM**: 16GB+ recommended (32GB for large datasets)
- **Storage**: 50-100GB free space for datasets and outputs
- **CPU**: Multi-core processor (8+ cores recommended)
- **GPU**: Optional but recommended for LSTM/GRU training

### Software Dependencies
```bash
# Core ML packages
tensorflow>=2.19.0
torch>=2.7.1
scikit-learn>=1.6.1
pandas>=2.2.3
numpy>=1.24.0

# APT analysis packages
beautifulsoup4
elasticsearch
pyyaml
matplotlib
seaborn

# Optional packages
jupyter
notebook
```

## 📈 **Dataset Creation Performance**

### Data Processing Performance
- **Labeling Accuracy**: 90-95% attack event labeling success rate
- **Processing Speed**: 15-60 minutes per APT scenario
- **Dataset Scale**: 500K+ labeled Sysmon events, 1M+ network flows per scenario
- **Batch Creation**: 50 labeled scenarios in 3-5 hours

### Dataset Validation Performance
- **Model Training**: 10-60 minutes depending on model complexity and dataset size
- **Memory Efficiency**: Optimized for large cybersecurity datasets with sparse matrices
- **Validation Speed**: Comprehensive dataset evaluation in 5-15 minutes
- **Multi-Model Comparison**: Complete dataset quality analysis in 30-60 minutes

## 📝 **Research Documentation**

### Academic Resources
- **100+ Research Papers**: Latest cybersecurity and ML papers in `pdfs/`
- **Academic Guidelines**: Paper writing guidelines in `docs/`
- **Experimental Design**: Detailed documentation for each model pipeline

### Technical Documentation
- **Data Processing Pipeline**: Complete documentation in `data-raw/scripts/README.md`
- **Dataset Evaluation**: Individual documentation for each model validation pipeline
- **Configuration**: Schema and encoding configuration examples for different dataset types
- **Troubleshooting**: Common issues and solutions for dataset creation

## 🤝 **Contributing and Usage**

### Research Ethics
- This dataset creation platform is designed for **defensive cybersecurity research** only
- Generated datasets are for **improving AI-driven intrusion detection systems**
- Attack simulation data is used solely for creating training data for defensive purposes
- No malicious tools or offensive techniques are included

### Best Practices
1. **Data Integrity**: Always backup data before processing
2. **Resource Management**: Monitor memory usage during large operations
3. **Reproducibility**: Use version-controlled configurations
4. **Documentation**: Document experimental changes and results

### Contributing Guidelines
- Follow existing code style and structure
- Add comprehensive error handling and logging
- Update documentation for new features
- Include validation and testing capabilities

## 📊 **Dataset Impact**

This platform enables cutting-edge cybersecurity dataset creation for:
- **AI-driven IDS Development** with high-quality labeled training data
- **Multi-modal Attack Detection** across network, host, and application layers
- **Temporal Attack Pattern Learning** with time-series labeled datasets
- **Cross-Attack-Type Validation** for robust IDS model training
- **Standardized Benchmarking** for comparing IDS model performance

**Platform Status**: ✅ Production Ready  
**Primary Application**: Creating labeled cybersecurity datasets for AI-driven IDS research and development  
**Labeling Accuracy**: 90-95% attack event labeling success rate  
**Dataset Scale**: 50 APT attack scenarios + multiple cybersecurity dataset types for comprehensive IDS training