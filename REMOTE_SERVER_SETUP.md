# Remote Server Setup Guide for Dataset Evaluation Pipeline

This guide provides the exact folder structure needed to run the dataset evaluation pipeline smoothly on a remote server. This setup supports all model types (SAE, LSTM-SAE, Isolation Forest, etc.).

## 📂 Complete Directory Structure

```
research/                              # Project root directory
├── analysis/                          # ML pipeline scripts and configurations
│   ├── sae/                           # SAE pipeline scripts
│   │   ├── 01-aggregation.py
│   │   ├── 02-encoding-bigbase-sae.py
│   │   ├── 02-encoding-unraveled-sae.py
│   │   ├── 03-training-sae.py
│   │   ├── 04-model-evaluation-sae.py
│   │   ├── 05-result-vis.py
│   │   ├── run_sae_pipeline.py        # Pipeline orchestrator
│   │   ├── run_sae.sh                 # Shell wrapper
│   │   ├── README_PIPELINE.md
│   │   └── PATH_VERIFICATION_REPORT.md
│   ├── lstm-sae/                      # LSTM-SAE pipeline scripts
│   │   ├── 01-aggregation.py
│   │   ├── 02-encoding-lstm-sae.py
│   │   ├── 03-training-lstm-sae.py
│   │   ├── 04-model-evaluation-lstm-sae.py
│   │   ├── 05-result-vis-lstm-sae.py
│   │   ├── run_lstm_sae_pipeline.py
│   │   └── smart_early_stopping.py
│   ├── gru-sae/                       # GRU-SAE pipeline scripts
│   │   ├── 01-aggregation.py
│   │   ├── 02-encoding-gru-sae.py
│   │   ├── 03-training-gru-sae.py
│   │   ├── 04-model-evaluation-gru-sae.py
│   │   ├── 05-result-vis-gru-sae.py
│   │   ├── run_gru_sae_pipeline.py
│   │   └── smart_early_stopping.py
│   ├── iforest/                       # Isolation Forest pipeline scripts
│   │   ├── 01-aggregation.py
│   │   ├── 02-encoding-bigbase-iforest.py
│   │   ├── 02-encoding-unraveled-iforest.py
│   │   ├── 03-training-iforest.py
│   │   ├── 04-model-evaluation-iforest.py
│   │   ├── 05-feature-space-iforest.py
│   │   ├── run_iforest_pipeline.py
│   │   └── run_iforest.sh
│   ├── config/                        # Configuration files
│   │   ├── schema-bigbase-sae-v1.json
│   │   ├── schema-bigbase-sae-v2.json
│   │   ├── schema-bigbase-lstm-sae-v1.json
│   │   ├── schema-bigbase-lstm-sae-v2.json
│   │   ├── schema-bigbase-gru-sae-v1.json
│   │   ├── schema-bigbase-gru-sae-v2.json
│   │   ├── schema-unraveled-sae-v1.json
│   │   ├── schema-unraveled-lstm-sae-v1.json
│   │   ├── schema-unraveled-gru-sae-v1.json
│   │   ├── encoding-bigbase-sae-v1.json
│   │   ├── encoding-bigbase-sae-v2.json
│   │   ├── encoding-bigbase-lstm-sae-v1.json
│   │   ├── encoding-bigbase-lstm-sae-v2.json
│   │   ├── encoding-bigbase-gru-sae-v1.json
│   │   ├── encoding-bigbase-gru-sae-v2.json
│   │   ├── encoding-bigbase-iforest-v1.json
│   │   ├── encoding-unraveled-sae-v1.json
│   │   ├── encoding-unraveled-lstm-sae-v1.json
│   │   ├── encoding-unraveled-gru-sae-v1.json
│   │   └── encoding-unraveled-iforest-v1.json
│   ├── comparative-evaluation/        # Cross-model analysis scripts
│   │   ├── comparative_analysis.py
│   │   ├── dataset_quality_analyzer.py
│   │   ├── roc_comparison_generator.py
│   │   └── simple_quality_analyzer.py
│   ├── utils/                         # Utility modules
│   │   ├── __init__.py
│   │   ├── paths.py                   # Path management utilities
│   │   ├── config.py                  # Configuration loading
│   │   └── encoding.py                # Data preprocessing utilities
│   └── __init__.py
├── datasets/                          # Raw cybersecurity datasets (REQUIRED - must be uploaded)
│   ├── bigbase/                       # Windows event logs dataset
│   │   ├── dataset-01.csv
│   │   ├── dataset-02.csv
│   │   ├── ...
│   │   ├── dataset-50.csv             # 50 CSV files total
│   │   └── reports/                   # Caldera attack reports
│   │       ├── cal-report-01.json
│   │       ├── entry_params-01.json
│   │       └── ...                    # Additional report files
│   ├── unraveled/                     # Multi-modal cybersecurity data
│   │   ├── host-logs/                 # Host log data
│   │   │   ├── audit/
│   │   │   ├── auth/
│   │   │   ├── filebeat/
│   │   │   ├── syslog/
│   │   │   └── windows/
│   │   ├── network-flows/             # Network flow data by weeks/days
│   │   │   ├── Week1_Day1-2_05262021-05272021/
│   │   │   ├── Week1_Day3_05282021/
│   │   │   ├── Week1_Day4_05292021/
│   │   │   └── ...                    # Additional week/day directories
│   │   └── nids/                      # Network intrusion detection data
│   │       └── all_snort
│   └── dapt2020/                      # DAPT2020 network flow dataset
│       ├── README.md
│       ├── enp0s3-monday-pvt.pcap_Flow.csv
│       ├── enp0s3-monday.pcap_Flow.csv
│       ├── util.py
│       └── ...                        # Additional PCAP flow files
├── data/                              # Pipeline outputs (auto-created)
│   ├── processed/                     # Stage 1 outputs
│   │   ├── bigbase-sae-v1.parquet
│   │   └── unraveled-sae-v1.parquet
│   └── encoded/                       # Stage 2 outputs
│       ├── bigbase-sae-v1/
│       │   ├── X_train_encoded.npz
│       │   ├── X_val_encoded.npz
│       │   ├── X_test_encoded.npz
│       │   ├── y_train.npy
│       │   ├── y_val.npy
│       │   ├── y_test.npy
│       │   └── column_transformer.joblib
│       └── unraveled-sae-v1/
│           └── [same structure as bigbase]
├── models/                            # Stage 3 outputs (auto-created)
│   └── (model files created by pipeline scripts)
├── artifacts/                         # Analysis outputs (auto-created)
│   ├── eval/                          # Model evaluation results
│   │   ├── sae/
│   │   ├── lstm-sae/
│   │   ├── gru-sae/
│   │   └── iforest/
│   ├── plots/                         # Visualizations
│   ├── metadata/                      # Dataset metadata
│   ├── comparative/                   # Cross-model analysis
│   ├── quality_analysis/              # Dataset quality reports
│   └── roc_comparison/                # ROC curve comparisons
├── docs/                              # Documentation files
├── pdfs/                              # Research papers and reports
├── others/                            # Miscellaneous files
├── data-raw/                          # Raw data collection artifacts
├── dataset-venv/                      # Python virtual environment
├── CLAUDE.md                          # Project documentation for Claude Code
└── REMOTE_SERVER_SETUP.md             # This file
```

## 🚀 Setup Instructions

### 1. Create Base Directory Structure
```bash
# Create the main project directory
mkdir -p dataset-eval
cd dataset-eval

# Create required subdirectories that won't be auto-created
mkdir -p analysis/{sae,lstm-sae,gru-sae,iforest,config,utils,comparative-evaluation}
mkdir -p datasets/{bigbase,unraveled,dapt2020}
mkdir -p data/{processed,encoded} models artifacts/{eval,plots,metadata,comparative,quality_analysis,roc_comparison}
mkdir -p docs pdfs others data-raw

# Create empty __init__.py files for Python imports
touch analysis/__init__.py
touch analysis/utils/__init__.py
```

### 2. Upload Code Files
Transfer these files to their respective directories:

**Pipeline Scripts** → `analysis/sae/` (or other model directories):
- `01-aggregation.py`
- `02-encoding-bigbase-sae.py`  
- `02-encoding-unraveled-sae.py`
- `03-training-sae.py`
- `04-model-evaluation-sae.py`
- `05-result-vis.py`
- `run_sae_pipeline.py`
- `run_sae.sh`
- `README_PIPELINE.md`

**Configuration Files** → `analysis/config/`:
- Schema configs: `schema-bigbase-{model}-v{version}.json`
- Schema configs: `schema-unraveled-{model}-v{version}.json`
- Encoding configs: `encoding-bigbase-{model}-v{version}.json`
- Encoding configs: `encoding-unraveled-{model}-v{version}.json`
- Where {model} = sae, lstm-sae, gru-sae, iforest
- Where {version} = v1, v2, etc.

**Utility Modules** → `analysis/utils/`:
- `paths.py`
- `config.py`
- `encoding.py`
- `__init__.py`

**Project Documentation** → root directory:
- `CLAUDE.md`
- `REMOTE_SERVER_SETUP.md`

### 3. Upload Raw Datasets (CRITICAL)
This is the most important step - your raw CSV data must be placed exactly here:

**Bigbase Dataset** → `datasets/bigbase/`:
```bash
datasets/bigbase/
├── dataset-01.csv
├── dataset-02.csv
├── ...
└── dataset-50.csv
```

**Unraveled Dataset** → `datasets/unraveled/`:
```bash
datasets/unraveled/
├── host-logs/
│   ├── audit/
│   ├── auth/
│   ├── filebeat/
│   ├── syslog/
│   └── windows/
├── network-flows/
│   ├── Week1_Day1-2_05262021-05272021/
│   ├── Week1_Day3_05282021/
│   └── [additional week/day directories]
└── nids/
```

**DAPT2020 Dataset** → `datasets/dapt2020/`:
```bash
datasets/dapt2020/
├── README.md
├── enp0s3-monday-pvt.pcap_Flow.csv
├── enp0s3-monday.pcap_Flow.csv
└── [additional PCAP flow files]
```

### 4. Make Scripts Executable
```bash
cd analysis/sae/
chmod +x run_sae.sh run_sae_pipeline.py
```

## ✅ Validation Checklist

Before running the pipeline, verify these critical requirements:

### Required Directories with Content:
- [ ] `datasets/bigbase/` contains 50 CSV files (dataset-01.csv to dataset-50.csv) + reports/
- [ ] `datasets/unraveled/` contains host-logs/, network-flows/, nids/ subdirectories
- [ ] `datasets/dapt2020/` contains PCAP flow CSV files + util.py
- [ ] `analysis/sae/` contains all 5 pipeline scripts + orchestrator + shell wrapper
- [ ] `analysis/lstm-sae/` contains LSTM-SAE pipeline scripts + orchestrator + early stopping
- [ ] `analysis/gru-sae/` contains GRU-SAE pipeline scripts + orchestrator + early stopping
- [ ] `analysis/iforest/` contains Isolation Forest pipeline scripts + orchestrator
- [ ] `analysis/comparative-evaluation/` contains cross-model analysis scripts
- [ ] `analysis/config/` contains all model-specific JSON configuration files
- [ ] `analysis/utils/` contains paths.py, config.py, encoding.py

### Docker Environment:
- [ ] Docker container has all required packages (tensorflow, scikit-learn, pandas, numpy, etc.)
- [ ] Container has access to project directory

### File Permissions:
- [ ] Pipeline scripts are executable (chmod +x)

## 🚦 Quick Start Test

Once everything is set up, test with:

```bash
# Navigate to SAE directory
cd analysis/sae/

# Test pipeline in Docker container
./run_sae.sh bigbase v1

# Or test other models
cd ../lstm-sae/
python run_lstm_sae_pipeline.py --dataset unraveled --version v1

cd ../iforest/
./run_iforest.sh bigbase v1
```

## 📝 Notes

- **Auto-created directories**: `data/`, `models/`, `artifacts/` are created by setup script
- **Multi-model support**: This structure supports SAE, LSTM-SAE, GRU-SAE, and Isolation Forest models
- **Critical uploads**: Only `datasets/` and code files need manual upload
- **Docker environment**: All Python dependencies should be pre-installed in the container
- **Pipeline orchestration**: Each model has its own pipeline orchestrator (SAE has shell wrapper)
- **Configuration files**: Include model-specific suffixes (-sae, -lstm-sae, -gru-sae, -iforest) for clarity

## 🐛 Common Issues

1. **"Raw data directory not found"**: Check that datasets are in exact paths shown above
2. **"Module not found"**: Ensure script runs from project root and Docker container has Python dependencies
3. **"Permission denied"**: Run `chmod +x` on shell scripts (run_sae.sh)
4. **Shell script syntax errors**: Ensure proper bash comment syntax (# not """)
5. **"Config file not found"**: Use exact config file names with model suffixes (-sae, -lstm)

## 📊 Expected Storage Requirements

- **Raw datasets**: 2-5 GB (depending on dataset size)
- **Processed data**: 1-3 GB per model/dataset combination
- **Trained models**: 100-500 MB per model
- **Evaluation artifacts**: 50-200 MB per model
- **Logs**: 10-50 MB per pipeline run
- **Total**: 10-20 GB for complete pipeline with multiple models