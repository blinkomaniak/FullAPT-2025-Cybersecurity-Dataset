# SAE Pipeline Execution Guide

This directory contains the complete SAE (Stacked Autoencoder) pipeline for cybersecurity anomaly detection.

## 🚀 Quick Start (Remote Server)

### Option 1: Using Shell Script (Recommended)
```bash
# Make scripts executable (first time only)
chmod +x run_sae.sh run_sae_pipeline.py

# Run for bigbase dataset
./run_sae.sh bigbase v1 ../config/schema-bigbase-sae-v1.json ../config/encoding-bigbase-sae-v1.json

# Run for unraveled dataset  
./run_sae.sh unraveled v1 ../config/schema-unraveled-sae-v1.json ../config/encoding-unraveled-sae-v1.json
```

### Option 2: Using Python Script Directly
```bash
# Run pipeline with required config files
python run_sae_pipeline.py --dataset bigbase --version v1 --schema ../config/schema-bigbase-sae-v1.json --encoding_config ../config/encoding-bigbase-sae-v1.json

python run_sae_pipeline.py --dataset unraveled --version v1 --schema ../config/schema-unraveled-sae-v1.json --encoding_config ../config/encoding-unraveled-sae-v1.json
```

## 📊 Pipeline Stages

The pipeline executes 5 stages in order, with specific usage of train/val/test splits:

### 1. **Stage 1: Data Aggregation** (`01-aggregation.py`)
**Data Usage**: Full dataset (no splits yet)
- **Function**: Combines raw CSV files from multiple sources
- **Processing**: Maps labels to binary (0=Benign, 1=Attack) 
- **Input**: Raw CSV files from `datasets/{dataset}/`
- **Output**: Single processed parquet file with all data
- **Split Status**: 🔄 No splitting - processes entire dataset as one unit

### 2. **Stage 2: Data Encoding** (`02-encoding-*-sae.py`) 
**Data Usage**: Creates train/val/test splits and encodes each separately
- **Splitting Strategy**: 70% train, 15% validation, 15% test (stratified by label)
- **Anti-Data-Snooping**: Splits data BEFORE fitting transformers
- **Training Data**: Used to fit column transformers (OneHot + StandardScaler)
- **Validation Data**: Transformed using fitted transformers (no re-fitting)
- **Test Data**: Transformed using fitted transformers (no re-fitting)
- **Function**: `train_test_split()` with stratification to preserve label distribution
- **Outputs**: 
  - `X_train_encoded.npz`, `y_train.npy` (70% - for model training)
  - `X_val_encoded.npz`, `y_val.npy` (15% - for validation during training)
  - `X_test_encoded.npz`, `y_test.npy` (15% - for final evaluation)
  - `column_transformer.joblib` (fitted on training data only)

### 3. **Stage 3: Model Training** (`03-training-sae.py`)
**Data Usage**: Training data (benign-only) + validation data (full) 
- **Training Set**: Filters `X_train` to **benign samples only** (y_train == 0)
  - **Rationale**: Anomaly detection - model learns normal patterns only
  - **Usage**: Trains autoencoder to reconstruct benign data
- **Validation Set**: Uses **full validation set** (benign + attack)
  - **Usage**: Monitors training progress and early stopping
  - **Metrics**: Validation loss on mixed benign/attack samples
- **Test Set**: ❌ **Not used** in this stage
- **Architecture**: Input → 1024 → 512 → 128 → 512 → 1024 → Input  
- **Training Strategy**: Benign-only training with full validation monitoring
- **Outputs**: Trained SAE model + training history + metadata

### 4. **Stage 4: Model Evaluation** (`04-model-evaluation-sae.py`)
**Data Usage**: Test data only (final evaluation)
- **Test Set**: Uses **full test set** (benign + attack samples)
  - **Usage**: Computes reconstruction errors for all test samples
  - **Anomaly Scoring**: Higher reconstruction error = more anomalous
- **Training Set**: ❌ **Not used** in this stage  
- **Validation Set**: ❌ **Not used** in this stage
- **Evaluation Process**:
  1. Load test data (`X_test`, `y_test`)
  2. Compute reconstruction errors using trained SAE
  3. Use reconstruction error as anomaly score
  4. Evaluate classification performance (ROC-AUC, PR-AUC, etc.)
- **Outputs**: Performance metrics + ROC/PR curves + threshold analysis

### 5. **Stage 5: Visualization** (`05-result-vis.py`)
**Data Usage**: Test data only (for visualization)
- **Test Set**: Uses **sampled test data** (max 50K samples for performance)
  - **Usage**: Creates latent space representations via encoder
  - **Visualization**: PCA and UMAP projections colored by true labels
- **Training Set**: ❌ **Not used** in this stage
- **Validation Set**: ❌ **Not used** in this stage  
- **Visualization Process**:
  1. Load test data and trained model
  2. Extract encoder (first half of autoencoder)
  3. Generate latent representations for test samples
  4. Apply dimensionality reduction (PCA, UMAP)
  5. Plot projections with benign/attack color coding
- **Outputs**: Latent space projection plots + analysis

## 🔄 Data Flow Summary

```
Stage 1: [Raw CSVs] → [Full Dataset.parquet]
                           ↓
Stage 2: [Full Dataset] → [Train(70%)] + [Val(15%)] + [Test(15%)]
                           ↓              ↓              ↓
Stage 3: [Train(benign-only)] + [Val(full)] → [Trained SAE Model]
                                              ↓
Stage 4:                                 [Test(full)] → [Performance Metrics]
                                              ↓  
Stage 5:                                 [Test(sampled)] → [Visualizations]
```

## 🎯 Key Design Principles

- **Anti-Data-Snooping**: Data splitting occurs before any preprocessing to prevent leakage
- **Anomaly Detection**: Model trains only on benign samples to learn normal patterns
- **Proper Evaluation**: Final evaluation uses completely unseen test data
- **Stratified Splitting**: Preserves original benign/attack ratio across all splits
- **Memory Efficiency**: Stage 5 samples test data to handle large datasets

## 🛠️ Advanced Usage

### Skip Stages (Resume Interrupted Runs)
```bash
# Skip stages 1-2 if already completed
python run_sae_pipeline.py --dataset bigbase --version v1 --schema ../config/schema-bigbase-sae-v1.json --encoding_config ../config/encoding-bigbase-sae-v1.json --skip-stages 1 2

# Skip only stage 1
python run_sae_pipeline.py --dataset bigbase --version v1 --schema ../config/schema-bigbase-sae-v1.json --encoding_config ../config/encoding-bigbase-sae-v1.json --skip-stages 1
```

### Quiet Mode (Minimal Output)
```bash
python run_sae_pipeline.py --dataset bigbase --version v1 --schema ../config/schema-bigbase-sae-v1.json --encoding_config ../config/encoding-bigbase-sae-v1.json --quiet
```

## 📝 Logging

All pipeline runs generate detailed logs:
- **Location**: `../../../logs/sae_pipeline_<dataset>_<version>_<timestamp>.log`
- **Content**: Command outputs, timing information, error messages
- **Format**: `[HH:MM:SS] LEVEL: message`

## 📂 Output Structure

After successful execution, you'll find:

```
data/
├── processed/               # Stage 1 outputs
│   └── <dataset>-sae-<version>.parquet
├── encoded/                 # Stage 2 outputs  
│   └── <dataset>-sae-<version>/
│       ├── X_train_encoded.npz
│       ├── X_val_encoded.npz
│       ├── X_test_encoded.npz
│       ├── y_train.npy, y_val.npy, y_test.npy
│       └── column_transformer.joblib
└── models/                  # Stage 3 outputs
    └── sae/<dataset>-<version>/
        ├── sae-model-<dataset>-<version>.keras
        ├── history-sae-model-<dataset>-<version>.json
        └── metadata-sae-model-<dataset>-<version>.json

artifacts/
└── eval/sae/<dataset>-<version>/    # Stage 4 & 5 outputs
    ├── metrics.json
    ├── roc_pr_curves.png
    ├── threshold_scores.png
    ├── pca_projection.png
    ├── umap_projection.png
    └── latent_space_projections.png
```

## ⚠️ Prerequisites

### System Requirements
- **Python**: 3.8+ with virtual environment activated
- **Memory**: 16GB+ RAM recommended (depends on dataset size)
- **Storage**: 10GB+ free space for artifacts
- **GPU**: Optional but recommended for faster training

### Data Requirements
Ensure raw datasets are in place:
```
datasets/
├── bigbase/               # 50 CSV files (dataset-01.csv to dataset-50.csv)
├── unraveled/
│   └── network-flows/     # Network flow CSV files by week/day
└── dapt2020/              # DAPT2020 PCAP flow files
```

### Environment Requirements
```bash
# For Docker environments - ensure packages are available
python -c "import tensorflow, sklearn, pandas, numpy; print('✅ All packages available')"

# For local environments with virtual environment
source ../../dataset-venv/bin/activate
```

## 🐛 Troubleshooting

### Common Issues

**1. "Raw data directory not found"**
- Check that datasets are properly extracted to `datasets/` directory
- Verify subdirectory structure for unraveled dataset

**2. "Python packages missing"**
- For Docker: Ensure container has required packages pre-installed
- For local: Activate virtual environment: `source ../../dataset-venv/bin/activate`
- Install missing packages: `pip install tensorflow scikit-learn pandas numpy`

**3. "GPU memory errors"**
- Reduce batch size in training script
- Add GPU memory growth configuration
- Use CPU-only training if needed

**4. "Pipeline timeout"**
- Individual stages timeout after 1 hour
- For large datasets, consider running stages individually
- Monitor system resources (RAM, disk space)

### Manual Stage Execution

If the orchestrator fails, you can run stages manually:

```bash
# Stage 1
python 01-aggregation.py --model sae --dataset bigbase --version v1 --schema ../config/schema-bigbase-sae-v1.json

# Stage 2  
python 02-encoding-bigbase-sae.py --dataset bigbase --version v1 --model sae --encoding_config ../config/encoding-bigbase-sae-v1.json

# Stage 3
python 03-training-sae.py --dataset bigbase --version v1 --model sae

# Stage 4
python 04-model-evaluation-sae.py --dataset bigbase --version v1 --model sae

# Stage 5
python 05-result-vis.py --dataset bigbase --version v1 --model sae
```

## 📊 Expected Runtime

Approximate execution times (may vary by system):

| Dataset   | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | Total   |
|-----------|---------|---------|---------|---------|---------|---------|
| bigbase   | 5 min   | 10 min  | 30 min  | 15 min  | 10 min  | ~70 min |
| unraveled | 3 min   | 8 min   | 25 min  | 12 min  | 8 min   | ~55 min |

## 🎯 Success Indicators

Pipeline completed successfully when you see:
- ✅ All 5 stages complete without errors
- 📊 Final metrics in `artifacts/eval/sae/<dataset>-<version>/metrics.json`
- 📈 Visualization plots generated
- 🎉 "Pipeline completed successfully!" message

## 📚 Further Information

- **Pipeline Architecture**: See main project CLAUDE.md
- **Model Details**: Check training metadata JSON files
- **Performance Analysis**: Review evaluation metrics and plots
- **Troubleshooting**: Check detailed logs in `logs/` directory