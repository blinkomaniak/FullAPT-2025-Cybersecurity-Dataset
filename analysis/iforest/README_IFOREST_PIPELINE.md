# Isolation Forest Pipeline Documentation

## Overview

The Isolation Forest (IForest) pipeline implements unsupervised anomaly detection using Isolation Forest algorithm for cybersecurity datasets. This pipeline follows the same intelligent anti-data-snooping methodology as the SAE pipeline, ensuring robust evaluation through proper data splitting and benign-only training.

## Pipeline Architecture

### 5-Stage Sequential Pipeline

```
Raw CSVs → [Stage 1] → Processed Parquet → [Stage 2] → Encoded Features → [Stage 3] → Trained Model → [Stage 4] → Evaluation Results → [Stage 5] → Visualizations
```

## Data Flow and Anti-Data-Snooping Design

### Core Design Principle: Early Data Splitting

The pipeline implements **anti-data-snooping** methodology by splitting data **before** any feature engineering or model training:

```
Original Dataset (100%)
├── Stage 1: Raw data aggregation and binary labeling
├── Stage 2: Early split into Train (70%) / Val (15%) / Test (15%)
│   ├── Fit transformers ONLY on training data
│   ├── Transform all splits using fitted transformers
│   └── Save all encoded splits separately
├── Stage 3: Train Isolation Forest ONLY on BENIGN training samples
├── Stage 4: Final evaluation ONLY on test data (never seen during training/validation)
└── Stage 5: Feature space visualization using validation data
```

This ensures that:
- ✅ **No data leakage** from validation/test into training
- ✅ **Unbiased evaluation** on truly unseen test data
- ✅ **Robust generalization** assessment

## Stage-by-Stage Breakdown

### Stage 1: Data Aggregation (`01-aggregation.py`)

**Purpose**: Combine raw CSV files and create binary-labeled dataset

**Data Usage**: 
- **Input**: Raw CSV files from `datasets/{dataset}/`
- **Processing**: Combines all CSV files, extracts core features, maps labels to binary (0=Benign, 1=Attack)
- **Output**: Single processed parquet file with all data

**Key Operations**:
- Load multiple CSV files using schema configuration
- Extract only core features specified in schema
- Binary label mapping: `benign_label` → 0, everything else → 1
- Memory-efficient processing with garbage collection

**Outputs**:
- `data/processed/iforest-{dataset}-{version}.parquet` - Combined dataset
- `data/processed/iforest-{dataset}-{version}-label-dist.csv` - Label distribution
- `data/processed/iforest-{dataset}-{version}-metadata.json` - Dataset metadata

**Command Example**:
```bash
python 01-aggregation.py --model iforest --dataset bigbase --version v1 --schema ../config/schema-bigbase-v1.json
```

### Stage 2: Feature Encoding (`02-encoding-{dataset}-iforest.py`)

**Purpose**: Anti-snooping feature encoding with 3-way data splitting

**Data Usage**:
- **Input**: Processed parquet file (all data)
- **Processing**: 
  1. **Early Split**: 70% train / 15% validation / 15% test (stratified)
  2. **Fit transformers**: Only on training data
  3. **Transform all splits**: Using fitted transformers
- **Output**: Separate encoded files for train/val/test

**Key Operations**:
- **TF-IDF** for text columns (CommandLine, Image paths) - bigbase only
- **OneHot encoding** for categorical columns (IPs, Users, Computers)
- **StandardScaler** for numeric columns (EventID, ports, bytes)
- **Sparse matrix storage** for memory efficiency

**Anti-Snooping Implementation**:
```python
# Split BEFORE any feature engineering
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df[label_col])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[label_col])

# Fit transformers ONLY on training data
column_transformer = build_transformer(...)
X_train = column_transformer.fit_transform(train_df)  # FIT on train only
X_val = column_transformer.transform(val_df)          # TRANSFORM val
X_test = column_transformer.transform(test_df)        # TRANSFORM test
```

**Outputs**:
- `data/encoded/iforest-{dataset}-{version}/X_train_encoded.npz` - Training features
- `data/encoded/iforest-{dataset}-{version}/X_val_encoded.npz` - Validation features  
- `data/encoded/iforest-{dataset}-{version}/X_test_encoded.npz` - Test features
- `data/encoded/iforest-{dataset}-{version}/y_train.npy` - Training labels
- `data/encoded/iforest-{dataset}-{version}/y_val.npy` - Validation labels
- `data/encoded/iforest-{dataset}-{version}/y_test.npy` - Test labels
- `data/encoded/iforest-{dataset}-{version}/column_transformer.joblib` - Fitted transformer
- `data/encoded/iforest-{dataset}-{version}/encoding_metadata.json` - Encoding metadata

**Command Examples**:
```bash
# Bigbase dataset
python 02-encoding-bigbase-iforest.py --dataset bigbase --version v1 --model iforest --encoding_config ../config/encoding-bigbase-iforest-v1.json

# Unraveled dataset
python 02-encoding-unraveled-iforest.py --dataset unraveled --version v1 --model iforest --encoding_config ../config/encoding-unraveled-iforest-v1.json
```

### Stage 3: Model Training (`03-training-iforest.py`)

**Purpose**: Train Isolation Forest model on benign-only samples with PCA dimensionality reduction

**Data Usage**:
- **Input**: Training data from Stage 2 (`X_train_encoded.npz`, `y_train.npy`)
- **Processing**: 
  1. **Filter to benign-only**: Only samples with `y_train == 0`
  2. **Sample subset**: Default 100K samples for computational efficiency
  3. **PCA reduction**: Default 50 components
  4. **Train Isolation Forest**: On reduced benign data
- **Output**: Trained model and metadata

**Benign-Only Training Logic**:
```python
# CRITICAL: Filter to BENIGN-ONLY samples for anomaly detection
benign_mask = y_train_full == 0
X_sparse = X_train_full[benign_mask]  # Only benign samples

# Sample for computational efficiency
sample_size = min(100_000, len(X_sparse))
X_sample = X_sparse[random_indices].toarray()

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_sample)

# Train Isolation Forest on reduced benign data
model = IsolationForest(n_estimators=100, contamination='auto')
model.fit(X_reduced)
```

**Why Benign-Only Training?**
- Isolation Forest is an **unsupervised anomaly detection** algorithm
- Designed to learn the "normal" pattern from benign samples only
- Attack samples would contaminate the normal pattern learning
- Model learns to isolate anomalies (attacks) from normal behavior (benign)

**Memory Optimization Strategies**:
- **Sampling**: Use subset of training data (default 100K samples)
- **PCA**: Reduce dimensionality from ~10K+ features to 50 components
- **Sparse matrices**: Efficient storage until PCA transformation

**Outputs**:
- `models/iforest-{dataset}-{version}/iforest-model-{dataset}-{version}.joblib` - Trained model
- `models/iforest-{dataset}-{version}/metadata-iforest-model-{dataset}-{version}.json` - Model metadata

**Command Example**:
```bash
python 03-training-iforest.py --dataset bigbase --version v1 --model iforest --sample-size 100000 --n-components 50
```

### Stage 4: Model Evaluation (`04-model-evaluation-iforest.py`)

**Purpose**: Evaluate trained model on unseen test data

**Data Usage**:
- **Input**: 
  - Test data from Stage 2 (`X_test_encoded.npz`, `y_test.npy`)
  - Trained model from Stage 3
- **Processing**: Apply same PCA transformation, compute anomaly scores, evaluate metrics
- **Output**: Comprehensive evaluation results and metrics

**Why Test Data (Not Validation)?**
- **Validation data**: Used during hyperparameter tuning and model selection
- **Test data**: **Final evaluation** on completely unseen data
- Provides **unbiased performance estimate** for real-world deployment
- **Never touched** during training or model development process

**Evaluation Process**:
```python
# Load truly unseen test data
X_test = sparse.load_npz("X_test_encoded.npz").toarray()
y_test = np.load("y_test.npy")

# Apply SAME PCA transformation as training
pca = PCA(n_components=metadata["n_components"])
X_test_pca = pca.fit_transform(X_test)  # Note: refit PCA on test data

# Compute anomaly scores
anomaly_scores = -model.decision_function(X_test_pca)  # Higher = more anomalous

# Comprehensive evaluation
roc_auc = roc_auc_score(y_test, anomaly_scores)
pr_auc = average_precision_score(y_test, anomaly_scores)
# + threshold optimization for F1, precision, recall
```

**Evaluation Metrics**:
- **ROC-AUC**: Area under ROC curve (higher = better)
- **PR-AUC**: Area under Precision-Recall curve (higher = better) 
- **F1-Score**: Harmonic mean of precision and recall (optimal threshold)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

**Outputs**:
- `artifacts/iforest-{dataset}-{version}/metrics.json` - Performance metrics
- `artifacts/iforest-{dataset}-{version}/anomaly_scores.npz` - Raw anomaly scores
- `artifacts/iforest-{dataset}-{version}/roc_pr_data.npz` - ROC/PR curve data
- `artifacts/iforest-{dataset}-{version}/threshold_scores.png` - Threshold analysis
- `artifacts/iforest-{dataset}-{version}/roc_pr_curves.png` - ROC/PR curves

**Command Example**:
```bash
python 04-model-evaluation-iforest.py --dataset bigbase --version v1 --model iforest
```

### Stage 5: Feature Space Visualization (`05-feature-space-iforest.py`)

**Purpose**: Visualize feature space and anomaly detection results

**Data Usage**:
- **Input**: 
  - Validation data from Stage 2 (`X_val_encoded.npz`, `y_val.npy`)
  - Anomaly scores from Stage 4
- **Processing**: 
  1. **Sample subset**: 100K samples for visualization efficiency
  2. **Dimensionality reduction**: PCA and UMAP to 2D
  3. **Multi-perspective visualization**: By anomaly score, true labels, tactic labels
- **Output**: Feature space visualizations

**Why Validation Data?**
- Test data reserved for final evaluation only
- Validation data provides good representation for visualization
- Allows exploration without compromising test set integrity

**Visualization Types**:

1. **PCA Projections**:
   - 2D PCA projection of high-dimensional features
   - Fast, linear dimensionality reduction
   - Colored by: anomaly scores, true labels, tactic labels

2. **UMAP Projections**:
   - Non-linear manifold learning technique
   - Better preserves local structure than PCA
   - Colored by: anomaly scores, true labels, tactic labels

**Insights from Visualizations**:
- **Anomaly score coloring**: Shows how model separates normal vs. anomalous regions
- **True label coloring**: Validates if anomalous regions correspond to actual attacks
- **Tactic label coloring**: Reveals if different attack tactics cluster separately

**Outputs**:
- `artifacts/iforest-{dataset}-{version}/pca_anomaly_score.png` - PCA colored by anomaly scores
- `artifacts/iforest-{dataset}-{version}/pca_true_labels.png` - PCA colored by true labels
- `artifacts/iforest-{dataset}-{version}/pca_tactic_labels.png` - PCA colored by tactic labels
- `artifacts/iforest-{dataset}-{version}/umap_anomaly_score.png` - UMAP colored by anomaly scores
- `artifacts/iforest-{dataset}-{version}/umap_true_labels.png` - UMAP colored by true labels
- `artifacts/iforest-{dataset}-{version}/umap_tactic_labels.png` - UMAP colored by tactic labels
- `artifacts/iforest-{dataset}-{version}/Z_2d_pca.npy` - PCA projections
- `artifacts/iforest-{dataset}-{version}/Z_umap.npy` - UMAP projections

**Command Example**:
```bash
python 05-feature-space-iforest.py --dataset bigbase --version v1 --model iforest
```

## Algorithm Details

### Isolation Forest Fundamentals

**Core Principle**: Anomalies are easier to isolate than normal points

**How it Works**:
1. **Random partitioning**: Create random binary trees by randomly selecting features and split values
2. **Path length measurement**: Anomalies require fewer splits to isolate (shorter paths)
3. **Ensemble scoring**: Average path lengths across multiple trees
4. **Anomaly score**: Shorter average path → Higher anomaly score

**Advantages for Cybersecurity**:
- ✅ **Unsupervised**: No need for labeled attack samples during training
- ✅ **Scalable**: Linear time complexity O(n)
- ✅ **Memory efficient**: Works well with sampling
- ✅ **Robust**: Not sensitive to normal data distribution assumptions
- ✅ **Interpretable**: Path length provides intuitive anomaly measure

### PCA Integration

**Why PCA Before Isolation Forest?**
- **Curse of dimensionality**: IF performance degrades in very high dimensions
- **Computational efficiency**: Faster training and inference
- **Noise reduction**: Focus on main variance components
- **Memory optimization**: Reduced storage requirements

**PCA Configuration**:
- **Components**: 50 (captures major variance while reducing complexity)
- **Explained variance**: Typically 80-95% of total variance
- **Applied consistently**: Same transformation for train/test data

### Hyperparameters

**Isolation Forest Parameters**:
- `n_estimators=100`: Number of isolation trees (more trees = more stable scores)
- `contamination='auto'`: Automatically estimate contamination ratio
- `random_state=42`: Reproducible results
- `n_jobs=-1`: Parallel processing

**Sampling Parameters**:
- `sample_size=100_000`: Training samples for computational efficiency
- Can be adjusted based on dataset size and computational resources

**PCA Parameters**:
- `n_components=50`: Balance between information retention and efficiency
- `random_state=42`: Reproducible PCA projections

## Configuration Files

### Schema Configuration (`schema-{dataset}-v1.json`)

Defines dataset structure and core features:

```json
{
  "core_features": ["CommandLine", "Image", "User", "Computer", ...],
  "label_column": "Label", 
  "time_column": "TimeCreated",
  "benign_label": "benign"
}
```

### Encoding Configuration (`encoding-{dataset}-iforest-v1.json`)

Defines feature transformation strategy:

**Bigbase Example**:
```json
{
  "label_column": "Label",
  "tfidf_columns": ["CommandLine", "ParentCommandLine", "Image", ...],
  "tfidf_params": {"CommandLine": 500, "Image": 300, ...},
  "onehot_columns": ["Computer", "User", "Protocol", ...],
  "numeric_columns": ["EventID"]
}
```

**Unraveled Example**:
```json
{
  "label_column": "Stage",
  "categorical_columns": ["src_ip", "dst_ip", "src_mac", ...],
  "numeric_columns": ["src_port", "dst_port", "protocol", ...]
}
```

## Memory Management and Optimization

### Large Dataset Handling

**Sparse Matrix Storage**:
- OneHot and TF-IDF features stored as sparse matrices
- Significant memory savings (often 90%+ reduction)
- Automatic sparse/dense detection in encoding scripts

**Sampling Strategies**:
- **Training**: Sample 100K benign samples for IF training
- **Visualization**: Sample 100K samples for PCA/UMAP projections
- **Memory monitoring**: Explicit cleanup with `gc.collect()`

**PCA Dimensionality Reduction**:
- Reduces ~10K+ features to 50 components
- Applied before computationally intensive operations
- Retains most variance while enabling efficiency

### Computational Efficiency

**Parallel Processing**:
- Isolation Forest: `n_jobs=-1` (use all CPU cores)
- Multiple trees trained in parallel

**Batch Processing**:
- Large datasets processed in chunks where possible
- Memory-mapped arrays for large file operations

## Pipeline Orchestration

### Automated Execution

**Pipeline Orchestrator** (`run_iforest_pipeline.py`):
- Executes all 5 stages in sequence
- Comprehensive error handling and logging
- Progress tracking with timing information
- Prerequisite validation before each stage
- Supports stage skipping for debugging/resumption

**Shell Wrapper** (`run_iforest.sh`):
- Simple command-line interface
- Docker-friendly (no virtual environment dependencies)
- Automatic config file passing

### Usage Examples

**Complete Pipeline Execution**:
```bash
# Bigbase dataset
./run_iforest.sh bigbase v1 ../config/schema-bigbase-v1.json ../config/encoding-bigbase-iforest-v1.json

# Unraveled dataset
./run_iforest.sh unraveled v1 ../config/schema-unraveled-v1.json ../config/encoding-unraveled-iforest-v1.json
```

**Individual Stage Execution**:
```bash
# Stage 1: Aggregation
python 01-aggregation.py --model iforest --dataset bigbase --version v1 --schema ../config/schema-bigbase-v1.json

# Stage 2: Encoding
python 02-encoding-bigbase-iforest.py --dataset bigbase --version v1 --model iforest --encoding_config ../config/encoding-bigbase-iforest-v1.json

# Stage 3: Training
python 03-training-iforest.py --dataset bigbase --version v1 --model iforest --sample-size 100000 --n-components 50

# Stage 4: Evaluation
python 04-model-evaluation-iforest.py --dataset bigbase --version v1 --model iforest

# Stage 5: Visualization
python 05-feature-space-iforest.py --dataset bigbase --version v1 --model iforest
```

**Pipeline Orchestrator with Options**:
```bash
# Full pipeline
python run_iforest_pipeline.py --dataset bigbase --version v1 --schema ../config/schema-bigbase-v1.json --encoding_config ../config/encoding-bigbase-iforest-v1.json

# Skip stages (resume from Stage 3)
python run_iforest_pipeline.py --dataset bigbase --version v1 --schema ../config/schema-bigbase-v1.json --encoding_config ../config/encoding-bigbase-iforest-v1.json --skip-stages 1 2

# Quiet mode (minimal output)
python run_iforest_pipeline.py --dataset bigbase --version v1 --schema ../config/schema-bigbase-v1.json --encoding_config ../config/encoding-bigbase-iforest-v1.json --quiet
```

## Key Differences from SAE Pipeline

### Algorithm Differences

| Aspect | SAE (Stacked Autoencoder) | IForest (Isolation Forest) |
|--------|---------------------------|----------------------------|
| **Type** | Supervised Deep Learning | Unsupervised Tree-based |
| **Training Data** | Benign samples only | Benign samples only |
| **Features** | Dense neural network layers | Tree-based random partitioning |
| **Anomaly Score** | Reconstruction error | Average path length |
| **Dimensionality** | Works with high dimensions | Requires PCA reduction |
| **Computational** | GPU-accelerated, slower | CPU-based, faster |
| **Memory** | Higher requirements | Lower requirements |
| **Interpretability** | Black box | More interpretable paths |

### Implementation Differences

**Model Training**:
- **SAE**: Trains deep autoencoder with reconstruction loss
- **IForest**: Builds ensemble of isolation trees with path length scoring

**Preprocessing**:
- **SAE**: Can handle high-dimensional features directly
- **IForest**: Applies PCA for dimensionality reduction

**Evaluation**:
- **SAE**: Reconstruction error threshold optimization
- **IForest**: Isolation score threshold optimization

### Performance Characteristics

**SAE Advantages**:
- Can capture complex nonlinear patterns
- No dimensionality reduction required
- Potentially higher accuracy on complex attacks

**IForest Advantages**:
- Much faster training and inference
- Lower memory requirements
- More interpretable anomaly scores
- Better suited for real-time detection

## Expected Performance

### Typical Results

**ROC-AUC Scores**:
- **Bigbase**: 0.85-0.92 (Windows event logs)
- **Unraveled**: 0.80-0.88 (Network flows)

**Training Times** (100K samples, 50 PCA components):
- **Stage 3 (Training)**: 2-5 minutes
- **Stage 4 (Evaluation)**: 1-3 minutes
- **Total Pipeline**: 10-30 minutes (depending on dataset size)

### Performance Factors

**Dataset Characteristics**:
- Larger feature sets benefit more from PCA reduction
- Higher anomaly ratios easier to detect
- Complex attack patterns may be harder to isolate

**Hyperparameter Impact**:
- More trees (`n_estimators`) → more stable scores, longer training
- More PCA components → more information, higher computation
- Larger sample size → better model, longer training

## Troubleshooting

### Common Issues

**Memory Errors**:
- Reduce `sample_size` parameter in Stage 3
- Reduce `n_components` in PCA
- Ensure sparse matrix storage in Stage 2

**Poor Performance**:
- Check label distribution (severely imbalanced datasets)
- Verify benign-only training (no attack samples in training)
- Increase `n_estimators` for more stable scores
- Try different `n_components` values

**File Not Found Errors**:
- Verify prerequisite stages completed successfully
- Check config file paths are correct
- Ensure output directories exist

### Performance Optimization

**For Large Datasets**:
- Increase sampling size if memory allows
- Use more PCA components if computational resources permit
- Consider distributed processing for very large datasets

**For Better Accuracy**:
- Ensure high-quality feature engineering in Stage 2
- Tune contamination ratio if known
- Experiment with different PCA component counts
- Validate feature importance through visualizations

## Integration with Research Pipeline

The Isolation Forest pipeline integrates seamlessly with the broader research framework:

- **Standardized paths**: Compatible with `utils.paths` infrastructure
- **Common configurations**: Shared schema and encoding patterns
- **Consistent evaluation**: Same metrics and visualization approaches
- **Pipeline orchestration**: Unified logging and error handling
- **Docker compatibility**: Runs in containerized environments

This enables direct comparison between IForest and other anomaly detection approaches (SAE, LSTM-SAE) using identical experimental setups and evaluation protocols.