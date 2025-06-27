# GRU-SAE Pipeline Documentation

## Overview

The GRU-SAE (Gated Recurrent Unit Stacked Autoencoder) pipeline implements temporal anomaly detection for cybersecurity datasets. As an alternative to LSTM-SAE, GRU-SAE offers faster training and more efficient memory usage while maintaining similar temporal modeling capabilities for security event sequences.

### Key Features
- **Efficient temporal modeling**: GRU layers process sequences with fewer parameters than LSTM
- **Faster training**: Reduced computational complexity enables quicker model iteration
- **Memory efficient**: Lower memory footprint suitable for large cybersecurity datasets
- **Smart early stopping**: Intelligent training termination with improvement rate analysis
- **Automated pipeline**: Complete 5-stage automation with error handling and flow control
- **Anti-data-snooping**: Early 3-way data splitting (train/val/test) prevents information leakage
- **Benign-only training**: Unsupervised anomaly detection trained exclusively on normal samples
- **APT-aware splitting**: Campaign-based data splitting for bigbase dataset
- **Configurable architecture**: User-defined GRU encoder/decoder structures
- **Production-ready**: Robust error handling, memory management, and scalable processing

## GRU vs LSTM Advantages

### Computational Efficiency
- **Fewer parameters**: GRU has 2 gates vs LSTM's 3 gates (forget, input, output)
- **Faster training**: 25-30% reduction in training time per epoch
- **Lower memory usage**: Reduced memory footprint for large sequence datasets

### Gradient Flow
- **Better gradient propagation**: Simpler gating mechanism reduces vanishing gradients
- **Stable training**: Less prone to exploding gradients in deep architectures
- **Convergence speed**: Often converges faster than LSTM on cybersecurity sequences

### Temporal Modeling
- **Similar performance**: Comparable anomaly detection capability to LSTM
- **Effective pattern capture**: Successfully learns temporal dependencies in security events
- **Robust to sequence length**: Maintains performance across various sequence lengths

## Pipeline Architecture

The GRU-SAE pipeline follows a 5-stage sequential architecture:

```
Stage 1: Data Aggregation → Stage 2: Sequence Encoding → Stage 3: Model Training → Stage 4: Evaluation → Stage 5: Visualization
```

Each stage produces artifacts consumed by subsequent stages, ensuring reproducible experiments.

## Stage-by-Stage Analysis

### Stage 1: Data Aggregation (`01-aggregation.py`)

**Purpose**: Combines raw CSV files into a unified dataset with proper labeling and metadata extraction.

**Key Functions**:

#### `get_apt_campaign(filename)`
Maps bigbase dataset files to APT campaigns for temporal analysis:
```python
def get_apt_campaign(filename):
    file_num = int(filename.split('dataset-')[1].split('.')[0])
    if 1 <= file_num <= 20: return "APT1"
    elif 21 <= file_num <= 30: return "APT2"
    # ... continues for APT3-APT6
```

#### DateTime Conversion Logic
Converts string timestamps to Unix epoch for numerical processing:
```python
if time_column == "UtcTime":
    df_all[time_column] = pd.to_datetime(df_all[time_column]).astype('int64') // 10**6
```

**Outputs**:
- `processed/{model}-{dataset}-{version}.parquet`: Combined dataset
- `metadata/{model}-{dataset}-{version}.json`: Schema and statistics
- `label_dist/{model}-{dataset}-{version}.csv`: Label distribution

**Dataset-Specific Handling**:
- **bigbase**: Adds APT_Campaign and Session_File metadata columns
- **unraveled**: Handles network flow subdirectories

### Stage 2: Sequence Encoding (`02-encoding-gru-sae.py`)

**Purpose**: Transforms tabular data into 3D sequences suitable for GRU processing with anti-data-snooping methodology.

**Key Functions**:

#### Early Data Splitting
```python
if DATASET == "bigbase" and 'APT_Campaign' in df.columns:
    TRAIN_APTS = ["APT1", "APT2", "APT3", "APT4"]  # 70%
    VAL_APTS = ["APT5"]                            # 15%
    TEST_APTS = ["APT6"]                           # 15%
```

#### Sparse-Aware Feature Selection
Handles high-dimensional sparse matrices without memory explosion:
```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=MAX_FEATURES, random_state=42)
X_train_reduced = svd.fit_transform(X_train_encoded)  # Works on sparse matrices
```

#### Memory Management
```python
if estimated_memory_gb > MEMORY_LIMIT_GB:
    reduction_factor = MEMORY_LIMIT_GB / estimated_memory_gb
    new_sample_size = int(sample_size * reduction_factor)
```

#### Sequence Creation
```python
def create_sequences_batched(data, labels, seq_len, batch_size=1000):
    for batch_start in range(0, len(data) - seq_len + 1, batch_size):
        batch_sequences = []
        for i in range(batch_start, min(batch_start + batch_size, len(data) - seq_len + 1)):
            sequence = data[i:i + seq_len]
            batch_sequences.append(sequence)
        yield np.array(batch_sequences), labels[batch_start + seq_len - 1:batch_start + len(batch_sequences) + seq_len - 1]
```

**Feature Engineering**:
- **TF-IDF**: Text columns (CommandLine, Image paths) → sparse vectors
- **OneHot**: Categorical columns (User, Protocol) → binary encoding
- **StandardScaler**: Numeric columns (EventID, ProcessId, UtcTime) → normalized values

**Outputs**:
- `X_train_sequences.npy`: 3D training sequences (samples × timesteps × features)
- `X_val_sequences.npy`: 3D validation sequences
- `X_test_sequences.npy`: 3D test sequences
- `y_train.npy`, `y_val.npy`, `y_test.npy`: Corresponding labels
- `feature_names.json`: Feature mapping for interpretability

### Stage 3: Model Training (`03-training-gru-sae.py`)

**Purpose**: Trains GRU autoencoder on benign sequences to learn normal behavioral patterns with intelligent early stopping.

**Enhanced Command Line Arguments**:
```bash
--stopping-strategy {standard,smart,adaptive}  # Early stopping strategy (default: smart)
--min-improvement FLOAT                        # Minimum improvement % for smart stopping (default: 0.5)
--patience INT                                 # Epochs to wait for improvement (default: 5)
--encoder-units INT [INT ...]                  # GRU encoder units (default: [256, 128])
--decoder-units INT [INT ...]                  # GRU decoder units (default: [128, 256])
--max-train INT                                # Max training samples for memory management
--max-val INT                                  # Max validation samples for memory management
```

**Architecture Functions**:

#### `build_gru_sae(input_dim, seq_len, encoder_units, decoder_units, dropout_rate)`
Creates configurable GRU autoencoder with user-defined architecture:
```python
# Encoder: Compresses sequences to latent representation
for i, units in enumerate(encoder_units):
    return_sequences = (i < len(encoder_units) - 1)
    x = GRU(units, return_sequences=return_sequences, dropout=dropout_rate)(x)

# Bridge: Repeats encoded vector for decoder input
x = RepeatVector(seq_len)(encoded)

# Decoder: Reconstructs original sequences
for i, units in enumerate(decoder_units):
    x = GRU(units, return_sequences=True, dropout=dropout_rate)(x)

# Output: Sequence reconstruction
outputs = TimeDistributed(Dense(input_dim, activation='linear'))(x)
```

**GRU Architecture Benefits**:
- **Simplified gating**: GRU uses reset and update gates (vs LSTM's 3 gates)
- **Fewer parameters**: Typically 25% fewer parameters than equivalent LSTM
- **Faster computation**: Reduced matrix operations per timestep
- **Better gradient flow**: Less complex internal state management

**Training Strategy**:
- **Benign-only**: Filters training data to normal samples only
- **Reconstruction loss**: MSE between input and reconstructed sequences
- **Smart early stopping**: Intelligent termination based on improvement analysis
- **Gradient clipping**: Stabilizes GRU training (`clipnorm=1.0`)
- **Memory management**: Automatic sampling for large datasets

**Smart Early Stopping System**:
The training script integrates three stopping strategies:

1. **Smart Strategy** (Recommended):
   ```python
   SmartEarlyStopping(
       min_improvement_pct=0.5,        # Require ≥0.5% relative improvement
       patience=5,                     # Wait 5 epochs for significant improvement
       improvement_window=10,          # Analyze 10-epoch improvement windows
       min_window_improvement=2.0,     # Require 2% improvement over window
       min_improvement_rate=0.1        # Require 0.1% improvement per epoch
   )
   ```

2. **Adaptive Strategy**:
   ```python
   AdaptiveEarlyStopping(
       min_improvement_ratio=0.001,    # Dynamic threshold based on loss magnitude
       base_patience=5                 # Standard patience counter
   )
   ```

3. **Standard Strategy**:
   ```python
   EarlyStopping(
       monitor='val_loss', 
       patience=5, 
       restore_best_weights=True
   )
   ```

**Configurable Architecture Examples**:
```bash
# Default architecture
--encoder-units 256 128 --decoder-units 128 256

# Deep architecture  
--encoder-units 512 256 128 64 --decoder-units 64 128 256 512

# Wide architecture
--encoder-units 512 512 --decoder-units 512 512

# Shallow architecture (faster training)
--encoder-units 128 --decoder-units 128
```

**GPU Configuration**: Restricts to GPU 0 only for multi-GPU systems:
```python
# Configure GPU - restrict to GPU 0 only
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"🚀 Using GPU 0 only")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
```

**Outputs**:
- `{model}-model-{dataset}-{version}.keras`: Final trained model
- `{model}-model-{dataset}-{version}-best.keras`: Best validation model
- `history-{model}-model-{dataset}-{version}.json`: Training metrics
- `metadata-{model}-model-{dataset}-{version}.json`: Enhanced architecture details

### Stage 4: Model Evaluation (`04-model-evaluation-gru-sae.py`)

**Purpose**: Evaluates trained model on test sequences and computes anomaly scores.

**Key Functions**:

#### Batch Processing for Memory Efficiency
```python
def evaluate_sequences_batched(model, sequences, batch_size=32):
    reconstruction_errors = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        reconstructed = model.predict(batch, verbose=0)
        # MSE per sequence
        mse = np.mean((batch - reconstructed) ** 2, axis=(1, 2))
        reconstruction_errors.extend(mse)
    return np.array(reconstruction_errors)
```

#### Anomaly Score Computation
Reconstruction error serves as anomaly score:
- **Normal sequences**: Low reconstruction error (model learned pattern)
- **Anomalous sequences**: High reconstruction error (unfamiliar pattern)

**Metrics Computed**:
- ROC-AUC: Area under receiver operating characteristic curve
- Precision/Recall: At various thresholds
- F1-Score: Harmonic mean of precision and recall
- Average Precision: Area under precision-recall curve

**Outputs**:
- `evaluation_results.json`: Performance metrics
- `reconstruction_errors.npy`: Per-sample anomaly scores
- `predictions.npy`: Binary predictions at optimal threshold

### Stage 5: Result Visualization (`05-result-vis-gru-sae.py`)

**Purpose**: Creates comprehensive visualizations for model analysis and result interpretation.

**Key Visualization Functions**:

#### `extract_gru_encoder(model)`
Extracts encoder portion for latent space analysis:
```python
def extract_gru_encoder(model):
    encoder_layers = []
    for layer in model.layers:
        if 'repeat_vector' in layer.name:
            break
        encoder_layers.append(layer)
    # Creates encoder-only model for feature extraction
```

#### `plot_sequence_analysis(reconstruction_errors, y_test)`
Analyzes temporal patterns in anomaly scores:
- Distribution of reconstruction errors by class
- Threshold selection visualization
- Temporal trends in anomaly detection

#### `plot_latent_space_analysis(encoded_sequences, y_test)`
Visualizes compressed representations:
- t-SNE projection of encoded sequences
- Class separation in latent space
- Clustering analysis of normal vs. anomalous patterns

**Generated Plots**:
1. **ROC Curve**: True positive rate vs. false positive rate
2. **Precision-Recall Curve**: Precision vs. recall trade-offs
3. **Reconstruction Error Distribution**: Normal vs. anomalous patterns
4. **Latent Space Visualization**: t-SNE of encoded sequences
5. **Confusion Matrix**: Classification performance at optimal threshold
6. **Temporal Analysis**: Time-series view of anomaly scores

## Automated Pipeline Runner (`run_gru_sae_pipeline.py`)

The automated pipeline runner provides complete 5-stage automation with intelligent flow control, error handling, and enhanced user experience.

### Key Features

#### **Configuration Presets**
Automated configuration file selection based on dataset and experiment type:
```bash
--config-preset basic     # Uses schema-bigbase-gru-sae-v1.json + encoding-bigbase-gru-sae-v1.json
--config-preset extended  # Uses schema-bigbase-gru-sae-v2.json + encoding-bigbase-gru-sae-v2.json  
--config-preset network   # Uses schema-unraveled-gru-sae-v1.json + encoding-unraveled-gru-sae-v1.json
```

#### **Pipeline Flow Control**
```bash
--start-from 3            # Start from Stage 3 (skip data processing)
--stop-at 4               # Stop after Stage 4 (skip visualization)
--skip-stages 1 2         # Skip Stages 1 and 2 (data already processed)
```

#### **Enhanced Output Management**
- **Color-coded progress**: Visual indicators for each stage
- **Smart output filtering**: Reduces training verbosity while preserving important information
- **Real-time streaming**: Live progress updates from all stages
- **Error handling**: Continue pipeline execution on stage failures with user confirmation

#### **Complete Parameter Integration**
```bash
# Architecture control
--encoder-units 512 256 128 --decoder-units 128 256 512

# Training control  
--stopping-strategy smart --min-improvement 1.0 --patience 3

# Memory management
--sample-size 2000000 --max-features 500

# Pipeline control
--start-from 1 --stop-at 5 --epochs 200
```

### Usage Examples

#### **Complete Pipeline Execution**
```bash
# Basic bigbase experiment
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase \
  --version v3 \
  --config-preset basic \
  --seq-len 50 \
  --stopping-strategy smart \
  --min-improvement 0.5

# Extended bigbase with advanced features
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase \
  --version v4 \
  --config-preset extended \
  --seq-len 100 \
  --encoder-units 512 256 128 \
  --decoder-units 128 256 512 \
  --stopping-strategy smart \
  --min-improvement 1.0 \
  --patience 3

# Unraveled network flow analysis
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset unraveled \
  --version v1 \
  --config-preset network \
  --subdir network-flows \
  --seq-len 10 \
  --sample-size 50000
```

#### **Partial Pipeline Execution**
```bash
# Resume from training (data already processed)
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v5 --config-preset basic \
  --start-from 3

# Training and evaluation only (skip visualization)
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v6 --config-preset extended \
  --start-from 3 --stop-at 4

# Skip failed stages and continue
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v7 --config-preset basic \
  --skip-stages 1 2  # Use pre-processed data
```

## Configuration System

### Enhanced Configuration Structure

The pipeline now supports an extended configuration system with dataset-specific optimizations:

#### **Schema Configurations (`analysis/config/schema-*-gru-sae-*.json`)**
Define dataset structure and core features:

#### `schema-bigbase-gru-sae-v1.json`
```json
{
  "core_features": ["Computer", "User", "EventID", "CommandLine", ...],
  "label_column": "Label",
  "time_column": "UtcTime",
  "benign_label": "normal"
}
```

### Encoding Configurations (`analysis/config/encoding-*-gru-sae-*.json`)
Specify feature transformations for GRU processing:

#### `encoding-bigbase-gru-sae-v1.json`
```json
{
  "tfidf_columns": ["CommandLine", "ParentCommandLine", "Image", ...],
  "tfidf_params": {"CommandLine": 500, "Image": 300, ...},
  "categorical_columns": ["Computer", "User", "Protocol", ...],
  "numeric_columns": ["EventID", "ProcessId", "UtcTime"],
  "time_column": "UtcTime"
}
```

#### `encoding-bigbase-gru-sae-v2.json`
Extended version with additional TF-IDF features:
- `OriginalFileName`, `TargetObject`, `TargetFilename`, `PipeName`
- Higher feature dimensionality for richer representations

## Data Flow Architecture

```
Raw CSVs → Aggregation → Parquet Dataset
                ↓
        Early Data Splitting (APT-aware)
                ↓
    Feature Engineering (TF-IDF + OneHot + Scaling)
                ↓
        Sequence Creation (3D arrays)
                ↓
    Benign-only GRU-SAE Training
                ↓
        Test Set Evaluation
                ↓
    Visualization & Analysis
```

## Memory Management Strategies

### 1. Sparse Matrix Handling
- TF-IDF creates sparse matrices for text features
- TruncatedSVD works directly on sparse data without dense conversion
- Avoids memory explosion with high-dimensional text features

### 2. Batch Processing
- Sequence creation in batches to prevent memory overflow
- Model evaluation in batches for large test sets
- Progressive loading and processing of data

### 3. Automatic Sample Size Reduction
- Estimates memory requirements before sequence creation
- Automatically reduces sample size if memory limit exceeded
- Maintains representative sampling across all classes

### 4. Explicit Cleanup
```python
del large_variables
tf.keras.backend.clear_session()
gc.collect()
```

## Performance Considerations

### Model Efficiency
- **Reduced parameters**: GRU uses 25-30% fewer parameters than LSTM
- **Faster training**: 20-25% faster training time per epoch
- **Lower memory usage**: Reduced GPU memory footprint for larger batches
- **Better convergence**: Often converges faster than LSTM on cybersecurity data

### Training Efficiency
- **Batch size**: Can use larger batches (32-64) due to lower memory usage
- **Early stopping**: Prevents overfitting and reduces training time
- **Learning rate scheduling**: Adaptive reduction improves convergence
- **GPU optimization**: Single GPU restriction prevents resource conflicts

### Evaluation Scalability
- **Batch processing**: Handles large test sets without memory issues
- **Sparse operations**: Maintains efficiency with high-dimensional features
- **Progressive visualization**: Generates plots incrementally for large datasets

## Usage Examples

### Complete GRU-SAE Pipeline for Bigbase Dataset

```bash
# Stage 1: Data Aggregation
python analysis/gru-sae/01-aggregation.py \
  --model gru-sae \
  --dataset bigbase \
  --version v1 \
  --schema analysis/config/schema-bigbase-gru-sae-v1.json

# Stage 2: Sequence Encoding  
python analysis/gru-sae/02-encoding-gru-sae.py \
  --dataset bigbase \
  --version v1 \
  --model gru-sae \
  --encoding_config analysis/config/encoding-bigbase-gru-sae-v1.json \
  --seq-len 100 \
  --sample-size 20000

# Stage 3: Model Training
python analysis/gru-sae/03-training-gru-sae.py \
  --dataset bigbase \
  --version v1 \
  --model gru-sae \
  --batch-size 32 \
  --epochs 50

# Stage 4: Evaluation
python analysis/gru-sae/04-model-evaluation-gru-sae.py \
  --dataset bigbase \
  --version v1 \
  --model gru-sae

# Stage 5: Visualization
python analysis/gru-sae/05-result-vis-gru-sae.py \
  --dataset bigbase \
  --version v1 \
  --model gru-sae
```

### Unraveled Network Flow Dataset

```bash
# Stage 1: Aggregation with subdirectory
python analysis/gru-sae/01-aggregation.py \
  --model gru-sae \
  --dataset unraveled \
  --version v1 \
  --subdir network-flows \
  --schema analysis/config/schema-unraveled-gru-sae-v1.json

# Stage 2: Sequence encoding for network flows
python analysis/gru-sae/02-encoding-gru-sae.py \
  --dataset unraveled \
  --version v1 \
  --model gru-sae \
  --encoding_config analysis/config/encoding-unraveled-gru-sae-v1.json \
  --seq-len 10 \
  --sample-size 20000

# Continue with stages 3-5...
```

## Dataset-Specific Considerations

### Bigbase Dataset (Windows Event Logs)
- **Temporal structure**: Sysmon events with chronological ordering
- **APT campaigns**: 6 distinct attack campaigns (APT1-APT6)
- **Text features**: Rich command lines and file paths requiring TF-IDF
- **Sequence length**: 50-100 events per sequence for behavioral patterns (shorter than LSTM due to efficiency)
- **Data splitting**: APT-aware splitting prevents campaign leakage

### Unraveled Dataset (Network Flows)
- **Temporal structure**: Network flows with timestamps
- **Mixed features**: IP addresses, ports, packet counts, byte counts
- **Shorter sequences**: 10-20 flows per sequence for network sessions (can handle longer sequences than LSTM)
- **Data splitting**: Timeline-based splitting (70% early, 15% mid, 15% late)
- **Categorical encoding**: IP/MAC addresses as categorical features

## Troubleshooting Common Issues

### Memory Allocation Errors
- **Symptom**: "Cannot allocate XXX GiB" during sequence creation
- **Solution**: GRU-SAE typically uses less memory, but reduce sample_size if needed
- **Prevention**: Use automatic memory estimation and reduction

### GPU Configuration Issues
- **Symptom**: Multiple GPU devices detected but conflicts occur
- **Solution**: GRU-SAE restricts to GPU 0 only automatically
- **Fix**: Verify GPU restriction is working with TensorFlow GPU device logging

### Training Convergence Issues
- **Symptom**: Model not converging or unstable training
- **Solution**: GRU often converges faster; try reducing learning rate or patience
- **Optimization**: Use smart early stopping with min_improvement 0.5-1.0%

## Advanced Configuration

### Custom GRU Architecture
```bash
python analysis/gru-sae/03-training-gru-sae.py \
  --encoder-units 512 256 128 \
  --decoder-units 128 256 512 \
  --dropout-rate 0.2 \
  --learning-rate 0.0005
```

### High-Dimensional Text Processing
```json
{
  "tfidf_params": {
    "CommandLine": 1000,
    "ParentCommandLine": 500,
    "TargetObject": 800
  }
}
```

### Memory-Optimized Processing
```bash
python analysis/gru-sae/02-encoding-gru-sae.py \
  --sample-size 15000 \
  --seq-len 50 \
  --max-features 5000
```

## GRU-SAE vs LSTM-SAE Comparison

### Training Speed
- **GRU-SAE**: 25-30% faster training time
- **Memory usage**: 20-25% lower GPU memory consumption
- **Convergence**: Often reaches optimal performance in fewer epochs

### Model Performance
- **Anomaly detection**: Comparable ROC-AUC and precision-recall performance
- **Temporal patterns**: Effectively captures security event sequences
- **Robustness**: More stable training with fewer hyperparameter sensitivities

### Use Case Recommendations
- **Choose GRU-SAE when**: Fast iteration, limited GPU memory, quick prototyping
- **Choose LSTM-SAE when**: Maximum performance needed, complex long-term dependencies
- **Both suitable for**: Most cybersecurity anomaly detection tasks

This documentation provides comprehensive guidance for implementing and troubleshooting the GRU-SAE pipeline across different cybersecurity datasets while maintaining temporal integrity and computational efficiency.