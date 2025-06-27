# LSTM-SAE Pipeline Documentation

## Overview

The LSTM-SAE (Long Short-Term Memory Stacked Autoencoder) pipeline implements temporal anomaly detection for cybersecurity datasets. Unlike traditional autoencoders that work on static feature vectors, LSTM-SAE processes sequential data to capture temporal patterns in security events.

### Key Features
- **Temporal modeling**: Processes sequences of events to capture behavioral patterns
- **Smart early stopping**: Intelligent training termination with improvement rate analysis
- **Automated pipeline**: Complete 5-stage automation with error handling and flow control
- **Anti-data-snooping**: Early 3-way data splitting (train/val/test) prevents information leakage
- **Benign-only training**: Unsupervised anomaly detection trained exclusively on normal samples
- **Memory-efficient**: Sparse matrix handling and batch processing for large datasets
- **APT-aware splitting**: Campaign-based data splitting for bigbase dataset
- **Configurable architecture**: User-defined LSTM encoder/decoder structures
- **Production-ready**: Robust error handling, memory management, and scalable processing

## Pipeline Architecture

The LSTM-SAE pipeline follows a 5-stage sequential architecture:

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

### Stage 2: Sequence Encoding (`02-encoding-lstm-sae.py`)

**Purpose**: Transforms tabular data into 3D sequences suitable for LSTM processing with anti-data-snooping methodology.

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

### Stage 3: Model Training (`03-training-lstm-sae.py`)

**Purpose**: Trains LSTM autoencoder on benign sequences to learn normal behavioral patterns with intelligent early stopping.

**Enhanced Command Line Arguments**:
```bash
--stopping-strategy {standard,smart,adaptive}  # Early stopping strategy (default: smart)
--min-improvement FLOAT                        # Minimum improvement % for smart stopping (default: 0.5)
--patience INT                                 # Epochs to wait for improvement (default: 5)
--encoder-units INT [INT ...]                  # LSTM encoder units (default: [256, 128])
--decoder-units INT [INT ...]                  # LSTM decoder units (default: [128, 256])
--max-train INT                                # Max training samples for memory management
--max-val INT                                  # Max validation samples for memory management
```

**Architecture Functions**:

#### `build_lstm_sae(input_dim, seq_len, encoder_units, decoder_units, dropout_rate)`
Creates configurable LSTM autoencoder with user-defined architecture:
```python
# Encoder: Compresses sequences to latent representation
for i, units in enumerate(encoder_units):
    return_sequences = (i < len(encoder_units) - 1)
    x = LSTM(units, return_sequences=return_sequences, dropout=dropout_rate)(x)

# Bridge: Repeats encoded vector for decoder input
x = RepeatVector(seq_len)(encoded)

# Decoder: Reconstructs original sequences
for i, units in enumerate(decoder_units):
    x = LSTM(units, return_sequences=True, dropout=dropout_rate)(x)

# Output: Sequence reconstruction
outputs = TimeDistributed(Dense(input_dim, activation='linear'))(x)
```

**Training Strategy**:
- **Benign-only**: Filters training data to normal samples only
- **Reconstruction loss**: MSE between input and reconstructed sequences
- **Smart early stopping**: Intelligent termination based on improvement analysis
- **Gradient clipping**: Stabilizes LSTM training (`clipnorm=1.0`)
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

**Enhanced Callbacks**:
```python
# Smart strategy example
callbacks = [
    SmartEarlyStopping(monitor='val_loss', min_improvement_pct=0.5, patience=5),
    ModelCheckpoint(save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-7)
]
```

**Configurable Architecture Examples**:
```bash
# Default architecture
--encoder-units 256 128 --decoder-units 128 256

# Deep architecture  
--encoder-units 512 256 128 64 --decoder-units 64 128 256 512

# Wide architecture
--encoder-units 512 512 --decoder-units 512 512

# Shallow architecture
--encoder-units 128 --decoder-units 128
```

**Outputs**:
- `{model}-model-{dataset}-{version}.keras`: Final trained model
- `{model}-model-{dataset}-{version}-best.keras`: Best validation model
- `history-{model}-model-{dataset}-{version}.json`: Training metrics
- `metadata-{model}-model-{dataset}-{version}.json`: Enhanced architecture details

### Stage 4: Model Evaluation (`04-model-evaluation-lstm-sae.py`)

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

### Stage 5: Result Visualization (`05-result-vis-lstm-sae.py`)

**Purpose**: Creates comprehensive visualizations for model analysis and result interpretation.

**Key Visualization Functions**:

#### `extract_lstm_encoder(model)`
Extracts encoder portion for latent space analysis:
```python
def extract_lstm_encoder(model):
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

## Automated Pipeline Runner (`run_lstm_sae_pipeline.py`)

The automated pipeline runner provides complete 5-stage automation with intelligent flow control, error handling, and enhanced user experience.

### Key Features

#### **Configuration Presets**
Automated configuration file selection based on dataset and experiment type:
```bash
--config-preset basic     # Uses schema-bigbase-lstm-sae-v1.json + encoding-bigbase-lstm-sae-v1.json
--config-preset extended  # Uses schema-bigbase-lstm-sae-v2.json + encoding-bigbase-lstm-sae-v2.json  
--config-preset network   # Uses schema-unraveled-v1.json + encoding-unraveled-lstm-sae-v1.json
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
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase \
  --version v3 \
  --config-preset basic \
  --seq-len 50 \
  --stopping-strategy smart \
  --min-improvement 0.5

# Extended bigbase with advanced features
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
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
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
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
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v5 --config-preset basic \
  --start-from 3

# Training and evaluation only (skip visualization)
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v6 --config-preset extended \
  --start-from 3 --stop-at 4

# Skip failed stages and continue
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v7 --config-preset basic \
  --skip-stages 1 2  # Use pre-processed data
```

### Output Management

#### **Smart Filtering System**
The pipeline runner implements intelligent output filtering to reduce terminal spam:

**Training Stage Filtering**:
- ✅ Shows: Epochs 1, 10, 20, 30... (every 10th epoch)
- ✅ Shows: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau messages  
- ✅ Shows: Training start/completion, architecture summary
- ❌ Hides: Individual batch progress bars (`1562/1562 [====] - ETA: 2s`)

**Encoding Stage Filtering**:
- ✅ Shows: Progress at 25%, 50%, 75%, 100%
- ❌ Hides: Intermediate tqdm progress updates

**Always Preserved**:
- ✅ Error messages, warnings, completion status
- ✅ GPU information, model summaries, file operations
- ✅ Lines with status emojis (📋, ✅, ❌, 🚀, etc.)

#### **Color-Coded Progress**
```
🚀 STAGE 1: DATA AGGREGATION
📋 Command: python analysis/lstm-sae/01-aggregation.py --model lstm-sae...
✅ Stage 1: Data Aggregation completed in 45.2s

🚀 STAGE 2: SEQUENCE ENCODING  
📋 Command: python analysis/lstm-sae/02-encoding-lstm-sae.py --dataset...
🧠 Smart Early Stopping Configuration:
   Minimum improvement: 0.5% or 0.000500
✅ Stage 2: Sequence Encoding completed in 156.8s
```

### Error Handling and Recovery

#### **Stage Failure Management**
```bash
❌ Stage 2: Sequence Encoding failed after 45.2s with return code 1
Continue with next stage? (y/n): y

🚀 STAGE 3: MODEL TRAINING
# ... continues with remaining stages
```

#### **Pipeline Summary**
```
🏁 LSTM-SAE PIPELINE COMPLETE
Total execution time: 47.3 minutes
⚠️ Failed stages: [2]
Please check the error messages above and rerun failed stages manually.

📁 Pipeline Artifacts:
  Processed Data: analysis/experiments/processed/lstm-sae-bigbase-v3.parquet
  Encoded Data:   analysis/experiments/encoded/lstm-sae-bigbase-v3/
  Trained Model:  analysis/experiments/models/lstm-sae-bigbase-v3/
  Evaluation:     analysis/experiments/eval/lstm-sae-bigbase-v3/
  Visualizations: analysis/experiments/vis/lstm-sae-bigbase-v3/
```

## Smart Early Stopping System (`smart_early_stopping.py`)

### Overview
The smart early stopping system addresses the critical problem of endless training with millicimal improvements (e.g., `0.0201 → 0.0200`), which can lead to 10+ hour training sessions with minimal benefit.

### Problem Statement
**Before**: Traditional `EarlyStopping(patience=5)` continues training for tiny improvements:
```
Epoch 45: val_loss = 0.0201  → wait = 0 ✅ (0.05% improvement = "improvement")
Epoch 46: val_loss = 0.0200  → wait = 0 ✅ (0.05% improvement = "improvement")  
... continues for all 150 epochs with 0.05% improvements
Result: 8+ hours of training time for negligible model improvement
```

**After**: Smart early stopping requires significant improvements:
```
Epoch 45: val_loss = 0.0201  → wait = 0 ✅ (new best)
Epoch 46: val_loss = 0.0200  → wait = 1 ⚠️ (0.05% < 0.5% required)
Epoch 47: val_loss = 0.0199  → wait = 2 ⚠️ (0.05% < 0.5% required)
Epoch 50: val_loss = 0.0196  → wait = 5 ❌ STOP! (patience reached)

🛑 Training stopped: No significant improvement for 5 epochs
⏱️ Saved 6+ hours of training time!
```

### Smart Early Stopping Algorithms

#### **1. SmartEarlyStopping** (Recommended)
Multi-criteria analysis with window-based improvement tracking:

```python
SmartEarlyStopping(
    monitor='val_loss',
    min_improvement_pct=0.5,        # Require ≥0.5% relative improvement
    min_improvement_abs=0.0005,     # Require ≥0.0005 absolute improvement
    patience=5,                     # Wait 5 epochs for significant improvement
    improvement_window=10,          # Analyze 10-epoch improvement windows
    min_window_improvement=2.0,     # Require 2% improvement over window
    min_improvement_rate=0.1,       # Require 0.1% improvement per epoch
    restore_best_weights=True
)
```

**Stopping Criteria**:
1. **Individual epoch improvement**: Must meet both relative (0.5%) and absolute (0.0005) thresholds
2. **Patience exhaustion**: No significant improvement for 5 consecutive epochs
3. **Window analysis**: Total improvement over 10 epochs must exceed 2%
4. **Improvement rate**: Average improvement per epoch must exceed 0.1%

#### **2. AdaptiveEarlyStopping** (Simpler Alternative)
Dynamic threshold adjustment based on loss magnitude:

```python
AdaptiveEarlyStopping(
    monitor='val_loss',
    base_patience=5,
    min_improvement_ratio=0.001,    # 0.1% of current loss value
    adaptive_threshold=True         # Adjust threshold based on loss scale
)
```

**Adaptive Thresholding**:
- For `val_loss = 0.01`: requires `0.0001` improvement (1%)
- For `val_loss = 0.001`: requires `0.00001` improvement (1%)
- Automatically scales with loss magnitude

#### **3. Factory Function**
Simplified callback creation:

```python
from smart_early_stopping import create_smart_callbacks

# Create complete callback set
callbacks = create_smart_callbacks(
    strategy='smart',
    min_improvement_pct=1.0,
    patience=3
)

# Returns: [SmartEarlyStopping, ReduceLROnPlateau]
```

### Integration Examples

#### **Training Script Integration**
```bash
# Aggressive stopping (fast iteration)
python analysis/lstm-sae/03-training-lstm-sae.py \
  --dataset bigbase --version v3 --model lstm-sae \
  --stopping-strategy smart \
  --min-improvement 1.0 \
  --patience 3

# Balanced stopping (recommended)
python analysis/lstm-sae/03-training-lstm-sae.py \
  --dataset bigbase --version v4 --model lstm-sae \
  --stopping-strategy smart \
  --min-improvement 0.5 \
  --patience 5

# Conservative stopping (thorough training)
python analysis/lstm-sae/03-training-lstm-sae.py \
  --dataset bigbase --version v5 --model lstm-sae \
  --stopping-strategy smart \
  --min-improvement 0.2 \
  --patience 8
```

#### **Expected Time Savings**
- **Before**: 150 epochs × 2-3 minutes = **5-8 hours**
- **After**: 40-60 epochs × 2-3 minutes = **1.5-3 hours**  
- **Savings**: **60-70% reduction** in training time

### Verbose Training Output
The smart early stopping system provides detailed feedback:

```
🧠 Smart Early Stopping Configuration:
   Minimum improvement: 0.5% or 0.000500
   Patience: 5 epochs
   Window analysis: 10 epochs
   Required window improvement: 2.0%
   Required improvement rate: 0.1%/epoch

Epoch 1/150 - loss: 0.0850 - val_loss: 0.0720
✅ Significant improvement: 15.28% (epoch 1)

Epoch 34/150 - loss: 0.0089 - val_loss: 0.0067
✅ Significant improvement: 1.2% (epoch 34)

Epoch 35/150 - loss: 0.0087 - val_loss: 0.0066
⚠️ Minor improvement: 0.149% - patience 1/5

Epoch 39/150 - loss: 0.0081 - val_loss: 0.0064
⚠️ Minor improvement: 0.153% - patience 5/5

🛑 Smart Early Stopping triggered at epoch 39
   Reason: No significant improvement for 5 epochs
   Best loss: 0.006700 (epoch 34)
   Restored weights from epoch 34
```

## Configuration System

### Enhanced Configuration Structure

The pipeline now supports an extended configuration system with dataset-specific optimizations:

#### **Schema Configurations (`analysis/config/schema-*.json`)**
Define dataset structure and core features:

#### `schema-bigbase-v1.json`
```json
{
  "core_features": ["Computer", "User", "EventID", "CommandLine", ...],
  "label_column": "Label",
  "time_column": "UtcTime",
  "benign_label": "normal"
}
```

### Encoding Configurations (`analysis/config/encoding-*-lstm-sae-*.json`)
Specify feature transformations for LSTM processing:

#### `encoding-bigbase-lstm-sae-v1.json`
```json
{
  "tfidf_columns": ["CommandLine", "ParentCommandLine", "Image", ...],
  "tfidf_params": {"CommandLine": 500, "Image": 300, ...},
  "categorical_columns": ["Computer", "User", "Protocol", ...],
  "numeric_columns": ["EventID", "ProcessId", "UtcTime"],
  "time_column": "UtcTime"
}
```

#### `encoding-bigbase-lstm-sae-v2.json`
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
    Benign-only LSTM-SAE Training
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

## Usage Examples

### Complete LSTM-SAE Pipeline for Bigbase Dataset

```bash
# Stage 1: Data Aggregation
python analysis/lstm-sae/01-aggregation.py \
  --model lstm-sae \
  --dataset bigbase \
  --version v1 \
  --schema analysis/config/schema-bigbase-v1.json

# Stage 2: Sequence Encoding  
python analysis/lstm-sae/02-encoding-lstm-sae.py \
  --dataset bigbase \
  --version v1 \
  --model lstm-sae \
  --encoding_config analysis/config/encoding-bigbase-lstm-sae-v1.json \
  --seq-len 100 \
  --sample-size 20000

# Stage 3: Model Training
python analysis/lstm-sae/03-training-lstm-sae.py \
  --dataset bigbase \
  --version v1 \
  --model lstm-sae \
  --batch-size 16 \
  --epochs 50

# Stage 4: Evaluation
python analysis/lstm-sae/04-model-evaluation-lstm-sae.py \
  --dataset bigbase \
  --version v1 \
  --model lstm-sae

# Stage 5: Visualization
python analysis/lstm-sae/05-result-vis-lstm-sae.py \
  --dataset bigbase \
  --version v1 \
  --model lstm-sae
```

### Unraveled Network Flow Dataset

```bash
# Stage 1: Aggregation with subdirectory
python analysis/lstm-sae/01-aggregation.py \
  --model lstm-sae \
  --dataset unraveled \
  --version v1 \
  --subdir network-flows \
  --schema analysis/config/schema-unraveled-v1.json

# Stage 2: Sequence encoding for network flows
python analysis/lstm-sae/02-encoding-lstm-sae.py \
  --dataset unraveled \
  --version v1 \
  --model lstm-sae \
  --encoding_config analysis/config/encoding-unraveled-lstm-sae-v1.json \
  --seq-len 10 \
  --sample-size 20000

# Continue with stages 3-5...
```

## Dataset-Specific Considerations

### Bigbase Dataset (Windows Event Logs)
- **Temporal structure**: Sysmon events with chronological ordering
- **APT campaigns**: 6 distinct attack campaigns (APT1-APT6)
- **Text features**: Rich command lines and file paths requiring TF-IDF
- **Sequence length**: 100-1000 events per sequence for behavioral patterns
- **Data splitting**: APT-aware splitting prevents campaign leakage

### Unraveled Dataset (Network Flows)
- **Temporal structure**: Network flows with timestamps
- **Mixed features**: IP addresses, ports, packet counts, byte counts
- **Shorter sequences**: 10-50 flows per sequence for network sessions
- **Data splitting**: Timeline-based splitting (70% early, 15% mid, 15% late)
- **Categorical encoding**: IP/MAC addresses as categorical features

## Performance Considerations

### Model Complexity
- **Input dimensionality**: Determined by encoding configuration
- **Sequence length**: Balances temporal context vs. computational cost
- **Architecture depth**: Default [256,128] encoder provides good compression
- **Memory usage**: 3D sequences require significant GPU memory

### Training Efficiency
- **Batch size**: Smaller batches (16-32) for memory-constrained environments
- **Early stopping**: Prevents overfitting and reduces training time
- **Learning rate scheduling**: Adaptive reduction improves convergence

### Evaluation Scalability
- **Batch processing**: Handles large test sets without memory issues
- **Sparse operations**: Maintains efficiency with high-dimensional features
- **Progressive visualization**: Generates plots incrementally for large datasets

## Troubleshooting Common Issues

### Memory Allocation Errors
- **Symptom**: "Cannot allocate XXX GiB" during sequence creation
- **Solution**: Reduce sample_size or increase MAX_FEATURES limit
- **Prevention**: Use automatic memory estimation and reduction

### Missing Time Column Errors
- **Symptom**: "UtcTime_numeric column not found"
- **Solution**: Ensure Stage 1 includes time_column in schema
- **Fix**: Use original "UtcTime" column with in-place conversion

### Sparse Matrix Memory Explosion
- **Symptom**: Memory usage jumps to 700+ GiB during feature selection
- **Solution**: Use TruncatedSVD directly on sparse matrices
- **Avoid**: Never call .toarray() on large sparse matrices

### File Naming Inconsistencies
- **Symptom**: Script expects different filename than exists
- **Solution**: Use consistent naming convention across all stages
- **Standard**: `02-encoding-lstm-sae.py` for LSTM-SAE encoding

## Advanced Configuration

### Custom LSTM Architecture
```bash
python analysis/lstm-sae/03-training-lstm-sae.py \
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
python analysis/lstm-sae/02-encoding-lstm-sae.py \
  --sample-size 10000 \
  --seq-len 50 \
  --max-features 5000
```

This documentation provides comprehensive guidance for implementing and troubleshooting the LSTM-SAE pipeline across different cybersecurity datasets while maintaining temporal integrity and memory efficiency.