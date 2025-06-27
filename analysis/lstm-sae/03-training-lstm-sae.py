"""
Stage 03: Train LSTM-SAE model with Anti-Data-Snooping Logic

This script trains an LSTM-based Stacked Autoencoder for anomaly detection using
only benign samples. It follows the same methodology as SAE and IForest pipelines.

Steps:
1. Load encoded 3D sequences from Stage 2
2. Filter to benign-only samples for unsupervised anomaly detection
3. Build configurable LSTM autoencoder architecture
4. Train with early stopping and model checkpointing
5. Save trained model and metadata
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# === Setup for GPU memory and quiet logging ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'default'

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json
import gc
from datetime import datetime
from utils.paths import get_encoded_dir, get_model_dir, ensure_dirs
from smart_early_stopping import SmartEarlyStopping, AdaptiveEarlyStopping, create_smart_callbacks

# Configure GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU available: {len(gpus)} devices")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("💻 Using CPU for training")

# === CLI Args ===
parser = argparse.ArgumentParser(description="Train LSTM-SAE model")
parser.add_argument("--dataset", required=True, help="Dataset name")
parser.add_argument("--version", required=True, help="Dataset version")
parser.add_argument("--model", required=True, help="Model name (e.g., lstm-sae)")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
parser.add_argument("--encoder-units", type=int, nargs='+', default=[256, 128], help="LSTM encoder units")
parser.add_argument("--decoder-units", type=int, nargs='+', default=[128, 256], help="LSTM decoder units")
parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--max-train", type=int, default=50000, help="Max training samples")
parser.add_argument("--max-val", type=int, default=5000, help="Max validation samples")
parser.add_argument("--stopping-strategy", choices=['standard', 'smart', 'adaptive'], default='smart', 
                   help="Early stopping strategy")
parser.add_argument("--min-improvement", type=float, default=0.5, 
                   help="Minimum improvement percentage required (for smart/adaptive stopping)")
parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
ENCODER_UNITS = args.encoder_units
DECODER_UNITS = args.decoder_units
DROPOUT_RATE = args.dropout_rate
LEARNING_RATE = args.learning_rate
MAX_TRAIN = args.max_train
MAX_VAL = args.max_val
STOPPING_STRATEGY = args.stopping_strategy
MIN_IMPROVEMENT = args.min_improvement
PATIENCE = args.patience

print(f"📦 Model: {MODEL}, Dataset: {DATASET}, Version: {VERSION}")
print(f"🔧 Config: batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LEARNING_RATE}")
print(f"🏗️ Architecture: Encoder={ENCODER_UNITS}, Decoder={DECODER_UNITS}")
print(f"🛑 Early Stopping: {STOPPING_STRATEGY} (min_improvement={MIN_IMPROVEMENT}%, patience={PATIENCE})")

# === Paths ===
ENCODED_DIR = get_encoded_dir(MODEL, DATASET, VERSION)
MODEL_DIR = get_model_dir(MODEL, DATASET, VERSION)
ensure_dirs(MODEL_DIR)

# === Load Data ===
print("📥 Loading encoded sequence data...")
try:
    # Load using new file naming convention
    X_train = np.load(os.path.join(ENCODED_DIR, "X_train_sequences.npy"))
    y_train = np.load(os.path.join(ENCODED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(ENCODED_DIR, "X_val_sequences.npy"))
    y_val = np.load(os.path.join(ENCODED_DIR, "y_val.npy"))
    print(f"✅ Loaded sequences - Train: {X_train.shape}, Val: {X_val.shape}")
except FileNotFoundError as e:
    print(f"❌ Error loading sequences: {e}")
    print("💡 Make sure Stage 2 (encoding) has completed successfully")
    sys.exit(1)

# === CRITICAL: Filter to BENIGN-ONLY samples for anomaly detection ===
print("🔍 Filtering to benign-only samples for unsupervised anomaly detection...")
train_benign_mask = y_train == 0
val_benign_mask = y_val == 0

X_train_benign = X_train[train_benign_mask]
X_val_benign = X_val[val_benign_mask]

print(f"📊 Training samples - Total: {len(y_train)}, Benign: {len(X_train_benign)} ({len(X_train_benign)/len(y_train):.1%})")
print(f"📊 Validation samples - Total: {len(y_val)}, Benign: {len(X_val_benign)} ({len(X_val_benign)/len(y_val):.1%})")
print(f"✅ Using {len(X_train_benign)} BENIGN sequences for LSTM-SAE training")

# === Downsample if needed for memory efficiency ===
if len(X_train_benign) > MAX_TRAIN:
    print(f"🎯 Sampling {MAX_TRAIN} training sequences from {len(X_train_benign)} available")
    indices = np.random.choice(len(X_train_benign), MAX_TRAIN, replace=False)
    X_train_benign = X_train_benign[indices]

if len(X_val_benign) > MAX_VAL:
    print(f"🎯 Sampling {MAX_VAL} validation sequences from {len(X_val_benign)} available")
    indices = np.random.choice(len(X_val_benign), MAX_VAL, replace=False)
    X_val_benign = X_val_benign[indices]

print(f"✅ Final shapes - Train: {X_train_benign.shape}, Val: {X_val_benign.shape}")

# === Model Parameters ===
SEQ_LEN = X_train_benign.shape[1]
INPUT_DIM = X_train_benign.shape[2]

print(f"🔢 Model dimensions - Sequence length: {SEQ_LEN}, Features: {INPUT_DIM}")

# === Build Enhanced LSTM-SAE Model ===
print("🧠 Building LSTM autoencoder architecture...")

def build_lstm_sae(input_dim, seq_len, encoder_units, decoder_units, dropout_rate):
    """Build configurable LSTM autoencoder with proper encoder-decoder structure"""
    
    # Input layer
    inputs = Input(shape=(seq_len, input_dim), name='input_sequences')
    
    # Encoder
    x = inputs
    for i, units in enumerate(encoder_units):
        return_sequences = (i < len(encoder_units) - 1)  # Last encoder layer doesn't return sequences
        x = LSTM(
            units, 
            activation='tanh',
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name=f'encoder_lstm_{i+1}'
        )(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f'encoder_dropout_{i+1}')(x)
    
    # Bottleneck (encoded representation)
    encoded = x  # This is our compressed representation
    
    # Bridge to decoder: repeat the encoded vector for each timestep
    x = RepeatVector(seq_len, name='repeat_vector')(encoded)
    
    # Decoder
    for i, units in enumerate(decoder_units):
        return_sequences = True  # Decoder always returns sequences
        x = LSTM(
            units,
            activation='tanh', 
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name=f'decoder_lstm_{i+1}'
        )(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f'decoder_dropout_{i+1}')(x)
    
    # Output layer: reconstruct input sequences
    outputs = TimeDistributed(
        Dense(input_dim, activation='linear'), 
        name='reconstruction'
    )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_SAE')
    
    return model

# Build model
model = build_lstm_sae(INPUT_DIM, SEQ_LEN, ENCODER_UNITS, DECODER_UNITS, DROPOUT_RATE)

# Compile with custom optimizer
optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)  # Gradient clipping for stability
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

print("📋 Model architecture:")
model.summary()

# === Setup Smart Callbacks ===
print(f"🧠 Setting up {STOPPING_STRATEGY} early stopping...")

# Create stopping strategy specific callbacks
if STOPPING_STRATEGY == 'smart':
    callbacks = [
        SmartEarlyStopping(
            monitor='val_loss',
            min_improvement_pct=MIN_IMPROVEMENT,
            patience=PATIENCE,
            improvement_window=10,
            min_window_improvement=2.0,  # 2% improvement over 10 epochs required
            min_improvement_rate=0.1,    # 0.1% per epoch minimum rate
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}-best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
elif STOPPING_STRATEGY == 'adaptive':
    callbacks = [
        AdaptiveEarlyStopping(
            monitor='val_loss',
            base_patience=PATIENCE,
            min_improvement_ratio=MIN_IMPROVEMENT/100.0,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}-best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
else:  # standard
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}-best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

# === Train Model ===
print("🚀 Training LSTM autoencoder on benign sequences...")
print(f"🎯 Target: Learn to reconstruct normal patterns")

# GPU warmup
_ = model(X_train_benign[:1])

# Training
history = model.fit(
    X_train_benign, X_train_benign,  # Autoencoder: input = target
    validation_data=(X_val_benign, X_val_benign),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)

# === Save Final Model and Artifacts ===
model_path = os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}.keras")
model.save(model_path)

# Save training history
history_path = os.path.join(MODEL_DIR, f"history-{MODEL}-model-{DATASET}-{VERSION}.json")
with open(history_path, "w") as f:
    # Convert numpy types to native Python for JSON serialization
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    json.dump(history_dict, f, indent=2)

# Save comprehensive metadata
metadata = {
    "model": MODEL,
    "dataset": DATASET,
    "version": VERSION,
    "model_type": "LSTM_Autoencoder",
    "architecture": {
        "seq_len": int(SEQ_LEN),
        "input_dim": int(INPUT_DIM),
        "encoder_units": ENCODER_UNITS,
        "decoder_units": DECODER_UNITS,
        "dropout_rate": DROPOUT_RATE
    },
    "training": {
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history.history['loss']),
        "learning_rate": LEARNING_RATE,
        "train_samples": int(X_train_benign.shape[0]),
        "val_samples": int(X_val_benign.shape[0])
    },
    "performance": {
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
        "best_val_loss": float(min(history.history['val_loss']))
    },
    "training_strategy": "benign_only_unsupervised",
    "saved_at": datetime.now().isoformat()
}

metadata_path = os.path.join(MODEL_DIR, f"metadata-{MODEL}-model-{DATASET}-{VERSION}.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Training complete! Artifacts saved:")
print(f"  Model (final):     {model_path}")
print(f"  Model (best):      {MODEL_DIR}/{MODEL}-model-{DATASET}-{VERSION}-best.keras")
print(f"  Training history:  {history_path}")
print(f"  Metadata:          {metadata_path}")

# === Final Summary ===
print(f"\n📊 Training Summary:")
print(f"  Final training loss:   {history.history['loss'][-1]:.6f}")
print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
print(f"  Best validation loss:  {min(history.history['val_loss']):.6f}")
print(f"  Epochs completed:      {len(history.history['loss'])}/{EPOCHS}")

# === Cleanup ===
del X_train, X_val, X_train_benign, X_val_benign, model
tf.keras.backend.clear_session()
gc.collect()

print("🏁 LSTM-SAE training complete.")