"""
Stage 03: Train SAE model on encoded dataset (BENIGN-ONLY Training)

- Loads pre-split and pre-encoded training/validation datasets
- Filters training data to BENIGN SAMPLES ONLY (y_train == 0) for anomaly detection
- Builds and trains a Stacked Autoencoder (SAE) model on normal samples only
- Uses a TensorFlow data generator for efficient batch processing
- Stores trained model, training history, and metadata to disk
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# === Setup for GPU memory and quiet logging ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = info, 2 = warning, 3 = error
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# ===
import argparse
import json
import numpy as np
from datetime import datetime
from scipy import sparse
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import gc

from utils.paths import (
    get_encoded_dir,
    get_model_dir,
    ensure_dirs
)

# === CLI Args ===
parser = argparse.ArgumentParser(description="Train SAE model on benign data")
parser.add_argument("--dataset", required=True, help="Dataset name")
parser.add_argument("--version", required=True, help="Dataset version")
parser.add_argument("--model", required=True, help="Model (sae, lstm-sae, etc.)")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model
print(f"📦 Dataset: {DATASET}, Version: {VERSION}, Model: {MODEL}")

# === Paths ===
ENCODED_DIR = get_encoded_dir(MODEL, DATASET, VERSION)
MODEL_DIR = get_model_dir(MODEL, DATASET, VERSION)
os.makedirs(MODEL_DIR, exist_ok=True)

X_train_path = os.path.join(ENCODED_DIR, "X_train_encoded.npz")
y_train_path = os.path.join(ENCODED_DIR, "y_train.npy")
X_val_path = os.path.join(ENCODED_DIR, "X_val_encoded.npz")
y_val_path = os.path.join(ENCODED_DIR, "y_val.npy")

# === Load Data ===
print("📥 Loading encoded training and validation data...")
X_train_full = sparse.load_npz(X_train_path)
y_train_full = np.load(y_train_path)
X_val = sparse.load_npz(X_val_path)
y_val = np.load(y_val_path)

print(f"✅ Full X_train: {X_train_full.shape}, X_val: {X_val.shape}")

# === CRITICAL: Filter to BENIGN-ONLY samples for anomaly detection ===
print("🔍 Filtering training data to BENIGN samples only...")
benign_mask = y_train_full == 0
X_train_benign = X_train_full[benign_mask]
y_train_benign = y_train_full[benign_mask]

print(f"📊 Training data filtering:")
print(f"  Original training samples: {X_train_full.shape[0]}")
print(f"  Benign samples: {X_train_benign.shape[0]} ({np.sum(benign_mask)/len(y_train_full):.1%})")
print(f"  Attack samples (filtered out): {np.sum(~benign_mask)} ({np.sum(~benign_mask)/len(y_train_full):.1%})")
print(f"✅ Training on {X_train_benign.shape[0]} BENIGN samples only")

# === Validation data analysis ===
val_benign = np.sum(y_val == 0)
val_attack = np.sum(y_val == 1)
print(f"📊 Validation data composition:")
print(f"  Benign: {val_benign} ({val_benign/len(y_val):.1%})")
print(f"  Attack: {val_attack} ({val_attack/len(y_val):.1%})")

# === GPU config ===
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# === Model Builder ===
def build_sae(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === TF Dataset Generator ===
def create_tf_dataset(X_csr, batch_size):
    def gen():
        for i in range(0, X_csr.shape[0], batch_size):
            X_batch = X_csr[i:i+batch_size].toarray().astype(np.float32)
            yield X_batch, X_batch

    output_signature = (
        tf.TensorSpec(shape=(None, X_csr.shape[1]), dtype=tf.float32),
        tf.TensorSpec(shape=(None, X_csr.shape[1]), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(gen, output_signature=output_signature)

# === Training Setup ===
BATCH_SIZE = 4096
EPOCHS = 10
STEPS_PER_EPOCH = X_train_benign.shape[0] // BATCH_SIZE
VALIDATION_STEPS = X_val.shape[0] // BATCH_SIZE
INPUT_DIM = X_train_benign.shape[1]

print(f"🚀 Training configuration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Steps per epoch: {STEPS_PER_EPOCH}")
print(f"  Validation steps: {VALIDATION_STEPS}")
print(f"  Input dimension: {INPUT_DIM}")

# NOTE: Training dataset uses ONLY benign samples, validation uses mixed for proper evaluation
train_dataset = create_tf_dataset(X_train_benign, BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
val_dataset = create_tf_dataset(X_val, BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

# === Build + Train ===
print("🚀 Starting model training...")
model = build_sae(INPUT_DIM)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=1
)

# === Save Artifacts ===
print("💾 Saving model and training metadata...")
model_path = os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}.keras")
model.save(model_path)

with open(os.path.join(MODEL_DIR, f"history-{MODEL}-model-{DATASET}-{VERSION}.json"), "w") as f:
    json.dump(history.history, f)

with open(os.path.join(MODEL_DIR, f"metadata-{MODEL}-model-{DATASET}-{VERSION}.json"), "w") as f:
    json.dump({
        "model": MODEL,
        "dataset": DATASET,
        "version": VERSION,
        "input_dim": INPUT_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "validation_steps": VALIDATION_STEPS,
        "total_train_samples": X_train_full.shape[0],
        "benign_train_samples": X_train_benign.shape[0],
        "benign_ratio": float(X_train_benign.shape[0] / X_train_full.shape[0]),
        "val_samples": X_val.shape[0],
        "val_benign": int(val_benign),
        "val_attack": int(val_attack),
        "training_strategy": "benign_only_anomaly_detection",
        "architecture": "stacked_autoencoder_1024_512_128_512_1024",
        "saved_at": datetime.now().isoformat()
    }, f, indent=2)

print("✅ Artifacts saved:")
print(f"  Model: {model_path}")
print(f"  History: {MODEL_DIR}/history-{MODEL}-model-{DATASET}-{VERSION}.json")
print(f"  Metadata: {MODEL_DIR}/metadata-{MODEL}-model-{DATASET}-{VERSION}.json") 


print("🏁 Training complete.")
print("📋 Summary:")
print(f"  ✅ Trained SAE on {X_train_benign.shape[0]} benign samples")
print(f"  ✅ Validated on {X_val.shape[0]} mixed samples ({val_benign} benign, {val_attack} attack)")
print(f"  ✅ Model ready for anomaly detection evaluation")

# === Cleanup ===
del model
del X_train_full
del X_train_benign
del X_val
del y_train_full
del y_train_benign
del y_val
K.clear_session()
gc.collect()