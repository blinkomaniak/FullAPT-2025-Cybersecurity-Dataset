"""
Stage 04: Evaluate LSTM-SAE model with Anti-Data-Snooping Logic

This script evaluates the trained LSTM autoencoder on the test set (never seen during training)
using reconstruction error as the anomaly score for sequence-based anomaly detection.

Steps:
1. Load test sequences and trained LSTM-SAE model
2. Compute reconstruction errors for all test sequences
3. Evaluate anomaly detection performance using comprehensive metrics
4. Save evaluation results and visualizations
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# === Setup for GPU memory and quiet logging ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'default'

import argparse
import numpy as np
import json
import time
import gc
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    roc_curve, precision_recall_curve
)

from utils.paths import get_encoded_dir, get_model_dir, get_eval_dir

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# === CLI Args ===
parser = argparse.ArgumentParser(description="Evaluate LSTM-SAE model")
parser.add_argument("--dataset", required=True, help="Dataset name")
parser.add_argument("--version", required=True, help="Dataset version")
parser.add_argument("--model", required=True, help="Model name (e.g., lstm-sae)")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model
BATCH_SIZE = args.batch_size

print(f"📦 Model: {MODEL}, Dataset: {DATASET}, Version: {VERSION}")

# === Paths ===
ENCODED_DIR = get_encoded_dir(MODEL, DATASET, VERSION)
MODEL_DIR = get_model_dir(MODEL, DATASET, VERSION)
EVAL_DIR = get_eval_dir(MODEL, DATASET, VERSION)
os.makedirs(EVAL_DIR, exist_ok=True)

# Updated paths for new file naming convention
X_test_path = os.path.join(ENCODED_DIR, "X_test_sequences.npy")
y_test_path = os.path.join(ENCODED_DIR, "y_test.npy")
model_path = os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}.keras")
metadata_path = os.path.join(MODEL_DIR, f"metadata-{MODEL}-model-{DATASET}-{VERSION}.json")

# === Load Data and Model ===
print("📥 Loading test sequences and trained model...")
try:
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    print(f"✅ Loaded test data: {X_test.shape}, labels: {y_test.shape}")
except FileNotFoundError as e:
    print(f"❌ Error loading test data: {e}")
    print("💡 Make sure Stage 2 (encoding) has completed successfully")
    sys.exit(1)

try:
    model = load_model(model_path, compile=False)
    print(f"✅ Loaded model from: {model_path}")
except FileNotFoundError:
    print(f"❌ Model not found: {model_path}")
    print("💡 Make sure Stage 3 (training) has completed successfully")
    sys.exit(1)

# Load metadata for context
try:
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"📋 Model architecture: {metadata.get('architecture', 'N/A')}")
except FileNotFoundError:
    print("⚠️ Model metadata not found, proceeding without it")
    metadata = {}

print(f"📊 Test data: {X_test.shape[0]} sequences, {X_test.shape[1]} timesteps, {X_test.shape[2]} features")

# === Compute Reconstruction Errors ===
print("🔍 Computing reconstruction errors for test sequences...")

def compute_reconstruction_errors_batch(model, X, batch_size=32):
    """Compute reconstruction errors for 3D sequence data in batches"""
    n_samples = X.shape[0]
    reconstruction_errors = []
    
    # Process in batches to avoid memory issues
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, n_samples, batch_size), total=n_batches, desc="Computing reconstruction errors"):
        batch_end = min(i + batch_size, n_samples)
        X_batch = X[i:batch_end]
        
        # Get reconstructions
        X_pred = model.predict(X_batch, verbose=0)
        
        # Compute reconstruction error for each sequence
        # Mean squared error across all timesteps and features
        batch_errors = np.mean(np.square(X_batch - X_pred), axis=(1, 2))
        reconstruction_errors.extend(batch_errors)
        
        # Cleanup to save memory
        del X_batch, X_pred
        gc.collect()
    
    return np.array(reconstruction_errors)

start_time = time.time()
reconstruction_errors = compute_reconstruction_errors_batch(model, X_test, BATCH_SIZE)
print(f"✅ Reconstruction errors computed in {time.time() - start_time:.2f}s")

print(f"📊 Reconstruction error statistics:")
print(f"   Mean: {np.mean(reconstruction_errors):.6f}")
print(f"   Std:  {np.std(reconstruction_errors):.6f}")
print(f"   Min:  {np.min(reconstruction_errors):.6f}")
print(f"   Max:  {np.max(reconstruction_errors):.6f}")

# === Save Raw Scores ===
np.savez_compressed(
    os.path.join(EVAL_DIR, "reconstruction_errors.npz"),
    errors=reconstruction_errors,
    y_labels=y_test
)

# === Compute Evaluation Metrics ===
print("📊 Computing evaluation metrics...")

# For anomaly detection: higher reconstruction error = more anomalous
anomaly_scores = reconstruction_errors
roc_auc = roc_auc_score(y_test, anomaly_scores)
pr_auc = average_precision_score(y_test, anomaly_scores)

# Threshold optimization
print("🔎 Optimizing detection threshold...")
thresholds = np.linspace(
    np.percentile(anomaly_scores, 5),
    np.percentile(anomaly_scores, 95),
    100
)

precisions, recalls, f1_scores = [], [], []

for thresh in tqdm(thresholds, desc="Threshold scan"):
    preds = (anomaly_scores > thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, preds, average='binary', zero_division=0
    )
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f1)

best_idx = int(np.argmax(f1_scores))
best_thresh = thresholds[best_idx]

# === Save Comprehensive Metrics ===
metrics = {
    "model": MODEL,
    "dataset": DATASET,
    "version": VERSION,
    "model_type": "LSTM_Autoencoder",
    "evaluation_data": "test_set",
    "performance": {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1_score": float(f1_scores[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "best_threshold": float(best_thresh)
    },
    "data_stats": {
        "test_samples": int(len(y_test)),
        "test_benign": int(np.sum(y_test == 0)),
        "test_attack": int(np.sum(y_test == 1)),
        "test_attack_ratio": float(np.mean(y_test))
    },
    "reconstruction_error_stats": {
        "mean": float(np.mean(reconstruction_errors)),
        "std": float(np.std(reconstruction_errors)),
        "min": float(np.min(reconstruction_errors)),
        "max": float(np.max(reconstruction_errors)),
        "median": float(np.median(reconstruction_errors))
    },
    "architecture": metadata.get("architecture", {}),
    "evaluated_at": datetime.now().isoformat()
}

with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# === Save ROC/PR Curve Data ===
fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, anomaly_scores)

np.savez_compressed(
    os.path.join(EVAL_DIR, "roc_pr_data.npz"),
    fpr=fpr, tpr=tpr,
    precision_curve=prec_curve, recall_curve=rec_curve,
    roc_auc=roc_auc, pr_auc=pr_auc
)

# === Generate Visualizations ===
print("📈 Generating evaluation plots...")

# Plot 1: Threshold vs Metrics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(thresholds, f1_scores, label="F1-score", linewidth=2)
plt.plot(thresholds, precisions, '--', label="Precision", alpha=0.7)
plt.plot(thresholds, recalls, '--', label="Recall", alpha=0.7)
plt.axvline(best_thresh, linestyle=':', color='red', label=f'Best threshold ({best_thresh:.4f})')
plt.xlabel("Reconstruction Error Threshold")
plt.ylabel("Score")
plt.title("Threshold vs Performance Metrics")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: ROC and PR Curves
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f"ROC (AUC: {roc_auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "evaluation_metrics.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(rec_curve, prec_curve, label=f"PR (AUC: {pr_auc:.4f})", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(EVAL_DIR, "precision_recall_curve.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Reconstruction Error Distribution
plt.figure(figsize=(10, 6))
benign_errors = reconstruction_errors[y_test == 0]
attack_errors = reconstruction_errors[y_test == 1]

plt.hist(benign_errors, bins=50, alpha=0.7, label=f'Benign (n={len(benign_errors)})', density=True)
plt.hist(attack_errors, bins=50, alpha=0.7, label=f'Attack (n={len(attack_errors)})', density=True)
plt.axvline(best_thresh, linestyle='--', color='red', label=f'Threshold ({best_thresh:.4f})')
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(EVAL_DIR, "reconstruction_error_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

print("✅ Evaluation complete! Results saved:")
print(f"📊 Performance Summary:")
print(f"   ROC-AUC: {roc_auc:.4f}")
print(f"   PR-AUC:  {pr_auc:.4f}")
print(f"   F1:      {f1_scores[best_idx]:.4f}")
print(f"   Precision: {precisions[best_idx]:.4f}")
print(f"   Recall:    {recalls[best_idx]:.4f}")

print(f"\n📁 Artifacts saved to: {EVAL_DIR}/")
print(" ├── metrics.json")
print(" ├── reconstruction_errors.npz")
print(" ├── roc_pr_data.npz")
print(" ├── evaluation_metrics.png")
print(" ├── precision_recall_curve.png")
print(" └── reconstruction_error_distribution.png")

# === Cleanup ===
del model, X_test, reconstruction_errors
tf.keras.backend.clear_session()
gc.collect()

print("🏁 LSTM-SAE evaluation complete.")