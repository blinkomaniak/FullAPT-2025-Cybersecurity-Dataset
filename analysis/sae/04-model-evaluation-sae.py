"""
Stage 04: Evaluation of SAE model on test split
- Computes reconstruction error on `X_test` (proper holdout evaluation)
- Generates evaluation metrics: ROC-AUC, PR-AUC, F1, Precision, Recall
- Saves threshold metrics and curves
- Evaluates model trained on benign-only data for anomaly detection
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# === Setup for GPU memory and quiet logging ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = info, 2 = warning, 3 = error
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# ===
import json
import numpy as np
import argparse
from tqdm import tqdm
from scipy import sparse
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
    roc_curve, precision_recall_curve
)
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow.keras.backend as K
import gc

from utils.paths import get_model_dir, get_encoded_dir, get_eval_dir

# === Config ===
# === CLI Arguments ===
parser = argparse.ArgumentParser(description="Model Evaluation")
parser.add_argument("--dataset", required=True, help="Dataset name (e.g., unraveled)")
parser.add_argument("--version", required=True, help="Version string (e.g., v1)")
parser.add_argument("--model", required=True, help="Model (e.g., sae, lstm-sae, etc.)")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model
SAMPLE_SIZE = 1_000_000
BATCH_SIZE = 4096

# === Paths ===
ENCODED_DIR = get_encoded_dir(MODEL, DATASET, VERSION)
MODEL_DIR = get_model_dir(MODEL, DATASET, VERSION)
EVAL_DIR = get_eval_dir(MODEL, DATASET, VERSION)
os.makedirs(EVAL_DIR, exist_ok=True)

X_test_path = os.path.join(ENCODED_DIR, "X_test_encoded.npz")
y_test_path = os.path.join(ENCODED_DIR, "y_test.npy")
model_path = os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}.keras")
recon_path = os.path.join(EVAL_DIR, f"reconstruction_errors.npz")

# === Load Data ===
print("📦 Loading test data and trained model...")
X_test = sparse.load_npz(X_test_path)
y_test = np.load(y_test_path)
sae_model = load_model(model_path, compile=False)

print(f"✅ Test data: {X_test.shape}")
print(f"📊 Test label distribution:")
test_benign = np.sum(y_test == 0)
test_attack = np.sum(y_test == 1)
print(f"  Benign: {test_benign} ({test_benign/len(y_test):.1%})")
print(f"  Attack: {test_attack} ({test_attack/len(y_test):.1%})")

# === Compute reconstruction errors ===
def compute_and_save_reconstruction_errors():
    reconstruction_errors = []
    n_batches = (X_test.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"🔍 Computing reconstruction errors on {X_test.shape[0]} test samples...")
    
    for i in tqdm(range(0, X_test.shape[0], BATCH_SIZE), total=n_batches, desc="🔍 Reconstructing"):
        X_batch = X_test[i:i+BATCH_SIZE].toarray()
        X_pred = sae_model.predict(X_batch, verbose=0)
        mse = np.mean((X_batch - X_pred) ** 2, axis=1)
        reconstruction_errors.extend(mse)

        if (i // BATCH_SIZE) % 10 == 0:
            K.clear_session()
            gc.collect()

    reconstruction_errors = np.array(reconstruction_errors)
    np.savez_compressed(
        recon_path,
        reconstruction_errors=reconstruction_errors,
        y_labels=y_test
    )
    print("✅ Reconstruction errors computed and saved.")
    
    # === Analyze reconstruction error distributions ===
    benign_errors = reconstruction_errors[y_test == 0]
    attack_errors = reconstruction_errors[y_test == 1]
    print(f"📊 Reconstruction error analysis:")
    print(f"  Benign samples - Mean: {np.mean(benign_errors):.6f}, Std: {np.std(benign_errors):.6f}")
    print(f"  Attack samples - Mean: {np.mean(attack_errors):.6f}, Std: {np.std(attack_errors):.6f}")
    print(f"  Attack/Benign ratio: {np.mean(attack_errors)/np.mean(benign_errors):.2f}x")
    
    return reconstruction_errors

# === Check or compute reconstruction errors ===
if os.path.exists(recon_path):
    user_input = input(f"⚠️ Found existing reconstruction_errors.npz. Recompute? (y/n): ")
    if user_input.lower() != 'y':
        print("📂 Loading existing reconstruction errors...")
        data = np.load(recon_path)
        reconstruction_errors = data["reconstruction_errors"]
        y_test_loaded = data["y_labels"]
        print(f"✅ Loaded {len(reconstruction_errors)} reconstruction errors")
    else:
        reconstruction_errors = compute_and_save_reconstruction_errors()
else:
    print("📂 No existing reconstruction_errors.npz found. Computing...")
    reconstruction_errors = compute_and_save_reconstruction_errors()

# === Compute metrics ===
print("📊 Computing evaluation metrics...")
roc_auc = roc_auc_score(y_test, reconstruction_errors)
pr_auc = average_precision_score(y_test, reconstruction_errors)

thresholds = np.linspace(reconstruction_errors.min(), reconstruction_errors.max(), 100)
precisions, recalls, f1_scores = [], [], []
for thresh in thresholds:
    preds = (reconstruction_errors > thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f1)

best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]

# Final predictions at best threshold
final_preds = (reconstruction_errors > best_thresh).astype(int)
final_p, final_r, final_f1, _ = precision_recall_fscore_support(y_test, final_preds, average='binary', zero_division=0)

metrics = {
    "model": MODEL,
    "dataset": DATASET,
    "version": VERSION,
    "evaluation_set": "test",
    "test_samples": int(len(y_test)),
    "test_benign": int(test_benign),
    "test_attack": int(test_attack),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "best_threshold": float(best_thresh),
    "best_f1_score": float(final_f1),
    "best_precision": float(final_p),
    "best_recall": float(final_r),
    "mean_recon_error_benign": float(np.mean(reconstruction_errors[y_test == 0])),
    "mean_recon_error_attack": float(np.mean(reconstruction_errors[y_test == 1])),
    "evaluation_strategy": "reconstruction_error_anomaly_detection"
}

print(f"🎯 Evaluation Results:")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  PR-AUC: {pr_auc:.4f}")
print(f"  Best F1: {final_f1:.4f} (Precision: {final_p:.4f}, Recall: {final_r:.4f})")
print(f"  Best Threshold: {best_thresh:.6f}")
with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# === Save ROC/PR data ===
fpr, tpr, _ = roc_curve(y_test, reconstruction_errors)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, reconstruction_errors)
np.savez_compressed(os.path.join(EVAL_DIR, "roc_pr_data.npz"),
    fpr=fpr, tpr=tpr,
    precision_curve=precision_curve, recall_curve=recall_curve,
    roc_auc=roc_auc, pr_auc=pr_auc
)

# === Plot Threshold vs Scores ===
print("📈 Plotting threshold-based metrics...")
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label='F1-score')
plt.plot(thresholds, precisions, '--', label='Precision')
plt.plot(thresholds, recalls, '--', label='Recall')
plt.axvline(best_thresh, linestyle=':', color='gray')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold vs F1, Precision, Recall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "threshold_scores.png"))

# === Plot ROC & PR ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.4f}", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(recall_curve, precision_curve, label=f"PR-AUC={pr_auc:.4f}", color='green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "roc_pr_curves.png"))

print("✅ Evaluation complete. Artifacts saved:")
print(f" ├── {EVAL_DIR}/metrics.json")
print(f" ├── {EVAL_DIR}/roc_pr_data.npz")
print(f" ├── {EVAL_DIR}/threshold_scores.png")
print(f" └── {EVAL_DIR}/roc_pr_curves.png")

# === Cleanup ===
del sae_model, X_test
K.clear_session()
gc.collect()