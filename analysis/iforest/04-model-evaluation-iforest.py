"""
Stage 04: Evaluate Isolation Forest model

- Loads test set and trained IF model
- Applies same PCA transformation
- Computes anomaly scores
- Evaluates ROC-AUC, PR-AUC, F1, Precision, Recall
- Saves scores, metrics, and visualizations
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from tqdm import tqdm
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
from scipy import sparse
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    roc_curve, precision_recall_curve
)
from sklearn.decomposition import PCA
import gc

from utils.paths import get_encoded_dir, get_model_dir, get_eval_dir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# === CLI Args ===
parser = argparse.ArgumentParser(description="Evaluate Isolation Forest model")
parser.add_argument("--dataset", required=True, help="Dataset name")
parser.add_argument("--version", required=True, help="Dataset version")
parser.add_argument("--model", required=True, help="Model name (e.g., iforest)")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model

# === Paths ===
ENCODED_DIR = get_encoded_dir(MODEL, DATASET, VERSION)
MODEL_DIR = get_model_dir(MODEL, DATASET, VERSION)
EVAL_DIR = get_eval_dir(MODEL, DATASET, VERSION)
os.makedirs(EVAL_DIR, exist_ok=True)

X_test_path = os.path.join(ENCODED_DIR, "X_test_encoded.npz")
y_test_path = os.path.join(ENCODED_DIR, "y_test.npy")
model_path = os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}.joblib")
pca_path = os.path.join(MODEL_DIR, f"pca-{MODEL}-model-{DATASET}-{VERSION}.joblib")
metadata_path = os.path.join(MODEL_DIR, f"metadata-{MODEL}-model-{DATASET}-{VERSION}.json")

# === Load artifacts ===
print("📦 Loading test data and trained model...")
X_test_sparse = sparse.load_npz(X_test_path)  # Keep as sparse
y_test = np.load(y_test_path)
model = joblib.load(model_path)
pca = joblib.load(pca_path)  # Load fitted PCA from training

with open(metadata_path) as f:
    metadata = json.load(f)

print(f"📊 Test data shape: {X_test_sparse.shape} (sparse)")

# === Apply PCA with batch processing ===
print(f"🧬 Applying PCA (n_components={metadata['n_components']}) with batch processing...")
start = time.time()

# Batch processing to avoid memory issues
batch_size = 10000
n_samples = X_test_sparse.shape[0]
X_test_pca_list = []

for i in tqdm(range(0, n_samples, batch_size), desc="PCA batches"):
    end_idx = min(i + batch_size, n_samples)
    batch_sparse = X_test_sparse[i:end_idx]
    batch_dense = batch_sparse.toarray()  # Densify only the batch
    batch_pca = pca.transform(batch_dense)  # Use fitted PCA (not fit_transform)
    X_test_pca_list.append(batch_pca)
    del batch_dense, batch_sparse  # Free memory

# Combine all batches
X_test_pca = np.vstack(X_test_pca_list)
del X_test_pca_list  # Free memory

print(f"PCA-reduced shape: {X_test_pca.shape} ({time.time() - start:.2f}s)")

# Clean up sparse matrix to free memory
del X_test_sparse
gc.collect()

# === Compute Anomaly Scores ===
print("🔍 Scoring anomalies...")
start = time.time()
anomaly_scores = -model.decision_function(X_test_pca)
print(f"Anomaly scores computed in {time.time() - start:.2f}s")

# === Save raw scores ===
np.savez_compressed(os.path.join(EVAL_DIR, "anomaly_scores.npz"),
    scores=anomaly_scores, y_labels=y_test
)

# === Compute Metrics ===
print("📊 Computing evaluation metrics...")
roc_auc = roc_auc_score(y_test, anomaly_scores)
pr_auc = average_precision_score(y_test, anomaly_scores)

thresholds = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 100)
precisions, recalls, f1_scores = [], [], []

for thresh in tqdm(thresholds, desc="🔎 Threshold scan"):
    preds = (anomaly_scores > thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f1)

best_idx = int(np.argmax(f1_scores))
best_thresh = thresholds[best_idx]

# === Save metrics ===
metrics = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "f1_score": f1_scores[best_idx],
    "precision": precisions[best_idx],
    "recall": recalls[best_idx],
    "best_threshold": float(best_thresh),
    "n_components": metadata["n_components"],
    "sample_size_train": metadata["sample_size"],
    "test_shape": list(X_test_pca.shape),
    "explained_variance_sum": metadata.get("explained_variance_sum"),
    "evaluated_at": datetime.now().isoformat()
}
with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# === Save ROC/PR curves ===
fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, anomaly_scores)
np.savez_compressed(os.path.join(EVAL_DIR, "roc_pr_data.npz"),
    fpr=fpr, tpr=tpr,
    precision_curve=prec_curve, recall_curve=rec_curve,
    roc_auc=roc_auc, pr_auc=pr_auc
)

# === Plot: Threshold vs F1, Precision, Recall ===
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label="F1-score")
plt.plot(thresholds, precisions, '--', label="Precision")
plt.plot(thresholds, recalls, '--', label="Recall")
plt.axvline(best_thresh, linestyle=':', color='gray')
plt.xlabel("Threshold (Anomaly Score)")
plt.ylabel("Score")
plt.title("Threshold vs F1, Precision, Recall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "threshold_scores.png"))

# === Plot: ROC + PR Curves ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"ROC-AUC: {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rec_curve, prec_curve, label=f"PR-AUC: {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "roc_pr_curves.png"))

print("✅ Evaluation complete. Artifacts saved to:")
print(" ├── metrics.json")
print(" ├── anomaly_scores.npz")
print(" ├── roc_pr_data.npz")
print(" ├── threshold_scores.png")
print(" └── roc_pr_curves.png")

