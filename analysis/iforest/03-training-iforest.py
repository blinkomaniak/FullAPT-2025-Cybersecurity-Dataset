"""
Stage 03: Train Isolation Forest model with tunable sampling and PCA

- Loads encoded sparse training data
- Samples a subset, densifies, and applies PCA
- Trains Isolation Forest on reduced data
- Saves model and metadata to disk
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import argparse
import numpy as np
import joblib
import json
from datetime import datetime
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from utils.paths import get_encoded_dir, get_model_dir, ensure_dirs

# === CLI Args ===
parser = argparse.ArgumentParser(description="Train Isolation Forest model with PCA")
parser.add_argument("--dataset", required=True, help="Dataset name")
parser.add_argument("--version", required=True, help="Dataset version")
parser.add_argument("--model", required=True, help="Model name (e.g., iforest)")
parser.add_argument("--sample-size", type=int, default=100_000, help="Number of samples to use from X_train")
parser.add_argument("--n-components", type=int, default=50, help="Number of PCA components to retain")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model
SAMPLE_SIZE = args.sample_size
N_COMPONENTS = args.n_components

print(f"📦 Model: {MODEL}, Dataset: {DATASET}, Version: {VERSION}")
print(f"🔢 Sampling: {SAMPLE_SIZE} rows | PCA components: {N_COMPONENTS}")

# === Paths ===
ENCODED_DIR = get_encoded_dir(MODEL, DATASET, VERSION)
MODEL_DIR = get_model_dir(MODEL, DATASET, VERSION)
os.makedirs(MODEL_DIR, exist_ok=True)

X_train_path = os.path.join(ENCODED_DIR, "X_train_encoded.npz")
y_train_path = os.path.join(ENCODED_DIR, "y_train.npy")

# === Load Training Data ===
print("📥 Loading training data and labels...")
X_train_full = sparse.load_npz(X_train_path)
y_train_full = np.load(y_train_path)

print(f"✅ Full training data: {X_train_full.shape}")

# === CRITICAL: Filter to BENIGN-ONLY samples for anomaly detection ===
print("🔍 Filtering training data to BENIGN samples only...")
benign_mask = y_train_full == 0
X_sparse = X_train_full[benign_mask]
y_train_benign = y_train_full[benign_mask]

print(f"📊 Training samples - Total: {len(y_train_full)}, Benign: {len(y_train_benign)} ({len(y_train_benign)/len(y_train_full):.1%})")
print(f"✅ Using {len(y_train_benign)} BENIGN samples for Isolation Forest training")

# === Sample from Sparse Matrix ===
n_rows = X_sparse.shape[0]
sample_size = min(SAMPLE_SIZE, n_rows)
idx = np.random.choice(n_rows, size=sample_size, replace=False)
X_sample = X_sparse[idx].toarray()

print(f"✅ Sampled shape (dense): {X_sample.shape}")

# === Apply PCA ===
print(f"🧬 Running PCA (n_components={N_COMPONENTS})...")
pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_reduced = pca.fit_transform(X_sample)
print(f"✅ PCA-reduced shape: {X_reduced.shape}")

# === Train Isolation Forest ===
print("🌲 Training Isolation Forest...")
model = IsolationForest(
    n_estimators=100,
    contamination='auto',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_reduced)

# === Save Model and PCA ===
model_path = os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}.joblib")
pca_path = os.path.join(MODEL_DIR, f"pca-{MODEL}-model-{DATASET}-{VERSION}.joblib")
joblib.dump(model, model_path)
joblib.dump(pca, pca_path)

# === Save Metadata ===
metadata = {
    "model": MODEL,
    "dataset": DATASET,
    "version": VERSION,
    "model_type": "IsolationForest",
    "n_estimators": model.n_estimators,
    "contamination": model.contamination,
    "sample_size": sample_size,
    "n_components": N_COMPONENTS,
    "original_shape": [int(n_rows), int(X_sparse.shape[1])],
    "pca_shape": list(X_reduced.shape),
    "explained_variance_ratio": list(pca.explained_variance_ratio_),
    "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
    "saved_at": datetime.now().isoformat()
}
with open(os.path.join(MODEL_DIR, f"metadata-{MODEL}-model-{DATASET}-{VERSION}.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Artifacts saved to:")
print(f"  - Model:    {model_path}")
print(f"  - PCA:      {pca_path}")
print(f"  - Metadata: metadata-{MODEL}-model-{DATASET}-{VERSION}.json")

# === Cleanup ===
del model, X_sample, X_reduced, X_sparse
print("🏁 Training complete.")

