"""
Feature Space Projection for Isolation Forest Results
- PCA and UMAP visualization
- Colored by anomaly score, true label, and tactic label
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.decomposition import PCA
import json
from scipy import sparse

from utils.paths import get_encoded_dir, get_eval_dir

def timed(label, func, *args, **kwargs):
    import time
    print(f"⏳ {label}...")
    start = time.time()
    result = func(*args, **kwargs)
    print(f"✅ {label} completed in {time.time() - start:.2f} sec")
    return result

# === CLI Args ===
parser = argparse.ArgumentParser(description="Feature Space Visualization for IF Model")
parser.add_argument("--dataset", required=True, help="Dataset name")
parser.add_argument("--version", required=True, help="Version")
parser.add_argument("--model", required=True, help="Model (e.g., iforest)")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model

ENCODED_DIR = get_encoded_dir(MODEL, DATASET, VERSION)
EVAL_DIR = get_eval_dir(MODEL, DATASET, VERSION)
os.makedirs(EVAL_DIR, exist_ok=True)

# === Load Data ===
print("📦 Loading validation data and scores...")
X_sparse = sparse.load_npz(os.path.join(ENCODED_DIR, "X_val_encoded.npz"))

# === Subsample before densifying ===
sample_size = 100_000
n_rows = X_sparse.shape[0]
sample_size = min(sample_size, n_rows)
idx = np.random.choice(n_rows, sample_size, replace=False)
print(f"Sampling {sample_size} rows before PCA/UMAP...")
X_sample = X_sparse[idx].toarray()

# === Load labels ===
y_val = np.load(os.path.join(ENCODED_DIR, "y_val.npy"))[idx]

# === Load or compute anomaly scores for validation data ===
# Note: Main evaluation uses test data, but visualization uses validation data
# We need to compute anomaly scores for validation data
print("🔍 Computing anomaly scores for validation data...")
from utils.paths import get_model_dir
import joblib

# Load trained model and PCA
MODEL_DIR = get_model_dir(MODEL, DATASET, VERSION)
model = joblib.load(os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}.joblib"))
pca = joblib.load(os.path.join(MODEL_DIR, f"pca-{MODEL}-model-{DATASET}-{VERSION}.joblib"))

# Apply PCA to sampled validation data and compute scores
X_val_pca = pca.transform(X_sample)
anomaly_scores = -model.decision_function(X_val_pca)

print(f"✅ Computed {len(anomaly_scores)} anomaly scores for validation samples")

# Load threshold from test evaluation (for reference)
try:
    with open(os.path.join(EVAL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    threshold = metrics["best_threshold"]
    print(f"🎯 Using threshold from test evaluation: {threshold:.4f}")
except FileNotFoundError:
    print("⚠️ No test metrics found, using median anomaly score as threshold")
    threshold = np.median(anomaly_scores)

# === Load TacticLabel
from utils.paths import get_processed_path
processed_path = get_processed_path(MODEL, DATASET, VERSION)
df = pd.read_parquet(processed_path)
# === Dynamically choose the label column based on dataset
if DATASET.lower() == "unraveled":
    label_col = "Stage"
elif DATASET.lower() == "bigbase":
    label_col = "TacticLabel"
else:
    raise ValueError(f"Unsupported dataset: {DATASET}")

# === Validate column exists
if label_col not in df.columns:
    raise ValueError(f"Missing column '{label_col}' in parquet file.")

# Load full label column (Stage or TacticLabel) for ALL rows

# WARNING: The relationship between df rows and X_val rows is complex due to stratified splitting
# For visualization purposes, we'll use a simpler approach: sample tactics from all data
print("⚠️ Note: Using random sampling of tactic labels for visualization")
print("   (Exact alignment with validation split is complex due to stratified sampling)")

# Sample tactics proportionally from the dataset
if len(df[label_col].unique()) > 20:
    # If too many unique tactics, sample the most common ones
    top_tactics = df[label_col].value_counts().head(10).index.tolist()
    tactics = np.random.choice(top_tactics, size=sample_size, replace=True)
else:
    # Sample from all available tactics
    all_tactics = df[label_col].values
    tactics = np.random.choice(all_tactics, size=sample_size, replace=True)

# === PCA Projection
print("🧬 Running PCA projection...")
start = time.time()
pca = PCA(n_components=2, random_state=42)
Z_pca = pca.fit_transform(X_sample)
print(f"PCA complete in {time.time() - start:.2f} sec")
np.save(os.path.join(EVAL_DIR, "Z_2d_pca.npy"), Z_pca)

# === UMAP Projection
print("🧬 Running UMAP projection...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
Z_umap = reducer.fit_transform(X_sample)
print(f"UMAP complete in {time.time() - start:.2f} sec")
np.save(os.path.join(EVAL_DIR, "Z_umap.npy"), Z_umap)

# === Common Plotting Helper
def plot_projection(Z, color_values, title, filename, cmap='inferno'):
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=color_values, cmap=cmap, s=6, alpha=0.6)
    if not isinstance(color_values[0], str):
        plt.colorbar(sc)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, filename))

# === Plot sets
print("🎨 Plotting PCA...")
plot_projection(Z_pca, anomaly_scores, "PCA — Anomaly Score (higher = more anomalous)", "pca_anomaly_score.png")
plot_projection(Z_pca, y_val, "PCA — True Label (Benign=0, Attack=1)", "pca_true_labels.png")

# Colored by TacticLabel
unique_tactics = sorted(set(tactics))
palette = sns.color_palette("hsv", len(unique_tactics))
color_map = {label: palette[i] for i, label in enumerate(unique_tactics)}
colors = [color_map[t] for t in tactics]
plot_projection(Z_pca, colors, "PCA — Tactic Label", "pca_tactic_labels.png", cmap=None)

print("🎨 Plotting UMAP...")
plot_projection(Z_umap, anomaly_scores, "UMAP — Anomaly Score", "umap_anomaly_score.png")
plot_projection(Z_umap, y_val, "UMAP — True Label (Benign=0, Attack=1)", "umap_true_labels.png")
plot_projection(Z_umap, colors, "UMAP — Tactic Label", "umap_tactic_labels.png", cmap=None)

print("✅ Visualization complete. Artifacts saved to:")
for f in ["pca_anomaly_score.png", "pca_true_labels.png", "pca_tactic_labels.png",
          "umap_anomaly_score.png", "umap_true_labels.png", "umap_tactic_labels.png"]:
    print(f" └── {f}")
