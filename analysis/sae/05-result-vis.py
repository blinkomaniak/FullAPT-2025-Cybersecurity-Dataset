"""
Latent Space Projection using SAE Bottleneck (Test Set Evaluation)
==================================================================
This script performs dimensionality reduction (PCA and UMAP) on the latent space
of a trained SAE model using the test set for proper evaluation visualization.

It provides:
- 2D PCA and UMAP projections of the bottleneck representation
- Visualization colored by true labels (benign vs attack)
- Analysis of how well the SAE separates different classes in latent space
- Proper evaluation using hold-out test data

Output artifacts include:
- 2D PCA and UMAP projections
- Label visualizations
- Saved arrays for external plotting
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# === Setup for GPU memory and quiet logging ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = info, 2 = warning, 3 = error
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.decomposition import PCA
from scipy import sparse
import umap
import gc
from tqdm import tqdm
from utils.paths import get_encoded_dir, get_model_dir, get_eval_dir

# === Config ===
# === CLI Arguments ===
parser = argparse.ArgumentParser(description="Latent Space visualization (PCA, UMAP)")
parser.add_argument("--dataset", required=True, help="Dataset name (e.g., unraveled)")
parser.add_argument("--version", required=True, help="Version string (e.g., v1)")
parser.add_argument("--model", required=True, help="Model (e.g., sae, lstm-sae, etc.")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model
MAX_SAMPLE_SIZE = 50_000  # Reduced for test set visualization
BATCH_SIZE = 4096

# === Paths ===
ENCODED_DIR = get_encoded_dir(MODEL, DATASET, VERSION)
MODEL_DIR = get_model_dir(MODEL, DATASET, VERSION)
EVAL_DIR = get_eval_dir(MODEL, DATASET, VERSION)
os.makedirs(EVAL_DIR, exist_ok=True)

print(f"📂 Loading test data from {ENCODED_DIR}")
X_test_path = os.path.join(ENCODED_DIR, "X_test_encoded.npz")
y_test_path = os.path.join(ENCODED_DIR, "y_test.npy")
model_path = os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}.keras")

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

# === Extract encoder ===
input_dim = X_test.shape[1]
input_layer = Input(shape=(input_dim,))
x = Dense(1024, activation='relu')(input_layer)
x = Dense(512, activation='relu')(x)
bottleneck = Dense(128, activation='relu')(x)
encoder = Model(inputs=input_layer, outputs=bottleneck)
for i in range(3):
    encoder.layers[i + 1].set_weights(sae_model.layers[i].get_weights())

print("🧬 Extracted encoder from trained SAE")

# === Sample from test set ===
n_test_samples = X_test.shape[0]
sample_size = min(MAX_SAMPLE_SIZE, n_test_samples)
idx = np.random.choice(n_test_samples, size=sample_size, replace=False)
X_sample = X_test[idx].toarray()
y_sample = y_test[idx]

print(f"📊 Using {sample_size} samples from test set for visualization")
sample_benign = np.sum(y_sample == 0)
sample_attack = np.sum(y_sample == 1)
print(f"  Sample - Benign: {sample_benign} ({sample_benign/len(y_sample):.1%}), Attack: {sample_attack} ({sample_attack/len(y_sample):.1%})")

# === Encode sample ===
print("🔐 Projecting into latent space...")
Z_latent = np.vstack([
    encoder.predict(X_sample[i:i + BATCH_SIZE], verbose=0)
    for i in range(0, X_sample.shape[0], BATCH_SIZE)
])

# === PCA and UMAP ===
print("🧬 Reducing dimensions...")
Z_pca = PCA(n_components=2).fit_transform(Z_latent)
Z_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42).fit_transform(Z_latent)

# === Save projections ===
np.save(os.path.join(EVAL_DIR, "Z_2d_pca.npy"), Z_pca)
np.save(os.path.join(EVAL_DIR, "Z_umap.npy"), Z_umap)
np.save(os.path.join(EVAL_DIR, "y_sample.npy"), y_sample)
np.save(os.path.join(EVAL_DIR, "test_sample_indices.npy"), idx)

# === Plot by class labels ===
print("🎨 Plotting latent space projections...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Colors for benign (blue) and attack (red)
colors = ['#2E86AB', '#F24236']  # Blue for benign, red for attack
labels = ['Benign', 'Attack']

# Plot PCA projection
ax = axes[0, 0]
for class_label in [0, 1]:
    mask = (y_sample == class_label)
    ax.scatter(Z_pca[mask, 0], Z_pca[mask, 1], 
              s=8, alpha=0.6, label=labels[class_label], color=colors[class_label])
ax.set_title("SAE Bottleneck: PCA Projection (Test Set)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot UMAP projection  
ax = axes[0, 1]
for class_label in [0, 1]:
    mask = (y_sample == class_label)
    ax.scatter(Z_umap[mask, 0], Z_umap[mask, 1], 
              s=8, alpha=0.6, label=labels[class_label], color=colors[class_label])
ax.set_title("SAE Bottleneck: UMAP Projection (Test Set)")
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot PCA with density
ax = axes[1, 0]
ax.scatter(Z_pca[y_sample == 0, 0], Z_pca[y_sample == 0, 1], 
          s=8, alpha=0.4, label='Benign', color=colors[0])
ax.scatter(Z_pca[y_sample == 1, 0], Z_pca[y_sample == 1, 1], 
          s=8, alpha=0.7, label='Attack', color=colors[1])
ax.set_title("PCA: Attack samples highlighted")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot UMAP with density
ax = axes[1, 1]
ax.scatter(Z_umap[y_sample == 0, 0], Z_umap[y_sample == 0, 1], 
          s=8, alpha=0.4, label='Benign', color=colors[0])
ax.scatter(Z_umap[y_sample == 1, 0], Z_umap[y_sample == 1, 1], 
          s=8, alpha=0.7, label='Attack', color=colors[1])
ax.set_title("UMAP: Attack samples highlighted")
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "latent_space_projections.png"), dpi=150, bbox_inches='tight')

# === Individual plots for better visibility ===
# PCA only
plt.figure(figsize=(10, 8))
for class_label in [0, 1]:
    mask = (y_sample == class_label)
    plt.scatter(Z_pca[mask, 0], Z_pca[mask, 1], 
              s=12, alpha=0.6, label=f'{labels[class_label]} (n={np.sum(mask)})', 
              color=colors[class_label])
plt.title(f"SAE Latent Space: PCA Projection\n({sample_size} test samples)", fontsize=14)
plt.xlabel("First Principal Component", fontsize=12)
plt.ylabel("Second Principal Component", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "pca_projection.png"), dpi=150, bbox_inches='tight')

# UMAP only
plt.figure(figsize=(10, 8))
for class_label in [0, 1]:
    mask = (y_sample == class_label)
    plt.scatter(Z_umap[mask, 0], Z_umap[mask, 1], 
              s=12, alpha=0.6, label=f'{labels[class_label]} (n={np.sum(mask)})', 
              color=colors[class_label])
plt.title(f"SAE Latent Space: UMAP Projection\n({sample_size} test samples)", fontsize=14)
plt.xlabel("UMAP Dimension 1", fontsize=12)
plt.ylabel("UMAP Dimension 2", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "umap_projection.png"), dpi=150, bbox_inches='tight')

# === Analysis Summary ===
print("\n📊 Latent Space Analysis Summary:")
print(f"  Model: {MODEL}")
print(f"  Dataset: {DATASET} (version {VERSION})")
print(f"  Visualization samples: {sample_size} from test set")
print(f"  Benign samples: {sample_benign} ({sample_benign/sample_size:.1%})")
print(f"  Attack samples: {sample_attack} ({sample_attack/sample_size:.1%})")
print(f"  Latent dimension: 128 → 2 (via PCA/UMAP)")

print("\n✅ Latent space projections saved to:")
print(f" ├── Combined view: {os.path.join(EVAL_DIR, 'latent_space_projections.png')}")
print(f" ├── PCA projection: {os.path.join(EVAL_DIR, 'pca_projection.png')}")
print(f" ├── UMAP projection: {os.path.join(EVAL_DIR, 'umap_projection.png')}")
print(f" └── Data arrays: Z_2d_pca.npy, Z_umap.npy, y_sample.npy")
print("🎯 Use these visualizations to assess SAE's ability to separate benign/attack patterns in latent space.")

# === Cleanup ===
del sae_model, encoder, X_sample, Z_latent
K.clear_session()
gc.collect()