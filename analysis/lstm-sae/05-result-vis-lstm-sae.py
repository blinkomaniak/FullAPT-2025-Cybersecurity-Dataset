"""
Stage 05: LSTM-SAE Result Visualization with Temporal Analysis

This script creates comprehensive visualizations for LSTM-SAE anomaly detection results.
It includes temporal sequence analysis, reconstruction error distributions, and latent
space projections tailored for 3D sequence data.

Steps:
1. Load evaluation results and model artifacts
2. Create temporal sequence visualizations  
3. Analyze reconstruction error patterns over time
4. Generate encoder latent space projections
5. Produce comprehensive result summaries
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# === Setup for GPU memory and quiet logging ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'default'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import gc
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.paths import get_encoded_dir, get_model_dir, get_eval_dir

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# === CLI Arguments ===
parser = argparse.ArgumentParser(description="Visualize LSTM-SAE anomaly detection results")
parser.add_argument("--dataset", required=True, help="Dataset name")
parser.add_argument("--version", required=True, help="Dataset version")
parser.add_argument("--model", required=True, help="Model name (e.g., lstm-sae)")
parser.add_argument("--sample-size", type=int, default=5000, help="Sample size for visualizations")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model
SAMPLE_SIZE = args.sample_size
BATCH_SIZE = 32

print(f"📦 Model: {MODEL}, Dataset: {DATASET}, Version: {VERSION}")

# === Paths ===
ENCODED_DIR = get_encoded_dir(MODEL, DATASET, VERSION)
MODEL_DIR = get_model_dir(MODEL, DATASET, VERSION)
EVAL_DIR = get_eval_dir(MODEL, DATASET, VERSION)
os.makedirs(EVAL_DIR, exist_ok=True)

# File paths for LSTM-SAE data
X_test_path = os.path.join(ENCODED_DIR, "X_test_sequences.npy")
y_test_path = os.path.join(ENCODED_DIR, "y_test.npy")
model_path = os.path.join(MODEL_DIR, f"{MODEL}-model-{DATASET}-{VERSION}.keras")
metrics_path = os.path.join(EVAL_DIR, "metrics.json")
recon_errors_path = os.path.join(EVAL_DIR, "reconstruction_errors.npz")

print("📥 Loading LSTM-SAE evaluation data and model...")
try:
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    print(f"✅ Test sequences: {X_test.shape}, labels: {y_test.shape}")
except FileNotFoundError as e:
    print(f"❌ Error loading test data: {e}")
    print("💡 Make sure Stage 4 (evaluation) has completed successfully")
    sys.exit(1)

try:
    with open(metrics_path) as f:
        metrics = json.load(f)
    print(f"✅ Loaded evaluation metrics")
except FileNotFoundError:
    print("❌ Evaluation metrics not found")
    print("💡 Make sure Stage 4 (evaluation) has completed successfully")
    sys.exit(1)

try:
    recon_data = np.load(recon_errors_path)
    reconstruction_errors = recon_data['errors']
    y_labels = recon_data['y_labels']
    print(f"✅ Loaded reconstruction errors: {len(reconstruction_errors)} samples")
except FileNotFoundError:
    print("❌ Reconstruction errors not found")
    print("💡 Make sure Stage 4 (evaluation) has completed successfully")
    sys.exit(1)

try:
    model = load_model(model_path, compile=False)
    print(f"✅ Loaded LSTM-SAE model")
except FileNotFoundError:
    print(f"❌ Model not found: {model_path}")
    sys.exit(1)

print(f"📊 Performance Summary:")
print(f"   ROC-AUC: {metrics['performance']['roc_auc']:.4f}")
print(f"   PR-AUC:  {metrics['performance']['pr_auc']:.4f}")
print(f"   F1:      {metrics['performance']['f1_score']:.4f}")

# === Sample data for visualization ===
if len(X_test) > SAMPLE_SIZE:
    print(f"🎯 Sampling {SAMPLE_SIZE} sequences from {len(X_test)} available for visualization")
    indices = np.random.choice(len(X_test), SAMPLE_SIZE, replace=False)
    X_vis = X_test[indices]
    y_vis = y_test[indices]
    errors_vis = reconstruction_errors[indices]
else:
    X_vis = X_test
    y_vis = y_test
    errors_vis = reconstruction_errors

print(f"✅ Visualization sample: {X_vis.shape} sequences")

# === Extract LSTM encoder for latent space analysis ===
print("🧠 Extracting LSTM encoder for latent space analysis...")

def extract_lstm_encoder(model):
    """Extract encoder part from LSTM autoencoder"""
    # Find the repeat vector layer (bridge between encoder and decoder)
    repeat_layer_idx = None
    for i, layer in enumerate(model.layers):
        if 'repeat' in layer.name.lower():
            repeat_layer_idx = i
            break
    
    if repeat_layer_idx is None:
        print("⚠️ Could not find repeat vector layer, using first half of model")
        # Fallback: use roughly first half of layers
        repeat_layer_idx = len(model.layers) // 2
    
    # Create encoder model up to the repeat vector
    encoder_output = model.layers[repeat_layer_idx - 1].output
    encoder = Model(inputs=model.input, outputs=encoder_output)
    return encoder

encoder = extract_lstm_encoder(model)
print(f"✅ Extracted encoder with output shape: {encoder.output_shape}")

# === Compute latent representations ===
print("🔐 Computing latent representations for sample sequences...")

def compute_latent_batch(encoder, X, batch_size=32):
    """Compute latent representations in batches"""
    n_samples = X.shape[0]
    latent_representations = []
    
    for i in tqdm(range(0, n_samples, batch_size), desc="Computing latent representations"):
        batch_end = min(i + batch_size, n_samples)
        X_batch = X[i:batch_end]
        latent_batch = encoder.predict(X_batch, verbose=0)
        latent_representations.append(latent_batch)
        
        # Cleanup
        del X_batch
        gc.collect()
    
    return np.vstack(latent_representations)

Z_latent = compute_latent_batch(encoder, X_vis, BATCH_SIZE)
print(f"✅ Latent representations computed: {Z_latent.shape}")

# === Dimensionality reduction for latent space ===
print("🧬 Reducing latent space dimensions for visualization...")
Z_pca = PCA(n_components=2, random_state=42).fit_transform(Z_latent)
Z_tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(Z_latent)//4)).fit_transform(Z_latent)

print(f"✅ PCA projection: {Z_pca.shape}")
print(f"✅ t-SNE projection: {Z_tsne.shape}")

# Save projections
np.savez_compressed(
    os.path.join(EVAL_DIR, "latent_projections.npz"),
    Z_pca=Z_pca,
    Z_tsne=Z_tsne,
    y_labels=y_vis,
    reconstruction_errors=errors_vis
)

# === Generate Comprehensive Visualizations ===
print("📈 Generating LSTM-SAE result visualizations...")

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# === Plot 1: Latent Space Projections ===
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# PCA by label
benign_mask = y_vis == 0
attack_mask = y_vis == 1

axes[0, 0].scatter(Z_pca[benign_mask, 0], Z_pca[benign_mask, 1], 
                  s=20, alpha=0.6, label=f'Benign (n={np.sum(benign_mask)})', color='blue')
axes[0, 0].scatter(Z_pca[attack_mask, 0], Z_pca[attack_mask, 1], 
                  s=20, alpha=0.6, label=f'Attack (n={np.sum(attack_mask)})', color='red')
axes[0, 0].set_title('LSTM Encoder Latent Space (PCA)')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# t-SNE by label
axes[0, 1].scatter(Z_tsne[benign_mask, 0], Z_tsne[benign_mask, 1], 
                  s=20, alpha=0.6, label=f'Benign (n={np.sum(benign_mask)})', color='blue')
axes[0, 1].scatter(Z_tsne[attack_mask, 0], Z_tsne[attack_mask, 1], 
                  s=20, alpha=0.6, label=f'Attack (n={np.sum(attack_mask)})', color='red')
axes[0, 1].set_title('LSTM Encoder Latent Space (t-SNE)')
axes[0, 1].set_xlabel('t-SNE 1')
axes[0, 1].set_ylabel('t-SNE 2')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# PCA colored by reconstruction error
scatter = axes[1, 0].scatter(Z_pca[:, 0], Z_pca[:, 1], 
                           c=errors_vis, s=20, alpha=0.6, cmap='viridis')
axes[1, 0].set_title('Latent Space Colored by Reconstruction Error (PCA)')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Reconstruction Error')

# t-SNE colored by reconstruction error
scatter = axes[1, 1].scatter(Z_tsne[:, 0], Z_tsne[:, 1], 
                           c=errors_vis, s=20, alpha=0.6, cmap='viridis')
axes[1, 1].set_title('Latent Space Colored by Reconstruction Error (t-SNE)')
axes[1, 1].set_xlabel('t-SNE 1')
axes[1, 1].set_ylabel('t-SNE 2')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='Reconstruction Error')

plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "latent_space_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# === Plot 2: Sequence Analysis ===
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Reconstruction error distribution
benign_errors = errors_vis[benign_mask]
attack_errors = errors_vis[attack_mask]

axes[0, 0].hist(benign_errors, bins=50, alpha=0.7, label=f'Benign (μ={np.mean(benign_errors):.4f})', 
               density=True, color='blue')
axes[0, 0].hist(attack_errors, bins=50, alpha=0.7, label=f'Attack (μ={np.mean(attack_errors):.4f})', 
               density=True, color='red')
axes[0, 0].axvline(metrics['performance']['best_threshold'], linestyle='--', color='black', 
                  label=f'Threshold ({metrics["performance"]["best_threshold"]:.4f})')
axes[0, 0].set_title('Reconstruction Error Distribution')
axes[0, 0].set_xlabel('Reconstruction Error')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Error vs sequence length (if we have variable length sequences)
# For fixed length, show error distribution by percentile
error_percentiles = np.percentile(errors_vis, [25, 50, 75, 90, 95, 99])
percentile_labels = ['25th', '50th', '75th', '90th', '95th', '99th']
axes[0, 1].bar(percentile_labels, error_percentiles, color='skyblue', alpha=0.7)
axes[0, 1].set_title('Reconstruction Error Percentiles')
axes[0, 1].set_xlabel('Percentile')
axes[0, 1].set_ylabel('Reconstruction Error')
axes[0, 1].grid(True, alpha=0.3)

# Sample sequence visualization
if len(X_vis) > 0:
    # Show a benign and attack sequence
    benign_idx = np.where(benign_mask)[0][0] if np.any(benign_mask) else 0
    attack_idx = np.where(attack_mask)[0][0] if np.any(attack_mask) else 0
    
    # Plot feature evolution over time for sample sequences
    timesteps = range(X_vis.shape[1])
    
    # Average feature value per timestep for visualization
    benign_seq_avg = np.mean(X_vis[benign_idx], axis=1)
    attack_seq_avg = np.mean(X_vis[attack_idx], axis=1) if attack_idx < len(X_vis) else benign_seq_avg
    
    axes[1, 0].plot(timesteps, benign_seq_avg, label=f'Benign (error={errors_vis[benign_idx]:.4f})', 
                   color='blue', linewidth=2)
    axes[1, 0].plot(timesteps, attack_seq_avg, label=f'Attack (error={errors_vis[attack_idx]:.4f})', 
                   color='red', linewidth=2)
    axes[1, 0].set_title('Sample Sequence Feature Evolution')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Average Feature Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature importance (variance across timesteps)
    feature_variance = np.var(X_vis.reshape(-1, X_vis.shape[2]), axis=0)
    top_features = np.argsort(feature_variance)[-20:]  # Top 20 most varying features
    
    axes[1, 1].bar(range(len(top_features)), feature_variance[top_features], color='green', alpha=0.7)
    axes[1, 1].set_title('Top 20 Most Variable Features')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'No sequences available', ha='center', va='center')
    axes[1, 1].text(0.5, 0.5, 'No sequences available', ha='center', va='center')

plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "sequence_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# === Plot 3: Performance Summary ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Performance metrics bar chart
metric_names = ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Precision', 'Recall']
metric_values = [
    metrics['performance']['roc_auc'],
    metrics['performance']['pr_auc'],
    metrics['performance']['f1_score'],
    metrics['performance']['precision'],
    metrics['performance']['recall']
]

axes[0].bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red', 'purple'], alpha=0.7)
axes[0].set_title('LSTM-SAE Performance Metrics')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1)
for i, v in enumerate(metric_values):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center')
axes[0].grid(True, alpha=0.3)

# Dataset distribution
dist_labels = ['Train Benign', 'Train Attack', 'Test Benign', 'Test Attack']
dist_values = [
    metrics['data_stats']['test_benign'],  # Using test stats as proxy
    metrics['data_stats']['test_attack'],
    metrics['data_stats']['test_benign'],
    metrics['data_stats']['test_attack']
]
axes[1].pie([dist_values[0] + dist_values[2], dist_values[1] + dist_values[3]], 
           labels=['Benign', 'Attack'], autopct='%1.1f%%', 
           colors=['lightblue', 'lightcoral'])
axes[1].set_title('Test Data Distribution')

# Architecture summary (text)
arch_info = metrics.get('architecture', {})
# Get final val loss safely
final_val_loss = metrics.get('performance', {}).get('final_val_loss', None)
final_val_loss_str = f"{final_val_loss:.6f}" if final_val_loss is not None else "N/A"

arch_text = f"""LSTM-SAE Architecture:

Sequence Length: {arch_info.get('seq_len', 'N/A')}
Input Features: {arch_info.get('input_dim', 'N/A')}
Encoder Units: {arch_info.get('encoder_units', 'N/A')}
Decoder Units: {arch_info.get('decoder_units', 'N/A')}
Dropout Rate: {arch_info.get('dropout_rate', 'N/A')}

Training:
Batch Size: {metrics.get('training', {}).get('batch_size', 'N/A')}
Epochs: {metrics.get('training', {}).get('epochs_completed', 'N/A')}
Final Val Loss: {final_val_loss_str}"""

axes[2].text(0.05, 0.95, arch_text, transform=axes[2].transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace')
axes[2].set_xlim(0, 1)
axes[2].set_ylim(0, 1)
axes[2].axis('off')
axes[2].set_title('Model Configuration')

plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "performance_summary.png"), dpi=300, bbox_inches='tight')
plt.close()

# === Save Summary Report ===
print("📝 Generating summary report...")

summary_report = {
    "model": MODEL,
    "dataset": DATASET,
    "version": VERSION,
    "visualization_summary": {
        "sample_size": len(X_vis),
        "latent_dimensions": Z_latent.shape[1],
        "visualization_techniques": ["PCA", "t-SNE"],
        "plots_generated": [
            "latent_space_analysis.png",
            "sequence_analysis.png", 
            "performance_summary.png"
        ]
    },
    "performance_highlights": {
        "roc_auc": metrics['performance']['roc_auc'],
        "pr_auc": metrics['performance']['pr_auc'],
        "f1_score": metrics['performance']['f1_score'],
        "best_threshold": metrics['performance']['best_threshold']
    },
    "sequence_statistics": {
        "sequence_length": X_vis.shape[1] if len(X_vis) > 0 else 0,
        "feature_count": X_vis.shape[2] if len(X_vis) > 0 else 0,
        "benign_sequences": int(np.sum(benign_mask)),
        "attack_sequences": int(np.sum(attack_mask)),
        "avg_benign_error": float(np.mean(benign_errors)) if len(benign_errors) > 0 else 0,
        "avg_attack_error": float(np.mean(attack_errors)) if len(attack_errors) > 0 else 0
    },
    "generated_at": datetime.now().isoformat()
}

with open(os.path.join(EVAL_DIR, "visualization_summary.json"), "w") as f:
    json.dump(summary_report, f, indent=2)

print("✅ LSTM-SAE visualization complete! Artifacts saved:")
print(f"📁 Results directory: {EVAL_DIR}/")
print(" ├── latent_space_analysis.png")
print(" ├── sequence_analysis.png")
print(" ├── performance_summary.png")
print(" ├── latent_projections.npz")
print(" └── visualization_summary.json")

print(f"\n📊 Visualization Summary:")
print(f"   Sample size: {len(X_vis)} sequences")
print(f"   Latent dims: {Z_latent.shape[1]}")
print(f"   Benign/Attack: {np.sum(benign_mask)}/{np.sum(attack_mask)}")
print(f"   Error separation: {np.mean(attack_errors):.4f} vs {np.mean(benign_errors):.4f}")

# === Cleanup ===
del model, encoder, X_vis, Z_latent, Z_pca, Z_tsne
tf.keras.backend.clear_session()
gc.collect()

print("🏁 LSTM-SAE visualization complete.")