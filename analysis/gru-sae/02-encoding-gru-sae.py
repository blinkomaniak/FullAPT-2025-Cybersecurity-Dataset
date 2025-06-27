"""
Stage 02: GRU Encoding with Anti-Data-Snooping Logic

This script performs feature encoding for GRU-SAE with proper temporal sequence creation
and early dataset splitting before fitting transformers. This prevents data leakage.

Steps:
1. Load and sort data chronologically
2. Split data temporally while preserving stratification (70% train, 15% val, 15% test)
3. Build sequences for each split separately
4. Apply feature encoding using training data only
5. Save encoded 3D sequences and metadata
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import sparse

from utils.paths import get_processed_path, get_encoded_dir, ensure_dirs
from utils.config import load_config
from utils.encoding import build_column_transformer

# === CLI Args ===
parser = argparse.ArgumentParser(description="Encode dataset into sequences for GRU-SAE")
parser.add_argument("--dataset", required=True, help="Dataset name")
parser.add_argument("--version", required=True, help="Dataset version")
parser.add_argument("--model", required=True, help="Model (e.g., gru-sae)")
parser.add_argument("--encoding_config", required=True, help="Path to encoding config JSON")
parser.add_argument("--sample-size", "--sample_size", type=int, default=2000, help="Number of sequences to sample per split")
parser.add_argument("--seq-len", type=int, default=50, help="Sequence length (timesteps) - optimized for memory constraints")
parser.add_argument("--max-features", type=int, default=1000, help="Maximum number of features after encoding (memory limit)")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model
SEQ_LEN = args.seq_len
MAX_FEATURES = args.max_features
SAMPLE_SIZE = args.sample_size  # Works with both --sample-size and --sample_size
ENCODING_CONFIG_PATH = args.encoding_config

# === Load parquet ===
parquet_path = get_processed_path(MODEL, DATASET, VERSION)
print(f"📦 Loading parquet: {parquet_path}")
df = pd.read_parquet(parquet_path)
print(f"📊 Loaded dataset: {df.shape}")

# === Load encoding config ===
print("⚙️ Loading encoding configuration...")
enc_config = load_config(ENCODING_CONFIG_PATH)
cat_cols = enc_config.get("categorical_columns", [])
num_cols = enc_config.get("numeric_columns", [])
label_col = enc_config["label_column"]
time_col = enc_config["time_column"]

# === Sort chronologically + validate time column ===
print("🧹 Sorting data chronologically...")
if time_col not in df.columns:
    raise ValueError(f"Time column '{time_col}' not found in dataset. Available columns: {list(df.columns)}")

# Sort by time to maintain temporal order
df = df.sort_values(time_col).reset_index(drop=True)
print(f"📊 Data sorted by {time_col}: {len(df)} total samples")

# Basic preprocessing
if cat_cols:
    available_cat_cols = [col for col in cat_cols if col in df.columns]
    if available_cat_cols:
        df[available_cat_cols] = df[available_cat_cols].fillna("missing").astype(str)
        print(f"📋 Processed {len(available_cat_cols)} categorical columns")
    else:
        print("⚠️ No categorical columns found in dataset")
        
if num_cols:
    available_num_cols = [col for col in num_cols if col in df.columns]
    if available_num_cols:
        df[available_num_cols] = df[available_num_cols].fillna(0).astype(np.float32)
        print(f"📋 Processed {len(available_num_cols)} numeric columns")
    else:
        print("⚠️ No numeric columns found in dataset")
        
available_feature_cols = [col for col in cat_cols + num_cols if col in df.columns]

# === Dataset-specific data splitting BEFORE encoding ===
print("🔀 Implementing dataset-specific data splitting strategy...")

if DATASET == "bigbase" and 'APT_Campaign' in df.columns:
    print("🎯 APT-aware splitting for bigbase dataset")
    
    # Strategic APT campaign allocation for generalization testing
    TRAIN_APTS = ["APT1", "APT2", "APT3", "APT4"]  # 4 APT types for training (45 sessions)
    VAL_APTS = ["APT2", "APT3"]                    # Subset of training APTs for validation  
    TEST_APTS = ["APT5", "APT6"]                   # Completely unseen APT types for testing (6 sessions)
    
    print(f"📊 APT allocation:")
    print(f"   Training APTs: {TRAIN_APTS} ({len([apt for apt in df['APT_Campaign'].unique() if apt in TRAIN_APTS])} campaigns)")
    print(f"   Validation APTs: {VAL_APTS} (subset of training)")
    print(f"   Test APTs: {TEST_APTS} ({len([apt for apt in df['APT_Campaign'].unique() if apt in TEST_APTS])} campaigns)")
    
    # Split by APT campaigns (prevents data leakage between attack types)
    train_df = df[df['APT_Campaign'].isin(TRAIN_APTS)].copy()
    test_df = df[df['APT_Campaign'].isin(TEST_APTS)].copy()
    
    # Validation set: sample from training APTs (within-attack-type validation)
    val_sessions_per_apt = 2  # Take 2 sessions per training APT for validation
    val_dfs = []
    train_filtered_dfs = []
    
    for apt in TRAIN_APTS:
        apt_df = train_df[train_df['APT_Campaign'] == apt]
        unique_sessions = apt_df['Session_File'].unique()
        
        if len(unique_sessions) > val_sessions_per_apt:
            # Select sessions for validation (deterministic selection)
            val_sessions = sorted(unique_sessions)[:val_sessions_per_apt]
            train_sessions = sorted(unique_sessions)[val_sessions_per_apt:]
            
            val_dfs.append(apt_df[apt_df['Session_File'].isin(val_sessions)])
            train_filtered_dfs.append(apt_df[apt_df['Session_File'].isin(train_sessions)])
        else:
            # If too few sessions, use 30% for validation
            val_size = int(0.3 * len(apt_df))
            val_dfs.append(apt_df[:val_size])
            train_filtered_dfs.append(apt_df[val_size:])
    
    # Create final splits
    train_df = pd.concat(train_filtered_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    
    # Sort each split chronologically to maintain temporal order within sessions
    train_df = train_df.sort_values([time_col]).reset_index(drop=True)
    val_df = val_df.sort_values([time_col]).reset_index(drop=True)  
    test_df = test_df.sort_values([time_col]).reset_index(drop=True)
    
    print(f"✅ APT-aware split complete:")
    print(f"   Train: {len(train_df)} events from {train_df['APT_Campaign'].nunique()} APTs, {train_df['Session_File'].nunique()} sessions")
    print(f"   Val:   {len(val_df)} events from {val_df['APT_Campaign'].nunique()} APTs, {val_df['Session_File'].nunique()} sessions")
    print(f"   Test:  {len(test_df)} events from {test_df['APT_Campaign'].nunique()} APTs, {test_df['Session_File'].nunique()} sessions")
    
else:
    print("⏰ Chronological splitting for unraveled dataset (timeline-based)")
    
    # For unraveled: use temporal splitting (preserves normal→attack→normal timeline)
    # Use stratified sampling within chronological windows
    window_size = len(df) // 10  # Use 10 temporal windows for stratification
    train_indices, temp_indices = [], []

    for i in range(0, len(df), window_size):
        window_end = min(i + window_size, len(df))
        window_df = df.iloc[i:window_end]
        
        if len(window_df) < 10:  # Skip very small windows
            continue
            
        window_train_size = int(0.7 * len(window_df))
        
        try:
            # Stratified split within this temporal window
            window_train, window_temp = train_test_split(
                window_df.index, 
                test_size=len(window_df) - window_train_size,
                stratify=window_df[label_col],
                random_state=42
            )
            train_indices.extend(window_train)
            temp_indices.extend(window_temp)
        except ValueError:  # If stratification fails (e.g., only one class)
            # Fall back to simple chronological split
            window_train = window_df.index[:window_train_size]
            window_temp = window_df.index[window_train_size:]
            train_indices.extend(window_train)
            temp_indices.extend(window_temp)

    # Second split: 15% val, 15% test from temp
    temp_df = df.loc[temp_indices]
    if len(temp_df) > 10:
        try:
            val_indices, test_indices = train_test_split(
                temp_df.index,
                test_size=0.5,
                stratify=temp_df[label_col],
                random_state=42
            )
        except ValueError:
            # Fall back to simple split
            mid_point = len(temp_indices) // 2
            val_indices = temp_indices[:mid_point]
            test_indices = temp_indices[mid_point:]
    else:
        val_indices = temp_indices[:len(temp_indices)//2]
        test_indices = temp_indices[len(temp_indices)//2:]

    # Create data splits
    train_df = df.loc[train_indices].sort_values(time_col).reset_index(drop=True)
    val_df = df.loc[val_indices].sort_values(time_col).reset_index(drop=True) 
    test_df = df.loc[test_indices].sort_values(time_col).reset_index(drop=True)

print(f"📊 Final split ratios - Train: {len(train_df)/len(df):.1%}, Val: {len(val_df)/len(df):.1%}, Test: {len(test_df)/len(df):.1%}")

# === Encode features using ONLY training data ===
print("🏗️ Building and fitting column transformer on training data only...")
column_transformer = build_column_transformer(
    [col for col in cat_cols if col in train_df.columns],
    [col for col in num_cols if col in train_df.columns]
)

# Fit transformer on training data only
X_train_encoded = column_transformer.fit_transform(train_df[available_feature_cols])
X_val_encoded = column_transformer.transform(val_df[available_feature_cols])
X_test_encoded = column_transformer.transform(test_df[available_feature_cols])

print(f"✅ Initial encoded shapes - Train: {X_train_encoded.shape}, Val: {X_val_encoded.shape}, Test: {X_test_encoded.shape}")

# === Sparse-Aware Feature Selection for Memory Management ===
if X_train_encoded.shape[1] > MAX_FEATURES:
    print(f"⚠️ Too many features ({X_train_encoded.shape[1]}) > limit ({MAX_FEATURES})")
    print("🎯 Applying SPARSE-AWARE dimensionality reduction...")
    
    from sklearn.decomposition import TruncatedSVD
    
    # ALWAYS use SVD for large sparse matrices (no conversion to dense!)
    print(f"🔄 Using TruncatedSVD for sparse matrix dimensionality reduction...")
    print(f"   Input: {X_train_encoded.shape} sparse matrix")
    print(f"   Target: {MAX_FEATURES} components")
    
    # TruncatedSVD works directly on sparse matrices - no memory explosion!
    svd = TruncatedSVD(n_components=MAX_FEATURES, random_state=42)
    
    print("🔄 Fitting SVD on training data...")
    X_train_reduced = svd.fit_transform(X_train_encoded)
    
    print("🔄 Transforming validation and test data...")
    X_val_reduced = svd.transform(X_val_encoded) 
    X_test_reduced = svd.transform(X_test_encoded)
    
    # Update encoded arrays (now dense but much smaller)
    X_train_encoded = X_train_reduced
    X_val_encoded = X_val_reduced 
    X_test_encoded = X_test_reduced
    
    print(f"✅ SVD reduction complete - reduced to {X_train_encoded.shape[1]} features")
    print(f"   Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
    
    # Clear intermediate variables
    del X_train_reduced, X_val_reduced, X_test_reduced
    gc.collect()

print(f"📊 Final encoded shapes - Train: {X_train_encoded.shape}, Val: {X_val_encoded.shape}, Test: {X_test_encoded.shape}")

# === Memory Usage Estimation ===
seq_memory_gb = (SAMPLE_SIZE * SEQ_LEN * X_train_encoded.shape[1] * 4) / (1024**3)  # 4 bytes per float32
print(f"💾 Estimated sequence memory usage: {seq_memory_gb:.2f} GB")

# Be more conservative with memory - 4GB limit instead of 8GB
memory_limit_gb = 4.0
if seq_memory_gb > memory_limit_gb:
    print(f"⚠️ Memory usage too high ({seq_memory_gb:.2f}GB > {memory_limit_gb}GB), reducing sample size...")
    max_safe_samples = int(memory_limit_gb * (1024**3) / (SEQ_LEN * X_train_encoded.shape[1] * 4))
    SAMPLE_SIZE = min(SAMPLE_SIZE, max_safe_samples)
    seq_memory_gb_new = (SAMPLE_SIZE * SEQ_LEN * X_train_encoded.shape[1] * 4) / (1024**3)
    print(f"🎯 Adjusted sample size to {SAMPLE_SIZE} (new memory: {seq_memory_gb_new:.2f}GB)")

def build_chronological_sequences(X, y, seq_len, max_sequences=None):
    """Build sequences maintaining chronological order"""
    n = X.shape[0]
    is_sparse = hasattr(X, "toarray")
    
    if n < seq_len:
        print(f"⚠️ Warning: Not enough data for sequences (need {seq_len}, have {n})")
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.array([], dtype=np.int32)
    
    # Calculate number of possible sequences
    n_sequences = n - seq_len + 1
    
    # Limit sequences if requested (for memory efficiency)
    if max_sequences and n_sequences > max_sequences:
        # Sample evenly across the temporal range
        step = n_sequences / max_sequences
        indices = np.round(np.arange(0, n_sequences, step)).astype(int)
        indices = indices[:max_sequences]  # Ensure we don't exceed max
        n_sequences = len(indices)
        print(f"📊 Sampling {n_sequences} sequences from {n - seq_len + 1} possible (step={step:.2f})")
    else:
        indices = np.arange(n_sequences)
        print(f"📊 Creating {n_sequences} sequences from {n} samples")
    
    # Pre-allocate arrays
    n_features = X.shape[1]
    X_seq = np.empty((n_sequences, seq_len, n_features), dtype=np.float32)
    y_seq = np.empty(n_sequences, dtype=np.int32)
    
    # Build sequences
    for j, i in enumerate(indices):
        end_idx = i + seq_len
        seq = X[i:end_idx]
        if is_sparse:
            seq = seq.toarray()
        X_seq[j] = seq
        y_seq[j] = y[end_idx - 1]  # Label from the last timestep
    
    return X_seq, y_seq

# === Build sequences for each split separately ===
print("🔁 Building chronological sequences for each split...")

# Build sequences with memory-efficient limits
max_seq_per_split = SAMPLE_SIZE // 3 if SAMPLE_SIZE else None

X_train_seq, y_train_seq = build_chronological_sequences(
    X_train_encoded, train_df[label_col].to_numpy(), SEQ_LEN, max_seq_per_split
)
X_val_seq, y_val_seq = build_chronological_sequences(
    X_val_encoded, val_df[label_col].to_numpy(), SEQ_LEN, max_seq_per_split
)
X_test_seq, y_test_seq = build_chronological_sequences(
    X_test_encoded, test_df[label_col].to_numpy(), SEQ_LEN, max_seq_per_split
)

print(f"✅ Sequence shapes:")
print(f"   Train: {X_train_seq.shape}, labels: {y_train_seq.shape}")
print(f"   Val:   {X_val_seq.shape}, labels: {y_val_seq.shape}")
print(f"   Test:  {X_test_seq.shape}, labels: {y_test_seq.shape}")

# === Label distribution analysis ===
print("\n📊 Label distribution analysis:")
train_benign = np.sum(y_train_seq == 0) if len(y_train_seq) > 0 else 0
train_attack = np.sum(y_train_seq == 1) if len(y_train_seq) > 0 else 0
val_benign = np.sum(y_val_seq == 0) if len(y_val_seq) > 0 else 0
val_attack = np.sum(y_val_seq == 1) if len(y_val_seq) > 0 else 0
test_benign = np.sum(y_test_seq == 0) if len(y_test_seq) > 0 else 0
test_attack = np.sum(y_test_seq == 1) if len(y_test_seq) > 0 else 0

if len(y_train_seq) > 0:
    print(f"Train - Benign: {train_benign} ({train_benign/len(y_train_seq):.1%}), Attack: {train_attack} ({train_attack/len(y_train_seq):.1%})")
if len(y_val_seq) > 0:
    print(f"Val   - Benign: {val_benign} ({val_benign/len(y_val_seq):.1%}), Attack: {val_attack} ({val_attack/len(y_val_seq):.1%})")
if len(y_test_seq) > 0:
    print(f"Test  - Benign: {test_benign} ({test_benign/len(y_test_seq):.1%}), Attack: {test_attack} ({test_attack/len(y_test_seq):.1%})")

# === Save outputs with consistent naming ===
encoded_dir = get_encoded_dir(MODEL, DATASET, VERSION)
ensure_dirs(encoded_dir)
print(f"\n💾 Saving encoded sequences to: {encoded_dir}")

# Save 3D sequences (use consistent naming with SAE/IForest)
np.save(os.path.join(encoded_dir, "X_train_sequences.npy"), X_train_seq)
np.save(os.path.join(encoded_dir, "X_val_sequences.npy"), X_val_seq)
np.save(os.path.join(encoded_dir, "X_test_sequences.npy"), X_test_seq)
np.save(os.path.join(encoded_dir, "y_train.npy"), y_train_seq)
np.save(os.path.join(encoded_dir, "y_val.npy"), y_val_seq)
np.save(os.path.join(encoded_dir, "y_test.npy"), y_test_seq)
joblib.dump(column_transformer, os.path.join(encoded_dir, "column_transformer.joblib"))

# Save metadata
meta = {
    "model": MODEL,
    "dataset": DATASET,
    "version": VERSION,
    "seq_len": SEQ_LEN,
    "train_shape": X_train_seq.shape,
    "val_shape": X_val_seq.shape,
    "test_shape": X_test_seq.shape,
    "n_features": X_train_seq.shape[2] if len(X_train_seq.shape) > 2 else 0,
    "n_timesteps": X_train_seq.shape[1] if len(X_train_seq.shape) > 1 else 0,
    "transformer_type": "TF-IDF + OneHot + StandardScaler" if DATASET == "bigbase" else "OneHot + StandardScaler",
    "splitting": "apt_aware_campaign_based" if DATASET == "bigbase" else "temporal_stratified_70_15_15",
    "sequence_building": "chronological",
    "train_benign_ratio": float(train_benign / len(y_train_seq)) if len(y_train_seq) > 0 else 0,
    "val_benign_ratio": float(val_benign / len(y_val_seq)) if len(y_val_seq) > 0 else 0,
    "test_benign_ratio": float(test_benign / len(y_test_seq)) if len(y_test_seq) > 0 else 0,
    "saved_at": datetime.now().isoformat()
}

# Add APT-specific metadata for bigbase
if DATASET == "bigbase" and 'APT_Campaign' in train_df.columns:
    meta["apt_splitting"] = {
        "train_apts": list(train_df['APT_Campaign'].unique()),
        "val_apts": list(val_df['APT_Campaign'].unique()),
        "test_apts": list(test_df['APT_Campaign'].unique()),
        "train_sessions": int(train_df['Session_File'].nunique()),
        "val_sessions": int(val_df['Session_File'].nunique()),
        "test_sessions": int(test_df['Session_File'].nunique()),
        "strategy": "unseen_apt_generalization"
    }

with open(os.path.join(encoded_dir, "encoding_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("📦 Artifacts saved:")
print(f"  Encoded train sequences: {encoded_dir}/X_train_sequences.npy")
print(f"  Encoded val sequences:   {encoded_dir}/X_val_sequences.npy")
print(f"  Encoded test sequences:  {encoded_dir}/X_test_sequences.npy")
print(f"  Labels:                  y_train.npy, y_val.npy, y_test.npy")
print(f"  Transformer:             column_transformer.joblib")
print(f"  Metadata:                encoding_metadata.json")
print("✅ Done.")

# === Cleanup ===
del X_train_encoded, X_val_encoded, X_test_encoded
del train_df, val_df, test_df, df
gc.collect()