"""
Stage 02: Encoding UNRAVELED dataset with Anti-Data-Snooping Logic

This script performs feature encoding for the UNRAVELED dataset, with early dataset splitting
into training, validation, and test sets before fitting the transformer. This prevents data leakage
(i.e., data snooping) from validation/test into training during the encoding stage.

Steps:
1. Load and split the dataset using stratified logic (70% train, 15% val, 15% test)
2. Apply appropriate preprocessing (missing value handling)
3. Build and fit column transformer using training data only
4. Transform training, validation, and test data
5. Save encoded splits, transformer, and metadata
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import argparse
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime
from utils.paths import get_processed_path, get_encoded_dir, ensure_dirs
from utils.config import load_config
from utils.encoding import build_column_transformer
from scipy.sparse import issparse

# === CLI Arguments ===
parser = argparse.ArgumentParser(description="Encode dataset with train/val split")
parser.add_argument("--dataset", required=True, help="Dataset name (e.g., unraveled)")
parser.add_argument("--version", required=True, help="Version string (e.g., v1)")
parser.add_argument("--model", required=True, help="Model (e.g., sae, lstm-sae)")
parser.add_argument("--encoding_config", required=True, help="Path to encoding config JSON")
args = parser.parse_args()

DATASET = args.dataset
VERSION = args.version
MODEL = args.model
ENCODING_CONFIG_PATH = args.encoding_config

# === Load Data ===
print("📥 Loading dataset...")
parquet_path = get_processed_path(MODEL, DATASET, VERSION)
df = pd.read_parquet(parquet_path)

# === Load Encoding Config ===
print("⚙️ Loading encoding configuration...")
enc_config = load_config(ENCODING_CONFIG_PATH)
cat_cols = enc_config.get("categorical_columns", [])
num_cols = enc_config.get("numeric_columns", [])
label_col = enc_config["label_column"]

# === Split before encoding ===
print("🔀 Splitting data into train/val/test before encoding...")
# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[label_col])
# Second split: 15% val, 15% test from the 30% temp
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[label_col])

print(f"✅ Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"📊 Split ratios - Train: {len(train_df)/len(df):.1%}, Val: {len(val_df)/len(df):.1%}, Test: {len(test_df)/len(df):.1%}")

# === Standardized Preprocessing ===
def preprocess(df_split):
    """Standardized preprocessing for all datasets - creates safe copy"""
    df_clean = df_split.copy()  # Safe copy to avoid modifying original
    
    # Handle categorical columns  
    if cat_cols:
        df_clean[cat_cols] = df_clean[cat_cols].fillna("missing").astype(str)
    
    # Handle numeric columns
    if num_cols:
        df_clean[num_cols] = df_clean[num_cols].fillna(0).astype(np.float32)
    
    return df_clean

print("🧹 Cleaning data and preparing types...")
train_df = preprocess(train_df)
val_df = preprocess(val_df)
test_df = preprocess(test_df)

# Check what columns are available vs. what's expected
available_cat_cols = [col for col in cat_cols if col in train_df.columns]
available_num_cols = [col for col in num_cols if col in train_df.columns]
missing_cat_cols = [col for col in cat_cols if col not in train_df.columns]
missing_num_cols = [col for col in num_cols if col not in train_df.columns]

print(f"📋 Dataset columns: {list(train_df.columns)}")
print(f"📋 Available categorical columns: {available_cat_cols}")
print(f"📋 Available numeric columns: {available_num_cols}")
if missing_cat_cols:
    print(f"⚠️ Missing categorical columns: {missing_cat_cols}")
if missing_num_cols:
    print(f"⚠️ Missing numeric columns: {missing_num_cols}")

# Only use available columns
all_feature_cols = available_cat_cols + available_num_cols
if not all_feature_cols:
    raise ValueError("No matching feature columns found! Check encoding config vs. actual dataset columns.")

# === ColumnTransformer ===
print("🏗️ Building and fitting transformer on training data only...")
column_transformer = build_column_transformer(
    available_cat_cols, available_num_cols
)

# Extract features and transform in one step
X_train = column_transformer.fit_transform(train_df[all_feature_cols])
X_val = column_transformer.transform(val_df[all_feature_cols])
X_test = column_transformer.transform(test_df[all_feature_cols])
y_train = train_df[label_col].to_numpy()
y_val = val_df[label_col].to_numpy()
y_test = test_df[label_col].to_numpy()

print(f"✅ X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"🔍 Data types - X_train: {type(X_train)}, sparse: {issparse(X_train)}")

# === Label Distribution Analysis ===
print("\n📊 Label distribution analysis:")
train_benign = np.sum(y_train == 0)
train_attack = np.sum(y_train == 1)
val_benign = np.sum(y_val == 0)
val_attack = np.sum(y_val == 1)
test_benign = np.sum(y_test == 0)
test_attack = np.sum(y_test == 1)

print(f"Train - Benign: {train_benign} ({train_benign/len(y_train):.1%}), Attack: {train_attack} ({train_attack/len(y_train):.1%})")
print(f"Val   - Benign: {val_benign} ({val_benign/len(y_val):.1%}), Attack: {val_attack} ({val_attack/len(y_val):.1%})")
print(f"Test  - Benign: {test_benign} ({test_benign/len(y_test):.1%}), Attack: {test_attack} ({test_attack/len(y_test):.1%})")

# === Save artifacts ===
encoded_dir = get_encoded_dir(MODEL, DATASET, VERSION)
ensure_dirs(encoded_dir)

print("💾 Saving encoded artifacts...")
# Save arrays appropriately based on type
if issparse(X_train):
    sparse.save_npz(os.path.join(encoded_dir, "X_train_encoded.npz"), X_train)
    sparse.save_npz(os.path.join(encoded_dir, "X_val_encoded.npz"), X_val)
    sparse.save_npz(os.path.join(encoded_dir, "X_test_encoded.npz"), X_test)
    print("📦 Saved as sparse matrices")
else:
    np.save(os.path.join(encoded_dir, "X_train_encoded.npy"), X_train)
    np.save(os.path.join(encoded_dir, "X_val_encoded.npy"), X_val)
    np.save(os.path.join(encoded_dir, "X_test_encoded.npy"), X_test)
    print("📦 Saved as dense arrays")
np.save(os.path.join(encoded_dir, "y_train.npy"), y_train)
np.save(os.path.join(encoded_dir, "y_val.npy"), y_val)
np.save(os.path.join(encoded_dir, "y_test.npy"), y_test)
joblib.dump(column_transformer, os.path.join(encoded_dir, "column_transformer.joblib"))

# === Save metadata ===
meta = {
    "model": MODEL,
    "dataset": DATASET,
    "version": VERSION,
    "train_shape": X_train.shape,
    "val_shape": X_val.shape,
    "test_shape": X_test.shape,
    "n_features": X_train.shape[1],
    "transformer_type": "OneHot + StandardScaler",
    "splitting": "stratified_70_15_15",
    "train_benign_ratio": float(train_benign / len(y_train)),
    "val_benign_ratio": float(val_benign / len(y_val)),
    "test_benign_ratio": float(test_benign / len(y_test)),
    "saved_at": datetime.now().isoformat()
}

with open(os.path.join(encoded_dir, "encoding_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("📦 Artifacts saved:")
print(f"  Encoded train data: {encoded_dir}/X_train_encoded.npz")
print(f"  Encoded val data:   {encoded_dir}/X_val_encoded.npz")
print(f"  Encoded test data:  {encoded_dir}/X_test_encoded.npz")
print(f"  Labels:             y_train.npy, y_val.npy, y_test.npy")
print(f"  Transformer:        column_transformer.joblib")
print(f"  Metadata:           encoding_metadata.json")
print("✅ Done.")

