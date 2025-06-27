"""
Stage 01: Data Aggregation & Label Mapping
Target Dataset: bigbase or unraveled
Output: processed parquet + label distribution + metadata
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import gc
import argparse
from utils.paths import (
    get_processed_path,
    get_label_dist_path,
    get_metadata_path,
    ensure_dirs
)
from utils.config import load_config

# === Parameters ===
parser = argparse.ArgumentParser(description="Aggregate raw CSVs and create labeled dataset.")
parser.add_argument("--model", type=str, required=True, help="Model (e.g., lstm-sae, sae, if, 1svm")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. bigbase, unraveled)")
parser.add_argument("--version", type=str, required=True, help="Version identifier (e.g. v1)")
parser.add_argument("--subdir", type=str, default="", help="Optional subdirectory inside dataset folder (e.g. 'network-flows')")
parser.add_argument("--schema", type=str, required=True, help="Path to schema JSON file")
args = parser.parse_args()

MODEL = args.model
DATASET = args.dataset
VERSION = args.version
RAW_DATASET_DIR = os.path.join("datasets", DATASET, args.subdir) if args.subdir else os.path.join("datasets", DATASET)

# === Feature schema ===
schema = load_config(args.schema)
core_features = schema["core_features"]
label_column = schema["label_column"]
time_column = schema["time_column"]
columns_to_keep = core_features + [label_column, time_column]  # Include time column
benign_label = schema["benign_label"]

# Add metadata columns for bigbase dataset
if DATASET == "bigbase":
    columns_to_keep.extend(['APT_Campaign', 'Session_File'])

# === APT Campaign Mapping for Bigbase Dataset ===
def get_apt_campaign(filename):
    """Map bigbase dataset files to APT campaigns"""
    if 'dataset-' not in filename:
        return "UNKNOWN"
    
    try:
        file_num = int(filename.split('dataset-')[1].split('.')[0])
        if 1 <= file_num <= 20: return "APT1"
        elif 21 <= file_num <= 30: return "APT2" 
        elif 31 <= file_num <= 38: return "APT3"
        elif 39 <= file_num <= 44: return "APT4"
        elif 45 <= file_num <= 47: return "APT5"
        elif 48 <= file_num <= 50: return "APT6"
        else: return "UNKNOWN"
    except:
        return "UNKNOWN"

# === Load all CSVs ===
all_csvs = sorted(glob(os.path.join(RAW_DATASET_DIR, '**', '*.csv'), recursive=True))
print(f"🔍 Found {len(all_csvs)} CSV files in {RAW_DATASET_DIR}")

dfs = []
for path in tqdm(all_csvs):
    try:
        df = pd.read_csv(path, usecols=lambda col: col in columns_to_keep, low_memory=False)

        # Fill missing columns if any
        for col in columns_to_keep:
            if col not in df.columns:
                df[col] = pd.NA

        # Label mapping: 0 = Benign, 1 = Attack
        df['Label'] = df[label_column].apply(lambda x: 0 if str(x).strip().lower() == benign_label else 1)

        # Add APT campaign metadata for bigbase dataset
        if DATASET == "bigbase":
            filename = os.path.basename(path)
            apt_campaign = get_apt_campaign(filename)
            df['APT_Campaign'] = apt_campaign
            df['Session_File'] = filename
            print(f"📁 {filename} → {apt_campaign} (events: {len(df)}, attacks: {df['Label'].sum()})")

        dfs.append(df)
        del df
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")

# === Combine all ===
df_all = pd.concat(dfs, ignore_index=True)
del dfs; gc.collect()
print(f"✅ Combined dataframe shape: {df_all.shape}")

# === DateTime Conversion for Bigbase Dataset ===
if time_column == "UtcTime" and time_column in df_all.columns:
    print(f"🕐 Converting {time_column} to numeric timestamp...")
    try:
        # Convert string datetime to Unix timestamp in milliseconds and OVERWRITE original column
        original_min = df_all[time_column].min()
        original_max = df_all[time_column].max()
        df_all[time_column] = pd.to_datetime(df_all[time_column]).astype('int64') // 10**6
        print(f"✅ Converted {time_column} from datetime to numeric:")
        print(f"   Original range: {original_min} to {original_max}")
        print(f"   Numeric range:  {df_all[time_column].min()} to {df_all[time_column].max()}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to convert {time_column} to numeric: {e}")
elif time_column in df_all.columns:
    print(f"✅ Time column {time_column} already in numeric format")

# === Save parquet ===
parquet_path = get_processed_path(MODEL, DATASET, VERSION)
ensure_dirs(os.path.dirname(parquet_path))
df_all.to_parquet(parquet_path, index=False)
print(f"💾 Saved processed dataset to {parquet_path}")

# === Save label distribution ===
label_dist_path = get_label_dist_path(MODEL, DATASET, VERSION)
df_all['Label'].value_counts().to_csv(label_dist_path)

# === Save metadata ===
metadata = {
    "model": MODEL,
    "dataset": DATASET,
    "version": VERSION,
    "total_rows": len(df_all),
    "total_columns": df_all.shape[1],
    "core_features_count": len(core_features),
    "core_features": core_features,
    "label_column": label_column,
    "time_column": time_column,
    "datetime_conversion": {
        "time_column": time_column,
        "conversion_applied": time_column == "UtcTime",
        "conversion_type": "string_datetime_to_unix_epoch_ms" if time_column == "UtcTime" else "no_conversion"
    },
    "memory_usage_mb": round(df_all.memory_usage(deep=True).sum() / (1024**2), 2)
}

# Add APT campaign statistics for bigbase
if DATASET == "bigbase" and 'APT_Campaign' in df_all.columns:
    apt_stats = {}
    for apt in df_all['APT_Campaign'].unique():
        apt_df = df_all[df_all['APT_Campaign'] == apt]
        apt_stats[apt] = {
            "sessions": apt_df['Session_File'].nunique(),
            "total_events": len(apt_df),
            "benign_events": int((apt_df['Label'] == 0).sum()),
            "attack_events": int((apt_df['Label'] == 1).sum()),
            "attack_ratio": float((apt_df['Label'] == 1).mean())
        }
    
    metadata["apt_campaigns"] = apt_stats
    metadata["total_apt_campaigns"] = len(apt_stats)
    print(f"\n📊 APT Campaign Statistics:")
    for apt, stats in apt_stats.items():
        print(f"   {apt}: {stats['sessions']} sessions, {stats['total_events']} events, {stats['attack_ratio']:.1%} attacks")

metadata_path = get_metadata_path(MODEL, DATASET, VERSION)
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"📝 Metadata saved to {metadata_path}")
print(f"📊 Label distribution saved to {label_dist_path}")

# === Clean up ===
del df_all
gc.collect()
print("✅ Stage 01 completed.")
