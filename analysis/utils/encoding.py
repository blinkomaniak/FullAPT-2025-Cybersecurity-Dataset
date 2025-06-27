import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib
import json
import os
from datetime import datetime

def prepare_data(df, categorical_cols, numeric_cols):
    """Legacy function - use standardized_preprocess instead"""
    df[categorical_cols] = df[categorical_cols].fillna("missing")
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df

def standardized_preprocess(df_split, categorical_cols=None, numeric_cols=None, text_cols=None):
    """
    Standardized preprocessing for all encoding scripts
    
    Args:
        df_split: DataFrame to preprocess
        categorical_cols: List of categorical columns for OneHot encoding
        numeric_cols: List of numeric columns for StandardScaler
        text_cols: List of text columns for TF-IDF (bigbase specific)
    
    Returns:
        df_clean: Preprocessed DataFrame copy
    """
    df_clean = df_split.copy()  # Safe copy to avoid modifying original
    
    # Handle text columns (for TF-IDF in bigbase)
    if text_cols:
        for col in text_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("missing").astype(str)
    
    # Handle categorical columns (for OneHot encoding)
    if categorical_cols:
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("missing").astype(str)
    
    # Handle numeric columns
    if numeric_cols:
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0).astype(np.float32)
    
    return df_clean

# Removed duplicate function - using the one below

def save_encoded_artifacts(X_sparse, y_labels, column_transformer, dataset, version):
    output_dir = f"data/encoded/{dataset}-{version}"
    os.makedirs(output_dir, exist_ok=True)

    sparse.save_npz(f"{output_dir}/X_encoded.npz", X_sparse)
    np.save(f"{output_dir}/y_labels.npy", y_labels)
    joblib.dump(column_transformer, f"{output_dir}/column_transformer.pkl")

    meta = {
        "dataset": dataset,
        "version": version,
        "X_shape": X_sparse.shape,
        "n_features": X_sparse.shape[1],
        "n_samples": X_sparse.shape[0],
        "sparsity": True,
        "transformer_type": "OneHot + StandardScaler",
        "saved_at": datetime.now().isoformat()
    }
    meta_path = f"artifacts/metadata/{dataset}-{version}-encoding.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved artifacts to: {output_dir}")
    print(f"📝 Metadata: {meta_path}")

def select_first_column(X):
    """Selector to extract a single column as Series."""
    return X.iloc[:,0]

def make_tfidf_pipeline(max_feats):
    """Builds a pipeline for TF-IDF encoding of a single column."""
    return Pipeline([
	('selector', FunctionTransformer(select_first_column, validate=False)),
	('tfidf', TfidfVectorizer(max_features=max_feats))
    ])

def build_column_transformer(cat_cols, num_cols):
    """Simpler transformer for fully numeric/categorical datasets (e.g., unraveled)."""
    return ColumnTransformer(transformers=[
	('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_cols),
	('scale', StandardScaler(), num_cols)
    ], sparse_threshold=0.1)

def build_bigbase_transformer(tfidf_cols, tfidf_params, onehot_cols, numeric_cols):
    """Full transformer pipeline for bigbase with TF-IDF, OneHot, and Scaler."""
    transformers = []

    for col in tfidf_cols:
        max_feats = tfidf_params.get(col, 300)
        transformers.append((f'tfidf_{col}', make_tfidf_pipeline(max_feats), [col]))

    if onehot_cols:
        transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True), onehot_cols))

    if numeric_cols:
        transformers.append(('numeric', StandardScaler(), numeric_cols))

    return ColumnTransformer(transformers, sparse_threshold=0.1)
