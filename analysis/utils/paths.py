"""
utils/paths.py
Centralized path management for dataset, model, version-based pipelines
"""
import os

def get_processed_path(model: str, dataset: str, version: str) -> str:
    return os.path.join("data", "processed", f"{dataset}-{model}-{version}.parquet")

def get_encoded_dir(model: str, dataset: str, version: str) -> str:
    return os.path.join("data", "encoded", f"{dataset}-{model}-{version}")

def get_model_dir(model: str, dataset: str, version: str) -> str:
    return os.path.join("models", model, f"{dataset}-{version}")

def get_eval_dir(model: str, dataset: str, version: str) -> str:
    return os.path.join("artifacts", "eval", model, f"{dataset}-{version}")

def get_plot_dir(model: str, dataset: str, version: str) -> str:
    return os.path.join("artifacts", "plots", model, f"{dataset}-{version}")

def get_metadata_path(model: str, dataset: str, version: str) -> str:
    return os.path.join("artifacts", "metadata", f"{dataset}-{model}-{version}.json")

def get_label_dist_path(model: str, dataset: str, version: str) -> str:
    return os.path.join("artifacts", "metadata", f"label_distribution-{dataset}-{model}-{version}.csv")

def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)

