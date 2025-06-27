import json
import os

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Config file not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)

