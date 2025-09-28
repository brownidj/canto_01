import json
import os

CONFIG_FILE = "config.json"

def load_config():
    """Load config dict from file, or return empty dict if missing/invalid."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(config: dict):
    """Save config dict to file (overwrites)."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Error saving config:", e)