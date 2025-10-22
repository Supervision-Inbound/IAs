# src/inferencia/utils_io.py
import json
import os
import pandas as pd # Although not used directly, keep for potential future use

def write_json(path: str, data):
    """Generic function to write data to a JSON file."""
    # Ensure the directory exists before writing
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            # Use ensure_ascii=False for proper UTF-8 encoding (e.g., accents)
            # Use indent=2 for readability
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"INFO: Successfully wrote JSON to {path}")
    except Exception as e:
        print(f"ERROR: Failed to write JSON to {path}. Error: {e}")

# The original write_hourly_json and write_daily_json functions
# have been removed as their modified versions are now within
# inferencia_core.py for the separate output strategy.

