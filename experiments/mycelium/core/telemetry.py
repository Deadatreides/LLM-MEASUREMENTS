"""
core/telemetry.py - Structured JSONL telemetry logging for Mycelium Swarm steps.
"""
import os
import json
import logging

logger = logging.getLogger("main")

def log_telemetry(step_data: dict):
    """
    Appends a single step telemetry record as a JSON-line to logs/telemetry.jsonl.
    Catches all exceptions to ensure that telemetry failures never crash the main application.
    """
    try:
        os.makedirs("logs", exist_ok=True)
        telemetry_file = os.path.join("logs", "telemetry.jsonl")
        
        # Ensure all data types are simple JSON-serializable types
        clean_data = {}
        for k, v in step_data.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                clean_data[k] = v
            elif isinstance(v, dict):
                clean_data[k] = {str(subkey): (subval if isinstance(subval, (int, float, str, bool)) or subval is None else str(subval)) for subkey, subval in v.items()}
            elif isinstance(v, list):
                clean_data[k] = [str(item) for item in v]
            else:
                clean_data[k] = str(v)

        # Write to JSONL file with UTF-8 encoding
        line = json.dumps(clean_data, ensure_ascii=False)
        with open(telemetry_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.warning(f"Telemetry logging failed: {e}")
