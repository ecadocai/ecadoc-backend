"""
Utility functions for the Floor Plan Agent API
"""
import os
import json
import time
import threading
from typing import Dict, Any
import time as _time
from modules.config.settings import settings

def delete_file(path: str):
    """Delete file with error handling"""
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"DEBUG: Deleted file: {path}")
    except Exception as e:
        print(f"Error deleting file {path}: {e}")

def delete_file_after_delay(path: str, delay_seconds: int):
    """Delete file after specified delay in seconds"""
    def delayed_delete():
        time.sleep(delay_seconds)
        delete_file(path)
    
    thread = threading.Thread(target=delayed_delete)
    thread.daemon = True
    thread.start()

def load_registry() -> Dict[str, Any]:
    """Load document registry from JSON file"""
    registry_path = os.path.join(settings.DATA_DIR, "registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_registry(reg: Dict[str, Any]):
    """Save document registry to JSON file"""
    registry_path = os.path.join(settings.DATA_DIR, "registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)

def validate_file_path(file_path: str, allowed_base_dir: str = None) -> bool:
    """Validate file path for security"""
    if allowed_base_dir is None:
        allowed_base_dir = settings.DATA_DIR
    
    abs_path = os.path.abspath(file_path)
    abs_base_dir = os.path.abspath(allowed_base_dir)
    
    # For debugging in production, only log if file doesn't exist or validation fails
    file_exists = os.path.exists(abs_path)
    path_within_base = abs_path.startswith(abs_base_dir)
    is_valid = path_within_base and file_exists
    
    if not is_valid:
        print(f"DEBUG: File validation failed:")
        print(f"  File path: {file_path}")
        print(f"  Absolute file path: {abs_path}")
        print(f"  Allowed base dir: {allowed_base_dir}")
        print(f"  Absolute base dir: {abs_base_dir}")
        print(f"  File exists: {file_exists}")
        print(f"  Path starts with base: {path_within_base}")
    
    return is_valid

def generate_unique_filename(base_name: str, extension: str, directory: str) -> str:
    """Generate a unique filename in the specified directory"""
    import uuid
    unique_id = uuid.uuid4().hex[:8]
    return os.path.join(directory, f"{base_name}_{unique_id}.{extension}")

def log_metric(event: str, **fields: Any) -> None:
    """Lightweight, structured metric logger.

    Prints a single-line record that is easy to grep in logs. Timestamps are
    unix milliseconds for quick charting. Never raises.
    """
    try:
        ts_ms = int(_time.time() * 1000)
        parts = [f"event={event}", f"ts={ts_ms}"]
        for k, v in fields.items():
            # Avoid spaces/newlines
            if isinstance(v, str):
                v = v.replace("\n", " ").replace(" ", "_")
            parts.append(f"{k}={v}")
        print("METRIC " + " ".join(parts))
    except Exception:
        pass
