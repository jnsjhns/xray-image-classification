from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def print_section(title: str, width: int = 78) -> None:
    """Print a visually highlighted section title in the terminal."""
    line = "=" * width
    print(f"\n{line}\n{title}\n{line}")


def print_kv(label: str, value: Any, label_width: int = 32) -> None:
    """Print a neatly formatted key-value pair."""
    print(f"{label:<{label_width}}: {value}")


def ensure_dir(path: Path) -> Path:
    """Ensure that a directory exists and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert NumPy / TensorFlow-like objects to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if hasattr(obj, "numpy"):
        tensor_value = obj.numpy()
        if isinstance(tensor_value, np.ndarray):
            return tensor_value.tolist()
        if isinstance(tensor_value, np.generic):
            return tensor_value.item()
        return tensor_value

    return obj


def save_json(path: Path, payload: dict[str, Any], indent: int = 4) -> None:
    """Save a dictionary as a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_payload = _to_json_serializable(payload)

    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable_payload, f, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return it as a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_histories(h1: Any, h2: Any | None = None) -> dict[str, list[float]]:
    """Merge Stage 1 and Stage 2 training histories into a single dictionary."""

    def _to_history_dict(history_obj: Any) -> dict[str, list[float]]:
        if hasattr(history_obj, "history"):
            history_obj = history_obj.history

        if not isinstance(history_obj, dict):
            raise TypeError("History must be a Keras History object or a dict.")

        return {
            k: [float(v_i) for v_i in v]
            for k, v in history_obj.items()
        }

    merged = _to_history_dict(h1)

    if h2 is not None:
        h2_dict = _to_history_dict(h2)
        for k, v in h2_dict.items():
            if k in merged:
                merged[k].extend(v)
            else:
                merged[k] = list(v)

    return merged