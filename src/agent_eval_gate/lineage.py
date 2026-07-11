"""SHA-256 lineage tracking for eval runs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any


def compute_bytes_hash(data: bytes, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def record_run_ledger(ledger_path: str, run_entry: dict[str, Any]) -> dict[str, Any]:
    """Append a run entry to the append-only ledger."""
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    ledger = []
    if os.path.exists(ledger_path):
        with open(ledger_path, "r", encoding="utf-8") as f:
            try:
                ledger = json.load(f)
            except json.JSONDecodeError:
                ledger = []
    ledger.append(run_entry)
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    return run_entry


def build_run_meta(
    tasks_bytes: bytes,
    sut_pins: dict[str, str],
    judge_pin: str,
    framework_versions: dict[str, str],
) -> dict[str, Any]:
    tasks_sha256 = compute_bytes_hash(tasks_bytes)
    return {
        "tasks_sha256": tasks_sha256,
        "judge_model": judge_pin,
        "sut_models": sut_pins,
        "framework_versions": framework_versions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
