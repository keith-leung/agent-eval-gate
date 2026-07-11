"""Base SUT adapter utilities."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.llm_client import LLMClient


def make_sut_output(task: Task, framework: str, model: str, raw_text: str, **kwargs) -> SUTOutput:
    from agent_eval_gate.llm_client import LLMClient

    # We need a temporary client to compute provenance hash, or just use raw fields.
    # For simplicity, compute a lightweight hash here without needing the full client.
    import hashlib, json
    payload = {
        "task_id": task.id,
        "prompt": task.prompt,
        "expected": str(task.expected),
        "framework": framework,
        "model": model,
        "raw_text": raw_text,
        "latency_ms": kwargs.get("latency_ms", 0.0),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    provenance_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return SUTOutput(
        task_id=task.id,
        framework=framework,
        model=model,
        raw_text=raw_text,
        provenance_hash=provenance_hash,
        **kwargs,
    )
