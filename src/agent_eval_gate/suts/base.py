"""Base SUT adapter utilities."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.llm_client import LLMClient


def make_sut_output(task: Task, framework: str, model: str, raw_text: str, **kwargs) -> SUTOutput:
    # Defensive: some SUT adapters may pass a non-str (e.g. a parsed dict or
    # an framework result object) as raw_text. Coerce to str at the source so
    # every downstream consumer (judge, pairwise, errored check) sees a str.
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

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
