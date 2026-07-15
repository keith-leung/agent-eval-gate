"""Anthropic API adapter.

Note: This uses the raw Anthropic SDK, not the Claude Agent SDK framework.
The Claude Agent SDK wraps Claude Code CLI and is not installable as a
Python package in this environment. We therefore evaluate Claude models
through the Anthropic Messages API directly, which is the honest substrate
for this SUT slot.
"""

from __future__ import annotations

import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "anthropic"


async def run(client, task: Task) -> SUTOutput:
    try:
        from anthropic import Anthropic

        anthropic_client = Anthropic(base_url=client.base_url, api_key=client.api_key)
        start = time.time()
        response = anthropic_client.messages.create(
            model=client.model,
            max_tokens=4096,
            system="You are a helpful assistant. Answer concisely.",
            messages=[{"role": "user", "content": task.prompt}],
        )
        latency_ms = (time.time() - start) * 1000.0
        raw_text = response.content[0].text if response.content else ""
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"Anthropic adapter failed: {exc}") from exc
