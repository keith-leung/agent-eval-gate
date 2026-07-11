"""AWS Strands adapter."""

from __future__ import annotations

import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "strands"


async def run(client, task: Task) -> SUTOutput:
    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel

        model = OpenAIModel(client.model, client_id=client.api_key, base_url=client.base_url)
        agent = Agent(model=model, system_prompt="You are a helpful assistant. Answer concisely.")

        start = time.time()
        result = agent(task.prompt)
        latency_ms = (time.time() - start) * 1000.0
        raw_text = result.message if hasattr(result, "message") else str(result)
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"AWS Strands adapter failed: {exc}") from exc
