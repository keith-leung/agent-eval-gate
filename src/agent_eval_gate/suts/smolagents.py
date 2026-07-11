"""Smolagents adapter."""

from __future__ import annotations

import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "smolagents"


async def run(client, task: Task) -> SUTOutput:
    try:
        from smolagents import CodeAgent, HfApiModel, LiteLLMModel

        # Prefer LiteLLM if the base_url is OpenAI-compatible
        model = LiteLLMModel(
            model_id=f"openai/{client.model}",
            api_base=client.base_url,
            api_key=client.api_key,
        )
        agent = CodeAgent(tools=[], model=model)

        import time
        start = time.time()
        result = agent.run(task.prompt)
        latency_ms = (time.time() - start) * 1000.0
        raw_text = str(result)
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"Smolagents adapter failed: {exc}") from exc
