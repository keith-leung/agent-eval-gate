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

        # Strands' OpenAIModel takes client_args (passed to the OpenAI client)
        # and model_id as a direct kwarg (Unpack[OpenAIConfig]). The positional
        # (model, client_id, base_url) signature used in older strands no
        # longer exists.
        model = OpenAIModel(
            client_args={
                "api_key": client.api_key,
                "base_url": client.base_url,
            },
            model_id=client.model,
        )
        agent = Agent(model=model, system_prompt="You are a helpful assistant. Answer concisely.")

        start = time.time()
        result = agent(task.prompt)
        latency_ms = (time.time() - start) * 1000.0
        raw_text = result.message if hasattr(result, "message") else str(result)
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"AWS Strands adapter failed: {exc}") from exc
