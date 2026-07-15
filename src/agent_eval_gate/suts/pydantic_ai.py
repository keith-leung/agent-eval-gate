"""PydanticAI adapter."""

from __future__ import annotations

import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "pydantic_ai"


async def run(client, task: Task) -> SUTOutput:
    try:
        from openai import AsyncOpenAI
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        # pydantic-ai 2.6 renamed OpenAIModel -> OpenAIChatModel and moved
        # base_url/api_key into an OpenAIProvider wrapping an AsyncOpenAI client.
        openai_client = AsyncOpenAI(base_url=client.base_url, api_key=client.api_key)
        provider = OpenAIProvider(openai_client=openai_client)
        model = OpenAIChatModel(client.model, provider=provider)
        agent = Agent(model=model, system_prompt="You are a helpful assistant. Answer concisely.")

        start = time.time()
        result = await agent.run(task.prompt)
        latency_ms = (time.time() - start) * 1000.0
        raw_text = result.output if hasattr(result, "output") else str(result)
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"PydanticAI adapter failed: {exc}") from exc
