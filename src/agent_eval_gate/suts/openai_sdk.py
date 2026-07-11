"""OpenAI Agents SDK adapter."""

from __future__ import annotations

import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "openai_sdk"


async def run(client, task: Task) -> SUTOutput:
    try:
        from agents import Agent, Runner
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(base_url=client.base_url, api_key=client.api_key)
        model = OpenAIChatCompletionsModel(openai_client=openai_client, model=client.model)

        agent = Agent(name="eval-agent", instructions="You are a helpful assistant. Answer concisely.", model=model)

        start = time.time()
        result = await Runner.run(agent, task.prompt)
        latency_ms = (time.time() - start) * 1000.0
        raw_text = result.final_output or ""
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"OpenAI Agents SDK adapter failed: {exc}") from exc
