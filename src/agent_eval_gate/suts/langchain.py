"""LangChain adapter."""

from __future__ import annotations

import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "langchain"


async def run(client, task: Task) -> SUTOutput:
    try:
        # langchain 1.3 moved create_react_agent out of langchain.agents.
        # The canonical home is now langgraph.prebuilt (the graph-based agent).
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        llm = ChatOpenAI(
            model=client.model,
            openai_api_base=client.base_url,
            openai_api_key=client.api_key,
            temperature=0,
        )

        agent = create_react_agent(
            model=llm,
            tools=[],
            prompt="You are a helpful assistant. Answer concisely.",
        )

        start = time.time()
        result = agent.invoke({"messages": [{"role": "user", "content": task.prompt}]})
        latency_ms = (time.time() - start) * 1000.0
        messages = result.get("messages", [])
        raw_text = ""
        if messages:
            last = messages[-1]
            raw_text = getattr(last, "content", str(last))
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"LangChain adapter failed: {exc}") from exc
