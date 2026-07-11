"""LangChain adapter."""

from __future__ import annotations

import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "langchain"


async def run(client, task: Task) -> SUTOutput:
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.tools import tool

        llm = ChatOpenAI(
            model=client.model,
            openai_api_base=client.base_url,
            openai_api_key=client.api_key,
            temperature=0,
        )

        @tool
        def dummy_tool(query: str) -> str:
            """A dummy tool for evaluation."""
            return f"Dummy result for: {query}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer concisely."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_react_agent(llm=llm, tools=[dummy_tool], prompt=prompt)
        executor = AgentExecutor(agent=agent, tools=[dummy_tool], verbose=False, handle_parsing_errors=True)

        start = time.time()
        result = executor.invoke({"input": task.prompt})
        latency_ms = (time.time() - start) * 1000.0
        raw_text = result.get("output", "")
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"LangChain adapter failed: {exc}") from exc
