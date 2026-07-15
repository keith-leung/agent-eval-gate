"""CrewAI adapter."""

from __future__ import annotations

import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "crewai"


async def run(client, task: Task) -> SUTOutput:
    try:
        # CrewAI 1.15 ships its own LLM wrapper that accepts an OpenAI-
        # compatible base_url directly — preferable to routing through
        # langchain_openai.ChatOpenAI, which crewai no longer requires.
        from crewai import Agent, Task as CrewTask, Crew, LLM

        llm = LLM(
            model=client.model,
            base_url=client.base_url,
            api_key=client.api_key,
        )

        agent = Agent(
            role="Evaluator",
            goal="Answer the task accurately and concisely.",
            backstory="You are a precise assistant.",
            llm=llm,
            verbose=False,
        )

        crew_task = CrewTask(
            description=task.prompt,
            agent=agent,
            expected_output="A concise, accurate answer.",
        )

        crew = Crew(agents=[agent], tasks=[crew_task], verbose=False)
        start = time.time()
        # The adapter runs inside an async event loop; crew.kickoff() is
        # synchronous and would deadlock — use kickoff_async().
        result = await crew.kickoff_async()
        latency_ms = (time.time() - start) * 1000.0
        raw_text = str(result.raw) if hasattr(result, "raw") else str(result)
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"CrewAI adapter failed: {exc}") from exc
