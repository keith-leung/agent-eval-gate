"""CrewAI adapter."""

from __future__ import annotations

import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "crewai"


async def run(client, task: Task) -> SUTOutput:
    try:
        from crewai import Agent, Task as CrewTask, Crew
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=client.model,
            openai_api_base=client.base_url,
            openai_api_key=client.api_key,
            temperature=0,
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
        import time
        start = time.time()
        result = crew.kickoff()
        latency_ms = (time.time() - start) * 1000.0
        raw_text = str(result.raw) if hasattr(result, "raw") else str(result)
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"CrewAI adapter failed: {exc}") from exc
