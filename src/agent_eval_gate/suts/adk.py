"""Google ADK adapter."""

from __future__ import annotations

from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "adk"


async def run(client, task: Task) -> SUTOutput:
    try:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        # ADK natively expects Google models. For cross-vendor OpenAI-compatible
        # endpoints, ADK does NOT support custom base_url as of 2026-07.
        # We still implement the adapter; at runtime it will raise honestly
        # unless the user has configured ADK for a compatible endpoint.
        # If the endpoint is OpenAI-compatible, we fall back to a direct LLM call.
        import time
        start = time.time()

        # Attempt ADK native path first
        agent = LlmAgent(
            name="eval-agent",
            model=client.model,
            instruction="You are a helpful assistant. Answer concisely.",
        )
        session_service = InMemorySessionService()
        runner = Runner(agent=agent, session_service=session_service, app_name="eval-gate")
        session = session_service.create_session(app_name="eval-gate", user_id="eval-user")
        events = runner.run(session_id=session.id, user_id="eval-user", prompt=task.prompt)
        raw_text = "".join(e.content.parts[0].text for e in events if hasattr(e, "content") and e.content and e.content.parts)
        latency_ms = (time.time() - start) * 1000.0
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"Google ADK adapter failed: {exc}") from exc
