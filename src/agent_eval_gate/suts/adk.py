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

        # ADK 2.4 locks to Google Gemini models — its model layer
        # (gemini_llm_connection / google_llm) has no OpenAI-compatible path.
        # SPEC §6 anticipated this: "verify at impl time whether ADK accepts a
        # custom OpenAI-compatible endpoint or pins Google's; document whichever."
        # It pins Google's. If the configured model is not a Gemini model,
        # this SUT cannot run on the shared OpenAI-compatible SUT gateway and
        # must be honestly marked unavailable rather than faked.
        model_lower = (client.model or "").lower()
        if "gemini" not in model_lower:
            raise RuntimeError(
                f"Google ADK locks to Gemini models and has no OpenAI-compatible "
                f"endpoint path (ADK 2.4 model layer is Gemini-only). Configured "
                f"SUT model '{client.model}' is not a Gemini model. To run this "
                f"SUT, configure a Google API key + a gemini-* model for the "
                f"'adk' provider in config.yaml. Marking unavailable."
            )

        import time
        start = time.time()
        agent = LlmAgent(
            name="eval_agent",
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
