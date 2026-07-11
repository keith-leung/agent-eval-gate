"""ReAct loop baseline (plain while-loop, NOT a peer SUT)."""

from __future__ import annotations

from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "react_loop"


async def run(client, task: Task) -> SUTOutput:
    """Minimal ReAct loop using the shared LLM client."""
    import time
    start = time.time()

    system_prompt = (
        "You are a ReAct-style reasoning agent. For each step, think about what to do next, "
        "then act. If you have enough information, provide the final answer. "
        "Format each thought as 'Thought: ...' and each action as 'Action: ...'. "
        "When done, output 'Final Answer: ...'."
    )
    prompt = task.prompt
    max_iterations = 5

    full_prompt = prompt
    for _ in range(max_iterations):
        response = await client.complete(system=system_prompt, user_message=full_prompt)
        full_prompt += f"\n\n{response}"
        if "Final Answer:" in response:
            break

    latency_ms = (time.time() - start) * 1000.0
    raw_text = full_prompt
    return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
