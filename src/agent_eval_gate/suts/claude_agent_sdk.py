"""Anthropic Claude Agent SDK adapter."""

from __future__ import annotations

import time
from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "claude_agent_sdk"


async def run(client, task: Task) -> SUTOutput:
    try:
        from anthropic import Anthropic

        # The Claude Agent SDK wraps Claude Code CLI and is genuinely locked
        # to Claude models. We use the Anthropic Messages API directly as the
        # honest substrate for this SUT slot.
        #
        # The Anthropic SDK appends "/v1/messages" to base_url itself; the
        # shared config's base_url already ends in "/v1", which would produce
        # "/v1/v1/messages". Strip a trailing "/v1" to avoid the double path.
        base = client.base_url
        if base.endswith("/v1"):
            base = base[:-3]

        anthropic_client = Anthropic(base_url=base, api_key=client.api_key)
        import time
        start = time.time()
        response = anthropic_client.messages.create(
            model=client.model,
            max_tokens=4096,
            system="You are a helpful assistant. Answer concisely.",
            messages=[{"role": "user", "content": task.prompt}],
        )
        latency_ms = (time.time() - start) * 1000.0
        # The gateway may return ThinkingBlock entries before the TextBlock;
        # extract the first block that actually carries text.
        raw_text = ""
        for block in response.content:
            if getattr(block, "text", None):
                raw_text = block.text
                break
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"Claude Agent SDK adapter failed: {exc}") from exc
