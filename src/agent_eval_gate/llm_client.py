"""Unified LLM client supporting OpenAI-compatible and Anthropic APIs.

Includes retry with exponential backoff for transient errors (429 rate
limit, 5xx) so a dense eval run (8 SUTs × 16 tasks + pairwise) does not
crash on the gateway's rate limiter.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAI, APIStatusError
from anthropic import AsyncAnthropic, Anthropic

from agent_eval_gate.protocols import Task


_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0  # seconds


def _is_retryable(exc: Exception) -> bool:
    """True for 429 / 5xx (transient gateway errors worth retrying)."""
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


class LLMClient:
    """Thin wrapper around OpenAI-compatible + Anthropic clients."""

    def __init__(
        self,
        provider: str = "gpt_agent",
        base_url: str = "https://gpt-agent.cc/v1",
        api_key: str = "",
        api_type: str = "openai",
        model: str = "step-3.7-flash",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        mode: str = "real",
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.api_type = api_type
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = mode

        if api_type == "anthropic":
            self._anthropic = AsyncAnthropic(base_url=base_url, api_key=api_key)
            self._sync_anthropic = Anthropic(base_url=base_url, api_key=api_key)
            self._openai = None
            self._sync_openai = None
        else:
            self._openai = AsyncOpenAI(base_url=base_url, api_key=api_key)
            self._sync_openai = OpenAI(base_url=base_url, api_key=api_key)
            self._anthropic = None
            self._sync_anthropic = None

    def _mock_response(
        self, system: str, user_message: str, messages: Optional[list[dict]]
    ) -> str:
        """Deterministic, input-derived, offline response for mock/CI mode.

        Emits a JSON object carrying the fields every judge/pairwise parser in
        this repo reads (``winner``/``score``/``confidence``/``faithfulness``),
        derived from a hash of the inputs so outputs vary per call (non-degenerate
        ranking) yet are fully deterministic and require no network.
        """
        import hashlib
        import json as _json

        basis = (system or "") + (user_message or "") + _json.dumps(messages or [], ensure_ascii=False)
        h = int(hashlib.sha256(basis.encode("utf-8")).hexdigest(), 16)
        score = round((h % 100) / 100.0, 2)
        winner = "A" if (h % 2 == 0) else "B"
        return _json.dumps({
            "winner": winner,
            "score": score,
            "confidence": 0.5,
            "faithfulness": score,
            "reasoning": "deterministic mock (offline CI mode)",
        }, ensure_ascii=False)

    async def complete(
        self,
        system: str = "",
        user_message: str = "",
        messages: Optional[list[dict]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tier: Optional[str] = None,  # accepted for API compat, unused (the client is already tier-scoped)
        **kwargs: Any,
    ) -> str:
        """Call the LLM and return the raw text response."""
        if self.mode == "mock":
            return self._mock_response(system, user_message, messages)
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.api_type == "anthropic":
            return await self._anthropic_complete(
                system=system, user_message=user_message, messages=messages,
                model=model, temperature=temperature, max_tokens=max_tokens,
                tools=tools, **kwargs,
            )
        return await self._openai_complete(
            system=system, user_message=user_message, messages=messages,
            model=model, temperature=temperature, max_tokens=max_tokens,
            tools=tools, **kwargs,
        )

    async def _openai_complete(
        self,
        system: str,
        user_message: str,
        messages: Optional[list[dict]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[list[dict]],
        **kwargs: Any,
    ) -> str:
        msgs = messages or []
        if not msgs and system:
            msgs.append({"role": "system", "content": system})
        if not msgs:
            msgs.append({"role": "user", "content": user_message})
        elif user_message:
            msgs.append({"role": "user", "content": user_message})

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            call_kwargs["tools"] = tools
        call_kwargs.update(kwargs)

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._openai.chat.completions.create(**call_kwargs)
                content = response.choices[0].message.content or ""
                return content
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc) or attempt == _MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
        raise last_exc  # unreachable

    async def _anthropic_complete(
        self,
        system: str,
        user_message: str,
        messages: Optional[list[dict]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[list[dict]],
        **kwargs: Any,
    ) -> str:
        msgs = messages or []
        if user_message:
            msgs.append({"role": "user", "content": user_message})
        if not msgs:
            msgs = [{"role": "user", "content": user_message or "Hello"}]

        call_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": msgs,
        }
        if system:
            call_kwargs["system"] = system
        if tools:
            call_kwargs["tools"] = tools
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        call_kwargs.update(kwargs)

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._anthropic.messages.create(**call_kwargs)
                # Extract first text block (skip ThinkingBlock).
                for block in response.content:
                    if getattr(block, "text", None):
                        return block.text
                return ""
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc) or attempt == _MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
        raise last_exc  # unreachable

    def complete_sync(
        self,
        system: str = "",
        user_message: str = "",
        messages: Optional[list[dict]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tier: Optional[str] = None,  # accepted for API compat, unused
        **kwargs: Any,
    ) -> str:
        """Synchronous completion (for judge paths that are not async).

        Runs the async ``complete`` in an event loop, with the same retry
        + backoff as the async path.
        """
        if self.mode == "mock":
            return self._mock_response(system, user_message, messages)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We are inside a running loop (e.g. the async gate). Create a
                # fresh loop in a thread to avoid "loop already running".
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        lambda: asyncio.run(self.complete(
                            system=system, user_message=user_message, messages=messages,
                            model=model, temperature=temperature, max_tokens=max_tokens,
                            **kwargs,
                        ))
                    ).result()
        except RuntimeError:
            pass
        return asyncio.run(self.complete(
            system=system, user_message=user_message, messages=messages,
            model=model, temperature=temperature, max_tokens=max_tokens,
            **kwargs,
        ))

    def compute_provenance_hash(self, task: Task, output: "SUTOutput") -> str:
        """Compute SHA-256 lineage hash for a task-output pair."""
        payload = {
            "task_id": task.id,
            "prompt": task.prompt,
            "expected": str(task.expected),
            "framework": output.framework,
            "model": output.model,
            "raw_text": output.raw_text,
            "latency_ms": output.latency_ms,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
