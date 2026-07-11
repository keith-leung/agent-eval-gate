"""Unified LLM client supporting OpenAI-compatible and Anthropic APIs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic, Anthropic

from agent_eval_gate.protocols import Task


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
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.api_type = api_type
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if api_type == "anthropic":
            self._anthropic = AsyncAnthropic(base_url=base_url, api_key=api_key)
            self._openai = None
        else:
            self._openai = AsyncOpenAI(base_url=base_url, api_key=api_key)
            self._anthropic = None

    async def complete(
        self,
        system: str = "",
        user_message: str = "",
        messages: Optional[list[dict]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM and return the raw text response."""
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

        response = await self._openai.chat.completions.create(**call_kwargs)
        content = response.choices[0].message.content or ""
        return content

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
        # If no messages, at least send something
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

        response = await self._anthropic.messages.create(**call_kwargs)
        content = response.content[0].text if response.content else ""
        return content

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
