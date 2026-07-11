"""Configuration loader — mode driven by --config flag, never env var."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    api_type: str = "openai"
    tiers: dict[str, dict[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tiers is None:
            self.tiers = {}


@dataclass
class JudgeConfig:
    provider: str
    tier: str


@dataclass
class SUTDefaultConfig:
    provider: str
    tier: str


@dataclass
class TracingConfig:
    langsmith_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    phoenix_endpoint: str = ""
    braintrust_api_key: str = ""


@dataclass
class EvalConfig:
    mode: str  # real | mock
    default_provider: str
    providers: dict[str, ProviderConfig]
    judge: JudgeConfig
    sut_default: SUTDefaultConfig
    tracing: TracingConfig

    @classmethod
    def from_file(cls, path: str | Path) -> "EvalConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)

        providers = {}
        for name, pdata in data.get("providers", {}).items():
            providers[name] = ProviderConfig(
                name=name,
                base_url=pdata.get("base_url", ""),
                api_key=pdata.get("api_key", ""),
                api_type=pdata.get("api_type", "openai"),
                tiers=pdata.get("tiers", {}),
            )

        tracing = TracingConfig(
            langsmith_api_key=data.get("langsmith_api_key", ""),
            langfuse_public_key=data.get("langfuse_public_key", ""),
            langfuse_secret_key=data.get("langfuse_secret_key", ""),
            phoenix_endpoint=data.get("phoenix_endpoint", ""),
            braintrust_api_key=data.get("braintrust_api_key", ""),
        )

        return cls(
            mode=data.get("mode", "real"),
            default_provider=data.get("default_provider", "gpt_agent"),
            providers=providers,
            judge=JudgeConfig(**data.get("judge", {})),
            sut_default=SUTDefaultConfig(**data.get("sut_default", {})),
            tracing=tracing,
        )

    def get_provider(self, name: str) -> ProviderConfig:
        if name not in self.providers:
            raise KeyError(f"Provider '{name}' not found in config. Available: {list(self.providers.keys())}")
        return self.providers[name]

    def get_sut_client(self, framework: str, model_override: Optional[str] = None) -> "LLMClient":
        from agent_eval_gate.llm_client import LLMClient

        provider_name = self.sut_default.provider
        tier = self.sut_default.tier
        provider = self.get_provider(provider_name)
        model = model_override or provider.tiers.get(tier, {}).get("model", "step-3.7-flash")
        return LLMClient(
            provider=provider_name,
            base_url=provider.base_url,
            api_key=provider.api_key,
            api_type=provider.api_type,
            model=model,
        )

    def get_judge_client(self) -> "LLMClient":
        from agent_eval_gate.llm_client import LLMClient

        provider_name = self.judge.provider
        tier = self.judge.tier
        provider = self.get_provider(provider_name)
        model = provider.tiers.get(tier, {}).get("model", "MiniMax-M2.7-highspeed")
        return LLMClient(
            provider=provider_name,
            base_url=provider.base_url,
            api_key=provider.api_key,
            api_type=provider.api_type,
            model=model,
        )
