"""Cross-vendor guard — structurally enforces judge != SUT vendor."""

from __future__ import annotations

from agent_eval_gate.protocols import ItemVerdict


VENDOR_MAP = {
    "gpt_agent": "stepfun",  # gpt-agent.cc serves StepFun models
    "minimax": "minimax",
    "mimo": "mimo",
    "anthropic": "anthropic",
    "mock": "mock",
}


def resolve_vendor(provider_name: str, model: str) -> str:
    if provider_name in VENDOR_MAP:
        return VENDOR_MAP[provider_name]
    # Heuristic: if model contains 'claude' -> anthropic, 'minimax' -> minimax
    lower = model.lower()
    if "claude" in lower:
        return "anthropic"
    if "minimax" in lower:
        return "minimax"
    if "mimo" in lower:
        return "mimo"
    return provider_name


def assert_cross_vendor(sut_provider: str, sut_model: str, judge_provider: str, judge_model: str) -> None:
    sut_vendor = resolve_vendor(sut_provider, sut_model)
    judge_vendor = resolve_vendor(judge_provider, judge_model)
    if sut_vendor == judge_vendor:
        raise RuntimeError(
            f"Cross-vendor guard violated: judge vendor ({judge_vendor}) == SUT vendor ({sut_vendor}). "
            f"SUT provider={sut_provider} model={sut_model}; judge provider={judge_provider} model={judge_model}."
        )
