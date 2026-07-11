"""inspect-ai export adapter — wraps B's scorers as inspect_ai.scorer.Score."""

from __future__ import annotations

from typing import Any

from agent_eval_gate.protocols import GateReport, ItemVerdict


def export_to_inspect(report: GateReport) -> list[dict[str, Any]]:
    """Produce inspect-ai-compatible Sample dicts from a GateReport."""
    samples = []
    for task_id, framework_verdicts in report.per_task_verdicts.items():
        for framework, verdict in framework_verdicts.items():
            sample = {
                "id": f"{task_id}::{framework}",
                "input": task_id,
                "target": framework,
                "metadata": {
                    "judge_model": verdict.judge_model,
                    "judge_vendor": verdict.judge_vendor,
                    "fallback_used": verdict.fallback_used,
                },
                "scores": [
                    {
                        "name": "agent_eval_gate.score",
                        "value": verdict.score,
                    }
                ],
            }
            samples.append(sample)
    return samples
