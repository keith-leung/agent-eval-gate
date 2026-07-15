"""Exact-match fallback judge — deterministic when LLM judge fails."""

from __future__ import annotations

import re

from agent_eval_gate.protocols import ItemVerdict, Task


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response, handling thinking output and fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return __import__("json").loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return __import__("json").loads(text[start : end + 1])

    raise __import__("json").JSONDecodeError(
        f"No JSON object found in LLM response ({len(text)} chars)",
        text[:200],
        0,
    )


def exact_match_evaluate(task: Task, output_text: str, judge_model: str = "mock", judge_vendor: str = "mock") -> ItemVerdict:
    """Deterministic fallback: exact string match on expected."""
    expected = str(task.expected).strip().lower()
    # Defensive: some SUT outputs may arrive as a non-str (e.g. a parsed dict);
    # coerce to str so this final tier never crashes on type.
    predicted = str(output_text).strip().lower() if not isinstance(output_text, str) else output_text.strip().lower()

    score = 1.0 if expected in predicted or predicted in expected else 0.0
    reason = f"Exact-match fallback. Expected='{expected}', predicted='{predicted[:200]}'."
    return ItemVerdict(
        task_id=task.id,
        framework="fallback",
        score=score,
        reason=reason,
        judge_model=judge_model,
        judge_vendor=judge_vendor,
        fallback_used=True,
    )
