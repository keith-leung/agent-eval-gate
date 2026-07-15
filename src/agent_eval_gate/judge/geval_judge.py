"""GEval-style judge via DeepEval, with a direct-LLM-prompt fallback.

SPEC §3 B2 requires a two-path evaluator: LLM judge primary, deterministic
fallback when the judge fails. DeepEval's GEval is the primary path; if it
raises, we fall back to a direct LLM prompt that asks the same cross-vendor
judge for a 0/1 score + reason. The exact-match fallback in exact_match.py
is the final deterministic tier, used only when the LLM call itself fails.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Optional

# DeepEval 2.5.x imports `from langchain.schema import HumanMessage`, but
# langchain 1.x split langchain.schema into langchain_core.messages. Inject a
# compatibility shim so DeepEval can import without downgrading langchain
# (which the SUT adapters depend on at 1.3). This runs once at module load.
if "langchain.schema" not in sys.modules:
    try:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # noqa: F401
        _shim = types.ModuleType("langchain.schema")
        _shim.HumanMessage = HumanMessage
        _shim.AIMessage = AIMessage
        _shim.SystemMessage = SystemMessage
        sys.modules["langchain.schema"] = _shim
    except Exception:
        pass  # if langchain_core isn't available, the GEval path will fail and fallback engages

from agent_eval_gate.protocols import ItemVerdict, Task
from agent_eval_gate.llm_client import LLMClient


def _geval_with_deepeval(client: LLMClient, task: Task, output_text: str, framework: str) -> Optional[ItemVerdict]:
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        metric = GEval(
            name=f"Eval-{task.id}",
            criteria=task.judge_criteria,
            evaluation_steps=[
                "Compare the 'actual_output' with 'expected_output'.",
                "Score 1.0 if the output satisfies the task criteria; otherwise 0.0.",
                "Provide a concise reason.",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.5,
        )
        test_case = LLMTestCase(
            input=task.prompt,
            actual_output=output_text,
            expected_output=str(task.expected),
        )
        metric.measure(test_case)
        score = float(metric.score) if metric.score is not None else 0.0
        reason = metric.reason or "No reason provided."
        return ItemVerdict(
            task_id=task.id,
            framework=framework,
            score=score,
            reason=reason,
            judge_model=client.model,
            judge_vendor=client.provider,
            fallback_used=False,
        )
    except Exception:
        return None  # signal fallback needed


def _geval_with_llm_prompt(client: LLMClient, task: Task, output_text: str, framework: str) -> ItemVerdict:
    from agent_eval_gate.judge.exact_match import _extract_json

    system = (
        "You are an evaluation judge. Score the agent output against the task criteria.\n"
        "Return ONLY valid JSON: {\"score\": <0.0|1.0>, \"reason\": \"<brief>\"}"
    )
    user = json.dumps({
        "task_prompt": task.prompt,
        "expected": str(task.expected),
        "actual_output": output_text,
    }, ensure_ascii=False)
    raw = client.complete_sync(
        system=system,
        user_message=user,
        temperature=0.0,
        max_tokens=256,
    )
    # MiniMax emits <think>...</think> before the answer; _extract_json handles
    # thinking blocks + ``` fences and pulls the JSON object out.
    try:
        parsed = _extract_json(raw) if isinstance(raw, str) else {}
        score = float(parsed.get("score", 0.0))
    except Exception:
        parsed = {}
        score = 0.0
    reason = parsed.get("reason", raw[:200] if isinstance(raw, str) else "") if isinstance(parsed, dict) else ""
    return ItemVerdict(
        task_id=task.id,
        framework=framework,
        score=score,
        reason=reason,
        judge_model=client.model,
        judge_vendor=client.provider,
        fallback_used=True,
    )


def geval_judge(client: LLMClient, task: Task, output_text: str, framework: str = "unknown") -> ItemVerdict:
    """Judge a single task output: DeepEval GEval primary, direct LLM prompt fallback."""
    result = _geval_with_deepeval(client, task, output_text, framework)
    if result is not None:
        return result
    return _geval_with_llm_prompt(client, task, output_text, framework)
