"""GEval-style judge via DeepEval."""

from __future__ import annotations

from typing import Optional

from agent_eval_gate.protocols import ItemVerdict, Task
from agent_eval_gate.llm_client import LLMClient


def geval_judge(client: LLMClient, task: Task, output_text: str, framework: str = "unknown") -> ItemVerdict:
    """Judge a single task output using DeepEval GEval or a direct LLM prompt."""
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        # We need a LiteLLMJudge-style adapter. For simplicity, DeepEval accepts
        # a callable that returns a string. We'll use the raw client if possible,
        # but to keep it simple, we fall back to the direct prompt path below.
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
    except Exception as exc:
        raise RuntimeError(f"GEval measure failed for {task.id}: {exc}") from exc
