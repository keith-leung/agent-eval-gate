"""Core eval gate — orchestrates SUT runs, judging, pairwise ranking, and drift."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional

from agent_eval_gate.config import EvalConfig
from agent_eval_gate.protocols import (
    Baseline,
    CellOutput,
    DriftReport,
    GateReport,
    ItemVerdict,
    OrdinalRanking,
    SUTOutput,
    Task,
)
from agent_eval_gate.task_set import load_task_set
from agent_eval_gate.lineage import build_run_meta, compute_file_hash
from agent_eval_gate.suts import SUT_REGISTRY, BASELINE_SUTS
from agent_eval_gate.judge import assert_cross_vendor, geval_judge, pairwise_rank, exact_match_evaluate
from agent_eval_gate.baseline import build_baseline, compare_to_baseline, push_baseline
from agent_eval_gate.export import export_to_inspect, setup_langsmith, setup_langfuse, setup_phoenix


async def _run_sut(client, framework: str, task: Task) -> SUTOutput:
    runner = SUT_REGISTRY.get(framework)
    if runner is None:
        raise RuntimeError(f"No SUT runner registered for framework '{framework}'.")
    return await runner(client, task)


async def run_suts(config: EvalConfig, tasks: list[Task], selected_frameworks: Optional[list[str]] = None) -> dict[str, dict[str, SUTOutput]]:
    results: dict[str, dict[str, SUTOutput]] = {}
    frameworks = selected_frameworks or list(SUT_REGISTRY.keys())

    for framework in frameworks:
        client = config.get_sut_client(framework)
        results[framework] = {}
        for task in tasks:
            try:
                output = await _run_sut(client, framework, task)
                results[framework][task.id] = output
            except Exception as exc:
                # Record failure but continue
                from agent_eval_gate.protocols import SUTOutput
                from agent_eval_gate.suts.base import make_sut_output
                output = make_sut_output(task, framework, client.model, f"ERROR: {exc}")
                output.raw_text = f"ERROR: {exc}"
                results[framework][task.id] = output
    return results


async def judge_outputs(
    config: EvalConfig,
    tasks: list[Task],
    sut_outputs: dict[str, dict[str, SUTOutput]],
) -> dict[str, dict[str, ItemVerdict]]:
    judge_client = config.get_judge_client()
    verdicts: dict[str, dict[str, ItemVerdict]] = {}

    for framework, task_outputs in sut_outputs.items():
        verdicts[framework] = {}
        for task in tasks:
            if task.id not in task_outputs:
                continue
            output = task_outputs[task.id]
            try:
                assert_cross_vendor(
                    config.sut_default.provider,
                    output.model,
                    config.judge.provider,
                    judge_client.model,
                )
                verdict = geval_judge(judge_client, task, output.raw_text, framework=framework)
            except Exception as exc:
                verdict = exact_match_evaluate(task, output.raw_text, judge_model=judge_client.model, judge_vendor=judge_client.provider)
            verdicts[framework][task.id] = verdict
    return verdicts


def compute_pairwise_ranking(
    config: EvalConfig,
    tasks: list[Task],
    sut_outputs: dict[str, dict[str, SUTOutput]],
    verdicts: dict[str, dict[str, ItemVerdict]],
) -> Optional[OrdinalRanking]:
    judge_client = config.get_judge_client()
    # Rank on the first task for demo purposes
    if not tasks:
        return None
    task = tasks[0]
    cells = []
    for framework, task_outputs in sut_outputs.items():
        if task.id in task_outputs:
            cells.append((framework, task.id, task_outputs[task.id].raw_text))
    if len(cells) < 2:
        return None
    return pairwise_rank(cells, judge_client, task)


async def run_gate(config: EvalConfig, baseline_path: Optional[str] = None, selected_frameworks: Optional[list[str]] = None) -> GateReport:
    tasks = load_task_set()
    tasks_bytes = "\n".join(t.prompt for t in tasks).encode("utf-8")

    # Run SUTs
    sut_outputs = await run_suts(config, tasks, selected_frameworks)

    # Judge
    verdicts = await judge_outputs(config, tasks, sut_outputs)

    # Pairwise
    ranking = compute_pairwise_ranking(config, tasks, sut_outputs, verdicts)

    # Build meta
    sut_pins = {}
    for fw in sut_outputs:
        if sut_outputs[fw]:
            first_task_id = list(sut_outputs[fw].keys())[0]
            sut_pins[fw] = sut_outputs[fw][first_task_id].model

    framework_versions = {fw: "unknown" for fw in sut_outputs}
    meta = build_run_meta(tasks_bytes, sut_pins, config.get_judge_client().model, framework_versions)

    # Drift
    drift: Optional[DriftReport] = None
    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        baseline = Baseline(**baseline_data)
        drift = compare_to_baseline(
            GateReport(
                run_id="current",
                run_meta=meta,
                per_task_verdicts=verdicts,
                pairwise_ranking=ranking,
            ),
            baseline,
        )

    gate_decision = "pass"
    if drift and drift.is_regression:
        gate_decision = "fail"

    report = GateReport(
        run_id=meta["timestamp"],
        run_meta=meta,
        per_task_verdicts=verdicts,
        pairwise_ranking=ranking,
        drift_vs_baseline=drift,
        gate_decision=gate_decision,
    )

    # Tracing
    setup_langsmith(config.tracing.langsmith_api_key or None)
    setup_langfuse(config.tracing.langfuse_public_key or None, config.tracing.langfuse_secret_key or None)
    setup_phoenix(config.tracing.phoenix_endpoint or None)

    # Export
    report._inspect_samples = export_to_inspect(report)  # type: ignore[attr-defined]

    return report
