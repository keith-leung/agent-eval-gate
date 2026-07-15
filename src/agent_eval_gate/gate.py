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
                from agent_eval_gate.suts.base import make_sut_output
                err_msg = f"ERROR: {exc}"[:500]
                output = make_sut_output(task, framework, client.model, err_msg)
                output.raw_text = err_msg
                results[framework][task.id] = output
            await asyncio.sleep(0.3)
        await asyncio.sleep(0.5)
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
    """Rank SUTs via per-task pairwise comparison, aggregating wins.

    Comparing on a single task (the old behavior) is noise when all SUTs
    return the same answer — the judge picks a deterministic but meaningless
    winner. Instead we run pairwise on EACH task where SUT outputs diverge,
    and aggregate wins across all such tasks. Tasks where every SUT returns
    identical output contribute no signal and are skipped.
    """
    judge_client = config.get_judge_client()
    if not tasks:
        return None

    frameworks = list(sut_outputs.keys())
    if len(frameworks) < 2:
        return None

    # Aggregate wins + comparisons across per-task pairwise runs.
    agg_wins: dict[str, float] = {f: 0.0 for f in frameworks}
    agg_total: dict[str, int] = {f: 0 for f in frameworks}
    agg_comparisons: list[dict] = []
    tasks_used = 0
    total_fallback = 0

    for task in tasks:
        cells = []
        outputs_for_task = {}
        for fw in frameworks:
            if task.id in sut_outputs.get(fw, {}):
                text = sut_outputs[fw][task.id].raw_text
                if isinstance(text, str) and not text.startswith("ERROR:"):
                    cells.append((fw, task.id, text))
                    outputs_for_task[fw] = text
        if len(cells) < 2:
            continue
        # Skip tasks where all SUT outputs are identical (no signal).
        unique_outputs = set(outputs_for_task.values())
        if len(unique_outputs) <= 1:
            continue
        tasks_used += 1
        ranking = pairwise_rank(cells, judge_client, task)
        for entry in ranking.ranking:
            fw = entry["framework"]
            agg_wins[fw] += entry.get("wins", 0)
            agg_total[fw] += entry.get("total_comparisons", 0)
        agg_comparisons.extend(ranking.pairwise_comparisons)
        total_fallback += ranking.judge_fallback_count

    if tasks_used == 0:
        # No diverging tasks — fall back to first-task ranking.
        task = tasks[0]
        cells = [(fw, task.id, sut_outputs[fw][task.id].raw_text)
                 for fw in frameworks if task.id in sut_outputs.get(fw, {})
                 and isinstance(sut_outputs[fw][task.id].raw_text, str)
                 and not sut_outputs[fw][task.id].raw_text.startswith("ERROR:")]
        if len(cells) < 2:
            return None
        return pairwise_rank(cells, judge_client, task)

    # Build final ranking from aggregated wins.
    final_ranking = sorted(frameworks, key=lambda f: agg_wins.get(f, 0.0), reverse=True)
    # BTL as win rate (Laplace-smoothed).
    btl = {}
    for fw in frameworks:
        w = agg_wins.get(fw, 0.0)
        n = max(agg_total.get(fw, 1), 1)
        btl[fw] = (w + 0.5) / (n + 1.0)

    return OrdinalRanking(
        ranking=[
            {"rank": r, "framework": fw, "btl_score": btl[fw],
             "wins": agg_wins[fw], "total_comparisons": agg_total[fw]}
            for r, fw in enumerate(final_ranking, 1)
        ],
        pairwise_comparisons=agg_comparisons,
        total_pairs=len(agg_comparisons),
        judge_fallback_count=total_fallback,
    )


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

    errored = _collect_errored_frameworks(sut_outputs)
    total_suts = len(sut_outputs)

    # A run where most SUTs errored is a broken gate, regardless of drift.
    # A 2-SUT "ranking" is not a valid cross-framework eval (SPEC §3 B1).
    if total_suts > 0 and len(errored) > total_suts // 2:
        gate_decision = "fail"
    elif drift:
        gate_decision = "fail" if drift.is_regression else "pass"
    else:
        # No baseline loaded: cannot make a regression pass/fail call.
        # "unknown" is honest; run.py treats it as exit 1 (not a clean pass).
        gate_decision = "unknown"

    report = GateReport(
        run_id=meta["timestamp"],
        run_meta=meta,
        per_task_verdicts=verdicts,
        pairwise_ranking=ranking,
        drift_vs_baseline=drift,
        gate_decision=gate_decision,
        errored_frameworks=errored,
        pairwise_task_count=len(tasks),
    )

    # Tracing
    setup_langsmith(config.tracing.langsmith_api_key or None)
    setup_langfuse(config.tracing.langfuse_public_key or None, config.tracing.langfuse_secret_key or None)
    setup_phoenix(config.tracing.phoenix_endpoint or None)

    # Export
    report._inspect_samples = export_to_inspect(report)  # type: ignore[attr-defined]

    return report


def _collect_errored_frameworks(sut_outputs: dict[str, dict[str, SUTOutput]]) -> list[str]:
    """Frameworks whose SUT run produced an ERROR marker instead of real output."""
    errored = set()
    for fw, task_outputs in sut_outputs.items():
        for output in task_outputs.values():
            # Defensive: some SUT adapters may produce a non-str raw_text
            # (e.g. a parsed dict); coerce before checking the error prefix.
            text = output.raw_text if isinstance(output.raw_text, str) else str(output.raw_text)
            if text.startswith("ERROR:"):
                errored.add(fw)
                break
    return sorted(errored)


_FRAMEWORK_PACKAGE_NAMES: dict[str, str] = {
    "openai_sdk": "openai-agents",
    "pydantic_ai": "pydantic-ai",
    "adk": "google-adk",
    "strands": "strands-agents",
    "langchain": "langchain",
    "claude_agent_sdk": "anthropic",
    "maf": "autogen-agentchat",
    "crewai": "crewai",
    "smolagents": "smolagents",
}


def _probe_framework_versions(framework_keys: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for key in framework_keys:
        pkg = _FRAMEWORK_PACKAGE_NAMES.get(key, key)
        try:
            versions[key] = importlib.metadata.version(pkg)
        except Exception:
            versions[key] = "unknown"
    return versions
