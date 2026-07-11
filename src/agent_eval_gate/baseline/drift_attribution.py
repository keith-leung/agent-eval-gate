"""Regression baseline + drift attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent_eval_gate.protocols import Baseline, DriftReport, GateReport
from agent_eval_gate.lineage import build_run_meta


def build_baseline(
    run_id: str,
    tasks_bytes: bytes,
    sut_pins: dict[str, str],
    judge_pin: str,
    framework_versions: dict[str, str],
    scores_per_task: dict[str, dict[str, float]],
    pairwise_ranking: list[dict],
    braintrust_experiment_id: Optional[str] = None,
) -> Baseline:
    meta = build_run_meta(tasks_bytes, sut_pins, judge_pin, framework_versions)
    return Baseline(
        id=run_id,
        tasks_sha256=meta["tasks_sha256"],
        model_pins=sut_pins,
        framework_pins=framework_versions,
        judge_pin=judge_pin,
        scores_per_task=scores_per_task,
        pairwise_ranking=pairwise_ranking,
        run_at=meta["timestamp"],
        braintrust_experiment_id=braintrust_experiment_id,
    )


def compare_to_baseline(
    new_report: GateReport,
    baseline: Baseline,
) -> DriftReport:
    """Attribute score delta to model / framework / data / judge drift."""
    new_meta = new_report.run_meta
    new_tasks_sha = new_meta.get("tasks_sha256", "")
    new_judge = new_meta.get("judge_model", "")

    # Data drift
    if new_tasks_sha != baseline.tasks_sha256:
        return DriftReport(
            axis="data",
            evidence=f"tasks_sha256 changed: baseline={baseline.tasks_sha256[:12]}... new={new_tasks_sha[:12]}...",
            recommendation="Re-baseline with the new task set.",
            is_regression=False,
            score_delta=None,
        )

    # Model drift
    new_sut_models = new_meta.get("sut_models", {})
    if new_sut_models != baseline.model_pins:
        changed = {k: (baseline.model_pins.get(k), new_sut_models.get(k)) for k in new_sut_models if baseline.model_pins.get(k) != new_sut_models.get(k)}
        return DriftReport(
            axis="model",
            evidence=f"SUT model pins changed: {changed}",
            recommendation="Verify the new model performs on par before promoting.",
            is_regression=True,
        )

    # Framework drift
    new_fw = new_meta.get("framework_versions", {})
    if new_fw != baseline.framework_pins:
        changed = {k: (baseline.framework_pins.get(k), new_fw.get(k)) for k in new_fw if baseline.framework_pins.get(k) != new_fw.get(k)}
        return DriftReport(
            axis="framework",
            evidence=f"Framework versions changed: {changed}",
            recommendation="Re-run baseline with the new framework version to isolate.",
            is_regression=True,
        )

    # Judge drift
    if new_judge != baseline.judge_pin:
        return DriftReport(
            axis="judge",
            evidence=f"Judge model changed: baseline={baseline.judge_pin} new={new_judge}",
            recommendation="Re-run old judge against new outputs to confirm SUT did not regress.",
            is_regression=False,
        )

    return DriftReport(
        axis="none",
        evidence="No axis changed relative to baseline.",
        recommendation="No action required.",
        is_regression=False,
    )
