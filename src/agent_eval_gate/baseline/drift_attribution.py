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


def _mean_score_delta(
    new_report: GateReport,
    baseline: Baseline,
) -> float | None:
    """Mean per-task score delta (new - baseline) across matching task ids.

    Returns None if no comparable scores exist (e.g. baseline has empty
    scores_per_task, or task sets don't overlap).
    """
    baseline_scores = baseline.scores_per_task or {}
    if not baseline_scores:
        return None
    deltas: list[float] = []
    # per_task_verdicts is {task_id: {framework: ItemVerdict}}
    # baseline_scores is {task_id: {framework: score}}
    for task_id, framework_verdicts in new_report.per_task_verdicts.items():
        for framework, verdict in framework_verdicts.items():
            base = baseline_scores.get(task_id, {}).get(framework)
            if base is None:
                continue
            deltas.append(verdict.score - base)
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


_REGRESSION_THRESHOLD = 0.05  # mean score drop beyond this = regression


def compare_to_baseline(
    new_report: GateReport,
    baseline: Baseline,
) -> DriftReport:
    """Attribute score delta to model / framework / data / judge drift.

    Per SPEC §3 B4: regression = (accuracy delta < -threshold) AND drift is
    NOT judge drift. is_regression therefore depends on the observed score
    delta, not merely on which axis changed.
    """
    new_meta = new_report.run_meta
    new_tasks_sha = new_meta.get("tasks_sha256", "")
    new_judge = new_meta.get("judge_model", "")

    # Data drift — baseline invalid for direct comparison; not a regression call.
    if new_tasks_sha != baseline.tasks_sha256:
        return DriftReport(
            axis="data",
            evidence=f"tasks_sha256 changed: baseline={baseline.tasks_sha256[:12]}... new={new_tasks_sha[:12]}...",
            recommendation="Re-baseline with the new task set.",
            is_regression=False,
            score_delta=None,
        )

    # Compute the comparable score delta once (data is unchanged here).
    delta = _mean_score_delta(new_report, baseline)

    # Model drift
    new_sut_models = new_meta.get("sut_models", {})
    if new_sut_models != baseline.model_pins:
        changed = {k: (baseline.model_pins.get(k), new_sut_models.get(k)) for k in new_sut_models if baseline.model_pins.get(k) != new_sut_models.get(k)}
        is_reg = delta is not None and delta < -_REGRESSION_THRESHOLD
        return DriftReport(
            axis="model",
            evidence=f"SUT model pins changed: {changed}; mean_score_delta={delta}",
            recommendation="Verify the new model performs on par before promoting.",
            is_regression=is_reg,
            score_delta=delta,
        )

    # Framework drift
    new_fw = new_meta.get("framework_versions", {})
    if new_fw != baseline.framework_pins:
        changed = {k: (baseline.framework_pins.get(k), new_fw.get(k)) for k in new_fw if baseline.framework_pins.get(k) != new_fw.get(k)}
        is_reg = delta is not None and delta < -_REGRESSION_THRESHOLD
        return DriftReport(
            axis="framework",
            evidence=f"Framework versions changed: {changed}; mean_score_delta={delta}",
            recommendation="Re-run baseline with the new framework version to isolate.",
            is_regression=is_reg,
            score_delta=delta,
        )

    # Judge drift — never a regression (the SUT didn't worsen; the judge moved).
    if new_judge != baseline.judge_pin:
        return DriftReport(
            axis="judge",
            evidence=f"Judge model changed: baseline={baseline.judge_pin} new={new_judge}; mean_score_delta={delta}",
            recommendation="Re-run old judge against new outputs to confirm SUT did not regress.",
            is_regression=False,
            score_delta=delta,
        )

    # No axis changed — regression iff scores actually dropped.
    is_reg = delta is not None and delta < -_REGRESSION_THRESHOLD
    return DriftReport(
        axis="none",
        evidence=f"No axis changed relative to baseline; mean_score_delta={delta}",
        recommendation="No action required." if not is_reg else "Scores dropped with no axis change — investigate judge noise.",
        is_regression=is_reg,
        score_delta=delta,
    )
