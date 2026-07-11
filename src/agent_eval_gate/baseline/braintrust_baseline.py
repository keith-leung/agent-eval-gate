"""Braintrust baseline substrate (immutable experiment snapshots)."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from agent_eval_gate.protocols import Baseline
from agent_eval_gate.lineage import record_run_ledger


def init_braintrust(api_key: str) -> None:
    """Initialize Braintrust if available and key is set."""
    if not api_key or api_key == "":
        return
    try:
        import braintrust  # noqa: F401
        os.environ["BRAINTRUST_API_KEY"] = api_key
    except ImportError:
        pass  # Braintrust not installed; skip


def push_baseline(baseline: Baseline, ledger_path: str = "runs/ledger.jsonl") -> Optional[str]:
    """Push baseline to Braintrust and append to local ledger."""
    init_braintrust("")
    experiment_id = None
    try:
        import braintrust
        project = braintrust.init_project(project_name="agent-eval-gate-baselines")
        experiment = project.create_experiment(name=baseline.id)
        experiment.log(
            metadata={
                "tasks_sha256": baseline.tasks_sha256,
                "judge_pin": baseline.judge_pin,
                "model_pins": baseline.model_pins,
                "framework_pins": baseline.framework_pins,
            }
        )
        experiment_id = experiment.id
    except Exception:
        pass

    record_run_ledger(ledger_path, {
        "run_id": baseline.id,
        "type": "baseline",
        "tasks_sha256": baseline.tasks_sha256,
        "judge_pin": baseline.judge_pin,
        "model_pins": baseline.model_pins,
        "framework_pins": baseline.framework_pins,
        "braintrust_experiment_id": experiment_id,
        "run_at": baseline.run_at,
    })
    return experiment_id
