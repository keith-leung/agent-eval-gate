"""CLI entrypoint: python -m agent_eval_gate.run --all"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Optional

# Disable OpenAI Agents SDK tracing BEFORE the `agents` package is imported
# anywhere. Without this, its BackendSpanExporter retries a (missing/invalid)
# OPENAI_API_KEY with exponential backoff on background threads that never exit,
# hanging the whole process — the CI hang Trae diagnosed. Must be set pre-import.
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

# Silence benign teardown noise. SUT adapters run in per-thread event loops
# (to isolate blocking framework calls); the framework HTTP clients they create
# schedule their async close during GC, after that loop is gone, emitting
# "Event loop is closed" tracebacks. These fire AFTER results are collected and
# have zero effect on the report or exit code. Filter exactly those two messages.
import logging as _logging


class _DropClosedLoopNoise(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:  # noqa: A003
        m = record.getMessage()
        return "Event loop is closed" not in m and "Task exception was never retrieved" not in m


_logging.getLogger("asyncio").addFilter(_DropClosedLoopNoise())

_prev_unraisablehook = sys.unraisablehook


def _quiet_unraisable(unraisable) -> None:
    exc = getattr(unraisable, "exc_value", None)
    if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
        return
    _prev_unraisablehook(unraisable)


sys.unraisablehook = _quiet_unraisable

from agent_eval_gate.config import EvalConfig
from agent_eval_gate.gate import run_gate
from agent_eval_gate.baseline import push_baseline, build_baseline
from agent_eval_gate.lineage import record_run_ledger
from agent_eval_gate.task_set import load_task_set


def main() -> int:
    parser = argparse.ArgumentParser(description="agent-eval-gate runner")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--all", action="store_true", help="Run full gate")
    parser.add_argument("--save-baseline", action="store_true", help="Save current run as baseline")
    parser.add_argument("--baseline-path", default="baselines/baseline.json", help="Baseline file path")
    parser.add_argument("--frameworks", default=None, help="Comma-separated framework list")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 1

    config = EvalConfig.from_file(args.config)
    selected = args.frameworks.split(",") if args.frameworks else None

    if args.all:
        report = asyncio.run(run_gate(config, baseline_path=args.baseline_path if os.path.exists(args.baseline_path) else None, selected_frameworks=selected))

        print("\n=== agent-eval-gate report ===")
        print(f"run_id: {report.run_id}")
        print(f"gate_decision: {report.gate_decision}")
        print(f"tasks_sha256: {report.run_meta.get('tasks_sha256', '')[:16]}...")
        if report.pairwise_ranking:
            print("\nPairwise ranking:")
            for entry in report.pairwise_ranking.ranking:
                sc = entry.get("btl_score")
                sc_str = f"{sc:.3f}" if isinstance(sc, (int, float)) else str(entry.get("status", "did_not_run"))
                print(f"  #{entry.get('rank', '-')} {entry['framework']} (score={sc_str}, wins={entry.get('wins', 0)})")
        if report.drift_vs_baseline:
            d = report.drift_vs_baseline
            print(f"\nDrift report: axis={d.axis} is_regression={d.is_regression}")
            print(f"  evidence: {d.evidence}")
            print(f"  recommendation: {d.recommendation}")
        else:
            print("\nDrift report: no baseline loaded.")

        # Save report
        os.makedirs("output", exist_ok=True)
        out_path = f"output/gate_report_{report.run_id.replace(':', '').replace('-', '').replace('T', '_')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": report.run_id,
                "run_meta": report.run_meta,
                "gate_decision": report.gate_decision,
                "errored_frameworks": report.errored_frameworks,
                "pairwise_ranking": report.pairwise_ranking.ranking if report.pairwise_ranking else None,
                "pairwise_btl_fallback_used": report.pairwise_ranking.btl_fallback_used if report.pairwise_ranking else None,
                "pairwise_judge_fallback_count": report.pairwise_ranking.judge_fallback_count if report.pairwise_ranking else 0,
                "drift_vs_baseline": report.drift_vs_baseline.__dict__ if report.drift_vs_baseline else None,
            }, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nReport saved to {out_path}")

        if args.save_baseline:
            tasks = load_task_set()
            tasks_bytes = "\n".join(t.prompt for t in tasks).encode("utf-8")
            sut_pins = report.run_meta.get("sut_models", {})
            baseline = build_baseline(
                run_id=report.run_id,
                tasks_bytes=tasks_bytes,
                sut_pins=sut_pins,
                judge_pin=report.run_meta.get("judge_model", ""),
                framework_versions=report.run_meta.get("framework_versions", {}),
                scores_per_task={},
                pairwise_ranking=report.pairwise_ranking.ranking if report.pairwise_ranking else [],
            )
            os.makedirs("baselines", exist_ok=True)
            with open(args.baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline.__dict__, f, indent=2, ensure_ascii=False, default=str)
            push_baseline(baseline)
            print(f"Baseline saved to {args.baseline_path}")

        return 0 if report.gate_decision == "pass" else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
