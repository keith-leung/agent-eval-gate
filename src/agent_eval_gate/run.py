"""CLI entrypoint: python -m agent_eval_gate.run --all"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Optional

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
                print(f"  #{entry['rank']} {entry['framework']} (score={entry['btl_score']:.3f}, wins={entry['wins']})")
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
                "pairwise_ranking": report.pairwise_ranking.ranking if report.pairwise_ranking else None,
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
