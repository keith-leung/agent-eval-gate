"""Minimal tests for agent-eval-gate."""

from __future__ import annotations

import json
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_task_set_count_and_types():
    from agent_eval_gate.task_set import load_task_set
    tasks = load_task_set()
    assert len(tasks) >= 16, f"Expected >=16 tasks, got {len(tasks)}"
    types = {t.type for t in tasks}
    assert types == {"qa", "structured", "tool_use", "faithfulness"}, f"Missing types: {types}"
    per_type = {}
    for t in tasks:
        per_type.setdefault(t.type, []).append(t)
    for ttype, items in per_type.items():
        assert len(items) >= 4, f"Type {ttype} has only {len(items)} tasks"


def test_protocols_import():
    from agent_eval_gate.protocols import (
        Baseline,
        DriftReport,
        GateReport,
        ItemVerdict,
        OrdinalRanking,
        SUTOutput,
        Task,
    )
    assert True


def test_config_loads():
    from agent_eval_gate.config import EvalConfig
    cfg = EvalConfig.from_file("config.ci.yaml")
    assert cfg.mode == "mock"
    assert "mock" in cfg.providers


def test_lineage_hash_stable():
    from agent_eval_gate.lineage import compute_bytes_hash
    h1 = compute_bytes_hash(b"hello")
    h2 = compute_bytes_hash(b"hello")
    assert h1 == h2
    assert len(h1) == 64


def test_extract_json_parser():
    from agent_eval_gate.judge.exact_match import _extract_json
    assert _extract_json('{"winner": "A"}') == {"winner": "A"}
    assert _extract_json('```json\n{"winner": "B"}\n```') == {"winner": "B"}
    assert _extract_json('Some text {"winner": "A"} more') == {"winner": "A"}


def test_cross_vendor_guard_raises():
    from agent_eval_gate.judge.cross_vendor_guard import assert_cross_vendor
    try:
        assert_cross_vendor("gpt_agent", "step-3.7-flash", "gpt_agent", "step-3.7-flash")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "Cross-vendor guard violated" in str(e)


def test_exact_match_fallback():
    from agent_eval_gate.protocols import Task
    from agent_eval_gate.judge.exact_match import exact_match_evaluate
    task = Task(id="t1", type="qa", prompt="p", expected="Paris")
    verdict = exact_match_evaluate(task, "The capital of France is Paris.")
    assert verdict.score == 1.0
    assert verdict.fallback_used is True


def test_btl_score():
    from agent_eval_gate.judge.pairwise_rank import _btl_score
    scores = _btl_score({"a": 3, "b": 1}, {"a": 4, "b": 4})
    assert scores["a"] > scores["b"]


if __name__ == "__main__":
    test_task_set_count_and_types()
    test_protocols_import()
    test_config_loads()
    test_lineage_hash_stable()
    test_extract_json_parser()
    test_cross_vendor_guard_raises()
    test_exact_match_fallback()
    test_btl_score()
    print("All tests passed.")
