"""Core protocols and data models for agent-eval-gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Task:
    """A single evaluation task."""
    id: str
    type: str  # qa | structured | tool_use | faithfulness
    prompt: str
    expected: Any
    context: Optional[str] = None
    tools: Optional[list[dict]] = None
    judge_criteria: str = "Evaluate whether the agent output correctly satisfies the task requirements."


@dataclass
class SUTOutput:
    """Output from a Subject Under Test (SUT)."""
    task_id: str
    framework: str
    model: str
    raw_text: str
    parsed_json: Optional[dict] = None
    tool_calls: Optional[list[dict]] = None
    latency_ms: float = 0.0
    tokens: Optional[dict] = None
    provenance_hash: str = ""


@dataclass
class ItemVerdict:
    """Judge verdict for a single (task, framework) cell."""
    task_id: str
    framework: str
    score: float  # 0.0 .. 1.0
    reason: str
    judge_model: str
    judge_vendor: str
    fallback_used: bool = False


@dataclass
class CellOutput:
    """Aggregated verdicts for one (framework, task) across judges."""
    framework: str
    task_id: str
    verdicts: list[ItemVerdict] = field(default_factory=list)

    @property
    def primary_score(self) -> float:
        if not self.verdicts:
            return 0.0
        return self.verdicts[0].score

    @property
    def primary_reason(self) -> str:
        if not self.verdicts:
            return ""
        return self.verdicts[0].reason


@dataclass
class Baseline:
    """Frozen regression baseline."""
    id: str
    tasks_sha256: str
    model_pins: dict[str, str]
    framework_pins: dict[str, str]
    judge_pin: str
    scores_per_task: dict[str, dict[str, float]]
    pairwise_ranking: list[dict]
    run_at: str
    braintrust_experiment_id: Optional[str] = None


@dataclass
class DriftAxis:
    NONE = "none"
    MODEL = "model"
    FRAMEWORK = "framework"
    DATA = "data"
    JUDGE = "judge"


@dataclass
class DriftReport:
    axis: str  # none | model | framework | data | judge
    evidence: str
    recommendation: str
    is_regression: bool
    score_delta: Optional[float] = None


@dataclass
class OrdinalRanking:
    ranking: list[dict]  # [{framework, score, wins, ...}]
    pairwise_comparisons: list[dict]
    total_pairs: int


@dataclass
class GateReport:
    run_id: str
    run_meta: dict
    per_task_verdicts: dict[str, dict[str, ItemVerdict]]  # task_id -> framework -> verdict
    pairwise_ranking: Optional[OrdinalRanking] = None
    drift_vs_baseline: Optional[DriftReport] = None
    gate_decision: str = "pass"  # pass | fail | unknown
