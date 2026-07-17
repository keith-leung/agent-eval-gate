# agent-eval-gate

Cross-framework agent regression gate. One shared task set, evaluated against nine agent frameworks, scored by a cross-vendor LLM-as-judge stack, with regression baseline and drift attribution.

## Problem

Proving an agent won't regress in production is harder than running a golden-set accuracy check. Three hard questions remain:

1. **Cross-vendor judging** — same-vendor judging is the athlete-as-referee anti-pattern. The fix is structural, not a prompt warning.
2. **Noisy absolute scores** — LLM judges give noisy absolute scores. Ordinal pairwise ranking (which config beats which) is more robust.
3. **Regression attribution** — a score dropped. Was it the model swap, the framework upgrade, the data drift, or the judge itself drifting? Without lineage + a baseline comparator, you can't tell.

## Architecture

### B1 — Multi-framework SUT harness

A common `AgentUnderTest` contract (`run(task) -> SUTOutput`) with thin adapters for each framework. The eval never imports framework-specific types beyond the adapter.

**Nine SUTs (LOCKED):**

1. OpenAI Agents SDK — `Agent` + `Runner`
2. PydanticAI — typed-output agent
3. Google ADK — Google's Agent Development Kit
4. AWS Strands — AWS agent SDK
5. LangChain — `Runnable` chain / `create_react_agent`
6. Anthropic Claude Agent SDK — safety-critical / extended-reasoning option
7. Microsoft Agent Framework (MAF) — 2026 merger of AutoGen + Semantic Kernel
8. CrewAI — role-based orchestration
9. Smolagents — HuggingFace lightweight code-agent

**ReAct loop** — the baseline/control condition, not a peer SUT. Built minimal from a plain while-loop.

### B2 — Cross-vendor LLM-as-judge core

- **Primary**: GEval-style LLM judge with per-task configurable criteria.
- **Fallback**: exact-match / deterministic scorer when the judge fails.
- **Structural guard**: `assert_cross_vendor()` enforces `judge_vendor != sut_vendor` at config time (raise, not warn).

### B3 — Ordinal pairwise ranking

- Uses a custom pairwise judge loop with a cross-vendor LLM judge (DeepEval `GEval` retained for absolute scoring). `ArenaGEval` was evaluated but does not fit a deterministic single-judge CI loop.
- **Bradley-Terry-Luce** score computation from pairwise wins — built in-house.
- Deterministic heuristic fallback per pair when a judge call fails.

### B4 — Regression baseline + drift attribution

- **Braintrust** for immutable experiment snapshots.
- Multi-axis drift attribution: model / framework / data / judge.
- Judge-drift vs SUT-regression distinction — the core question this repo exists to answer.
- SHA-256 lineage over task set + append-only ledger.

### B5 — Shared task set

16 hand-authored tasks across 4 types, aligned with standard benchmark task styles:

| Type | Count | Benchmark style |
|------|-------|-----------------|
| qa | 4 | GAIA-style general assistant |
| structured | 4 | Typed-output / Pydantic schema |
| tool_use | 4 | Tau-Bench-style tool calling |
| faithfulness | 4 | Output-vs-context grounding |

### B6 — inspect-ai + LangSmith integration

- **inspect-ai** (TOP-BILL): export adapter wraps local scorers as `inspect_ai.scorer.Score`.
- **LangSmith / Langfuse / Phoenix**: env-var driven tracing, off when keys unset.

## Setup

```bash
conda create -n agent-eval-gate python=3.11 -y
conda activate agent-eval-gate
pip install -e .[dev]
```

Copy `config.example.yaml` to `config.yaml` and fill real keys.

## Usage

```bash
# Run full gate
python -m agent_eval_gate.run --all

# CI / mock mode
python -m agent_eval_gate.run --config config.ci.yaml --all
```

## Configuration

Mode is selected by `--config` flag, never by environment variable:

- `config.yaml` — real-LLM run, gitignored
- `config.ci.yaml` — mock mode, committed

## Keywords

This repo demonstrates the 2026 cross-framework eval vocabulary:

**SUT frameworks:** OpenAI Agents SDK, PydanticAI, Google ADK, AWS Strands, LangChain, Anthropic Claude Agent SDK, Microsoft Agent Framework, CrewAI, Smolagents

**Judge + observability:** inspect-ai (UK AISI), DeepEval (GEval + ArenaGEval), Braintrust, LangSmith, Langfuse, Phoenix / Arize Phoenix

**Method terms:** cross-vendor LLM-as-judge, judge-model independence, self-preference bias mitigation, ordinal pairwise ranking, Bradley-Terry-Luce, regression baseline, drift attribution, SHA-256 lineage

**Standard benchmarks:** GAIA, SWE-bench Verified, Tau-Bench, Cybench / GDM CTF

## License

MIT
