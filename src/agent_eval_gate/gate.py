"""Core eval gate — orchestrates SUT runs, judging, pairwise ranking, and drift."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import threading
import time
from typing import Any, Optional

from agent_eval_gate.config import EvalConfig


def _stage(msg: str) -> None:
    """Diagnostic stage logger with timestamp and active thread count."""
    print(f"[{time.strftime('%H:%M:%S')}][thr={threading.active_count()}] {msg}", flush=True)


_DELAY_ENV = os.environ.get("B_MOCK_LLM_DELAY")
_MOCK_DELAY = float(_DELAY_ENV) if _DELAY_ENV else None
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


# Real measured SUT latencies are ~1-6s each (see _diag_sut_timing). Several
# adapters (crewai/smolagents/langchain/strands/adk/claude) call SYNC-BLOCKING
# framework methods (crew.kickoff / agent.run / executor.invoke) inside an
# `async def`. Awaiting them in the shared event loop freezes it, which is why a
# serial run appeared to "hang" and prompted (wrong) 45s-timeout + task-cutting
# workarounds. Fix: run each adapter in its OWN worker thread (own event loop)
# via to_thread, all SUT*task cells concurrently under a bounded semaphore, with
# a real wall-clock timeout that now actually fires. No task reduction.
SUT_CALL_TIMEOUT = 90.0   # generous; real calls ~1-6s. Only catches true hangs.
SUT_CONCURRENCY = 8
JUDGE_CONCURRENCY = 8      # geval_judge is sync; offloaded to threads, run concurrently.
JUDGE_CALL_TIMEOUT = 60.0  # per judge/pairwise call; real calls ~3s. A single stalled
                           # judge connection must not hang the whole phase (openai
                           # SDK's own default is 600s — far too long for a gate).


async def run_suts(config: EvalConfig, tasks: list[Task], selected_frameworks: Optional[list[str]] = None) -> dict[str, dict[str, SUTOutput]]:
    frameworks = selected_frameworks or list(SUT_REGISTRY.keys())

    # Mock/CI mode: the SUT adapters drive real third-party frameworks that need a
    # live endpoint. Under config.ci.yaml there is none, so previously every SUT
    # retried the dead endpoint (~14s each) and the OpenAI Agents SDK tracing
    # threads kept the process alive → CI hang. In mock mode we short-circuit to
    # deterministic, offline, per-(framework, task) outputs, which still exercises
    # the full gate (judge → pairwise → BTL → drift → decision) with no network.
    if _MOCK_DELAY is not None:
        _stage(f"run_suts: start (B_MOCK_LLM_DELAY={_MOCK_DELAY}s, stub mode)")
        from agent_eval_gate.suts.base import make_sut_output as _mk
        results_delay: dict[str, dict[str, SUTOutput]] = {fw: {} for fw in frameworks}
        clients_delay = {fw: config.get_sut_client(fw) for fw in frameworks}
        sem_delay = asyncio.Semaphore(SUT_CONCURRENCY)

        async def _delay_one(fw: str, task: Task) -> None:
            async with sem_delay:
                def _stub():
                    time.sleep(_MOCK_DELAY)
                    return _mk(task, fw, clients_delay[fw].model, f"[stub:{fw}] {task.id}")
                try:
                    out = await asyncio.wait_for(
                        asyncio.to_thread(_stub),
                        timeout=SUT_CALL_TIMEOUT,
                    )
                    results_delay[fw][task.id] = out
                except Exception as exc:
                    results_delay[fw][task.id] = _mk(task, fw, clients_delay[fw].model, f"ERROR: {exc}")

        await asyncio.gather(*[_delay_one(fw, t) for fw in frameworks for t in tasks])
        _stage(f"run_suts: done ({len(frameworks)}x{len(tasks)} stub cells)")
        return results_delay

    if getattr(config, "mode", "real") == "mock":
        from agent_eval_gate.suts.base import make_sut_output as _mk
        mock_results: dict[str, dict[str, SUTOutput]] = {fw: {} for fw in frameworks}
        for fw in frameworks:
            model = config.get_sut_client(fw).model
            for task in tasks:
                raw = f"[mock:{fw}] deterministic answer for {task.id}"
                mock_results[fw][task.id] = _mk(task, fw, model, raw)
        print(f"[suts] MOCK mode: {len(frameworks)}x{len(tasks)} deterministic cells (no network)", flush=True)
        return mock_results

    results: dict[str, dict[str, SUTOutput]] = {fw: {} for fw in frameworks}
    clients = {fw: config.get_sut_client(fw) for fw in frameworks}
    sem = asyncio.Semaphore(SUT_CONCURRENCY)
    t_phase = time.time()

    def _error_output(fw: str, task: Task, exc: BaseException) -> SUTOutput:
        from agent_eval_gate.suts.base import make_sut_output as _mk
        err_msg = f"ERROR: {type(exc).__name__}: {exc}"[:500]
        o = _mk(task, fw, clients[fw].model, err_msg)
        o.raw_text = err_msg
        return o

    async def _one(fw: str, task: Task) -> None:
        runner = SUT_REGISTRY.get(fw)
        if runner is None:
            results[fw][task.id] = _error_output(fw, task, RuntimeError(f"No SUT runner for '{fw}'"))
            return
        async with sem:
            t0 = time.time()
            try:
                # Adapter is a coroutine, but several do sync-blocking work; run
                # it in a dedicated worker thread with its own event loop so it
                # cannot block the shared loop, and wait_for can actually time out.
                out = await asyncio.wait_for(
                    asyncio.to_thread(lambda: asyncio.run(runner(clients[fw], task))),
                    timeout=SUT_CALL_TIMEOUT,
                )
                results[fw][task.id] = out
                status = "ok"
            except Exception as exc:  # noqa: BLE001 — includes asyncio.TimeoutError
                results[fw][task.id] = _error_output(fw, task, exc)
                status = f"ERR({type(exc).__name__})"
            print(f"[suts] {fw:18s} {task.id:10s} {time.time()-t0:5.1f}s  {status}", flush=True)

    await asyncio.gather(*[_one(fw, t) for fw in frameworks for t in tasks])
    print(f"[suts] phase done: {len(frameworks)}x{len(tasks)} cells in {time.time()-t_phase:.1f}s", flush=True)
    return results


async def judge_outputs(
    config: EvalConfig,
    tasks: list[Task],
    sut_outputs: dict[str, dict[str, SUTOutput]],
) -> dict[str, dict[str, ItemVerdict]]:
    judge_client = config.get_judge_client()
    verdicts: dict[str, dict[str, ItemVerdict]] = {fw: {} for fw in sut_outputs}

    # geval_judge is SYNC (uses complete_sync). Running 160 of them serially was
    # the second serial-blocking bottleneck (~8 min). Offload each to a worker
    # thread and run them concurrently under a bounded semaphore.
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    t_phase = time.time()

    def _judge_one_sync(framework: str, task: Task, output: SUTOutput) -> ItemVerdict:
        try:
            assert_cross_vendor(
                config.sut_default.provider, output.model,
                config.judge.provider, judge_client.model,
            )
            return geval_judge(judge_client, task, output.raw_text, framework=framework)
        except Exception:
            return exact_match_evaluate(task, output.raw_text, judge_model=judge_client.model, judge_vendor=judge_client.provider)

    async def _one(framework: str, task: Task, output: SUTOutput):
        async with sem:
            try:
                v = await asyncio.wait_for(
                    asyncio.to_thread(_judge_one_sync, framework, task, output),
                    timeout=JUDGE_CALL_TIMEOUT,
                )
            except Exception:
                # Judge stalled or errored (incl. TimeoutError) → deterministic
                # offline exact-match fallback, so one bad connection can't hang
                # the phase.
                v = exact_match_evaluate(
                    task, output.raw_text,
                    judge_model=judge_client.model, judge_vendor=judge_client.provider,
                )
            return framework, task.id, v

    coros = [
        _one(fw, task, task_outputs[task.id])
        for fw, task_outputs in sut_outputs.items()
        for task in tasks if task.id in task_outputs
    ]
    for framework, task_id, verdict in await asyncio.gather(*coros):
        verdicts[framework][task_id] = verdict
    print(f"[judge] {len(coros)} verdicts in {time.time()-t_phase:.1f}s", flush=True)
    return verdicts


async def compute_pairwise_ranking(
    config: EvalConfig,
    tasks: list[Task],
    sut_outputs: dict[str, dict[str, SUTOutput]],
    verdicts: dict[str, dict[str, ItemVerdict]],
) -> Optional[OrdinalRanking]:
    """Rank SUTs via per-task pairwise comparison, aggregating wins."""
    judge_client = config.get_judge_client()
    _t_pw = time.time()
    if not tasks:
        return None

    frameworks = list(sut_outputs.keys())
    if len(frameworks) < 2:
        return None

    agg_wins: dict[str, float] = {f: 0.0 for f in frameworks}
    agg_total: dict[str, int] = {f: 0 for f in frameworks}
    agg_per_pair_wins: dict[tuple[str, str], int] = {}
    agg_per_pair_total: dict[tuple[str, str], int] = {}
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
        unique_outputs = set(outputs_for_task.values())
        if len(unique_outputs) <= 1:
            continue
        tasks_used += 1
        ranking = await pairwise_rank(cells, judge_client, task)
        for entry in ranking.ranking:
            fw = entry["framework"]
            agg_wins[fw] += entry.get("wins", 0)
            agg_total[fw] += entry.get("total_comparisons", 0)
        for comp in ranking.pairwise_comparisons:
            if comp["winner"] is None:      # undecided — no win signal
                continue
            a, b = comp["pair"]
            for key in [(a, b), (b, a)]:
                agg_per_pair_total[key] = agg_per_pair_total.get(key, 0) + (1 if key == (a, b) else 0)
            if comp["winner"] == a:
                agg_per_pair_wins[(a, b)] = agg_per_pair_wins.get((a, b), 0) + 1
        agg_comparisons.extend(ranking.pairwise_comparisons)
        total_fallback += ranking.judge_fallback_count

    if tasks_used == 0:
        task = tasks[0]
        cells = [(fw, task.id, sut_outputs[fw][task.id].raw_text)
                 for fw in frameworks if task.id in sut_outputs.get(fw, {})
                 and isinstance(sut_outputs[fw][task.id].raw_text, str)
                 and not sut_outputs[fw][task.id].raw_text.startswith("ERROR:")]
        if len(cells) < 2:
            return None
        return await pairwise_rank(cells, judge_client, task)

    print(f"[pairwise] {tasks_used} diverging tasks, {len(agg_comparisons)} comparisons in {time.time()-_t_pw:.1f}s", flush=True)

    # Global BTL MLE fit from aggregated per-pair matrices (scipy L-BFGS-B).
    from agent_eval_gate.judge.pairwise_rank import _btl_score
    btl_scores, _hessian_cis, btl_fallback = _btl_score(
        agg_wins, agg_total, agg_per_pair_wins, agg_per_pair_total
    )

    # Confidence intervals via bootstrap over the decided pairwise outcomes. The
    # numerical-Hessian CI degenerates to [0,1] on well-separated rankings, so we
    # resample the comparisons with replacement, refit BTL each time, and take
    # 2.5/97.5 percentiles per SUT — a real, informative interval.
    import random as _random
    decided = [c for c in agg_comparisons if c.get("winner") is not None]
    _boot: dict[str, list[float]] = {f: [] for f in frameworks}
    _t_boot = time.time()
    try:
        if len(decided) >= 8:
            _rng = _random.Random(1234)
            _n = len(decided)
            for _ in range(150):
                bw = {f: 0.0 for f in frameworks}
                bt = {f: 0 for f in frameworks}
                bpw: dict[tuple[str, str], int] = {}
                bpt: dict[tuple[str, str], int] = {}
                for _i in range(_n):
                    comp = decided[_rng.randrange(_n)]
                    a, b = comp["pair"]
                    w = comp["winner"]
                    bt[a] += 1; bt[b] += 1; bw[w] += 1
                    for key in [(a, b), (b, a)]:
                        bpt[key] = bpt.get(key, 0) + (1 if key == (a, b) else 0)
                    if w == a:
                        bpw[(a, b)] = bpw.get((a, b), 0) + 1
                # with_ci=False skips the per-fit Hessian; small max_iter keeps each
                # refit bounded even on (near-)separable resamples.
                bs, _bc, _bf = _btl_score(bw, bt, bpw, bpt, with_ci=False, max_iter=50)
                for fw in frameworks:
                    if bt.get(fw, 0) > 0:
                        _boot[fw].append(bs.get(fw, 0.0))
    except Exception:
        _boot = {f: [] for f in frameworks}  # any failure → point-estimate CI below
    print(f"[pairwise] bootstrap CI ({sum(len(v) for v in _boot.values()) and 150} resamples) in {time.time()-_t_boot:.1f}s", flush=True)

    def _ci(fw: str) -> list[float]:
        vals = sorted(_boot.get(fw, []))
        if len(vals) >= 20:
            lo = vals[int(0.025 * (len(vals) - 1))]
            hi = vals[int(0.975 * (len(vals) - 1))]
            return [round(lo, 4), round(hi, 4)]
        s = round(btl_scores.get(fw, 0.0), 4)   # too few decided pairs to bootstrap
        return [s, s]

    # B-3: SUTs with 0 comparisons (did not run) get no btl_score.
    final_entries = []
    ranked_fw = sorted(
        [f for f in frameworks if agg_total.get(f, 0) > 0],
        key=lambda f: btl_scores.get(f, 0.0), reverse=True
    )
    for r, fw in enumerate(ranked_fw, 1):
        final_entries.append({
            "rank": r, "framework": fw, "btl_score": btl_scores.get(fw, 0.0),
            "confidence_interval": _ci(fw),
            "wins": agg_wins.get(fw, 0), "total_comparisons": agg_total.get(fw, 0),
            "status": "ran",
        })
    # Append did_not_run SUTs at the end with status marker.
    for fw in frameworks:
        if agg_total.get(fw, 0) == 0:
            final_entries.append({
                "rank": len(ranked_fw) + 1, "framework": fw, "btl_score": None,
                "wins": 0, "total_comparisons": 0, "status": "did_not_run",
            })

    return OrdinalRanking(
        ranking=final_entries,
        pairwise_comparisons=agg_comparisons,
        total_pairs=len(agg_comparisons),
        judge_fallback_count=total_fallback,
        btl_fallback_used=btl_fallback,
    )


async def run_gate(config: EvalConfig, baseline_path: Optional[str] = None, selected_frameworks: Optional[list[str]] = None) -> GateReport:
    _stage("run_gate: start")
    tasks = load_task_set()
    tasks_bytes = "\n".join(t.prompt for t in tasks).encode("utf-8")

    # Run SUTs
    _stage("run_suts: start")
    sut_outputs = await run_suts(config, tasks, selected_frameworks)
    _stage("run_suts: done")

    # Judge
    _stage("judge_outputs: start")
    verdicts = await judge_outputs(config, tasks, sut_outputs)
    _stage("judge_outputs: done")

    # Pairwise
    _stage("compute_pairwise: start")
    ranking = await compute_pairwise_ranking(config, tasks, sut_outputs, verdicts)
    _stage("compute_pairwise: done")

    # Build meta
    _stage("build_meta: start")
    sut_pins = {}
    for fw in sut_outputs:
        if sut_outputs[fw]:
            first_task_id = list(sut_outputs[fw].keys())[0]
            sut_pins[fw] = sut_outputs[fw][first_task_id].model

    framework_versions = _probe_framework_versions(list(sut_outputs.keys()))
    meta = build_run_meta(tasks_bytes, sut_pins, config.get_judge_client().model, framework_versions)
    _stage("build_meta: done")

    # Drift
    _stage("drift: start")
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
    _stage("drift: done")
    _stage("setup_langsmith: start")
    setup_langsmith(config.tracing.langsmith_api_key or None)
    _stage("setup_langsmith: done")
    _stage("setup_langfuse: start")
    setup_langfuse(config.tracing.langfuse_public_key or None, config.tracing.langfuse_secret_key or None)
    _stage("setup_langfuse: done")
    _stage("setup_phoenix: start")
    setup_phoenix(config.tracing.phoenix_endpoint or None)
    _stage("setup_phoenix: done")

    # Export
    _stage("export_to_inspect: start")
    report._inspect_samples = export_to_inspect(report)  # type: ignore[attr-defined]
    _stage("export_to_inspect: done")

    _stage("run_gate: return")
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
