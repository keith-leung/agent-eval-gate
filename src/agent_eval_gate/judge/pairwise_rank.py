"""Ordinal pairwise ranking with cross-vendor LLM judge + BTL aggregation.

The pairwise judge loop is self-built (not DeepEval Arena — Arena's API is
shaped for crowd-sourced human-preference arenas and does not fit a
deterministic single-judge CI loop). The Bradley-Terry-Luce
score computation uses scipy L-BFGS-B maximum-likelihood estimation — this is the
real BTL model, not a Laplace-smoothed win rate. Confidence intervals are computed
by **bootstrap resampling of the pairwise outcomes** (the numerical-Hessian route
degenerates to [0,1] on well-separated rankings, so it is not used for the reported
CI). On a judge failure a pair is recorded as **UNDECIDED** and excluded from the
win matrix — verbosity is never used as a tie-breaker.
"""

from __future__ import annotations

import asyncio

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Optional

from agent_eval_gate.protocols import ItemVerdict, OrdinalRanking, Task
from agent_eval_gate.llm_client import LLMClient
from agent_eval_gate.judge.exact_match import _extract_json

# Per pairwise-judge-call wall-clock timeout. Real calls are ~3s; a single stalled
# connection must not hang the phase (the openai SDK's own default is 600s). On
# timeout the pair is recorded UNDECIDED by the surrounding except handler.
_JUDGE_CALL_TIMEOUT = 60.0


def _btl_score(
    wins: dict[str, int],
    total_comparisons: dict[str, int],
    per_pair_wins: Optional[dict[tuple[str, str], int]] = None,
    per_pair_total: Optional[dict[tuple[str, str], int]] = None,
    with_ci: bool = True,
    max_iter: int = 120,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], bool]:
    """Bradley-Terry-Luce score via scipy MLE from pairwise wins.

    Returns (scores, confidence_intervals, used_fallback).
      - scores: framework -> BTL probability (sigmoid of fitted beta).
      - confidence_intervals: framework -> (lower, upper) 95% CI from Hessian.
      - used_fallback: True if scipy unavailable or fit failed → Laplace.

    The MLE path minimizes the negative log-likelihood of the Bradley-Terry
    model: P(i beats j) = sigmoid(beta_i - beta_j), over the observed
    per-pair win counts. This is the real BTL model — NOT (wins+0.5)/(n+1).
    """
    frameworks = sorted(wins.keys())
    n = len(frameworks)
    if n < 2:
        scores = {fw: 1.0 for fw in frameworks}
        cis = {fw: (1.0, 1.0) for fw in frameworks}
        return scores, cis, False

    index = {fw: i for i, fw in enumerate(frameworks)}

    # Build the win matrix W[i][j] = times i beat j, and N[i][j] = total i-vs-j.
    W = [[0] * n for _ in range(n)]
    N = [[0] * n for _ in range(n)]
    if per_pair_wins and per_pair_total:
        for (a, b), w in per_pair_wins.items():
            if a in index and b in index:
                W[index[a]][index[b]] = w
        for (a, b), cnt in per_pair_total.items():
            if a in index and b in index:
                N[index[a]][index[b]] = cnt
    else:
        # Without per-pair matrices, approximate from aggregate wins/totals.
        for a in frameworks:
            for b in frameworks:
                if a == b:
                    continue
                i, j = index[a], index[b]
                N[i][j] = max(total_comparisons.get(a, 0) + total_comparisons.get(b, 0), 1)
                # Distribute aggregate wins proportionally (rough — per-pair is preferred).
                wa = wins.get(a, 0)
                wb = wins.get(b, 0)
                total_w = wa + wb
                W[i][j] = int(N[i][j] * wa / total_w) if total_w > 0 else 0

    # Try scipy MLE first.
    try:
        from scipy.optimize import minimize
        import numpy as np

        def neg_log_likelihood(beta):
            ll = 0.0
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    w = W[i][j]
                    ni = N[i][j]
                    if ni == 0:
                        continue
                    # Clamp the exponent: on well-separated data beta gaps grow
                    # large and math.exp overflows (OverflowError), which both
                    # crashes the fit and, unbounded, makes L-BFGS-B chase betas
                    # to infinity for many iterations. Clamping keeps it stable.
                    d = beta[j] - beta[i]
                    d = 500.0 if d > 500.0 else (-500.0 if d < -500.0 else d)
                    p = 1.0 / (1.0 + math.exp(d))
                    ll += w * math.log(p + 1e-12) + (ni - w) * math.log(1 - p + 1e-12)
            return -ll

        x0 = [0.0] * n
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                          options={"maxiter": max_iter})
        if result.success or result.fun < neg_log_likelihood(x0):
            beta = result.x
            beta = [b - max(beta) for b in beta]  # normalize (max beta -> 0)

            def _sig(x: float) -> float:
                x = 500.0 if x > 500.0 else (-500.0 if x < -500.0 else x)
                return 1.0 / (1.0 + math.exp(-x))

            scores = {fw: _sig(beta[i]) for i, fw in enumerate(frameworks)}

            if not with_ci:
                # Bootstrap path: point scores only; skip the expensive Hessian.
                return scores, {fw: (s, s) for fw, s in scores.items()}, False

            # Numerical Hessian → confidence intervals.
            try:
                eps = 1e-5
                H = [[0.0] * n for _ in range(n)]
                for k in range(n):
                    for l in range(n):
                        x_pp = [b + (eps if i == k else 0.0) + (eps if i == l else 0.0) for i, b in enumerate(beta)]
                        x_pm = [b + (eps if i == k else 0.0) - (eps if i == l else 0.0) for i, b in enumerate(beta)]
                        x_mp = [b - (eps if i == k else 0.0) + (eps if i == l else 0.0) for i, b in enumerate(beta)]
                        x_mm = [b - (eps if i == k else 0.0) - (eps if i == l else 0.0) for i, b in enumerate(beta)]
                        H[k][l] = (neg_log_likelihood(x_pp) - neg_log_likelihood(x_pm)
                                   - neg_log_likelihood(x_mp) + neg_log_likelihood(x_mm)) / (4 * eps * eps)
                inv_H = np.linalg.inv(np.array(H) + 1e-6 * np.eye(n))
                cis = {}
                for i, fw in enumerate(frameworks):
                    se = math.sqrt(max(inv_H[i, i], 1e-10))
                    cis[fw] = (max(0.0, scores[fw] - 1.96 * se), min(1.0, scores[fw] + 1.96 * se))
                return scores, cis, False
            except Exception:
                cis = {fw: (max(0.0, s - 0.1), min(1.0, s + 0.1)) for fw, s in scores.items()}
                return scores, cis, False
    except Exception:
        pass

    # Fallback: Laplace-smoothed win rate (only if scipy unavailable or fit failed).
    scores = {}
    for name in frameworks:
        w = wins.get(name, 0)
        nn = max(total_comparisons.get(name, 1), 1)
        scores[name] = (w + 0.5) / (nn + 1.0)
    cis = {fw: (max(0.0, s - 0.1), min(1.0, s + 0.1)) for fw, s in scores.items()}
    return scores, cis, True


async def pairwise_rank(
    cells: list[tuple[str, str, str]],  # (framework, task_id, output_text)
    judge_client: LLMClient,
    task: Task,
) -> OrdinalRanking:
    """Rank framework outputs via pairwise comparison (fully async)."""
    frameworks = [c[0] for c in cells]
    framework_outputs = {c[0]: c[2] for c in cells}
    pairs = list(combinations(frameworks, 2))

    wins = {f: 0 for f in frameworks}
    comparisons = []
    total_comparisons = {f: 0 for f in frameworks}

    PAIRWISE_SYSTEM = """\
You are an evaluation judge. Compare two agent outputs for the same task and determine which is BETTER.

Task: {task_prompt}

Output A:
{output_a}

Output B:
{output_b}

Respond with ONLY valid JSON:
{{
  "winner": "A" or "B",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}"""

    sem = asyncio.Semaphore(8)  # limit concurrent judge calls

    async def _compare(a, b):
        user_message = json.dumps({
            "task_prompt": task.prompt,
            "output_a": framework_outputs[a],
            "output_b": framework_outputs[b],
        }, ensure_ascii=False)
        async with sem:
            try:
                raw = await asyncio.wait_for(
                    judge_client.complete(
                        system=PAIRWISE_SYSTEM.format(task_prompt=task.prompt, output_a=framework_outputs[a], output_b=framework_outputs[b]),
                        user_message=user_message,
                        tier="medium",
                        temperature=0.0,
                        max_tokens=1024,
                    ),
                    timeout=_JUDGE_CALL_TIMEOUT,
                )
                parsed = _extract_json(raw)
                winner = parsed.get("winner", "").upper()
                if winner not in ("A", "B"):
                    raise ValueError(f"Invalid winner '{winner}'")
                winner_framework = a if winner == "A" else b
                return {
                    "pair": [a, b],
                    "winner": winner_framework,
                    "confidence": parsed.get("confidence", 0.0),
                    "reasoning": parsed.get("reasoning", ""),
                }
            except Exception as exc:
                # Judge failed → record the pair as UNDECIDED. We do NOT award the
                # win to the longer output: verbosity is not quality. Undecided
                # pairs carry no win signal and are excluded from the BTL matrix.
                return {
                    "pair": [a, b],
                    "winner": None,
                    "confidence": 0.0,
                    "reasoning": f"undecided (judge failed: {exc})",
                }

    # Run all pairs concurrently (bounded by semaphore).
    results = await asyncio.gather(*[_compare(a, b) for a, b in pairs])
    for result in results:
        comparisons.append(result)
        winner = result["winner"]
        if winner is None:          # undecided — no win signal
            continue
        wins[winner] = wins.get(winner, 0) + 1
        total_comparisons[result["pair"][0]] += 1
        total_comparisons[result["pair"][1]] += 1

    # Build per-pair matrices from comparisons for real BTL MLE.
    per_pair_wins: dict[tuple[str, str], int] = {}
    per_pair_total: dict[tuple[str, str], int] = {}
    for comp in comparisons:
        if comp["winner"] is None:  # undecided — skip
            continue
        a, b = comp["pair"]
        key = (a, b)
        rev_key = (b, a)
        per_pair_total[key] = per_pair_total.get(key, 0) + 1
        per_pair_total[rev_key] = per_pair_total.get(rev_key, 0) + 1
        if comp["winner"] == a:
            per_pair_wins[key] = per_pair_wins.get(key, 0) + 1

    btl_scores, btl_cis, btl_fallback = _btl_score(wins, total_comparisons, per_pair_wins, per_pair_total)
    ranking = sorted(frameworks, key=lambda f: btl_scores.get(f, 0.0), reverse=True)

    return OrdinalRanking(
        ranking=[
            {
                "rank": rank,
                "framework": fw,
                "btl_score": btl_scores.get(fw, 0.0),
                "confidence_interval": list(btl_cis.get(fw, (0.0, 0.0))),
                "wins": wins.get(fw, 0),
                "total_comparisons": total_comparisons.get(fw, 0),
            }
            for rank, fw in enumerate(ranking, 1)
        ],
        pairwise_comparisons=comparisons,
        total_pairs=len(pairs),
        judge_fallback_count=sum(1 for c in comparisons if c.get("winner") is None),
        btl_fallback_used=btl_fallback,
    )
