"""Ordinal pairwise ranking with DeepEval Arena + BTL aggregation."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

from agent_eval_gate.protocols import ItemVerdict, OrdinalRanking, Task
from agent_eval_gate.llm_client import LLMClient
from agent_eval_gate.judge.exact_match import _extract_json


def _btl_score(wins: dict[str, int], total_comparisons: dict[str, int]) -> dict[str, float]:
    """Bradley-Terry-Luce score from pairwise wins."""
    scores = {}
    for name in wins:
        w = wins.get(name, 0)
        n = max(total_comparisons.get(name, 1), 1)
        # Simplified BTL: win rate as score; add epsilon to avoid zeros.
        scores[name] = (w + 0.5) / (n + 1.0)
    return scores


def pairwise_rank(
    cells: list[tuple[str, str, str]],  # (framework, task_id, output_text)
    judge_client: LLMClient,
    task: Task,
) -> OrdinalRanking:
    """Rank framework outputs via pairwise comparison."""
    frameworks = [c[0] for c in cells]
    framework_outputs = {c[0]: c[2] for c in cells}
    pairs = list(combinations(frameworks, 2))

    wins = {f: 0 for f in frameworks}
    comparisons = []
    total_comparisons = {f: 0 for f in frameworks}
    concurrency = int(os.environ.get("LLM_CONCURRENCY", "3"))

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

    def _compare(pair_idx, a, b):
        user_message = json.dumps({
            "task_prompt": task.prompt,
            "output_a": framework_outputs[a],
            "output_b": framework_outputs[b],
        }, ensure_ascii=False)
        raw = None
        try:
            raw = judge_client.complete_sync(
                system=PAIRWISE_SYSTEM.format(task_prompt=task.prompt, output_a=framework_outputs[a], output_b=framework_outputs[b]),
                user_message=user_message,
                tier="medium",
                temperature=0.0,
                max_tokens=1024,
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
                "fallback": False,
            }
        except Exception as exc:
            # Deterministic fallback: longer output wins (heuristic)
            len_a = len(framework_outputs[a])
            len_b = len(framework_outputs[b])
            fb_winner = a if len_a >= len_b else b
            return {
                "pair": [a, b],
                "winner": fb_winner,
                "confidence": 0.0,
                "reasoning": f"Heuristic fallback (LLM failed: {exc})",
                "fallback": True,
            }

    with ThreadPoolExecutor(max_workers=min(concurrency, len(pairs))) as pool:
        futures = {
            pool.submit(_compare, idx, a, b): (idx, a, b)
            for idx, (a, b) in enumerate(pairs, 1)
        }
        for future in as_completed(futures):
            idx, a, b = futures[future]
            result = future.result()
            winner = result["winner"]
            wins[winner] = wins.get(winner, 0) + 1
            total_comparisons[a] += 1
            total_comparisons[b] += 1
            result.pop("fallback", None)
            comparisons.append(result)

    btl_scores = _btl_score(wins, total_comparisons)
    ranking = sorted(frameworks, key=lambda f: btl_scores.get(f, 0.0), reverse=True)

    return OrdinalRanking(
        ranking=[
            {
                "rank": rank,
                "framework": fw,
                "btl_score": btl_scores.get(fw, 0.0),
                "wins": wins.get(fw, 0),
                "total_comparisons": total_comparisons.get(fw, 0),
            }
            for rank, fw in enumerate(ranking, 1)
        ],
        pairwise_comparisons=comparisons,
        total_pairs=len(pairs),
    )
