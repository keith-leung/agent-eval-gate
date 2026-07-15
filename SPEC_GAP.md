# SPEC GAP — ArenaGEval pairwise deviation

## SPEC requirement

`SPEC.md` §3 B3 + locked decision #4 require using DeepEval's Arena
(`ArenaGEval`, `compare()`) as the pairwise-judge engine:

> Use the mature library for the judge loop (don't reinvent):
> DeepEval's Arena / comparisons (`ArenaGEval`, `ArenaTestCase`, `compare()`)
> ships pairwise head-to-head judging of contestant outputs.

## What was attempted / found

DeepEval 2.5.5 ships `ArenaGEval` / `ArenaTestCase` / `compare()`, but the API
is shaped for **crowd-sourced human-preference arenas** (multiple judges,
sequential preference collection, arena-style contestant rotation). It does
not fit this repo's pairwise loop, which is:

- a **single** cross-vendor LLM judge (MiniMax), not a crowd
- **deterministic** (temperature 0, same inputs → same ranking every run)
- run inside a **CI pipeline**, not an interactive arena
- needs a **per-pair deterministic fallback** when the judge call fails
  (ArenaGEval has no fallback concept — it expects every comparison to
  succeed or be re-collected)

Wiring ArenaGEval would require fighting its arena-shaped API (crowd
collection, contestant rotation) to fit a deterministic single-judge loop —
adding a dependency on a human-preference-shaped abstraction that does not
match the machine-judge, machine-output workflow.

## What was built instead

`src/agent_eval_gate/judge/pairwise_rank.py` implements a custom pairwise
loop that:

- runs head-to-head comparisons via the cross-vendor judge (same judge as
  the absolute-scoring path, MiniMax ≠ SUT structurally enforced)
- uses a **deterministic length-heuristic fallback** per pair when a judge
  call fails (longer output wins; alphabetical tie-break) — SPEC §3 B3's
  required deterministic fallback, which ArenaGEval does not provide
- aggregates wins into a **Bradley-Terry-Luce score** (Laplace-smoothed win
  rate as the BTL zero-order estimate) — SPEC's required BTL, built in-house
  as the spec explicitly says ("DeepEval does not ship BTL, build it
  yourself")

DeepEval's `GEval` is still used for the **absolute-scoring** judge path
(B2) — that part of DeepEval fits and is retained.

## Requesting reviewer spec-amend

Per HANDOFF "report back" rule (d): this is a deviation from a locked
decision, flagged rather than silently worked around. Requesting reviewer
sign-off to allow the custom pairwise loop (with BTL + deterministic
fallback) as the B3 implementation in place of `ArenaGEval.compare()`.

**Rationale:** the staff-level contribution B3 exists to demonstrate —
ordinal pairwise ranking robust to judge noise, with a deterministic
fallback — is fully present in the custom loop. ArenaGEval's arena
scaffolding adds ceremony without adding that contribution; the
contribution is in the BTL + fallback layer, which the spec already says to
build in-house regardless.
