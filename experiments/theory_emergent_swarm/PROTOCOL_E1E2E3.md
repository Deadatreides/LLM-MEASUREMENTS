# PROTOCOL — E1 (F2 FILTER), E2 (T_hard consensus+det-SUM), E3 (ρ_FILTER/ρ_SUM)

Written before any live call for this batch. Thresholds are copied
verbatim from `ESW_THEORY.md` §7 (already committed before this
PROTOCOL existed) — never re-derived, never moved.

```
E1  F2 FILTER, consensus exact-set rate     predicted 0.20-0.50 vs baseline 0.025   falsify if < 0.10
E2  T_hard consensus-FILTER + det-SUM, Δ    predicted >= 0.25  vs r_m*_B = 0.000    falsify if Δ < 0.05
E3  rho_FILTER < 0.4 (variance), rho_SUM > 0.8 (bias)                                cost 0, derived from E1/E2 + FORK-1 disk data
```

Baselines quoted for comparison only (read-only, from
`agent_delta0_new_grid/metrics/delta.json` and
`agent_fork1_three_paths/metrics/path_b_delta.json`): `r_∪_LLM_F2 = 0.025`,
`r_m*_B = 0.000`.

## 1. Mechanism (same for E1 and E2 — this is the "consensus-FILTER"
   referenced by both)

Both F2 and T_hard's FILTER atom asks a model for ONE thing: the list of
matching record ids, in a single call (same prompt shape DELTA-0/FORK-1
already used for their own single-executor FILTER atom — this package's
only change is querying **all 6 registry models**, not one, per task).
"Field decomposition" happens entirely in **post-processing** of that one
response via `oracles.extract_ids` — never as a separate call per
candidate id (that would cost K calls/model/task, ~15x over budget, and
is NOT what "whole-экстракты" in the request refers to).

**Candidate-id universe** for task `t`: all ids in `t["records"]` (K=18
for F2, K_HARD=14 for T_hard) — not just ids some model happened to
mention. A model's silence on an id is an implicit NO vote.

**Per-model reliability** `π_m`: balanced accuracy `(TPR_m + TNR_m)/2`,
NOT raw accuracy. Self-defined choice, reason: ids are positional/
task-local (unlike HETEROSTEP's `parcels`/`tariff`, which are the same
semantic field across every task), so there is no cross-task "this id"
identity to calibrate on directly — instead π_m is pooled over every
`(task, candidate_id)` pair the model voted on. Raw accuracy would let a
trivial always-NO model look strong under the ~75-89% true-negative base
rate (matched count is always 2-5 of K=14-18); balanced accuracy does not.

**Calibration = leave-one-out across the same N_TEST=40** (per family,
never pooled across F2/T_hard — different difficulty, different
generator). For task `T`, `π_m` is estimated from the OTHER 39 test
tasks' golden labels only — `T`'s own label is never used to weight the
vote that scores `T`. Self-defined choice, reason: maximizes
comparability with DELTA-0/FORK-1's existing N=40 baselines (no separate
calibration split shrinks the already-small N), while remaining
leak-free per left-out task (standard LOO-CV property).

**Condorcet-admissible set** `A(T) = {m : π_m(LOO excl. T) > 0.5}`.

**Per-id consensus vote**: majority among `A(T)`. Tie (only possible at
even `|A(T)|`, e.g. 2-2): defer to the single admitted model with the
highest LOO-π for `T`. `A(T) = ∅`: every id defaults to NO (consensus
id-set = ∅) — direct application of the arch1 PASS/FAIL/INAPPLICABLE
trichotomy (the swarm abstains rather than guesses); scored as a FAIL
under the exact-set checkpoint, **never** golden-substituted (DELTA-0/
FORK-1 chain-break rule, reapplied).

**Consensus id-set** = `{id : consensus vote = YES}`.

## 2. E1 scoring

`oracles.check_filter(", ".join(sorted(consensus_ids)), task["matched_ids"])`
per test task → `v ∈ {0,1}`. `r_E1 = mean(v)` over the 40 F2 test tasks.
Compare to the thresholds above. `Δ_E1 = r_E1 - 0.025`.

## 3. E2 scoring

Same mechanism on T_hard's own 40 test tasks (different generator,
different id prefix, different prompt wording — `atoms_hard.
build_filter_prompt`). Consensus id-set →
`det_atoms.aggregate_det(consensus_ids, "SUM", task["id_to_amount"])`
(always deterministic — SUM is **never** asked of an LLM anywhere in
this package, FORK-1's own locked rule) →
`oracles.within_tolerance(sum_computed, task["final_oracle"])` → `v`.
`r_E2 = mean(v)`. `Δ_E2 = r_E2 - r_m*_B (0.000)`. Compare to thresholds.

Chain-break discipline (DELTA-0/FORK-1, reapplied unchanged): this
package's consensus mechanism has no intermediate chain to break — FILTER
and SUM are the only two steps and SUM is deterministic — so the only
failure mode is FILTER's own consensus id-set being wrong, which SUM then
faithfully propagates (never golden-substituted).

## 4. E3 (0 new calls)

`ρ_FILTER`: computed on E1's (F2) and E2's (T_hard) own newly-collected
per-model per-id correctness matrix, same formula as `rho_holdout.py`
(`ρ = [P(both wrong) − P(wrong)²] / [P(wrong)(1−P(wrong))]`), reported
**separately per family** (`ρ_FILTER_F2`, `ρ_FILTER_Thard`) since
`ESW_THEORY.md` §7's single symbol `ρ_FILTER` does not specify which —
both are tested against the `<0.4` bar, not silently collapsed to one.

`ρ_SUM`: reused from FORK-1's own already-collected whole-task numeric
answers (`agent_fork1_three_paths/metrics/path_b_whole_detail.json`,
read-only, 0 new calls) — per-model per-task correctness of the DIRECT
(non-decomposed) numeric final answer, same ρ formula. This is bias-type
error (FORK-1 forensics: 0/155 correct despite correct format on ALL 6
models) — expected `ρ_SUM` high, near-degenerate if `P(wrong)≈1` for a
model (denominator → 0, reported as `nan`/flagged, not silently dropped).

## 5. Fixed constants (before any call)

```
TEMPERATURE = 0.0
SEED_BASE_ESW_FILTER = 97000       # stable_hash(task_id, "FILTER", model_id) + this
MAX_TOKENS_FILTER    = 220          # same allowance DELTA-0/FORK-1 used for FILTER
```

## 6. Gate before budget

Smoke test on the first 3 test tasks × 6 models (18 calls) per family
before the full 40×6=240-call campaign per family (480 total). Verifies:
no crashes, `extract_ids` parses non-trivially, LOO-reliability code runs
without divide-by-zero on the tiny slice. Abort and report if broken —
do not spend the full budget on unverified code.

## 7. Isolation

Writes only to `theory_emergent_swarm/`. Reads `agent_delta0_new_grid/
metrics/delta.json`, `agent_fork1_three_paths/metrics/path_b_delta.json`,
`agent_fork1_three_paths/metrics/path_b_whole_detail.json` read-only
(reference constants + E3's ρ_SUM data). Loads `experiment11/configs/
model_registry.py` by direct file path (sys.path-collision lesson).
Never writes to any `agent_*` package or `experiment*/`.
