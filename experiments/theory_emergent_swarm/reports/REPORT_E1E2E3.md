# REPORT — E1, E2, E3 (F2/T_hard consensus-FILTER live test of ESW theory)

Pre-registration: `PROTOCOL_E1E2E3.md` (written before any call). Thresholds
copied from `ESW_THEORY.md` §7, never moved. All numbers below are the
first and only run — no retries, no threshold adjustment after seeing
results.

## Verdicts

| # | prediction | threshold | observed | verdict |
|---|---|---|---|---|
| E1 | F2 FILTER consensus exact-set, 0.20-0.50 vs baseline 0.025 | falsify if <0.10 | **r=0.025, Δ=+0.000** | **FALSIFIED** |
| E2 | T_hard consensus-FILTER+det-SUM, Δ≥0.25 vs r_m*_B=0.000 | falsify if Δ<0.05 | **r=0.025, Δ=+0.025** | **FALSIFIED** |
| E3a | ρ_FILTER_F2 < 0.4 | | **+0.150** | confirmed |
| E3b | ρ_FILTER_Thard < 0.4 | | **+0.103** | confirmed |
| E3c | ρ_SUM > 0.8 | | **undefined** (P(wrong)=1.000 exactly, 0/240) | see §3 |

FORK-1's own pre-registered basket does **not** flip:
`FORK_LLM_SWARM` requires `Δ_B_LLM ≥ 0.05`; E2 delivered `+0.025`. Basket
stays `FORK_TOOL_ONLY`.

## 1. What was measured

Both experiments used the SAME mechanism (`src/consensus_filter.py`,
locked in `PROTOCOL_E1E2E3.md` before any call): one FILTER call per
(task, model) — 6 models × 40 test tasks = 240 calls per family, 480
total — extracted id-sets turned into per-candidate-id binary votes in
post-processing, per-model reliability estimated as leave-one-out
balanced accuracy, Condorcet-admissible set = `{m : π_m > 0.5}`,
per-id majority vote among admitted models.

Gate-before-budget cleared first: `scripts/smoke_consensus.py` (36 calls,
3 tasks × 6 models × 2 families) ran clean before either full campaign.

## 2. Diagnosis — the mechanism works; the checkpoint doesn't fit it

The Condorcet filter worked exactly as designed. On F2, it correctly
excluded `smollm2-1.7b` (TPR=0.000 — literally never finds a true
positive, TNR=0.978 — a textbook "always vote NO" degenerate model that
raw accuracy would have rated as merely mediocre, 0.215 error rate) and
kept `qwen3-1.7b` (balanced accuracy 0.806, the strongest individual
voter) alongside 3-4 others. This is the balanced-accuracy design
decision (`PROTOCOL_E1E2E3.md` §1) doing exactly its job.

**Per-id, consensus voting genuinely beats the best single model:**

| family | consensus per-id accuracy | best single model | gain |
|---|---|---|---|
| F2 (n=720 id-decisions) | **0.840** | qwen3-1.7b, 0.799 | **+0.042** |
| T_hard (n=560 id-decisions) | **0.782** | internvl3-2b, 0.780 | +0.002 |

This is the same qualitative effect as ESW-0 (condorcet beats best-single,
0.861 vs 0.783) — smaller in magnitude here, present on F2, negligible on
T_hard, but never negative. The swarm mechanism is not broken.

**What kills the task-level rate is conjunction width.** F2's exact-set
checkpoint and T_hard's SUM checkpoint both require EVERY one of K
candidate ids (K=18 for F2, K_HARD=14 for T_hard) to be voted correctly
in the SAME task — a K-way AND over binary decisions, not the 4-way AND
HETEROSTEP's `whole_grid` fields presented in ESW-0. Raising the measured
per-id accuracy to the K-th power predicts the task-level rate almost
exactly:

```
F2:     0.840^18 = 0.043   (observed 0.025 -- lower, consistent with
                             within-task rho_FILTER_F2=+0.150: errors on
                             different ids in the SAME task are not fully
                             independent, so the true joint is below the
                             naive-independence estimate)
T_hard: 0.782^14 = 0.032   (observed 0.025 -- same pattern, rho=+0.103)
```

Both land within ~1-2pp of the observed rate using nothing but the
measured per-id accuracy and K. **This is not a failed swarm — it is a
swarm whose real, positive, per-decision gain is being raised to a power
wide enough to erase it.** ESW-0's HETEROSTEP fields (K=4) sit in a
regime where this exponent is survivable (0.84^4≈0.50, order-of-magnitude
consistent with the measured 0.861 task-level rate once you account for
the fact HETEROSTEP's own per-field accuracies were higher and more
lopsided across fields than F2/T_hard's uniform ~0.78-0.84). F2/T_hard's
per-id decomposition (K=14-18) does not.

This refines `ESW_THEORY.md` §3.2's convergence bound: the bound is
correctly stated PER FIELD, but was implicitly read as transferring to
the task level whenever `A_f ≠ ∅` for enough fields — it does not, once
`|F|` (here, K candidate ids) grows past a handful. **Swarm Gain
`G_f = k_eff·γ_f²` governs each field in isolation; nothing in §3
addresses what happens to `P(all fields correct) = Π_f P_f` as `|F|`
grows — that product decays even when every individual `P_f` improves.**

## 3. E3 — ρ_SUM's undefined result is not a falsification

`rho_SUM` could not be computed: `P(wrong) = 1.000` exactly (0/240 — all
6 models wrong on all 40 T_hard whole-task numeric answers, matching
FORK-1's own forensics bit-for-bit: 0/155 single-line-wrong + 85
multi-line, 0 correct either way). The correlation formula's denominator
`P(wrong)(1-P(wrong))` is then exactly 0 — division by zero, reported as
`NaN`, not silently coerced to a number.

This is the **degenerate limiting case of the bias claim**, not evidence
against it: universal, deterministic failure across every model on every
task is a stronger form of "shared failure" than any finite `ρ<1` could
express. `run_e3.py`'s mechanical verdict label
(`BELOW_0.8_or_NAN_bias_claim_falsified`) is misleading here and should
be read with this paragraph, not on its own — the label conflates "the
sample formula is undefined" with "the claim is false"; disclosed here
rather than silently relabeled after the fact.

`ρ_FILTER_F2 = +0.150` and `ρ_FILTER_Thard = +0.103` both land
comfortably under the 0.4 bar — genuinely dispersed, low-correlation
errors, exactly the "variance, not bias" signature `ESW_THEORY.md` §3.4
predicted for a FILTER-type atom. **The bias/variance router was right
about WHICH regime each atom is in. It was silent on conjunction width,
which is what actually decided E1/E2's outcome.**

## 4. Cost

480 live calls total (240 F2 + 240 T_hard), plus 36 smoke calls = 516.
E3: 0 new calls (reused `metrics/filter_calls.jsonl` + FORK-1's
`path_b_whole_detail.json`, both already on disk).

## 5. Non-claims

- This does not retest E0 (HETEROSTEP, K=4 fields) — that result stands
  as measured, untouched by E1/E2's outcome on a structurally different
  checkpoint (K=14-18 fields).
- The conjunction-width diagnosis is a POST-HOC explanation of an
  observed result, offered with supporting arithmetic (the `p^K`
  match), not itself a pre-registered, falsifiable prediction — it should
  be treated as a hypothesis for a FUTURE experiment (e.g., measure
  task-level rate as a function of K directly), not as confirmed theory.
- Per-id accuracy (0.840, 0.782) is measured on the SAME 40 test tasks
  used for LOO calibration — standard leave-one-out property (no task
  scores itself), but still a single N=40 sample per family, not
  cross-validated against a wholly separate holdout the way ESW-0's
  50/50 split was.
