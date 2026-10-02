# K1: an Albert Kahn design office as a swarm architecture — English summary

Translated from the Russian original, [`REPORT_K1.md`](REPORT_K1.md). The protocol was written before
any new call and is in [`../PROTOCOL_K1.md`](../PROTOCOL_K1.md) (Russian). If this summary and the
original disagree, the original wins.

The run made 720 live calls in 614 s. The full-context baseline was re-used from disk, with 0 new
calls. Thresholds were not moved.

## Setup

* **Task.** F2, test split, 40 tasks. Each task puts 18 records into the context, and the model must
  return the records that match the task's condition. Scoring:
  * **exact set** — the whole set must be right;
  * the task's final yes/no answer.
* **Models.** 6 local models, all 1–2B: qwen2.5-coder-1.5b, llama-3.2-1b, internvl3-2b, qwen3-1.7b,
  gemma-3-1b, smollm2-1.7b.
* **Pipeline:**
  1. A script, not an LLM, splits the context into 3 blocks of 6 records. An LLM planner scored 22 %
     in an earlier experiment (LIFE-9).
  2. Each of the 6 models reads each of the 3 blocks independently.
  3. **In-discipline check:** an id outside its own block is a hallucination and is dropped. Format
     is validated. Both checks are deterministic and make 0 calls.
  4. **Assembly:** votes for a record are taken only from that record's own block.
  5. Consensus and decoding, as in E4 (π > 0.60).
  6. A one-line yes/no consultation runs when the margin is low. It was not needed.

## 1. First — refuting my own claim

The pre-registered threshold `K1-emergence` was badly worded: "the assembled answer is correct, and no
single model solved the whole set on the full context". By that wording the result was **34 of 40**.
That number is an artifact: it compared the full pipeline (blocks + checks + consensus + decoding)
against a single model's *raw* extraction. The two sides ran under different conditions.

The correct comparison sends the single model through the **identical** pipeline:

| | |
|---|---|
| assembled pipeline, 6 models | **0.900** |
| best single model, same pipeline (`qwen3-1.7b`) | **0.850** |
| contribution of the swarm | **+0.050** |
| oracle "any one of the 6", same pipeline | 0.975 |
| **strict emergence** | **0** |

**There is no emergence.** The assembled answer always lies inside the oracle "at least one model got
it". The 34 is kept in `metrics/k1_result.json` next to the 0: the old value is not deleted, so the
wording defect stays visible.

This is the second emergence check that gives zero; G1 gave zero structurally. The only place
emergence was ever measured is E0, where assembly went by the *fields of one joint answer* (0.870
against an oracle of 0.840). Composition by *context blocks* does not produce it.

## 2. What held — and this is the main result

### 2.1 Positional recall decay removed

`probe_lost_middle.py` (0 GPU) ran on data already collected. Recall falls **monotonically** from
0.609 at position 0 to 0.214 at position 17. This is not "lost in the middle": there is no U-shaped
dip. Finding decays — the model says "no" to late records more and more often.

Split into 3 blocks of 6 records:

| | full context | Kahn blocks |
|---|---|---|
| recall, head (positions 0–5) | 0.508 | 0.595 |
| recall, tail (positions 12–17) | **0.304** | **0.574** |
| **head − tail gap** | **0.204** | **0.021** |

The gap shrank tenfold. The pre-registered threshold `K1-flat` was ≤ 0.12, and the result is 0.021. A
record's position stopped affecting whether it is found. Individual tail positions gained between
+0.30 and +0.375.

### 2.2 Decomposition helps WEAK models more than strong ones

Exact set after decoding, full context → blocks:

| model | full | blocks | gain |
|---|---|---|---|
| qwen2.5-coder-1.5b | 0.175 | **0.700** | **+0.525** |
| llama-3.2-1b | 0.400 | **0.725** | **+0.325** |
| internvl3-2b | 0.575 | 0.825 | +0.250 |
| qwen3-1.7b | 0.725 | 0.850 | +0.125 |
| gemma-3-1b | 0.000 | 0.050 | +0.050 |
| smollm2-1.7b | 0.000 | 0.000 | 0 |

Every model that can do the task at all improved, and **the weaker the model, the bigger the gain**.
This is Kahn's thesis in numbers: the right division of labour lifts mediocre workers to a level where
their work adds up. `qwen2.5-coder` went up fourfold and nearly caught the leader.

### 2.3 Task result

| F2 test, n = 40 | |
|---|---|
| exact set, full context + decoding (E4) | 0.750 |
| **exact set, blocks + assembly + decoding** | **0.900** |
| final yes/no | **1.000** (whole-task `r_m*` = 0.525) |
| hallucinations dropped by the in-discipline check (id outside its block) | 5 |

The pre-registered threshold `K1-set` was ≥ 0.85. The result is 0.900: `CONFIRMED`.

## 3. Which parts of the Kahn scheme actually carried load

| Kahn element | implemented as | measured contribution |
|---|---|---|
| division into disciplines (blocks) | deterministic, by schema | **main: +0.150 exact set, gap 0.204 → 0.021** |
| workers | 6 models per block | +0.050 over the best single model |
| in-discipline check | id outside block → drop | 5 hallucinations caught |
| norm control | format validation | 0 calls, no format errors |
| **cross-discipline check** | implemented | **degenerate: F2 blocks are independent, the seams have nothing to check** |
| chief-specialist consultation | E6 mechanism | not needed: 0.900 reached without it |
| 3 revisions | margin gating | not needed |

**Honestly: one Kahn discipline out of seven did the work — the split into blocks.** The rest either
degenerated on this task (cross-discipline check), were not needed (consultation, revisions), or gave
a small contribution (the swarm of workers, the in-discipline check). K1 cannot be presented as a
confirmation of the whole Kahn scheme.

## 4. Non-claims

* **No emergence** (strict measure = 0). The claim in §1 was refuted by me, on the same run.
* The cross-discipline check was **not tested in substance** on F2, because the blocks are
  independent. Testing it needs a task with dependencies between blocks; that is the next test bed.
* `STRUCTURE_DET = true`: decoding, the in-discipline check and norm control are deterministic. The
  deterministic filter's `filter_det = 1.000` was not beaten (K1: 0.900).
* The final yes/no of 1.000 next to an exact set of 0.900 reflects how soft a boolean answer is. The
  honest number is 0.900.
* Block size 6 was chosen from the measured recall profile, where positions 0–5 are the high-recall
  zone. It was **not tuned**: sizes 3 and 9 were not tried.
* 40 tasks, one task family, one split.

## 5. Where this leads

The last three batches add up to one statement:

> **The gain comes from decomposition — of the context (K1), of the answer structure (E4), of the
> question (E6). The pool of models adds 0 to +0.08 on top and never produces emergence.**

Emergence was measured exactly once: in E0, where assembly went by the FIELDS of one joint answer.
Neither a mode vote on the final number (G1) nor composition by context blocks (K1) reproduced it. If
it is reachable at all, it is only where **different models contribute different PARTS of one
answer** — not different opinions about the whole, and not different pieces of the input.

---

Terms used in the original:
* **ВДП** — in-discipline check;
* **МДП** — cross-discipline check;
* **нормоконтроль** — norm control, a standards check before anything is issued;
* **ГИП** — chief project engineer.

These are the standard checks of a Soviet/post-Soviet design institute.
