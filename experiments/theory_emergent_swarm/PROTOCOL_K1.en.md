# PROTOCOL — K1: Albert Kahn's design office as a swarm architecture (English version)

English translation of [`PROTOCOL_K1.md`](PROTOCOL_K1.md) (Russian original). This is a pre-registration
and is translated as written, mistakes included. If the two disagree, the original wins.

> **Translator's note, 2026-10-04.** The mechanism stated in §0 is wrong: "the model says 'no' to late
> records more and more often". A later check on the same data found that the yes rate is nearly flat by
> position; what decays is discrimination. See the correction in
> [`reports/REPORT_K1.en.md`](reports/REPORT_K1.en.md), §2.1. The recall decay itself stands.

Written BEFORE a single new call. The thresholds do not move.

## 0. Measured premise (0 GPU, `probe_lost_middle.py`)

Extraction recall falls **monotonically with the record's position in the context**: recall 0.609 at
position 0 → **0.214** at position 17. Threefold.

This is **not** "lost in the middle": there is no U-shaped dip, accuracy is 0.762 at the edges against
0.739 in the middle. It is a decay of finding: the model says "no" to late records more and more often,
which raises true negatives and collapses recall.

The direct consequence: **splitting a long context into blocks moves every record into the
high-recall zone.** 18 records → 3 blocks of 6 → every record sits at positions 0–5, where recall is
0.46–0.61 instead of 0.21–0.31 in the tail. This is a measured justification for decomposition, not a
hypothesis.

## 1. Mapping the Kahn scheme onto measured capabilities

| Kahn | component | what it is based on (measured) |
|---|---|---|
| chief project engineer / division into disciplines | **the task schema, not an LLM** | LIFE-9: LLM planner 22% |
| discipline | **context block** (6 records) | positional recall decay, §0 |
| worker | 1–6 models per block | Probe A: the pool is not differentiated → choose by measured π, not by "specialty" |
| chief specialist's consultation | **a point question about one line** | E6: an atom is easier than the whole, 1.000 for qwen3/internvl3 |
| in-discipline check | the id belongs to its own block + format | deterministic, 0 calls |
| cross-discipline check | consistency at the seams between blocks | deterministic, 0 calls |
| norm control | format validation before release | deterministic, 0 calls |
| assembling the drawing sets | **composition of blocks** | E0: composition by fields exceeded the oracle |
| 3 revisions | iterations while the margin is low | the margin signal reproduced 5 times |

**The key point, the reason for the whole construction:** composition is the only mechanism that can
produce emergence.
* A mode vote structurally cannot: G1 found 0 emergent tasks, and the ceiling is the "any one" oracle.
* Assembly from blocks can: if model A got block 1 and model B got block 2, the assembled answer is
  correct even though no one produced it whole.

## 2. Pipeline

```
SCHEMA -> 3 blocks of 6 records (deterministic, not an LLM)
   │
WORKERS: 6 models x 3 blocks, independently
   │
IN-DISCIPLINE CHECK: id outside its block -> drop (hallucination), format -> norm control
   │
ASSEMBLY: votes for a record are taken from ITS block -> support vector over all 18
   │
CONSENSUS + DECODING (as in E4, pi>0.60)
   │
CONSULTATION: if the margin is low -- a point yes/no on one line (E6)
   │
RELEASE
```

## 3. Pre-registered thresholds

| code | quantity | prediction | falsification |
|---|---|---|---|
| **K1-recall** | recall at original positions 12–17 (currently **0.28**) | ≥ 0.45 | **< 0.35** |
| K1-flat | recall gap between positions 0–5 and 12–17 (currently **0.32**) | ≤ 0.12 | > 0.20 |
| **K1-set** | exact set after assembly + decoding (E4 gives **0.750**) | ≥ 0.85 | < 0.78 |
| **K1-emergence** | tasks where the assembly is correct but NO model got the whole set on the full context | **≥ 3** | **0 → composition does not produce emergence** |

**K1-emergence is the main one.** It is the only threshold in the whole project that tests emergence
directly rather than through a proxy. G1 gave zero by construction; if composition also gives zero,
there is no emergence anywhere.

## 4. Budget

40 tasks of F2 TEST × 3 blocks × 6 models = **720 calls**. These are the same tasks that carry
`r_m*=0.525`, E1 and E4, so the comparison needs no transfer assumption.

A block is ~400 tokens in and ~40 out → ≈0.86 s per call → **≈10 minutes**. The baseline (full context)
is already on disk, 0 new calls.

The total token volume barely grows: 3 blocks of 1/3 of the context ≈ the same context, plus triple
prompt overhead. **Decomposition here is nearly free in tokens and should pay for itself in recall.**

## 5. Non-claims

- **The cross-discipline check is degenerate here.** Blocks in F2 are independent (records do not refer
  to each other), so there is nothing to check at the seams. The cross-discipline check carries real
  load only on tasks with dependencies between blocks. Here it is implemented, but its contribution is
  not measured. Do not present this as a test of the cross-discipline check.
- `STRUCTURE_DET = true`: decoding, the in-discipline check and norm control are deterministic.
  `filter_det = 1.000` is an unbeaten ceiling.
- **Record order is preserved.** Inside a block records keep their original order, and blocks are cut
  consecutively, without shuffling — otherwise measuring the positional effect would lose its meaning.
- 40 tasks, one family, one split.
