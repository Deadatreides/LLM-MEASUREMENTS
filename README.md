# Measurements from a year of local LLM work

A working lab notebook, published as-is: **232 reports and 548 harness scripts**
from ~70 experiments run on one desktop machine over a year.

Most of what is here is **negative results**. That is not an apology — it is the
reason to publish. Hypotheses that died under measurement are the expensive part
of this work, and almost nobody publishes them.

**The hardware is the point, not a limitation.** Everything below was measured
on a Ryzen 5 1600 (Zen 1), a **GTX 1660 SUPER with 6 GiB**, 24 GB of RAM and a
SATA SSD. No cluster, no A100, no cloud credits. If you are running models on a
machine like that, these numbers transfer to you directly.

---

## Findings you can use today

Every row below names the file that produced it. If a claim here is not in that
file, it is a bug — open an issue.

| finding | what was measured | where |
|---|---|---|
| **`llama-bench` feeds random tokens** | input is generated with `std::rand() % n_vocab` — and on Windows `RAND_MAX` is 32 767 against a vocabulary of 201 088. Anything routing-dependent (MoE expert selection, cache behaviour) cannot be measured with it; use `llama-server`. This invalidates a whole category of benchmark results, including several of my own. | [`hardware/REVISION-CLUSTERS-HILBERT-GLASS.md`](hardware/REVISION-CLUSTERS-HILBERT-GLASS.md) §6, [`hardware/HANDOFF-2026-09-15.md`](hardware/HANDOFF-2026-09-15.md) |
| **Batching does not rescue MoE on CPU** | ×1.23 total at batch 8 — a batch of 8 tokens cost about 6.5 single tokens, so it does not pay for itself. | [`experiments/swebench_pro/reports/REPORT_SWE1_PILOT.md`](experiments/swebench_pro/reports/REPORT_SWE1_PILOT.md) |
| **Quality follows a power law in depth, with no cliff** | `PPL ~ d^−2.18`, i.e. `PPL(d) = PPL_full · (d_full/d)^2.18`. An earlier "cliff" I reported turned out to be an artefact of channel ordering, and the conclusion that depended on it was withdrawn. | [`hardware/AUDIT-GPTOSS-PPL.md`](hardware/AUDIT-GPTOSS-PPL.md) |
| **KV at `q4_0` costs 10.7 % perplexity, not 1.5 %** | a number I had published earlier was wrong by 7×, and this is the correction. Kept here deliberately. | [`hardware/PHASE0-RESULTS.md`](hardware/PHASE0-RESULTS.md) |
| **A swarm of weak models did not beat the best single model** | `E_score = −0.069 ± 0.027` over 30 tasks. The candidate pool of best-of-N was stronger *before* any selection was applied. | [`experiments/mycelium/lab/REPORT.md`](experiments/mycelium/lab/REPORT.md) |

---

## Layout

```
experiments/     ~70 directories, one per experiment
  <name>/
    README.md      what this experiment is
    PROTOCOL.md    conditions and method, written before the run
    BLOCKERS.md    what stopped it, where it stopped
    reports/       results
    *.py           the harness that produced them
hardware/        measurements about the machine itself: throughput,
                 KV cache economics, MoE offload, build costs
```

Notable groups:

* **`experiments/mycelium/`** — an attempt to build a system from a pool of weak
  local models that beats the best single model in the pool. It does not work,
  and `AUDIT_RESTORE.md` is the post-mortem of why.
* **`experiments/theory_emergent_swarm/`** — the theory that followed, tested
  against LongBench v2 rather than against itself.
* **`experiments/arch0..arch2/`** — architecture cycles: dependency model,
  complexity budgets, a small combinator language and its runner.
* **`experiments/agent_life*/`, `agent_a*/`** — selection, retention and budget
  experiments over recorded runs.

## What is not here

**Raw run artefacts.** They are roughly 16 GB of `.jsonl` and `.json` across
these experiments, and putting them in git would make the repository unusable
for the 10 MB of text that is actually worth reading. Each experiment keeps its
aggregate metrics and its report; if you want the raw runs for a specific
experiment to re-derive a number, open an issue and say which one.

## Honesty notes

* Several conclusions in these reports were later **retracted by better
  measurement**, and the retractions are left in place rather than edited out.
  Where a report was superseded, the newer one says so.
* Paths that pointed at the author's machine were replaced with `<PROJECT_ROOT>`,
  `<HOME>` and `<MODELS>`. Nothing else was edited for publication.
* Single machine, single operator. Runs drift about 6 % between sessions, which
  is why comparisons here are interleaved rather than run in series.

## Licence

Reports and text: CC BY 4.0. Harness code: MIT. Use the numbers, cite the
directory.
