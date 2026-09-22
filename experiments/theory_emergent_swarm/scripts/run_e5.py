"""run_e5.py — E5: margin admission (pi > 0.60) + structural decoding,
verified OUT-OF-SAMPLE on the never-used TRAIN splits (F2: 80 tasks,
T_hard: 40 -- none touched by E1-E4).

Thresholds pre-registered in reports/REPORT_E4.md SS8, written BEFORE this
script ran, and repeated here verbatim. This is a genuine out-of-sample
test of E4's post-hoc hypothesis, not another sweep: PI_ADMIT is a fixed
constant, never a loop.

720 live calls (120 tasks x 6 models). Mechanism identical to E1/E2
(already smoke-verified); only the task-id list changes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_f2 as TF2       # noqa: E402
import task_generator_hard as THARD   # noqa: E402
import atoms_hard as AH               # noqa: E402
import consensus_filter as CF         # noqa: E402
import structure_decode as SD         # noqa: E402
import det_atoms as DA                # noqa: E402
import oracles as OR                  # noqa: E402
import model_registry_11 as MR        # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = AGENT_DIR / "metrics"

PI_ADMIT = 0.60          # FIXED, from REPORT_E4.md SS4 post-hoc plateau. Never swept here.
THR_F2_SET = 0.60        # pre-registered in REPORT_E4.md SS8
THR_HARD_SET = 0.50


def final_f2(task, ids) -> bool:
    agg = DA.aggregate_det(ids, task["op"], task["id_to_amount"])
    return DA.derive_det(agg, task["threshold"], task["comparator"]) == task["final_oracle"]


def final_hard(task, ids) -> bool:
    return OR.within_tolerance(DA.aggregate_det(ids, "SUM", task["id_to_amount"]),
                               task["final_oracle"])


def run_family(label, gen_mod, prompt_builder, max_tokens, final_fn, model_ids) -> dict:
    d = gen_mod.build_tasks()
    tasks_by_id, ids = d["tasks"], d["train"]     # TRAIN split -- never used in E1-E4
    print(f"[E5/{label}] out-of-sample n={len(ids)} calls={len(ids) * len(model_ids)}", flush=True)

    t0 = time.time()
    votes = CF.collect_votes(model_ids, tasks_by_id, ids, prompt_builder, max_tokens,
                             family=f"E5-{label}")
    print(f"[E5/{label}] collected in {time.time() - t0:.1f}s", flush=True)

    n = len(ids)
    swarm_set = swarm_fin = single_set = single_fin = 0
    per_model_set = {m: 0 for m in model_ids}
    adm_sizes = []
    per_task = {}

    for tid in ids:
        task = tasks_by_id[tid]
        golden = frozenset(task["matched_ids"])
        cids = CF.candidate_ids(task)
        classes = SD.enumerate_classes(task)
        rel = CF.loo_reliabilities(tasks_by_id, ids, model_ids, votes, tid)

        admitted = sorted(m for m in model_ids if rel[m] > PI_ADMIT)
        adm_sizes.append(len(admitted))
        sup = SD.swarm_support(task, votes[tid], admitted)
        _k, dec, _s = SD.decode(sup, cids, classes)
        swarm_set += int(dec == golden)
        swarm_fin += int(final_fn(task, dec))

        best_m = max(sorted(model_ids), key=lambda m: rel[m])
        _k2, sdec, _s2 = SD.decode(SD.single_support(task, votes[tid], best_m), cids, classes)
        single_set += int(sdec == golden)
        single_fin += int(final_fn(task, sdec))

        for m in model_ids:
            _k3, mdec, _s3 = SD.decode(SD.single_support(task, votes[tid], m), cids, classes)
            per_model_set[m] += int(mdec == golden)

        per_task[tid] = {"golden": sorted(golden), "decoded": sorted(dec),
                        "admitted": admitted, "best_single": best_m}

    return {
        "n": n, "pi_admit": PI_ADMIT, "mean_admitted": sum(adm_sizes) / n,
        "swarm_set": swarm_set / n, "swarm_final": swarm_fin / n,
        "single_honest_set": single_set / n, "single_honest_final": single_fin / n,
        "gain_set": (swarm_set - single_set) / n,
        "per_model_set": {m: c / n for m, c in per_model_set.items()},
        "oracle_best_model_set": max(per_model_set.values()) / n,
        "per_task": per_task,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model_ids = list(MR.MODEL_IDS)

    f2 = run_family("F2", TF2, TF2.build_filter_prompt, TF2.MAX_TOKENS_FILTER, final_f2, model_ids)
    hard = run_family("Thard", THARD, AH.build_filter_prompt, AH.MAX_TOKENS_FILTER, final_hard,
                      model_ids)

    gain_both_positive = f2["gain_set"] > 0 and hard["gain_set"] > 0
    set_bars_met = f2["swarm_set"] >= THR_F2_SET and hard["swarm_set"] >= THR_HARD_SET

    verdict_swarm = "SWARM_CONFIRMED_OOS" if gain_both_positive else "SWARM_REJECTED"
    verdict_thr = "THRESHOLD_HELD_OOS" if set_bars_met else "THRESHOLD_OVERFIT"

    result = {
        "STRUCTURE_DET": True, "split": "TRAIN (never used in E1-E4)",
        "pi_admit": PI_ADMIT,
        "thresholds": {"gain_both_positive": True, "F2_set": THR_F2_SET,
                      "Thard_set": THR_HARD_SET},
        "F2": f2, "T_hard": hard,
        "verdict_swarm": verdict_swarm, "verdict_threshold": verdict_thr,
        "tool_ceiling_filter_det": 1.000,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "e5_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    p = print
    p("\n=== E5: margin admission + decoding, OUT-OF-SAMPLE (train splits) ===\n")
    for lbl, f, bar in (("F2", f2, THR_F2_SET), ("T_hard", hard, THR_HARD_SET)):
        p(f"--- {lbl} (n={f['n']}, mean|A|={f['mean_admitted']:.1f}) ---")
        p(f"  swarm + decode      exact-set : {f['swarm_set']:.3f}   (bar {bar:.2f})")
        p(f"  best single + decode exact-set: {f['single_honest_set']:.3f}")
        p(f"  GAIN                          : {f['gain_set']:+.3f}")
        p(f"  swarm final                   : {f['swarm_final']:.3f}")
        p(f"  best single final             : {f['single_honest_final']:.3f}\n")
    p(f"verdict swarm     : {verdict_swarm}")
    p(f"verdict threshold : {verdict_thr}")


if __name__ == "__main__":
    main()
