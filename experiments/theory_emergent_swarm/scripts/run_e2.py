"""run_e2.py — E2: T_hard consensus-FILTER + det-SUM on the full 40 test
tasks x 6 models (240 calls). SUM is ALWAYS deterministic (FORK-1's
locked rule, reapplied unchanged) -- never asked of an LLM anywhere in
this script. PROTOCOL_E1E2E3.md §3: Delta predicted >=0.25 vs
r_m*_B=0.000, falsify if Delta<0.05. Run only after smoke_consensus.py
passes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_hard as THARD   # noqa: E402
import atoms_hard as AH               # noqa: E402
import det_atoms as DA                # noqa: E402
import consensus_filter as CF         # noqa: E402
import oracles as OR                  # noqa: E402
import model_registry_11 as MR        # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = AGENT_DIR / "metrics"

R_M_STAR_B = 0.000   # agent_fork1_three_paths/metrics/path_b_delta.json, read-only reference
THR_PREDICT = 0.25
THR_FALSIFY = 0.05


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    d = THARD.build_tasks()
    tasks_by_id, test_ids = d["tasks"], d["test"]
    print(f"E2: T_hard test set n={len(test_ids)}, models={len(MR.MODEL_IDS)}, "
         f"calls={len(test_ids) * len(MR.MODEL_IDS)}", flush=True)

    t0 = time.time()
    votes = CF.collect_votes(MR.MODEL_IDS, tasks_by_id, test_ids, AH.build_filter_prompt,
                             AH.MAX_TOKENS_FILTER, family="E2-Thard")
    print(f"collect_votes done in {time.time() - t0:.1f}s", flush=True)

    cons = CF.run_consensus(tasks_by_id, test_ids, MR.MODEL_IDS, votes)

    per_task = {}
    n_pass = 0
    for tid in test_ids:
        c = cons[tid]
        task = tasks_by_id[tid]
        sum_computed = DA.aggregate_det(c["consensus_ids"], "SUM", task["id_to_amount"])
        passed = OR.within_tolerance(sum_computed, task["final_oracle"])
        per_task[tid] = {
            "v": int(passed), "consensus_ids": sorted(c["consensus_ids"]),
            "golden": task["matched_ids"], "sum_computed": sum_computed,
            "final_oracle": task["final_oracle"], "admitted": c["admitted"],
            "loo_rel": c["loo_rel"],
        }
        n_pass += int(passed)

    r_e2 = n_pass / len(test_ids)
    delta_e2 = r_e2 - R_M_STAR_B
    if delta_e2 < THR_FALSIFY:
        verdict = "FALSIFIED"
    elif delta_e2 >= THR_PREDICT:
        verdict = "CONFIRMED"
    else:
        verdict = "ABOVE_FLOOR_BELOW_PREDICTION"

    fork1_llm_swarm = delta_e2 >= 0.05   # FORK-1's own pre-registered basket, criteria written before this theory existed

    result = {
        "r_E2": r_e2, "n_pass": n_pass, "n": len(test_ids),
        "r_m_star_B": R_M_STAR_B, "delta_E2": delta_e2,
        "threshold_predict": THR_PREDICT, "threshold_falsify": THR_FALSIFY,
        "verdict": verdict, "fork1_basket_flips_to_FORK_LLM_SWARM": fork1_llm_swarm,
        "per_task": per_task,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "e2_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nr_E2 = {r_e2:.3f}  ({n_pass}/{len(test_ids)})", flush=True)
    print(f"Delta_E2 = {delta_e2:+.3f}  (r_m*_B = {R_M_STAR_B:.3f})", flush=True)
    print(f"verdict = {verdict}", flush=True)
    print(f"FORK-1 basket flips to FORK_LLM_SWARM: {fork1_llm_swarm}", flush=True)


if __name__ == "__main__":
    main()
