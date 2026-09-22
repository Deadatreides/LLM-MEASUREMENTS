"""run_e1.py — E1: F2 FILTER, consensus exact-set rate on the full 40
test tasks x 6 models (240 calls). PROTOCOL_E1E2E3.md §2: thresholds
predicted 0.20-0.50 vs baseline r_union_LLM_F2=0.025, falsify if <0.10.
Run only after smoke_consensus.py passes (gate-before-budget).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_f2 as TF2   # noqa: E402
import consensus_filter as CF     # noqa: E402
import oracles as OR              # noqa: E402
import model_registry_11 as MR    # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = AGENT_DIR / "metrics"

BASELINE_R_UNION_LLM_F2 = 0.025    # agent_delta0_new_grid/metrics/delta.json, read-only reference
THR_PREDICT_LOW, THR_PREDICT_HIGH = 0.20, 0.50
THR_FALSIFY = 0.10


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    d = TF2.build_tasks()
    tasks_by_id, test_ids = d["tasks"], d["test"]
    print(f"E1: F2 test set n={len(test_ids)}, models={len(MR.MODEL_IDS)}, "
         f"calls={len(test_ids) * len(MR.MODEL_IDS)}", flush=True)

    t0 = time.time()
    votes = CF.collect_votes(MR.MODEL_IDS, tasks_by_id, test_ids, TF2.build_filter_prompt,
                             TF2.MAX_TOKENS_FILTER, family="E1-F2")
    print(f"collect_votes done in {time.time() - t0:.1f}s", flush=True)

    cons = CF.run_consensus(tasks_by_id, test_ids, MR.MODEL_IDS, votes)

    per_task = {}
    n_pass = 0
    for tid in test_ids:
        c = cons[tid]
        task = tasks_by_id[tid]
        check = OR.check_filter(", ".join(sorted(c["consensus_ids"])), task["matched_ids"])
        per_task[tid] = {
            "v": check["v"], "consensus_ids": sorted(c["consensus_ids"]),
            "golden": task["matched_ids"], "admitted": c["admitted"],
            "loo_rel": c["loo_rel"],
        }
        n_pass += check["v"]

    r_e1 = n_pass / len(test_ids)
    delta_e1 = r_e1 - BASELINE_R_UNION_LLM_F2
    if r_e1 < THR_FALSIFY:
        verdict = "FALSIFIED"
    elif THR_PREDICT_LOW <= r_e1 <= THR_PREDICT_HIGH:
        verdict = "CONFIRMED_IN_RANGE"
    else:
        verdict = "ABOVE_RANGE_NOT_FALSIFIED"

    result = {
        "r_E1": r_e1, "n_pass": n_pass, "n": len(test_ids),
        "baseline_r_union_llm_f2": BASELINE_R_UNION_LLM_F2, "delta_E1": delta_e1,
        "threshold_predict": [THR_PREDICT_LOW, THR_PREDICT_HIGH], "threshold_falsify": THR_FALSIFY,
        "verdict": verdict, "per_task": per_task,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "e1_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nr_E1 = {r_e1:.3f}  ({n_pass}/{len(test_ids)})", flush=True)
    print(f"Delta_E1 = {delta_e1:+.3f}  (baseline {BASELINE_R_UNION_LLM_F2:.3f})", flush=True)
    print(f"verdict = {verdict}", flush=True)


if __name__ == "__main__":
    main()
