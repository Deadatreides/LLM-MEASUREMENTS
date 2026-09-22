"""union_hard.py — FORK-1 Path B, K6: r_∪_B, two sub-measurements.
PROTOCOL.md §2.5: B-DET (FILTER=det, SUM=det, 0 LLM calls) and B-LLM
(FILTER=LLM executor=m*_B, SUM=det -- SUM is NEVER asked of an LLM in
either variant). Chain-break discipline identical to DELTA-0 (reapplied,
not rederived): FILTER v=0 -> union_pass=False immediately, no golden
substitution, SUM never called for that task.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import atoms_hard as AH            # noqa: E402
import det_atoms as DA             # noqa: E402
import oracles as OR               # noqa: E402
import model_registry_11 as MR     # noqa: E402
import call_log                    # noqa: E402

TEMPERATURE = 0.0
SEED_BASE_UNION_HARD = 95000


def stable_hash(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest, 16) % 10 ** 6


def draw_seed(task_id: str, atom: str, model_id: str) -> int:
    return SEED_BASE_UNION_HARD + stable_hash(task_id, atom, model_id)


def run_chain_det(task: dict) -> dict:
    """B-DET: FILTER=det, SUM=det. 0 LLM calls."""
    ids = DA.filter_det(task["records"], task["target_category"], task["target_region"])
    filter_check = OR.check_filter(", ".join(sorted(ids)), task["matched_ids"])
    if filter_check["v"] != 1:
        return {"union_pass": False, "break_at": "FILTER", "total_cost": 0, "calls": [],
               "filter_extracted": filter_check["extracted"]}
    sum_value = DA.aggregate_det(ids, "SUM", task["id_to_amount"])
    passed = OR.within_tolerance(sum_value, task["final_oracle"])
    return {"union_pass": passed, "break_at": None if passed else "SUM",
           "total_cost": 0, "calls": [], "sum_computed": sum_value}


def run_chain_llm(task: dict, executor_model: str, llm) -> dict:
    """B-LLM: FILTER=LLM (executor_model), SUM=det (always, per PROTOCOL §2.3)."""
    calls = []
    total_cost = 0
    task_id = task["task_id"]

    seed = draw_seed(task_id, "FILTER", executor_model)
    t0 = time.time()
    raw = MR.generate(llm, executor_model, AH.build_filter_prompt(task), temperature=TEMPERATURE,
                      top_p=1.0, seed=seed, max_tokens=AH.MAX_TOKENS_FILTER)
    ms = (time.time() - t0) * 1000.0
    n_in, n_out = raw.get("input_tokens") or 0, raw.get("output_tokens") or 0
    failed = bool(raw.get("generation_failed"))
    calls.append({"atom": "FILTER", "cost": n_in + n_out, "failed": failed,
                 "raw_text": raw.get("raw_text", "")})
    total_cost += n_in + n_out
    call_log.log_union_call(model=executor_model, task_id=task_id, atom="FILTER", seed=seed,
                            n_in=n_in, n_out=n_out, ms=ms, failed=failed)
    if failed:
        return {"union_pass": False, "break_at": "FILTER_CALL_FAILED", "total_cost": total_cost,
               "calls": calls}

    filter_check = OR.check_filter(raw.get("raw_text", ""), task["matched_ids"])
    calls[-1]["v"] = filter_check["v"]
    if filter_check["v"] != 1:
        # PROTOCOL.md correction (reapplied from DELTA-0): chain breaks HERE,
        # no golden substitution, SUM is never called for this task.
        return {"union_pass": False, "break_at": "FILTER", "total_cost": total_cost,
               "calls": calls, "filter_extracted": filter_check["extracted"]}
    real_ids = filter_check["extracted"]   # == golden here (v==1), but this IS the real output

    sum_value = DA.aggregate_det(real_ids, "SUM", task["id_to_amount"])
    passed = OR.within_tolerance(sum_value, task["final_oracle"])
    return {"union_pass": passed, "break_at": None if passed else "SUM",
           "total_cost": total_cost, "calls": calls, "sum_computed": sum_value}


def measure_union_det(tasks_by_id: dict, task_ids: list) -> dict:
    return {t: run_chain_det(tasks_by_id[t]) for t in task_ids}


def measure_union_llm(executor_model: str, tasks_by_id: dict, task_ids: list) -> dict:
    llm, load_t = MR.load_model(executor_model)
    print(f"[union-hard] {executor_model} загружена за {load_t:.1f}с", flush=True)
    results = {}
    for t_id in task_ids:
        results[t_id] = run_chain_llm(tasks_by_id[t_id], executor_model, llm)
    del llm
    return results


def compute_r_union(results: dict, task_ids: list) -> dict:
    n = len(task_ids)
    n_pass = sum(1 for t in task_ids if results[t]["union_pass"])
    mean_cost = sum(results[t]["total_cost"] for t in task_ids) / n if n else 0.0
    break_at_counts: dict = {}
    for t in task_ids:
        if not results[t]["union_pass"]:
            k = results[t]["break_at"]
            break_at_counts[k] = break_at_counts.get(k, 0) + 1
    return {"r_union_lower": n_pass / n if n else 0.0, "n_pass": n_pass, "n": n,
           "mean_cost": mean_cost, "break_at_counts": break_at_counts}
