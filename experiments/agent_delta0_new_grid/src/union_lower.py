"""union_lower.py — DELTA-0 C9: M1 oracle-route, real-E chain.
PROTOCOL.md §5/§6 (locked corrections #1-#2): golden PLAN (which atoms,
their static params — never model-invented), real E (actual model
outputs chained forward, never golden-substituted). Chain breaks at the
FIRST failing checkpoint: `union_pass=False` immediately, remaining
atoms are NEVER called for that task (no rescue, no inflated cost).
Executor model = m* for all 3 atoms (PROTOCOL.md §9.1).
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import atoms as AT                # noqa: E402
import oracles as OR              # noqa: E402
import model_registry_11 as MR    # noqa: E402
import call_log                   # noqa: E402

TEMPERATURE = 0.0
SEED_BASE_UNION = 80000


def stable_hash(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest, 16) % 10 ** 6


def draw_seed(task_id: str, atom: str, model_id: str) -> int:
    return SEED_BASE_UNION + stable_hash(task_id, atom, model_id)


def _call_atom(atom_name: str, prompt: str, model_id: str, llm, task_id: str, max_tokens: int) -> dict:
    seed = draw_seed(task_id, atom_name, model_id)
    t0 = time.time()
    raw = MR.generate(llm, model_id, prompt, temperature=TEMPERATURE, top_p=1.0,
                      seed=seed, max_tokens=max_tokens)
    ms = (time.time() - t0) * 1000.0
    n_in, n_out = raw.get("input_tokens") or 0, raw.get("output_tokens") or 0
    failed = bool(raw.get("generation_failed"))
    call_log.log_union_call(model=model_id, task_id=task_id, atom=atom_name, seed=seed,
                            n_in=n_in, n_out=n_out, ms=ms, failed=failed)
    return {"raw_text": raw.get("raw_text", ""), "cost": n_in + n_out, "failed": failed}


def run_chain_for_task(task: dict, executor_model: str, llm) -> dict:
    calls = []
    total_cost = 0

    # -- FILTER --
    c = _call_atom("FILTER", AT.build_filter_prompt(task), executor_model, llm,
                   task["task_id"], AT.MAX_TOKENS_FILTER)
    calls.append({"atom": "FILTER", "cost": c["cost"], "failed": c["failed"], "raw_text": c["raw_text"]})
    total_cost += c["cost"]
    if c["failed"]:
        return {"union_pass": False, "break_at": "FILTER_CALL_FAILED", "calls": calls,
               "total_cost": total_cost}
    filter_check = OR.check_filter(c["raw_text"], task["matched_ids"])
    calls[-1]["v"] = filter_check["v"]
    if filter_check["v"] != 1:
        # PROTOCOL.md correction #1: chain breaks HERE, no golden substitution,
        # AGGREGATE/DERIVE are never called for this task.
        return {"union_pass": False, "break_at": "FILTER", "calls": calls,
               "total_cost": total_cost, "filter_extracted": filter_check["extracted"]}
    real_ids = filter_check["extracted"]   # == golden here (v==1), but this IS the real output

    # -- AGGREGATE (compact id->amount slice for the REAL extracted ids only) --
    id_amount_slice = {i: task["id_to_amount"][i] for i in real_ids if i in task["id_to_amount"]}
    c = _call_atom("AGGREGATE", AT.build_aggregate_prompt(id_amount_slice, task["op"]),
                   executor_model, llm, task["task_id"], AT.MAX_TOKENS_AGGREGATE)
    calls.append({"atom": "AGGREGATE", "cost": c["cost"], "failed": c["failed"], "raw_text": c["raw_text"]})
    total_cost += c["cost"]
    if c["failed"]:
        return {"union_pass": False, "break_at": "AGGREGATE_CALL_FAILED", "calls": calls,
               "total_cost": total_cost}
    agg_check = OR.check_aggregate(c["raw_text"], task["aggregate_value"])
    calls[-1]["v"] = agg_check["v"]
    if agg_check["v"] != 1:
        return {"union_pass": False, "break_at": "AGGREGATE", "calls": calls,
               "total_cost": total_cost, "aggregate_extracted": agg_check["extracted"]}
    real_agg_value = agg_check["extracted"]

    # -- DERIVE --
    c = _call_atom("DERIVE", AT.build_derive_prompt(real_agg_value, task["threshold"], task["comparator"]),
                   executor_model, llm, task["task_id"], AT.MAX_TOKENS_DERIVE)
    calls.append({"atom": "DERIVE", "cost": c["cost"], "failed": c["failed"], "raw_text": c["raw_text"]})
    total_cost += c["cost"]
    if c["failed"]:
        return {"union_pass": False, "break_at": "DERIVE_CALL_FAILED", "calls": calls,
               "total_cost": total_cost}
    derive_check = OR.check_derive(c["raw_text"], task["final_oracle"])
    calls[-1]["v"] = derive_check["v"]
    union_pass = derive_check["v"] == 1
    return {"union_pass": union_pass, "break_at": None if union_pass else "DERIVE",
           "calls": calls, "total_cost": total_cost}


def measure_union_lower(executor_model: str, tasks_by_id: dict, task_ids: list) -> dict:
    llm, load_t = MR.load_model(executor_model)
    print(f"[union] {executor_model} загружена за {load_t:.1f}с", flush=True)
    results = {}
    for t_id in task_ids:
        results[t_id] = run_chain_for_task(tasks_by_id[t_id], executor_model, llm)
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
