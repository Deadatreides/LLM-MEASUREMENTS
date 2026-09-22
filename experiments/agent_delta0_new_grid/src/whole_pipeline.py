"""whole_pipeline.py — DELTA-0 C6/C7: whole-task r_m*(B) measurement for
every registry model, then m* = argmax. PROTOCOL.md §4: whole prompt asks
ONLY for the final ДА/НЕТ; `cost = input+output tokens`; `cost>B -> FAIL`
(not "almost"); temperature=0 (fallback 0.1 if unsupported, recorded).
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import oracles as OR              # noqa: E402
import model_registry_11 as MR    # noqa: E402
import call_log                   # noqa: E402

TEMPERATURE = 0.0
SEED_BASE_WHOLE = 70000
MAX_TOKENS_WHOLE_GEN = 30   # generous for a 1-word ДА/НЕТ answer + minor preamble;
                            # independent of B (a measurement cap, not a truncation setting)


def stable_hash(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest, 16) % 10 ** 6


def draw_seed(task_id: str, model_id: str) -> int:
    return SEED_BASE_WHOLE + stable_hash(task_id, model_id)


def build_whole_prompt(task: dict) -> str:
    return f"База транзакций:\n{task['db_text']}\n\n{task['query_text']}"


def measure_whole_for_model(model_id: str, tasks_by_id: dict, task_ids: list, B: int,
                            temperature: float = TEMPERATURE) -> dict:
    llm, load_t = MR.load_model(model_id)
    print(f"[whole] {model_id} загружена за {load_t:.1f}с", flush=True)
    results = {}
    for t_id in task_ids:
        task = tasks_by_id[t_id]
        prompt = build_whole_prompt(task)
        seed = draw_seed(t_id, model_id)
        t0 = time.time()
        raw = MR.generate(llm, model_id, prompt, temperature=temperature, top_p=1.0,
                          seed=seed, max_tokens=MAX_TOKENS_WHOLE_GEN)
        ms = (time.time() - t0) * 1000.0
        n_in = raw.get("input_tokens") or 0
        n_out = raw.get("output_tokens") or 0
        cost = n_in + n_out
        failed = bool(raw.get("generation_failed"))
        if failed:
            v, over_budget = 0, False
        elif cost > B:
            v, over_budget = 0, True     # PROTOCOL.md §4: cost>B -> FAIL, not "almost"
        else:
            check = OR.check_whole_final(raw.get("raw_text", ""), task["final_oracle"])
            v, over_budget = check["v"], False
        results[t_id] = {"v": v, "cost": cost, "n_in": n_in, "n_out": n_out,
                         "failed": failed, "over_budget": over_budget,
                         "raw_text": raw.get("raw_text", "")}
        call_log.log_whole_call(model=model_id, task_id=t_id, seed=seed, n_in=n_in,
                                n_out=n_out, ms=ms, failed=failed)
    del llm
    return results


def measure_all_models(model_ids: list, tasks_by_id: dict, task_ids: list, B: int) -> dict:
    return {m: measure_whole_for_model(m, tasks_by_id, task_ids, B) for m in model_ids}


def compute_r_m(results_by_model: dict, task_ids: list) -> dict:
    out = {}
    n = len(task_ids)
    for m, results in results_by_model.items():
        n_pass = sum(1 for t in task_ids if results[t]["v"] == 1)
        mean_cost = sum(results[t]["cost"] for t in task_ids) / n if n else 0.0
        out[m] = {"r_m": n_pass / n if n else 0.0, "n_pass": n_pass, "n": n, "mean_cost": mean_cost}
    return out


def select_m_star(r_m_table: dict) -> str:
    """argmax r_m; ties -> lower mean cost; ties -> lexicographically
    smaller model_id (PROTOCOL.md §4)."""
    return min(r_m_table, key=lambda m: (-r_m_table[m]["r_m"], r_m_table[m]["mean_cost"], m))
