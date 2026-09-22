"""consensus_filter.py — E1/E2 core mechanism. See PROTOCOL_E1E2E3.md for
the full design and every self-defined premise (candidate-id universe,
balanced-accuracy reliability, leave-one-out calibration, tie-break,
empty-admitted-set default).

One FILTER call per (task, model) — field decomposition (per-id binary
vote) happens entirely in post-processing of that single response via
`oracles.extract_ids`, never as a separate call per candidate id.
"""

from __future__ import annotations

import hashlib
import time

import model_registry_11 as MR
import oracles as OR
import call_log

TEMPERATURE = 0.0
SEED_BASE_ESW_FILTER = 97000


def stable_hash(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest, 16) % 10 ** 6


def draw_seed(task_id: str, model_id: str) -> int:
    return SEED_BASE_ESW_FILTER + stable_hash(task_id, "FILTER", model_id)


def candidate_ids(task: dict) -> list:
    return sorted(r["id"] for r in task["records"])


def collect_votes(model_ids, tasks_by_id: dict, task_ids: list, prompt_builder, max_tokens: int,
                  family: str) -> dict:
    """-> {task_id: {model_id: {"extracted": frozenset, "raw_text": str, "failed": bool, "cost": int}}}"""
    votes = {t: {} for t in task_ids}
    for model_id in model_ids:
        llm, load_t = MR.load_model(model_id)
        print(f"[consensus-{family}] {model_id} loaded in {load_t:.1f}s", flush=True)
        for task_id in task_ids:
            task = tasks_by_id[task_id]
            seed = draw_seed(task_id, model_id)
            t0 = time.time()
            raw = MR.generate(llm, model_id, prompt_builder(task), temperature=TEMPERATURE,
                              top_p=1.0, seed=seed, max_tokens=max_tokens)
            ms = (time.time() - t0) * 1000.0
            n_in = raw.get("input_tokens") or 0
            n_out = raw.get("output_tokens") or 0
            failed = bool(raw.get("generation_failed"))
            raw_text = raw.get("raw_text", "")
            extracted = OR.extract_ids(raw_text) if not failed else frozenset()
            votes[task_id][model_id] = {
                "extracted": extracted, "raw_text": raw_text, "failed": failed,
                "cost": n_in + n_out,
            }
            call_log.log_filter_call(model=model_id, task_id=task_id, family=family, seed=seed,
                                     n_in=n_in, n_out=n_out, ms=ms, failed=failed,
                                     extracted=sorted(extracted), raw_text=raw_text)
        del llm
    return votes


def golden_map(task: dict) -> dict:
    matched = set(task["matched_ids"])
    return {cid: (1 if cid in matched else 0) for cid in candidate_ids(task)}


def per_id_votes(task: dict, votes_for_task: dict) -> dict:
    cids = candidate_ids(task)
    out = {}
    for model_id, v in votes_for_task.items():
        extracted = v["extracted"]
        out[model_id] = {cid: (1 if cid in extracted else 0) for cid in cids}
    return out


def loo_reliabilities(tasks_by_id: dict, task_ids: list, model_ids, votes: dict,
                      exclude_task_id: str) -> dict:
    """Balanced accuracy per model, leave-one-out excluding `exclude_task_id`.
    `T`'s own golden labels are never used to weight the vote that scores `T`."""
    stats = {m: {"tp": 0, "fn": 0, "tn": 0, "fp": 0} for m in model_ids}
    for tid in task_ids:
        if tid == exclude_task_id:
            continue
        task = tasks_by_id[tid]
        golden = golden_map(task)
        pv = per_id_votes(task, votes[tid])
        for model_id in model_ids:
            mv = pv[model_id]
            for cid, g in golden.items():
                v = mv[cid]
                if g == 1 and v == 1:
                    stats[model_id]["tp"] += 1
                elif g == 1 and v == 0:
                    stats[model_id]["fn"] += 1
                elif g == 0 and v == 0:
                    stats[model_id]["tn"] += 1
                else:
                    stats[model_id]["fp"] += 1
    rel = {}
    for model_id, s in stats.items():
        tpr = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        tnr = s["tn"] / (s["tn"] + s["fp"]) if (s["tn"] + s["fp"]) > 0 else 0.0
        rel[model_id] = (tpr + tnr) / 2.0
    return rel


def consensus_ids_for_task(task: dict, votes_for_task: dict, admitted: list,
                           loo_rel: dict) -> set:
    cids = candidate_ids(task)
    pv = per_id_votes(task, votes_for_task)
    result: set = set()
    if not admitted:
        return result   # A(T) = empty -> abstain -> empty set, scored FAIL, never golden-substituted
    best_m = max(admitted, key=lambda m: loo_rel[m])
    for cid in cids:
        yes = sum(1 for m in admitted if pv[m][cid] == 1)
        no = sum(1 for m in admitted if pv[m][cid] == 0)
        if yes > no:
            include = True
        elif no > yes:
            include = False
        else:
            include = bool(pv[best_m][cid])
        if include:
            result.add(cid)
    return result


def run_consensus(tasks_by_id: dict, task_ids: list, model_ids, votes: dict) -> dict:
    """-> {task_id: {"consensus_ids": set, "admitted": list, "loo_rel": dict}}"""
    out = {}
    for tid in task_ids:
        loo_rel = loo_reliabilities(tasks_by_id, task_ids, model_ids, votes, tid)
        admitted = sorted(m for m in model_ids if loo_rel[m] > 0.5)
        cons_ids = consensus_ids_for_task(tasks_by_id[tid], votes[tid], admitted, loo_rel)
        out[tid] = {"consensus_ids": cons_ids, "admitted": admitted, "loo_rel": loo_rel}
    return out
