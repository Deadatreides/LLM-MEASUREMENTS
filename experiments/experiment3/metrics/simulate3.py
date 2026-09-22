"""
Spec section 16: mathematical simulation for N=1,2,4,8,16 independent
generations, grounded in the REAL measured error correlation (not an
assumed independence model). Implemented as bootstrap resampling (with
replacement) from the actual observed claim vectors within a fixed
condition (same task/model/prompt/temperature -- the "same_settings"
cells, which is where independent-repeat correlation was actually
measured). This directly encodes whatever correlation truly exists in the
data, rather than forcing a single-parameter rho model.

Compares two resolution strategies per claim, given N draws:
  A. majority vote      -- always resolves to whichever label has more
                            votes among the determinate (CORRECT/INCORRECT)
                            draws; never abstains.
  B. conflict-first      -- resolves only if ALL determinate draws agree;
                            otherwise the claim is left UNRESOLVED (a
                            signal for further checking, not a guess).
"""
import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))

from compute_metrics3 import load_runs, load_claims, build_vectors, RUNS_DIR, METRICS_DIR

N_VALUES = [1, 2, 4, 8, 16]
N_TRIALS = 800
RNG_SEED = 12345


def majority_vote(labels):
    det = [l for l in labels if l in ("CORRECT", "INCORRECT")]
    if not det:
        return None
    c = collections.Counter(det)
    top, top_n = c.most_common(1)[0]
    n_second = c.most_common(2)[1][1] if len(c) > 1 else 0
    if top_n == n_second:
        return "TIE"
    return top


def conflict_first(labels):
    det = [l for l in labels if l in ("CORRECT", "INCORRECT")]
    if not det:
        return None
    if len(set(det)) == 1:
        return det[0]
    return "UNRESOLVED"


def simulate_condition(claim_vectors_pool, claim_ids, rng):
    """claim_vectors_pool: list of {claim_id: classification} dicts (the
    real, independently-observed A1s for one fixed condition)."""
    out = {}
    for n in N_VALUES:
        n_at_least_one_fully_correct = 0
        maj_resolved = 0
        maj_correct = 0
        maj_incorrect = 0
        cf_resolved = 0
        cf_unresolved = 0
        cf_correct = 0
        cf_incorrect = 0
        n_unanimous_wrong = 0
        n_claim_trials = 0
        n_conflict_detected_trials = 0

        for _ in range(N_TRIALS):
            draw = [rng.choice(claim_vectors_pool) for _ in range(n)]
            if any(all(v == "CORRECT" for v in d.values()) for d in draw):
                n_at_least_one_fully_correct += 1

            trial_has_conflict = False
            for cid in claim_ids:
                labels = [d.get(cid) for d in draw]
                mv = majority_vote(labels)
                cfv = conflict_first(labels)
                if mv is not None:
                    n_claim_trials += 1
                    if mv != "TIE":
                        maj_resolved += 1
                        if mv == "CORRECT":
                            maj_correct += 1
                        else:
                            maj_incorrect += 1
                    if cfv == "UNRESOLVED":
                        cf_unresolved += 1
                        trial_has_conflict = True
                    elif cfv is not None:
                        cf_resolved += 1
                        if cfv == "CORRECT":
                            cf_correct += 1
                        else:
                            cf_incorrect += 1
                            n_unanimous_wrong += 1
            if trial_has_conflict:
                n_conflict_detected_trials += 1

        out[str(n)] = {
            "p_at_least_one_fully_correct": round(n_at_least_one_fully_correct / N_TRIALS, 3),
            "p_conflict_detected_in_trial": round(n_conflict_detected_trials / N_TRIALS, 3),
            "majority_vote": {
                "resolve_rate": round(maj_resolved / n_claim_trials, 3) if n_claim_trials else None,
                "accuracy_when_resolved": round(maj_correct / maj_resolved, 3) if maj_resolved else None,
                "p_majority_error": round(maj_incorrect / n_claim_trials, 3) if n_claim_trials else None,
            },
            "conflict_first": {
                "resolve_rate": round(cf_resolved / n_claim_trials, 3) if n_claim_trials else None,
                "unresolved_rate": round(cf_unresolved / n_claim_trials, 3) if n_claim_trials else None,
                "accuracy_when_resolved": round(cf_correct / cf_resolved, 3) if cf_resolved else None,
                "p_unanimous_wrong": round(n_unanimous_wrong / n_claim_trials, 3) if n_claim_trials else None,
            },
        }
    return out


def main():
    runs = load_runs()
    claim_rows = load_claims()
    vecs = build_vectors(claim_rows)

    cells = collections.defaultdict(list)
    for r in runs:
        cells[(r["task_id"], r["model_id"], r["prompt_id"], r["temperature"])].append(r["run_id"])

    rng = random.Random(RNG_SEED)
    per_task_results = {}
    pooled_pool = []
    claim_ids_all = set()

    for (task_id, model_id, prompt_id, t), run_ids in cells.items():
        if len(run_ids) < 4:
            continue  # need a reasonably-sized pool to bootstrap from
        pool = [vecs[r] for r in run_ids]
        claim_ids = sorted(set().union(*[set(v.keys()) for v in pool]))
        key = f"{task_id}|{model_id}|{prompt_id}|T{t}"
        per_task_results[key] = simulate_condition(pool, claim_ids, rng)
        pooled_pool.extend(pool)
        claim_ids_all.update(claim_ids)

    # one pooled simulation across ALL same-settings cells combined (coarser,
    # but shows the aggregate picture across the whole experiment)
    pooled = simulate_condition(pooled_pool, sorted(claim_ids_all), rng) if pooled_pool else {}

    out = {"n_trials": N_TRIALS, "n_conditions_simulated": len(per_task_results),
           "pooled_across_all_same_settings_cells": pooled, "per_condition": per_task_results}
    with open(os.path.join(METRICS_DIR, "simulation3.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"simulated {len(per_task_results)} conditions, {N_TRIALS} trials each")
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
