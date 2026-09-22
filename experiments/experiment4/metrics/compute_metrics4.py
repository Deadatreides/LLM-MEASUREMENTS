"""
Core Experiment 4 metrics: hypothesis coverage, error concentration/
diversity, correct-in-pool (empirical vs naive-independence), false
consensus, conflict information, Oracle@N vs majority@N vs single, and
marginal utility -- for N=1,2,4,8,16, computed from real data via
without-replacement subsampling of the real generation pools (16 real
draws for the main condition, 8 for the temperature/model control series
-- see spec section 3).
"""
import collections
import itertools
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))

from hypothesis import build_hypothesis_rows, load_a1_runs

EXP4_DIR = r"<PROJECT_ROOT>\trace-probe\experiment4"
METRICS_DIR = os.path.join(EXP4_DIR, "metrics")
RUNS_DIR = os.path.join(EXP4_DIR, "runs")

N_VALUES = [1, 2, 4, 8, 16]
N_TRIALS = 500
RNG_SEED = 20260814


def build_claim_pool(hyp_rows, run_ids):
    """(task_id, claim_id) -> {run_id: hypothesis_id} restricted to the
    given run_ids (only rows where a hypothesis was actually stated --
    OMITTED runs contribute no entry for that claim, exactly like a
    generation that didn't address it)."""
    pool = collections.defaultdict(dict)
    for r in hyp_rows:
        if r["run_id"] in run_ids:
            pool[(r["task_id"], r["claim_id"])][r["run_id"]] = r["hypothesis_id"]
    return pool


def subsample(items, n, rng, max_pop):
    """Without-replacement subsample of size n from items (a list). If
    n >= len(items), returns items unchanged (can't subsample more than
    exist)."""
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)


def entropy_and_stats(hyp_ids):
    """hyp_ids: list of hypothesis_id strings (one per generation that
    addressed the claim). Returns entropy (bits), largest share, n_eff
    (inverse-Simpson), n_unique."""
    if not hyp_ids:
        return {"entropy_bits": None, "largest_share": None, "n_eff": None, "n_unique": 0}
    counts = collections.Counter(hyp_ids)
    total = len(hyp_ids)
    probs = [v / total for v in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    n_eff = 1.0 / sum(p * p for p in probs)
    return {
        "entropy_bits": round(entropy, 3),
        "largest_share": round(max(counts.values()) / total, 3),
        "n_eff": round(n_eff, 2),
        "n_unique": len(counts),
    }


def compute_claim_metrics(claim_pool_full, status_lookup, condition_name, n_values=N_VALUES, n_trials=N_TRIALS, rng=None):
    """claim_pool_full: {run_id: hypothesis_id} for one (task,claim) in one
    condition. status_lookup: hyp_id -> 'CORRECT'/'INCORRECT'/'AMBIGUOUS'.
    Returns per-N stats via without-replacement subsampling."""
    run_ids = list(claim_pool_full.keys())
    pop_size = len(run_ids)
    if pop_size == 0:
        return None

    out = {"population_size": pop_size, "by_n": {}}
    full_hyp_ids = [claim_pool_full[r] for r in run_ids]
    out["full_population_stats"] = entropy_and_stats(full_hyp_ids)
    out["full_population_stats"]["correct_share"] = round(
        sum(1 for h in full_hyp_ids if status_lookup.get(h) == "CORRECT") / pop_size, 3
    )
    out["full_population_stats"]["incorrect_share"] = round(
        sum(1 for h in full_hyp_ids if status_lookup.get(h) == "INCORRECT") / pop_size, 3
    )

    for n in n_values:
        if n > pop_size:
            continue
        trials = []
        exact = (n == pop_size)
        n_trial_iters = 1 if exact else n_trials
        for _ in range(n_trial_iters):
            sample_ids = list(run_ids) if exact else rng.sample(run_ids, n)
            sample_hyps = [claim_pool_full[r] for r in sample_ids]
            trials.append(sample_hyps)

        coverage = []
        correct_present = []
        incorrect_count = []
        unsupported_count = []
        majority_correct = []
        unanimous_wrong = []
        conflict_flags = []
        for sample_hyps in trials:
            uniq = set(sample_hyps)
            coverage.append(len(uniq))
            has_correct = any(status_lookup.get(h) == "CORRECT" for h in uniq)
            correct_present.append(has_correct)
            incorrect_count.append(sum(1 for h in uniq if status_lookup.get(h) == "INCORRECT"))
            unsupported_count.append(sum(1 for h in uniq if status_lookup.get(h) not in ("CORRECT", "INCORRECT")))
            conflict_flags.append(len(uniq) > 1)
            if len(uniq) == 1:
                only = next(iter(uniq))
                unanimous_wrong.append(status_lookup.get(only) == "INCORRECT")
            else:
                unanimous_wrong.append(False)
            # majority / plurality
            c = collections.Counter(sample_hyps)
            top_count = c.most_common(1)[0][1]
            top_hyps = [h for h, v in c.items() if v == top_count]
            # tie -> not resolved as correct unless ALL tied top hyps are correct
            maj_is_correct = all(status_lookup.get(h) == "CORRECT" for h in top_hyps)
            majority_correct.append(maj_is_correct)

        n_t = len(trials)
        # H1/H2 (spec sections 16-17): condition correct_present on
        # whether THAT SAME trial had a conflict, or was unanimous.
        conflict_trials = [cp for cp, cf in zip(correct_present, conflict_flags) if cf]
        no_conflict_trials = [cp for cp, cf in zip(correct_present, conflict_flags) if not cf]

        out["by_n"][str(n)] = {
            "n_trials": n_t, "exact_population": exact,
            "mean_coverage": round(sum(coverage) / n_t, 3),
            "p_correct_present": round(sum(correct_present) / n_t, 3),
            "mean_incorrect_hyp_count": round(sum(incorrect_count) / n_t, 3),
            "mean_unsupported_hyp_count": round(sum(unsupported_count) / n_t, 3),
            "p_majority_correct": round(sum(majority_correct) / n_t, 3),
            "p_unanimous_wrong_false_consensus": round(sum(unanimous_wrong) / n_t, 3),
            "p_conflict": round(sum(conflict_flags) / n_t, 3),
            "p_correct_present_given_conflict": round(sum(conflict_trials) / len(conflict_trials), 3) if conflict_trials else None,
            "p_correct_present_given_no_conflict": round(sum(no_conflict_trials) / len(no_conflict_trials), 3) if no_conflict_trials else None,
            "n_conflict_trials": len(conflict_trials), "n_no_conflict_trials": len(no_conflict_trials),
        }

    return out


def naive_independence_model(p1, n):
    """P(at least one correct among n) if draws were truly independent
    Bernoulli(p1)."""
    return 1 - (1 - p1) ** n


def main():
    condition = sys.argv[1] if len(sys.argv) > 1 else "main"
    hyp_rows = build_hypothesis_rows()
    all_runs = load_a1_runs()
    runs_by_id = {r["run_id"]: r for r in all_runs}

    # IMPORTANT: hypothesis_id strings like "CORRECT_1", "INCORRECT_1",
    # "TYPE=STRING", "ARGC=1" are reused across DIFFERENT claims, and the
    # SAME literal id can mean different things (e.g. "TYPE=STRING" is the
    # correct hypothesis for a claim expecting a string argument, but would
    # be the incorrect one for a claim expecting an int). The status
    # lookup must therefore be scoped per (task_id, claim_id), never
    # global -- a global dict here would silently let the last-processed
    # claim's status overwrite every other claim's meaning of that id.
    status_lookup_by_claim = collections.defaultdict(dict)
    for r in hyp_rows:
        status_lookup_by_claim[(r["task_id"], r["claim_id"])][r["hypothesis_id"]] = r["status"]

    rng = random.Random(RNG_SEED)

    conditions = {
        "main_qwen3_P1_T0.5": lambda r: r["model_id"] == "qwen3-1.7b-q4_0-unsloth" and r["prompt_id"] == "P1" and r["temperature"] == 0.5,
        "qwen3_P1_T0.0": lambda r: r["model_id"] == "qwen3-1.7b-q4_0-unsloth" and r["prompt_id"] == "P1" and r["temperature"] == 0.0,
        "qwen3_P1_T1.0": lambda r: r["model_id"] == "qwen3-1.7b-q4_0-unsloth" and r["prompt_id"] == "P1" and r["temperature"] == 1.0,
        "llama_P1_T0.5": lambda r: r["model_id"] == "llama-3.2-1b-instruct-q4_0" and r["prompt_id"] == "P1" and r["temperature"] == 0.5,
        "qwen_coder_P1_T0.5": lambda r: r["model_id"] == "qwen2.5-coder-1.5b-instruct-q4_0" and r["prompt_id"] == "P1" and r["temperature"] == 0.5,
        "smollm_P1_T0.5": lambda r: r["model_id"] == "smollm2-360m-instruct-bartowski-q5_k_m" and r["prompt_id"] == "P1" and r["temperature"] == 0.5,
    }

    result = {}
    for cond_name, pred in conditions.items():
        run_ids = set(r["run_id"] for r in all_runs if pred(r))
        pool = build_claim_pool(hyp_rows, run_ids)
        cond_result = {}
        for (task_id, claim_id), claim_pool in pool.items():
            local_status = status_lookup_by_claim[(task_id, claim_id)]
            cm = compute_claim_metrics(claim_pool, local_status, cond_name, rng=rng)
            if cm:
                cond_result[f"{task_id}/{claim_id}"] = cm
        result[cond_name] = cond_result

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "metrics4_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # also save hypothesis status lookup (scoped per claim) + population
    # sizes for downstream scripts
    with open(os.path.join(METRICS_DIR, "status_lookup.json"), "w", encoding="utf-8") as f:
        json.dump({f"{k[0]}/{k[1]}": v for k, v in status_lookup_by_claim.items()}, f, ensure_ascii=False, indent=2)

    print("conditions:", list(result.keys()))
    print("claims per condition:", {k: len(v) for k, v in result.items()})


if __name__ == "__main__":
    main()
