"""
Section 25 (mixed-model pools) and section 23 (marginal utility) for
Experiment 4. Reuses the claim pools built in compute_metrics4.py.
"""
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))

from hypothesis import build_hypothesis_rows, load_a1_runs
from compute_metrics4 import build_claim_pool, entropy_and_stats, METRICS_DIR, N_TRIALS, RNG_SEED

MODELS = ["qwen3-1.7b-q4_0-unsloth", "llama-3.2-1b-instruct-q4_0",
          "qwen2.5-coder-1.5b-instruct-q4_0", "smollm2-360m-instruct-bartowski-q5_k_m"]


def main():
    hyp_rows = build_hypothesis_rows()
    all_runs = load_a1_runs()

    status_lookup_by_claim = collections.defaultdict(dict)
    for r in hyp_rows:
        status_lookup_by_claim[(r["task_id"], r["claim_id"])][r["hypothesis_id"]] = r["status"]

    # per-model run_id pools at P1/T0.5
    per_model_runs = {}
    for m in MODELS:
        per_model_runs[m] = set(r["run_id"] for r in all_runs if r["model_id"] == m and r["prompt_id"] == "P1" and r["temperature"] == 0.5)

    per_model_pool = {m: build_claim_pool(hyp_rows, rids) for m, rids in per_model_runs.items()}

    rng = random.Random(RNG_SEED + 1)

    # -- SAME MODEL x N vs MIXED MODELS x N (Oracle-style correct-present) --
    claim_keys = set()
    for m in MODELS:
        claim_keys |= set(per_model_pool[m].keys())

    mixed_results = {}
    same_model_results = {"qwen3-1.7b-q4_0-unsloth_alone": {}}

    for per_model_n, label in [(1, "N=4_(1+1+1+1)"), (2, "N=8_(2+2+2+2)")]:
        n_total = per_model_n * len(MODELS)
        correct_present_trials = []
        maj_correct_trials = []
        n_claims_used = 0
        for key in claim_keys:
            pools = []
            statuses = status_lookup_by_claim[key]
            ok = True
            for m in MODELS:
                cp = per_model_pool[m].get(key, {})
                if len(cp) < per_model_n:
                    ok = False
                    break
                pools.append(cp)
            if not ok:
                continue
            n_claims_used += 1
            cp_trials = []
            mj_trials = []
            for _ in range(N_TRIALS):
                sample = []
                for cp in pools:
                    run_ids = list(cp.keys())
                    chosen = rng.sample(run_ids, per_model_n)
                    sample.extend(cp[r] for r in chosen)
                uniq = set(sample)
                has_correct = any(statuses.get(h) == "CORRECT" for h in uniq)
                cp_trials.append(has_correct)
                c = collections.Counter(sample)
                top_count = c.most_common(1)[0][1]
                top_hyps = [h for h, v in c.items() if v == top_count]
                mj_trials.append(all(statuses.get(h) == "CORRECT" for h in top_hyps))
            correct_present_trials.append(sum(cp_trials) / N_TRIALS)
            maj_correct_trials.append(sum(mj_trials) / N_TRIALS)

        mixed_results[label] = {
            "n_total_generations": n_total, "n_claims_used": n_claims_used,
            "mean_p_correct_present": round(sum(correct_present_trials) / len(correct_present_trials), 3) if correct_present_trials else None,
            "mean_p_majority_correct": round(sum(maj_correct_trials) / len(maj_correct_trials), 3) if maj_correct_trials else None,
        }

        # same-model (qwen3 alone) comparison at the SAME total N
        q_pool = per_model_pool["qwen3-1.7b-q4_0-unsloth"]
        cp_trials = []
        mj_trials = []
        n_claims_q = 0
        for key, cp in q_pool.items():
            if len(cp) < n_total:
                continue
            n_claims_q += 1
            statuses = status_lookup_by_claim[key]
            run_ids = list(cp.keys())
            local_cp = []
            local_mj = []
            for _ in range(N_TRIALS):
                sample_ids = rng.sample(run_ids, n_total)
                sample = [cp[r] for r in sample_ids]
                uniq = set(sample)
                local_cp.append(any(statuses.get(h) == "CORRECT" for h in uniq))
                c = collections.Counter(sample)
                top_count = c.most_common(1)[0][1]
                top_hyps = [h for h, v in c.items() if v == top_count]
                local_mj.append(all(statuses.get(h) == "CORRECT" for h in top_hyps))
            cp_trials.append(sum(local_cp) / N_TRIALS)
            mj_trials.append(sum(local_mj) / N_TRIALS)
        same_model_results["qwen3-1.7b-q4_0-unsloth_alone"][label] = {
            "n_total_generations": n_total, "n_claims_used": n_claims_q,
            "mean_p_correct_present": round(sum(cp_trials) / len(cp_trials), 3) if cp_trials else None,
            "mean_p_majority_correct": round(sum(mj_trials) / len(mj_trials), 3) if mj_trials else None,
        }

    # -- marginal utility (section 23): delta coverage / correct-present / conflict between N steps --
    with open(os.path.join(METRICS_DIR, "metrics4_summary.json"), encoding="utf-8") as f:
        m4 = json.load(f)
    main_cond = m4["main_qwen3_P1_T0.5"]
    n_seq = ["1", "2", "4", "8", "16"]
    agg = {n: {"coverage": [], "correct_present": [], "conflict": []} for n in n_seq}
    for claim, cm in main_cond.items():
        for n in n_seq:
            if n in cm["by_n"]:
                agg[n]["coverage"].append(cm["by_n"][n]["mean_coverage"])
                agg[n]["correct_present"].append(cm["by_n"][n]["p_correct_present"])
                agg[n]["conflict"].append(cm["by_n"][n]["p_conflict"])
    mean_by_n = {}
    for n in n_seq:
        if agg[n]["coverage"]:
            mean_by_n[n] = {
                "mean_coverage": sum(agg[n]["coverage"]) / len(agg[n]["coverage"]),
                "mean_correct_present": sum(agg[n]["correct_present"]) / len(agg[n]["correct_present"]),
                "mean_conflict": sum(agg[n]["conflict"]) / len(agg[n]["conflict"]),
            }
    marginal = {}
    steps = [("1", "2"), ("2", "4"), ("4", "8"), ("8", "16")]
    for a, b in steps:
        if a in mean_by_n and b in mean_by_n:
            extra_gens = int(b) - int(a)
            marginal[f"{a}->{b}"] = {
                "delta_coverage": round(mean_by_n[b]["mean_coverage"] - mean_by_n[a]["mean_coverage"], 4),
                "delta_correct_present": round(mean_by_n[b]["mean_correct_present"] - mean_by_n[a]["mean_correct_present"], 4),
                "delta_conflict": round(mean_by_n[b]["mean_conflict"] - mean_by_n[a]["mean_conflict"], 4),
                "extra_generations": extra_gens,
                "correct_present_gain_per_extra_generation": round(
                    (mean_by_n[b]["mean_correct_present"] - mean_by_n[a]["mean_correct_present"]) / extra_gens, 5
                ),
            }

    out = {"mixed_vs_same_model": {"mixed": mixed_results, "same_model": same_model_results},
           "marginal_utility_main_condition": marginal}
    with open(os.path.join(METRICS_DIR, "mixed_and_marginal.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
