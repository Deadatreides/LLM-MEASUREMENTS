"""
Experiment 5 core metrics: semantic/error entropy, Oracle@N / Pass@1 /
G_N / delta-G_N, state-transition usefulness, entropy quartiles, A/B/C/D
quadrants, temperature comparison, systematic-trap detection, budget.

Clustering is fully mechanical (no LLM judge, consistent with every prior
experiment in this series): a CORRECT answer is one cluster (all
reference-verified equivalent); an INCORRECT answer's cluster is its
error_signature, already computed mechanically by experiment 2's
verification pipeline (reused unmodified here).
"""
import collections
import itertools
import json
import math
import os
import random
import sys

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment5"
RUNS_DIR = os.path.join(EXP_DIR, "runs")
METRICS_DIR = os.path.join(EXP_DIR, "metrics")

N_VALUES = [1, 2, 4, 8, 16]
N_TRIALS = 500
RNG_SEED = 20260815


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_main_index(model_id):
    return load_jsonl(os.path.join(RUNS_DIR, f"index.{model_id}.jsonl"))


def cluster_of(row):
    if row["correctness"]:
        return "CORRECT"
    sig = row["error_signature"] or f"UNSIGNATURED_{row['error_class']}"
    return sig


def entropy_bits(counts):
    total = sum(counts.values())
    if total == 0:
        return None
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def cell_stats(rows):
    """rows: list of index rows for one (task_id, temperature) cell (all
    16 seeds). Returns H_sem, H_error, cluster distributions, Pass@1."""
    clusters = [cluster_of(r) for r in rows]
    n = len(clusters)
    counts_all = collections.Counter(clusters)
    h_sem = entropy_bits(counts_all)

    incorrect_clusters = [c for c in clusters if c != "CORRECT"]
    counts_err = collections.Counter(incorrect_clusters)
    h_error_among_incorrect = entropy_bits(counts_err) if incorrect_clusters else None
    # H_error among ALL answers (correct answers contribute 0 "error" mass;
    # defined here as entropy of the full distribution restricted to error
    # clusters only, with correct treated as its own non-error mass -- i.e.
    # entropy computed over counts_err but normalized by n, not by n_incorrect)
    if incorrect_clusters:
        probs_all_norm = [c / n for c in counts_err.values()]
        h_error_among_all = -sum(p * math.log2(p) for p in probs_all_norm if p > 0)
    else:
        h_error_among_all = 0.0

    n_correct = counts_all.get("CORRECT", 0)
    pass_at_1 = n_correct / n if n else None
    dominant_share = max(counts_all.values()) / n if n else None
    dominant_cluster = counts_all.most_common(1)[0][0] if counts_all else None
    n_eff = 1.0 / sum((c / n) ** 2 for c in counts_all.values()) if n else None

    return {
        "n": n, "pass_at_1": round(pass_at_1, 4) if pass_at_1 is not None else None,
        "h_sem_bits": round(h_sem, 4) if h_sem is not None else None,
        "h_error_among_incorrect_bits": round(h_error_among_incorrect, 4) if h_error_among_incorrect is not None else None,
        "h_error_among_all_bits": round(h_error_among_all, 4),
        "n_unique_clusters": len(counts_all), "n_unique_error_clusters": len(counts_err),
        "dominant_cluster": dominant_cluster, "dominant_cluster_share": round(dominant_share, 4) if dominant_share is not None else None,
        "n_eff_clusters": round(n_eff, 3) if n_eff is not None else None,
        "cluster_counts": dict(counts_all),
    }


def oracle_and_g(rows, rng, n_values=N_VALUES, n_trials=N_TRIALS):
    """Subsampling without replacement from the real N=16 pool of one cell."""
    run_ids = [r["run_id"] for r in rows]
    correctness = {r["run_id"]: r["correctness"] for r in rows}
    pop = len(run_ids)
    out = {}
    for n in n_values:
        if n > pop:
            continue
        exact = (n == pop)
        trials = 1 if exact else n_trials
        successes = 0
        for _ in range(trials):
            sample = run_ids if exact else rng.sample(run_ids, n)
            if any(correctness[r] for r in sample):
                successes += 1
        out[str(n)] = round(successes / trials, 4)
    return out


def state_transition_stats(rows_sorted_by_seed):
    """Section 5: walk the seeds in a fixed (seed) order as a retrospective
    "what if we had stopped here" rollout -- generations are independent;
    order is an analysis choice, not a claim about sequential dependence."""
    seen_correct = False
    seen_error_sigs = set()
    counts = collections.Counter()
    for r in rows_sorted_by_seed:
        cl = cluster_of(r)
        if r["correctness"]:
            if seen_correct:
                counts["repeated_correct"] += 1
            else:
                counts["new_correct"] += 1
            seen_correct = True
        else:
            if cl in seen_error_sigs:
                counts["repeated_error"] += 1
            else:
                counts["new_error"] += 1
            seen_error_sigs.add(cl)
    total = len(rows_sorted_by_seed)
    return {k: round(v / total, 4) for k, v in counts.items()} | {"raw_counts": dict(counts), "n": total}


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b-q4_0-unsloth"
    rows = load_main_index(model_id)
    rows = [r for r in rows if not r["generation_failed"]]
    rng = random.Random(RNG_SEED)

    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["task_id"], r["temperature"])].append(r)

    cell_results = {}
    for (task_id, t), cell_rows in by_cell.items():
        cell_rows_sorted = sorted(cell_rows, key=lambda r: r["seed"])
        stats = cell_stats(cell_rows_sorted)
        oracle = oracle_and_g(cell_rows_sorted, rng)
        transitions = state_transition_stats(cell_rows_sorted)
        g_n = {n: round(v - stats["pass_at_1"], 4) if stats["pass_at_1"] is not None else None for n, v in oracle.items()}
        n_keys = sorted(oracle.keys(), key=int)
        delta_g = {}
        for i in range(1, len(n_keys)):
            a, b = n_keys[i - 1], n_keys[i]
            delta_g[f"{a}->{b}"] = round(oracle[b] - oracle[a], 4)

        tokens = [r["output_tokens"] or 0 for r in cell_rows]
        times = [r["generation_time_sec"] or 0 for r in cell_rows]
        cell_results[f"{task_id}/T{t}"] = {
            "task_id": task_id, "temperature": t, **stats,
            "oracle_at_n": oracle, "g_n": g_n, "delta_g": delta_g,
            "state_transitions": transitions,
            "total_output_tokens": sum(tokens), "total_time_sec": round(sum(times), 2),
            "mean_output_tokens": round(sum(tokens) / len(tokens), 1), "mean_time_sec": round(sum(times) / len(times), 3),
        }

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "metrics5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cell_results, f, ensure_ascii=False, indent=2)

    print(f"n_rows={len(rows)} n_cells={len(cell_results)}")
    print("wrote metrics5_summary.json")


if __name__ == "__main__":
    main()
