"""
Core metrics for Experiment 3: per-run accuracy, pairwise error
correlation (agreement / Jaccard / conditional probabilities), error
clustering + N_eff, and the three comparison series (temperature / prompt
/ model). Works entirely from runs/a1_index.*.jsonl + runs/claims_index.jsonl
-- no RAW re-reading needed.
"""
import collections
import glob
import itertools
import json
import math
import os

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment3"
RUNS_DIR = os.path.join(EXP_DIR, "runs")
METRICS_DIR = os.path.join(EXP_DIR, "metrics")

DETERMINATE = ("CORRECT", "INCORRECT")


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


def load_runs():
    rows = []
    for shard in sorted(glob.glob(os.path.join(RUNS_DIR, "a1_index.*.jsonl"))):
        rows.extend(load_jsonl(shard))
    return [r for r in rows if not r["generation_failed"]]


def load_claims():
    return load_jsonl(os.path.join(RUNS_DIR, "claims_index.jsonl"))


def build_vectors(claim_rows):
    """run_id -> {claim_id: classification}"""
    vecs = collections.defaultdict(dict)
    for c in claim_rows:
        vecs[c["run_id"]][c["claim_id"]] = c["classification"]
    return vecs


def run_accuracy(vec):
    vals = list(vec.values())
    n = len(vals)
    n_correct = sum(1 for v in vals if v == "CORRECT")
    n_addr = sum(1 for v in vals if v in DETERMINATE)
    return {
        "accuracy_strict": round(n_correct / n, 3) if n else None,  # OMITTED/AMBIGUOUS count against
        "accuracy_among_addressed": round(n_correct / n_addr, 3) if n_addr else None,
        "n_correct": n_correct, "n_incorrect": sum(1 for v in vals if v == "INCORRECT"),
        "n_omitted": sum(1 for v in vals if v == "OMITTED"), "n_ambiguous": sum(1 for v in vals if v == "AMBIGUOUS"),
    }


def pair_stats(vec_a, vec_b):
    shared = [k for k in vec_a if k in vec_b and vec_a[k] in DETERMINATE and vec_b[k] in DETERMINATE]
    if not shared:
        return None
    a_err = {k for k in shared if vec_a[k] == "INCORRECT"}
    b_err = {k for k in shared if vec_b[k] == "INCORRECT"}
    both_err = a_err & b_err
    union_err = a_err | b_err
    agree = sum(1 for k in shared if (vec_a[k] == "INCORRECT") == (vec_b[k] == "INCORRECT"))
    return {
        "n_shared_claims": len(shared),
        "agreement": round(agree / len(shared), 3),
        "jaccard_error": round(len(both_err) / len(union_err), 3) if union_err else None,
        "p_error_b_given_error_a": round(len(both_err) / len(a_err), 3) if a_err else None,
        "p_error_a_given_error_b": round(len(both_err) / len(b_err), 3) if b_err else None,
        "n_a_err": len(a_err), "n_b_err": len(b_err), "n_both_err": len(both_err),
        "both_correct": sum(1 for k in shared if vec_a[k] == "CORRECT" and vec_b[k] == "CORRECT"),
        "both_incorrect": len(both_err),
        "one_correct_one_incorrect": sum(1 for k in shared if (vec_a[k] == "CORRECT") != (vec_b[k] == "CORRECT")),
    }


def aggregate_pairs(pair_list):
    if not pair_list:
        return None
    n_pairs = len(pair_list)
    total_shared = sum(p["n_shared_claims"] for p in pair_list)
    total_agree = sum(round(p["agreement"] * p["n_shared_claims"]) for p in pair_list)
    total_both_err = sum(p["n_both_err"] for p in pair_list)
    total_a_err = sum(p["n_a_err"] for p in pair_list)
    total_b_err = sum(p["n_b_err"] for p in pair_list)
    total_union_err = total_a_err + total_b_err - total_both_err
    both_correct = sum(p["both_correct"] for p in pair_list)
    both_incorrect = sum(p["both_incorrect"] for p in pair_list)
    one_one = sum(p["one_correct_one_incorrect"] for p in pair_list)
    return {
        "n_pairs": n_pairs, "n_claim_comparisons": total_shared,
        "mean_agreement": round(total_agree / total_shared, 3) if total_shared else None,
        "jaccard_error": round(total_both_err / total_union_err, 3) if total_union_err else None,
        "p_error_b_given_error_a": round(total_both_err / total_a_err, 3) if total_a_err else None,
        "p_error_a_given_error_b": round(total_both_err / total_b_err, 3) if total_b_err else None,
        "both_correct_rate": round(both_correct / total_shared, 3) if total_shared else None,
        "both_incorrect_rate": round(both_incorrect / total_shared, 3) if total_shared else None,
        "one_correct_one_incorrect_rate": round(one_one / total_shared, 3) if total_shared else None,
    }


def error_clusters(claim_rows, subset_run_ids=None):
    """cluster identity = (task_id, claim_id); size = # runs where INCORRECT."""
    counts = collections.Counter()
    for c in claim_rows:
        if subset_run_ids is not None and c["run_id"] not in subset_run_ids:
            continue
        if c["classification"] == "INCORRECT":
            counts[(c["task_id"], c["claim_id"])] += 1
    total = sum(counts.values())
    if total == 0:
        return {"n_unique_clusters": 0, "total_error_instances": 0, "largest_cluster_share": None, "n_eff": None, "clusters": {}}
    n_eff = 1.0 / sum((v / total) ** 2 for v in counts.values())
    return {
        "n_unique_clusters": len(counts),
        "total_error_instances": total,
        "largest_cluster_share": round(max(counts.values()) / total, 3),
        "n_eff": round(n_eff, 2),
        "clusters": {f"{k[0]}/{k[1]}": v for k, v in counts.most_common()},
    }


def main():
    runs = load_runs()
    claim_rows = load_claims()
    vecs = build_vectors(claim_rows)
    runs_by_id = {r["run_id"]: r for r in runs}

    # -- per-run accuracy --------------------------------------------------
    acc_rows = []
    for run_id, vec in vecs.items():
        r = runs_by_id.get(run_id)
        if not r:
            continue
        a = run_accuracy(vec)
        acc_rows.append({**a, "run_id": run_id, "task_id": r["task_id"], "model_id": r["model_id"],
                          "prompt_id": r["prompt_id"], "temperature": r["temperature"], "seed": r["seed"]})

    # -- cells: (task_id, model_id, prompt_id, temperature) -> [run_id,...]
    cells = collections.defaultdict(list)
    for r in runs:
        key = (r["task_id"], r["model_id"], r["prompt_id"], r["temperature"])
        cells[key].append(r["run_id"])

    # same-settings pairwise stats per cell (different seeds)
    same_settings_pairs = []
    per_cell_pairs = {}
    for key, run_ids in cells.items():
        pl = []
        for ra, rb in itertools.combinations(sorted(run_ids), 2):
            ps = pair_stats(vecs[ra], vecs[rb])
            if ps:
                pl.append(ps)
                same_settings_pairs.append(ps)
        if pl:
            per_cell_pairs[key] = aggregate_pairs(pl)

    result = {
        "n_runs": len(runs), "n_claim_rows": len(claim_rows),
        "accuracy_overall": {
            "mean_accuracy_strict": round(sum(a["accuracy_strict"] for a in acc_rows if a["accuracy_strict"] is not None) / len(acc_rows), 3) if acc_rows else None,
        },
        "same_settings_aggregate": aggregate_pairs(same_settings_pairs),
        "overall_error_clusters": error_clusters(claim_rows),
    }

    # -- SERIES A: same model/prompt, different temperature ---------------
    # The primary model is the one run across the full temperature sweep
    # (series A) -- identify it that way rather than picking an arbitrary
    # P1 run, since several other models also have single P1@T0.5 cells
    # (series C) and file-load order must not silently determine this.
    temp_counts = collections.Counter(r["model_id"] for r in runs if r["prompt_id"] == "P1")
    primary_model = temp_counts.most_common(1)[0][0]
    temps = sorted(set(r["temperature"] for r in runs if r["model_id"] == primary_model))
    series_a = {}
    for t in temps:
        run_ids_at_t = [r["run_id"] for r in runs if r["prompt_id"] == "P1" and r["temperature"] == t and r["model_id"] == primary_model]
        accs = [a["accuracy_strict"] for a in acc_rows if a["run_id"] in run_ids_at_t]
        # pairwise within this T (across tasks' cells, same T, same prompt, same model)
        pl = []
        for key, run_ids in cells.items():
            if key[1] == primary_model and key[2] == "P1" and key[3] == t:
                for ra, rb in itertools.combinations(sorted(run_ids), 2):
                    ps = pair_stats(vecs[ra], vecs[rb])
                    if ps:
                        pl.append(ps)
        text_hashes = set(r["result_hash"] for r in runs if r["run_id"] in run_ids_at_t and r["result_hash"])
        series_a[str(t)] = {
            "n_runs": len(run_ids_at_t),
            "mean_accuracy": round(sum(accs) / len(accs), 3) if accs else None,
            "n_unique_raw_texts": len(text_hashes),
            "text_diversity_ratio": round(len(text_hashes) / len(run_ids_at_t), 3) if run_ids_at_t else None,
            "pairwise": aggregate_pairs(pl),
            "error_clusters": error_clusters(claim_rows, subset_run_ids=set(run_ids_at_t)),
        }
    result["series_A_temperature_sweep"] = {"fixed_model": primary_model, "fixed_prompt": "P1", "by_temperature": series_a}

    # -- SERIES B: same model/temperature, different prompt ----------------
    mid_t = 0.5
    series_b = {}
    for p in ("P1", "P2", "P3"):
        run_ids = [r["run_id"] for r in runs if r["model_id"] == primary_model and r["temperature"] == mid_t and r["prompt_id"] == p]
        accs = [a["accuracy_strict"] for a in acc_rows if a["run_id"] in run_ids]
        pl = []
        for key, rids in cells.items():
            if key[1] == primary_model and key[3] == mid_t and key[2] == p:
                for ra, rb in itertools.combinations(sorted(rids), 2):
                    ps = pair_stats(vecs[ra], vecs[rb])
                    if ps:
                        pl.append(ps)
        text_hashes = set(r["result_hash"] for r in runs if r["run_id"] in run_ids and r["result_hash"])
        series_b[p] = {
            "n_runs": len(run_ids), "mean_accuracy": round(sum(accs) / len(accs), 3) if accs else None,
            "n_unique_raw_texts": len(text_hashes),
            "text_diversity_ratio": round(len(text_hashes) / len(run_ids), 3) if run_ids else None,
            "pairwise": aggregate_pairs(pl),
            "error_clusters": error_clusters(claim_rows, subset_run_ids=set(run_ids)),
        }
    result["series_B_prompt_sweep"] = {"fixed_model": primary_model, "fixed_temperature": mid_t, "by_prompt": series_b}

    # -- SERIES C: same prompt/temperature, different model ----------------
    models_present = sorted(set(r["model_id"] for r in runs if r["temperature"] == mid_t and r["prompt_id"] == "P1"))
    series_c = {}
    for m in models_present:
        run_ids = [r["run_id"] for r in runs if r["model_id"] == m and r["temperature"] == mid_t and r["prompt_id"] == "P1"]
        accs = [a["accuracy_strict"] for a in acc_rows if a["run_id"] in run_ids]
        pl = []
        for key, rids in cells.items():
            if key[1] == m and key[3] == mid_t and key[2] == "P1":
                for ra, rb in itertools.combinations(sorted(rids), 2):
                    ps = pair_stats(vecs[ra], vecs[rb])
                    if ps:
                        pl.append(ps)
        text_hashes = set(r["result_hash"] for r in runs if r["run_id"] in run_ids and r["result_hash"])
        series_c[m] = {
            "n_runs": len(run_ids), "mean_accuracy": round(sum(accs) / len(accs), 3) if accs else None,
            "n_unique_raw_texts": len(text_hashes),
            "text_diversity_ratio": round(len(text_hashes) / len(run_ids), 3) if run_ids else None,
            "pairwise": aggregate_pairs(pl),
            "error_clusters": error_clusters(claim_rows, subset_run_ids=set(run_ids)),
        }
    result["series_C_model_sweep"] = {"fixed_prompt": "P1", "fixed_temperature": mid_t, "by_model": series_c}

    # -- cross-model pairwise (M_i vs M_j at same prompt/temperature) ------
    cross_model = {}
    for ma, mb in itertools.combinations(models_present, 2):
        pl = []
        for task_id in sorted(set(r["task_id"] for r in runs)):
            rids_a = [r["run_id"] for r in runs if r["model_id"] == ma and r["temperature"] == mid_t and r["prompt_id"] == "P1" and r["task_id"] == task_id]
            rids_b = [r["run_id"] for r in runs if r["model_id"] == mb and r["temperature"] == mid_t and r["prompt_id"] == "P1" and r["task_id"] == task_id]
            for ra in rids_a:
                for rb in rids_b:
                    ps = pair_stats(vecs[ra], vecs[rb])
                    if ps:
                        pl.append(ps)
        cross_model[f"{ma}__vs__{mb}"] = aggregate_pairs(pl)
    result["cross_model_pairwise"] = cross_model

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "metrics3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(METRICS_DIR, "run_accuracy.jsonl"), "w", encoding="utf-8") as f:
        for a in acc_rows:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"n_runs={len(runs)} n_claims={len(claim_rows)} models={models_present}")
    print("wrote metrics3_summary.json, run_accuracy.jsonl")


if __name__ == "__main__":
    main()
