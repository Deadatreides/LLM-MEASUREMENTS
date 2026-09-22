"""
Computes all the metrics described in the experiment spec from
runs/index_enriched.jsonl (run enrich_index.py first). Writes several small
JSON/JSONL files under experiment/metrics/. Does not read any RAW files --
everything here works from the compact index only, as required.

Deliberately keeps TEXT_DIVERSITY (distinct raw outputs), ANSWER_DIVERSITY
(distinct extracted final answers) and ERROR_DIVERSITY (distinct verified
error signatures) as separate numbers -- they measure different things and
must not be collapsed into one "diversity" score (see spec section 33).
"""
import collections
import itertools
import json
import math
import os

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment"
METRICS_DIR = os.path.join(EXP_DIR, "metrics")

PASS_STATUSES = {"REFERENCE_PASS", "MECHANICAL_PASS"}
FAIL_STATUSES = {"REFERENCE_FAIL", "MECHANICAL_FAIL"}
# UNVERIFIED / TECHNICAL_FAILURE are excluded from pass/fail accuracy math
# entirely; counted separately so they are never silently folded into
# "incorrect".

MIN_N_FOR_STATS = 20  # below this, aggregates are flagged INSUFFICIENT_SAMPLE

TEMPERATURE_CONDITIONS = ["T0_P1.0", "T0.3_P1.0", "T0.7_P1.0", "T1.0_P1.0"]
PROMPT_IDS = ["P1", "P2", "P3"]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (round(max(0.0, lo), 4), round(min(1.0, hi), 4))


def status_of(row):
    s = row["verification"]["status"]
    if s in PASS_STATUSES:
        return "PASS"
    if s in FAIL_STATUSES:
        return "FAIL"
    return None  # UNVERIFIED / TECHNICAL_FAILURE -> excluded from vectors


def norm_answer(row):
    a = row["result"]["extracted_final_answer"]
    if a is None:
        return None
    return a.strip().lower()


# ---------------------------------------------------------------------------
# 1. CELL-LEVEL metrics: one row per (model, task, prompt, condition)
# ---------------------------------------------------------------------------

def compute_cell_metrics(rows):
    cells = collections.defaultdict(list)
    for r in rows:
        key = (r["model_id"], r["task_id"], r["prompt_id"], r["condition_id"])
        cells[key].append(r)

    out = []
    for (model_id, task_id, prompt_id, condition_id), crows in cells.items():
        n_repeats = len(crows)
        valid = [r for r in crows if status_of(r) is not None]
        n_valid = len(valid)
        n_pass = sum(1 for r in valid if status_of(r) == "PASS")
        n_fail = n_valid - n_pass
        n_technical = sum(
            1 for r in crows if r["verification"]["status"] in ("TECHNICAL_FAILURE", "UNVERIFIED")
        )

        output_hashes = [r.get("output_text_hash") for r in crows if r.get("output_text_hash")]
        unique_outputs = len(set(output_hashes))

        answers = [norm_answer(r) for r in crows if norm_answer(r) is not None]
        unique_answers = len(set(answers))

        fail_rows = [r for r in valid if status_of(r) == "FAIL"]
        error_classes = [r["verification"]["error_class"] for r in fail_rows if r["verification"]["error_class"]]
        error_sigs = [r["verification"]["error_signature"] for r in fail_rows if r["verification"]["error_signature"]]
        unique_error_classes = len(set(error_classes))
        unique_error_sigs = len(set(error_sigs))
        if error_sigs:
            sig_counts = collections.Counter(error_sigs)
            error_persistence = round(max(sig_counts.values()) / len(error_sigs), 3)
        else:
            error_persistence = None

        out.append(
            {
                "model_id": model_id,
                "task_id": task_id,
                "prompt_id": prompt_id,
                "condition_id": condition_id,
                "n_repeats": n_repeats,
                "n_valid": n_valid,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "n_technical_or_unverified": n_technical,
                "pass_rate": round(n_pass / n_valid, 3) if n_valid else None,
                "text_diversity_unique_outputs": unique_outputs,
                "text_diversity_ratio": round(unique_outputs / n_repeats, 3) if n_repeats else None,
                "answer_diversity_unique_answers": unique_answers,
                "answer_diversity_ratio": round(unique_answers / len(answers), 3) if answers else None,
                "error_diversity_unique_classes": unique_error_classes,
                "error_diversity_unique_signatures": unique_error_sigs,
                "error_persistence": error_persistence,
            }
        )
    return out


# ---------------------------------------------------------------------------
# 2. MODEL-LEVEL summary
# ---------------------------------------------------------------------------

def compute_model_summary(rows):
    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model_id"]].append(r)

    out = {}
    for model_id, mrows in by_model.items():
        valid = [r for r in mrows if status_of(r) is not None]
        n_pass = sum(1 for r in valid if status_of(r) == "PASS")
        n_valid = len(valid)
        n_tech = sum(1 for r in mrows if r["verification"]["status"] == "TECHNICAL_FAILURE")
        n_unverified = sum(1 for r in mrows if r["verification"]["status"] == "UNVERIFIED")
        n_gen_failed = sum(1 for r in mrows if r["result_status"] == "generation_failed")

        gen_times = [r["performance"]["generation_time_sec"] for r in mrows if r["performance"]["generation_time_sec"] is not None]
        out_tokens = [r["performance"]["output_tokens"] for r in mrows if r["performance"]["output_tokens"]]
        tps = [r["performance"]["tokens_per_sec"] for r in mrows if r["performance"]["tokens_per_sec"]]

        by_domain = collections.defaultdict(lambda: {"pass": 0, "valid": 0})
        for r in valid:
            d = r["domain"]
            by_domain[d]["valid"] += 1
            if status_of(r) == "PASS":
                by_domain[d]["pass"] += 1

        lo, hi = wilson_ci(n_pass, n_valid) if n_valid else (None, None)

        out[model_id] = {
            "n_rows_total": len(mrows),
            "n_valid": n_valid,
            "n_pass": n_pass,
            "n_technical_failure": n_tech,
            "n_unverified": n_unverified,
            "n_generation_failed": n_gen_failed,
            "overall_pass_rate": round(n_pass / n_valid, 3) if n_valid else None,
            "overall_pass_rate_95ci": [lo, hi],
            "insufficient_sample": n_valid < MIN_N_FOR_STATS,
            "pass_rate_by_domain": {
                d: {
                    "pass_rate": round(v["pass"] / v["valid"], 3) if v["valid"] else None,
                    "n": v["valid"],
                    "insufficient_sample": v["valid"] < MIN_N_FOR_STATS,
                }
                for d, v in by_domain.items()
            },
            "mean_generation_time_sec": round(sum(gen_times) / len(gen_times), 3) if gen_times else None,
            "mean_output_tokens": round(sum(out_tokens) / len(out_tokens), 1) if out_tokens else None,
            "mean_tokens_per_sec": round(sum(tps) / len(tps), 2) if tps else None,
            "total_generation_time_sec": round(sum(gen_times), 1) if gen_times else None,
        }
    return out


# ---------------------------------------------------------------------------
# 3. Temperature comparison (T=0 vs T>0), per model
# ---------------------------------------------------------------------------

def compute_temperature_comparison(rows, cell_metrics):
    # restrict to the top_p=1.0 temperature sweep, P1 prompt only (isolates
    # the temperature factor without also varying prompt)
    cells_by_model = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in cell_metrics:
        if c["condition_id"] in TEMPERATURE_CONDITIONS and c["prompt_id"] == "P1":
            cells_by_model[c["model_id"]][c["condition_id"]].append(c)

    out = {}
    for model_id, by_cond in cells_by_model.items():
        cond_stats = {}
        for cond_id in TEMPERATURE_CONDITIONS:
            ccells = by_cond.get(cond_id, [])
            n_valid = sum(c["n_valid"] for c in ccells)
            n_pass = sum(c["n_pass"] for c in ccells)
            mean_text_div = (
                round(sum(c["text_diversity_ratio"] for c in ccells if c["text_diversity_ratio"] is not None) / len(ccells), 3)
                if ccells else None
            )
            mean_err_div = (
                round(sum(c["error_diversity_unique_signatures"] for c in ccells) / len(ccells), 3)
                if ccells else None
            )
            cond_stats[cond_id] = {
                "n_valid": n_valid,
                "pass_rate": round(n_pass / n_valid, 3) if n_valid else None,
                "mean_text_diversity_ratio": mean_text_div,
                "mean_unique_error_signatures_per_cell": mean_err_div,
                "insufficient_sample": n_valid < MIN_N_FOR_STATS,
            }
        out[model_id] = cond_stats

    # error signature set churn: T0 vs each T>0, per model, unioned across tasks
    churn = {}
    rows_by_model_cond_task = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        if r["condition_id"] in TEMPERATURE_CONDITIONS and r["prompt_id"] == "P1" and status_of(r) == "FAIL":
            sig = r["verification"]["error_signature"]
            if sig:
                rows_by_model_cond_task[r["model_id"]][(r["condition_id"], r["task_id"])].add(sig)

    for model_id in cells_by_model:
        model_churn = {}
        base_sigs = set()
        for (cond, task), sigs in rows_by_model_cond_task[model_id].items():
            if cond == "T0_P1.0":
                base_sigs |= sigs
        for cond_id in TEMPERATURE_CONDITIONS[1:]:
            other_sigs = set()
            for (cond, task), sigs in rows_by_model_cond_task[model_id].items():
                if cond == cond_id:
                    other_sigs |= sigs
            model_churn[cond_id] = {
                "new_error_signatures_vs_T0": sorted(other_sigs - base_sigs),
                "disappeared_error_signatures_vs_T0": sorted(base_sigs - other_sigs),
                "shared_error_signatures": sorted(base_sigs & other_sigs),
            }
        churn[model_id] = model_churn

    return {"per_condition_stats": out, "error_signature_churn_vs_T0": churn}


# ---------------------------------------------------------------------------
# 4. Prompt comparison (P1 vs P2 vs P3), per model
# ---------------------------------------------------------------------------

def compute_prompt_comparison(cell_metrics):
    cells_by_model = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in cell_metrics:
        if c["condition_id"] == "T0_P1.0":  # hold temperature fixed at the deterministic baseline
            cells_by_model[c["model_id"]][c["prompt_id"]].append(c)

    out = {}
    for model_id, by_prompt in cells_by_model.items():
        pstats = {}
        for pid in PROMPT_IDS:
            pcells = by_prompt.get(pid, [])
            n_valid = sum(c["n_valid"] for c in pcells)
            n_pass = sum(c["n_pass"] for c in pcells)
            mean_text_div = (
                round(sum(c["text_diversity_ratio"] for c in pcells if c["text_diversity_ratio"] is not None) / len(pcells), 3)
                if pcells else None
            )
            pstats[pid] = {
                "n_valid": n_valid,
                "pass_rate": round(n_pass / n_valid, 3) if n_valid else None,
                "mean_text_diversity_ratio": mean_text_div,
                "insufficient_sample": n_valid < MIN_N_FOR_STATS,
            }
        out[model_id] = pstats
    return out


# ---------------------------------------------------------------------------
# 5. Pairwise error-correlation (the main metric, spec section 16)
# ---------------------------------------------------------------------------

def build_vectors(rows):
    """(model,prompt,condition) -> {repeat_idx: {task_id: 'PASS'/'FAIL'}}"""
    vecs = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        st = status_of(r)
        if st is None:
            continue
        key = (r["model_id"], r["prompt_id"], r["condition_id"])
        vecs[key][r["repeat_index"]][r["task_id"]] = st
    return vecs


def pair_stats(vec_a, vec_b):
    shared_tasks = set(vec_a) & set(vec_b)
    if not shared_tasks:
        return None
    agree = 0
    both_fail = 0
    a_fail = 0
    b_fail = 0
    a_pass = 0
    a_pass_b_fail = 0
    for t in shared_tasks:
        sa, sb = vec_a[t], vec_b[t]
        if sa == sb:
            agree += 1
        if sa == "FAIL":
            a_fail += 1
        if sa == "PASS":
            a_pass += 1
            if sb == "FAIL":
                a_pass_b_fail += 1
        if sb == "FAIL":
            b_fail += 1
        if sa == "FAIL" and sb == "FAIL":
            both_fail += 1
    union_fail = a_fail + b_fail - both_fail
    return {
        "n_tasks": len(shared_tasks),
        "agree": agree,
        "both_fail": both_fail,
        "a_fail": a_fail,
        "b_fail": b_fail,
        "a_pass": a_pass,
        "a_pass_b_fail": a_pass_b_fail,
        "union_fail": union_fail,
    }


def aggregate_pair_stats(pair_results):
    n_tasks = sum(p["n_tasks"] for p in pair_results)
    if n_tasks == 0:
        return None
    agree = sum(p["agree"] for p in pair_results)
    both_fail = sum(p["both_fail"] for p in pair_results)
    a_fail = sum(p["a_fail"] for p in pair_results)
    b_fail = sum(p["b_fail"] for p in pair_results)
    union_fail = sum(p["union_fail"] for p in pair_results)
    a_pass = sum(p["a_pass"] for p in pair_results)
    a_pass_b_fail = sum(p["a_pass_b_fail"] for p in pair_results)
    return {
        "n_pairs": len(pair_results),
        "n_task_comparisons": n_tasks,
        "pairwise_agreement": round(agree / n_tasks, 3),
        "jaccard_error_overlap": round(both_fail / union_fail, 3) if union_fail else None,
        "p_error_b_given_error_a": round(both_fail / a_fail, 3) if a_fail else None,
        "p_success_b_given_error_a": round((a_fail - both_fail) / a_fail, 3) if a_fail else None,
        "p_error_b_given_success_a": round(a_pass_b_fail / a_pass, 3) if a_pass else None,
        "insufficient_sample": n_tasks < MIN_N_FOR_STATS,
    }


def compute_correlation(rows, per_model=True):
    vecs = build_vectors(rows)

    categories = {
        "same_settings": [],
        "diff_temperature_same_prompt": [],
        "diff_prompt_same_temperature": [],
        "diff_temperature_and_prompt": [],
    }
    per_model_categories = collections.defaultdict(lambda: {k: [] for k in categories})

    keys = list(vecs.keys())
    for i, key_a in enumerate(keys):
        model_a, prompt_a, cond_a = key_a
        for key_b in keys[i:]:
            model_b, prompt_b, cond_b = key_b
            if model_a != model_b:
                continue  # correlation is always within the SAME model
            same_prompt = prompt_a == prompt_b
            same_cond = cond_a == cond_b

            reps_a = vecs[key_a]
            reps_b = vecs[key_b]

            if same_prompt and same_cond:
                # same settings: only compare distinct repeat indices (i<j), avoid self-pairs
                for ra, rb in itertools.combinations(sorted(reps_a.keys()), 2):
                    ps = pair_stats(reps_a[ra], reps_b[rb])
                    if ps:
                        categories["same_settings"].append(ps)
                        per_model_categories[model_a]["same_settings"].append(ps)
            elif same_prompt and not same_cond:
                for ra in reps_a:
                    for rb in reps_b:
                        ps = pair_stats(reps_a[ra], reps_b[rb])
                        if ps:
                            categories["diff_temperature_same_prompt"].append(ps)
                            per_model_categories[model_a]["diff_temperature_same_prompt"].append(ps)
            elif not same_prompt and same_cond:
                for ra in reps_a:
                    for rb in reps_b:
                        ps = pair_stats(reps_a[ra], reps_b[rb])
                        if ps:
                            categories["diff_prompt_same_temperature"].append(ps)
                            per_model_categories[model_a]["diff_prompt_same_temperature"].append(ps)
            else:
                for ra in reps_a:
                    for rb in reps_b:
                        ps = pair_stats(reps_a[ra], reps_b[rb])
                        if ps:
                            categories["diff_temperature_and_prompt"].append(ps)
                            per_model_categories[model_a]["diff_temperature_and_prompt"].append(ps)

    def finalize(cat_dict):
        result = {}
        for cat, pair_list in cat_dict.items():
            agg = aggregate_pair_stats(pair_list)
            result[cat] = agg
        return result

    out = {"overall": finalize(categories)}
    if per_model:
        out["per_model"] = {m: finalize(cats) for m, cats in per_model_categories.items()}
    return out


# ---------------------------------------------------------------------------
def main():
    enriched_path = os.path.join(EXP_DIR, "runs", "index_enriched.jsonl")
    if not os.path.exists(enriched_path):
        raise SystemExit("run enrich_index.py first")
    rows = load_jsonl(enriched_path)

    os.makedirs(METRICS_DIR, exist_ok=True)

    cell_metrics = compute_cell_metrics(rows)
    with open(os.path.join(METRICS_DIR, "cell_metrics.jsonl"), "w", encoding="utf-8") as f:
        for c in cell_metrics:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    model_summary = compute_model_summary(rows)
    with open(os.path.join(METRICS_DIR, "model_summary.json"), "w", encoding="utf-8") as f:
        json.dump(model_summary, f, ensure_ascii=False, indent=2)

    temp_comp = compute_temperature_comparison(rows, cell_metrics)
    with open(os.path.join(METRICS_DIR, "temperature_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(temp_comp, f, ensure_ascii=False, indent=2)

    prompt_comp = compute_prompt_comparison(cell_metrics)
    with open(os.path.join(METRICS_DIR, "prompt_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(prompt_comp, f, ensure_ascii=False, indent=2)

    correlation = compute_correlation(rows)
    with open(os.path.join(METRICS_DIR, "error_correlation.json"), "w", encoding="utf-8") as f:
        json.dump(correlation, f, ensure_ascii=False, indent=2)

    print(f"n_rows={len(rows)} n_cells={len(cell_metrics)} n_models={len(model_summary)}")
    print("Wrote: cell_metrics.jsonl, model_summary.json, temperature_comparison.json, "
          "prompt_comparison.json, error_correlation.json")


if __name__ == "__main__":
    main()
