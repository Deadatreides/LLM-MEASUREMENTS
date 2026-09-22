"""
Metrics for Experiment 2 (artifact pipeline vs monolithic best-of-N).
Reads runs/{baseline,pipeline_runs,seams,artifacts}.<model_id>.jsonl for
every model found, merges them, and writes metrics/*.json.
"""
import collections
import glob
import json
import os

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment2"
RUNS_DIR = os.path.join(EXP_DIR, "runs")
METRICS_DIR = os.path.join(EXP_DIR, "metrics")

PASS = "MECHANICAL_PASS"


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


def merge_all(prefix):
    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, f"{prefix}.*.jsonl"))):
        rows.extend(load_jsonl(path))
    return rows


def models_present(baseline_rows, pipeline_rows):
    return sorted(set(r["model_id"] for r in baseline_rows) | set(r["model_id"] for r in pipeline_rows))


# ---------------------------------------------------------------------------
# BASELINE best-of-N
# ---------------------------------------------------------------------------

def compute_baseline(baseline_rows, artifacts_by_id, model_id, task_ids, n_values=(1, 2, 4, 8)):
    by_task = collections.defaultdict(list)
    for r in baseline_rows:
        if r["model_id"] == model_id:
            by_task[r["task_id"]].append(r)
    for t in by_task:
        by_task[t].sort(key=lambda r: r["sample_idx"])

    out = {}
    for n in n_values:
        n_solved = 0
        n_tasks = 0
        total_in = 0
        total_out = 0
        total_calls = 0
        for task_id in task_ids:
            samples = by_task.get(task_id, [])[:n]
            if not samples:
                continue
            n_tasks += 1
            solved = any(s["reference_status"] == PASS for s in samples)
            n_solved += int(solved)
            for s in samples:
                art = artifacts_by_id.get(s["artifact_id"])
                if art:
                    total_in += art["input_tokens"] or 0
                    total_out += art["output_tokens"] or 0
                total_calls += 1
        out[f"N={n}"] = {
            "n_tasks": n_tasks,
            "solved_rate": round(n_solved / n_tasks, 3) if n_tasks else None,
            "n_solved": n_solved,
            "avg_calls_per_task": round(total_calls / n_tasks, 2) if n_tasks else None,
            "avg_input_tokens_per_task": round(total_in / n_tasks, 1) if n_tasks else None,
            "avg_output_tokens_per_task": round(total_out / n_tasks, 1) if n_tasks else None,
            "avg_total_tokens_per_task": round((total_in + total_out) / n_tasks, 1) if n_tasks else None,
        }
    return out


# ---------------------------------------------------------------------------
# PIPELINE solved rate + budget, per variant
# ---------------------------------------------------------------------------

def compute_pipeline_variant(pipeline_rows, model_id, variant, task_ids):
    rows = [r for r in pipeline_rows if r["model_id"] == model_id and r["pipeline_variant"] == variant]
    if not rows:
        return None
    n = len(rows)
    n_initial_solved = sum(1 for r in rows if r["initial_solved"])
    n_final_solved = sum(1 for r in rows if r["final_solved"])
    avg_calls = sum(r["n_llm_calls"] for r in rows) / n
    avg_in = sum(r["total_input_tokens"] for r in rows) / n
    avg_out = sum(r["total_output_tokens"] for r in rows) / n
    avg_wall = sum(r["wall_time_sec"] for r in rows) / n
    avg_gen = sum(r["total_generation_time_sec"] for r in rows) / n
    mutation_scores = [r["mutation_score_a2"] for r in rows if r["mutation_score_a2"] is not None]

    # per-task solved (any attempt for that task solved) -- for a fair
    # comparison against baseline's per-task solved rate
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    n_tasks_final_solved = sum(1 for t in task_ids if any(r["final_solved"] for r in by_task.get(t, [])))
    n_tasks_initial_solved = sum(1 for t in task_ids if any(r["initial_solved"] for r in by_task.get(t, [])))
    n_tasks_present = sum(1 for t in task_ids if by_task.get(t))

    repaired = [r for r in rows if r["n_repairs_used"] > 0]
    n_repaired = len(repaired)
    n_repair_success = sum(1 for r in repaired if (not r["initial_solved"]) and r["final_solved"])
    n_repair_regressed = sum(1 for r in repaired if r["repair_regressed"])
    n_initial_fail = sum(1 for r in rows if not r["initial_solved"])

    return {
        "n_attempts": n,
        "attempt_level_initial_solved_rate": round(n_initial_solved / n, 3),
        "attempt_level_final_solved_rate": round(n_final_solved / n, 3),
        "task_level_solved_rate_any_attempt_initial": round(n_tasks_initial_solved / n_tasks_present, 3) if n_tasks_present else None,
        "task_level_solved_rate_any_attempt_final": round(n_tasks_final_solved / n_tasks_present, 3) if n_tasks_present else None,
        "n_tasks_present": n_tasks_present,
        "avg_llm_calls": round(avg_calls, 2),
        "avg_input_tokens": round(avg_in, 1),
        "avg_output_tokens": round(avg_out, 1),
        "avg_total_tokens": round(avg_in + avg_out, 1),
        "avg_wall_time_sec": round(avg_wall, 3),
        "avg_generation_time_sec": round(avg_gen, 3),
        "avg_mutation_score_a2": round(sum(mutation_scores) / len(mutation_scores), 3) if mutation_scores else None,
        "n_repaired_attempts": n_repaired,
        "patch_success_rate": round(n_repair_success / n_repaired, 3) if n_repaired else None,
        "patch_regression_rate": round(n_repair_regressed / n_repaired, 3) if n_repaired else None,
        "n_initial_fail": n_initial_fail,
        "economics": {
            "solved_task_per_1k_output_tokens": round(1000 * n_tasks_final_solved / (avg_out * n_tasks_present), 4) if avg_out and n_tasks_present else None,
            "solved_task_per_wall_sec": round(n_tasks_final_solved / (avg_wall * n_tasks_present), 5) if avg_wall and n_tasks_present else None,
        },
    }


# ---------------------------------------------------------------------------
# Drift categories (spec section 21) + CASE1-5 (spec section 10)
# ---------------------------------------------------------------------------

def compute_drift_and_cases(pipeline_rows, model_id, variant):
    rows = [r for r in pipeline_rows if r["model_id"] == model_id and r["pipeline_variant"] == variant]
    drift = collections.Counter()
    cases = collections.Counter()
    for r in rows:
        a2_good = r["seam_a2_reference_status"] == PASS
        a3_good = r["initial_reference_status"] == PASS
        seam_pass = r["initial_seam_a2_a3_status"] == PASS

        if not a3_good and not seam_pass:
            drift["USEFUL"] += 1
        elif not a3_good and seam_pass:
            drift["MISSED_DANGEROUS"] += 1
        elif a3_good and not seam_pass:
            drift["FALSE"] += 1
        else:
            drift["CORRECT_ACCEPT"] += 1

        if a2_good and not a3_good:
            cases["CASE1_a2good_a3bad"] += 1
            if not seam_pass:
                cases["CASE4_a2_detects_a3_error"] += 1
        if (not a2_good) and a3_good:
            cases["CASE2_a2bad_a3good"] += 1
        if (not a2_good) and (not a3_good) and seam_pass:
            cases["CASE3_both_bad_but_seam_passes"] += 1
        if seam_pass and not a3_good:
            cases["CASE5_seam_passes_but_reference_fails"] += 1

    n = len(rows)
    n_bad = drift["USEFUL"] + drift["MISSED_DANGEROUS"]
    n_good = drift["FALSE"] + drift["CORRECT_ACCEPT"]
    return {
        "n_attempts": n,
        "drift_counts": dict(drift),
        "drift_rate": round(n_bad / n, 3) if n else None,
        "drift_caught_rate": round(drift["USEFUL"] / n_bad, 3) if n_bad else None,
        "false_accept_rate": round(drift["MISSED_DANGEROUS"] / n_bad, 3) if n_bad else None,
        "false_reject_rate": round(drift["FALSE"] / n_good, 3) if n_good else None,
        "case_counts": dict(cases),
    }


# ---------------------------------------------------------------------------
def main():
    baseline_rows = merge_all("baseline")
    pipeline_rows = merge_all("pipeline_runs")
    artifact_rows = merge_all("artifacts")
    artifacts_by_id = {a["artifact_id"]: a for a in artifact_rows}

    task_ids = sorted(set(r["task_id"] for r in baseline_rows) | set(r["task_id"] for r in pipeline_rows))
    models = models_present(baseline_rows, pipeline_rows)

    result = {"models": models, "task_ids": task_ids, "baseline": {}, "pipeline": {}, "drift": {}}

    for model_id in models:
        result["baseline"][model_id] = compute_baseline(baseline_rows, artifacts_by_id, model_id, task_ids)
        result["pipeline"][model_id] = {}
        result["drift"][model_id] = {}
        for variant in ("PIPELINE_1", "PIPELINE_2", "PIPELINE_2R2", "PIPELINE_3"):
            pv = compute_pipeline_variant(pipeline_rows, model_id, variant, task_ids)
            if pv:
                result["pipeline"][model_id][variant] = pv
                result["drift"][model_id][variant] = compute_drift_and_cases(pipeline_rows, model_id, variant)

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "metrics2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"models={models}")
    print(f"baseline_rows={len(baseline_rows)} pipeline_rows={len(pipeline_rows)} artifact_rows={len(artifact_rows)}")
    print("wrote metrics/metrics2_summary.json")


if __name__ == "__main__":
    main()
