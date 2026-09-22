"""
Stage 9 (prompt diversity) and stage 11 (independent-model comparison),
built entirely from experiments 1 and 2's EXISTING data -- no new
generation. Both reused datasets are explicitly smaller/narrower than
this experiment's main series (see AUDIT.md); results here are labeled
accordingly and kept separate from the main N=16 findings.
"""
import collections
import json
import math
import os

EXP1_INDEX = r"<PROJECT_ROOT>\trace-probe\experiment\runs\index.jsonl"
EXP2_BASELINE_DIR = r"<PROJECT_ROOT>\trace-probe\experiment2\runs"
METRICS_DIR = os.path.dirname(__file__)

PASS_STATUSES = {"MECHANICAL_PASS", "REFERENCE_PASS"}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def entropy_bits(counts):
    total = sum(counts.values())
    if not total:
        return None
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def cluster_exp1(row):
    v = row["verification"]
    return "CORRECT" if v["status"] in PASS_STATUSES else (v["error_signature"] or f"UNSIG_{v['error_class']}")


# ---------------------------------------------------------------------------
# stage 9: prompt diversity (reused from experiment 1: qwen3, CODE_01-03, T=0, N=5, P1/P2/P3)
# ---------------------------------------------------------------------------

def prompt_diversity():
    rows = load_jsonl(EXP1_INDEX)
    rows = [r for r in rows if r["domain"] == "code" and r["model_id"] == "qwen3-1.7b-q4_0-unsloth"
            and r["condition_id"] == "T0_P1.0" and r["task_id"] in ("CODE_01", "CODE_02", "CODE_03")]

    by_prompt = collections.defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)

    out = {}
    for pid, prows in by_prompt.items():
        n = len(prows)
        n_correct = sum(1 for r in prows if r["verification"]["status"] in PASS_STATUSES)
        clusters = collections.Counter(cluster_exp1(r) for r in prows)
        out[pid] = {
            "n": n, "pass_rate": round(n_correct / n, 4) if n else None,
            "n_unique_clusters": len(clusters), "h_sem_bits": round(entropy_bits(clusters), 4) if clusters else None,
        }

    # same-model / same-prompt / different-T comparison at N=5 (from this experiment's own T sweep,
    # but truncated to N=5 for a fair apples-to-apples N comparison against the prompt series)
    return {"note": "reused from experiment 1: qwen3-1.7b, CODE_01-03 only, T=0 only, N=5 (< required N=8) -- small, illustrative, not the main series",
            "by_prompt": out}


# ---------------------------------------------------------------------------
# stage 11: independent-model comparison (reused from experiment 2 baseline: N=8, T=0.3, 4 models, 6 tasks)
# ---------------------------------------------------------------------------

def model_comparison():
    import glob
    baseline_rows = []
    for p in glob.glob(os.path.join(EXP2_BASELINE_DIR, "baseline.*.jsonl")):
        baseline_rows.extend(load_jsonl(p))
    artifact_rows = []
    for p in glob.glob(os.path.join(EXP2_BASELINE_DIR, "artifacts.*.jsonl")):
        artifact_rows.extend(load_jsonl(p))
    art_by_id = {a["artifact_id"]: a for a in artifact_rows}

    models = sorted(set(r["model_id"] for r in baseline_rows))
    tasks = sorted(set(r["task_id"] for r in baseline_rows))

    def cluster_exp2(r):
        return "CORRECT" if r["reference_status"] == "MECHANICAL_PASS" else (r["error_signature"] or f"UNSIG_{r['error_class']}")

    # same-model x8 (one model, all 8 samples) vs 4-models x2-each, per task
    same_model_oracle = collections.defaultdict(list)
    mixed_oracle = []
    same_model_diversity = collections.defaultdict(list)
    mixed_diversity = []

    by_model_task = collections.defaultdict(list)
    for r in baseline_rows:
        by_model_task[(r["model_id"], r["task_id"])].append(r)

    for task in tasks:
        for m in models:
            rows = by_model_task.get((m, task), [])
            if len(rows) < 8:
                continue
            has_correct = any(r["reference_status"] == "MECHANICAL_PASS" for r in rows)
            same_model_oracle[m].append(has_correct)
            clusters = collections.Counter(cluster_exp2(r) for r in rows)
            same_model_diversity[m].append(len(clusters))

        # mixed: first 2 samples from each of the 4 models
        mixed_rows = []
        ok = True
        for m in models:
            rows = by_model_task.get((m, task), [])
            if len(rows) < 2:
                ok = False
                break
            mixed_rows.extend(sorted(rows, key=lambda r: r["sample_idx"])[:2])
        if ok:
            has_correct = any(r["reference_status"] == "MECHANICAL_PASS" for r in mixed_rows)
            mixed_oracle.append(has_correct)
            clusters = collections.Counter(cluster_exp2(r) for r in mixed_rows)
            mixed_diversity.append(len(clusters))

    out = {
        "note": "reused from experiment 2 baseline: N=8, T=0.3 fixed, 4 models, 6 tasks -- no new generation",
        "same_model_x8_oracle": {m: round(sum(v) / len(v), 4) for m, v in same_model_oracle.items()},
        "same_model_x8_mean_n_clusters": {m: round(sum(v) / len(v), 3) for m, v in same_model_diversity.items()},
        "mixed_4models_x2_oracle": round(sum(mixed_oracle) / len(mixed_oracle), 4) if mixed_oracle else None,
        "mixed_4models_x2_mean_n_clusters": round(sum(mixed_diversity) / len(mixed_diversity), 3) if mixed_diversity else None,
        "n_tasks_used": len(tasks),
    }
    return out


def main():
    out = {"prompt_diversity_stage9": prompt_diversity(), "model_comparison_stage11": model_comparison()}
    with open(os.path.join(METRICS_DIR, "cross_reuse5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
