"""
Stage 3: token-level entropy trajectory shape features, for the small
illustrative logits sample (N=8, T=0.5, CODE_06=hard vs CODE_02=easy).
Does NOT average the whole trajectory to one number first -- keeps H(t)
and derives shape descriptors from it (spec section 3 requirement).
"""
import json
import os
import statistics
import sys

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment5"
RUNS_DIR = os.path.join(EXP_DIR, "runs")
METRICS_DIR = os.path.dirname(__file__)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_peaks(h, threshold=0.5, min_gap=3):
    """A 'peak' = a local run where entropy rises above threshold after
    being below it -- coarse, mechanical, not a claim of deep signal
    processing rigor."""
    peaks = []
    above = False
    last_peak = -min_gap
    for i, v in enumerate(h):
        if v >= threshold and not above:
            if i - last_peak >= min_gap:
                peaks.append(i)
                last_peak = i
            above = True
        elif v < threshold:
            above = False
    return peaks


def shape_features(h):
    n = len(h)
    if n == 0:
        return None
    mean_h = statistics.mean(h)
    median_h = statistics.median(h)
    max_h = max(h)
    sorted_h = sorted(h)
    p90 = sorted_h[int(0.9 * (n - 1))]
    var_h = statistics.pvariance(h) if n > 1 else 0.0
    start_h = statistics.mean(h[: max(1, n // 10)])
    end_h = statistics.mean(h[-max(1, n // 10):])
    # simple linear slope via least squares
    xs = list(range(n))
    mx = statistics.mean(xs)
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (v - mean_h) for x, v in zip(xs, h)) / denom if denom else 0.0
    peaks = count_peaks(h)
    return {
        "n_tokens": n, "mean_H": round(mean_h, 4), "median_H": round(median_h, 4),
        "max_H": round(max_h, 4), "p90_H": round(p90, 4), "variance_H": round(var_h, 4),
        "entropy_at_start": round(start_h, 4), "entropy_at_end": round(end_h, 4),
        "slope": round(slope, 6), "number_of_peaks": len(peaks), "position_of_peaks": peaks,
    }


def main():
    path = os.path.join(RUNS_DIR, "index.qwen3-1.7b-q4_0-unsloth.logits.jsonl")
    rows = load_jsonl(path)
    rows = [r for r in rows if r["temperature"] == 0.5 and r["task_id"] in ("CODE_06", "CODE_02") and not r["generation_failed"]]

    out = {}
    for r in rows:
        meta_path = os.path.join(RUNS_DIR, r["raw_path"], "meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        lp = meta.get("logprobs")
        if not lp:
            continue
        h = lp["entropy_per_token_bits"]
        feats = shape_features(h)
        feats["correctness"] = r["correctness"]
        feats["captured_mass_mean"] = round(statistics.mean(lp["captured_mass"]), 4)
        out[r["run_id"]] = feats

    by_task = {"CODE_02_easy": {}, "CODE_06_hard": {}}
    for run_id, feats in out.items():
        task = "CODE_02_easy" if "CODE_02" in run_id else "CODE_06_hard"
        by_task[task][run_id] = feats

    summary = {}
    for task, runs in by_task.items():
        if not runs:
            continue
        means = {k: statistics.mean(v[k] for v in runs.values()) for k in ("mean_H", "max_H", "p90_H", "variance_H", "number_of_peaks")}
        n_correct = sum(1 for v in runs.values() if v["correctness"])
        summary[task] = {"n_runs": len(runs), "n_correct": n_correct, **{f"avg_{k}": round(v, 4) for k, v in means.items()}}

    with open(os.path.join(METRICS_DIR, "trajectory5_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"per_run": out, "by_task": summary}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
