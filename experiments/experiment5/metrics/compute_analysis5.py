"""
Experiment 5, stages 6/7/8/10/12/14: entropy quartiles, A/B/C/D quadrants,
temperature comparison, Pearson/Spearman correlations with bootstrap CIs,
systematic-trap detection, and budget/information-gain-per-token.

Reads metrics/metrics5_summary.json (one row per (task_id, temperature)
cell, written by compute_metrics5.py).
"""
import json
import math
import os
import random
import statistics
import sys

METRICS_DIR = os.path.dirname(__file__)
RNG_SEED = 20260816
N_BOOTSTRAP = 2000


def load_cells():
    with open(os.path.join(METRICS_DIR, "metrics5_summary.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# stage 6: entropy quartiles
# ---------------------------------------------------------------------------

def quartile_analysis(cells):
    items = [(k, v) for k, v in cells.items() if v["h_sem_bits"] is not None]
    items.sort(key=lambda kv: kv[1]["h_sem_bits"])
    n = len(items)
    q_bounds = [0, n // 4, n // 2, (3 * n) // 4, n]
    quartiles = {}
    labels = ["Q1_low_H", "Q2", "Q3", "Q4_high_H"]
    for i, label in enumerate(labels):
        chunk = items[q_bounds[i]:q_bounds[i + 1]]
        if not chunk:
            continue
        pass1 = [v["pass_at_1"] for _, v in chunk if v["pass_at_1"] is not None]
        oracle8 = [v["oracle_at_n"].get("8") for _, v in chunk if v["oracle_at_n"].get("8") is not None]
        g8 = [v["g_n"].get("8") for _, v in chunk if v["g_n"].get("8") is not None]
        delta_8_16 = [v["delta_g"].get("8->16") for _, v in chunk if v["delta_g"].get("8->16") is not None]
        dom_share = [v["dominant_cluster_share"] for _, v in chunk if v["dominant_cluster_share"] is not None]
        n_clusters = [v["n_unique_clusters"] for _, v in chunk]
        new_correct = [v["state_transitions"].get("new_correct", 0) for _, v in chunk]
        repeated_err = [v["state_transitions"].get("repeated_error", 0) for _, v in chunk]
        quartiles[label] = {
            "n_cells": len(chunk), "h_range": [chunk[0][1]["h_sem_bits"], chunk[-1][1]["h_sem_bits"]],
            "mean_pass_at_1": round(statistics.mean(pass1), 4) if pass1 else None,
            "mean_oracle_at_8": round(statistics.mean(oracle8), 4) if oracle8 else None,
            "mean_g_8": round(statistics.mean(g8), 4) if g8 else None,
            "mean_delta_8_to_16": round(statistics.mean(delta_8_16), 4) if delta_8_16 else None,
            "mean_dominant_cluster_share": round(statistics.mean(dom_share), 4) if dom_share else None,
            "mean_n_unique_clusters": round(statistics.mean(n_clusters), 2),
            "mean_new_correct_rate": round(statistics.mean(new_correct), 4),
            "mean_repeated_error_rate": round(statistics.mean(repeated_err), 4),
            "cells": [k for k, _ in chunk],
        }
    return quartiles


# ---------------------------------------------------------------------------
# stage 7: A/B/C/D quadrants (H x accuracy)
# ---------------------------------------------------------------------------

def quadrant_analysis(cells):
    items = [(k, v) for k, v in cells.items() if v["h_sem_bits"] is not None and v["pass_at_1"] is not None]
    h_vals = sorted(v["h_sem_bits"] for _, v in items)
    a_vals = sorted(v["pass_at_1"] for _, v in items)
    h_med = h_vals[len(h_vals) // 2]
    a_med = a_vals[len(a_vals) // 2]

    quads = {"A_lowH_highAcc": [], "B_lowH_lowAcc": [], "C_highH_highAcc": [], "D_highH_lowAcc": []}
    for k, v in items:
        low_h = v["h_sem_bits"] <= h_med
        high_acc = v["pass_at_1"] >= a_med
        if low_h and high_acc:
            quads["A_lowH_highAcc"].append(k)
        elif low_h and not high_acc:
            quads["B_lowH_lowAcc"].append(k)
        elif not low_h and high_acc:
            quads["C_highH_highAcc"].append(k)
        else:
            quads["D_highH_lowAcc"].append(k)

    out = {"h_median": round(h_med, 4), "accuracy_median": round(a_med, 4), "quadrants": {}}
    for name, keys in quads.items():
        sub = [cells[k] for k in keys]
        out["quadrants"][name] = {
            "n_cells": len(sub), "cells": keys,
            "mean_pass_at_1": round(statistics.mean([c["pass_at_1"] for c in sub]), 4) if sub else None,
            "mean_h_sem": round(statistics.mean([c["h_sem_bits"] for c in sub]), 4) if sub else None,
            "mean_dominant_cluster_share": round(statistics.mean([c["dominant_cluster_share"] for c in sub if c["dominant_cluster_share"] is not None]), 4) if sub else None,
            "mean_delta_8_to_16": round(statistics.mean([c["delta_g"].get("8->16", 0) for c in sub]), 4) if sub else None,
        }
    return out


# ---------------------------------------------------------------------------
# stage 8: temperature comparison
# ---------------------------------------------------------------------------

def temperature_comparison(cells):
    by_t = {}
    temps = sorted(set(v["temperature"] for v in cells.values()))
    for t in temps:
        sub = [v for v in cells.values() if v["temperature"] == t]
        by_t[str(t)] = {
            "n_cells": len(sub),
            "mean_h_sem": round(statistics.mean([c["h_sem_bits"] for c in sub if c["h_sem_bits"] is not None]), 4),
            "mean_h_error_among_incorrect": round(
                statistics.mean([c["h_error_among_incorrect_bits"] for c in sub if c["h_error_among_incorrect_bits"] is not None]), 4
            ) if any(c["h_error_among_incorrect_bits"] is not None for c in sub) else None,
            "mean_pass_at_1": round(statistics.mean([c["pass_at_1"] for c in sub]), 4),
            "mean_oracle_at_8": round(statistics.mean([c["oracle_at_n"].get("8", c["pass_at_1"]) for c in sub]), 4),
            "mean_g_8": round(statistics.mean([c["g_n"].get("8", 0) for c in sub]), 4),
            "mean_repeated_error_rate": round(statistics.mean([c["state_transitions"].get("repeated_error", 0) for c in sub]), 4),
            "mean_new_correct_rate": round(statistics.mean([c["state_transitions"].get("new_correct", 0) for c in sub]), 4),
            "total_output_tokens": sum(c["total_output_tokens"] for c in sub),
            "total_time_sec": round(sum(c["total_time_sec"] for c in sub), 1),
        }
    for t_key, d in by_t.items():
        d["g_8_per_1k_tokens"] = round(1000 * d["mean_g_8"] / (d["total_output_tokens"] / d["n_cells"]), 5) if d["total_output_tokens"] else None
    return by_t


# ---------------------------------------------------------------------------
# stage 10: correlations + bootstrap CI
# ---------------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def spearman(xs, ys):
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks
    return pearson(rank(xs), rank(ys))


def bootstrap_ci(xs, ys, fn, rng, n_boot=N_BOOTSTRAP):
    n = len(xs)
    if n < 4:
        return None
    idx = list(range(n))
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in range(n)]
        bx = [xs[i] for i in sample]
        by = [ys[i] for i in sample]
        r = fn(bx, by)
        if r is not None:
            vals.append(r)
    if not vals:
        return None
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return round(lo, 3), round(hi, 3)


def correlation_suite(cells, rng):
    items = list(cells.values())

    def pair(key_x, key_y, transform_x=None, transform_y=None):
        xs, ys = [], []
        for v in items:
            x = transform_x(v) if transform_x else v.get(key_x)
            y = transform_y(v) if transform_y else v.get(key_y)
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if len(xs) < 4:
            return {"n": len(xs), "note": "INSUFFICIENT_SAMPLE"}
        p = pearson(xs, ys)
        s = spearman(xs, ys)
        p_ci = bootstrap_ci(xs, ys, pearson, rng)
        s_ci = bootstrap_ci(xs, ys, spearman, rng)
        return {"n": len(xs), "pearson_r": round(p, 3) if p is not None else None, "pearson_95ci": p_ci,
                "spearman_rho": round(s, 3) if s is not None else None, "spearman_95ci": s_ci}

    g8 = lambda v: v["g_n"].get("8")
    dg = lambda v: v["delta_g"].get("8->16")

    return {
        "H_sem_vs_G8": pair(None, None, lambda v: v["h_sem_bits"], g8),
        "H_sem_vs_deltaG": pair(None, None, lambda v: v["h_sem_bits"], dg),
        "H_error_vs_G8": pair(None, None, lambda v: v["h_error_among_incorrect_bits"], g8),
        "H_error_vs_deltaG": pair(None, None, lambda v: v["h_error_among_incorrect_bits"], dg),
        "dominant_cluster_share_vs_G8": pair(None, None, lambda v: v["dominant_cluster_share"], g8),
        "dominant_cluster_share_vs_repeated_error": pair(
            None, None, lambda v: v["dominant_cluster_share"], lambda v: v["state_transitions"].get("repeated_error")
        ),
        "n_sample_note": f"n_cells={len(items)} (18 task x temperature cells) -- small-sample, all CIs must be read with that in mind",
    }


# ---------------------------------------------------------------------------
# stage 12: systematic trap vs useful-compute candidates
# ---------------------------------------------------------------------------

def trap_detection(cells):
    traps = []
    useful = []
    for k, v in cells.items():
        if v["pass_at_1"] is None or v["pass_at_1"] >= 0.5:
            continue  # only interesting where Pass@1 is low
        oracle16 = v["oracle_at_n"].get("16", v["oracle_at_n"].get("8"))
        grew = (oracle16 - v["pass_at_1"]) if oracle16 is not None else 0
        record = {"cell": k, "pass_at_1": v["pass_at_1"], "oracle_top": oracle16,
                  "growth": round(grew, 4), "h_sem": v["h_sem_bits"], "dominant_share": v["dominant_cluster_share"]}
        if grew < 0.15 and v["dominant_cluster_share"] and v["dominant_cluster_share"] >= 0.7:
            traps.append(record)
        elif grew >= 0.3:
            useful.append(record)
    return {"systematic_trap_candidates": traps, "useful_compute_candidates": useful}


# ---------------------------------------------------------------------------
def main():
    cells = load_cells()
    rng = random.Random(RNG_SEED)

    out = {
        "quartiles_by_H_sem": quartile_analysis(cells),
        "quadrants_ABCD": quadrant_analysis(cells),
        "temperature_comparison": temperature_comparison(cells),
        "correlations": correlation_suite(cells, rng),
        "trap_detection": trap_detection(cells),
    }
    with open(os.path.join(METRICS_DIR, "analysis5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote analysis5_summary.json")
    print(json.dumps(out["temperature_comparison"], indent=2))


if __name__ == "__main__":
    main()
