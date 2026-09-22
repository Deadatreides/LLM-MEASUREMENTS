"""
The "virtual seam" (spec section 14): for every pair of independent A1s,
classify each shared claim as AGREE / CONFLICT / UNKNOWN (no LLM judge --
derived from each side's own CORRECT/INCORRECT/OMITTED/AMBIGUOUS
classification, see configs/claims.py:pairwise_claim_relation).

Also tests H1/H2 directly (spec section 15): among CONFLICT pairs, is it
true that "exactly one side is correct" (this is definitionally true given
how CONFLICT is derived here -- see REPORT3.md for the honest caveat on
H1). Among AGREE pairs, what fraction agree on being WRONG -- this is the
real, non-tautological test of H2 ("AGREEMENT != CORRECTNESS").
"""
import collections
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))

from claims import pairwise_claim_relation
from compute_metrics3 import load_runs, load_claims, build_vectors, RUNS_DIR, METRICS_DIR


def seam_tally(vec_a, vec_b):
    shared = set(vec_a) & set(vec_b)
    tally = collections.Counter()
    both_correct = both_incorrect = 0
    for k in shared:
        rel = pairwise_claim_relation(vec_a[k], vec_b[k])
        tally[rel] += 1
        if rel == "AGREE":
            if vec_a[k] == "CORRECT":
                both_correct += 1
            else:
                both_incorrect += 1
    return tally, both_correct, both_incorrect


def main():
    runs = load_runs()
    claim_rows = load_claims()
    vecs = build_vectors(claim_rows)

    cells = collections.defaultdict(list)
    for r in runs:
        cells[(r["task_id"], r["model_id"], r["prompt_id"], r["temperature"])].append(r["run_id"])

    categories = {"same_settings": collections.Counter(), "diff_temperature_same_prompt_model": collections.Counter(),
                  "diff_prompt_same_temp_model": collections.Counter(), "diff_model_same_prompt_temp": collections.Counter()}
    both_c = {k: 0 for k in categories}
    both_i = {k: 0 for k in categories}

    keys = list(cells.keys())
    for i, key_a in enumerate(keys):
        task_a, model_a, prompt_a, t_a = key_a
        for key_b in keys[i:]:
            task_b, model_b, prompt_b, t_b = key_b
            if task_a != task_b:
                continue  # claims are task-specific; only compare within the same task
            same_model = model_a == model_b
            same_prompt = prompt_a == prompt_b
            same_t = t_a == t_b

            if key_a == key_b:
                cat = "same_settings"
                pairs = list(itertools.combinations(sorted(cells[key_a]), 2))
            elif same_model and same_prompt and not same_t:
                cat = "diff_temperature_same_prompt_model"
                pairs = [(ra, rb) for ra in cells[key_a] for rb in cells[key_b]]
            elif same_model and same_t and not same_prompt:
                cat = "diff_prompt_same_temp_model"
                pairs = [(ra, rb) for ra in cells[key_a] for rb in cells[key_b]]
            elif same_prompt and same_t and not same_model:
                cat = "diff_model_same_prompt_temp"
                pairs = [(ra, rb) for ra in cells[key_a] for rb in cells[key_b]]
            else:
                continue

            for ra, rb in pairs:
                tally, bc, bi = seam_tally(vecs[ra], vecs[rb])
                categories[cat] += tally
                both_c[cat] += bc
                both_i[cat] += bi

    result = {}
    for cat, tally in categories.items():
        total = sum(tally.values())
        n_agree = tally["AGREE"]
        result[cat] = {
            "n_claim_pairs": total,
            "AGREE": tally["AGREE"], "CONFLICT": tally["CONFLICT"], "UNKNOWN": tally["UNKNOWN"],
            "agree_rate": round(n_agree / total, 3) if total else None,
            "conflict_rate": round(tally["CONFLICT"] / total, 3) if total else None,
            "unknown_rate": round(tally["UNKNOWN"] / total, 3) if total else None,
            "among_agree__both_correct": both_c[cat], "among_agree__both_incorrect": both_i[cat],
            "H2_agree_but_both_wrong_rate": round(both_i[cat] / n_agree, 3) if n_agree else None,
        }

    with open(os.path.join(METRICS_DIR, "seam3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
