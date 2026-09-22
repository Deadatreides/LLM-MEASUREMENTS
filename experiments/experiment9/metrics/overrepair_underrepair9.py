"""
Sections 9-11, TABLE 4-5: over-repair / under-repair / independent-branch
preservation, computed against the brute-force MRS_GROUND_TRUTH
(mrs_bruteforce9.py), not the conservative graph-affected set --
matching section 9's literal framing ("система решает: один claim сломан
-> пересчитать всю задачу" is a statement about what's ACTUALLY needed,
not about what the graph conservatively flags).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))
from schema9 import CLAIMS, BRANCH_A_CLAIMS, BRANCH_B_CLAIMS

METRICS_DIR = os.path.dirname(__file__)


def main():
    case_results = json.load(open(os.path.join(METRICS_DIR, "case_abcd_results9.json"), encoding="utf-8"))
    mrs_results = json.load(open(os.path.join(METRICS_DIR, "mrs_bruteforce9_results.json"), encoding="utf-8"))
    mrs_by_key = {(m["task_id"], m["injected_claim"]): m for m in mrs_results}

    n = len(case_results)
    over_repair = 0
    under_repair = 0
    full_recomputation = 0
    only_necessary_branch = 0
    extra_branch_included = 0
    branch_cases = 0

    rows = []
    for r in case_results:
        key = (r["task_id"], r["injected_claim"])
        m = mrs_by_key[key]
        true_mrs = set(m["mrs_ground_truth_candidates"][0])
        predicted = set(r["predicted_affected_set"])

        is_over = predicted > true_mrs  # strict superset
        is_under = not (true_mrs <= predicted)  # misses something truly needed
        is_full = predicted == set(CLAIMS)

        if is_over:
            over_repair += 1
        if is_under:
            under_repair += 1
        if is_full:
            full_recomputation += 1

        if r["case"] == "CASE_C_BRANCH":
            branch_cases += 1
            other_branch = set(BRANCH_B_CLAIMS) if r["injected_claim"] in BRANCH_A_CLAIMS else set(BRANCH_A_CLAIMS)
            if predicted & other_branch:
                extra_branch_included += 1
            else:
                only_necessary_branch += 1

        rows.append({"task_id": r["task_id"], "injected_claim": r["injected_claim"], "case": r["case"],
                      "true_mrs": sorted(true_mrs), "predicted": sorted(predicted),
                      "over_repair": is_over, "under_repair": is_under, "full_recomputation": is_full})

    summary = {
        "n_cases": n,
        "over_repair_rate": round(over_repair / n, 4),
        "under_repair_rate": round(under_repair / n, 4),
        "full_recomputation_rate": round(full_recomputation / n, 4),
        "independent_branch_preservation": round(only_necessary_branch / branch_cases, 4) if branch_cases else None,
        "branch_cases_n": branch_cases, "branch_cases_contaminated": extra_branch_included,
    }
    with open(os.path.join(METRICS_DIR, "overrepair_underrepair9_results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
