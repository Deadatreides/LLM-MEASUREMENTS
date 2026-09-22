"""
Sections 2, 8, 9, 10, 19: for all 20 tasks, inject exactly one error at
each of the 6 possible claim positions (36 combinations x... 20 tasks x 6
positions = 120 injected cases), classify the resulting (mostly-correct,
one-corrupted) decomposition mechanically, compute PREDICTED_AFFECTED_SET
(dependency_engine9, from classified text) and compare against
TRUE_AFFECTED_SET (ground_truth9, from the known injection point + frozen
graph -- computed independently, never calling the same code).

Injection-point -> CASE mapping (section 2):
  C6_FINAL_TOTAL  -> CASE A (local/terminal, no downstream) and CASE D
                     (this DAG's only merge node IS the terminal claim, so
                     A and D coincide structurally here -- noted honestly,
                     not engineered around)
  C1_ROOT         -> CASE B (shared upstream, must propagate through BOTH
                     branches)
  C2/C3 (method)  -> CASE C (branch-only, other branch must stay clean)
  C4/C5 (subtotal)-> CASE C (branch-only, downstream-of-branch variant)

Section 19's "same DAG, different injected node" check is exactly this
script's per-task loop over all 6 positions -- the affected-set SIZE and
CONTENT must differ correctly according to graph position, not collapse
to "the whole task is bad" regardless of where the injection happened.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from schema9 import TASKS, TASK_IDS, CLAIMS, DEPENDS_ON
from classifiers9 import classify_claim
from injection9 import inject_error
from dependency_engine9 import predicted_affected_set
from ground_truth9 import true_affected_set

SECTION_FOR_CLAIM = {"C1_ROOT": "A1_ROOT", "C2_METHOD_A": "A2_METHOD_A", "C3_METHOD_B": "A3_METHOD_B",
                      "C4_SUBTOTAL_A": "A4_SUBTOTAL_A", "C5_SUBTOTAL_B": "A5_SUBTOTAL_B", "C6_FINAL_TOTAL": "A6_FINAL_TOTAL"}

CASE_OF_CLAIM = {"C6_FINAL_TOTAL": "CASE_A_LOCAL_AND_D_MERGE", "C1_ROOT": "CASE_B_UPSTREAM",
                  "C2_METHOD_A": "CASE_C_BRANCH", "C3_METHOD_B": "CASE_C_BRANCH",
                  "C4_SUBTOTAL_A": "CASE_C_BRANCH", "C5_SUBTOTAL_B": "CASE_C_BRANCH"}


def evaluate_one(task_id, injected_claim):
    task = TASKS[task_id]
    sections, desc = inject_error(task, injected_claim)

    local_status = {}
    for cid in CLAIMS:
        text = sections[SECTION_FOR_CLAIM[cid]]
        local_status[cid], _ = classify_claim(cid, text, task)

    predicted, root_origins, effective = predicted_affected_set(local_status, DEPENDS_ON, CLAIMS)
    true_set = true_affected_set([injected_claim])

    tp = predicted & true_set
    false_expansion = predicted - true_set  # claims flagged that are actually independent
    missed_dependency = true_set - predicted  # claims truly affected but not flagged

    precision = len(tp) / len(predicted) if predicted else (1.0 if not true_set else 0.0)
    recall = len(tp) / len(true_set) if true_set else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "task_id": task_id, "injected_claim": injected_claim, "case": CASE_OF_CLAIM[injected_claim],
        "injection_description": desc, "local_status": local_status,
        "predicted_affected_set": sorted(predicted), "true_affected_set": sorted(true_set),
        "root_origins_predicted": sorted(root_origins),
        "false_expansion": sorted(false_expansion), "missed_dependency": sorted(missed_dependency),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "exact_match": predicted == true_set,
    }


def main():
    results = []
    for task_id in TASK_IDS:
        for claim_id in CLAIMS:
            results.append(evaluate_one(task_id, claim_id))

    out_path = os.path.join(os.path.dirname(__file__), "case_abcd_results9.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # aggregate by case
    by_case = collections.defaultdict(list)
    for r in results:
        by_case[r["case"]].append(r)

    print(f"n_results={len(results)}")
    for case, rows in sorted(by_case.items()):
        n = len(rows)
        exact = sum(1 for r in rows if r["exact_match"])
        mean_p = sum(r["precision"] for r in rows) / n
        mean_r = sum(r["recall"] for r in rows) / n
        mean_f1 = sum(r["f1"] for r in rows) / n
        total_fe = sum(len(r["false_expansion"]) for r in rows)
        total_md = sum(len(r["missed_dependency"]) for r in rows)
        print(f"{case:28} n={n:3} exact_match={exact}/{n} mean_P={mean_p:.4f} mean_R={mean_r:.4f} mean_F1={mean_f1:.4f} total_FALSE_EXPANSION={total_fe} total_MISSED_DEPENDENCY={total_md}")

    # section 19: same DAG, different node -- check affected-set sizes are structurally sane per task
    print("\n--- section 19 sanity: per-task affected-set size by injection point ---")
    by_task = collections.defaultdict(dict)
    for r in results:
        by_task[r["task_id"]][r["injected_claim"]] = len(r["true_affected_set"])
    violations = 0
    for task_id, sizes in by_task.items():
        # C1 (upstream) must have the largest true affected set (whole graph); C6 (terminal) the smallest (1)
        if sizes["C1_ROOT"] != 6 or sizes["C6_FINAL_TOTAL"] != 1:
            violations += 1
    print(f"tasks with correct C1=6/C6=1 structural sizing: {len(by_task) - violations}/{len(by_task)}")


if __name__ == "__main__":
    main()
