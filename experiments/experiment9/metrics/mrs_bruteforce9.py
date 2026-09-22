"""
Section 17: MRS_GROUND_TRUTH via actual perturb-and-reverify, not
heuristic. For each of the 120 injected cases (case_abcd_evaluation9.py),
brute-force all 2^6=64 subsets of claims: "repair" a subset by setting
those claims' artifact text back to the reference-correct version while
leaving every other claim exactly as it is in the corrupted decomposition
(including the ONE originally-injected claim, if it is excluded from the
candidate subset). Classify + propagate the resulting hybrid text and
check whether C6_FINAL_TOTAL's EFFECTIVE status becomes CORRECT. The
minimal such subset (by size, ties kept) is MRS_GROUND_TRUTH.

Key finding anticipated from case_abcd_evaluation9.py's own local_status
output: because injection9.inject_error() only corrupts ONE artifact's
text and leaves all downstream artifacts' text at their reference-correct
content, downstream claims' OWN text is often still factually fine even
though the graph-based "affected set" (ground_truth9) conservatively
includes them. This can make MRS_GROUND_TRUTH SMALLER than the
graph-affected set -- the brute force is what actually confirms or
refutes that, rather than assuming it.
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from schema9 import TASKS, TASK_IDS, CLAIMS, DEPENDS_ON
from classifiers9 import classify_claim
from injection9 import inject_error, reference_correct_sections
from dependency_engine9 import propagate, predicted_affected_set
from ground_truth9 import true_affected_set

SECTION_FOR_CLAIM = {"C1_ROOT": "A1_ROOT", "C2_METHOD_A": "A2_METHOD_A", "C3_METHOD_B": "A3_METHOD_B",
                      "C4_SUBTOTAL_A": "A4_SUBTOTAL_A", "C5_SUBTOTAL_B": "A5_SUBTOTAL_B", "C6_FINAL_TOTAL": "A6_FINAL_TOTAL"}


def check_repair(task, corrupted_sections, repair_subset):
    ref = reference_correct_sections(task)
    hybrid = dict(corrupted_sections)
    for cid in repair_subset:
        hybrid[SECTION_FOR_CLAIM[cid]] = ref[SECTION_FOR_CLAIM[cid]]

    local_status = {}
    for cid in CLAIMS:
        local_status[cid], _ = classify_claim(cid, hybrid[SECTION_FOR_CLAIM[cid]], task)
    effective = propagate(local_status, DEPENDS_ON)
    return effective["C6_FINAL_TOTAL"][0] == "CORRECT"


def brute_force_mrs(task_id, injected_claim):
    task = TASKS[task_id]
    corrupted_sections, desc = inject_error(task, injected_claim)

    minimal_sets = []
    min_size = None
    for size in range(0, len(CLAIMS) + 1):
        for subset in itertools.combinations(CLAIMS, size):
            if check_repair(task, corrupted_sections, subset):
                minimal_sets.append(subset)
                min_size = size
        if minimal_sets:
            break  # found the smallest size that works; collect all ties at this size, then stop

    graph_affected = true_affected_set([injected_claim])
    return {
        "task_id": task_id, "injected_claim": injected_claim, "injection_description": desc,
        "mrs_ground_truth_candidates": [sorted(s) for s in minimal_sets], "mrs_ground_truth_size": min_size,
        "graph_affected_set": sorted(graph_affected), "graph_affected_size": len(graph_affected),
        "mrs_smaller_than_graph_affected": min_size is not None and min_size < len(graph_affected),
    }


def main():
    results = []
    for task_id in TASK_IDS:
        for claim_id in CLAIMS:
            results.append(brute_force_mrs(task_id, claim_id))

    out_path = os.path.join(os.path.dirname(__file__), "mrs_bruteforce9_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n = len(results)
    n_smaller = sum(1 for r in results if r["mrs_smaller_than_graph_affected"])
    n_equal = sum(1 for r in results if r["mrs_ground_truth_size"] == r["graph_affected_size"])
    print(f"n_cases={n}")
    print(f"MRS_ground_truth SMALLER than graph-affected set: {n_smaller}/{n}")
    print(f"MRS_ground_truth EQUAL to graph-affected set: {n_equal}/{n}")
    for r in results[:8]:
        print(r["task_id"], r["injected_claim"], "mrs=", r["mrs_ground_truth_candidates"], "graph_affected=", r["graph_affected_set"])


if __name__ == "__main__":
    main()
