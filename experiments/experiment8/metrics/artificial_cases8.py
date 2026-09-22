"""
Section 11: 7 hand-constructed control cases, built AFTER the correct
decomposition schema/classifiers were frozen (sections 2/23), used ONLY as
a sanity check of the seam/dependency machinery -- NOT as a benchmark of
real model quality (spec explicitly forbids drawing model-quality
conclusions from these). All text is hand-authored, not generated.

Base task: MATH_01 (simple interest, P=500, r=4%, t=3, correct final=60,
wrong-shortcut final=62.432 via compound interest).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from classifiers8 import classify_math_claim
from dependency_engine import propagate, minimal_repair_set, error_taxonomy_label
from seams8 import seam_dependency_status, seam_consistency_math
from schema8 import MATH_TASKS, MATH_CLAIMS, MATH_DEPENDS_ON

TASK = MATH_TASKS["MATH_01"]
CLAIM_ORDER = MATH_CLAIMS


def run_case(name, expected_summary, sections):
    local = {cid: classify_math_claim("MATH_01", cid, sections[{"C1_METHOD": "A1_METHOD", "C2_VALUES": "A2_VALUES",
                                                                  "C3_COMPUTATION": "A3_COMPUTATION", "C4_FINAL_ANSWER": "A4_FINAL_ANSWER"}[cid]])[0]
             for cid in CLAIM_ORDER}
    eff = propagate(local, MATH_DEPENDS_ON)
    mrs, root_origins = minimal_repair_set(eff, MATH_DEPENDS_ON, CLAIM_ORDER)
    labels = {cid: error_taxonomy_label(cid, eff, root_origins) for cid in CLAIM_ORDER}
    seam_c1_c3 = seam_consistency_math(TASK, sections, local["C1_METHOD"], local["C3_COMPUTATION"], None)
    seam_deps = [seam_dependency_status(cid, local[cid], eff, MATH_DEPENDS_ON) for cid in CLAIM_ORDER]
    seam_deps = [s for s in seam_deps if s]
    return {
        "case": name, "expected": expected_summary, "local": local, "effective": eff,
        "mrs": mrs, "root_origins": root_origins, "error_labels": labels,
        "seam_consistency_A1_A3": seam_c1_c3, "seam_dependency": seam_deps,
    }


CASES = []

# CASE 1: one artifact wrong (terminal-only slip), rest correct
CASES.append(run_case(
    "CASE_1_single_artifact_wrong",
    "MRS={C4}, root_origins={C4}, everything else NO_ERROR",
    {"A1_METHOD": "We use simple interest: I = P * r * t.",
     "A2_VALUES": "P=500, r=4%, t=3 years.",
     "A3_COMPUTATION": "500 * 0.04 * 3 = 60.",
     "A4_FINAL_ANSWER": "The final answer is 65."},  # slip, unrelated to the (correct) chain above
))

# CASE 2: upstream (A2) wrong, A3/A4 formally consistent with the wrong value
CASES.append(run_case(
    "CASE_2_upstream_wrong_downstream_consistent",
    "root_origins={C2}, MRS={C2,C3,C4}, C3/C4 labeled ERROR_DEPENDENCY not ERROR_LOCAL",
    {"A1_METHOD": "We use simple interest: I = P * r * t.",
     "A2_VALUES": "P=500, r=4%, t=5 years.",  # wrong: t should be 3
     "A3_COMPUTATION": "500 * 0.04 * 5 = 100.",  # consistent with the wrong t=5
     "A4_FINAL_ANSWER": "The final answer is 100."},
))

# CASE 3: two artifacts (A1 method vs A3 computation) contradict each other
CASES.append(run_case(
    "CASE_3_artifacts_contradict",
    "SEAM_CONSISTENCY(A1,A3) = VIOLATED (A1 claims simple interest but A3 computes compound interest)",
    {"A1_METHOD": "We use simple interest: I = P * r * t, not compound interest.",
     "A2_VALUES": "P=500, r=4%, t=3 years.",
     "A3_COMPUTATION": "500 * (1.04^3 - 1) = 62.432.",  # actually did compound, contradicting A1
     "A4_FINAL_ANSWER": "The final answer is 62.432."},
))

# CASE 4: one artifact has an internal contradiction
CASES.append(run_case(
    "CASE_4_internal_contradiction",
    "C1_METHOD = AMBIGUOUS (both simple and compound interest mentioned in the same artifact)",
    {"A1_METHOD": "We use simple interest (I=P*r*t). Actually, let's use the compound interest formula instead.",
     "A2_VALUES": "P=500, r=4%, t=3 years.",
     "A3_COMPUTATION": "500 * 0.04 * 3 = 60.",
     "A4_FINAL_ANSWER": "The final answer is 60."},
))

# CASE 5: A1 and A2 both individually correct, but A3 combines them wrongly
# (error lives ONLY in how A3 uses A1+A2, not in A1 or A2 themselves)
CASES.append(run_case(
    "CASE_5_error_only_in_combination",
    "root_origins={C3} (not C1/C2), MRS={C3,C4}",
    {"A1_METHOD": "We use simple interest: I = P * r * t.",
     "A2_VALUES": "P=500, r=4%, t=3 years.",
     "A3_COMPUTATION": "500 * 0.04 + 3 = 23.",  # wrong operator: + instead of *, matches neither 60 nor 62.432
     "A4_FINAL_ANSWER": "The final answer is 23."},
))

# CASE 6: correct answer, different wording (should NOT be flagged as an error)
CASES.append(run_case(
    "CASE_6_correct_but_different_wording",
    "everything CORRECT despite paraphrase",
    {"A1_METHOD": "This calls for the standard formula where interest equals principal times rate times time -- simple interest, since it is not compounded.",
     "A2_VALUES": "The principal is five hundred dollars (500), the annual rate is 4 percent, over a period of 3 years.",
     "A3_COMPUTATION": "Multiplying gives 500 x 0.04 x 3, which comes out to 60.",
     "A4_FINAL_ANSWER": "Final answer: 60."},
))

# CASE 7: two independent, both-correct formulations (checked as a pair)
CASE7_A = {"A1_METHOD": "Simple interest: I = P * r * t.", "A2_VALUES": "P=500, r=4%, t=3.",
           "A3_COMPUTATION": "500*0.04*3=60.", "A4_FINAL_ANSWER": "60."}
CASE7_B = {"A1_METHOD": "This is simple, not compound, interest: multiply principal by rate by time.",
           "A2_VALUES": "Principal 500, rate 0.04, time 3 years.",
           "A3_COMPUTATION": "500 times 0.04 times 3 equals 60.", "A4_FINAL_ANSWER": "The answer is 60."}
res_a = run_case("CASE_7a", "CORRECT", CASE7_A)
res_b = run_case("CASE_7b", "CORRECT", CASE7_B)
conflict = any(res_a["local"][c] == "CORRECT" and res_b["local"][c] == "INCORRECT" for c in CLAIM_ORDER) or \
           any(res_a["local"][c] == "INCORRECT" and res_b["local"][c] == "CORRECT" for c in CLAIM_ORDER)
CASES.append({"case": "CASE_7_two_independent_correct", "expected": "both CORRECT, no false SEAM_CONFLICT",
              "case_a": res_a, "case_b": res_b, "false_conflict_detected": conflict})


def main():
    import json
    out_path = os.path.join(os.path.dirname(__file__), "artificial_cases8_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(CASES, f, ensure_ascii=False, indent=2, default=str)

    for c in CASES:
        print("===", c["case"], "===")
        print("expected:", c["expected"])
        if "error_labels" in c:
            print("local:", c["local"])
            print("mrs:", c["mrs"], "root_origins:", c["root_origins"])
            print("labels:", c["error_labels"])
            print("seam_consistency A1/A3:", c["seam_consistency_A1_A3"])
        else:
            print("case_a local:", c["case_a"]["local"])
            print("case_b local:", c["case_b"]["local"])
            print("false_conflict_detected:", c["false_conflict_detected"])
        print()


if __name__ == "__main__":
    main()
