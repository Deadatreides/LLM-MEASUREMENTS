"""
Spec section 17: a synthetic control group, independent of any model
generation, to check the conflict-detection machinery itself (not the
models) against ground-truth-known cases. Hand-constructed claim vectors
for CODE_01 (is_palindrome), using the same claim_id taxonomy as the real
experiment, with deliberately controlled error correlation:

  G   -- fully correct vector (reference)
  G2  -- also fully correct, but imagine it was phrased differently
         (same classifications -- classification doesn't depend on
         wording once mechanically extracted, so G2 == G here; kept as a
         separate named vector for clarity of which case it feeds)
  E1  -- one independent error (case-sensitivity claim wrong)
  E2  -- a DIFFERENT independent error (spaces/punctuation claim wrong)
  E3  -- the SAME error as E1 (fully correlated with E1)
  E4  -- E1's error AND a second, independent error (partially correlated)

CASE A (one correct + one wrong):     G  vs E1
CASE B (two different wrong):         E1 vs E2
CASE C (same error twice):            E1 vs E3
CASE D (both correct, "different phrasing"): G  vs G2
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))

from claims import pairwise_claim_relation
from compute_metrics3 import pair_stats, METRICS_DIR

ALL_CLAIMS = ["C1_NAME", "C2_ARGC", "C3_ARG_TYPE", "C4_RETURN_TYPE", "C5_SEMANTICS_CASE", "C6_EDGE_EMPTY", "C7_EDGE_SPACES"]

G = {c: "CORRECT" for c in ALL_CLAIMS}
G2 = dict(G)  # same underlying facts, hypothetically different wording -> same mechanical classification
E1 = dict(G); E1["C5_SEMANTICS_CASE"] = "INCORRECT"
E2 = dict(G); E2["C7_EDGE_SPACES"] = "INCORRECT"
E3 = dict(G); E3["C5_SEMANTICS_CASE"] = "INCORRECT"  # identical error to E1
E4 = dict(G); E4["C5_SEMANTICS_CASE"] = "INCORRECT"; E4["C7_EDGE_SPACES"] = "INCORRECT"


def seam(vec_a, vec_b):
    out = {}
    for c in ALL_CLAIMS:
        out[c] = pairwise_claim_relation(vec_a[c], vec_b[c])
    return out


def run_case(name, vec_a, vec_b, expectation):
    s = seam(vec_a, vec_b)
    ps = pair_stats(vec_a, vec_b)
    n_conflict = sum(1 for v in s.values() if v == "CONFLICT")
    n_agree = sum(1 for v in s.values() if v == "AGREE")
    return {
        "case": name, "expectation": expectation, "per_claim_seam": s,
        "n_conflict": n_conflict, "n_agree": n_agree,
        "pair_stats": ps,
    }


def main():
    results = [
        run_case(
            "CASE_A_one_correct_one_wrong", G, E1,
            "conflict should localize to exactly the erroneous claim (C5); rest AGREE-correct",
        ),
        run_case(
            "CASE_B_two_different_wrong", E1, E2,
            "conflict on BOTH C5 and C7 (each disagrees on the other's error); no false sense of agreement",
        ),
        run_case(
            "CASE_C_same_error_twice", E1, E3,
            "full AGREE on all claims including the shared wrong one (C5) -- the H2 danger case: agreement, but wrong",
        ),
        run_case(
            "CASE_D_both_correct_different_wording", G, G2,
            "full AGREE, all correct -- the good case, contrasted directly against CASE C",
        ),
    ]

    # pass/fail check against the stated expectations, purely mechanical
    checks = {}
    a = results[0]
    checks["CASE_A_localizes_conflict_to_C5_only"] = (a["n_conflict"] == 1 and a["per_claim_seam"]["C5_SEMANTICS_CASE"] == "CONFLICT")
    b = results[1]
    checks["CASE_B_shows_two_conflicts"] = (b["n_conflict"] == 2)
    c = results[2]
    checks["CASE_C_shows_full_agreement_despite_shared_error"] = (c["n_conflict"] == 0 and c["n_agree"] == len(ALL_CLAIMS))
    d = results[3]
    checks["CASE_D_shows_full_agreement_and_correct"] = (d["n_conflict"] == 0 and d["n_agree"] == len(ALL_CLAIMS))

    out = {"results": results, "checks": checks, "all_checks_passed": all(checks.values())}
    with open(os.path.join(METRICS_DIR, "control_experiment.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    for k, v in checks.items():
        print(k, "->", "PASS" if v else "FAIL")
    print("ALL PASSED:", out["all_checks_passed"])


if __name__ == "__main__":
    main()
