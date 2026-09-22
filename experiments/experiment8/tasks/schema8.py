"""
FROZEN PROTOCOL (spec section 23) -- artifact/claim/dependency schema for
experiment 8. Written and reviewed BEFORE any experiment-8 generation
happens. Not edited after seeing generation results (if a genuine
infrastructure bug is later found, spec section 15/23 says: fix it and
REDO the affected part of the experiment, not silently patch the rule).

Two task families:

MATH_01..04 (new, section 3: need multi-step tasks with a genuine
dependency chain and a well-known "shortcut" wrong method, not just
random arithmetic slips):
  A1_METHOD       -- states which formula/method applies
  A2_VALUES       -- restates the relevant numeric inputs
  A3_COMPUTATION  -- performs the arithmetic using A1's method + A2's values
  A4_FINAL_ANSWER -- states the final numeric answer
  DEPENDS_ON: A3 depends on A1 AND A2; A4 depends on A3.
  Each task is deliberately built around a real, well-documented wrong
  shortcut (averaging two speeds instead of total distance/total time;
  additive instead of sequential percentage discounts; etc.) so that real
  model errors -- not injected ones -- exercise the dependency-propagation
  machinery in the MAIN experiment. Numbers are chosen so the "shortcut"
  method provably gives a DIFFERENT final number than the correct method
  (checked by hand below), so C1 (method) is verifiable independently of
  whether C4 (final number) happens to coincide.

CODE_01 (is_palindrome) / CODE_06 (two_sum) (reused verbatim from
experiments 1-7: same question, reference_implementation, reference_tests):
  A1_CONTRACT     -- signature + core behavior (claims C1-C5, reused from
                     claims_def_code.py, unmodified)
  A2_EDGE_CASES   -- edge case handling (claims C6-C7, reused unmodified)
  A3_IMPLEMENTATION -- actual Python code, mechanically tested against
                     REFERENCE_TESTS via subprocess (claim C8_TESTS_PASS,
                     LEVEL 0 mechanical, independent of what A1/A2 claimed
                     -- SEAM_CONSISTENCY separately checks whether A3's
                     actual signature agrees with what A1 claimed, see
                     metrics/seams8.py)
  DEPENDS_ON: none of C1-C8 are made to depend on each other for CODE
  tasks -- C8 is checked by running real tests against the real task, not
  against the model's own (possibly wrong) A1/A2 text, so it is not
  "locally consistent with a wrong premise" in the section-6 sense. The
  A1-vs-A3 relationship is instead captured as a SEAM (agreement, not
  correctness-dependence).
"""

MATH_TASKS = {
    "MATH_01": {
        "task_id": "MATH_01",
        "question": (
            "A bank account starts with a principal of $500. It earns SIMPLE interest "
            "(not compound) at a rate of 4% per year for 3 years. What is the total "
            "interest earned, not including the principal?"
        ),
        "correct_method_desc": "simple interest: I = P * r * t",
        "wrong_method_desc": "compound interest formula (P*(1+r)^t - P)",
        "values": {"P": 500, "r_pct": 4, "r": 0.04, "t": 3},
        "intermediate": 60,  # I = 500*0.04*3
        "final_answer": 60,
        "wrong_shortcut_final_answer": 62.432,  # 500*(1.04**3 - 1), for SEAM_CONSISTENCY checks
    },
    "MATH_02": {
        "task_id": "MATH_02",
        "question": (
            "A student scored 80 on a test worth 30% of the final grade, and 90 on a "
            "test worth 70% of the final grade. What is the student's final weighted "
            "average grade?"
        ),
        "correct_method_desc": "weighted average: sum(score_i * weight_i)",
        "wrong_method_desc": "simple unweighted average ((80+90)/2)",
        "values": {"s1": 80, "w1_pct": 30, "w1": 0.3, "s2": 90, "w2_pct": 70, "w2": 0.7},
        "intermediate": 87,  # 80*0.3+90*0.7 = 24+63 = 87
        "final_answer": 87,
        "wrong_shortcut_final_answer": 85,  # (80+90)/2
    },
    "MATH_03": {
        "task_id": "MATH_03",
        "question": (
            "A car travels 60 miles at 30 mph, then another 60 miles at 60 mph. What is "
            "the car's average speed for the ENTIRE trip (defined as total distance "
            "divided by total time, not the average of the two speeds)?"
        ),
        "correct_method_desc": "total distance / total time",
        "wrong_method_desc": "simple average of the two speeds ((30+60)/2)",
        "values": {"d1": 60, "v1": 30, "d2": 60, "v2": 60},
        "intermediate": 3,  # total time = 2h + 1h = 3h (total distance 120mi)
        "final_answer": 40,  # 120/3 ; wrong-shortcut gives 45, provably different
        "wrong_shortcut_final_answer": 45,  # (30+60)/2
    },
    "MATH_04": {
        "task_id": "MATH_04",
        "question": (
            "An item costs $200. It is first discounted by 20%, then an ADDITIONAL 10% "
            "off the already-discounted price (sequential discounts, not combined into "
            "one 30% discount). What is the final price?"
        ),
        "correct_method_desc": "sequential discounts: 200*0.8 then *0.9",
        "wrong_method_desc": "additive discount (200*0.7 = 30% off combined)",
        "values": {"price": 200, "d1_pct": 20, "d1": 0.2, "d2_pct": 10, "d2": 0.1},
        "intermediate": 160,  # 200*0.8
        "final_answer": 144,  # 160*0.9 ; wrong-shortcut gives 140, provably different
        "wrong_shortcut_final_answer": 140,  # 200*0.7
    },
}

MATH_ARTIFACTS = ["A1_METHOD", "A2_VALUES", "A3_COMPUTATION", "A4_FINAL_ANSWER"]
MATH_CLAIMS = ["C1_METHOD", "C2_VALUES", "C3_COMPUTATION", "C4_FINAL_ANSWER"]
MATH_CLAIM_ARTIFACT = {"C1_METHOD": "A1_METHOD", "C2_VALUES": "A2_VALUES", "C3_COMPUTATION": "A3_COMPUTATION", "C4_FINAL_ANSWER": "A4_FINAL_ANSWER"}
MATH_DEPENDS_ON = {"C1_METHOD": [], "C2_VALUES": [], "C3_COMPUTATION": ["C1_METHOD", "C2_VALUES"], "C4_FINAL_ANSWER": ["C3_COMPUTATION"]}

CODE_ARTIFACTS = ["A1_CONTRACT", "A2_EDGE_CASES", "A3_IMPLEMENTATION"]
# claim_id -> artifact_id for the reused claims_def_code.py claims (C1-C5 -> A1, C6-C7 -> A2)
CODE_CLAIM_ARTIFACT_BY_TASK = {
    "CODE_01": {"C1_NAME": "A1_CONTRACT", "C2_ARGC": "A1_CONTRACT", "C3_ARG_TYPE": "A1_CONTRACT", "C4_RETURN_TYPE": "A1_CONTRACT",
                "C5_SEMANTICS_CASE": "A1_CONTRACT", "C6_EDGE_EMPTY": "A2_EDGE_CASES", "C7_EDGE_SPACES": "A2_EDGE_CASES", "C8_TESTS_PASS": "A3_IMPLEMENTATION"},
    "CODE_06": {"C1_NAME": "A1_CONTRACT", "C2_ARGC": "A1_CONTRACT", "C3_RETURN_TYPE": "A1_CONTRACT", "C4_SEMANTICS_INDICES": "A1_CONTRACT",
                "C5_SEMANTICS_SMALLEST_I": "A1_CONTRACT", "C6_EDGE_ONE_SOLUTION": "A2_EDGE_CASES", "C7_EDGE_NOT_SAME": "A2_EDGE_CASES", "C8_TESTS_PASS": "A3_IMPLEMENTATION"},
}
CODE_DEPENDS_ON = {}  # C8 independently mechanically checked; A1-vs-A3 handled as a SEAM, not a dependency (see module docstring)

ALL_TASK_IDS = list(MATH_TASKS.keys()) + ["CODE_01", "CODE_06"]
TASK_FAMILY = {**{t: "MATH" for t in MATH_TASKS}, "CODE_01": "CODE", "CODE_06": "CODE"}
