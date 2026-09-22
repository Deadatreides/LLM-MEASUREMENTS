"""
Section 10 baseline: mechanical Oracle-equivalent check for a FLAT (P
mode) answer -- one verdict for the whole answer, no localization
possible by construction (that is the point of comparing against it).
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))
from ast_checks import extract_code
from mech_check_code import classify_c8_tests_pass

_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


def _numbers_in(text):
    out = []
    for m in _NUM_RE.finditer(text):
        s = m.group(0).replace("$", "").replace(",", "").replace("%", "")
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def check_p_mode_math(text, expected_final_answer):
    anchored = list(re.finditer(r"\b(final answer|answer|result|total)\b\s*(is|:|=)?\s*\$?(-?\d[\d,]*\.?\d*)", text, re.I))
    if anchored:
        s = anchored[-1].group(3).replace(",", "")
        try:
            v = float(s)
            return ("CORRECT" if abs(v - expected_final_answer) <= 0.01 else "INCORRECT"), s
        except ValueError:
            pass
    nums = _numbers_in(text)
    if not nums:
        return "UNKNOWN", None
    v = nums[-1]
    return ("CORRECT" if abs(v - expected_final_answer) <= 0.01 else "INCORRECT"), str(v)


def check_p_mode_code(text, task):
    code, _ = extract_code(text, task.get("function_name"))
    if not code:
        return "UNKNOWN", "no code extracted"
    status, evidence = classify_c8_tests_pass(code, task["reference_tests"])
    return status, evidence
