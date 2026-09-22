"""
Section 5-7: builds the reference-correct 6-section decomposition for a
task (hand-templated, not model-generated -- exactly the "correct DAG
fixed BEFORE injecting an error" section 5/7 requires), then injects
exactly ONE minimal, targeted error into exactly one claim's artifact
text, leaving every other section byte-for-byte unchanged.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))
from schema9 import CLAIM_ARTIFACT


def reference_correct_sections(task):
    return {
        "A1_ROOT": f"This problem has two parts that must be handled separately: {task['keywordA']} and {task['keywordB']}, then combined into one total.",
        "A2_METHOD_A": f"For {task['keywordA']}: multiply the quantity ({task['qtyA']}) by the rate ({task['rateA']}).",
        "A3_METHOD_B": f"For {task['keywordB']}: multiply the quantity ({task['qtyB']}) by the rate ({task['rateB']}).",
        "A4_SUBTOTAL_A": f"{task['qtyA']} * {task['rateA']} = {task['subtotalA']}",
        "A5_SUBTOTAL_B": f"{task['qtyB']} * {task['rateB']} = {task['subtotalB']}",
        "A6_FINAL_TOTAL": f"{task['subtotalA']} + {task['subtotalB']} = {task['final_total']}. Final total: {task['final_total']}",
    }


def _wrong_number(correct, bump=1.37):
    """Deterministic, reproducible wrong value: always provably different
    from `correct` (never coincides after rounding)."""
    wrong = round(correct * bump + 3, 4)
    if abs(wrong - correct) < 0.01:
        wrong += 5
    return wrong


def inject_error(task, claim_id):
    """Returns (corrupted_sections, description) with exactly ONE artifact
    text replaced; all others are identical to reference_correct_sections."""
    sections = dict(reference_correct_sections(task))
    artifact_id = CLAIM_ARTIFACT[claim_id]

    if claim_id == "C1_ROOT":
        sections[artifact_id] = f"This problem is about {task['keywordA']} only; compute the total cost for that."
        desc = f"root now omits {task['keywordB']!r} entirely"
    elif claim_id == "C2_METHOD_A":
        wrong_rate = _wrong_number(task["rateA"])
        sections[artifact_id] = f"For {task['keywordA']}: multiply the quantity ({task['qtyA']}) by the rate ({wrong_rate})."
        desc = f"method A states wrong rate {wrong_rate} instead of {task['rateA']}"
    elif claim_id == "C3_METHOD_B":
        wrong_rate = _wrong_number(task["rateB"])
        sections[artifact_id] = f"For {task['keywordB']}: multiply the quantity ({task['qtyB']}) by the rate ({wrong_rate})."
        desc = f"method B states wrong rate {wrong_rate} instead of {task['rateB']}"
    elif claim_id == "C4_SUBTOTAL_A":
        wrong_sub = _wrong_number(task["subtotalA"])
        sections[artifact_id] = f"{task['qtyA']} * {task['rateA']} = {wrong_sub}"
        desc = f"subtotal A computed as {wrong_sub} instead of {task['subtotalA']}"
    elif claim_id == "C5_SUBTOTAL_B":
        wrong_sub = _wrong_number(task["subtotalB"])
        sections[artifact_id] = f"{task['qtyB']} * {task['rateB']} = {wrong_sub}"
        desc = f"subtotal B computed as {wrong_sub} instead of {task['subtotalB']}"
    elif claim_id == "C6_FINAL_TOTAL":
        wrong_total = _wrong_number(task["final_total"])
        sections[artifact_id] = f"{task['subtotalA']} + {task['subtotalB']} = {wrong_total}. Final total: {wrong_total}"
        desc = f"final total stated as {wrong_total} instead of {task['final_total']}"
    else:
        raise ValueError(claim_id)

    return sections, desc
