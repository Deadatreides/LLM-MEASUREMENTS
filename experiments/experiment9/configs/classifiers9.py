"""
Section 5/9: mechanical (regex-only) claim classifiers, parameterized by
each task's schema9.py entry (all 20 tasks share the same 6-claim
structure, so one generic classifier per claim type suffices, unlike
experiment 8's per-task functions). Frozen alongside schema9.py, not
retuned after seeing generations.
"""
import re

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


def _has_number_near(text, target, tol=0.01):
    for m in _NUM_RE.finditer(text):
        s = m.group(0).replace("$", "").replace(",", "").replace("%", "")
        try:
            v = float(s)
        except ValueError:
            continue
        if abs(v - target) <= tol:
            return True
    return False


def _has_keyword(text, keyword):
    return re.search(re.escape(keyword), text, re.I) is not None


_GENERIC_TWO_PARTS_RE = re.compile(r"\btwo (parts|categories|components|pieces)\b|\bseparately\b|\bhandled? separately\b|\beach part\b", re.I)


def classify_c1_root(text, task):
    """CORRECT if the root either names both branches explicitly OR uses a
    generic "two separate parts" framing (found necessary after checking
    real generations: many correct decompositions state the STRUCTURE in
    A1 -- "this problem has two parts" -- and save the branch-specific
    names for A2/A3, which is a legitimate way to satisfy the claim, not
    an omission. Requiring literal keyword repetition in A1 specifically
    was over-strict and produced a large, spurious UNKNOWN rate purely
    from this one claim, cascading through propagation even when every
    other claim -- including the final numeric answer -- was correct)."""
    has_a = _has_keyword(text, task["keywordA"])
    has_b = _has_keyword(text, task["keywordB"])
    if has_a and has_b:
        return "CORRECT", f"mentions both {task['keywordA']!r} and {task['keywordB']!r}"
    if _GENERIC_TWO_PARTS_RE.search(text):
        return "CORRECT", "generic two-parts framing (branch names not required here)"
    if has_a or has_b:
        return "INCORRECT", "mentions only one branch"
    return "OMITTED", None


_GENERIC_PART_RE = {"A": re.compile(r"\bpart a\b|\bfirst part\b", re.I), "B": re.compile(r"\bpart b\b|\bsecond part\b", re.I)}


def classify_c_method(text, task, which):
    """which: 'A' or 'B'. CORRECT if the text mentions EITHER the branch's
    domain keyword OR a generic 'part A'/'first part' alias (found
    necessary the same way as classify_c1_root's fix: the D-mode prompt
    template itself introduces "the first part"/"the second part" phrasing
    for A4/A5, and some models -- mostly qwen2.5-coder in this run --
    consistently echo "part A"/"part B" rather than the task's domain
    keyword throughout, which the artifact's fixed position already
    disambiguates; that is not an omission), AND both its quantity and
    rate numbers (evidence that the method correctly identifies what
    needs to be multiplied)."""
    keyword = task["keywordA"] if which == "A" else task["keywordB"]
    qty = task["qtyA"] if which == "A" else task["qtyB"]
    rate = task["rateA"] if which == "A" else task["rateB"]
    if not (_has_keyword(text, keyword) or _GENERIC_PART_RE[which].search(text)):
        return "OMITTED", None
    if _has_number_near(text, qty) and _has_number_near(text, rate):
        return "CORRECT", f"mentions {keyword!r} (or generic part-{which} alias) with qty={qty} and rate={rate}"
    if _numbers_in(text):
        return "INCORRECT", f"mentions {keyword!r} (or generic alias) but not matching qty/rate"
    return "OMITTED", None


def classify_c1_method_a(text, task):
    return classify_c_method(text, task, "A")


def classify_c1_method_b(text, task):
    return classify_c_method(text, task, "B")


def classify_c_subtotal(text, task, which):
    target = task["subtotalA"] if which == "A" else task["subtotalB"]
    if _has_number_near(text, target):
        return "CORRECT", f"shows subtotal {target}"
    if _numbers_in(text):
        return "INCORRECT", f"computation shown but not {target}"
    return "OMITTED", None


def classify_c_subtotal_a(text, task):
    return classify_c_subtotal(text, task, "A")


def classify_c_subtotal_b(text, task):
    return classify_c_subtotal(text, task, "B")


def classify_c_final_total(text, task):
    expected = task["final_total"]
    anchored = list(re.finditer(r"\b(final answer|final total|answer|total|result)\b\s*(is|:|=)?\s*\$?(-?\d[\d,]*\.?\d*)", text, re.I))
    if anchored:
        s = anchored[-1].group(3).replace(",", "")
        try:
            v = float(s)
            return ("CORRECT" if abs(v - expected) <= 0.01 else "INCORRECT"), s
        except ValueError:
            pass
    nums = _numbers_in(text)
    if not nums:
        return "OMITTED", None
    v = nums[-1]
    return ("CORRECT" if abs(v - expected) <= 0.01 else "INCORRECT"), str(v)


CLASSIFIERS = {
    "C1_ROOT": classify_c1_root,
    "C2_METHOD_A": classify_c1_method_a,
    "C3_METHOD_B": classify_c1_method_b,
    "C4_SUBTOTAL_A": classify_c_subtotal_a,
    "C5_SUBTOTAL_B": classify_c_subtotal_b,
    "C6_FINAL_TOTAL": classify_c_final_total,
}


def classify_claim(claim_id, text, task):
    if not text:
        return "OMITTED", None
    return CLASSIFIERS[claim_id](text.lower(), task)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))
    from schema9 import TASKS

    task = TASKS["BRANCH_01"]
    cases = [
        ("C1_ROOT", "We need to compute the cost of shirts and pants separately.", "CORRECT"),
        ("C1_ROOT", "We need to compute the cost of shirts only.", "INCORRECT"),
        ("C2_METHOD_A", "2 shirts at $20 each means multiplying 2 by 20.", "CORRECT"),
        ("C2_METHOD_A", "3 shirts at $25 each.", "INCORRECT"),
        ("C4_SUBTOTAL_A", "2 * 20 = 40", "CORRECT"),
        ("C4_SUBTOTAL_A", "2 * 20 = 45", "INCORRECT"),
        ("C6_FINAL_TOTAL", "The final total is 145.", "CORRECT"),
        ("C6_FINAL_TOTAL", "The final total is 150.", "INCORRECT"),
    ]
    failures = 0
    for claim_id, text, expected in cases:
        got, ev = classify_claim(claim_id, text, task)
        ok = got == expected
        if not ok:
            failures += 1
        print(f"{'OK ' if ok else 'FAIL'} {claim_id}: expected={expected} got={got} :: {text!r}")
    print(f"\n{failures} failures out of {len(cases)} self-test cases")
