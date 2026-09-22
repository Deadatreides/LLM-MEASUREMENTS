"""
Section 5/9: mechanical (regex-only, never LLM-judge) claim classifiers for
the 4 MATH tasks' C1-C4 claims. Frozen alongside tasks/schema8.py -- not
retuned after seeing real generations. Every classifier below is checked
against held-out synthetic phrasings in _SELFTEST (run via
`python classifiers8.py`) BEFORE being trusted on real data, same
discipline as experiments 6/7's local_classifiers.py.

CODE tasks reuse claims_def_code.py's C1-C7 classifiers unmodified (via
configs/claims_code.py) plus a new C8_TESTS_PASS (mechanical pytest-style
execution, LEVEL 0, see configs/run_tests.py).
"""
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+")
_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


def _sentences(text):
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


def _numbers_in(text):
    out = []
    for m in _NUM_RE.finditer(text):
        s = m.group(0).replace("$", "").replace(",", "").replace("%", "")
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def _has_number_near(text, target, tol=0.01, window=60):
    """True if `target` appears as a number within `window` chars of any
    occurrence (crude but sufficient: we only need presence, not position,
    since these are short single-purpose artifacts)."""
    for m in _NUM_RE.finditer(text):
        s = m.group(0).replace("$", "").replace(",", "").replace("%", "")
        try:
            v = float(s)
        except ValueError:
            continue
        if abs(v - target) <= tol:
            return True
    return False


_NEG_WORD_RE = re.compile(r"\b(not|never|n't|without|isn'?t|instead of|rather than|different from|unlike|as opposed to|in contrast to)\b", re.I)


def _classify_method_claim(text, pos_re, neg_re, window=25):
    """Shared negation-aware method-claim classifier (found needed via
    Phase 4 artificial CASE_3/CASE_6/CASE_7 sanity checks -- a plain
    pos/neg regex pair misreads "...not compound interest..." or
    "...since it is not compounded..." as evidence FOR the wrong method,
    the same class of bug fixed repeatedly in experiments 3/6/7's claim
    classifiers). A neg-pattern match immediately preceded by a negation
    word within `window` chars does NOT count as asserting the wrong
    method -- it is dropped (and does not by itself prove the right
    method either; a genuine positive match is still required for
    CORRECT)."""
    pos = pos_re.search(text)
    neg_matches = list(neg_re.finditer(text))
    real_neg = None
    for m in neg_matches:
        pre = text[max(0, m.start() - window): m.start()]
        if not _NEG_WORD_RE.search(pre):
            real_neg = m
            break
    if pos and real_neg:
        return "AMBIGUOUS", f"pos={pos.group(0)!r} neg={real_neg.group(0)!r}"
    if pos:
        return "CORRECT", pos.group(0)
    if real_neg:
        return "INCORRECT", real_neg.group(0)
    return "OMITTED", None


# ---------------------------------------------------------------- MATH_01
_M01_POS_RE = re.compile(r"\bsimple interest\b|\bp\s*\*?\s*r\s*\*?\s*t\b|\bprincipal\s*(x|\*|times|by)\s*(the\s*)?rate\s*(x|\*|times|by)\s*(the\s*)?time\b", re.I)
_M01_NEG_RE = re.compile(r"\bcompound interest\b|\(1\s*\+\s*r\)|\bcompound\w*\b", re.I)


def _c1_method_math01(text):
    return _classify_method_claim(text, _M01_POS_RE, _M01_NEG_RE)


def _c2_values_math01(text):
    ok = _has_number_near(text, 500) and (_has_number_near(text, 4) or _has_number_near(text, 0.04)) and _has_number_near(text, 3)
    if ok:
        return "CORRECT", "500/4%/3 present"
    if _numbers_in(text):
        return "INCORRECT", "some numbers present but not matching 500/4%/3"
    return "OMITTED", None


def _c3_computation_math01(text):
    if _has_number_near(text, 60):
        return "CORRECT", "shows intermediate 60"
    if _numbers_in(text):
        return "INCORRECT", "computation shown but not 60"
    return "OMITTED", None


def _c4_final_math01(text):
    return _generic_final_answer(text, 60)


# ---------------------------------------------------------------- MATH_02
_M02_POS_RE = re.compile(r"\bweighted average\b|\*\s*0?\.3\b|\*\s*30\s*%|\bweight\w*\b", re.I)
_M02_NEG_RE = re.compile(r"\bsimple average\b|\bunweighted\b|\(80\s*\+\s*90\)\s*/\s*2\b|\baverage of the two\b", re.I)


def _c1_method_math02(text):
    return _classify_method_claim(text, _M02_POS_RE, _M02_NEG_RE)


def _c2_values_math02(text):
    ok = _has_number_near(text, 80) and _has_number_near(text, 90) and (_has_number_near(text, 30) or _has_number_near(text, 0.3)) and (_has_number_near(text, 70) or _has_number_near(text, 0.7))
    if ok:
        return "CORRECT", "80/90/30%/70% present"
    if _numbers_in(text):
        return "INCORRECT", "some numbers present but not matching all four"
    return "OMITTED", None


def _c3_computation_math02(text):
    if _has_number_near(text, 24) and _has_number_near(text, 63):
        return "CORRECT", "shows 24 and 63"
    if _has_number_near(text, 87):
        return "CORRECT", "shows 87 as intermediate sum"
    if _numbers_in(text):
        return "INCORRECT", "computation shown but not matching"
    return "OMITTED", None


def _c4_final_math02(text):
    return _generic_final_answer(text, 87)


# ---------------------------------------------------------------- MATH_03
_M03_POS_RE = re.compile(r"\btotal distance\b.{0,40}\btotal time\b|\bdistance\s*/\s*time\b", re.I)
_M03_NEG_RE = re.compile(r"\baverage of the two speeds\b|\(30\s*\+\s*60\)\s*/\s*2\b|\bsimple average\b.{0,20}speed", re.I)


def _c1_method_math03(text):
    return _classify_method_claim(text, _M03_POS_RE, _M03_NEG_RE)


def _c2_values_math03(text):
    ok = _has_number_near(text, 60) and _has_number_near(text, 30) and _has_number_near(text, 60) and _has_number_near(text, 60)
    # 60 appears 3x (distance, distance, speed2) plus 30 (speed1) -- presence-only check
    ok = _has_number_near(text, 30) and _has_number_near(text, 60)
    if ok:
        return "CORRECT", "30/60 present"
    if _numbers_in(text):
        return "INCORRECT", "numbers present but not matching"
    return "OMITTED", None


def _c3_computation_math03(text):
    if _has_number_near(text, 120) and _has_number_near(text, 3):
        return "CORRECT", "shows total distance 120 and total time 3"
    if _numbers_in(text):
        return "INCORRECT", "computation shown but not matching 120/3"
    return "OMITTED", None


def _c4_final_math03(text):
    return _generic_final_answer(text, 40)


# ---------------------------------------------------------------- MATH_04
_M04_POS_RE = re.compile(r"\bsequential\w*\b|\bsuccessiv\w*\b|\balready.discounted\b|\bthen\b.{0,20}\b(10|ten)\s*%.{0,20}\bof\b", re.I)
_M04_NEG_RE = re.compile(r"\b(combined|total|additive)\b.{0,15}\b30\s*%|\b30\s*%\s*off\b|\badd\w*\b.{0,10}(20|10)\s*%", re.I)


def _c1_method_math04(text):
    return _classify_method_claim(text, _M04_POS_RE, _M04_NEG_RE)


def _c2_values_math04(text):
    ok = _has_number_near(text, 200) and (_has_number_near(text, 20) or _has_number_near(text, 0.2)) and (_has_number_near(text, 10) or _has_number_near(text, 0.1))
    if ok:
        return "CORRECT", "200/20%/10% present"
    if _numbers_in(text):
        return "INCORRECT", "numbers present but not matching"
    return "OMITTED", None


def _c3_computation_math04(text):
    if _has_number_near(text, 160):
        return "CORRECT", "shows intermediate 160"
    if _numbers_in(text):
        return "INCORRECT", "computation shown but not 160"
    return "OMITTED", None


def _c4_final_math04(text):
    return _generic_final_answer(text, 144)


def _generic_final_answer(text, expected):
    """C4 is checked against the LAST number preceded by an answer-ish
    keyword within the text; falls back to the last standalone number in
    the text if no keyword anchor is found."""
    anchored = list(re.finditer(r"\b(final answer|answer|result|total)\b\s*(is|:|=)?\s*\$?(-?\d[\d,]*\.?\d*)", text, re.I))
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


MATH_CLASSIFIERS = {
    "MATH_01": {"C1_METHOD": _c1_method_math01, "C2_VALUES": _c2_values_math01, "C3_COMPUTATION": _c3_computation_math01, "C4_FINAL_ANSWER": _c4_final_math01},
    "MATH_02": {"C1_METHOD": _c1_method_math02, "C2_VALUES": _c2_values_math02, "C3_COMPUTATION": _c3_computation_math02, "C4_FINAL_ANSWER": _c4_final_math02},
    "MATH_03": {"C1_METHOD": _c1_method_math03, "C2_VALUES": _c2_values_math03, "C3_COMPUTATION": _c3_computation_math03, "C4_FINAL_ANSWER": _c4_final_math03},
    "MATH_04": {"C1_METHOD": _c1_method_math04, "C2_VALUES": _c2_values_math04, "C3_COMPUTATION": _c3_computation_math04, "C4_FINAL_ANSWER": _c4_final_math04},
}


def classify_math_claim(task_id, claim_id, text):
    return MATH_CLASSIFIERS[task_id][claim_id](text)


_SELFTEST = {
    ("MATH_01", "C1_METHOD"): [
        ("We use simple interest: I = P * r * t.", "CORRECT"),
        ("This requires the compound interest formula (1+r)^t.", "INCORRECT"),
        ("The bank pays money.", "OMITTED"),
    ],
    ("MATH_01", "C2_VALUES"): [
        ("Principal P=500, rate r=4%, time t=3 years.", "CORRECT"),
        ("We have some numbers like 10 and 20.", "INCORRECT"),
    ],
    ("MATH_01", "C3_COMPUTATION"): [
        ("500 * 0.04 * 3 = 60.", "CORRECT"),
        ("500 * 0.04 * 3 = 61.", "INCORRECT"),
    ],
    ("MATH_01", "C4_FINAL_ANSWER"): [
        ("The final answer is 60.", "CORRECT"),
        ("The final answer is $62.43.", "INCORRECT"),
        ("So the total interest is 60 dollars.", "CORRECT"),
    ],
    ("MATH_03", "C1_METHOD"): [
        ("Average speed = total distance / total time = 120/3.", "CORRECT"),
        ("Just take the average of the two speeds: (30+60)/2.", "INCORRECT"),
    ],
    ("MATH_03", "C3_COMPUTATION"): [
        ("Total distance is 120 miles, total time is 3 hours.", "CORRECT"),
        ("Total distance is 120 miles, total time is 4 hours.", "INCORRECT"),
    ],
    ("MATH_04", "C1_METHOD"): [
        ("Apply the discounts sequentially: first 20% off, then 10% off the already-discounted price.", "CORRECT"),
        ("Combine into one 30% off discount.", "INCORRECT"),
    ],
    ("MATH_04", "C4_FINAL_ANSWER"): [
        ("Final answer: 144.", "CORRECT"),
        ("Final answer: 140.", "INCORRECT"),
    ],
}

if __name__ == "__main__":
    failures = 0
    total = 0
    for (task, claim), cases in _SELFTEST.items():
        for text, expected in cases:
            total += 1
            got, ev = classify_math_claim(task, claim, text)
            ok = got == expected
            if not ok:
                failures += 1
            print(f"{'OK ' if ok else 'FAIL'} {task}/{claim}: expected={expected} got={got} :: {text!r}")
    print(f"\n{failures} failures out of {total} self-test cases")
