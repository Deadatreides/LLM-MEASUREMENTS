"""
Sections 14-15: two standalone diagnostic cases (not part of the 20-task
main DAG -- these test a mechanism, not this experiment's specific task
family). The mechanical claim classifiers used elsewhere in this
experiment have NO way to discover an undeclared shared premise on their
own (they only propagate edges that are already IN the frozen schema) --
so the honest question is narrower and more useful: can a minimal,
separate heuristic at least flag a PLAUSIBLE hidden dependency as
UNCERTAIN instead of silently asserting independence, WITHOUT also
massively over-flagging claims that merely share a number by coincidence
(section 15's SEMANTIC_SIMILARITY != DEPENDENCY warning)?

detect_shared_premise(text_a, text_b): a number is treated as a candidate
shared premise only if (a) it is "distinctive" (excludes 0/1/2/3, which
appear everywhere by chance) AND (b) it co-occurs with at least one shared
non-trivial context word in both texts (e.g., both mention "fee", not
just both mention "5"). This is a deliberately narrow, conservative
heuristic -- it is expected to catch the constructed HIDDEN case and
correctly stay silent on the constructed FALSE case, but it is NOT a
general hidden-dependency solver, and that limitation is reported as such
in REPORT9.md, not hidden.
"""
import re

_STOPWORDS = {"the", "a", "an", "is", "of", "to", "and", "for", "in", "on", "at", "with", "this", "that", "applied", "once", "has", "holds"}
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_TRIVIAL_NUMS = {0, 1, 2, 3}


def _numbers(text):
    out = set()
    for m in _NUM_RE.finditer(text):
        try:
            v = float(m.group(0).replace(",", ""))
            if v not in _TRIVIAL_NUMS:
                out.add(v)
        except ValueError:
            continue
    return out


def _context_words(text, number_str):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def detect_shared_premise(text_a, text_b):
    nums_a, nums_b = _numbers(text_a), _numbers(text_b)
    shared_nums = nums_a & nums_b
    if not shared_nums:
        return {"status": "NO_SHARED_NUMBER", "flag": False}

    words_a, words_b = _context_words(text_a, ""), _context_words(text_b, "")
    shared_words = words_a & words_b
    if shared_words:
        return {"status": "UNCERTAIN_POSSIBLE_HIDDEN_DEPENDENCY", "flag": True,
                "shared_numbers": sorted(shared_nums), "shared_context_words": sorted(shared_words)}
    return {"status": "SHARED_NUMBER_NO_CONTEXT_OVERLAP", "flag": False, "shared_numbers": sorted(shared_nums)}


HIDDEN_DEPENDENCY_CASE = {
    "claim_a_text": "Shipping cost for order A uses a fixed handling fee of $5, applied once: 5 + base_cost_A.",
    "claim_b_text": "Shipping cost for order B uses a fixed handling fee of $5, applied once: 5 + base_cost_B.",
    "expected": "UNCERTAIN_POSSIBLE_HIDDEN_DEPENDENCY (both reference the same 'handling fee' premise)",
}

FALSE_DEPENDENCY_CASE = {
    "claim_a_text": "Team A has 5 members working on the project.",
    "claim_b_text": "The meeting room holds 5 people at most.",
    "expected": "no dependency flagged (same number, unrelated context: team size vs room capacity)",
}


def main():
    import json
    hidden_result = detect_shared_premise(HIDDEN_DEPENDENCY_CASE["claim_a_text"], HIDDEN_DEPENDENCY_CASE["claim_b_text"])
    false_result = detect_shared_premise(FALSE_DEPENDENCY_CASE["claim_a_text"], FALSE_DEPENDENCY_CASE["claim_b_text"])

    out = {
        "hidden_dependency_case": {**HIDDEN_DEPENDENCY_CASE, "detected": hidden_result},
        "false_dependency_case": {**FALSE_DEPENDENCY_CASE, "detected": false_result},
        "hidden_correctly_flagged": hidden_result["flag"] is True,
        "false_correctly_not_flagged": false_result["flag"] is False,
    }
    with open("hidden_false_dependency9_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
