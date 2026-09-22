"""
Section 5/9/13: negation-aware classifiers for LOCAL_RETRY answers -- short
free-text prose responses about ONE specific claim, not the structured
JSON CONTRACT the claims_def.py classifiers expect. A different mechanism
is needed here on purpose (same lesson as experiment 6's
_classify_spaces_not_stripped): reusing the JSON-field regex as-is would
either not fire (wrong field shape) or under/over-match free prose.

Each classifier: sentence-window based (split on . ! ?), returns
(classification, evidence_sentence) with classification in
CORRECT / INCORRECT / OMITTED. AMBIGUOUS is reserved for when a single
sentence matches both a positive and a negative signal (rare, but must not
be silently resolved either way).

Every classifier below was checked against a handful of held-out synthetic
phrasings (see _SELFTEST at the bottom, run once at import time in
__main__) covering common ways a small local model actually answers this
kind of question, BEFORE being used on real generations. None were tuned
after seeing real experiment 7 local-retry output.
"""
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text):
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


def _first_match(sentences, anchor_re, pos_re, neg_re):
    for sent in sentences:
        if not anchor_re.search(sent):
            continue
        has_pos = bool(pos_re.search(sent)) if pos_re else False
        has_neg = bool(neg_re.search(sent)) if neg_re else False
        if has_pos and has_neg:
            return "AMBIGUOUS", sent.strip()
        if has_pos:
            return "CORRECT", sent.strip()
        if has_neg:
            return "INCORRECT", sent.strip()
    return "OMITTED", None


# ---- C7_EDGE_SPACES (qwen3 / CODE_01) -- reused verbatim from experiment6
_SP_ANCHOR = re.compile(r"\b(space|punctuation)\w*", re.I)
_SP_TARGET = re.compile(r"\b(strip\w*|remov\w*|ignor\w*)\b", re.I)
_SP_NEG = re.compile(r"\b(not|never|n't)\b", re.I)
_SP_PRESERVE = re.compile(r"\b(preserv\w*|remain\w*|kept|keep\w*|includ\w*|as.is)\b", re.I)


def classify_spaces_not_stripped(text):
    for sent in _sentences(text):
        if not _SP_ANCHOR.search(sent):
            continue
        for m in _SP_TARGET.finditer(sent):
            if re.search(r"^\s*case\b", sent[m.end(): m.end() + 15]):
                continue
            pre = sent[:m.start()]
            if _SP_NEG.search(pre):
                return "CORRECT", sent.strip()
            return "INCORRECT", sent.strip()
        if _SP_PRESERVE.search(sent):
            return "CORRECT", sent.strip()
    return "OMITTED", None


# ---- C2_ARGC "exactly 1 argument" (llama/coder, several tasks)
_ARGC_ANCHOR = re.compile(r"\bargument\w*|\bparamet\w*", re.I)
_ARGC_ONE = re.compile(r"\b(one|1|single|exactly one)\b", re.I)
_ARGC_MANY = re.compile(r"\b(two|three|four|2|3|4|multiple|several|more than one|additional)\b", re.I)


def classify_argc_one(text):
    return _first_match(_sentences(text), _ARGC_ANCHOR, _ARGC_ONE, _ARGC_MANY)


# ---- C5_SEMANTICS_CASE "comparison is case-insensitive" (coder/CODE_01)
_CASE_ANCHOR = re.compile(r"\bcase\b", re.I)
_CASE_POS = re.compile(r"\b(insensitive|ignor\w*|regardless of case|not.{0,10}sensitive)\b", re.I)
_CASE_NEG = re.compile(r"\b(sensitive|matters?|distinguish\w*|matter)\b", re.I)


def classify_case_insensitive(text):
    return _first_match(_sentences(text), _CASE_ANCHOR, _CASE_POS, _CASE_NEG)


# ---- C7_EDGE_NO_EVEN "list with no evens returns 0" (qwen3/CODE_02)
# ---- C6_EDGE_EMPTY   "empty list returns 0"           (coder/CODE_02)
_RETURN0_ANCHOR = re.compile(r"\breturn\w*|\bresult\b|\braise\w*|\berror\b|\bexception\b|\boutput\w*", re.I)
_RETURN0_POS = re.compile(r"\b(0|zero)\b", re.I)
_RETURN0_NEG = re.compile(r"\b(none|null|error|exception|-1|raise\w*)\b", re.I)


def classify_returns_zero(text):
    return _first_match(_sentences(text), _RETURN0_ANCHOR, _RETURN0_POS, _RETURN0_NEG)


# ---- C7_EDGE_NOT_SAME "i and j must be different indices, i<j" (llama/CODE_06)
_IDX_ANCHOR = re.compile(r"\bindex|\bindices|\bi\b.{0,10}\bj\b", re.I)
_IDX_POS = re.compile(r"\b(different|distinct|not.{0,5}same|i\s*<\s*j|smaller|must not be equal)\b", re.I)
_IDX_NEG = re.compile(r"\b(same index|equal|can be the same|i\s*==\s*j|i\s*=\s*j)\b", re.I)


def classify_indices_distinct(text):
    return _first_match(_sentences(text), _IDX_ANCHOR, _IDX_POS, _IDX_NEG)


# ---- C3_RETURN_TYPE "return type is a tuple/pair of indices" (coder/CODE_06)
_RT_ANCHOR = re.compile(r"\breturn\w*|\btype\b", re.I)
_RT_POS = re.compile(r"\b(tuple|pair)\b.{0,20}\b(index|indices|position)|\b(index|indices|position)\w*.{0,20}\btuple\b", re.I)
_RT_NEG = re.compile(r"\b(list of value|the values themselves|list\b(?!.{0,15}index)|integer\b(?!.{0,15}index)|single value)\b", re.I)


def classify_return_tuple_of_indices(text):
    return _first_match(_sentences(text), _RT_ANCHOR, _RT_POS, _RT_NEG)


CLASSIFIERS = {
    ("CODE_01", "C7_EDGE_SPACES"): classify_spaces_not_stripped,
    ("CODE_01", "C2_ARGC"): classify_argc_one,
    ("CODE_01", "C5_SEMANTICS_CASE"): classify_case_insensitive,
    ("CODE_02", "C7_EDGE_NO_EVEN"): classify_returns_zero,
    ("CODE_02", "C2_ARGC"): classify_argc_one,
    ("CODE_02", "C6_EDGE_EMPTY"): classify_returns_zero,
    ("CODE_03", "C2_ARGC"): classify_argc_one,
    ("CODE_06", "C7_EDGE_NOT_SAME"): classify_indices_distinct,
    ("CODE_06", "C3_RETURN_TYPE"): classify_return_tuple_of_indices,
}


def classify_local_answer(task_id, claim_id, text):
    fn = CLASSIFIERS.get((task_id, claim_id))
    if fn is None:
        raise ValueError(f"no local classifier registered for ({task_id}, {claim_id})")
    return fn(text.lower())


_SELFTEST = {
    ("CODE_01", "C7_EDGE_SPACES"): [
        ("Spaces and punctuation should not be stripped; they remain part of the string for comparison.", "CORRECT"),
        ("The function should strip spaces before comparing, ignoring case.", "INCORRECT"),
        ("This is unrelated commentary about performance.", "OMITTED"),
    ],
    ("CODE_01", "C2_ARGC"): [
        ("The function takes exactly one argument, the string s.", "CORRECT"),
        ("It takes two arguments: the string and a case-sensitivity flag.", "INCORRECT"),
        ("The function computes a palindrome check.", "OMITTED"),
    ],
    ("CODE_01", "C5_SEMANTICS_CASE"): [
        ("The comparison should be case-insensitive, ignoring uppercase vs lowercase.", "CORRECT"),
        ("The comparison is case-sensitive, so case matters.", "INCORRECT"),
    ],
    ("CODE_02", "C7_EDGE_NO_EVEN"): [
        ("If there are no even numbers, the function should return 0.", "CORRECT"),
        ("If there are no even numbers, the function should return None.", "INCORRECT"),
    ],
    ("CODE_02", "C6_EDGE_EMPTY"): [
        ("For an empty list, the result should be 0.", "CORRECT"),
        ("For an empty list, it should raise an error.", "INCORRECT"),
    ],
    ("CODE_03", "C2_ARGC"): [
        ("fibonacci takes exactly one argument, n.", "CORRECT"),
        ("It takes two arguments, n and a memo dictionary.", "INCORRECT"),
    ],
    ("CODE_06", "C7_EDGE_NOT_SAME"): [
        ("Yes, i and j must be different indices, with i smaller than j.", "CORRECT"),
        ("The two indices can be the same index if needed.", "INCORRECT"),
    ],
    ("CODE_06", "C3_RETURN_TYPE"): [
        ("The return type should be a tuple of the two indices.", "CORRECT"),
        ("It should return a list of the matching values themselves.", "INCORRECT"),
    ],
}

if __name__ == "__main__":
    failures = 0
    for (task, claim), cases in _SELFTEST.items():
        for text, expected in cases:
            got, ev = classify_local_answer(task, claim, text)
            ok = got == expected
            if not ok:
                failures += 1
            print(f"{'OK ' if ok else 'FAIL'} {task}/{claim}: expected={expected} got={got} :: {text!r}")
    print(f"\n{failures} failures out of {sum(len(v) for v in _SELFTEST.values())} self-test cases")
