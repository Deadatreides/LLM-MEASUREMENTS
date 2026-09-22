"""
Simple, objective, regex-based extraction of answer/code and light textual
characteristics from raw model output. Deliberately conservative: anything
that can't be reliably determined is left as None/"UNKNOWN" rather than
guessed (see spec: don't fabricate a classification just to fill a field).
"""
import re

FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*:\s*(.+)", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
UNCERTAINTY_RE = re.compile(
    r"\b(not sure|might be|maybe|possibly|i think|unsure|approximately|i'm not certain|"
    r"could be|perhaps|it seems)\b",
    re.IGNORECASE,
)
SELF_CORRECTION_RE = re.compile(
    r"\b(wait,|actually,|let me reconsider|i made a mistake|correction:|on second thought|"
    r"let me recompute|let me redo|scratch that)\b",
    re.IGNORECASE,
)


def extract_final_answer_text(text):
    matches = FINAL_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip().strip(".").strip(), len(matches)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        return lines[-1], 0
    return None, 0


def normalize_numeric(s):
    if s is None:
        return None
    s = s.strip()
    s = s.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not matches:
        return None
    # if the "final answer" line is itself a full expression (e.g.
    # "128 - 47 - 36 = 55"), the intended answer is the last number, not
    # the first operand.
    try:
        v = float(matches[-1])
        if v.is_integer():
            return int(v)
        return v
    except ValueError:
        return None


def normalize_text_answer(s):
    if s is None:
        return None
    s = s.strip().strip(".").strip().lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s.strip()


def extract_code(text, function_name):
    fences = CODE_FENCE_RE.findall(text)
    if fences:
        # prefer the fence that actually defines the target function
        for f in fences:
            if f"def {function_name}" in f:
                return f.strip(), "fenced"
        return fences[-1].strip(), "fenced"
    if f"def {function_name}" in text:
        idx = text.index(f"def {function_name}")
        return text[idx:].strip(), "unfenced"
    return None, None


def text_characteristics(text):
    return {
        "length_chars": len(text),
        "has_explicit_uncertainty": bool(UNCERTAINTY_RE.search(text)),
        "has_self_correction_language": bool(SELF_CORRECTION_RE.search(text)),
        "has_contradiction": None,  # not reliably detectable automatically -> UNKNOWN
    }
