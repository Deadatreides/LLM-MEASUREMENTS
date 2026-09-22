"""oracles.py — COPY (not import) of `agent_fork1_three_paths/src/
oracles.py`, byte-identical logic. Checkpoint/final parsers and checkers,
all LLM-judge-free, all local `v in {0,1}`. Negative-lookahead numeric
extractor (`^\\s*\\d+[.)](?!\\d)\\s*`, the "13.1" bug DELTA-0 found and
fixed) carried forward unchanged. `_ID_RE` already generalized in FORK-1
to match both F2's "TXN-####" and T_hard's "REC-####".
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_LIST_PREFIX_RE = re.compile(r"^\s*\d+[.)](?!\d)\s*")
_ID_RE = re.compile(r"[A-Z]+-\d{4,}")
_TOLERANCE = 0.01


def _strip_list_prefix(line: str) -> str:
    return _LIST_PREFIX_RE.sub("", line)


def _lines(text: str) -> list:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _as_number(s: str):
    s = (s or "").strip().strip('"\'`*').rstrip(".").strip()
    s = re.sub(r"^[€$£]\s*|\s*(?:EUR|USD|GBP)$", "", s).strip()
    m = _NUMBER_RE.fullmatch(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


# -- FILTER checkpoint: extracted id-SET vs golden id-set --------------------------------

def extract_ids(text: str) -> frozenset:
    return frozenset(_ID_RE.findall(text or ""))


def check_filter(text: str, golden_ids: list) -> dict:
    extracted = extract_ids(text)
    golden = frozenset(golden_ids)
    v = 1 if extracted == golden else 0
    return {"v": v, "extracted": sorted(extracted), "golden": sorted(golden)}


# -- AGGREGATE / numeric-final checkpoint: tier-1/tier-2 extraction, no guessing ---------

def extract_number(text: str):
    stripped_whole = _strip_list_prefix((text or "").strip())
    whole = _as_number(stripped_whole)
    if whole is not None:
        return whole
    hits = [v for ln in _lines(text) if (v := _as_number(_strip_list_prefix(ln))) is not None]
    if len(hits) == 1:
        return hits[0]
    return None


def check_aggregate(text: str, golden_value: float) -> dict:
    extracted = extract_number(text)
    v = 1 if (extracted is not None and abs(extracted - golden_value) <= _TOLERANCE) else 0
    return {"v": v, "extracted": extracted, "golden": golden_value}


def within_tolerance(a: float, b: float) -> bool:
    return abs(a - b) <= _TOLERANCE


# -- DERIVE / F2-style final checkpoint: ДА/НЕТ token, unambiguous only ------------------

_YES_RE = re.compile(r"^д\s*а\.?$", re.IGNORECASE)
_NO_RE = re.compile(r"^н\s*е\s*т\.?$", re.IGNORECASE)


def extract_yes_no(text: str):
    whole = _strip_list_prefix((text or "").strip()).strip()
    if _YES_RE.match(whole):
        return "ДА"
    if _NO_RE.match(whole):
        return "НЕТ"
    yes_lines = [ln for ln in _lines(text) if _YES_RE.match(_strip_list_prefix(ln))]
    no_lines = [ln for ln in _lines(text) if _NO_RE.match(_strip_list_prefix(ln))]
    if len(yes_lines) == 1 and not no_lines:
        return "ДА"
    if len(no_lines) == 1 and not yes_lines:
        return "НЕТ"
    return None


def check_derive(text: str, golden_final: str) -> dict:
    extracted = extract_yes_no(text)
    v = 1 if extracted == golden_final else 0
    return {"v": v, "extracted": extracted, "golden": golden_final}


def check_whole_final(text: str, golden_final: str) -> dict:
    return check_derive(text, golden_final)
