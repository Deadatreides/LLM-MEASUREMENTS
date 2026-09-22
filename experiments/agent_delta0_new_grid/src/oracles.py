"""oracles.py — DELTA-0 C3: checkpoint/final parsers and checkers, all
LLM-judge-free, all local `v in {0,1}`. Written fresh for this package
(PROTOCOL.md §10 — copying a utility within the package is allowed, this
is not an import of any prior package's code): the numeric extractor
already builds in the negative-lookahead fix LIFE-9 had to discover the
hard way this session ("13.1" must not be eaten down to "1" by a
list-prefix stripper) — applied from the start here, not rediscovered.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_LIST_PREFIX_RE = re.compile(r"^\s*\d+[.)](?!\d)\s*")   # lookahead: don't eat "13.1"'s "13."
_ID_RE = re.compile(r"TXN-\d{4,}")
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


# -- AGGREGATE checkpoint: numeric extraction (tier-1/tier-2), no oracle-independent guess --

def extract_number(text: str):
    """Tier 1: whole (list-prefix-stripped) response is a bare number.
    Tier 2: exactly one value-only line. Else None (INAPPLICABLE-equivalent).
    """
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


# -- DERIVE / final checkpoint: ДА/НЕТ token, unambiguous only ---------------------------

_YES_RE = re.compile(r"^д\s*а\.?$", re.IGNORECASE)
_NO_RE = re.compile(r"^н\s*е\s*т\.?$", re.IGNORECASE)


def extract_yes_no(text: str):
    """Tier 1: whole (stripped) response is exactly ДА or НЕТ.
    Tier 2: exactly one line is exactly ДА or НЕТ (and not both present).
    Else None (ambiguous/absent -> v=0, never guessed).
    """
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
    """Whole-task measurement (C6): PROTOCOL.md §4 -- asks ONLY for the
    final ДА/НЕТ, identical parser to `check_derive`."""
    return check_derive(text, golden_final)
