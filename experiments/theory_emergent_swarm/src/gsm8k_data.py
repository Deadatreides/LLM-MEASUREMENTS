"""gsm8k_data.py — GSM8K loader + answer seam. The golden `#### N` line is
used ONLY for scoring, never shown to a model and never used to build any
hypothesis space (there is none -- the answer domain is unbounded).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "gsm8k_test.parquet"
SPLIT_SEED = 20260824

_GOLD_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_LIST_PREFIX_RE = re.compile(r"^\s*\d+[.)](?!\d)\s*")      # LIFE-9 lesson, negative lookahead
_ANSWER_LINE_RE = re.compile(r"ANSWER\s*[:=]\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def _to_float(s: str):
    try:
        return float((s or "").replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def golden(answer_field: str):
    m = _GOLD_RE.search(answer_field or "")
    return _to_float(m.group(1)) if m else None


def load(n: int, seed: int = SPLIT_SEED) -> list:
    """-> [{'task_id','question','gold'}], deterministic sample of the test split."""
    df = pd.read_parquet(DATA)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    out = []
    for i, row in df.iterrows():
        g = golden(row["answer"])
        if g is None:
            continue
        out.append({"task_id": f"GSM_{i:04d}", "question": row["question"].strip(), "gold": g})
    return out


def build_solve_prompt(task: dict) -> str:
    return (
        "Solve the math problem. Show your calculation briefly.\n"
        "Finish with a final line in exactly this format:\n"
        "ANSWER: <number>\n\n"
        f"Problem: {task['question']}"
    )


def build_verify_prompt(task: dict, candidate: float) -> str:
    cand = int(candidate) if float(candidate).is_integer() else candidate
    return (
        "Check whether the proposed answer to the math problem is correct.\n"
        "Recompute briefly, then finish with a final line in exactly this format:\n"
        "VERDICT: YES\n"
        "or\n"
        "VERDICT: NO\n\n"
        f"Problem: {task['question']}\n\n"
        f"Proposed answer: {cand}"
    )


def extract_answer(text: str):
    """Tier 1: the requested ANSWER: line. Tier 2: last number in the text."""
    stripped = "\n".join(_LIST_PREFIX_RE.sub("", ln) for ln in (text or "").splitlines())
    hits = _ANSWER_LINE_RE.findall(stripped)
    if hits:
        return _to_float(hits[-1])
    nums = _NUM_RE.findall(stripped)
    return _to_float(nums[-1]) if nums else None


_VERDICT_RE = re.compile(r"VERDICT\s*[:=]\s*(YES|NO)", re.IGNORECASE)


def extract_verdict(text: str):
    hits = _VERDICT_RE.findall(text or "")
    if hits:
        return hits[-1].upper()
    t = (text or "").strip().upper()
    if t.startswith("YES"):
        return "YES"
    if t.startswith("NO"):
        return "NO"
    return None


def same(a, b, tol: float = 1e-4) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))
