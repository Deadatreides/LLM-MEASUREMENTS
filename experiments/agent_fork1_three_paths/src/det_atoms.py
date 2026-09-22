"""det_atoms.py — FORK-1 Path A / Path B-DET: pure Python, deterministic
re-implementation of the FILTER/AGGREGATE/DERIVE predicate. Independent
of the generator's own golden computation (not a passthrough of
`task["matched_ids"]`/`task["aggregate_value"]`) so the self-test
(PROTOCOL.md §1.4: 40/40 v=1 on golden F2, else HARD_STOP) is a genuine
cross-check, not a tautology. 0 LLM calls anywhere in this module.
"""

from __future__ import annotations


def filter_det(records: list, category: str, region: str) -> set:
    return {r["id"] for r in records if r["category"] == category and r["region"] == region}


def aggregate_det(ids, op: str, id_to_amount: dict) -> float:
    amounts = [id_to_amount[i] for i in ids if i in id_to_amount]
    if op == "SUM":
        return round(sum(amounts), 2) if amounts else 0.0
    if op == "COUNT":
        return float(len(amounts))
    if op == "MAX":
        return round(max(amounts), 2) if amounts else 0.0
    raise ValueError(op)


def derive_det(aggregate_value: float, threshold: float, comparator: str) -> str:
    if comparator == ">=":
        result = aggregate_value >= threshold
    elif comparator == "<":
        result = aggregate_value < threshold
    else:
        raise ValueError(comparator)
    return "ДА" if result else "НЕТ"
