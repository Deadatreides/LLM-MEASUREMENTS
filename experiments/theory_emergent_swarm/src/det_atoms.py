"""det_atoms.py — COPY (not import) of `agent_fork1_three_paths/src/
det_atoms.py`, byte-identical logic. Pure Python, deterministic
re-implementation of FILTER/AGGREGATE/DERIVE. Reused here as the E2 SUM
step (never asked of an LLM) and as a verifier, never a generator.
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
