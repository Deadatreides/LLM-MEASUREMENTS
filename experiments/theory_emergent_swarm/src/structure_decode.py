"""structure_decode.py — E4 core. Projects a noisy id-set onto the
nearest admissible STRUCTURE (equivalence class over record attributes)
instead of scoring 18 independent bits. See PROTOCOL_E4.md.

`agreement(s, C) = sum_i [ s_i if i in C else (1 - s_i) ]`, which for
binary votes equals `K - |S xor C|` -- so argmax agreement IS
nearest-codeword (minimum Hamming distance) decoding.

STRUCTURE_DET: class enumeration reads record["category"]/["region"]
deterministically. Flagged everywhere, per PROTOCOL_E4.md SS5 -- this is a
hybrid, never presented as a swarm beating a tool.
"""

from __future__ import annotations

import consensus_filter as CF


def enumerate_classes(task: dict) -> dict:
    """-> {(category, region): frozenset(ids)}. All non-empty classes, NO
    size filter (2-5 guarantee is generator knowledge, deliberately unused)."""
    classes: dict = {}
    for r in task["records"]:
        classes.setdefault((r["category"], r["region"]), set()).add(r["id"])
    return {k: frozenset(v) for k, v in classes.items()}


def agreement(support: dict, all_ids: list, member_ids: frozenset) -> float:
    total = 0.0
    for i in all_ids:
        s = support.get(i, 0.0)
        total += s if i in member_ids else (1.0 - s)
    return total


def decode(support: dict, all_ids: list, classes: dict) -> tuple:
    """-> (best_key, best_member_ids, best_score). Deterministic tie-break:
    lexicographically smallest (category, region) key wins."""
    best_key = None
    best_score = None
    for key in sorted(classes):
        score = agreement(support, all_ids, classes[key])
        if best_score is None or score > best_score + 1e-12:
            best_key, best_score = key, score
    return best_key, classes[best_key], best_score


def swarm_support(task: dict, votes_for_task: dict, admitted: list) -> dict:
    """Plain mode among Condorcet-admitted models -- E0's own measured
    choice (log-odds lost there, 0.806 vs 0.861), reused not reinvented.
    Empty admitted set -> all-zero support (abstain), never golden-filled."""
    cids = CF.candidate_ids(task)
    pv = CF.per_id_votes(task, votes_for_task)
    if not admitted:
        return {i: 0.0 for i in cids}
    return {i: sum(pv[m][i] for m in admitted) / len(admitted) for i in cids}


def single_support(task: dict, votes_for_task: dict, model_id: str) -> dict:
    cids = CF.candidate_ids(task)
    pv = CF.per_id_votes(task, votes_for_task)
    return {i: float(pv[model_id][i]) for i in cids}
