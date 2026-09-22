"""eigen_consensus.py — LABEL-FREE reliability from the agreement matrix's
leading eigenvector (spectral meta-learner, Parisi et al. PNAS 2014, here
for K-way choice).

Why this matters for the project: every consensus rule so far (E0's
Condorcet, G1's pi>0.5*pi_best, M2's reliability ranking) needed LABELLED
calibration data to estimate pi_m. That is the binding practical
constraint -- on a real task you have no labels. The agreement structure
alone contains the information: for conditionally-independent voters,

    P(i agrees with j) = p_i*p_j + (1-p_i)(1-p_j)/(K-1)

is monotone increasing in both p_i and p_j, so the off-diagonal agreement
matrix is approximately rank-1 and its leading eigenvector ORDERS the
voters by reliability without ever seeing a gold answer.

This module only ranks; whether the ranking matches the labelled truth is
an empirical question, measured explicitly (Spearman) rather than assumed.
"""

from __future__ import annotations

import math


def agreement_matrix(votes: dict, models: list, task_ids: list) -> list:
    """votes[task][model] -> label or None. Diagonal is left at 0 so the
    self-agreement term cannot dominate the eigenvector."""
    n = len(models)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            both = agree = 0
            for t in task_ids:
                a, b = votes[t].get(models[i]), votes[t].get(models[j])
                if a is None or b is None:
                    continue
                both += 1
                agree += int(a == b)
            A[i][j] = agree / both if both else 0.0
    return A


def leading_eigenvector(A: list, iters: int = 500, tol: float = 1e-12) -> list:
    """Power iteration. Sign is fixed so the vector is non-negative."""
    n = len(A)
    v = [1.0 / math.sqrt(n)] * n
    for _ in range(iters):
        w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x * x for x in w))
        if norm < tol:
            return [1.0 / n] * n
        w = [x / norm for x in w]
        if sum(abs(w[i] - v[i]) for i in range(n)) < tol:
            v = w
            break
        v = w
    if sum(v) < 0:
        v = [-x for x in v]
    return [max(x, 0.0) for x in v]


def eigen_weights(votes: dict, models: list, task_ids: list) -> dict:
    """-> {model: weight >= 0}, no labels used anywhere."""
    v = leading_eigenvector(agreement_matrix(votes, models, task_ids))
    s = sum(v) or 1.0
    return {m: v[i] / s for i, m in enumerate(models)}


def weighted_vote(per_model: dict, weights: dict, tie_order: list):
    """Weighted plurality. Deterministic tie-break by `tie_order`."""
    tally: dict = {}
    for m, lab in per_model.items():
        if lab is None:
            continue
        tally[lab] = tally.get(lab, 0.0) + weights.get(m, 0.0)
    if not tally:
        return None, 0.0
    best = max(tally.items(), key=lambda kv: (kv[1], -tie_order.index(kv[0])
                                              if kv[0] in tie_order else 0))
    second = sorted((v for k, v in tally.items() if k != best[0]), reverse=True)
    return best[0], best[1] - (second[0] if second else 0.0)


def spearman(a: list, b: list) -> float:
    """Rank correlation, for checking eigen order against labelled truth."""
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((r - ma) ** 2 for r in ra))
    db = math.sqrt(sum((r - mb) ** 2 for r in rb))
    return num / (da * db) if da and db else float("nan")
