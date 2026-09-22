"""
State classification S1-S6, computed ONLY from the first k=8 generations
of a unit -- never from Oracle@16, never from any later/other-arm data
(spec section 10's anti-leakage requirement). Thresholds are FIXED and
principled (chosen before looking at outcomes), not tuned by searching
over the dataset -- this is a small (n=36) dataset and tuning thresholds
on it would just be overfitting dressed up as a finding.
"""
import collections
import math


def cluster_of(row):
    if row["correctness_status"] == "CORRECT":
        return "CORRECT"
    if row["correctness_status"] == "UNKNOWN":
        return "UNKNOWN"
    sig = row.get("error_signature") or f"UNSIG_{row.get('error_class')}"
    return sig


def entropy_bits(counts):
    total = sum(counts.values())
    if not total:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def first_k_features(rows_sorted_by_seed, k=8):
    """rows_sorted_by_seed: ALL available rows for a unit, sorted by seed
    ascending. Only the first k are used -- everything below is computed
    strictly from that slice."""
    first_k = rows_sorted_by_seed[:k]
    first_half = rows_sorted_by_seed[: k // 2]

    clusters_k = [cluster_of(r) for r in first_k]
    counts_k = collections.Counter(clusters_k)
    n = len(first_k)

    pass_at_k = sum(1 for c in clusters_k if c == "CORRECT") / n if n else None
    unknown_rate_k = sum(1 for c in clusters_k if c == "UNKNOWN") / n if n else None
    oracle_at_k = 1.0 if "CORRECT" in counts_k else 0.0
    oracle_at_half = 1.0 if any(cluster_of(r) == "CORRECT" for r in first_half) else 0.0
    growth_half_to_k = oracle_at_k - oracle_at_half

    dominant_cluster, dominant_count = (counts_k.most_common(1)[0] if counts_k else (None, 0))
    dominant_share = dominant_count / n if n else None
    n_unique = len(counts_k)
    h_sem = entropy_bits(counts_k)
    n_eff = 1.0 / sum((c / n) ** 2 for c in counts_k.values()) if n else None

    # roughly-balanced two-way split between a CORRECT cluster and an
    # INCORRECT cluster -- candidate CONFLICT signature
    two_way_conflict = False
    if n_unique == 2 and "CORRECT" in counts_k:
        shares = sorted(v / n for v in counts_k.values())
        if shares[0] >= 0.3:  # neither side below 30% -- genuinely split, not just one outlier
            two_way_conflict = True

    return {
        "n": n, "pass_at_k": round(pass_at_k, 4) if pass_at_k is not None else None,
        "unknown_rate_k": round(unknown_rate_k, 4) if unknown_rate_k is not None else None,
        "oracle_at_k": oracle_at_k, "oracle_at_half": oracle_at_half, "growth_half_to_k": round(growth_half_to_k, 4),
        "dominant_cluster": dominant_cluster, "dominant_cluster_share": round(dominant_share, 4) if dominant_share is not None else None,
        "n_unique_clusters": n_unique, "h_sem_bits": round(h_sem, 4), "n_eff_clusters": round(n_eff, 3) if n_eff is not None else None,
        "two_way_conflict": two_way_conflict,
    }


def classify_state(features):
    """Fixed decision tree, documented thresholds:
      CONFIDENT  = dominant_cluster_share >= 0.75  (6+ of 8 agree)
      UNCERTAIN  = dominant_cluster_share <  0.75
    """
    f = features
    if f["unknown_rate_k"] is not None and f["unknown_rate_k"] >= 0.5:
        return "S5_UNKNOWN", "unknown_rate_k >= 0.5"

    confident = f["dominant_cluster_share"] is not None and f["dominant_cluster_share"] >= 0.75

    if confident:
        if f["dominant_cluster"] == "CORRECT":
            return "S1_CONFIDENT_CORRECT", "dominant_cluster_share>=0.75 and dominant=CORRECT"
        return "S3_CONFIDENT_WRONG", "dominant_cluster_share>=0.75 and dominant=error cluster"

    # uncertain region
    if f["two_way_conflict"]:
        return "S6_CONFLICT", "n_unique_clusters==2, roughly balanced CORRECT vs INCORRECT"
    if f["growth_half_to_k"] > 0:
        return "S2_UNCERTAIN_PRODUCTIVE", "uncertain, oracle still grew within observed window (half->k)"
    return "S4_NOISY", "uncertain, oracle flat/non-growing within observed window"


ACTION_BY_STATE = {
    "S1_CONFIDENT_CORRECT": "STOP",
    "S2_UNCERTAIN_PRODUCTIVE": "SAME_MODEL",
    "S3_CONFIDENT_WRONG": "SWITCH_MODEL",
    "S4_NOISY": "STOP",
    "S5_UNKNOWN": "CHANGE_PROMPT",  # stand-in for "seek verification" in this code-only setup (see REPORT6.md)
    "S6_CONFLICT": "SWITCH_MODEL",
}
