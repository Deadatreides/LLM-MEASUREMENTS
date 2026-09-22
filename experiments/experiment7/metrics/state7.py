"""
Section 6/7/10: state features computed STRICTLY from the first 8
generations of a unit (task_id, model_id), at two levels:

  - TASK level: is the whole CONTRACT artifact defect-free (no INCORRECT/
    AMBIGUOUS claim anywhere in it)? Drives the SAME_MODEL/SWITCH_MODEL
    decision, which acts on the whole artifact.
  - TARGET-CLAIM level: just the one claim selected by
    metrics/target_claim.py's fixed priority rule. Drives whether
    LOCAL_RETRY is worth trying and is the unit of the H5-style
    CONFIDENT_WRONG/SAME-vs-SWITCH comparison at claim granularity.

Anti-leakage: every function here takes an already-restricted "first 8"
list and never looks past it. Oracle@16 / continuation-arm data is used
elsewhere (outcome building), never here.

STATE A / B / C (spec section 10), computed independently at each level:
  STATE_A_STABLE_ERROR: low entropy AND dominant cluster is an error
                         cluster AND oracle@k is low (<1.0)
  STATE_B_PRODUCTIVE_UNCERTAINTY: not low-entropy-error, AND oracle grows
                         somewhere across the first-8 window (checked at
                         four checkpoints k=2,4,6,8, not just half-vs-full,
                         to avoid experiment 6's degenerate two-point
                         growth signal)
  STATE_C_NOISE:        not low-entropy-error AND oracle does not grow
                         anywhere across the first-8 window
Thresholds (entropy "low" <=1.0 bit, dominant share ">=0.75") are fixed
before looking at any experiment-7 outcome data -- same values used in
experiment 6's classifier, not re-tuned here.
"""
import collections
import math


def _entropy(counts):
    total = sum(counts.values())
    if not total:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def task_defect_signature(claim_rows_one_run):
    """claim_rows_one_run: the 7 claim rows for ONE generation (run_id).
    Returns 'CLEAN' if no claim is INCORRECT/AMBIGUOUS, else a frozenset
    signature of which claim_ids are broken (a crude but mechanical
    error-cluster id for the whole artifact)."""
    bad = frozenset(r["claim_id"] for r in claim_rows_one_run if r["classification"] in ("INCORRECT", "AMBIGUOUS"))
    return "CLEAN" if not bad else bad


def task_level_features(runs_first8, claims_by_run):
    """runs_first8: list of run_ids (already restricted to first 8 seeds,
    ascending). claims_by_run: {run_id: [7 claim rows]}."""
    n = len(runs_first8)
    sigs = [task_defect_signature(claims_by_run[rid]) for rid in runs_first8]
    counts = collections.Counter(sigs)
    dominant_sig, dominant_n = counts.most_common(1)[0]
    dominant_share = dominant_n / n if n else None

    oracle_track = []
    for k in (2, 4, 6, 8):
        if k > n:
            break
        oracle_track.append(1.0 if "CLEAN" in sigs[:k] else 0.0)
    growth_anywhere = any(oracle_track[i] > oracle_track[i - 1] for i in range(1, len(oracle_track)))

    pass_at_k = counts.get("CLEAN", 0) / n if n else None
    oracle_at_k = 1.0 if "CLEAN" in sigs else 0.0
    h_bits = round(_entropy(counts), 4)

    return {
        "n": n, "pass_at_k": round(pass_at_k, 4) if pass_at_k is not None else None,
        "oracle_at_k": oracle_at_k, "dominant_signature_is_clean": dominant_sig == "CLEAN",
        "dominant_share": round(dominant_share, 4) if dominant_share is not None else None,
        "n_unique_signatures": len(counts), "h_bits": h_bits,
        "oracle_track_k2_4_6_8": oracle_track, "growth_anywhere_in_first8": growth_anywhere,
    }


def claim_level_features(claim_rows_first8):
    """claim_rows_first8: the target claim's rows across the first 8 runs
    of this unit, ascending by seed."""
    n = len(claim_rows_first8)
    cls_seq = [r["classification"] for r in claim_rows_first8]
    counts = collections.Counter(cls_seq)
    dominant_cls, dominant_n = counts.most_common(1)[0]
    dominant_share = dominant_n / n if n else None

    oracle_track = []
    for k in (2, 4, 6, 8):
        if k > n:
            break
        oracle_track.append(1.0 if "CORRECT" in cls_seq[:k] else 0.0)
    growth_anywhere = any(oracle_track[i] > oracle_track[i - 1] for i in range(1, len(oracle_track)))

    pass_at_k = counts.get("CORRECT", 0) / n if n else None
    oracle_at_k = 1.0 if "CORRECT" in cls_seq else 0.0
    h_bits = round(_entropy(counts), 4)

    return {
        "n": n, "counts": dict(counts), "pass_at_k": round(pass_at_k, 4) if pass_at_k is not None else None,
        "oracle_at_k": oracle_at_k, "dominant_cls": dominant_cls,
        "dominant_share": round(dominant_share, 4) if dominant_share is not None else None,
        "h_bits": h_bits, "oracle_track_k2_4_6_8": oracle_track, "growth_anywhere_in_first8": growth_anywhere,
    }


def classify_state(features, dominant_is_error_fn, low_entropy_threshold=1.0, dominant_share_threshold=0.75):
    """Generic A/B/C classifier usable at either level.
    dominant_is_error_fn(features) -> bool: whether the dominant cluster
    represents an error (task: dominant_signature_is_clean is False;
    claim: dominant_cls != 'CORRECT' and dominant_cls != 'OMITTED')."""
    f = features
    low_entropy = f["h_bits"] <= low_entropy_threshold and (f["dominant_share"] or 0) >= dominant_share_threshold

    if low_entropy and dominant_is_error_fn(f) and f["oracle_at_k"] < 1.0:
        return "STATE_A_STABLE_ERROR"
    if f["growth_anywhere_in_first8"]:
        return "STATE_B_PRODUCTIVE_UNCERTAINTY"
    return "STATE_C_NOISE"
