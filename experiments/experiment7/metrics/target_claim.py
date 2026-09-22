"""
Section 9: deterministic "which claim is problematic" rule, computed
STRICTLY from the first 8 generations of a (task, model, prompt=P1,
temperature=0.5) unit's already-existing claims_index rows (experiment 3's
data for llama/coder is literally seeds 1-8; qwen3's is seeds 1-8 of its
existing 16). Fixed BEFORE any local-retry generation happens -- applied
mechanically to historical data, no target-claim choice is made by looking
at experiment 7's own local-retry outcomes.

Priority order (verbatim from spec section 9):
  1. mechanically detected conflict (this claim shows BOTH a CORRECT and an
     INCORRECT classification across the first 8 -- true disagreement, not
     just noise)
  2. claim with a confirmed error (INCORRECT is the dominant class, no
     conflict)
  3. claim with a stable error cluster (INCORRECT dominant AND at least 2
     generations share near-identical evidence text)
  4. claim with maximum conflict/entropy among independent generations
     (excluding OMITTED-dominant claims, which carry little information)
  5. UNKNOWN claim (AMBIGUOUS is the dominant class) -- only case where a
     "cannot cleanly verify" claim is targeted, and only because it CAN be
     locally re-asked

If none of a task's 7 claims shows anything other than 100% CORRECT or
100% OMITTED in the first 8 -> LOCAL_RETRY_NOT_APPLICABLE (nothing to fix,
not simulated).
"""
import collections
import math


def _entropy(counts):
    total = sum(counts.values())
    if not total:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _norm_evidence(ev):
    if not ev:
        return None
    return " ".join(ev.lower().split())


def claim_features(claim_rows_first8):
    """claim_rows_first8: list of claims_index rows for ONE claim_id,
    already restricted to the first 8 seeds of one (task, model) unit."""
    cls_counts = collections.Counter(r["classification"] for r in claim_rows_first8)
    n = len(claim_rows_first8)
    dominant_cls, dominant_n = cls_counts.most_common(1)[0]
    has_conflict = cls_counts.get("CORRECT", 0) > 0 and cls_counts.get("INCORRECT", 0) > 0

    incorrect_evidence = [
        _norm_evidence(r["evidence"]) for r in claim_rows_first8 if r["classification"] == "INCORRECT"
    ]
    ev_counts = collections.Counter(e for e in incorrect_evidence if e)
    stable_error_cluster = bool(ev_counts) and ev_counts.most_common(1)[0][1] >= 2

    return {
        "n": n, "counts": dict(cls_counts), "dominant_cls": dominant_cls,
        "dominant_share": dominant_n / n if n else None,
        "has_conflict": has_conflict, "stable_error_cluster": stable_error_cluster,
        "entropy_bits": round(_entropy(cls_counts), 4),
        "omitted_dominant": dominant_cls == "OMITTED" and dominant_n / n >= 0.75,
    }


def select_target_claim(claims_by_id_first8, claim_order):
    """claims_by_id_first8: {claim_id: [rows]}. claim_order: task's fixed
    claim_id list (for deterministic tie-breaks, NOT for scoring). Returns
    (target_claim_id, priority_tier, features_by_claim) or
    (None, 'LOCAL_RETRY_NOT_APPLICABLE', features_by_claim)."""
    feats = {cid: claim_features(rows) for cid, rows in claims_by_id_first8.items()}

    # tier 1: mechanically detected conflict
    tier1 = [cid for cid in claim_order if feats[cid]["has_conflict"]]
    if tier1:
        return tier1[0], "TIER1_MECHANICAL_CONFLICT", feats

    # tier 2: confirmed error (INCORRECT dominant, majority)
    tier2 = [cid for cid in claim_order if feats[cid]["dominant_cls"] == "INCORRECT" and feats[cid]["dominant_share"] >= 0.5]
    if tier2:
        return tier2[0], "TIER2_CONFIRMED_ERROR", feats

    # tier 3: stable error cluster (subset of tier2 already covers dominant
    # INCORRECT; this tier catches INCORRECT-present-but-not-majority cases
    # where repeated identical evidence still marks a stable pattern)
    tier3 = [cid for cid in claim_order if feats[cid]["stable_error_cluster"]]
    if tier3:
        return tier3[0], "TIER3_STABLE_ERROR_CLUSTER", feats

    # tier 4: max conflict/entropy among non-OMITTED-dominant claims
    candidates = [cid for cid in claim_order if not feats[cid]["omitted_dominant"] and feats[cid]["entropy_bits"] > 0]
    if candidates:
        best = max(candidates, key=lambda cid: (feats[cid]["entropy_bits"], -claim_order.index(cid)))
        return best, "TIER4_MAX_ENTROPY", feats

    # tier 5: AMBIGUOUS dominant ("UNKNOWN", locally re-askable)
    tier5 = [cid for cid in claim_order if feats[cid]["dominant_cls"] == "AMBIGUOUS"]
    if tier5:
        return tier5[0], "TIER5_UNKNOWN_AMBIGUOUS", feats

    return None, "LOCAL_RETRY_NOT_APPLICABLE", feats
