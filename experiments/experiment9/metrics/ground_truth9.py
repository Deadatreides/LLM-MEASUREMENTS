"""
GROUND TRUTH side (section 6's key methodological fix relative to
experiment 8: ground truth must NOT be computed by the same code that
produces the prediction). For artificial control cases, TRUE_AFFECTED_SET
is set BY CONSTRUCTION -- we know exactly which claim(s) we corrupted, so
we compute the true affected set with a bare graph traversal over the
FROZEN schema9.DEPENDS_ON, never touching classifiers9.py or any
generated/injected text. dependency_engine9.py (the prediction side) is a
completely separate module that never imports this one and vice versa.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))
from schema9 import DEPENDS_ON


def _downstream_of(cid):
    reverse = {}
    for c, deps in DEPENDS_ON.items():
        for d in deps:
            reverse.setdefault(d, []).append(c)
    seen = set()
    frontier = [cid]
    while frontier:
        cur = frontier.pop()
        for nxt in reverse.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def true_affected_set(injected_claim_ids):
    """injected_claim_ids: the claim(s) we, the experimenters, actually
    replaced with wrong text. Returns the TRUE affected set: each injected
    claim plus everything transitively downstream of it, per the frozen
    graph -- independent of any classifier or propagation code."""
    affected = set(injected_claim_ids)
    for cid in injected_claim_ids:
        affected |= _downstream_of(cid)
    return affected


def true_mrs_by_construction(injected_claim_ids):
    """For a SINGLE injected origin (the common case in this experiment's
    CASE A-D), the true minimal repair set is exactly the same as the true
    affected set for a linear/tree-shaped propagation graph, since fixing
    the origin(s) without regenerating what's downstream leaves stale,
    now-orphaned text there (same reasoning as experiment 8 section 15)."""
    return true_affected_set(injected_claim_ids)
