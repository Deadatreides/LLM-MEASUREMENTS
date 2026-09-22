"""
PREDICTION side (mirrors experiment 8's dependency_engine.py, same
propagation rule and same root-origin-based ERROR_LOCAL/ERROR_DEPENDENCY
fix). Deliberately kept as a SEPARATE module from ground_truth9.py
(section 6's independence requirement) -- this module only ever looks at
CLASSIFIED claim statuses (the output of classifiers9.py applied to
actual text), never at which claim we, the experimenters, chose to
corrupt.
"""
import collections

STATUS_MAP = {"CORRECT": "CORRECT", "INCORRECT": "INCORRECT", "AMBIGUOUS": "AMBIGUOUS", "OMITTED": "UNKNOWN", "UNKNOWN": "UNKNOWN"}


def normalize_status(raw_status):
    return STATUS_MAP.get(raw_status, "UNKNOWN")


def propagate(local_status_by_claim, depends_on):
    normalized = {cid: normalize_status(s) for cid, s in local_status_by_claim.items()}
    memo = {}

    def resolve(cid):
        if cid in memo:
            return memo[cid]
        own = normalized.get(cid, "UNKNOWN")
        deps = depends_on.get(cid, [])
        if own == "INCORRECT":
            memo[cid] = ("INCORRECT", "LOCAL")
            return memo[cid]
        if not deps:
            memo[cid] = (own, "NONE" if own == "CORRECT" else "LOCAL_UNKNOWN")
            return memo[cid]
        dep_results = [resolve(d)[0] for d in deps if d in normalized]
        if "INCORRECT" in dep_results:
            memo[cid] = ("INCORRECT", "PROPAGATED")
            return memo[cid]
        if "UNKNOWN" in dep_results or "AMBIGUOUS" in dep_results:
            memo[cid] = ("UNKNOWN", "PROPAGATED_UNKNOWN")
            return memo[cid]
        memo[cid] = (own, "NONE" if own == "CORRECT" else "LOCAL_UNKNOWN")
        return memo[cid]

    for cid in normalized:
        resolve(cid)
    return memo


def downstream_of(cid, depends_on):
    reverse = collections.defaultdict(list)
    for c, deps in depends_on.items():
        for d in deps:
            reverse[d].append(c)
    seen = set()
    frontier = [cid]
    while frontier:
        cur = frontier.pop()
        for nxt in reverse.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def root_origins_of(effective_by_claim, claim_order):
    origins = [cid for cid in claim_order if effective_by_claim.get(cid, ("UNKNOWN", "NONE")) == ("INCORRECT", "LOCAL")]

    def ancestors(cid, depends_on, seen=None):
        seen = seen or set()
        for d in depends_on.get(cid, []):
            if d not in seen:
                seen.add(d)
                ancestors(d, depends_on, seen)
        return seen

    return origins


def predicted_affected_set(local_status_by_claim, depends_on, claim_order):
    """The SYSTEM's prediction: every claim whose EFFECTIVE status (after
    propagation over the observed local classifications) is INCORRECT.
    This is what section 8 calls PREDICTED_AFFECTED_SET -- computed
    purely from classified text, with no knowledge of where we actually
    injected the error."""
    effective = propagate(local_status_by_claim, depends_on)
    affected = {cid for cid in claim_order if effective[cid][0] == "INCORRECT"}

    def ancestors(cid, seen=None):
        seen = seen or set()
        for d in depends_on.get(cid, []):
            if d not in seen:
                seen.add(d)
                ancestors(d, seen)
        return seen

    root_origins = [cid for cid in affected if not (ancestors(cid) & affected)]
    return affected, root_origins, effective
