"""
Section 5/6/12: local vs effective claim correctness, propagated through
the frozen DEPENDS_ON graph (tasks/schema8.py). Frozen alongside the
schema -- the propagation ALGORITHM itself is part of the protocol, not
tuned after seeing which tasks/models it flatters.

Status mapping note: the mechanical classifiers (configs/classifiers8.py,
configs/claims_code_wrapper.py) produce 4 raw labels -- CORRECT,
INCORRECT, OMITTED, AMBIGUOUS -- inherited from the claims_def.py
convention used throughout experiments 3-7. Spec section 5 asks for
exactly CORRECT/INCORRECT/UNKNOWN/AMBIGUOUS. OMITTED ("claim never
addressed") is mapped onto UNKNOWN for every top-level status/metric in
this experiment (we cannot establish correctness for a claim that was
never stated, which is exactly what UNKNOWN means) -- the original
OMITTED/AMBIGUOUS distinction is kept in raw evidence fields for
diagnostic purposes but never affects localization/MRS arithmetic.

Propagation rule (section 6, applied literally): a downstream claim that
is locally consistent with a WRONG upstream premise is NOT correct.
  effective_status(claim):
    - own_local INCORRECT -> effective INCORRECT, origin=LOCAL (this claim
      has its own independently-detectable defect, regardless of upstream)
    - else if any dependency's effective_status == INCORRECT -> effective
      INCORRECT, origin=PROPAGATED
    - else if any dependency's effective_status in (UNKNOWN, AMBIGUOUS) ->
      effective UNKNOWN, origin=PROPAGATED_UNKNOWN (can't vouch for a
      claim built on an unverifiable premise, but this is NOT the same as
      a confirmed error)
    - else (all deps effectively CORRECT, own_local not INCORRECT) ->
      effective = own_local (CORRECT or UNKNOWN), origin=NONE
"""
import collections

STATUS_MAP = {"CORRECT": "CORRECT", "INCORRECT": "INCORRECT", "AMBIGUOUS": "AMBIGUOUS", "OMITTED": "UNKNOWN", "UNKNOWN": "UNKNOWN"}


def normalize_status(raw_status):
    return STATUS_MAP.get(raw_status, "UNKNOWN")


def propagate(local_status_by_claim, depends_on):
    """local_status_by_claim: {claim_id: raw_status}. depends_on:
    {claim_id: [claim_id, ...]}. Returns {claim_id: (effective_status, origin)}."""
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


def error_taxonomy_label(cid, effective_by_claim, root_origins):
    """Section 12: ERROR_LOCAL / ERROR_DEPENDENCY / UNKNOWN for one claim.

    IMPORTANT (found while validating on artificial CASE 2, see
    metrics/artificial_cases8.py): a downstream claim that is "formally
    consistent with a wrong upstream premise" (section 6) will ALSO
    usually fail its OWN local check, because our mechanical classifiers
    compare each claim to absolute ground truth, not to what its sibling
    claims specifically asserted. propagate()'s per-claim "origin" tag
    therefore says LOCAL for such claims too (their own text really does
    deviate from truth) -- using that tag directly would mislabel every
    claim in a cascade as an independent ERROR_LOCAL, defeating the
    point of section 12's distinction. The graph-level root-origin set
    (dependency_engine.minimal_repair_set's second return value) is what
    correctly separates "the earliest, graph-independent defect" from
    "everything downstream of it, whether or not it also happens to
    deviate from truth on its own" -- use THAT, not the raw per-claim tag.

    (ERROR_GLOBAL is reserved for cross-artifact SEAM inconsistencies the
    dependency graph itself cannot express -- computed in seams8.py.)"""
    status, _ = effective_by_claim[cid]
    if status == "CORRECT":
        return "NO_ERROR"
    if status == "UNKNOWN":
        return "UNKNOWN"
    if cid in root_origins:
        return "ERROR_LOCAL"
    return "ERROR_DEPENDENCY"


def downstream_of(cid, depends_on):
    """All claims that (transitively) depend on cid."""
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


def minimal_repair_set(effective_by_claim, depends_on, claim_order):
    """Section 21: MRS = origin claims (LOCAL-origin, effectively
    INCORRECT) UNION everything transitively downstream of each origin.

    This is NOT just the root-cause claim(s) -- section 15 explicitly
    warns that repairing only the origin (e.g. A1) and leaving A2/A3/A4's
    ALREADY-GENERATED TEXT untouched is wrong, because that downstream
    text still literally embeds the old (now-orphaned) values/derivation
    and would stay wrong even after the origin is fixed, unless it too is
    regenerated. So MRS = the full "affected subgraph" rooted at each
    origin claim, not a single point fix. A claim already downstream of
    another origin is not double-listed as its own separate origin (its
    own local defect, if any, is a symptom of the same root, per
    propagate()'s LOCAL-still-fires-on-its-own-deviation behavior --
    origins are deduplicated by checking no ancestor is itself an origin)."""
    origins = [cid for cid in claim_order
               if effective_by_claim.get(cid, ("UNKNOWN", "NONE")) == ("INCORRECT", "LOCAL")]

    def ancestors(cid, seen=None):
        seen = seen or set()
        for d in depends_on.get(cid, []):
            if d not in seen:
                seen.add(d)
                ancestors(d, seen)
        return seen

    root_origins = [o for o in origins if not (ancestors(o) & set(origins))]

    mrs = set()
    for o in root_origins:
        mrs.add(o)
        mrs |= downstream_of(o, depends_on)
    return sorted(mrs, key=lambda c: claim_order.index(c) if c in claim_order else 999), root_origins
