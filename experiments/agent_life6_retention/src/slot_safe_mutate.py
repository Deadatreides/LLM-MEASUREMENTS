"""slot_safe_mutate.py — LIFE-6 Branch A's single lever (PROTOCOL.md §3):
wraps `arch2.mutate.mutate` (never edited -- reused by import) so that if
the parent already has `Imp(s)=1` on some slot `s`, the accepted mutation
usually does not touch `children[s]`.

Design (see PROTOCOL.md for why this reading was chosen over the task's
literal "запрещены ИЛИ p_safe=0.05"): ONE continuous knob, not two
mechanisms. `p_safe=0.0` = full prohibition; `p_safe=0.05` (Branch A's
value) = usually forbidden, occasionally (5%) let through anyway so a
protected slot is never PERMANENTLY frozen (which could itself be a
pathological, un-asked-for side effect).

**Why a structural guard comes before the diff** (found this session by
reading `arch2/mutate.py` in full, not assumed): `addresses(root)` always
includes the node's own empty address, and `set_at(root, [], new_node)`
replaces the WHOLE root. Four of the six operators (M3's `wrap_budget`/
`wrap_par`, M4's `refine`, M5, M6) can therefore rewrite `child["root"]`
entirely -- it may end up `SWITCH`/`BUDGET`/a bare `CALL`/an arbitrary
donor subtree, with no `"children"` key, or a DIFFERENT number of
children than the parent's 4 slots. `mutate()`'s own report never carries
a slot/address (checked exhaustively: only `{"operator", "donor_id"?,
"composite_id"?}`), so there is no way to know what changed except by
diffing the actual output -- and a naive `child["root"]["children"][s]`
index would `KeyError`/silently misbehave on these whole-root-rewrite
outputs. So: first confirm `child["root"]` is still `ASSEMBLE`-shaped
with the same child count as `parent["root"]`; if not, treat the mutation
as touching EVERY protected slot (can't be attributed to one, and
"can't prove it's safe" must mean "not safe" for this lever to mean
anything). Only when structure is preserved does a per-index diff of the
protected slots' JSON make sense.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
ARCH2 = ROOT / "arch2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ARCH2) not in sys.path:
    sys.path.insert(0, str(ARCH2))

import mutate as M   # noqa: E402

MAX_RETRIES = 10


def _slot_json(node: dict) -> str:
    return json.dumps(node, sort_keys=True, ensure_ascii=False)


def _touches_protected_slot(parent_root: dict, child_root: dict, protected: frozenset) -> bool:
    """True = this mutation must be treated as unsafe (either it
    provably changed a protected slot, or the top-level structure was
    rewritten wholesale and slot attribution is impossible)."""
    if not protected:
        return False
    p_children = parent_root.get("children") or []
    c_children = child_root.get("children") or []
    if child_root.get("op") != "ASSEMBLE" or len(c_children) != len(p_children):
        return True   # whole-root rewrite -- can't attribute, treat as touching everything
    for s in protected:
        if s >= len(c_children):
            return True
        if _slot_json(c_children[s]) != _slot_json(p_children[s]):
            return True
    return False


def slot_safe_mutate(parent: dict, rng, ctx: dict, registry, gen: int, imp_fn,
                     p_safe: float = 0.05, max_retries: int = MAX_RETRIES
                     ) -> Tuple[Optional[dict], dict]:
    """Same call signature/contract as `arch2.mutate.mutate`
    (`(parent, rng, ctx, registry, gen) -> (child|None, report)`) plus
    `imp_fn`/`p_safe`/`max_retries` -- `orchestrator.py` binds the extra
    args via a closure before passing this to `default_channel.
    make_default_reproduce(..., mutate_fn=...)`, which only ever calls
    the 5-positional-arg shape."""
    protected = imp_fn(parent)   # frozenset of slot indices with Imp(s)=1

    if not protected or rng.random() < p_safe:
        child, report = M.mutate(parent, rng, ctx, registry, gen)
        report = dict(report or {})
        report["slot_safe_bypassed"] = bool(protected)
        return child, report

    parent_root = parent.get("root", {})
    for attempt in range(max_retries):
        child, report = M.mutate(parent, rng, ctx, registry, gen)
        if child is None:
            continue
        if _touches_protected_slot(parent_root, child.get("root", {}), protected):
            continue
        report = dict(report or {})
        report["slot_safe_bypassed"] = False
        report["slot_safe_attempts"] = attempt
        return child, report

    # Exhausted -- matches `mutate()`'s own exhaustion convention (return
    # None, no operator). Does NOT fall back to an unsafe mutation: that
    # would defeat the entire lever (PROTOCOL.md §3).
    return None, {"operator": None, "slot_safe_bypassed": False, "slot_safe_exhausted": True}
