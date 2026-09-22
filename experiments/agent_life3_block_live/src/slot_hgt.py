"""slot_hgt.py — `transfer_slot`: копия (не импорт — изоляция пакетов)
`agent_life2_complementary/src/slot_hgt.py` -- логика НЕ менялась
(PROTOCOL.md §4.1: p=0.35, комплементарность как единственный критерий
донора, `donor_score_fn` нигде в `ctx`, откат на `arch2.mutate.mutate`).
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
ARCH2 = ROOT / "arch2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ARCH2) not in sys.path:
    sys.path.insert(0, str(ARCH2))

import genotype as G   # noqa: E402
import mutate as M     # noqa: E402

MAX_RETRIES = 10


def transfer_slot(parent: dict, rng: random.Random, ctx: dict, registry, gen: int,
                  imp_fn) -> Tuple[Optional[dict], dict]:
    donors = ctx.get("donors") or []
    if not donors:
        child, report = M.mutate(parent, rng, ctx, registry, gen)
        return child, {**report, "hgt_attempted": True, "hgt_fallback_reason": "no_donors"}

    parent_root = parent.get("root", {})
    if parent_root.get("op") != "ASSEMBLE":
        child, report = M.mutate(parent, rng, ctx, registry, gen)
        return child, {**report, "hgt_attempted": True, "hgt_fallback_reason": "parent_not_assemble"}

    n_slots = len(parent_root.get("children", []))
    if n_slots == 0:
        child, report = M.mutate(parent, rng, ctx, registry, gen)
        return child, {**report, "hgt_attempted": True, "hgt_fallback_reason": "no_slots"}

    parent_imp = imp_fn(parent)

    valid_by_slot: dict = {}
    for s in range(n_slots):
        if s in parent_imp:
            continue
        candidates = [d for d in donors
                     if d.get("root", {}).get("op") == "ASSEMBLE"
                     and len(d["root"].get("children", [])) > s
                     and s in imp_fn(d)]
        if candidates:
            valid_by_slot[s] = candidates

    if not valid_by_slot:
        child, report = M.mutate(parent, rng, ctx, registry, gen)
        return child, {**report, "hgt_attempted": True, "hgt_fallback_reason": "no_complementary_donor"}

    slots_available = sorted(valid_by_slot)
    for _ in range(MAX_RETRIES):
        s = rng.choice(slots_available)
        donor = rng.choice(valid_by_slot[s])

        new_root = copy.deepcopy(parent_root)
        new_root["children"][s] = copy.deepcopy(donor["root"]["children"][s])

        child = G.genotype(new_root, gen=gen, parent_ids=[parent["complex_id"]],
                           origin="mutate:LIFE3_transfer_slot")
        if G.is_valid(child, registry) and child["complex_id"] != parent["complex_id"]:
            return child, {"operator": "TRANSFER_SLOT", "donor_id": donor.get("complex_id"),
                          "slot": s, "complementary": True, "hgt_attempted": True,
                          "hgt_fallback_reason": None}

    child, report = M.mutate(parent, rng, ctx, registry, gen)
    return child, {**report, "hgt_attempted": True, "hgt_fallback_reason": "retries_exhausted"}
