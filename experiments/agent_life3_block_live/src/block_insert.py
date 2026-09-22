"""block_insert.py — PROTOCOL.md §4.4: `BLOCK_INSERT`, второй рычаг этого
пакета. Вставляет CALL на ЗАРЕГИСТРИРОВАННЫЙ блок (композит) в слот,
предпочитая комплементарные слоты (`Imp(s,parent)=0`), но не требуя этого
жёстко (в отличие от `transfer_slot` -- "предпочтительно", см. PROTOCOL).
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


def block_insert(parent: dict, rng: random.Random, ctx: dict, registry, gen: int,
                 block_registry: dict, imp_fn) -> Tuple[Optional[dict], dict]:
    """`block_registry`: `{molecule_id: {"slot": int, ...}}`. Ноль
    зарегистрированных блоков (CTRL, реестр выключен навсегда) -> всегда
    откат на `M.mutate`, тот же код-путь, один флаг снаружи."""
    parent_root = parent.get("root", {})
    if parent_root.get("op") != "ASSEMBLE" or not block_registry:
        child, report = M.mutate(parent, rng, ctx, registry, gen)
        return child, {**report, "block_insert_attempted": True,
                       "block_fallback_reason": "no_blocks_or_not_assemble"}

    n_slots = len(parent_root.get("children", []))
    candidates = [(mid, e["slot"]) for mid, e in block_registry.items() if e["slot"] < n_slots]
    if not candidates:
        child, report = M.mutate(parent, rng, ctx, registry, gen)
        return child, {**report, "block_insert_attempted": True,
                       "block_fallback_reason": "no_slot_in_range"}

    parent_imp = imp_fn(parent)
    complementary = [(mid, s) for mid, s in candidates if s not in parent_imp]
    pool = complementary if complementary else candidates   # "предпочтительно", не жёстко

    for _ in range(MAX_RETRIES):
        mid, s = pool[rng.randrange(len(pool))]
        new_root = copy.deepcopy(parent_root)
        new_root["children"][s] = {"op": "CALL", "molecule": mid, "params": {}}

        child = G.genotype(new_root, gen=gen, parent_ids=[parent["complex_id"]],
                           origin="mutate:LIFE3_block_insert")
        if G.is_valid(child, registry) and child["complex_id"] != parent["complex_id"]:
            return child, {"operator": "BLOCK_INSERT", "molecule_id": mid, "slot": s,
                          "complementary": bool(complementary and (mid, s) in complementary),
                          "block_insert_attempted": True, "block_fallback_reason": None}

    child, report = M.mutate(parent, rng, ctx, registry, gen)
    return child, {**report, "block_insert_attempted": True,
                   "block_fallback_reason": "retries_exhausted"}
