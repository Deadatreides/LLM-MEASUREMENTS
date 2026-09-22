"""slot_hgt.py — `transfer_slot`: ЕДИНСТВЕННЫЙ механический рычаг этого
пакета (PROTOCOL.md §5). Донор допустим для слота `s` ТОЛЬКО если
`Imp(s,donor)=1 ∧ Imp(s,recipient)=0` — комплементарность покрытия, НЕ
skeleton/semantic similarity (`ctx["donor_score_fn"]` не читается здесь
вовсе и не кладётся в `ctx` нигде в пакете — единственный способ выбора и
слота, и донора в этом файле -- равномерный `rng.choice`).

Откат при отсутствии допустимого донора -- на немодифицированный
`arch2.mutate.mutate(...)`, НИКОГДА не на similarity-HGT (LIFE-1's
`swap_assemble_slot` сюда не импортирован и не скопирован -- это ровно тот
оператор, который пакет заменяет).
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
    """`imp_fn(genotype_dict) -> frozenset[int]` -- ImpSet замкнутый на
    (ds, panel, best_single) вызывающей стороной (см. `metrics_lib.imp_set`),
    кэшируется по `complex_id` вызывающей стороной для скорости.

    `ctx["donors"]` -- archive ∪ alive, эталоны уже исключены (тот же
    механизм, что LIFE-1/A4: `[ev.archive[c] for c in ev.archive if c not
    in set(ev.references)]`)."""
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

    # -- допустимые (slot, donor) пары: Imp(s,donor)=1 ∧ Imp(s,recipient)=0 --
    valid_by_slot: dict = {}
    for s in range(n_slots):
        if s in parent_imp:
            continue                      # реципиент уже Imp на этом слоте -- переносить некуда
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
        s = rng.choice(slots_available)                 # равномерно по допустимым слотам
        donor = rng.choice(valid_by_slot[s])             # равномерно по допустимым донорам

        new_root = copy.deepcopy(parent_root)
        new_root["children"][s] = copy.deepcopy(donor["root"]["children"][s])

        child = G.genotype(new_root, gen=gen, parent_ids=[parent["complex_id"]],
                           origin="mutate:LIFE2_transfer_slot")
        if G.is_valid(child, registry) and child["complex_id"] != parent["complex_id"]:
            # Комплементарность гарантирована КОНСТРУКЦИЕЙ выбора (donor/slot
            # взяты из valid_by_slot) -- slot_match=True всегда для успешного
            # transfer_slot; проверяется здесь же как sanity, не пересчитывается
            # заново метриками (PROTOCOL.md §1: slot_match -- корректность, не находка).
            return child, {"operator": "TRANSFER_SLOT", "donor_id": donor.get("complex_id"),
                          "slot": s, "complementary": True, "hgt_attempted": True,
                          "hgt_fallback_reason": None}

    child, report = M.mutate(parent, rng, ctx, registry, gen)
    return child, {**report, "hgt_attempted": True, "hgt_fallback_reason": "retries_exhausted"}
