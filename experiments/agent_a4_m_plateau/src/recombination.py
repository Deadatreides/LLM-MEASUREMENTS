"""recombination.py — swap_assemble_slot: cross-slot рекомбинация с уклоном.

Копия (не импорт) `agent_a3_mstar/src/recombination.py` — оператор сам по
себе НЕ меняется в A4 (PROTOCOL.md §4): слот выбирается с уклоном
P(LOOKUP)=P(COMPUTE)=2*P(READ)=2*P(FORMAT), вероятность применения —
внешний параметр (`effective_p_cross` в `src/orchestrator.py::
custom_reproduce`, с клэмпом `min_frac_local`), не константа здесь.
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

# PROTOCOL.md §4: P(LOOKUP)=P(COMPUTE)=2*P(READ)=2*P(FORMAT), веса на слоты
# 0=READ,1=FORMAT,2=LOOKUP,3=COMPUTE (heterostep.STEP_KINDS порядок).
SLOT_WEIGHTS = (1.0, 1.0, 2.0, 2.0)

MAX_RETRIES = 10


def swap_assemble_slot(parent: dict, rng: random.Random, ctx: dict, registry, gen: int) -> Tuple[Optional[dict], dict]:
    """Перенос ЦЕЛОГО поддерева выбранного (с уклоном) слота ASSEMBLE от донора.

    Донор выбирается тем же приёмом, что `arch2.mutate.m5_horizontal_transfer`
    (взвешивание через `ctx["donor_score_fn"]`, если задан) -- не переписывает
    донорскую логику заново, только ограничивает выбор ASSEMBLE-корневыми
    донорами и слот -- выбранным с уклоном индексом, а не случайным узлом
    произвольной глубины по всему дереву донора.

    При неудаче (нет валидных ASSEMBLE-доноров, результат невалиден/дубликат)
    -- до MAX_RETRIES попыток, затем откат на немодифицированный
    `arch2.mutate.mutate(...)`.
    """
    donors = ctx.get("donors") or []
    if not donors:
        return None, {"operator": None}

    parent_root = parent.get("root", {})
    if parent_root.get("op") != "ASSEMBLE":
        return M.mutate(parent, rng, ctx, registry, gen)

    n_slots = len(parent_root.get("children", []))
    if n_slots == 0:
        return M.mutate(parent, rng, ctx, registry, gen)

    weights = list(SLOT_WEIGHTS[:n_slots]) if n_slots <= len(SLOT_WEIGHTS) else [1.0] * n_slots

    assemble_donors = [d for d in donors if d.get("root", {}).get("op") == "ASSEMBLE"
                      and len(d["root"].get("children", [])) >= n_slots]
    if not assemble_donors:
        return M.mutate(parent, rng, ctx, registry, gen)

    score_fn = ctx.get("donor_score_fn")

    def pick_donor():
        if score_fn is not None:
            w = []
            for d in assemble_donors:
                try:
                    w.append(max(1e-6, float(score_fn(d))))
                except Exception:                              # noqa: BLE001
                    w.append(1e-6)
            return rng.choices(assemble_donors, weights=w, k=1)[0]
        return rng.choice(assemble_donors)

    for _ in range(MAX_RETRIES):
        donor = pick_donor()
        slot_idx = rng.choices(range(n_slots), weights=weights, k=1)[0]

        new_root = copy.deepcopy(parent_root)
        new_root["children"][slot_idx] = copy.deepcopy(donor["root"]["children"][slot_idx])

        child = G.genotype(new_root, gen=gen, parent_ids=[parent["complex_id"]],
                           origin="mutate:A4_swap_assemble_slot")
        if G.is_valid(child, registry) and child["complex_id"] != parent["complex_id"]:
            return child, {"operator": "SWAP_SLOT", "donor_id": donor.get("complex_id"),
                          "slot": slot_idx}

    return M.mutate(parent, rng, ctx, registry, gen)
