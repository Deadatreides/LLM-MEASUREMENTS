"""recombination.py — Усиленный cross-slot M5 оператор рекомбинации для ASSEMBLE."""

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

import genotype as G
import mutate as M

def enhanced_m5_cross_slot(parent: dict, rng: random.Random, ctx: dict, registry, gen: int) -> Tuple[Optional[dict], dict]:
    """Усиленная рекомбинация: перенос целостного поддерева слота ASSEMBLE от донора."""
    donors = ctx.get("donors") or []
    if not donors:
        return None, {"operator": None}

    parent_root = parent.get("root", {})
    if parent_root.get("op") != "ASSEMBLE":
        # Запасной вариант на случай не-ASSEMBLE корня (не должен возникать на HETEROSTEP)
        return M.mutate(parent, rng, ctx, registry, gen)

    # Выбираем донора (предпочтительно ASSEMBLE-корневого)
    score_fn = ctx.get("donor_score_fn")
    assemble_donors = [d for d in donors if d.get("root", {}).get("op") == "ASSEMBLE"]
    target_donors = assemble_donors if assemble_donors else donors

    if score_fn is not None:
        weights = []
        for d in target_donors:
            try:
                weights.append(max(1e-6, float(score_fn(d))))
            except Exception:
                weights.append(1e-6)
        donor = rng.choices(target_donors, weights=weights, k=1)[0]
    else:
        donor = rng.choice(target_donors)

    donor_root = donor.get("root", {})
    if donor_root.get("op") != "ASSEMBLE":
        return M.mutate(parent, rng, ctx, registry, gen)

    # Выбираем слот (0..3) для обмена
    n_slots = len(parent_root.get("children", []))
    if n_slots == 0 or len(donor_root.get("children", [])) < n_slots:
        return M.mutate(parent, rng, ctx, registry, gen)

    slot_idx = rng.randrange(n_slots)
    donor_slot_subtree = copy.deepcopy(donor_root["children"][slot_idx])

    new_root = copy.deepcopy(parent_root)
    new_root["children"][slot_idx] = donor_slot_subtree

    child = G.genotype(new_root, gen=gen, parent_ids=[parent["complex_id"]], origin="mutate:M5_enhanced")
    if G.is_valid(child, registry) and child["complex_id"] != parent["complex_id"]:
        return child, {"operator": "M5", "donor_id": donor.get("complex_id")}

    # Если точечный перенос слота не дал валидного уникального мутанта, пробуем несколько раз или откат к обычному mutate
    for _ in range(10):
        slot_idx = rng.randrange(n_slots)
        donor = rng.choice(target_donors)
        donor_root = donor.get("root", {})
        if donor_root.get("op") == "ASSEMBLE" and len(donor_root.get("children", [])) >= n_slots:
            donor_slot_subtree = copy.deepcopy(donor_root["children"][slot_idx])
            new_root = copy.deepcopy(parent_root)
            new_root["children"][slot_idx] = donor_slot_subtree
            child = G.genotype(new_root, gen=gen, parent_ids=[parent["complex_id"]], origin="mutate:M5_enhanced")
            if G.is_valid(child, registry) and child["complex_id"] != parent["complex_id"]:
                return child, {"operator": "M5", "donor_id": donor.get("complex_id")}

    return M.mutate(parent, rng, ctx, registry, gen)
