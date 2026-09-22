"""default_channel.py — LIFE-5 Phase A: зафиксированный операционный
дефолт после провала этажа блока (LIFE-3/4, `UNIT_LIVE_FAIL`/`UNIT4_FAIL`,
1/6 оба раза). ASSEMBLE + комплементарный slot-HGT (`p_slot_hgt=0.35`,
`transfer_slot` из `slot_hgt.py`, копия LIFE-2/3/4 БЕЗ ИЗМЕНЕНИЙ), без
регистрации блоков и БЕЗ `BLOCK_INSERT` вообще -- не как CTRL-плечо
LIFE-3/4 (пустой навсегда `block_registry`, диспетчер всё ещё катает
кубик на несуществующую ветку), а буквально без этого кода-пути: диспетчер
здесь -- ровно два исхода (`roll<p_slot_hgt` -> `transfer_slot`, иначе ->
`mutate`), один `rng.random()` на попытку, как во всех прежних пакетах.

`births.jsonl`'s схема строк ИДЕНТИЧНА LIFE-3/4 (те же поля, `molecule_id`/
`block_insert_attempted`/`block_fallback_reason` всегда `None`/`False`) --
намеренно, чтобы `slot_map.py` мог обрабатывать births любого из 28
прогонов (24 существующих + 4 новых) одним кодом, без спецслучаев по
источнику.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402
import slot_hgt as ST       # noqa: E402

E = LD.E
M = LD.M

P_SLOT_HGT = 0.35


def _write_births_row(births_path: Path, *, gen: int, child_id, parent_id, report: dict,
                      admitted: bool, delta_impset) -> None:
    births_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "gen": gen, "child_id": child_id, "parent_id": parent_id,
        "operator": report.get("operator"),
        "donor_id": report.get("donor_id"), "molecule_id": report.get("molecule_id"),
        "slot": report.get("slot"), "complementary": report.get("complementary"),
        "hgt_attempted": bool(report.get("hgt_attempted", False)),
        "hgt_fallback_reason": report.get("hgt_fallback_reason"),
        "block_insert_attempted": bool(report.get("block_insert_attempted", False)),
        "block_fallback_reason": report.get("block_fallback_reason"),
        "delta_impset": delta_impset, "admitted": admitted,
    }
    with open(births_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_default_reproduce(births_path: Path, imp_fn, p_slot_hgt: float = P_SLOT_HGT):
    """Ровно два исхода -- ни регистрации, ни `BLOCK_INSERT` не существует
    как код-путь в этом диспетчере (Phase A, PROTOCOL.md)."""
    def default_reproduce(ev, gen: int, results: dict, control_ids: list) -> None:
        reference_cids_live = set(ev.references)
        ctx = {
            "generators": [m for m in ev.registry.generator_ids()],
            "observables": [],
            "feasible_params": ev.backend.feasible_params("heterostep"),
            "donors": [ev.archive[c] for c in ev.archive if c not in reference_cids_live],
            "composites": list(ev.registry.composite_ids()),
            "assemble_n_steps": ev.assemble_n_steps,
            # ПРЕДНАМЕРЕННО нет "donor_score_fn" -- ни здесь, ни в откате на M.mutate.
        }

        n_fresh = int(E.POP_SIZE * E.FRESH_FRACTION)
        n_children = max(0, E.POP_SIZE - len(ev.population) - n_fresh)
        parents = ev.elite or ev.population
        parent_w = [max(1e-6, ev.accounts[c].w) for c in parents] if parents else []
        added = 0

        for _ in range(n_children * 4):
            if added >= n_children or not parents:
                break
            parent_id = ev.rng.choices(parents, weights=parent_w, k=1)[0]
            parent = ev.genotypes[parent_id]

            roll = ev.rng.random()               # РОВНО один вызов
            if roll < p_slot_hgt:
                child, report = ST.transfer_slot(parent, ev.rng, ctx, ev.registry, gen, imp_fn)
            else:
                child, report = M.mutate(parent, ev.rng, ctx, ev.registry, gen)
                report = dict(report or {})
                report.setdefault("hgt_attempted", False)

            delta = None
            if child is not None:
                delta = len(imp_fn(child)) - len(imp_fn(parent))
            admitted = bool(child and child["complex_id"] not in ev.genotypes)
            _write_births_row(births_path, gen=gen, child_id=(child["complex_id"] if child else None),
                              parent_id=parent_id, report=report or {}, admitted=admitted,
                              delta_impset=delta)
            if admitted:
                ev._admit(child, gen)
                ev._emit_birth_edges(child, gen, report, parent_id)
                added += 1

        fresh = 0
        for _ in range(n_fresh * 6):
            if fresh >= n_fresh:
                break
            g, report = M.random_genotype(ev.rng, ctx, ev.registry, gen)
            delta = None
            if g is not None:
                delta = len(imp_fn(g)) - 0
            admitted = bool(g and g["complex_id"] not in ev.genotypes)
            _write_births_row(births_path, gen=gen, child_id=(g["complex_id"] if g else None),
                              parent_id=None,
                              report={**(report or {}), "hgt_attempted": False},
                              admitted=admitted, delta_impset=delta)
            if admitted:
                ev._admit(g, gen)
                ev._emit_birth_edges(g, gen, report, None)
                fresh += 1

    return default_reproduce
