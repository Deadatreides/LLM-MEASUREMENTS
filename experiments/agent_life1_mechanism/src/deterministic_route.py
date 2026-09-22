"""deterministic_route.py — замена `arch2.heterostep_seeds._greedy_order`
(PROTOCOL.md §5). Та ломает тай-брейк через `max(set(ds.models), key=gain)`
-- порядок обхода `set` строк зависит от `PYTHONHASHSEED` процесса, поэтому
НЕ воспроизводима между отдельными запусками (`agent_a5_live_m/BLOCKERS.md`
находка 2). Здесь -- явный алфавитный тай-брейк (канонический порядок) плюс
полный перебор ничьих (для Gate R1 в `PROTOCOL.md` §3: нужно найти, КАКОЙ
именно из 6 вариантов дал архивный прогон A5, не полагаясь на процессный
хэш-порядок).

`route`/`_slot`/`_fallback` из `arch2.heterostep_seeds` переиспользуются БЕЗ
ИЗМЕНЕНИЙ (чистые конструкторы генотипа, не зависят от tie-break) -- сюда
не скопированы.
"""

from __future__ import annotations

import live_dataset as LD

HSTEP = LD.HSTEP
HSEED = LD.HSEED
G = LD.G


def _gain(ds, task_ids, kind, model, covered) -> int:
    return len({t for t in task_ids if ds.cells[(t, kind, model)]["status"] == "PASS"} - covered)


def greedy_order_deterministic(ds, task_ids: list, kind: str) -> list:
    """Канонический (воспроизводимый между процессами) жадный порядок:
    тай-брейк -- алфавитно по `model_id`. Совпадает с
    `heterostep_seeds._greedy_order` побитово на НЕ-ничейных шагах;
    расходится только там, где та зависит от хэш-порядка."""
    covered: set = set()
    chosen: list = []
    rest = set(ds.models)
    while rest:
        gains = {m: _gain(ds, task_ids, kind, m, covered) for m in rest}
        best = max(gains.values())
        if best == 0 and chosen:
            break
        tied = sorted(m for m, g in gains.items() if g == best)
        pick = tied[0]
        chosen.append(pick)
        covered |= {t for t in task_ids if ds.cells[(t, kind, pick)]["status"] == "PASS"}
        rest.discard(pick)
    return chosen


def enumerate_greedy_orders(ds, task_ids: list, kind: str) -> list:
    """Полный перебор ВСЕХ жадных порядков, различающихся выбором на любом
    ничейном шаге (PROTOCOL.md §5: для READ на этой сетке -- 6 вариантов;
    для FORMAT/LOOKUP/COMPUTE -- ровно 1, ничьих нет). Список отсортирован
    для воспроизводимости; каждый элемент -- tuple(model_id, ...)."""

    def rec(covered, chosen, rest):
        if not rest:
            return [tuple(chosen)]
        gains = {m: _gain(ds, task_ids, kind, m, covered) for m in rest}
        best = max(gains.values())
        if best == 0 and chosen:
            return [tuple(chosen)]
        tied = sorted(m for m, g in gains.items() if g == best)
        out = []
        for m in tied:
            new_covered = covered | {t for t in task_ids if ds.cells[(t, kind, m)]["status"] == "PASS"}
            out.extend(rec(new_covered, chosen + [m], rest - {m}))
        return out

    orders = rec(set(), [], set(ds.models))
    return sorted(set(orders))


def all_order_variants(ds, task_ids: list) -> list:
    """Декартово произведение вариантов по всем 4 родам -> список
    {kind: [model,...]} словарей. На этой сетке даёт ровно 6 (READ) x 1 x 1
    x 1 = 6 полных вариантов затравки (PROTOCOL.md §5)."""
    per_kind = {k: enumerate_greedy_orders(ds, task_ids, k) for k in HSTEP.STEP_KINDS}
    variants = [{}]
    for kind, orders in per_kind.items():
        variants = [{**v, kind: order} for v in variants for order in orders]
    return variants


def routes_from_orders(orders: dict) -> dict:
    """{kind: greedy_order} -> {"b1":..., "b2":..., "b3":...} per-kind
    model-list dicts, ровно та форма, что принимает `heterostep_seeds.route`."""
    return {
        "b1": {k: [orders[k][0]] for k in HSTEP.STEP_KINDS},
        "b2": {k: orders[k][:2] for k in HSTEP.STEP_KINDS},
        "b3": {k: list(orders[k]) for k in HSTEP.STEP_KINDS},
    }


def build_references(orders: dict) -> dict:
    """{kind: order} -> {"REF_B1_route": genotype, "REF_B2_greedy2": genotype,
    "REF_B3_greedy_cover": genotype} -- HSEED.route() переиспользован БЕЗ
    ИЗМЕНЕНИЙ (чистый конструктор ASSEMBLE-дерева)."""
    r = routes_from_orders(orders)
    return {
        "REF_B1_route": G.genotype(HSEED.route(r["b1"]), gen=0, origin="seed:REF_B1_route"),
        "REF_B2_greedy2": G.genotype(HSEED.route(r["b2"]), gen=0, origin="seed:REF_B2_greedy2"),
        "REF_B3_greedy_cover": G.genotype(HSEED.route(r["b3"]), gen=0, origin="seed:REF_B3_greedy_cover"),
    }


def find_matching_variant(ds, task_ids: list, target_ids: dict) -> dict:
    """PROTOCOL.md §3, Gate R1. `target_ids` = {"REF_B1_route": cid, "REF_B2_greedy2": cid,
    "REF_B3_greedy_cover": cid} восстановленные из трасс архивного прогона A5.
    Перебирает все варианты `all_order_variants`, возвращает первый совпавший
    ПОЛНОСТЬЮ (все 3 complex_id равны target) вместе с самими orders, либо
    {"matched": False, "checked": [...]} если ни один не совпал."""
    variants = all_order_variants(ds, task_ids)
    checked = []
    for orders in variants:
        refs = build_references(orders)
        got = {name: g["complex_id"] for name, g in refs.items()}
        checked.append(got)
        if got == target_ids:
            return {"matched": True, "orders": orders, "references": refs, "complex_ids": got}
    return {"matched": False, "checked": checked, "n_variants": len(variants)}
