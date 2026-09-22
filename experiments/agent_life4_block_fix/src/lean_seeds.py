"""lean_seeds.py — копия (не импорт) `agent_life3_block_live/src/
lean_seeds.py`. Бедный старт, Imp-пары найдены поверх ПЕРЕИСПОЛЬЗОВАННОЙ
(скопированной) сетки/панели LIFE-3 -- те же PASS-паттерны, тот же грид
(LIFE-4 PROTOCOL.md: grid=reuse). `M.random_genotype` не используется
(LIFE-2 установила: он не производит валидных ASSEMBLE на реестре
HETEROSTEP) -- все генотипы строятся явно из `heterostep_seeds.route`.

LIFE-4 Change 1 (единственное изменение в этом файле): `assert_lean_
population`'s вызов `ML.imp_set` теперь передаёт `registry` (сигнатура
`imp_set` изменилась везде, см. `metrics_lib.py`) -- в реальности seed-
генотипы никогда не содержат `cx.`-CALL (строятся напрямую из
`HSEED.route`), так что это не меняет никакое поведение здесь, просто
держит сигнатуру согласованной.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402
import metrics_lib as ML    # noqa: E402

HSTEP = LD.HSTEP
HSEED = LD.HSEED
G = LD.G

STEP_KINDS = HSTEP.STEP_KINDS
REFERENCE_NAMES = HSEED.REFERENCE_NAMES


def _gain(ds, panel, kind, model, covered):
    return len({t for t in panel if ds.cells[(t, kind, model)]["status"] == "PASS"} - covered)


def greedy_order_deterministic(ds, panel: list, kind: str) -> list:
    """Алфавитный тай-брейк, не хэш-порядок-зависимый
    `heterostep_seeds._greedy_order`."""
    covered: set = set()
    chosen: list = []
    rest = set(ds.models)
    while rest:
        gains = {m: _gain(ds, panel, kind, m, covered) for m in rest}
        best = max(gains.values())
        if best == 0 and chosen:
            break
        tied = sorted(m for m, g in gains.items() if g == best)
        pick = tied[0]
        chosen.append(pick)
        covered |= {t for t in panel if ds.cells[(t, kind, pick)]["status"] == "PASS"}
        rest.discard(pick)
    return chosen


def build_b1_b2_b3(ds, panel: list) -> dict:
    orders = {kind: greedy_order_deterministic(ds, panel, kind) for kind in STEP_KINDS}
    return {
        "b1": {k: [orders[k][0]] for k in STEP_KINDS},
        "b2": {k: orders[k][:2] for k in STEP_KINDS},
        "b3": {k: list(orders[k]) for k in STEP_KINDS},
        "orders": orders,
    }


def find_robust_pairs(ds, panel: list, best_single: dict) -> dict:
    """{slot_index: [(m1,m2), ...]} -- ВСЕ 2-модельные пары, дающие
    Imp(slot)=1 на ЭТОЙ панели. Найдено заново (не скопировано с LIFE-3),
    т.к. панель/сетка reuse'нута из LIFE-3, но пары пересчитаны прямо
    здесь -- не полагается на LIFE-3's собственные найденные значения.
    Одномодельный слот структурно не может дать Imp=1 -- минимум 2
    модели, проверено конструкцией (PROTOCOL.md §4.2)."""
    models = list(ds.models)
    out: dict = {}
    for s, kind in enumerate(STEP_KINDS):
        pairs = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                cov = ML.covered_set(ds, panel, s, [m1, m2])
                if len(cov) > best_single[s]:
                    pairs.append((m1, m2))
        out[s] = pairs
    return out


def _root_key(root: dict) -> str:
    import json as _json
    return _json.dumps(root, sort_keys=True, ensure_ascii=False)


def build_seeds_pop20_lean(ds, panel: list, best_single: dict) -> dict:
    """-> {name: genotype}. 3 эталона + РОВНО 20 СТРУКТУРНО РАЗЛИЧНЫХ
    не-эталонных: 1 нулевой (`b1`) + однослотовые Imp-пары (найдены
    динамически, round-robin по слотам) + заполнители без блоков --
    добор до ровно 20 РАЗЛИЧНЫХ (не имён -- корней дерева, PROTOCOL.md §5).
    Дедуп по каноническому JSON корня ДО построения генотипа -- иначе
    зацикливание по `models[idx % len(models)]` тихо создаёт дубликаты под
    разными именами (найдено LIFE-3 смоук-тестом одной клетки)."""
    routes = build_b1_b2_b3(ds, panel)
    b1 = routes["b1"]
    models = list(ds.models)

    out: dict = {}
    seen: set = set()

    def add(name: str, root: dict) -> bool:
        key = _root_key(root)
        if key in seen:
            return False
        seen.add(key)
        out[name] = root
        return True

    out["REF_B1_route"] = HSEED.route(b1)
    out["REF_B2_greedy2"] = HSEED.route(routes["b2"])
    out["REF_B3_greedy_cover"] = HSEED.route(routes["b3"])

    add("P_zero_b1", HSEED.route(b1))

    # -- источник 1: однослотовые Imp-пары, round-robin по слотам --
    robust = find_robust_pairs(ds, panel, best_single)
    per_slot_idx = {s: 0 for s in robust}
    progressed = True
    while progressed and (len(out) - 3) < 20:
        progressed = False
        for s, kind in enumerate(STEP_KINDS):
            if (len(out) - 3) >= 20:
                break
            pairs = robust.get(s, [])
            i = per_slot_idx[s]
            if i >= len(pairs):
                continue
            m1, m2 = pairs[i]
            per_slot_idx[s] += 1
            per_kind = dict(b1)
            per_kind[kind] = [m1, m2]
            if add(f"P_single_slot_{kind}_{i}", HSEED.route(per_kind)):
                progressed = True

    # -- источник 2: один атом на все 4 слота, по одному на КАЖДУЮ модель (<=6, без дублей) --
    for i, m in enumerate(models):
        if (len(out) - 3) >= 20:
            break
        add(f"P_single_model_{i}", HSEED.route({k: [m] for k in STEP_KINDS}))

    # -- источник 3: b1 с ОДНИМ слотом переключённым на альтернативную ОДИНОЧНУЮ
    # модель (не b1's выбор) -- всё ещё ImpSet=0 (одна модель на слот), даёт
    # много вариантов, если источников 1-2 не хватило до 20 --
    for s, kind in enumerate(STEP_KINDS):
        if (len(out) - 3) >= 20:
            break
        for m in models:
            if (len(out) - 3) >= 20:
                break
            if m == b1[kind][0]:
                continue
            per_kind = dict(b1)
            per_kind[kind] = [m]
            add(f"P_alt_single_{kind}_{m[:8]}", HSEED.route(per_kind))

    # -- источник 4 (запасной): обратный порядок уже найденных Imp-пар --
    # PAR(a,b) и PAR(b,a) покрывают одно и то же множество (Imp не меняется),
    # но структурно различны -- используется, только если 1-3 не хватило.
    for s, kind in enumerate(STEP_KINDS):
        if (len(out) - 3) >= 20:
            break
        for (m1, m2) in robust.get(s, []):
            if (len(out) - 3) >= 20:
                break
            per_kind = dict(b1)
            per_kind[kind] = [m2, m1]
            add(f"P_single_slot_{kind}_rev_{m1[:6]}_{m2[:6]}", HSEED.route(per_kind))

    assert (len(out) - 3) == 20, (
        f"lean seed construction produced only {len(out) - 3} distinct non-reference "
        f"genotypes (need 20) -- источников 1-4 не хватило, см. BLOCKERS.md")

    return {name: G.genotype(root, gen=0, origin=f"seed:{name}") for name, root in out.items()}


def assert_lean_population(seeds: dict, ds, panel: list, best_single: dict, registry) -> None:
    """PROTOCOL.md §5: два обязательных assert'а до generation 0. LIFE-4:
    `registry` обязателен для `ML.imp_set` (Change 1) -- seed-генотипы
    никогда не содержат cx.-CALL, так что разворот здесь не меняет
    результат, только держит сигнатуру согласованной с остальным кодом."""
    non_ref = {name: g for name, g in seeds.items() if name not in REFERENCE_NAMES}
    n_distinct = len({g["complex_id"] for g in non_ref.values()})
    assert n_distinct == 20, (
        f"lean assert #1 failed: {n_distinct} distinct non-reference complex_id, expected 20 "
        f"(names: {sorted(non_ref)})")

    violators = []
    for name, g in non_ref.items():
        imp = ML.imp_set(ds, panel, best_single, g, registry)
        if len(imp) >= 2:
            violators.append((name, g["complex_id"], sorted(imp)))
    assert not violators, (
        f"lean assert #2 failed: {len(violators)} non-reference seed(s) with |ImpSet|>=2: "
        f"{violators}")
