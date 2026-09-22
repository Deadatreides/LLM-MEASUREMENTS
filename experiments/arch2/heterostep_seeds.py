"""heterostep_seeds.py — стартовые генотипы популяции кампании a1 (пятый цикл).

Аналог `library.py` для полигона HETEROSTEP. `SWITCH("task_family", ...)` здесь
НЕПРИМЕНИМ: у HETEROSTEP один "family" и реестр не поддерживает SWITCH вовсе
(`heterostep.Registry.observe` бросает `NotImplementedError`, `observable_names()`
пуст, SPEC.md §33) — все затравки ASSEMBLE-корневые НАПРЯМУЮ, без обёртки-диспетчера.

Особенность, отличающая эталоны отсюда от `library.py`'s B0-B3: «слепой повтор»
структурно бессмыслен для этого полигона. Без ASSEMBLE у CALL+CHECK нет вида на
шаг задачи (`state.task["step"]` не установлен), и голый `SEQ(CALL, CHECK)` вне
ASSEMBLE получает ровно INAPPLICABLE, не частичный успех (`heterostep.py::
_seam_heterostep`). Поэтому вместо B0 (голый повтор) — ТРИ ПРЕДРЕГИСТРИРОВАННЫЕ
базовые линии плана ASSEMBLE (`C:\\Users\\user\\.claude\\plans\\cached-hopping-aho.md`
§4, `REPORT_ASSEMBLE.md` §2), не измерения задним числом:

  REF_B1_route        маршрут по перечислению (эксп. 14 §G0, оптимум глубины 1)
  REF_B2_greedy2       жадная ПАРА моделей на род (глубина ≤2)
  REF_B3_greedy_cover  жадное покрытие до нулевого прироста (потолок объединения)

B1/B2/B3 вычисляются ЗДЕСЬ из тех же ячеек train-сетки, что видит Runner —
не хардкод конкретных имён моделей, а функция от данных (см. `_greedy_order`).
Числа сверены с независимым Python-подсчётом ДО того, как ASSEMBLE был написан
(REPORT_ASSEMBLE.md §2: 0.4600 / 0.5800 / 0.7000 на train, до последнего знака).

B3 — де-факто ПЛАНКА для отбора: `evolve.py` вычисляет gate ДИНАМИЧЕСКИ как
сильнейший эталон НА ТЕКУЩЕМ СРЕЗЕ (§394-395 evolve.py), а r(B3) равен
теоретическому потолку (объединение всех 6 моделей, ceiling.py-стиль расчёт) —
популяция не может побить его по r СТРУКТУРНО, только по СТОИМОСТИ при том же r.
Это и есть операционализация V3 (план §5) существующей логикой extinction/D3
(`lower_e > 1.0`), без единой правки семантики отбора.
"""

from __future__ import annotations

import genotype as G
import heterostep as H

REFERENCE_NAMES = ("REF_B1_route", "REF_B2_greedy2", "REF_B3_greedy_cover")
BASELINE_NAME = "REF_B1_route"            # линейка стоимости (дешевле B2/B3 по построению)
GATE_REFERENCE_NAME = "REF_B3_greedy_cover"   # планка: потолок объединения


# -- строительные блоки ------------------------------------------------------------------


def _slot(model_id: str) -> dict:
    return G.SEQ(G.CALL(f"gen.{model_id}", prompt=H.PROMPT_VARIANT, temperature=H.TEMPERATURE,
                        seed_slot=0, max_tokens=H.MAX_TOKENS_STEP), G.CHECK(H.SEAM_ID))


def _fallback(model_ids) -> dict:
    """Один слот ASSEMBLE: одна модель напрямую, или PAR-откат при >1 модели --
    ленивый выход по первому PASS (runner._exec_par), а не подделка результата."""
    model_ids = list(model_ids)
    if len(model_ids) == 1:
        return _slot(model_ids[0])
    return G.PAR(*[_slot(m) for m in model_ids])


def route(per_kind: dict) -> dict:
    """ASSEMBLE над {род: [модели по приоритету отката]} — общий конструктор и для
    маршрута глубины 1 (списки длины 1), и для отката (списки длиной > 1)."""
    return G.ASSEMBLE(*[_fallback(per_kind[k]) for k in H.STEP_KINDS])


# -- маршруты, вычисленные из данных -------------------------------------------------------


def _greedy_order(ds: H.Dataset, task_ids: list, kind: str) -> list:
    """Порядок моделей по МАРГИНАЛЬНОМУ покрытию (жадное покрытие множества):
    первая — лучшая одиночная, дальше — та, что добавляет больше НОВЫХ решённых
    задач среди ещё не покрытых. Останавливается на нулевом приросте (план §1,
    таблица «жадная по покрытию» — 0.700, совпадает с объединением всех 6)."""
    covered: set = set()
    chosen: list = []
    rest = set(ds.models)

    def gain(m):
        return len({t for t in task_ids if ds.cells[(t, kind, m)]["status"] == "PASS"} - covered)

    while rest:
        best = max(rest, key=gain)
        g = gain(best)
        if g == 0 and chosen:
            break
        chosen.append(best)
        covered |= {t for t in task_ids if ds.cells[(t, kind, best)]["status"] == "PASS"}
        rest.discard(best)
    return chosen


def b1_route(ds: H.Dataset, task_ids: list) -> dict:
    """Депф-1: только лучшая одиночная модель на род — первый элемент жадного
    порядка. Тождественно маршруту перечисления эксп. 14 §G0 (план §1: ранг 1 из
    1296 по цепочке, доказано отдельно, не переоткрывается здесь)."""
    return {k: [_greedy_order(ds, task_ids, k)[0]] for k in H.STEP_KINDS}


def b2_route(ds: H.Dataset, task_ids: list) -> dict:
    """Жадная ПАРА (глубина ≤2) на род — жёсткий алгоритм-конкурент V3."""
    return {k: _greedy_order(ds, task_ids, k)[:2] for k in H.STEP_KINDS}


def b3_route(ds: H.Dataset, task_ids: list) -> dict:
    """Полное жадное покрытие (до нулевого прироста) — потолок объединения,
    планка отбора (V3)."""
    return {k: _greedy_order(ds, task_ids, k) for k in H.STEP_KINDS}


# -- сборка стартовой популяции -------------------------------------------------------------


def seed_complexes(ds: H.Dataset, task_ids: list) -> dict:
    """{имя: генотип}. Три эталона (REFERENCE_NAMES) + структурно разнородные
    стартовые точки популяции (не эталоны — конкурируют, умирают, размножаются)."""
    b1, b2, b3 = b1_route(ds, task_ids), b2_route(ds, task_ids), b3_route(ds, task_ids)
    models = list(ds.models)
    out: dict = {}

    # -- эталоны --
    out["REF_B1_route"] = route(b1)
    out["REF_B2_greedy2"] = route(b2)
    out["REF_B3_greedy_cover"] = route(b3)

    # -- слабый старт: одна модель на ВСЕ 4 слота, без отката (эксп. 14 плечо
    # B_STEP_SAME по духу: r~0.05 по всей цепочке на train) --
    out["P_single_model_x4"] = route({k: [models[0]] for k in H.STEP_KINDS})
    out["P_single_model_x4_alt"] = route({k: [models[-1]] for k in H.STEP_KINDS})

    # -- иной маршрут глубины 1: ВТОРОЙ по покрытию элемент жадного порядка на
    # каждый род (там, где он существует) -- структурно отличная от B1 точка,
    # не заведомо хуже (жадный порядок ранжирует по маргинальному, не абсолютному
    # вкладу; второй элемент способен обгонять первый по абсолютному покрытию) --
    out["P_second_best_route"] = route({
        k: [_greedy_order(ds, task_ids, k)[1] if len(_greedy_order(ds, task_ids, k)) > 1
            else _greedy_order(ds, task_ids, k)[0]]
        for k in H.STEP_KINDS
    })

    # -- частичный откат: B1 с расширением только ОДНОГО слота -- полшага в
    # сторону B2, естественная стартовая точка для M3 (add_par_branch) продолжить --
    out["P_par_lookup_only"] = route({**b1, "LOOKUP": b2["LOOKUP"]})
    out["P_par_compute_only"] = route({**b1, "COMPUTE": b2["COMPUTE"]})

    return {name: G.genotype(root, gen=0, origin=f"seed:{name}") for name, root in out.items()}
