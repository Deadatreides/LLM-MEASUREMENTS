"""lean_seeds.py — PROTOCOL.md §4: бедный старт. `M.random_genotype` НЕ
используется здесь (разведка подтвердила: он структурно не может
произвести валидный ASSEMBLE на реестре HETEROSTEP -- его `leaf()` строит
`library.attempt(...)` с `max_tokens∈{150,320}`/швами
`seam.multistep_arith`/`seam.enriched_code`, которых реестр HETEROSTEP не
знает; `heterostep.Registry.observe` вдобавок безусловно бросает
`NotImplementedError`). Все генотипы строятся ЯВНО из
`heterostep_seeds.route`/`._slot`/`._fallback` (чистые конструкторы дерева).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402

HSTEP = LD.HSTEP
HSEED = LD.HSEED
G = LD.G

STEP_KINDS = HSTEP.STEP_KINDS
REFERENCE_NAMES = HSEED.REFERENCE_NAMES

# -- панель-устойчивые однослотовые Imp-пары (разведка, PROTOCOL.md §3) --
# индекс слота -> список 2-моделных пар, каждая даёт Imp(slot)=1 на любой
# из 5 панелей LIFE-1. FORMAT (индекс 1) отсутствует намеренно -- Imp(FORMAT)
# структурно недостижим (один атом покрывает 80/80).
_ROBUST_PAIRS = {
    0: [  # READ
        ("gemma-3-it-1b-q5_k_s", "internvl3-2b-q4_k_m"),
        ("internvl3-2b-q4_k_m", "llama-3.2-1b-instruct-q4_0"),
        ("internvl3-2b-q4_k_m", "qwen3-1.7b-q4_0-unsloth"),
        ("internvl3-2b-q4_k_m", "qwen2.5-coder-1.5b-instruct-q4_0"),
        ("qwen2.5-coder-1.5b-instruct-q4_0", "qwen3-1.7b-q4_0-unsloth"),
    ],
    2: [  # LOOKUP
        ("gemma-3-it-1b-q5_k_s", "qwen3-1.7b-q4_0-unsloth"),
        ("qwen3-1.7b-q4_0-unsloth", "smollm2-1.7b-instruct-q4_k_m"),
    ],
    3: [  # COMPUTE
        ("gemma-3-it-1b-q5_k_s", "internvl3-2b-q4_k_m"),
        ("gemma-3-it-1b-q5_k_s", "qwen2.5-coder-1.5b-instruct-q4_0"),
        ("internvl3-2b-q4_k_m", "qwen2.5-coder-1.5b-instruct-q4_0"),
        ("llama-3.2-1b-instruct-q4_0", "qwen2.5-coder-1.5b-instruct-q4_0"),
        ("qwen2.5-coder-1.5b-instruct-q4_0", "qwen3-1.7b-q4_0-unsloth"),
        ("qwen2.5-coder-1.5b-instruct-q4_0", "smollm2-1.7b-instruct-q4_k_m"),
    ],
}


def _gain(ds, panel, kind, model, covered):
    return len({t for t in panel if ds.cells[(t, kind, model)]["status"] == "PASS"} - covered)


def greedy_order_deterministic(ds, panel: list, kind: str) -> list:
    """Копия `agent_life1_mechanism/src/deterministic_route.py::
    greedy_order_deterministic` -- тай-брейк алфавитный по model_id, не
    хэш-порядок-зависимый `set()`+`max()` (`heterostep_seeds._greedy_order`,
    не используется в этом пакете)."""
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


def build_seeds_pop20_lean(ds, panel: list) -> dict:
    """-> {name: genotype}. 3 эталона + 1 нулевой (`b1`) + 13 однослотовых
    (панель-устойчивые пары §3) + 6 заполнителей без блоков. PROTOCOL.md §4
    требует ровно 20 РАЗЛИЧНЫХ complex_id среди не-эталонных -- проверяется
    `assert_lean_population` ниже, не здесь."""
    routes = build_b1_b2_b3(ds, panel)
    b1, b2, b3, orders = routes["b1"], routes["b2"], routes["b3"], routes["orders"]
    models = list(ds.models)

    out: dict = {}
    out["REF_B1_route"] = HSEED.route(b1)
    out["REF_B2_greedy2"] = HSEED.route(b2)
    out["REF_B3_greedy_cover"] = HSEED.route(b3)

    # -- 1 нулевой: чистый b1 (лучшая одиночная модель на каждый слот, ImpSet=0) --
    out["P_zero_b1"] = HSEED.route(b1)

    # -- 13 однослотовых: b1 с ОДНИМ слотом заменённым на комплементарную пару --
    for s, pairs in _ROBUST_PAIRS.items():
        kind = STEP_KINDS[s]
        for i, pair in enumerate(pairs):
            name = f"P_single_slot_{kind}_{i}"
            per_kind = dict(b1)
            per_kind[kind] = list(pair)
            out[name] = HSEED.route(per_kind)

    # -- 6 заполнителей без блоков: один атом на ВСЕ 4 слота (ImpSet=0 по построению) --
    out["P_single_model_first"] = HSEED.route({k: [models[0]] for k in STEP_KINDS})
    out["P_single_model_last"] = HSEED.route({k: [models[-1]] for k in STEP_KINDS})
    for idx in range(1, min(4, len(models) - 1)):
        out[f"P_single_model_{idx}"] = HSEED.route({k: [models[idx]] for k in STEP_KINDS})
    # добор до ровно 6 заполнителей, если моделей меньше 6 (не должно случиться здесь)
    fill_idx = min(4, len(models) - 1)
    while sum(1 for n in out if n.startswith("P_single_model_")) < 6 and fill_idx < len(models):
        out[f"P_single_model_extra_{fill_idx}"] = HSEED.route({k: [models[fill_idx]] for k in STEP_KINDS})
        fill_idx += 1

    return {name: G.genotype(root, gen=0, origin=f"seed:{name}") for name, root in out.items()}


def assert_lean_population(seeds: dict, ds, panel: list, best_single: dict) -> None:
    """PROTOCOL.md §4: два обязательных assert'а до generation 0. Бросает
    AssertionError с диагностикой -- вызывающая сторона (`orchestrator.py`)
    ловит и пишет в BLOCKERS.md, evolve не запускается."""
    import metrics_lib as ML   # noqa: E402  (локальный импорт -- избежать цикла на уровне модуля)

    non_ref = {name: g for name, g in seeds.items() if name not in REFERENCE_NAMES}
    n_distinct = len({g["complex_id"] for g in non_ref.values()})
    assert n_distinct == 20, (
        f"lean assert #1 failed: {n_distinct} distinct non-reference complex_id, expected 20 "
        f"(names: {sorted(non_ref)})")

    violators = []
    for name, g in non_ref.items():
        imp = ML.imp_set(ds, panel, best_single, g)
        if len(imp) >= 2:
            violators.append((name, g["complex_id"], sorted(imp)))
    assert not violators, (
        f"lean assert #2 failed: {len(violators)} non-reference seed(s) with |ImpSet|>=2: "
        f"{violators}")
