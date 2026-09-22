"""mutate.py — операторы M1-M6 (SPEC.md §9). Все — операции над ДАННЫМИ.

Каждый мутант проходит валидатор ДО любого исполнения; невалидный отбрасывается
бесплатно. Это главный экономический фильтр слоя, поэтому мутации намеренно
«наивные»: дешевле сгенерировать десяток кандидатов и отсеять, чем усложнять оператор.

Вероятности операторов зафиксированы здесь ДО прогона и по ходу не подстраиваются
(норма проекта: пороги не смягчаются задним числом).
"""

from __future__ import annotations

import copy
import random
from typing import Callable, Optional

import genotype as G

# SPEC.md §12: произвольные константы, зафиксированы до прогона.
OPERATOR_WEIGHTS = {"M1": 0.30, "M2": 0.20, "M3": 0.20, "M4": 0.15, "M5": 0.10, "M6": 0.05}
MAX_TRIES = 40

COMPOSITE_PREFIX = "cx."


def is_composite_id(mid) -> bool:
    return isinstance(mid, str) and mid.startswith(COMPOSITE_PREFIX)


def _any_plain_params(root: dict) -> Optional[dict]:
    """Параметры любого НЕкомпозитного CALL в этом же генотипе.

    Нужны, когда композит заменяется обычной моделью: у композита сетка пуста, а
    выдумывать max_tokens нельзя — он зависит от семейства задачи и обязан совпадать
    с тем, что стоит в соседних ветвях.
    """
    for _, n in addresses(root):
        if n.get("op") == "CALL" and not is_composite_id(n.get("molecule")) and n.get("params"):
            return dict(n["params"])
    return None


# -- адресация узлов ---------------------------------------------------------------------


def addresses(node: dict, addr: Optional[list] = None) -> list:
    """[(address, node)] — адрес это список шагов вида ('children', i) / ('cases', key)
    / ('default',) / ('child',). Нужен, чтобы заменить узел на месте."""
    addr = addr or []
    out = [(list(addr), node)]
    op = node.get("op")
    if op in ("SEQ", "PAR", "ASSEMBLE"):
        # ASSEMBLE в одной ветке с SEQ/PAR НАМЕРЕННО (SPEC.md §31: форма поля
        # идентична): все шесть операторов M1-M6 видят поддеревья слотов ASSEMBLE
        # без единой правки самих операторов — только через это членство.
        for i, ch in enumerate(node["children"]):
            out += addresses(ch, addr + [("children", i)])
    elif op == "SWITCH":
        for key in sorted(node["cases"]):
            out += addresses(node["cases"][key], addr + [("cases", key)])
        out += addresses(node["default"], addr + [("default",)])
    elif op == "BUDGET":
        out += addresses(node["child"], addr + [("child",)])
    return out


def get_at(root: dict, address: list) -> dict:
    node = root
    for step in address:
        node = node[step[0]][step[1]] if len(step) == 2 else node[step[0]]
    return node


def set_at(root: dict, address: list, new_node: dict) -> dict:
    if not address:
        return new_node
    root = copy.deepcopy(root)
    node = root
    for step in address[:-1]:
        node = node[step[0]][step[1]] if len(step) == 2 else node[step[0]]
    last = address[-1]
    if len(last) == 2:
        node[last[0]][last[1]] = new_node
    else:
        node[last[0]] = new_node
    return root


def _nodes_of(root: dict, op: str) -> list:
    return [(a, n) for a, n in addresses(root) if n.get("op") == op]


# -- операторы ------------------------------------------------------------------------------


def m1_swap_molecule(root: dict, rng: random.Random, ctx: dict) -> Optional[dict]:
    """Заменить генератор на другой того же интерфейса.

    Ось «семейство модели» — сильнейшая из измеренных (F6/F8, +9.9 п.п. при равном
    бюджете), поэтому M1 — самый вероятный оператор."""
    calls = _nodes_of(root, "CALL")
    if not calls:
        return None
    addr, node = rng.choice(calls)
    options = [m for m in ctx["generators"] if m != node["molecule"]]
    if not options:
        return None
    new_mid = rng.choice(options)

    # У композита сетка параметров ПУСТА, у обычной модели — обязательна. Раньше M1
    # просто переносил старые params на новую молекулу; при участии композитов такой
    # мутант всегда невалиден, и вставка композита молча не работала (SPEC.md §13.3).
    if is_composite_id(new_mid):
        params = {}
    elif is_composite_id(node.get("molecule")):
        params = _any_plain_params(root)
        if params is None:
            return None
    else:
        params = dict(node.get("params") or {})
    return set_at(root, addr, {"op": "CALL", "molecule": new_mid, "params": params})


def m2_shift_param(root: dict, rng: random.Random, ctx: dict) -> Optional[dict]:
    """Сдвинуть скалярный параметр по объявленной сетке (или лимит BUDGET)."""
    calls = [(a, n) for a, n in _nodes_of(root, "CALL")
             if n["params"] and not is_composite_id(n.get("molecule"))]
    budgets = _nodes_of(root, "BUDGET")
    choices = [("call", a, n) for a, n in calls] + [("budget", a, n) for a, n in budgets]
    if not choices:
        return None
    kind, addr, node = rng.choice(choices)
    if kind == "budget":
        step = rng.choice([0.5, 0.75, 1.5, 2.0])
        new_limit = max(50, int(node["limit_tokens"] * step))
        return set_at(root, addr, dict(node, limit_tokens=new_limit))

    grid = ctx["feasible_params"]
    key = rng.choice([k for k in node["params"] if len(grid.get(k, ())) > 1] or list(node["params"]))
    options = [v for v in grid.get(key, ()) if v != node["params"][key]]
    if not options:
        return None
    params = dict(node["params"])
    params[key] = rng.choice(options)
    return set_at(root, addr, dict(node, params=params))


def m3_structural(root: dict, rng: random.Random, ctx: dict) -> Optional[dict]:
    """Вставить/удалить узел: обернуть в BUDGET, добавить/убрать ветвь PAR,
    заменить поддерево на STOP."""
    what = rng.choice(["wrap_budget", "add_par_branch", "drop_par_branch", "to_stop", "wrap_par"])
    addrs = addresses(root)

    if what == "wrap_budget":
        addr, node = rng.choice(addrs)
        if node.get("op") == "BUDGET":
            return None
        return set_at(root, addr, G.BUDGET(rng.choice([300, 500, 700, 1000]), copy.deepcopy(node)))

    if what == "wrap_par":
        addr, node = rng.choice(addrs)
        if node.get("op") in ("STOP", "CHECK"):
            return None
        return set_at(root, addr, G.PAR(copy.deepcopy(node), copy.deepcopy(node)))

    pars = _nodes_of(root, "PAR")
    if what in ("add_par_branch", "drop_par_branch"):
        if not pars:
            return None
        addr, node = rng.choice(pars)
        children = list(node["children"])
        if what == "add_par_branch":
            children.append(copy.deepcopy(rng.choice(children)))
        else:
            if len(children) < 2:
                return None
            children.pop(rng.randrange(len(children)))
        return set_at(root, addr, dict(node, children=children))

    # to_stop: превратить поддерево в отказ тратить. Прямое следствие E4.3:
    # на высокобарьерных подтипах предельная цена в 13 раз выше.
    inner = [(a, n) for a, n in addrs if a and n.get("op") != "STOP"]
    if not inner:
        return None
    addr, _ = rng.choice(inner)
    return set_at(root, addr, G.STOP("barrier_too_high"))


def m4_switch(root: dict, rng: random.Random, ctx: dict) -> Optional[dict]:
    """Расщепить узел по наблюдаемой / добавить / убрать ветвь SWITCH."""
    switches = _nodes_of(root, "SWITCH")
    what = rng.choice(["refine", "add_case", "drop_case"]) if switches else "refine"

    if what == "refine":
        addrs = [(a, n) for a, n in addresses(root) if n.get("op") != "CHECK"]
        addr, node = rng.choice(addrs)
        obs = rng.choice(ctx["observables"])
        values = list(G.OBSERVABLES[obs])
        value = rng.choice(values)
        alt = rng.choice([G.STOP("barrier_too_high"), copy.deepcopy(node)])
        return set_at(root, addr, G.SWITCH(obs, {value: copy.deepcopy(node)}, alt))

    addr, node = rng.choice(switches)
    obs = node["obs"]
    if what == "add_case":
        free = [v for v in G.OBSERVABLES[obs] if v not in node["cases"]]
        if not free:
            return None
        cases = dict(node["cases"])
        cases[rng.choice(free)] = rng.choice([G.STOP("barrier_too_high"),
                                              copy.deepcopy(node["default"])])
        return set_at(root, addr, dict(node, cases=cases))

    if not node["cases"]:
        return None
    cases = dict(node["cases"])
    cases.pop(rng.choice(list(cases)))
    return set_at(root, addr, dict(node, cases=cases))


def m5_horizontal_transfer(root: dict, rng: random.Random, ctx: dict) -> Optional[dict]:
    """Перенос поддерева от донора (живого ИЛИ архивного).

    Единственный способ, которым мёртвая линия влияет на будущее: генотип в архиве
    навсегда, но ни субсидии, ни места в популяции донор не получает (SPEC.md §9.2)."""
    donors = ctx.get("donors") or []
    if not donors:
        return None

    # Донор выбирается не равномерно, а по donor_score = 0.5*skel + 0.5*(1 - sem):
    # структурно совместимый, но фенотипически ДОПОЛНЯЮЩИЙ (SPEC.md §15). Прививать
    # подграф от донора, решающего ровно то же самое, — это и есть избыточность.
    score_fn = ctx.get("donor_score_fn")
    if score_fn is not None:
        weights = []
        for d in donors:
            try:
                weights.append(max(1e-6, float(score_fn(d))))
            except Exception:                                  # noqa: BLE001
                weights.append(1e-6)
        donor = rng.choices(donors, weights=weights, k=1)[0]
    else:
        donor = rng.choice(donors)

    donor_nodes = [n for a, n in addresses(donor["root"]) if n.get("op") != "CHECK"]
    if not donor_nodes:
        return None
    graft = copy.deepcopy(rng.choice(donor_nodes))
    addr, _ = rng.choice(addresses(root))
    ctx.setdefault("report", {})["donor_id"] = donor.get("complex_id")
    return set_at(root, addr, graft)


def m6_compress(root: dict, rng: random.Random, ctx: dict) -> Optional[dict]:
    """Заменить поддерево ссылкой на ЗАРЕГИСТРИРОВАННЫЙ комплекс-молекулу.

    Работает только когда автокатализ уже случился (§10); до первой регистрации
    оператор не срабатывает — это ожидаемо, а не сбой."""
    composites = ctx.get("composites") or []
    if not composites:
        return None
    addr, node = rng.choice([(a, n) for a, n in addresses(root) if n.get("op") != "CHECK"])
    mid = rng.choice(composites)
    ctx.setdefault("report", {})["composite_id"] = mid
    return set_at(root, addr, {"op": "CALL", "molecule": mid, "params": {}})


OPERATORS: dict[str, Callable] = {
    "M1": m1_swap_molecule, "M2": m2_shift_param, "M3": m3_structural,
    "M4": m4_switch, "M5": m5_horizontal_transfer, "M6": m6_compress,
}


# -- точка входа -----------------------------------------------------------------------------


def operator_weights(ctx: dict) -> dict:
    """Веса операторов. Единственная зависимость от состояния — наличие композитов:
    пока их нет, M6 не может сработать в принципе, и его вес — мёртвый груз. Значение
    M6_WEIGHT_WITH_COMPOSITES объявлено в SPEC.md §18 ДО прогона."""
    w = dict(OPERATOR_WEIGHTS)
    if ctx.get("composites"):
        import heredity

        w["M6"] = heredity.M6_WEIGHT_WITH_COMPOSITES
    return w


def mutate(parent: dict, rng: random.Random, ctx: dict, registry, gen: int):
    """-> (child, report) либо (None, report). report = {"operator", "donor_id", ...}.

    Отчёт нужен слою наследственности: без него нельзя эмитить `crossed_from`
    (какой донор дал подграф) и `instantiated_from` (какая молекула вставлена) —
    родство восстановить постфактум невозможно (SPEC.md §14).

    ctx: {"generators", "observables", "feasible_params", "donors", "composites",
          "donor_score_fn"(опц.)}
    """
    weights_map = operator_weights(ctx)
    names = list(weights_map)
    weights = [weights_map[n] for n in names]
    for _ in range(MAX_TRIES):
        op_name = rng.choices(names, weights=weights, k=1)[0]
        ctx["report"] = {"operator": op_name}
        try:
            new_root = OPERATORS[op_name](copy.deepcopy(parent["root"]), rng, ctx)
        except Exception:                                   # noqa: BLE001
            new_root = None
        if new_root is None:
            continue
        child = G.genotype(new_root, gen=gen, parent_ids=[parent["complex_id"]],
                           origin=f"mutate:{op_name}")
        if G.is_valid(child, registry) and child["complex_id"] != parent["complex_id"]:
            return child, dict(ctx["report"])
    return None, {"operator": None}


def random_genotype(rng: random.Random, ctx: dict, registry, gen: int, depth: int = 0):
    """-> (генотип, report). Свежий случайный генотип «с нуля» — гарантирует ненулевую
    меру исследования даже если популяция сойдётся (SPEC.md §6, защита от вырождения)."""
    for _ in range(MAX_TRIES):
        ctx["report"] = {"operator": "random"}
        try:
            root = _random_node(rng, ctx, budget_depth=2)
        except (IndexError, KeyError):
            # Пустой `ctx["observables"]` (SPEC.md §33: реестры без наблюдаемых,
            # например `heterostep.Registry`) делает ветку "switch" внутри
            # `_random_node` невыполнимой -- тот же исход, что и провал одного из
            # MAX_TRIES попыток, а не крах кампании. Для существующих реестров
            # (`observables` всегда непустой) это ветвление никогда не срабатывает.
            continue
        child = G.genotype(root, gen=gen, parent_ids=[], origin="random")
        if G.is_valid(child, registry):
            return child, dict(ctx["report"])
    return None, {"operator": "random"}


def _random_node(rng: random.Random, ctx: dict, budget_depth: int) -> dict:
    import heredity
    import library

    prompts = list(ctx["feasible_params"]["prompt"])
    slots = list(ctx["feasible_params"]["seed_slot"])
    plain = [m for m in ctx["generators"] if not is_composite_id(m)]
    composites = ctx.get("composites") or []

    def leaf():
        # Третий путь вставки композита (SPEC.md §17.3), помимо M1 и M6: свежий
        # случайный генотип может взять уже зарегистрированную молекулу как лист.
        if composites and rng.random() < heredity.P_COMPOSITE_LEAF:
            mid = rng.choice(composites)
            ctx.setdefault("report", {})["composite_id"] = mid
            return library.composite_attempt(mid)
        return library.attempt(rng.choice(prompts), rng.choice(plain), rng.choice(slots))

    if budget_depth <= 0:
        return leaf()

    # ASSEMBLE участвует в случайной генерации ТОЛЬКО когда ctx явно объявляет число
    # шагов задачи через `assemble_n_steps` (SPEC.md §31/§33). Без этого флага список
    # и веса побайтово прежние -- ни новой ветки, ни изменения вероятностей для
    # существующих реестров MSARITH/кода, которые этот флаг никогда не ставят (иначе
    # 25 случаев `test_random_genotypes_are_valid` могли бы начать видеть ASSEMBLE
    # против пустого HETEROSTEP-реестра и валиться на unknown_molecule).
    n_steps = ctx.get("assemble_n_steps")
    ops = ["leaf", "par", "switch", "budget", "stop"]
    weights = [0.40, 0.25, 0.20, 0.10, 0.05]
    if n_steps:
        ops = ops + ["assemble"]
        weights = weights + [0.15]           # вес не перенормирует остальные (SPEC.md §33)

    choice = rng.choices(ops, weights=weights, k=1)[0]
    if choice == "leaf":
        return leaf()
    if choice == "stop":
        return G.STOP("barrier_too_high")
    if choice == "par":
        k = rng.randint(2, 3)
        return G.PAR(*[_random_node(rng, ctx, budget_depth - 1) for _ in range(k)])
    if choice == "budget":
        return G.BUDGET(rng.choice([300, 500, 700, 1000]), _random_node(rng, ctx, budget_depth - 1))
    if choice == "assemble":
        # Слоты стартуют ПЛОСКИМИ (глубина 1, как маршрут эксп. 14 -- B1 из плана).
        # Откат внутри слота ищет МУТАЦИЯ (M3 wrap_par/add_par_branch, уже
        # существующие операторы, применимые благодаря членству ASSEMBLE в
        # `addresses()`) -- ни одного нового оператора мутации не понадобилось.
        return G.ASSEMBLE(*[leaf() for _ in range(n_steps)])
    obs = rng.choice(ctx["observables"])
    value = rng.choice(list(G.OBSERVABLES[obs]))
    return G.SWITCH(obs, {value: _random_node(rng, ctx, budget_depth - 1)},
                    _random_node(rng, ctx, budget_depth - 1))
