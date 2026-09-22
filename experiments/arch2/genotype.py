"""genotype.py — Complex Layer, уровень 1: генотип как ДАННЫЕ.

Генотип комплекса — сериализуемое дерево из восьми узлов (SPEC.md §1.1, §30) над
ссылками на молекулы. Никакого императивного кода, никаких побочных эффектов: этот
модуль только описывает, канонизирует и ПРОВЕРЯЕТ структуру. Исполняет `runner.py`.

Восьмой узел, ASSEMBLE, добавлен в пятом цикле (SPEC.md §30-34) поверх честно
пересмотренного запрета §10/§20/§29 — с измеренным основанием (эксп. 14, G1 = +0.3800,
CI [+0.27,+0.49]): язык из семи узлов не мог выразить «шаг i от модели X, шаг j от
модели Y», и маршрут в эксп. 14 подбирался перечислением, а не отбором.

Валидатор здесь — главный экономический фильтр всего слоя: невалидный мутант должен
отсеиваться до единого вызова модели, потому что дефицитный ресурс — GPU, а не CPU.

Зависимости: реестр молекул передаётся СВЕРХУ как duck-typed объект (инверсия
зависимостей, как `graph_view` в arch1/seam_engine.py) — модуль не импортирует
`registry.py` на уровне модуля и потому тестируется на игрушечных реестрах.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Iterator, Optional

SPEC_VERSION = 1

# -- пределы структуры (SPEC.md §12: произвольные, зафиксированы до прогона) ----------
MAX_DEPTH = 6
MAX_NODES = 40

# Верхняя оценка длины промпта, когда реестр не сообщает свою. Обоснование величины:
# фактические `input_tokens` в experiment12/runs12/phase0_full_exp12-v1.json лежат в
# диапазоне ~140-420; 800 — заведомо консервативная граница, используемая ТОЛЬКО для
# статической оценки сверху, не для учёта стоимости (там всегда фактические токены).
DEFAULT_MAX_INPUT_TOKENS = 800

OPS = ("CALL", "CHECK", "SEQ", "PAR", "SWITCH", "BUDGET", "STOP", "ASSEMBLE")

# Закрытый список наблюдаемых (SPEC.md §1.2). Расширять только правкой СПЕКИ.
OBSERVABLES: dict[str, tuple[str, ...]] = {
    "last_status": ("PASS", "FAIL", "INAPPLICABLE", "ERROR", "NONE"),
    "collision_subtype": ("syntax", "contract_interface", "dependency", "semantic", "value", "other"),
    "same_failure": ("SAME", "DIFFERENT", "NA"),
    "attempts_used": ("0", "1", "2", "3+"),
    "tokens_bucket": ("0", "low", "mid", "high"),
    "task_family": ("arithmetic", "code"),
    "dominant_share_bucket": ("low", "high", "NA"),
    "n_unique_signatures": ("1", "2", "3+", "NA"),
}

# Поля, не участвующие в структурной тождественности комплекса (SPEC.md §7.2).
_PROVENANCE_FIELDS = ("complex_id", "gen", "parent_ids", "origin", "born_at")


# -- конструкторы (удобство, не обязательны: генотип — обычный dict/JSON) -------------


def CALL(molecule: str, **params) -> dict:
    return {"op": "CALL", "molecule": molecule, "params": dict(params)}


def CHECK(seam: str) -> dict:
    return {"op": "CHECK", "seam": seam}


def SEQ(*children: dict) -> dict:
    return {"op": "SEQ", "children": list(children)}


def PAR(*children: dict) -> dict:
    return {"op": "PAR", "children": list(children)}


def SWITCH(obs: str, cases: dict, default: dict) -> dict:
    return {"op": "SWITCH", "obs": obs, "cases": dict(cases), "default": default}


def BUDGET(limit_tokens: int, child: dict) -> dict:
    return {"op": "BUDGET", "limit_tokens": int(limit_tokens), "child": child}


def STOP(reason: str) -> dict:
    return {"op": "STOP", "reason": reason}


def ASSEMBLE(*children: dict) -> dict:
    """SPEC.md §31. Дочерний узел `i` -- слот, производящий значение шага `i` задачи;
    исход RESOLVED только если подтверждены ВСЕ слоты. Форма поля идентична SEQ/PAR
    (список под `children`) НАМЕРЕННО: `mutate.addresses`, `children_of`, `walk`,
    `cost_upper_bound` и валидатор трактуют ASSEMBLE как SEQ/PAR всюду, где расходится
    только СЕМАНТИКА исполнения (`runner._exec_assemble`), а не форма дерева."""
    return {"op": "ASSEMBLE", "children": list(children)}


def genotype(root: dict, *, gen: int = 0, parent_ids: Optional[list] = None,
             origin: str = "seed") -> dict:
    """Собирает полный генотип и проставляет `complex_id` по структуре."""
    g = {
        "complex_id": None,
        "spec_version": SPEC_VERSION,
        "gen": int(gen),
        "parent_ids": list(parent_ids or []),
        "origin": origin,
        "root": root,
    }
    g["complex_id"] = complex_id(g)
    return g


# -- канонизация и идентичность --------------------------------------------------------


def canonical_json(g: dict) -> str:
    """Каноническая форма СТРУКТУРЫ: происхождение выброшено, ключи отсортированы.

    Два генотипа с разной историей, но одинаковым деревом, дают одну строку — и,
    значит, один `complex_id`. Дубликаты в популяции схлопываются бесплатно, до GPU.
    """
    body = {k: v for k, v in g.items() if k not in _PROVENANCE_FIELDS}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def complex_id(g: dict) -> str:
    return "cx-" + hashlib.sha256(canonical_json(g).encode("utf-8")).hexdigest()[:6]


# -- обход ------------------------------------------------------------------------------


def children_of(node: dict) -> list:
    op = node.get("op")
    if op in ("SEQ", "PAR", "ASSEMBLE"):
        return list(node.get("children") or [])
    if op == "SWITCH":
        return list((node.get("cases") or {}).values()) + ([node["default"]] if "default" in node else [])
    if op == "BUDGET":
        return [node["child"]] if "child" in node else []
    return []


def walk(node: dict, path: str = "root") -> Iterator[tuple[str, dict]]:
    """(node_path, node) в порядке обхода сверху вниз. node_path попадает в трассу."""
    yield path, node
    op = node.get("op")
    if op in ("SEQ", "PAR", "ASSEMBLE"):
        for i, ch in enumerate(node.get("children") or []):
            yield from walk(ch, f"{path}/{op.lower()}[{i}]")
    elif op == "SWITCH":
        for key in sorted((node.get("cases") or {}).keys()):
            yield from walk(node["cases"][key], f"{path}/case[{key}]")
        if "default" in node:
            yield from walk(node["default"], f"{path}/default")
    elif op == "BUDGET":
        if "child" in node:
            yield from walk(node["child"], f"{path}/budget")


def depth(node: dict) -> int:
    kids = children_of(node)
    return 1 if not kids else 1 + max(depth(k) for k in kids)


def n_nodes(node: dict) -> int:
    return 1 + sum(n_nodes(k) for k in children_of(node))


def molecules_used(g: dict) -> set:
    out = set()
    for _, node in walk(g["root"]):
        if node.get("op") == "CALL":
            out.add(node.get("molecule"))
        elif node.get("op") == "CHECK":
            out.add(node.get("seam"))
    return out


# -- верхняя оценка стоимости ------------------------------------------------------------


def cost_upper_bound(node: dict, registry=None) -> int:
    """Статическая верхняя граница токенов. Используется валидатором и BUDGET-узлом.

    SEQ/PAR/ASSEMBLE — сумма (худший случай: у PAR ни одна ветвь не дала PASS,
    исполнены все; у ASSEMBLE все слоты ВСЕГДА исполняются, так что сумма — не только
    худший, но и типичный случай);
    SWITCH — максимум по ветвям (исполняется ровно одна);
    BUDGET — min(лимит, оценка потомка).
    """
    op = node.get("op")
    if op == "CALL":
        max_in = DEFAULT_MAX_INPUT_TOKENS
        if registry is not None and hasattr(registry, "max_input_tokens"):
            max_in = registry.max_input_tokens(node.get("molecule"))
        return int(max_in) + int((node.get("params") or {}).get("max_tokens", 0))
    if op in ("CHECK", "STOP"):
        return 0
    if op in ("SEQ", "PAR", "ASSEMBLE"):
        return sum(cost_upper_bound(c, registry) for c in node.get("children") or [])
    if op == "SWITCH":
        kids = children_of(node)
        return max((cost_upper_bound(c, registry) for c in kids), default=0)
    if op == "BUDGET":
        inner = cost_upper_bound(node["child"], registry) if "child" in node else 0
        return min(int(node.get("limit_tokens", 0)), inner)
    return 0


# -- валидатор ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "CALL": {"op", "molecule", "params"},
    "CHECK": {"op", "seam"},
    "SEQ": {"op", "children"},
    "PAR": {"op", "children"},
    "SWITCH": {"op", "obs", "cases", "default"},
    "BUDGET": {"op", "limit_tokens", "child"},
    "STOP": {"op", "reason"},
    "ASSEMBLE": {"op", "children"},
}


class _NullRegistry:
    """Реестр-заглушка: пропускает любые id. Только для тестов структуры."""

    def has_molecule(self, mid): return True
    def kind_of(self, mid): return "generator"
    def params_grid(self, mid): return None
    def max_input_tokens(self, mid): return DEFAULT_MAX_INPUT_TOKENS


def validate(g: dict, registry=None) -> list:
    """-> список нарушений (пустой = валиден). Никогда не бросает на кривом входе."""
    reg = registry if registry is not None else _NullRegistry()
    v: list = []

    if not isinstance(g, dict):
        return ["genotype_not_a_dict"]
    if g.get("spec_version") != SPEC_VERSION:
        v.append(f"spec_version:expected {SPEC_VERSION}, got {g.get('spec_version')}")
    root = g.get("root")
    if not isinstance(root, dict):
        return v + ["root_missing_or_not_a_dict"]

    if g.get("complex_id") not in (None, complex_id(g)):
        v.append(f"complex_id_mismatch:{g.get('complex_id')} != {complex_id(g)}")

    try:
        d, n = depth(root), n_nodes(root)
    except Exception as exc:                                    # noqa: BLE001
        return v + [f"malformed_tree:{type(exc).__name__}: {exc}"]
    if d > MAX_DEPTH:
        v.append(f"depth:{d} > {MAX_DEPTH}")
    if n > MAX_NODES:
        v.append(f"n_nodes:{n} > {MAX_NODES}")

    for path, node in walk(root):
        v.extend(_validate_node(path, node, reg))

    return v


def _validate_node(path: str, node: dict, reg) -> list:
    v: list = []
    if not isinstance(node, dict):
        return [f"{path}:node_not_a_dict"]
    op = node.get("op")
    if op not in OPS:
        return [f"{path}:unknown_op:{op!r}"]

    missing = _REQUIRED_FIELDS[op] - set(node)
    if missing:
        v.append(f"{path}:missing_fields:{sorted(missing)}")
    extra = set(node) - _REQUIRED_FIELDS[op]
    if extra:
        v.append(f"{path}:unexpected_fields:{sorted(extra)}")
    if missing:
        return v  # дальше проверять нечего

    if op == "CALL":
        mid = node["molecule"]
        if not reg.has_molecule(mid):
            v.append(f"{path}:unknown_molecule:{mid}")
        elif reg.kind_of(mid) != "generator":
            v.append(f"{path}:CALL_requires_generator, got {reg.kind_of(mid)}:{mid}")
        else:
            v.extend(_validate_params(path, mid, node["params"], reg))

    elif op == "CHECK":
        sid = node["seam"]
        if not reg.has_molecule(sid):
            v.append(f"{path}:unknown_seam:{sid}")
        elif reg.kind_of(sid) != "seam":
            v.append(f"{path}:CHECK_requires_seam, got {reg.kind_of(sid)}:{sid}")

    elif op in ("SEQ", "PAR", "ASSEMBLE"):
        # ASSEMBLE намеренно в одной ветке с PAR: непроверка "unreachable after stop"
        # применяется только к SEQ (op == "SEQ" ниже) — тот же выбор, что уже сделан
        # для PAR, и по той же причине: слоты ASSEMBLE не образуют линейный поток
        # управления, к которому относится понятие "недостижимый код".
        kids = node["children"]
        if not isinstance(kids, list) or not kids:
            v.append(f"{path}:{op}_children_must_be_nonempty_list")
        elif op == "SEQ":
            for i, ch in enumerate(kids[:-1]):
                if isinstance(ch, dict) and ch.get("op") == "STOP":
                    v.append(f"{path}:unreachable_after_stop_at_child_{i}")

    elif op == "SWITCH":
        obs = node["obs"]
        if obs not in OBSERVABLES:
            v.append(f"{path}:unknown_obs:{obs}")
        else:
            allowed = set(OBSERVABLES[obs])
            bad = set((node["cases"] or {}).keys()) - allowed
            if bad:
                v.append(f"{path}:unknown_case_values:{sorted(bad)} for obs {obs}")
        if not isinstance(node["cases"], dict):
            v.append(f"{path}:cases_must_be_dict")
        if not isinstance(node["default"], dict):
            v.append(f"{path}:default_must_be_a_node")

    elif op == "BUDGET":
        lim = node["limit_tokens"]
        if not isinstance(lim, int) or isinstance(lim, bool) or lim <= 0:
            v.append(f"{path}:limit_tokens_must_be_positive_int, got {lim!r}")
        if not isinstance(node["child"], dict):
            v.append(f"{path}:child_must_be_a_node")

    elif op == "STOP":
        if not isinstance(node["reason"], str) or not node["reason"]:
            v.append(f"{path}:reason_must_be_nonempty_str")

    return v


def _validate_params(path: str, mid: str, params, reg) -> list:
    if not isinstance(params, dict):
        return [f"{path}:params_must_be_dict"]
    grid = reg.params_grid(mid)
    if grid is None:
        return []
    v: list = []
    unknown = set(params) - set(grid)
    if unknown:
        v.append(f"{path}:unknown_params:{sorted(unknown)}")
    missing = set(grid) - set(params)
    if missing:
        v.append(f"{path}:missing_params:{sorted(missing)}")
    for key, allowed in grid.items():
        if key in params and params[key] not in allowed:
            v.append(f"{path}:param_off_grid:{key}={params[key]!r} not in {list(allowed)}")
    return v


def is_valid(g: dict, registry=None) -> bool:
    return not validate(g, registry)


# -- сериализация -------------------------------------------------------------------------


def describe(node: dict, indent: int = 0) -> str:
    """Компактная читаемая запись дерева — чтобы «кто выжил» можно было прочитать
    глазами, а не разбирать JSON."""
    pad = "  " * indent
    op = node.get("op")
    if op == "CALL":
        p = node.get("params") or {}
        mol = node["molecule"].replace("gen.", "")
        if not p:
            return f"{pad}CALL {mol}"
        return (f"{pad}CALL {mol} [{p.get('prompt')}, T={p.get('temperature')}, "
                f"slot={p.get('seed_slot')}, max={p.get('max_tokens')}]")
    if op == "CHECK":
        return f"{pad}CHECK {node['seam'].replace('seam.', '')}"
    if op == "STOP":
        return f"{pad}STOP({node['reason']})"
    if op in ("SEQ", "PAR", "ASSEMBLE"):
        lines = [f"{pad}{op}"]
        lines += [describe(c, indent + 1) for c in node["children"]]
        return "\n".join(lines)
    if op == "BUDGET":
        return f"{pad}BUDGET({node['limit_tokens']})\n" + describe(node["child"], indent + 1)
    if op == "SWITCH":
        lines = [f"{pad}SWITCH {node['obs']}"]
        for key in sorted(node["cases"]):
            lines.append(f"{pad}  case {key}:")
            lines.append(describe(node["cases"][key], indent + 2))
        lines.append(f"{pad}  default:")
        lines.append(describe(node["default"], indent + 2))
        return "\n".join(lines)
    return f"{pad}<{op}?>"


def dumps(g: dict) -> str:
    return json.dumps(g, ensure_ascii=False, sort_keys=True, indent=2)


def loads(text: str) -> dict:
    return json.loads(text)
