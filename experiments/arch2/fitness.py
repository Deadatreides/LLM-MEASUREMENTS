"""fitness.py — НЕЗАВИСИМЫЙ пересчёт фитнеса из трасс на диске (SPEC.md §2).

Этот модуль намеренно НЕ импортирует `runner.py` и ничего не исполняет: он читает
только то, что записано, и пересчитывает всё заново. Причина конкретна и взята из
истории проекта: оба крупных дефекта измерения (ложные FAIL при извлечении, шаг 1
сужения) жили ВНУТРИ исполнителя и потому были невидимы его собственным метрикам.
Независимый пересчёт ловит ровно этот класс. Контрольный случай в tests/ проверяет
отсутствие импорта механически, а не по договорённости.

Фитнес векторный: f = (r, −c, u). Скалярной свёртки нет (I-19: скалярная агрегация
в старом Мицелии была АНТИкоррелирована с истиной, ρ = −0.429).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Optional

RESOLVED = "RESOLVED"


# -- чтение трасс ---------------------------------------------------------------------


# Журналы, лежащие рядом с трассами, но трассами НЕ являющиеся. Список явный:
# молча пропускать записи без complex_id нельзя — так дефект измерения и прячется.
NON_TRACE_FILES = frozenset({"heredity.jsonl"})


def load_results(runs_dir: Path, experiment_id: str) -> dict:
    """-> {complex_id: {task_id: record}}. Читает append-only JSONL; при повторных
    записях одной пары (комплекс, задача) выигрывает последняя (журнал дописывается)."""
    out: dict = {}
    base = Path(runs_dir) / experiment_id
    if not base.exists():
        return out
    for path in sorted(base.glob("*.jsonl")):
        if path.name in NON_TRACE_FILES:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "complex_id" not in rec or "task_id" not in rec:
                raise ValueError(
                    f"{path.name}: запись без complex_id/task_id — это не трасса. "
                    f"Добавьте файл в NON_TRACE_FILES явно, а не пропускайте молча.")
            out.setdefault(rec["complex_id"], {})[rec["task_id"]] = rec
    return out


# -- базовые величины -----------------------------------------------------------------


def resolved_set(per_task: dict, task_ids: Iterable[str]) -> set:
    return {t for t in task_ids if (per_task.get(t) or {}).get("outcome") == RESOLVED}


def metrics(per_task: dict, task_ids: Iterable[str]) -> dict:
    task_ids = list(task_ids)
    n = len(task_ids)
    solved = resolved_set(per_task, task_ids)
    total_cost = sum((per_task.get(t) or {}).get("cost", 0) for t in task_ids)
    n_evaluated = sum(1 for t in task_ids if t in per_task)
    return {
        "n_tasks": n, "n_evaluated": n_evaluated, "n_resolved": len(solved),
        "r": (len(solved) / n) if n else 0.0,
        "total_cost": total_cost,
        "c": (total_cost / len(solved)) if solved else float("inf"),
        "resolved": sorted(solved),
    }


def unique_solves(per_task: dict, others: list, task_ids: Iterable[str]) -> list:
    """U(G): решённое этим комплексом и НИ ОДНИМ из `others` (обычно — элита)."""
    mine = resolved_set(per_task, task_ids)
    for other in others:
        mine -= resolved_set(other, task_ids)
    return sorted(mine)


# -- парные сравнения ------------------------------------------------------------------


def paired_delta_r(a: dict, b: dict, task_ids: Iterable[str], n_boot: int = 10000,
                   seed: int = 24680) -> dict:
    """ΔR = r(a) − r(b), парно по задачам, bootstrap 95% CI + знаковый тест.

    Тот же метод, что во всех предыдущих кампаниях проекта
    (experiment13/harness.py::bootstrap_rate_diff / paired_sign_test) — воспроизведён
    здесь, а не импортирован, чтобы `fitness.py` оставался автономным.
    """
    task_ids = list(task_ids)
    pairs = [((a.get(t) or {}).get("outcome") == RESOLVED,
              (b.get(t) or {}).get("outcome") == RESOLVED) for t in task_ids]
    n = len(pairs)
    if n == 0:
        return {"point": 0.0, "ci_95": [0.0, 0.0], "n_pairs": 0, "p_value": 1.0,
                "a_wins": 0, "b_wins": 0}

    point = (sum(1 for x, _ in pairs if x) - sum(1 for _, y in pairs if y)) / n
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append((sum(1 for x, _ in sample if x) - sum(1 for _, y in sample if y)) / n)
    diffs.sort()

    n10 = sum(1 for x, y in pairs if x and not y)
    n01 = sum(1 for x, y in pairs if y and not x)
    m = n10 + n01
    if m == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        p = min(1.0, sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m * 2)

    return {
        "point": point,
        "ci_95": [diffs[int(0.025 * n_boot)], diffs[min(int(0.975 * n_boot), n_boot - 1)]],
        "n_pairs": n, "p_value": p, "a_wins": n10, "b_wins": n01,
    }


def bootstrap_efficiency(a: dict, b: dict, task_ids: Iterable[str], n_boot: int = 2000,
                         seed: int = 13579) -> dict:
    """E = c(b)/c(a) с парным bootstrap по задачам. E > 1 ⟺ a дешевле на решённую
    задачу. Ресэмплируются ЗАДАЧИ (а не попытки): единица независимости — задача."""
    task_ids = list(task_ids)
    rows = [((a.get(t) or {}).get("cost", 0), (a.get(t) or {}).get("outcome") == RESOLVED,
             (b.get(t) or {}).get("cost", 0), (b.get(t) or {}).get("outcome") == RESOLVED)
            for t in task_ids]
    n = len(rows)

    def _ratio(sample):
        ca = sum(r[0] for r in sample); na = sum(1 for r in sample if r[1])
        cb = sum(r[2] for r in sample); nb = sum(1 for r in sample if r[3])
        if na == 0 or nb == 0 or ca == 0:
            return None
        return (cb / nb) / (ca / na)

    point = _ratio(rows)
    if n == 0 or point is None:
        return {"point": None, "ci_95": None, "n_pairs": n}
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        r = _ratio(sample)
        if r is not None:
            vals.append(r)
    vals.sort()
    if not vals:
        return {"point": point, "ci_95": None, "n_pairs": n}
    return {"point": point,
            "ci_95": [vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]],
            "n_pairs": n}


def paired_ratio_ci(res_a: dict, gate_a: dict, res_b: dict, gate_b: dict,
                    task_ids: Iterable[str], n_boot: int = 10000, seed: int = 20260819) -> dict:
    """Критерий регрессии V1 (SPEC.md §26): ratio = E_A / E_B, парный bootstrap.

    E_A = эффективность комплекса A относительно ЕГО планки, E_B = то же для B, оба
    пересчитываются на ОДНОЙ И ТОЙ ЖЕ выборке задач в каждом ресэмпле (общий rng на
    шаг) -- иначе ratio смешивал бы дисперсию двух независимых bootstrap.

    Провал V1 <=> point(ratio) < 1 И ci_95[1](ratio) < 1 -- формализация ручного
    расчёта, уже сделанного при подготовке REPORT_HEREDITY.md §3.
    """
    task_ids = list(task_ids)
    rows = [((res_a.get(t) or {}).get("cost", 0), (res_a.get(t) or {}).get("outcome") == RESOLVED,
             (gate_a.get(t) or {}).get("cost", 0), (gate_a.get(t) or {}).get("outcome") == RESOLVED,
             (res_b.get(t) or {}).get("cost", 0), (res_b.get(t) or {}).get("outcome") == RESOLVED,
             (gate_b.get(t) or {}).get("cost", 0), (gate_b.get(t) or {}).get("outcome") == RESOLVED)
            for t in task_ids]
    n = len(rows)

    def _eff(sample, i_cost, i_solved, j_cost, j_solved):
        c_num = sum(r[i_cost] for r in sample); n_num = sum(1 for r in sample if r[i_solved])
        c_den = sum(r[j_cost] for r in sample); n_den = sum(1 for r in sample if r[j_solved])
        if n_num == 0 or n_den == 0 or c_num == 0:
            return None
        return (c_den / n_den) / (c_num / n_num)

    def _ratio(sample):
        ea = _eff(sample, 0, 1, 2, 3)
        eb = _eff(sample, 4, 5, 6, 7)
        if ea is None or eb is None or eb == 0:
            return None
        return ea / eb

    point = _ratio(rows) if n else None
    if n == 0 or point is None:
        return {"point": None, "ci_95": None, "n_pairs": n, "regression": None}

    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        r = _ratio(sample)
        if r is not None:
            vals.append(r)
    if not vals:
        return {"point": point, "ci_95": None, "n_pairs": n, "regression": None}
    vals.sort()
    ci = [vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]]
    return {"point": point, "ci_95": ci, "n_pairs": n,
            "regression": bool(point < 1.0 and ci[1] < 1.0)}


def efficiency_gain(c_self: float, c_base: float) -> float:
    """E = c(B₀)/c(G) > 1 означает «дешевле эталона на решённую задачу»."""
    if c_self in (0, float("inf")) or math.isinf(c_self):
        return 0.0
    return c_base / c_self


# -- Парето и nиширование ----------------------------------------------------------------


def dominates(a: tuple, b: tuple) -> bool:
    """Все координаты максимизируются (c подаётся как −c)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_layers(points: dict) -> list:
    """{key: (r, −c, u)} -> список слоёв: слой 0 — недоминируемые, и так далее."""
    remaining = dict(points)
    layers = []
    while remaining:
        front = [k for k, v in remaining.items()
                 if not any(dominates(w, v) for k2, w in remaining.items() if k2 != k)]
        if not front:                       # защита от вырождения на равных точках
            front = list(remaining)
        layers.append(sorted(front))
        for k in front:
            remaining.pop(k)
    return layers


def shared_scores(results: dict, ids: list, task_ids: Iterable[str]) -> dict:
    """Fitness sharing по ПОКРЫТИЮ: решение задачи, которую решают многие, стоит
    меньше. Защита от вырождения популяции в один комплекс (SPEC.md §6).
    Обоснование — F8: смешанный пул закрывает РАЗНЫЕ слепые зоны."""
    task_ids = list(task_ids)
    solved_by = {t: sum(1 for cid in ids if t in resolved_set(results.get(cid, {}), [t]))
                 for t in task_ids}
    out = {}
    for cid in ids:
        mine = resolved_set(results.get(cid, {}), task_ids)
        out[cid] = sum(1.0 / solved_by[t] for t in mine if solved_by[t] > 0)
    return out


# -- сводка по поколению --------------------------------------------------------------------


def evaluate_population(results: dict, ids: list, task_ids: Iterable[str], baseline_id: str,
                        elite_ids: Optional[list] = None, n_boot: int = 2000) -> dict:
    """Полный вектор фитнеса для каждого комплекса + производные гейты."""
    task_ids = list(task_ids)
    elite_ids = list(elite_ids or [])
    base = results.get(baseline_id, {})
    base_m = metrics(base, task_ids)
    elite_results = [results.get(e, {}) for e in elite_ids if e != baseline_id]

    out = {}
    for cid in ids:
        per_task = results.get(cid, {})
        m = metrics(per_task, task_ids)
        uniq = unique_solves(per_task, elite_results, task_ids) if elite_results else []
        delta = paired_delta_r(per_task, base, task_ids, n_boot=n_boot)
        out[cid] = {
            **m,
            "u": len(uniq) / len(task_ids) if task_ids else 0.0,
            "unique_solves": uniq,
            "delta_r": delta,
            "efficiency_gain": efficiency_gain(m["c"], base_m["c"]),
            "fitness_vector": (m["r"], -m["c"] if not math.isinf(m["c"]) else -1e18,
                               len(uniq) / len(task_ids) if task_ids else 0.0),
        }
    out["_baseline"] = {"complex_id": baseline_id, **base_m}
    return out
