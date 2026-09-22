"""heredity.py — наследственность Complex Layer (SPEC.md §13-19).

Четыре механизма, закрывающие три пробела первого цикла:
  1. типизированные рёбра между complex_id (родство больше не теряется);
  2. placement-score как ДИАГНОСТИКА — решение принимают явные CASE-условия;
  3. decay/uses/shadow — обратимое затухание, ортогональное смерти D1/D2/D3;
  4. служебные функции замкнутого цикла «комплекс -> молекула -> снова в генотипах».

Зависимости: только `genotype.py` (уровень 1). Runner, реестр и популяция здесь не
нужны — модуль оперирует id, масками исходов и генотипами, поэтому тестируется без
GPU и без реестра.

Ключевое проектное решение, взятое из источника идеи
(`ptg/PTG+MCP1.3/ptg_core.py`, ПАТЧ 9/10): скалярный placement НЕ является критерием.
Там это сказано дословно — «один скаляр — источник багов», и решение о
continues/reinforces принимается набором CASE-условий. Здесь так же: `place()` считает
диагностику, а `classify_relation()` выносит решение по трём явным правилам.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import genotype as G

# -- константы (SPEC.md §18, зафиксированы ДО прогона) ---------------------------------
W_SEM = 0.40
W_TEMP = 0.20
W_DENS = 0.20
W_SKEL = 0.20

SEM_HIGH = 0.90
ECHO_MIN_GEN = 2
TEMP_HALF_LIFE_GENS = 3.0

W0 = 1.0
LAMBDA = 3.0
RHO = 0.5
W_FLOOR = 0.25

MAX_REGISTER_PER_GEN = 1
M6_WEIGHT_WITH_COMPOSITES = 0.15
P_COMPOSITE_LEAF = 0.25
H0_GENERATIONS = 8

EDGE_TYPES = ("mutated_from", "crossed_from", "reinforces", "echo",
              "compressed_into", "instantiated_from")
LINEAGE_EDGES = ("mutated_from", "crossed_from")

RESOLVED = "RESOLVED"


# -- 1. журнал рёбер ---------------------------------------------------------------------


class EdgeLog:
    """Append-only журнал рёбер наследственности.

    Удаления нет вообще — как у Storage в arch1. Инвариант NO_MERGE (SPEC.md §14.1)
    держится тем, что здесь нет и не может быть операции слияния двух id: похожие
    комплексы получают РЕБРО, а не общий идентификатор.
    """

    def __init__(self):
        self._edges: list = []
        self._out: dict = {}
        self._in: dict = {}

    def add(self, src: str, dst: str, edge_type: str, gen: int,
            evidence: Optional[dict] = None) -> dict:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type: {edge_type}")
        edge = {"src": src, "dst": dst, "type": edge_type, "gen": int(gen),
                "evidence": dict(evidence or {})}
        self._edges.append(edge)
        self._out.setdefault(src, []).append(edge)
        self._in.setdefault(dst, []).append(edge)
        return edge

    def __len__(self) -> int:
        return len(self._edges)

    def all(self) -> list:
        return list(self._edges)

    def counts(self) -> dict:
        return dict(Counter(e["type"] for e in self._edges))

    def edges_of(self, cid: str, types: Optional[Iterable[str]] = None,
                 direction: str = "out") -> list:
        pool = self._out.get(cid, []) if direction == "out" else self._in.get(cid, [])
        if types is None:
            return list(pool)
        types = set(types)
        return [e for e in pool if e["type"] in types]

    # -- линии --

    def ancestors(self, cid: str) -> set:
        """Транзитивное замыкание вверх по mutated_from/crossed_from."""
        seen, stack = set(), [cid]
        while stack:
            cur = stack.pop()
            for e in self.edges_of(cur, LINEAGE_EDGES, "out"):
                if e["dst"] not in seen:
                    seen.add(e["dst"])
                    stack.append(e["dst"])
        return seen

    def descendants(self, cid: str) -> set:
        seen, stack = set(), [cid]
        while stack:
            cur = stack.pop()
            for e in self.edges_of(cur, LINEAGE_EDGES, "in"):
                if e["src"] not in seen:
                    seen.add(e["src"])
                    stack.append(e["src"])
        return seen

    def same_lineage(self, a: str, b: str) -> bool:
        """Одна линия = один является предком другого, либо у них общий предок."""
        if a == b:
            return True
        anc_a, anc_b = self.ancestors(a) | {a}, self.ancestors(b) | {b}
        return bool(anc_a & anc_b)

    def to_jsonl(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in self._edges:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        tmp.replace(path)

    @classmethod
    def from_jsonl(cls, path: Path) -> "EdgeLog":
        log = cls()
        path = Path(path)
        if not path.exists():
            return log
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                log.add(e["src"], e["dst"], e["type"], e["gen"], e.get("evidence"))
        return log


# -- 2. каркас генотипа --------------------------------------------------------------------


def skeleton(g: dict) -> dict:
    """Подпись каркаса: из чего собран комплекс, безотносительно конкретных молекул.

    Нужна, чтобы отличать «тот же приём другими молекулами» от «другой приём»:
    для отчёта о разнообразии элиты это независимая от семейства моделей ось.
    """
    root = g["root"] if "root" in g else g
    ops, obs, par_widths = Counter(), set(), []
    for _, node in G.walk(root):
        ops[node.get("op")] += 1
        if node.get("op") == "SWITCH":
            obs.add(node.get("obs"))
        if node.get("op") == "PAR":
            par_widths.append(len(node.get("children") or []))
    return {
        "ops": dict(ops),
        "depth": G.depth(root),
        "switch_obs": sorted(obs),
        "has_stop": ops.get("STOP", 0) > 0,
        "par_widths": par_widths,
    }


def _multiset_jaccard(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return inter / union if union else 1.0


def skel_similarity(sa: dict, sb: dict) -> float:
    ops = _multiset_jaccard(sa["ops"], sb["ops"])
    ddepth = abs(sa["depth"] - sb["depth"]) / max(G.MAX_DEPTH, 1)
    depth_term = max(0.0, 1.0 - ddepth)
    same_obs = 1.0 if sa["switch_obs"] == sb["switch_obs"] else 0.0
    return 0.5 * ops + 0.3 * depth_term + 0.2 * same_obs


def niche_key(g: dict) -> str:
    """Ниша = стабильный хеш каркаса, БЕЗ cost_bin (SPEC.md §23).

    Квартили стоимости были рассмотрены и отклонены: при ELITE_SIZE=8 и
    наблюдённых 6-8 различных каркасах в элите (REPORT_HEREDITY.md §4) они
    фрагментировали бы ниши до одного представителя без всякого отбора внутри.
    """
    import hashlib

    canon = json.dumps(skeleton(g), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


# -- 3. фенотипическая близость ------------------------------------------------------------


def resolved_mask(per_task: dict, task_ids: Iterable[str]) -> dict:
    return {t: (per_task.get(t) or {}).get("outcome") == RESOLVED for t in task_ids}


def sem_similarity(a: dict, b: dict) -> float:
    """Жаккар решённых задач ТОЛЬКО по пересечению оценённых.

    Считать по объединению нельзя: комплекс получал бы сходство/уникальность за
    задачи, которых конкурент не видел. Ровно этот дефект был найден в E3 и
    исправлен там же — здесь он не повторяется по построению.
    """
    common = set(a) & set(b)
    if not common:
        return 0.0
    sa = {t for t in common if (a.get(t) or {}).get("outcome") == RESOLVED}
    sb = {t for t in common if (b.get(t) or {}).get("outcome") == RESOLVED}
    if not sa and not sb:
        return 1.0            # оба не решили ничего на общем срезе — один и тот же режим
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def temporal_affinity(gen_a: int, gen_b: int, half_life: float = TEMP_HALF_LIFE_GENS) -> float:
    if half_life <= 0:
        return 1.0 if gen_a == gen_b else 0.0
    return float(0.5 ** (abs(gen_a - gen_b) / half_life))


def density_term(cid: str, results: dict, peers: Iterable[str],
                 task_ids: Iterable[str], sem_high: float = SEM_HIGH) -> float:
    """1 − доля соседей, к которым фенотип этого комплекса уже близок.

    Низкое значение = «такого добра тут и так много»; используется как ранжирующий
    штраф при выборе кандидата на регистрацию, не как приговор.
    """
    peers = [p for p in peers if p != cid]
    if not peers:
        return 1.0
    mine = results.get(cid, {})
    close = sum(1 for p in peers
                if sem_similarity(mine, results.get(p, {})) >= sem_high)
    return 1.0 - close / len(peers)


def place(sem: float, temp: float, density: float, skel: float) -> float:
    """Диагностический скаляр (SPEC.md §15). НЕ критерий смерти и НЕ фитнес."""
    return W_SEM * sem + W_TEMP * temp + W_DENS * density + W_SKEL * skel


def classify_relation(sem: float, dgen: int, same_lineage: bool,
                      sem_high: float = SEM_HIGH,
                      echo_min_gen: int = ECHO_MIN_GEN) -> Optional[str]:
    """Явные CASE-условия вместо argmax скаляра (SPEC.md §15).

    -> "echo" | "reinforces" | None (независимая линия).
    """
    if sem < sem_high:
        return None
    if same_lineage:
        return "echo" if dgen >= echo_min_gen else None
    return "reinforces"


def donor_score(skel: float, sem: float) -> float:
    """Для M5: структурно совместимый, но фенотипически ДОПОЛНЯЮЩИЙ донор.

    Прививать подграф от донора, который решает ровно то же самое, бессмысленно —
    это и есть та самая избыточность, от которой защищает §14.1.
    """
    return 0.5 * skel + 0.5 * (1.0 - sem)


# -- 4. decay / uses / shadow ---------------------------------------------------------------


def weight(dgen_idle: int, uses: int, w0: float = W0, lam: float = LAMBDA,
           rho: float = RHO) -> float:
    """w(t) = W0 · exp(−Δgen_idle/λ) · (1 + ρ·uses)  (SPEC.md §16)."""
    return float(w0 * math.exp(-max(0, dgen_idle) / lam) * (1.0 + rho * max(0, uses)))


def is_shadow(w: float, floor: float = W_FLOOR) -> bool:
    return w < floor


# -- 5. служебное для замкнутого цикла ------------------------------------------------------


def composite_calls(g: dict, composite_prefix: str = "cx.") -> list:
    """Пути и id всех CALL на зарегистрированные комплексы-молекулы."""
    root = g["root"] if "root" in g else g
    out = []
    for path, node in G.walk(root):
        if node.get("op") == "CALL" and str(node.get("molecule", "")).startswith(composite_prefix):
            out.append({"node_path": path, "molecule": node["molecule"]})
    return out


def uses_composite(g: dict, molecule_id: Optional[str] = None) -> bool:
    calls = composite_calls(g)
    if molecule_id is None:
        return bool(calls)
    return any(c["molecule"] == molecule_id for c in calls)


def cost_profile_from_traces(per_task: dict) -> dict:
    """Измеренный, а не оценённый профиль стоимости (SPEC.md §17, критерий 4)."""
    costs = sorted(rec.get("cost", 0) for rec in per_task.values())
    calls = sorted(rec.get("n_calls", 0) for rec in per_task.values())
    if not costs:
        return {"n_observations": 0, "median_tokens": None, "median_calls": None}
    mid = len(costs) // 2
    # Медиана И среднее: у маршрутизирующих комплексов распределение бимодально
    # (ветвь STOP стоит 0), и одна медиана читалась бы как «молекула бесплатна».
    return {
        "n_observations": len(costs),
        "median_tokens": costs[mid], "median_calls": calls[mid],
        "mean_tokens": sum(costs) / len(costs), "mean_calls": sum(calls) / len(calls),
        "zero_cost_share": sum(1 for c in costs if c == 0) / len(costs),
        "min_tokens": costs[0], "max_tokens": costs[-1],
    }


def registration_gate(cid: str, metrics: dict, gate_metrics: dict, genotype_obj: dict,
                      registry, n_min: int) -> dict:
    """Все четыре критерия §17. -> {"ok": bool, "reasons": [...], "checks": {...}}.

    Критерий 2 — «не доминируется планкой»: планка обязана быть НЕ ЛУЧШЕ по обеим осям
    сразу. Это ровно «лежит на Парето (r, −c) относительно планки».
    """
    checks, reasons = {}, []

    n_eval = metrics.get("n_evaluated", 0)
    checks["n_evaluated"] = n_eval
    if n_eval < n_min:
        reasons.append(f"n_evaluated {n_eval} < {n_min}")

    r, c = metrics.get("r", 0.0), metrics.get("c", float("inf"))
    gr, gc = gate_metrics.get("r", 0.0), gate_metrics.get("c", float("inf"))
    dominated = (gr >= r) and (gc <= c)
    checks["dominated_by_gate"] = dominated
    checks["r"], checks["c"], checks["gate_r"], checks["gate_c"] = r, c, gr, gc
    if dominated:
        reasons.append(f"dominated by gate (r {r:.3f}<={gr:.3f} and c {c:.1f}>={gc:.1f})")

    violations = G.validate(genotype_obj, registry)
    checks["genotype_violations"] = violations
    if violations:
        reasons.append(f"invalid genotype: {violations[:1]}")

    return {"ok": not reasons, "reasons": reasons, "checks": checks}


# -- 6. усиление отбора (SPEC.md §21-28) ----------------------------------------------------


def fail_gate(delta_r: dict) -> bool:
    """Провал среза = верхняя граница ΔR относительно планки строго ниже нуля.

    Тот же предикат, что уже использует скрининг D1 (`evolve._d1_screen_deaths`) —
    не новое понятие, а его переиспользование для dual-slice D3 (SPEC.md §25).
    """
    ci = (delta_r or {}).get("ci_95") or [0.0, 0.0]
    return ci[1] < 0.0


COMPOSITE_PREFIX = "cx."


def n_raw_calls(g: dict, composite_prefix: str = COMPOSITE_PREFIX) -> int:
    """Число CALL на НЕкомпозитную молекулу — часть tie-break парсимонии (SPEC.md §24)."""
    root = g["root"] if "root" in g else g
    return sum(1 for _, n in G.walk(root)
              if n.get("op") == "CALL" and not str(n.get("molecule", "")).startswith(composite_prefix))


def composite_nesting_depth(g: dict, registry, composite_prefix: str = COMPOSITE_PREFIX,
                            _seen: Optional[frozenset] = None) -> int:
    """0, если композитов нет; иначе 1 + максимум по вложенным композитам.

    Рекурсия идёт по уже ЗАМОРОЖЕННЫМ геномам композитов (регистрация всегда позже
    по времени, чем регистрация того, что она в себя включает), поэтому зациклиться
    не может; `_seen` — дополнительная защита от случайного цикла на синтетических
    данных в тестах, а не от реального сценария.
    """
    calls = composite_calls(g, composite_prefix)
    if not calls:
        return 0
    seen = _seen or frozenset()
    best = 0
    for c in calls:
        mid = c["molecule"]
        if mid in seen or not getattr(registry, "is_composite", lambda _m: False)(mid):
            continue
        inner = registry.get(mid).get("genotype")
        if inner is None:
            continue
        best = max(best, 1 + composite_nesting_depth(inner, registry, composite_prefix,
                                                      seen | {mid}))
    return best if best else (1 if calls else 0)


def complexity_key(g: dict, registry) -> tuple:
    """(n_nodes, n_raw_calls, composite_nesting_depth) — tie-break парсимонии
    (SPEC.md §24), в этом самом порядке компонент кортежа."""
    root = g["root"] if "root" in g else g
    return (G.n_nodes(root), n_raw_calls(g), composite_nesting_depth(g, registry))


def dominated_by_registered(candidate_r: float, candidate_c: float, candidate_nodes: int,
                            registered: list) -> Optional[dict]:
    """Первая уже зарегистрированная молекула, которая доминирует кандидата по (r,-c)
    И при этом candidate СЛОЖНЕЕ неё по числу узлов -- или None (SPEC.md §24).

    Дешёвая, но более слабая молекула кандидата НЕ отклоняет: доминирования по
    (r,-c) в эту сторону нет. Это ровно дешёвый край Парето, который §17 обязан
    пропускать (см. test_accepts_cheaper_even_if_slightly_worse).
    """
    for rec in registered:
        rr, rc, rn = rec.get("r", 0.0), rec.get("c", float("inf")), rec.get("n_nodes")
        if rn is None:
            continue
        dominates = (rr >= candidate_r) and (rc <= candidate_c) and (rr > candidate_r or rc < candidate_c)
        if dominates and candidate_nodes > rn:
            return rec
    return None
