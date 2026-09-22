"""validate_esw0.py — ESW-0: нулевая по стоимости проверка ядра теории
эмерджентного роя на УЖЕ СУЩЕСТВУЮЩИХ данных (0 новых generate()).

Читает `agent_life9_decomp_alpha/metrics/live_grid/whole_grid.json`
(100 задач × 6 моделей, целостные ответы, лежит на диске с LIFE-3),
перепарсивает их поправкой LIFE-9 (list-prefix), извлекает ПОЛЯ
совместного ответа и сравнивает четыре оценщика:

  best_single   -- лучшая одиночная модель, ВЫБРАННАЯ НА TRAIN (честно)
  plain         -- простое большинство по всем 6
  weighted      -- лог-оддс взвешивание (Naive Bayes / Dawid-Skene M-шаг)
  condorcet     -- большинство ТОЛЬКО среди моделей с pi_{m,f} > 1/2

Ключевой результат (40 случайных 50/50 разбиений): condorcet побеждает
best_single на 40/40. Наивное большинство ПРОИГРЫВАЕТ -- условие
Кондорсе несущее, не декоративное.

Запуск:  python theory_emergent_swarm/validate_esw0.py
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment14"))
sys.path.insert(0, str(ROOT / "experiment14" / "tasks"))

import seams as S14          # noqa: E402
import heterostep as T14     # noqa: E402

GRID = ROOT / "agent_life9_decomp_alpha" / "metrics" / "live_grid" / "whole_grid.json"
GEN_SEED = 150002
FIELDS = ("parcels", "ship_date", "tariff", "shipping_cost")
N_SPLITS = 40
SPLIT_SEED = 7

# поправка разбора, найденная LIFE-9 (нумерованный список ломает _ASSIGN_RE);
# negative lookahead -- чтобы "13.1" не съедалось до "1" (урок того же прогона)
_LIST_PREFIX_RE = re.compile(r"^\s*\d+[.)](?!\d)\s*")


def _strip(text: str) -> str:
    return "\n".join(_LIST_PREFIX_RE.sub("", ln) for ln in (text or "").splitlines())


def eq(v, oracle) -> bool:
    if v is None:
        return False
    if isinstance(oracle, float):
        return isinstance(v, (int, float)) and abs(v - oracle) <= 0.011
    return v == oracle


def load_fields() -> tuple:
    tasks = T14.build_tasks(n=200, seed=GEN_SEED)
    rows = json.loads(GRID.read_text(encoding="utf-8"))["rows"]
    per = defaultdict(dict)
    for r in rows:
        res = S14.whole_seam(_strip(r.get("raw_output", "")), tasks[r["task_id"]]["steps"])
        per[r["task_id"]][r["model"]] = {sr["name"]: sr.get("extracted") for sr in res["step_results"]}
    full6 = sorted(t for t in per if len(per[t]) == 6)
    models = sorted({r["model"] for r in rows})
    return tasks, per, full6, models


def reliabilities(per, tasks, models, train_ids) -> dict:
    """pi_{m,f} -- точность модели m по полю f, оценённая ТОЛЬКО на train."""
    rel = {}
    for m in models:
        for i, f in enumerate(FIELDS):
            n = sum(1 for t in train_ids if eq(per[t][m][f], tasks[t]["steps"][i]["oracle"]))
            rel[(m, f)] = n / len(train_ids)
    return rel


def consensus_score(per, tasks, models, rel, test_ids, mode: str, thr: float = 0.5) -> float:
    ok = 0
    for t in test_ids:
        good = True
        for i, f in enumerate(FIELDS):
            if mode == "condorcet":
                pool = [m for m in models if rel[(m, f)] >= thr]
                if not pool:                                  # A_f пусто -> лучший доступный
                    pool = [max(models, key=lambda m: rel[(m, f)])]
            else:
                pool = models
            tally = defaultdict(float)
            for m in pool:
                v = per[t][m][f]
                if v is None:
                    continue
                key = round(v, 2) if isinstance(v, float) else v
                if mode == "weighted":
                    p = min(max(rel[(m, f)], 0.02), 0.98)
                    tally[key] += math.log(p / (1 - p))
                else:
                    tally[key] += 1.0
            if not tally:
                good = False
                break
            if not eq(max(tally.items(), key=lambda kv: kv[1])[0], tasks[t]["steps"][i]["oracle"]):
                good = False
                break
        if good:
            ok += 1
    return ok / len(test_ids)


def task_level_r(per, tasks, m, ids) -> float:
    return sum(1 for t in ids
               if all(eq(per[t][m][f], tasks[t]["steps"][i]["oracle"])
                      for i, f in enumerate(FIELDS))) / len(ids)


def best_single_honest(per, tasks, models, train_ids, test_ids) -> float:
    """Модель выбирается на TRAIN, оценивается на TEST -- без подглядывания."""
    m = max(models, key=lambda mm: task_level_r(per, tasks, mm, train_ids))
    return task_level_r(per, tasks, m, test_ids)


def main() -> None:
    tasks, per, full6, models = load_fields()
    print(f"задач с полным покрытием 6 моделей: {len(full6)}\n")

    print("точность ПО ПОЛЯМ (в скобках -- r уровня задачи):")
    for m in sorted(models, key=lambda mm: -task_level_r(per, tasks, mm, full6)):
        accs = [sum(1 for t in full6 if eq(per[t][m][f], tasks[t]["steps"][i]["oracle"])) / len(full6)
                for i, f in enumerate(FIELDS)]
        print(f"  {m[:34]:34s} [{task_level_r(per, tasks, m, full6):.2f}] "
              + " ".join(f"{f[:5]}={a:.2f}" for f, a in zip(FIELDS, accs)))

    res = defaultdict(list)
    rng = random.Random(SPLIT_SEED)
    for _ in range(N_SPLITS):
        ids = full6[:]
        rng.shuffle(ids)
        tr, te = ids[:50], ids[50:]
        rel = reliabilities(per, tasks, models, tr)
        res["best_single"].append(best_single_honest(per, tasks, models, tr, te))
        res["plain"].append(consensus_score(per, tasks, models, rel, te, "plain"))
        res["weighted"].append(consensus_score(per, tasks, models, rel, te, "weighted"))
        res["condorcet"].append(consensus_score(per, tasks, models, rel, te, "condorcet"))

    print(f"\n{N_SPLITS} случайных 50/50 разбиений, среднее [min..max]:")
    for k, label in (("best_single", "best single (train-picked)"), ("plain", "plain majority"),
                     ("weighted", "log-odds weighted"), ("condorcet", "CONDORCET-filtered")):
        v = res[k]
        print(f"  {label:28s} {statistics.mean(v):.3f}  [{min(v):.2f}..{max(v):.2f}]")

    wins = sum(1 for a, b in zip(res["condorcet"], res["best_single"]) if a > b)
    print(f"\n  condorcet > best_single на {wins}/{N_SPLITS} разбиениях")

    out = {"n_tasks": len(full6), "n_splits": N_SPLITS,
           "means": {k: statistics.mean(v) for k, v in res.items()},
           "condorcet_wins_over_best_single": wins}
    (Path(__file__).resolve().parent / "metrics" / "esw0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
