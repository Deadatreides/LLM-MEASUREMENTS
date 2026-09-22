"""rho_holdout.py — измеряет intra-class correlation ошибок rho_f по
каждому полю на HOLDOUT (тестовая половина того же 50/50 разбиения,
seed=7, что в validate_esw0.py). 0 новых вызовов -- те же данные
whole_grid.json.

rho_f = [P(оба неверны) - P(неверна)^2] / [P(неверна)(1-P(неверна))]

rho=0 -- независимость. rho>0 -- общий провал (декорреляция бы помогла).
rho<0 -- анти-корреляция (модели ошибаются РАЗНО) -- k_eff = k/(1+(k-1)rho)
формально ПРЕВЫШАЕТ k при rho<0: разнородный пул 6 разных семейств моделей
не делит failure modes.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiment14"))
sys.path.insert(0, str(ROOT / "experiment14" / "tasks"))

import seams as S14        # noqa: E402
import heterostep as T14   # noqa: E402

GRID = ROOT / "agent_life9_decomp_alpha" / "metrics" / "live_grid" / "whole_grid.json"
GEN_SEED = 150002
FIELDS = ("parcels", "ship_date", "tariff", "shipping_cost")
SPLIT_SEED = 7   # same as validate_esw0.py -- holdout = test half of the identical split

_LIST_PREFIX_RE = re.compile(r"^\s*\d+[.)](?!\d)\s*")


def _strip(text: str) -> str:
    return "\n".join(_LIST_PREFIX_RE.sub("", ln) for ln in (text or "").splitlines())


def eq(v, oracle) -> bool:
    if v is None:
        return False
    if isinstance(oracle, float):
        return isinstance(v, (int, float)) and abs(v - oracle) <= 0.011
    return v == oracle


def load_fields():
    tasks = T14.build_tasks(n=200, seed=GEN_SEED)
    rows = json.loads(GRID.read_text(encoding="utf-8"))["rows"]
    per = defaultdict(dict)
    for r in rows:
        res = S14.whole_seam(_strip(r.get("raw_output", "")), tasks[r["task_id"]]["steps"])
        per[r["task_id"]][r["model"]] = {sr["name"]: sr.get("extracted") for sr in res["step_results"]}
    full6 = sorted(t for t in per if len(per[t]) == 6)
    models = sorted({r["model"] for r in rows})
    return tasks, per, full6, models


def rho_for_field(per, tasks, models, ids, field_idx: int, field: str):
    wrong = {m: [not eq(per[t][m][field], tasks[t]["steps"][field_idx]["oracle"]) for t in ids]
            for m in models}
    p_bar = statistics.mean(statistics.mean(w) for w in wrong.values())
    pair_both = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            pair_both.append(statistics.mean(a and b for a, b in zip(wrong[m1], wrong[m2])))
    p_both = statistics.mean(pair_both)
    denom = p_bar * (1 - p_bar)
    rho = (p_both - p_bar ** 2) / denom if denom > 1e-9 else float("nan")
    return rho, p_bar


def main() -> None:
    import random
    tasks, per, full6, models = load_fields()
    rng = random.Random(SPLIT_SEED)
    ids = full6[:]
    rng.shuffle(ids)
    holdout = ids[50:]   # test half -- identical split to validate_esw0.py

    print(f"holdout n={len(holdout)} (test half, seed={SPLIT_SEED}, same split as validate_esw0.py)\n")
    out = {}
    for i, f in enumerate(FIELDS):
        rho, pbar = rho_for_field(per, tasks, models, holdout, i, f)
        k_eff_at_6 = 6 / (1 + 5 * rho) if (1 + 5 * rho) > 0 else float("inf")
        print(f"  {f:15s}  P(wrong)={pbar:.3f}   rho={rho:+.3f}   k_eff(k=6)={k_eff_at_6:.1f}")
        out[f] = {"p_wrong": pbar, "rho": rho, "k_eff_at_6": k_eff_at_6}

    (Path(__file__).resolve().parent / "metrics" / "rho_holdout.json").write_text(
        json.dumps({"n_holdout": len(holdout), "split_seed": SPLIT_SEED, "fields": out},
                  ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
