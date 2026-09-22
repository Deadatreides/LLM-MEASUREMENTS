"""ceiling.py — потолок алфавита и природа провалов (0 GPU, по сохранённому сырью).

Отвечает на вопрос, который три цикла отбора обходили стороной: система не бьёт
сильнейшую модель — потому что отбор плох, или потому что отбирать не из чего?

Считает по `experiment12/runs12/phase0_full_exp12-v1.json` (1320 первичных вызовов,
6 моделей x 55 задач x 4 seed):

  1. ПОТОЛОК АЛФАВИТА  -- доля задач, решаемых хоть кем-то хоть когда-то;
  2. РАЗЛОЖЕНИЕ ПУЛА   -- универсальные / поле боя / безнадёжные;
  3. ПОТОЛОК ПРЕВОСХОДСТВА -- задачи, где лучшая модель провалила ВСЕ 4 seed,
                          а кто-то другой решил (только здесь портфель может выиграть);
  4. ПОРЯДОК РАЗНООБРАЗИЯ -- N_eff = N/(1+(N-1)rho) -> 1/rho, жёсткий предел;
  5. МОЩНОСТЬ          -- сколько задач нужно, чтобы измеренный эффект стал значим;
  6. УСТОЙЧИВОСТЬ ВЫБОРА -- split-half: тот же ли портфель лучший на другой половине;
  7. ПРИРОДА ПРОВАЛОВ  -- FAIL (неспособность) против INAPPLICABLE (протокол вывода).

Пункт 7 — главный: он определяет, ЧТО именно оптимизировал слой комплексов.
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "experiment12" / "runs12" / "phase0_full_exp12-v1.json"
METRICS = Path(__file__).resolve().parent / "metrics"
SEEDS = (1, 2, 3, 4)
N_BOOT = 20000


def load():
    with open(PHASE0, encoding="utf-8") as f:
        d = json.load(f)
    ir = d["initial_records"]
    models = sorted({r["model"] for r in ir})
    status = {(r["model"], r["task_id"], r["seed"]): r["status"] for r in ir}
    tokens = {(r["model"], r["task_id"], r["seed"]): r["total_tokens"] for r in ir}
    tasks = sorted({r["task_id"] for r in ir})
    full = [t for t in tasks if all((m, t, s) in status for m in models for s in SEEDS)]
    return models, status, tokens, full


def solved_by(status, models, t):
    return {m for m in models if any(status[(m, t, s)] == "PASS" for s in SEEDS)}


def paired_bootstrap(rows, n_boot=N_BOOT, seed=7):
    rng = random.Random(seed)
    n = len(rows)
    point = sum(a for a, _ in rows) / n - sum(b for _, b in rows) / n
    diffs = []
    for _ in range(n_boot):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(a for a, _ in s) / n - sum(b for _, b in s) / n)
    diffs.sort()
    return point, diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def n_eff(oracle: float, p1: float) -> float:
    """Сколько НЕЗАВИСИМЫХ попыток с вероятностью p1 дали бы наблюдённый oracle."""
    if not 0 < oracle < 1:
        return float("inf")
    return math.log(1 - oracle) / math.log(1 - p1)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    models, status, tokens, full = load()
    solvers = {t: solved_by(status, models, t) for t in full}

    universal = [t for t in full if len(solvers[t]) == len(models)]
    hopeless = [t for t in full if not solvers[t]]
    battle = [t for t in full if 1 <= len(solvers[t]) <= len(models) - 1]
    ceiling = len(full) - len(hopeless)

    o4 = {m: sum(any(status[(m, t, s)] == "PASS" for s in SEEDS) for t in full) / len(full)
          for m in models}
    best_single = max(o4, key=o4.get)
    p1 = {m: sum(status[(m, t, s)] == "PASS" for t in full for s in SEEDS) / (len(full) * 4)
          for m in models}

    best_sub, best_val = None, -1.0
    for S in itertools.combinations(models, 4):
        v = sum(any(status[(m, t, s)] == "PASS" for m in S)
                for t in full for s in SEEDS) / (len(full) * 4)
        if v > best_val:
            best_val, best_sub = v, S

    rescue = [t for t in full if best_single not in solvers[t] and solvers[t]]

    p_pool = sum(p1.values()) / len(models)
    within = sum(any(status[(m, t, s)] == "PASS" for s in SEEDS)
                 for m in models for t in full) / (len(models) * len(full))
    cross = sum(any(status[(m, t, s)] == "PASS" for m in S)
                for S in itertools.combinations(models, 4)
                for t in full for s in SEEDS) / (15 * len(full) * 4)
    nw, nc = n_eff(within, p_pool), n_eff(cross, p_pool)
    rho_w, rho_c = (4 / nw - 1) / 3, (4 / nc - 1) / 3

    def rows_for(pool):
        return [(sum(any(status[(m, t, s)] == "PASS" for m in best_sub) for s in SEEDS) / 4,
                 1.0 if any(status[(best_single, t, s)] == "PASS" for s in SEEDS) else 0.0)
                for t in pool]

    res = {}
    for label, pool in (("full_pool", full), ("battleground", battle)):
        pt, lo, hi = paired_bootstrap(rows_for(pool))
        sd = (hi - lo) / 3.92 * math.sqrt(len(pool))
        need = math.ceil(7.85 * sd * sd / (pt * pt)) if pt else None
        res[label] = {"n": len(pool), "delta": pt, "ci_95": [lo, hi],
                      "significant": lo > 0, "sd": sd, "n_needed_for_80pct_power": need}

    rng = random.Random(11)
    agree = 0
    for _ in range(200):
        sh = full[:]
        rng.shuffle(sh)
        halves = (sh[:len(sh) // 2], sh[len(sh) // 2:])
        picks = [max(itertools.combinations(models, 4),
                     key=lambda S: sum(any(status[(m, t, s)] == "PASS" for m in S)
                                       for t in h for s in SEEDS))
                 for h in halves]
        agree += picks[0] == picks[1]

    counts = collections.Counter(status.values())
    by_group = {}
    for name, pool in (("universal", universal), ("battleground", battle), ("hopeless", hopeless)):
        c = collections.Counter(status[(m, t, s)] for m in models for t in pool for s in SEEDS)
        tot = sum(c.values()) or 1
        by_group[name] = {k: c[k] / tot for k in ("PASS", "FAIL", "INAPPLICABLE")}
    c_rescue = collections.Counter(status[(best_single, t, s)] for t in rescue for s in SEEDS)
    tot_r = sum(c_rescue.values()) or 1

    lazy = {}
    for name, pool in (("universal", universal), ("battleground", battle), ("hopeless", hopeless)):
        tk = rs = 0
        for t in pool:
            for m in best_sub:
                tk += tokens[(m, t, 1)]
                if status[(m, t, 1)] == "PASS":
                    rs += 1
                    break
        lazy[name] = {"n": len(pool), "tokens": tk, "resolved": rs}
    tot_tok = sum(v["tokens"] for v in lazy.values())
    tot_res = sum(v["resolved"] for v in lazy.values())

    out = {
        "source": str(PHASE0.relative_to(ROOT)),
        "n_tasks": len(full), "n_models": len(models), "n_calls": len(full) * 24,
        "alphabet_ceiling": {"solvable_by_someone": ceiling, "share": ceiling / len(full),
                             "hopeless": len(hopeless)},
        "pool_decomposition": {"universal": len(universal), "battleground": len(battle),
                               "hopeless": len(hopeless),
                               "share_that_selection_can_affect": len(battle) / len(full)},
        "best_single": {"model": best_single, "oracle_at_4": o4[best_single], "p1": p1[best_single]},
        "best_portfolio": {"models": list(best_sub), "oracle_at_4": best_val,
                           "mean_p1": sum(p1[m] for m in best_sub) / 4,
                           "advantage": best_val - o4[best_single]},
        "superiority_ceiling": {"n_rescue_tasks": len(rescue),
                                "max_advantage_share": len(rescue) / len(full),
                                "best_model_failure_mode_on_rescue":
                                    {k: c_rescue[k] / tot_r for k in ("INAPPLICABLE", "FAIL")}},
        "diversity_order": {"within_model": {"oracle_at_4": within, "n_eff": nw, "rho": rho_w,
                                             "asymptotic_ceiling": 1 / rho_w},
                            "cross_model": {"oracle_at_4": cross, "n_eff": nc, "rho": rho_c,
                                            "asymptotic_ceiling": 1 / rho_c}},
        "power": res,
        "portfolio_choice_stability": {"split_half_agreement": agree / 200, "n_splits": 200},
        "failure_nature": {"overall": {k: counts[k] / sum(counts.values())
                                       for k in ("PASS", "FAIL", "INAPPLICABLE")},
                           "by_group": by_group},
        "cost_structure_lazy_portfolio": {
            **lazy, "total_tokens": tot_tok, "total_resolved": tot_res,
            "tokens_per_resolved": tot_tok / tot_res,
            "tokens_per_resolved_if_hopeless_skipped":
                (tot_tok - lazy["hopeless"]["tokens"]) / tot_res,
            "free_efficiency_gain": tot_tok / (tot_tok - lazy["hopeless"]["tokens"]),
        },
    }

    METRICS.mkdir(parents=True, exist_ok=True)
    path = METRICS / "ceiling.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

    a = out["alphabet_ceiling"]
    print(f"ПОТОЛОК АЛФАВИТА: {a['solvable_by_someone']}/{len(full)} = {a['share']:.4f} "
          f"(безнадёжных {a['hopeless']})")
    d = out["pool_decomposition"]
    print(f"РАЗЛОЖЕНИЕ ПУЛА: универсальных {d['universal']}, поле боя {d['battleground']}, "
          f"безнадёжных {d['hopeless']} -> отбор влияет на {d['share_that_selection_can_affect']:.1%}")
    print(f"лучшая одиночная {out['best_single']['oracle_at_4']:.4f} "
          f"({out['best_single']['model']}, p1={out['best_single']['p1']:.4f})")
    print(f"лучший портфель  {out['best_portfolio']['oracle_at_4']:.4f} "
          f"(mean p1={out['best_portfolio']['mean_p1']:.4f}), "
          f"превосходство {out['best_portfolio']['advantage']:+.4f}")
    s = out["superiority_ceiling"]
    print(f"ПОТОЛОК ПРЕВОСХОДСТВА: {s['n_rescue_tasks']} задач = {s['max_advantage_share']:+.4f}; "
          f"лучшая модель там провалилась по формату в "
          f"{s['best_model_failure_mode_on_rescue']['INAPPLICABLE']:.0%} случаев")
    dv = out["diversity_order"]
    print(f"ПОРЯДОК РАЗНООБРАЗИЯ: внутри модели N_eff={dv['within_model']['n_eff']:.2f} "
          f"(потолок {dv['within_model']['asymptotic_ceiling']:.2f}); "
          f"между моделями N_eff={dv['cross_model']['n_eff']:.2f} "
          f"(потолок {dv['cross_model']['asymptotic_ceiling']:.2f})")
    for k, v in res.items():
        print(f"МОЩНОСТЬ {k}: dR={v['delta']:+.4f} CI[{v['ci_95'][0]:+.4f},{v['ci_95'][1]:+.4f}] "
              f"n={v['n']} нужно ~{v['n_needed_for_80pct_power']}")
    print(f"УСТОЙЧИВОСТЬ ВЫБОРА ПОРТФЕЛЯ (split-half): "
          f"{out['portfolio_choice_stability']['split_half_agreement']:.1%}")
    fn = out["failure_nature"]
    print(f"ПРИРОДА ПРОВАЛОВ: PASS {fn['overall']['PASS']:.1%}, FAIL {fn['overall']['FAIL']:.1%}, "
          f"INAPPLICABLE {fn['overall']['INAPPLICABLE']:.1%}")
    for g, v in fn["by_group"].items():
        print(f"    {g:13s} FAIL {v['FAIL']:.1%}  INAPPLICABLE {v['INAPPLICABLE']:.1%}")
    cs = out["cost_structure_lazy_portfolio"]
    print(f"СТОИМОСТЬ: {cs['tokens_per_resolved']:.1f} ток/решённую; без безнадёжных "
          f"{cs['tokens_per_resolved_if_hopeless_skipped']:.1f} (E={cs['free_efficiency_gain']:.3f} бесплатно)")
    print(f"\nсохранено -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
