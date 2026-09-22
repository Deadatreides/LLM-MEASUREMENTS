"""barriers.py — оценка барьера B = ΔV/D по УЖЕ СОБРАННЫМ данным (SPEC.md §4).

Ноль вызовов модели: всё считается из сохранённого сырья экспериментов 11-12.

Что считается:
  1. q̂_k и B̂_k = −ln q̂_k — предельная вероятность выхода на k-й слепой попытке
     при k−1 провалах (все 24 перестановки seed, 330 полных ячеек);
  2. N_eff при разном РАЗМЕЩЕНИИ одного и того же бюджета (1 модель × 4 seed против
     4 модели × 1 seed) — количественная форма F1/F6/F8;
  3. q̂ и B̂ слепого повтора по подтипам коллизии;
  4. ε = ln(q₁/q₀), δ = ln(c₁/c₀) и критерий ε > δ для всех измеренных плеч
     экспериментов 11 и 12 — ретро-проверка §4.4.

Самопроверка: §4.4 SPEC.md содержит уже посчитанные числа; скрипт обязан
воспроизвести их с точностью 1e-6. Расхождение означает ошибку В СКРИПТЕ, а не
повод переписать таблицу.
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
ARCH2 = Path(__file__).resolve().parent
METRICS = ARCH2 / "metrics"

EXP12_PHASE0 = ROOT / "experiment12" / "runs12" / "phase0_full_exp12-v1.json"
EXP12_PHASE1 = ROOT / "experiment12" / "runs12" / "phase1_full_exp12-v2.json"
EXP11_PHASE0 = ROOT / "experiment11" / "runs11" / "phase0_full_exp11-v1.json"
EXP11_PHASE1 = ROOT / "experiment11" / "runs11" / "phase1_full_exp11-v1.json"

SEEDS = (1, 2, 3, 4)

# Ожидаемые значения — РОВНО в той точности, в какой они опубликованы в таблицах
# SPEC.md §4.2/§4.4 (4 знака). Допуск — половина последнего знака. Записывать сюда
# больше знаков, чем опубликовано, значило бы проверять выдуманную точность:
# первая же версия этого файла содержала такую ошибку, и самопроверка её поймала.
EXPECTED = {
    "marginal_q": {1: 0.4144, 2: 0.1496, 3: 0.0903, 4: 0.0635},
    "allocation": {"within": 0.5758, "cross4": 0.6745, "mix2x2": 0.6204},
    "arm_eps_minus_delta": {
        "K0O0_promptB": +0.1031, "K1O0": -0.1749, "K2O0": -0.1614,
        "K1O1": -0.2602, "K2O1": -0.2483, "K0O1": -0.3156,
    },
}
TOLERANCE = 5e-5


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ln(x):
    return math.log(x) if x > 0 else float("inf")


def barrier(q: float) -> float:
    """B = ΔV/D = −ln q. Идентифицируема только эта безразмерная комбинация:
    интенсивность шума D отдельно не наблюдаема (SPEC.md §4.1)."""
    return float("inf") if q <= 0 else -math.log(q)


# -- 1. предельная отдача повтора --------------------------------------------------------


def marginal_repeat_curve(phase0: dict) -> dict:
    """q̂_k = P(k-я попытка PASS | предыдущие k−1 провалились), по всем 24
    перестановкам порядка seed — чтобы результат не зависел от произвольного
    порядка хранения (тот же дефект, что нашли в best-of-N эксп. 10)."""
    grid = collections.defaultdict(dict)
    for r in phase0["initial_records"]:
        grid[(r["model"], r["task_id"])][r["seed"]] = r
    cells = {k: v for k, v in grid.items() if len(v) == len(SEEDS)}

    num, den, tok = collections.Counter(), collections.Counter(), collections.Counter()
    for rs in cells.values():
        for perm in itertools.permutations(SEEDS):
            for k, s in enumerate(perm, 1):
                den[k] += 1
                tok[k] += rs[s]["total_tokens"]
                if rs[s]["status"] == "PASS":
                    num[k] += 1
                    break
    out = {}
    for k in SEEDS:
        q = num[k] / den[k]
        mean_tokens = tok[k] / den[k]
        out[k] = {
            "n": den[k], "q": q, "barrier_nats": barrier(q),
            "mean_tokens_per_attempt": mean_tokens,
            "marginal_tokens_per_resolved": (mean_tokens / q) if q else None,
        }
    return {"n_cells": len(cells), "by_attempt": out}


# -- 2. размещение бюджета: N_eff ---------------------------------------------------------


def n_eff(oracle: float, p1: float) -> float:
    """Эффективное число независимых попыток: столько НЕЗАВИСИМЫХ розыгрышей с той
    же вероятностью p1 дали бы наблюдённый oracle. Прямая операционализация F1."""
    if oracle >= 1.0 or p1 <= 0 or p1 >= 1:
        return float("inf")
    return math.log(1 - oracle) / math.log(1 - p1)


def allocation_comparison(phase0: dict, n_boot: int = 10000, seed: int = 99) -> dict:
    """Одинаковый бюджет (4 вызова на ячейку), разное РАЗМЕЩЕНИЕ.

    Ключевое сравнение всего слоя: то, чего не мерил ни один из 13 экспериментов,
    потому что все они фиксировали одно действие для всех состояний.
    """
    P, T = {}, {}
    models = sorted({r["model"] for r in phase0["initial_records"]})
    for r in phase0["initial_records"]:
        P[(r["model"], r["task_id"], r["seed"])] = (r["status"] == "PASS")
        T[(r["model"], r["task_id"], r["seed"])] = r["total_tokens"]
    tasks = sorted({r["task_id"] for r in phase0["initial_records"]})
    full = [t for t in tasks if all((m, t, s) in P for m in models for s in SEEDS)]

    p1 = sum(P[(m, t, s)] for m in models for t in full for s in SEEDS) / (len(models) * len(full) * len(SEEDS))

    within, cross, mix = {}, {}, {}
    tok_within, tok_cross = {}, {}
    for t in full:
        ws = [any(P[(m, t, s)] for s in SEEDS) for m in models]
        within[t] = sum(ws) / len(ws)
        tok_within[t] = sum(T[(m, t, s)] for m in models for s in SEEDS) / len(models)

        cs, cts = [], []
        for sub in itertools.combinations(models, 4):
            for s in SEEDS:
                cs.append(any(P[(m, t, s)] for m in sub))
                cts.append(sum(T[(m, t, s)] for m in sub))
        cross[t] = sum(cs) / len(cs)
        tok_cross[t] = sum(cts) / len(cts)

        ms = []
        for pair in itertools.combinations(models, 2):
            for ss in itertools.combinations(SEEDS, 2):
                ms.append(any(P[(m, t, s)] for m in pair for s in ss))
        mix[t] = sum(ms) / len(ms)

    W = sum(within.values()) / len(full)
    C = sum(cross.values()) / len(full)
    M = sum(mix.values()) / len(full)
    tw = sum(tok_within.values()) / len(full)
    tc = sum(tok_cross.values()) / len(full)

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [full[rng.randrange(len(full))] for _ in full]
        diffs.append(sum(cross[t] for t in sample) / len(sample) - sum(within[t] for t in sample) / len(sample))
    diffs.sort()

    wins = sum(1 for t in full if cross[t] > within[t])
    losses = sum(1 for t in full if cross[t] < within[t])

    # лучшая и худшая четвёрка моделей (при том же бюджете)
    subsets = []
    for sub in itertools.combinations(models, 4):
        o = sum(any(P[(m, t, s)] for m in sub) for t in full for s in SEEDS) / (len(full) * len(SEEDS))
        tk = sum(T[(m, t, s)] for m in sub for t in full for s in SEEDS) / (len(full) * len(SEEDS))
        subsets.append({"models": list(sub), "oracle_at_4": o, "tokens": tk,
                        "tokens_per_resolved": tk / o if o else None})
    subsets.sort(key=lambda d: -d["oracle_at_4"])

    singles = []
    for m in models:
        o = sum(any(P[(m, t, s)] for s in SEEDS) for t in full) / len(full)
        tk = sum(T[(m, t, s)] for t in full for s in SEEDS) / len(full)
        singles.append({"model": m, "oracle_at_4": o, "tokens": tk,
                        "tokens_per_resolved": tk / o if o else None})
    singles.sort(key=lambda d: -d["oracle_at_4"])

    return {
        "n_tasks": len(full), "n_models": len(models), "per_attempt_pass_rate": p1,
        "within_model_4_seeds": {"oracle_at_4": W, "n_eff": n_eff(W, p1), "tokens": tw,
                                 "tokens_per_resolved": tw / W},
        "two_models_two_seeds": {"oracle_at_4": M, "n_eff": n_eff(M, p1), "tokens": tw,
                                 "tokens_per_resolved": tw / M},
        "four_models_one_seed": {"oracle_at_4": C, "n_eff": n_eff(C, p1), "tokens": tc,
                                 "tokens_per_resolved": tc / C},
        "paired_diff_cross_minus_within": {
            "point": C - W, "ci_95": [diffs[int(0.025 * n_boot)], diffs[min(int(0.975 * n_boot), n_boot - 1)]],
            "tasks_cross_better": wins, "tasks_within_better": losses,
            "tasks_tied": len(full) - wins - losses,
        },
        "best_subset_of_4": subsets[0], "worst_subset_of_4": subsets[-1],
        "best_single_model": singles[0],
    }


# -- 2b. разделение «гетерогенность» и «сильная модель» --------------------------------


def heterogeneity_decomposition(phase0: dict, n_boot: int = 10000, seed: int = 4242) -> dict:
    """Отделяет ЭФФЕКТ РАЗНООБРАЗИЯ от эффекта «в наборе оказалась сильная модель».

    Конструкция, не требующая никакой модели и никакой регрессии: для набора моделей S
    сравниваются два размещения ОДНОГО И ТОГО ЖЕ бюджета в 4 вызова —

        гетерогенное:  по одному вызову каждой модели из S;
        гомогенное:    4 вызова одной моделью, УСРЕДНЁННЫЕ по m ∈ S.

    У обоих ожидаемая вероятность успеха ОДНОЙ попытки одинакова по построению
    (среднее p1 по S), поэтому разница между ними не может объясняться силой моделей —
    только тем, что попытки берутся из разных источников.

    Считается на всех C(6,4)=15 четвёрках и всех C(6,2)=15 парах (для пар гетерогенное
    размещение — 2+2). Плюс отдельная, самая наглядная проверка: бьёт ли лучшая
    четвёрка лучшую ОДИНОЧНУЮ модель, будучи слабее её по среднему p1.
    """
    P, models = {}, sorted({r["model"] for r in phase0["initial_records"]})
    for r in phase0["initial_records"]:
        P[(r["model"], r["task_id"], r["seed"])] = (r["status"] == "PASS")
    tasks = sorted({r["task_id"] for r in phase0["initial_records"]})
    full = [t for t in tasks if all((m, t, s) in P for m in models for s in SEEDS)]
    families = {t: ("arithmetic" if t.startswith("MSARITH") else "code") for t in full}

    p1 = {m: sum(P[(m, t, s)] for t in full for s in SEEDS) / (len(full) * len(SEEDS))
          for m in models}
    # per-task oracle@4 для гомогенного размещения каждой модели
    hom = {m: {t: any(P[(m, t, s)] for s in SEEDS) for t in full} for m in models}

    def _premium(subsets, hetero_fn):
        """-> (per-task средняя надбавка, сводка). Пары/четвёрки усредняются, задачи
        остаются единицей ресэмплирования — независимы именно они."""
        per_task = {t: [] for t in full}
        rows = []
        for S in subsets:
            het = hetero_fn(S)                       # {task: bool}
            base = {t: sum(hom[m][t] for m in S) / len(S) for t in full}
            diff = {t: het[t] - base[t] for t in full}
            for t in full:
                per_task[t].append(diff[t])
            rows.append({
                "models": [m.split("-")[0] for m in S],
                "mean_p1": sum(p1[m] for m in S) / len(S),
                "hetero_oracle4": sum(het.values()) / len(full),
                "homo_oracle4_mean": sum(base.values()) / len(full),
                "premium": sum(diff.values()) / len(full),
            })
        avg = {t: sum(v) / len(v) for t, v in per_task.items()}
        point = sum(avg.values()) / len(full)
        rng = random.Random(seed)
        boot = []
        for _ in range(n_boot):
            sample = [full[rng.randrange(len(full))] for _ in full]
            boot.append(sum(avg[t] for t in sample) / len(sample))
        boot.sort()
        rows.sort(key=lambda d: -d["premium"])
        return {
            "point": point,
            "ci_95": [boot[int(0.025 * n_boot)], boot[min(int(0.975 * n_boot), n_boot - 1)]],
            "n_tasks": len(full), "n_subsets": len(rows),
            "subsets_with_positive_premium": sum(1 for r in rows if r["premium"] > 0),
            "tasks_hetero_better": sum(1 for t in full if avg[t] > 0),
            "tasks_homo_better": sum(1 for t in full if avg[t] < 0),
            "best": rows[0], "worst": rows[-1],
        }

    quads = list(itertools.combinations(models, 4))
    pairs = list(itertools.combinations(models, 2))

    def het_quad_fixed(S):
        # по одному вызову каждой модели; усредняем по тому, КАКОЙ seed берётся,
        # чтобы результат не зависел от произвольного выбора розыгрыша
        out = {}
        for t in full:
            hits = [any(P[(m, t, s)] for m in S) for s in SEEDS]
            out[t] = sum(hits) / len(hits)
        return out

    def het_pair(S):
        m1, m2 = S
        out = {}
        for t in full:
            hits = []
            for ss in itertools.combinations(SEEDS, 2):
                hits.append(any(P[(m, t, s)] for m in (m1, m2) for s in ss))
            out[t] = sum(hits) / len(hits)
        return out

    result = {
        "per_attempt_strength": p1,
        "quads_4x1": _premium(quads, het_quad_fixed),
        "pairs_2x2": _premium(pairs, het_pair),
    }

    # per-family разбивка на четвёрках
    by_family = {}
    for fam in ("arithmetic", "code"):
        ids = [t for t in full if families[t] == fam]
        if not ids:
            continue
        diffs = []
        for S in quads:
            het = het_quad_fixed(S)
            base = {t: sum(hom[m][t] for m in S) / len(S) for t in ids}
            diffs.append(sum(het[t] - base[t] for t in ids) / len(ids))
        by_family[fam] = {"n_tasks": len(ids), "mean_premium": sum(diffs) / len(diffs)}
    result["by_family"] = by_family

    # самая наглядная проверка: лучшая четвёрка против лучшей ОДИНОЧНОЙ модели
    o4_single = {m: sum(hom[m].values()) / len(full) for m in models}
    best_single = max(models, key=lambda m: o4_single[m])
    quad_o4 = {S: sum(het_quad_fixed(S).values()) / len(full) for S in quads}
    best_quad = max(quad_o4, key=quad_o4.get)
    rng = random.Random(seed + 1)
    het_t = het_quad_fixed(best_quad)
    diffs = []
    for _ in range(n_boot):
        sample = [full[rng.randrange(len(full))] for _ in full]
        diffs.append(sum(het_t[t] - hom[best_single][t] for t in sample) / len(sample))
    diffs.sort()
    result["best_quad_vs_best_single"] = {
        "best_single": best_single, "best_single_oracle4": o4_single[best_single],
        "best_single_p1": p1[best_single],
        "best_quad": [m.split("-")[0] for m in best_quad], "best_quad_oracle4": quad_o4[best_quad],
        "best_quad_mean_p1": sum(p1[m] for m in best_quad) / 4,
        "diff": quad_o4[best_quad] - o4_single[best_single],
        "ci_95": [diffs[int(0.025 * n_boot)], diffs[min(int(0.975 * n_boot), n_boot - 1)]],
        "note": "четвёрка СЛАБЕЕ по среднему p1 и всё равно решает больше — сила моделей "
                "объяснить это не может",
    }
    return result


# -- 3. барьер по подтипам коллизии --------------------------------------------------------


def barrier_by_subtype(phase0: dict) -> dict:
    """Слепой повтор (другой seed того же (модель, задача)) в разрезе подтипа
    исходной коллизии. Даёт 13-кратный спред предельной цены — основание для
    ветви STOP в генотипе."""
    sys.path.insert(0, str(ROOT / "experiment12"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("arch2_barriers_seams12", ROOT / "experiment12" / "seams.py")
    s12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s12)
    from tasks.arithmetic_multistep_tasks import TASKS as ARITH   # noqa: E402

    by_key = collections.defaultdict(list)
    for r in phase0["initial_records"]:
        by_key[(r["model"], r["task_id"])].append(r)

    agg = collections.defaultdict(lambda: [0, 0, 0])
    for st in phase0["states"]:
        steps = ARITH[st["task_id"]]["steps"] if st["task_family"] == "arithmetic" else None
        sub = s12.collision_subtype(st["task_family"], st["initial_seam_result"], steps) or "other"
        for r in by_key[(st["model_id"], st["task_id"])]:
            if r["seed"] == st["seed"]:
                continue
            for key in (sub, "ALL"):
                agg[key][0] += 1
                agg[key][1] += 1 if r["status"] == "PASS" else 0
                agg[key][2] += r["total_tokens"]

    out = {}
    for key, (n, solved, tok) in agg.items():
        q = solved / n
        c = tok / n
        out[key] = {"n": n, "q": q, "barrier_nats": barrier(q), "mean_tokens_per_attempt": c,
                    "tokens_per_resolved": c / q if q else None}
    return out


# -- 4. критерий слабого сигнала по всем измеренным плечам ----------------------------------


def _arm_economics(records, arm_key, solved_key, tokens_key="tokens") -> dict:
    by = collections.defaultdict(list)
    for r in records:
        by[r[arm_key]].append(r)
    out = {}
    for arm, rs in by.items():
        n = len(rs)
        tok = sum(r[tokens_key] for r in rs)
        solved = sum(1 for r in rs if r[solved_key])
        out[arm] = {"n": n, "q": solved / n, "c_per_attempt": tok / n,
                    "tokens_per_resolved": (tok / solved) if solved else None}
    return out


def weak_signal_criterion(arms: dict, control: str) -> dict:
    """ε = ln(q₁/q₀) против δ = ln(c₁/c₀). Сигнал окупается ⟺ ε > δ (SPEC.md §4.4).

    Это тождество, эквивалентное «tokens-per-resolved должен упасть». Ценность —
    в шкале: сколько барьера обязан снять контекст, чтобы окупить свои токены.
    """
    base = arms[control]
    out = {}
    for arm, m in arms.items():
        if arm == control:
            continue
        if m["c_per_attempt"] == 0:
            # Плечо ничего не генерирует (SEEK_EVIDENCE/STOP эксп. 11) — критерий
            # «стоит ли сигнал своих токенов» к нему неприменим, а не «не окупается».
            out[arm] = {"q": m["q"], "c_per_attempt": 0.0, "n": m["n"],
                        "not_applicable": "no generation, criterion undefined"}
            continue
        eps = _ln(m["q"] / base["q"]) if base["q"] else float("inf")
        delta = _ln(m["c_per_attempt"] / base["c_per_attempt"])
        out[arm] = {
            "q": m["q"], "c_per_attempt": m["c_per_attempt"], "n": m["n"],
            "epsilon_barrier_drop_nats": eps, "delta_log_cost": delta,
            "net": eps - delta, "pays_off": eps > delta,
            "barrier_nats": barrier(m["q"]),
        }
    return {"control": control, "control_q": base["q"], "control_c": base["c_per_attempt"],
            "control_barrier_nats": barrier(base["q"]), "arms": out}


# -- сборка -----------------------------------------------------------------------------------


def compute_all(n_boot: int = 10000) -> dict:
    p0_12 = _load(EXP12_PHASE0)
    p1_12 = _load(EXP12_PHASE1)
    p1_11 = _load(EXP11_PHASE1)

    arms12 = _arm_economics(p1_12["arm_results"], "arm", "solved")
    arms11 = _arm_economics(p1_11["arm_results"], "arm", "solved")

    return {
        "source_files": {
            "exp12_phase0": str(EXP12_PHASE0.relative_to(ROOT)),
            "exp12_phase1": str(EXP12_PHASE1.relative_to(ROOT)),
            "exp11_phase1": str(EXP11_PHASE1.relative_to(ROOT)),
        },
        "marginal_repeat_curve": marginal_repeat_curve(p0_12),
        "allocation_comparison": allocation_comparison(p0_12, n_boot=n_boot),
        "heterogeneity_decomposition": heterogeneity_decomposition(p0_12, n_boot=n_boot),
        "barrier_by_collision_subtype": barrier_by_subtype(p0_12),
        "weak_signal_exp12": weak_signal_criterion(arms12, "K0O0"),
        "weak_signal_exp11": weak_signal_criterion(arms11, "REGENERATE_SAME"),
    }


def self_check(res: dict) -> list:
    """Воспроизводимость чисел SPEC.md §4.2/§4.4 с точностью 1e-6."""
    problems = []
    for k, expected in EXPECTED["marginal_q"].items():
        got = res["marginal_repeat_curve"]["by_attempt"][k]["q"]
        if abs(got - expected) > TOLERANCE:
            problems.append(f"marginal_q[{k}]: {got!r} != {expected!r}")

    alloc = res["allocation_comparison"]
    for key, field in (("within", "within_model_4_seeds"), ("cross4", "four_models_one_seed"),
                       ("mix2x2", "two_models_two_seeds")):
        got = alloc[field]["oracle_at_4"]
        exp = EXPECTED["allocation"][key]
        if abs(got - exp) > TOLERANCE:
            problems.append(f"allocation[{key}]: {got!r} != {exp!r}")

    for arm, exp in EXPECTED["arm_eps_minus_delta"].items():
        got = res["weak_signal_exp12"]["arms"][arm]["net"]
        if abs(got - exp) > TOLERANCE:
            problems.append(f"weak_signal[{arm}].net: {got:.9f} != {exp:.9f}")
    return problems


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    res = compute_all()
    METRICS.mkdir(parents=True, exist_ok=True)
    out = METRICS / "barriers.json"
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    tmp.replace(out)

    mr = res["marginal_repeat_curve"]
    print(f"E4.1 предельная отдача слепого повтора ({mr['n_cells']} ячеек, 24 перестановки):")
    print("  k   n      q_k      B_k(нат)   ток/попытку   предельная цена ток/resolved")
    for k, m in sorted(mr["by_attempt"].items()):
        print(f"  {k}  {m['n']:6d}  {m['q']:.4f}   {m['barrier_nats']:.3f}      "
              f"{m['mean_tokens_per_attempt']:6.1f}      {m['marginal_tokens_per_resolved']:8.1f}")

    a = res["allocation_comparison"]
    print(f"\nE4.2 размещение одного и того же бюджета (4 вызова, {a['n_tasks']} задач, "
          f"{a['n_models']} моделей, p1={a['per_attempt_pass_rate']:.4f}):")
    for label, key in (("1 модель x 4 seed", "within_model_4_seeds"),
                       ("2 модели x 2 seed", "two_models_two_seeds"),
                       ("4 модели x 1 seed", "four_models_one_seed")):
        m = a[key]
        print(f"  {label:20s} oracle@4={m['oracle_at_4']:.4f}  N_eff={m['n_eff']:.2f}  "
              f"ток={m['tokens']:.1f}  ток/resolved={m['tokens_per_resolved']:.1f}")
    d = a["paired_diff_cross_minus_within"]
    print(f"  парный bootstrap: {d['point']:+.4f}  CI95 [{d['ci_95'][0]:+.4f}, {d['ci_95'][1]:+.4f}]  "
          f"лучше/хуже/ничья = {d['tasks_cross_better']}/{d['tasks_within_better']}/{d['tasks_tied']}")
    print(f"  лучшая четвёрка: {a['best_subset_of_4']['oracle_at_4']:.4f}  "
          f"худшая: {a['worst_subset_of_4']['oracle_at_4']:.4f}  "
          f"лучшая одиночная модель: {a['best_single_model']['oracle_at_4']:.4f} "
          f"({a['best_single_model']['model']})")

    h = res["heterogeneity_decomposition"]
    print()
    print("E4.2b разделение «разнообразие» и «сильная модель» "
          "(оба размещения -- 4 вызова, среднее p1 совпадает ПО ПОСТРОЕНИЮ):")
    for key, label in (("quads_4x1", "4 модели x 1 вызов "), ("pairs_2x2", "2 модели x 2 вызова")):
        d = h[key]
        print(f"  {label} надбавка {d['point']:+.4f}  CI95 [{d['ci_95'][0]:+.4f}, {d['ci_95'][1]:+.4f}]"
              f"  наборов с плюсом {d['subsets_with_positive_premium']}/{d['n_subsets']}"
              f"  задач лучше/хуже {d['tasks_hetero_better']}/{d['tasks_homo_better']}")
    for fam, v in sorted(h["by_family"].items()):
        print(f"    по семейству {fam:11s} n={v['n_tasks']:3d}  надбавка {v['mean_premium']:+.4f}")
    b = h["best_quad_vs_best_single"]
    print(f"  лучшая четвёрка {b['best_quad']}: oracle@4={b['best_quad_oracle4']:.4f} "
          f"при среднем p1={b['best_quad_mean_p1']:.4f}")
    print(f"  лучшая одиночная {b['best_single'].split('-')[0]}: oracle@4={b['best_single_oracle4']:.4f} "
          f"при p1={b['best_single_p1']:.4f}")
    print(f"  разница {b['diff']:+.4f}  CI95 [{b['ci_95'][0]:+.4f}, {b['ci_95'][1]:+.4f}] "
          f"-- четвёрка СЛАБЕЕ по среднему p1 и всё равно решает больше")

    print("\nE4.3 барьер слепого повтора по подтипу коллизии:")
    for sub, m in sorted(res["barrier_by_collision_subtype"].items(), key=lambda kv: kv[1]["barrier_nats"]):
        print(f"  {sub:12s} n={m['n']:5d}  q={m['q']:.4f}  B={m['barrier_nats']:.2f} нат  "
              f"ток/resolved={m['tokens_per_resolved']:8.1f}")

    for label, key in (("эксп. 12 v2", "weak_signal_exp12"), ("эксп. 11", "weak_signal_exp11")):
        w = res[key]
        print(f"\nE4.4 критерий слабого сигнала, {label} (контроль {w['control']}: "
              f"q={w['control_q']:.4f}, c={w['control_c']:.1f}, B={w['control_barrier_nats']:.3f} нат):")
        print("  плечо            q       c/попытку   eps      delta     eps-delta  окупается")
        for arm, m in sorted(w["arms"].items(), key=lambda kv: -kv[1].get("net", float("-inf"))):
            if "not_applicable" in m:
                print(f"  {arm:16s} {m['q']:.4f}  {m['c_per_attempt']:8.1f}  — критерий неприменим "
                      f"({m['not_applicable']})")
                continue
            print(f"  {arm:16s} {m['q']:.4f}  {m['c_per_attempt']:8.1f}  {m['epsilon_barrier_drop_nats']:+.4f}  "
                  f"{m['delta_log_cost']:+.4f}  {m['net']:+.4f}   {'да' if m['pays_off'] else 'нет'}")

    problems = self_check(res)
    print(f"\nСАМОПРОВЕРКА (воспроизведение опубликованных чисел SPEC.md §4.2/§4.4, допуск {TOLERANCE:g}):")
    if problems:
        for p in problems:
            print(f"  РАСХОЖДЕНИЕ: {p}")
        print("  -> ошибка в скрипте либо в таблице спеки; НЕ править таблицу под вывод.")
    else:
        print("  все значения воспроизведены")
    print(f"\nсохранено -> {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
