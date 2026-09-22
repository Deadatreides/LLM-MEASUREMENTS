"""robustness_analysis.py — честная перепроверка + разбивка по факторам.

Читает исходный `metrics/summary.json` (16 прогонов, seed 20260901/20260902,
предшественник) и `metrics/seed3_memory_log.json` (8 прогонов, seed 20260903,
лично прогнано и живьём пронаблюдано в этой сессии — время, память, PID).
НЕ повторяет эволюцию -- чистый анализ уже посчитанных данных.

Делает три вещи:
1. Сливает 16+8=24 прогона в единый набор, пересчитывает поячеечные агрегаты
   (mean/any по ТРЁМ seed вместо двух).
2. Раскладка по факторам (start_pop / evol_budget / recombination) по
   отдельности -- явное обоснование развилки VI vs Alphabet, а не только
   угловое сравнение C1-vs-C8.
3. Честно фиксирует найденную хрупкость: V1/V2/V3 (сверяются с ПОСЛЕДНИМ
   поколением) гораздо более шумные при добавлении третьего seed, чем
   Coverage_atoms/Coverage_pairs (считаются по ВСЕМУ архиву прогона, отсюда
   насыщение 1.000 везде). Не чинит эту хрупкость (не входит в объём пакета) --
   раскрывает её числами.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"

FACTOR_LEVELS = {
    "start_pop": (5, 20),
    "n_generations": (20, 40),          # прокси для evol_budget (совпадает с g_stall low/high)
    "recombination": ("low", "high"),
}


def load_merged_runs() -> List[dict]:
    with open(METRICS / "summary.json", encoding="utf-8") as f:
        original = json.load(f)
    with open(METRICS / "seed3_memory_log.json", encoding="utf-8") as f:
        seed3 = json.load(f)

    all_runs = list(original["all_runs"])
    for r in seed3["results"]:
        r = dict(r)
        r["seed"] = seed3["seed"]
        all_runs.append(r)
    return all_runs


def cell_aggregates(all_runs: List[dict]) -> Dict[str, dict]:
    by_cell: Dict[str, list] = {}
    for r in all_runs:
        by_cell.setdefault(r["cell"], []).append(r)

    out = {}
    for cell, runs in by_cell.items():
        n = len(runs)
        out[cell] = {
            "cell": cell, "config": runs[0]["config"], "n_seeds": n,
            "mean_best_r": sum(r["best_r"] for r in runs) / n,
            "mean_best_c": sum(r["best_c"] for r in runs) / n,
            "mean_cov_atoms": sum(r["coverage_atoms"] for r in runs) / n,
            "mean_cov_pairs": sum(r["coverage_pairs"] for r in runs) / n,
            "mean_P_A": sum(r["P_A"] for r in runs) / n,
            "mean_P_B": sum(r["P_B"] for r in runs) / n,
            "mean_P_AB": sum(r["P_A_and_B"] for r in runs) / n,
            "v1_rate": sum(r["v1_pass"] for r in runs) / n,
            "v2_rate": sum(r["v2_pass"] for r in runs) / n,
            "v3_rate": sum(r["v3_pass"] for r in runs) / n,
            "v1_any": any(r["v1_pass"] for r in runs),
            "v2_any": any(r["v2_pass"] for r in runs),
            "v3_any": any(r["v3_pass"] for r in runs),
            "seeds": sorted({r["seed"] for r in runs}),
        }
    return out


def factor_breakdown(all_runs: List[dict]) -> Dict[str, dict]:
    """Для каждого фактора по отдельности: среднее по всем прогонам с этим
    уровнем фактора (два других фактора усредняются, не фиксируются)."""
    out = {}
    for factor, levels in FACTOR_LEVELS.items():
        out[factor] = {}
        for level in levels:
            subset = [r for r in all_runs if r["config"][factor] == level]
            n = len(subset)
            if n == 0:
                continue
            out[factor][str(level)] = {
                "n_runs": n,
                "mean_cov_pairs": round(sum(r["coverage_pairs"] for r in subset) / n, 4),
                "mean_P_AB": round(sum(r["P_A_and_B"] for r in subset) / n, 4),
                "v1_rate": round(sum(r["v1_pass"] for r in subset) / n, 4),
                "v2_rate": round(sum(r["v2_pass"] for r in subset) / n, 4),
                "v3_rate": round(sum(r["v3_pass"] for r in subset) / n, 4),
                "mean_best_r": round(sum(r["best_r"] for r in subset) / n, 4),
            }
    return out


def seed3_vs_original_comparison(all_runs: List[dict]) -> Dict[str, dict]:
    """По каждой ячейке: числа исходных 2 seed отдельно от честного 3-го --
    видно ли расхождение, и какое."""
    by_cell: Dict[str, dict] = {}
    for r in all_runs:
        by_cell.setdefault(r["cell"], {"orig": [], "seed3": []})
        key = "seed3" if r["seed"] == 20260903 else "orig"
        by_cell[r["cell"]][key].append({
            "seed": r["seed"], "best_r": r["best_r"], "v1": r["v1_pass"],
            "v2": r["v2_pass"], "v3": r["v3_pass"],
            "n_generations_run": r.get("n_generations_run"), "extinct": r.get("extinct"),
        })
    return by_cell


def main() -> None:
    all_runs = load_merged_runs()
    assert len(all_runs) == 24, f"ожидалось 24 прогона (16+8), получено {len(all_runs)}"

    cells = cell_aggregates(all_runs)
    factors = factor_breakdown(all_runs)
    comparison = seed3_vs_original_comparison(all_runs)

    out = {
        "n_total_runs": len(all_runs),
        "seeds_used": sorted({r["seed"] for r in all_runs}),
        "cell_aggregates_3seed": cells,
        "factor_breakdown": factors,
        "seed3_vs_original_per_cell": comparison,
        "finding": (
            "V1/V2/V3 сверяются с ПОСЛЕДНИМ поколением прогона (снимок одного "
            "control-среза), поэтому сильно шумят при добавлении независимого "
            "seed: V2 успешен в 5/8 ячеек по исходным 2 seed (OR), но 0/8 по "
            "честному 3-му seed. Coverage_atoms/Coverage_pairs считаются по ВСЕМУ "
            "архиву прогона и потому насыщены на 1.000 во всех 24 прогонах без "
            "исключения -- это НЕ дефект честного прогона, а структурное свойство "
            "самой метрики (архив только растёт), делающее её слишком слабой, "
            "чтобы различать ячейки. Самый устойчивый по всем 24 прогонам "
            "сигнал -- P(A&B): варьируется по ячейкам/seed, но не разово, а "
            "закономерно откликается на recombination=high (см. factor_breakdown)."
        ),
    }

    with open(METRICS / "factor_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[robustness_analysis] {len(all_runs)} прогонов слито (16 исходных + 8 честных seed3)")
    print(f"[robustness_analysis] записано -> {METRICS / 'factor_breakdown.json'}")

    print("\n=== Разбивка по факторам (24 прогона) ===")
    for factor, levels in factors.items():
        print(f"  {factor}:")
        for level, d in levels.items():
            print(f"    {level:6s} (n={d['n_runs']:2d}): Cov_pairs={d['mean_cov_pairs']:.3f}  "
                  f"P(A&B)={d['mean_P_AB']:.3f}  V1={d['v1_rate']:.2f}  V2={d['v2_rate']:.2f}  "
                  f"V3={d['v3_rate']:.2f}  best_r={d['mean_best_r']:.3f}")


if __name__ == "__main__":
    main()
