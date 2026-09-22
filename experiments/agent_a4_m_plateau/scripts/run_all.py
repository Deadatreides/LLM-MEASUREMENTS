"""run_all.py — Единая точка входа пакета A4.

Порядок (PROTOCOL.md §1): verify_baselines -> 12 прогонов (4 p_cross x 3 seed,
archive-only claims) -> metrics/curve_m.json (J(p_cross), m*, плато, V3_HIT,
корзина вердикта) -> reports/REPORT_A4.md.

Живой прогресс по ходу (время/RSS на каждый прогон), тот же стиль, что A3.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent_a4_m_plateau"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import psutil                                   # noqa: E402

import scripts.verify_baselines as verify_b      # noqa: E402
from src.orchestrator import run_cell_experiment  # noqa: E402

P_CROSS_LEVELS = (0.50, 0.65, 0.80, 0.90)
SEEDS = (20261101, 20261102, 20261103)

RUNS_DIR = AGENT_DIR / "runs"
METRICS_DIR = AGENT_DIR / "metrics"

LAMBDA = 0.05
N_REF = 30
M_TIE_EPS = 0.005
PLATEAU_P_AB_EPS = 0.015
PLATEAU_J_EPS = 0.01
V3_HIT_MIN_SEEDS = 2   # PROTOCOL.md §6.3: >=2 из 3, не "хотя бы 1"


def compute_j(mean_p_ab: float, mean_nodes: float) -> float:
    """PROTOCOL.md §5: J = mean(p_A^B) - lambda * mean(nodes) / n_ref.
    "nodes" = best_nodes_archive (archive-only, не снимковая элита -- та же
    дисциплина "только archive решает", что и остальные метрики пакета)."""
    return mean_p_ab - LAMBDA * mean_nodes / N_REF


def select_m_star(level_stats: dict) -> dict:
    """PROTOCOL.md §6.1: m* = argmax mean(J), тай-брейк к меньшему p_cross
    при Δmean(J) < 0.005."""
    ordered = sorted(level_stats.items(), key=lambda kv: kv[0])
    best_level, best_j = None, float("-inf")
    for p_cross, stats in ordered:
        j = stats["J"]
        if best_level is None or j > best_j + M_TIE_EPS:
            best_level, best_j = p_cross, j
        elif abs(j - best_j) <= M_TIE_EPS and p_cross < best_level:
            best_level, best_j = p_cross, max(j, best_j)
    return {"p_cross_star": best_level, "J_star": level_stats[best_level]["J"]}


def check_plateau(level_stats: dict) -> dict:
    """PROTOCOL.md §6.2: на ДВУХ соседних ВЕРХНИХ уровнях одновременно
    Δp_A^B < 0.015 И ΔJ < 0.01 -> PLATEAU=yes. "Верхние" = два наибольших
    уровня сетки (0.80->0.90 для {0.50,0.65,0.80,0.90})."""
    ordered = sorted(level_stats.items(), key=lambda kv: kv[0])
    p_top1, s_top1 = ordered[-2]
    p_top2, s_top2 = ordered[-1]
    delta_p_ab = s_top2["mean_p_A_and_B"] - s_top1["mean_p_A_and_B"]
    delta_j = s_top2["J"] - s_top1["J"]
    plateau = delta_p_ab < PLATEAU_P_AB_EPS and delta_j < PLATEAU_J_EPS
    return {
        "plateau": plateau, "top_levels": [p_top1, p_top2],
        "delta_p_A_and_B": round(delta_p_ab, 4), "delta_J": round(delta_j, 6),
    }


def compute_v3_hit(all_runs: list) -> dict:
    """PROTOCOL.md §6.3: V3_archive истинен у >=2 из 3 seed на ОДНОМ уровне."""
    by_level: dict = {}
    for r in all_runs:
        by_level.setdefault(r["p_cross_nominal"], []).append(r["v3_archive"])
    hits = {p: sum(v) for p, v in by_level.items()}
    hit_levels = [p for p, n in hits.items() if n >= V3_HIT_MIN_SEEDS]
    return {"hits_per_level": hits, "v3_hit": bool(hit_levels), "hit_levels": sorted(hit_levels)}


def select_verdict_basket(plateau: dict, v3: dict, level_stats: dict) -> dict:
    """PROTOCOL.md §6.4: приоритет V3_ARCHIVE_HIT > M_PLATEAU_V3_FAIL > M_STILL_RISING."""
    if v3["v3_hit"]:
        basket = "V3_ARCHIVE_HIT"
    else:
        top_levels = plateau["top_levels"]
        best_r_on_plateau = max(level_stats[p]["mean_best_r_archive"] for p in top_levels)
        if plateau["plateau"] and best_r_on_plateau < 0.70:
            basket = "M_PLATEAU_V3_FAIL"
        else:
            basket = "M_STILL_RISING"
    return {"basket": basket}


def main() -> int:
    print("=== ПАКЕТ A4: СТАРТ ===")

    print("\n--- Шаг 1: verify_baselines ---")
    ret = verify_b.main()
    if ret != 0:
        print("[run_all] ОШИБКА: verify_baselines не пройден -- СТОП.")
        return 1

    print("\n--- Шаг 2: 12 прогонов (4 p_cross x 3 seed) ---")
    all_runs = []
    proc = psutil.Process()
    overall_t0 = time.time()

    for p_cross in P_CROSS_LEVELS:
        for seed in SEEDS:
            t0 = time.time()
            rss_before = proc.memory_info().rss / (1024 * 1024)
            print(f"[p_cross={p_cross:.2f} seed={seed}] старт, RSS до={rss_before:.1f} MB ...",
                  flush=True)

            res = run_cell_experiment(p_cross, seed, RUNS_DIR)

            elapsed = time.time() - t0
            rss_after = proc.memory_info().rss / (1024 * 1024)
            all_runs.append(res)
            print(f"[p_cross={p_cross:.2f}(eff={res['p_cross_effective']:.2f}) seed={seed}] "
                  f"готово за {elapsed:.1f}с | RSS после={rss_after:.1f} MB | "
                  f"n_gen={res['n_generations_run']} extinct={res['extinct']} | "
                  f"best_r_arch={res['best_r_archive']:.3f} best_c_arch={res['best_c_archive']:.1f} "
                  f"nodes={res['best_nodes_archive']} | "
                  f"V1/2/3_archive={res['v1_archive']}/{res['v2_archive']}/{res['v3_archive']} | "
                  f"p_A^B={res['p_A_and_B_archive']:.3f} Cov_a={res['coverage_atoms']:.3f} "
                  f"Cov_p={res['coverage_pairs']:.3f} | "
                  f"[для истории] V3_snap={res['v3_snap']}", flush=True)

    overall_elapsed = time.time() - overall_t0
    print(f"\n[run_all] все 12 прогонов завершены за {overall_elapsed:.1f}с")

    print("\n--- Шаг 3: metrics/summary.json ---")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"elapsed_seconds": round(overall_elapsed, 2), "all_runs": all_runs}, f,
                  ensure_ascii=False, indent=2)
    print(f"[run_all] записан -> {METRICS_DIR / 'summary.json'}")

    print("\n--- Шаг 4: metrics/curve_m.json (J(p_cross), m*, плато, V3_HIT, корзина) ---")
    level_stats = {}
    for p_cross in P_CROSS_LEVELS:
        runs = [r for r in all_runs if r["p_cross_nominal"] == p_cross]
        n = len(runs)
        mean_p_ab = sum(r["p_A_and_B_archive"] for r in runs) / n
        mean_nodes = sum(r["best_nodes_archive"] for r in runs) / n
        level_stats[p_cross] = {
            "n_seeds": n, "p_cross_effective": runs[0]["p_cross_effective"],
            "mean_p_A_and_B": round(mean_p_ab, 4),
            "mean_p_A": round(sum(r["p_A_archive"] for r in runs) / n, 4),
            "mean_p_B": round(sum(r["p_B_archive"] for r in runs) / n, 4),
            "mean_coverage_atoms": round(sum(r["coverage_atoms"] for r in runs) / n, 4),
            "mean_coverage_pairs": round(sum(r["coverage_pairs"] for r in runs) / n, 4),
            "mean_best_nodes": round(mean_nodes, 4),
            "mean_best_r_archive": round(sum(r["best_r_archive"] for r in runs) / n, 4),
            "mean_best_c_archive": round(sum(r["best_c_archive"] for r in runs) / n, 4),
            "v1_archive_rate": round(sum(r["v1_archive"] for r in runs) / n, 4),
            "v2_archive_rate": round(sum(r["v2_archive"] for r in runs) / n, 4),
            "v3_archive_rate": round(sum(r["v3_archive"] for r in runs) / n, 4),
            "mean_best_r_snap": round(sum(r["best_r_snap"] for r in runs) / n, 4),
            "v3_snap_rate": round(sum(r["v3_snap"] for r in runs) / n, 4),
            "J": round(compute_j(mean_p_ab, mean_nodes), 6),
        }

    m_star = select_m_star(level_stats)
    plateau = check_plateau(level_stats)
    v3 = compute_v3_hit(all_runs)
    verdict = select_verdict_basket(plateau, v3, level_stats)

    curve_m = {
        "p_cross_levels": list(P_CROSS_LEVELS), "seeds": list(SEEDS),
        "level_stats": {str(k): v for k, v in level_stats.items()},
        "m_star": m_star, "plateau": plateau, "v3_hit": v3, "verdict": verdict,
    }
    with open(METRICS_DIR / "curve_m.json", "w", encoding="utf-8") as f:
        json.dump(curve_m, f, ensure_ascii=False, indent=2)

    print(f"[run_all] m* = {m_star['p_cross_star']}  (J={m_star['J_star']:.6f})")
    print(f"[run_all] плато (верхние уровни {plateau['top_levels']}): {plateau['plateau']} "
          f"(Δp_A^B={plateau['delta_p_A_and_B']:.4f}, ΔJ={plateau['delta_J']:.6f})")
    print(f"[run_all] V3_HIT: {v3['v3_hit']}  (уровни-попадания: {v3['hit_levels']}, "
          f"счёт по уровням: {v3['hits_per_level']})")
    print(f"\n[run_all] ВЕРДИКТ-КОРЗИНА: {verdict['basket']}")
    print(f"[run_all] записан -> {METRICS_DIR / 'curve_m.json'}")

    print("\n=== ПАКЕТ A4: ПРОГОН ЗАВЕРШЁН (отчёт пишется отдельно) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
