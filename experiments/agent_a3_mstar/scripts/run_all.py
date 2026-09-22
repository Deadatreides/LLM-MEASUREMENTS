"""run_all.py — Единая точка входа пакета A3.

Порядок (PROTOCOL.md §1): verify_baselines -> 10 прогонов (5 p_cross x 2 seed)
-> metrics/curve_m.json (J(p_cross), m*, плато, SNR) -> reports/REPORT_A3.md.

Живой прогресс по ходу (время/RSS на каждый прогон) -- та же наблюдаемость,
что честная перепроверка A2 (psutil, не апостериорная оценка).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent_a3_mstar"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import psutil                                   # noqa: E402

import scripts.verify_baselines as verify_b      # noqa: E402
from src.orchestrator import run_cell_experiment  # noqa: E402

P_CROSS_LEVELS = (0.05, 0.15, 0.30, 0.45, 0.60)
SEEDS = (20261001, 20261002)

RUNS_DIR = AGENT_DIR / "runs"
METRICS_DIR = AGENT_DIR / "metrics"
REPORTS_DIR = AGENT_DIR / "reports"

LAMBDA = 0.05
N_REF = 30
J_TIE_EPS = 0.005
PLATEAU_EPS = 0.01
PLATEAU_FROM = 0.30
SNR_MIN_DIFF = 0.5


def compute_j(mean_p_ab: float, mean_n_nodes: float) -> float:
    """PROTOCOL.md §6.1: J = mean(p_A^B) - lambda * mean(n_nodes) / n_ref."""
    return mean_p_ab - LAMBDA * mean_n_nodes / N_REF


def select_m_star(level_stats: dict) -> dict:
    """PROTOCOL.md §6.1: m* = argmax J, тай-брейк к меньшему p_cross при ΔJ<0.005."""
    ordered = sorted(level_stats.items(), key=lambda kv: kv[0])
    best_level, best_j = None, float("-inf")
    for p_cross, stats in ordered:
        j = stats["J"]
        if best_level is None or j > best_j + J_TIE_EPS:
            best_level, best_j = p_cross, j
        elif abs(j - best_j) <= J_TIE_EPS and p_cross < best_level:
            best_level, best_j = p_cross, max(j, best_j)
    return {"p_cross_star": best_level, "J_star": level_stats[best_level]["J"]}


def check_plateau(level_stats: dict) -> dict:
    """PROTOCOL.md §6.2: рост p_A^B < 0.01 при p_cross>=0.30 -> плато."""
    ordered = sorted(level_stats.items(), key=lambda kv: kv[0])
    deltas = []
    plateau = True
    for i in range(1, len(ordered)):
        p_prev, s_prev = ordered[i - 1]
        p_cur, s_cur = ordered[i]
        if p_cur < PLATEAU_FROM:
            continue
        delta = s_cur["mean_p_A_and_B"] - s_prev["mean_p_A_and_B"]
        deltas.append({"from": p_prev, "to": p_cur, "delta_p_A_and_B": round(delta, 4)})
        if delta >= PLATEAU_EPS:
            plateau = False
    return {"plateau": plateau, "deltas_from_030": deltas}


def check_snr(level_stats: dict, p_cross_star: float) -> dict:
    """PROTOCOL.md §6.3: archive явно выше snap на m* -> SNR-эффект признан."""
    s = level_stats[p_cross_star]
    diff = s["v3_archive_rate"] - s["v3_snap_rate"]
    snr_limit = diff >= SNR_MIN_DIFF or (s["v3_snap_rate"] == 0 and s["v3_archive_rate"] > 0)
    return {
        "snr_limit": snr_limit, "v3_archive_rate": s["v3_archive_rate"],
        "v3_snap_rate": s["v3_snap_rate"], "diff": round(diff, 4),
    }


def main() -> int:
    print("=== ПАКЕТ A3: СТАРТ ===")

    print("\n--- Шаг 1: verify_baselines ---")
    ret = verify_b.main()
    if ret != 0:
        print("[run_all] ОШИБКА: verify_baselines не пройден -- СТОП.")
        return 1

    print("\n--- Шаг 2: 10 прогонов (5 p_cross x 2 seed) ---")
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
            print(f"[p_cross={p_cross:.2f} seed={seed}] готово за {elapsed:.1f}с | "
                  f"RSS после={rss_after:.1f} MB | n_gen={res['n_generations_run']} "
                  f"extinct={res['extinct']} | best_r_snap={res['best_r_snap']:.3f} "
                  f"best_r_archive={res['best_r_archive']:.3f} | "
                  f"V1/2/3_archive={res['v1_archive']}/{res['v2_archive']}/{res['v3_archive']} | "
                  f"V1/2/3_snap={res['v1_snap']}/{res['v2_snap']}/{res['v3_snap']} | "
                  f"p_A^B={res['p_A_and_B_archive']:.3f} Cov_a={res['coverage_atoms']:.3f} "
                  f"Cov_p={res['coverage_pairs']:.3f}", flush=True)

    overall_elapsed = time.time() - overall_t0
    print(f"\n[run_all] все 10 прогонов завершены за {overall_elapsed:.1f}с")

    print("\n--- Шаг 3: metrics/summary.json ---")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"elapsed_seconds": round(overall_elapsed, 2), "all_runs": all_runs}, f,
                  ensure_ascii=False, indent=2)
    print(f"[run_all] записан -> {METRICS_DIR / 'summary.json'}")

    print("\n--- Шаг 4: metrics/curve_m.json (J(p_cross), m*, плато, SNR) ---")
    level_stats = {}
    for p_cross in P_CROSS_LEVELS:
        runs = [r for r in all_runs if r["p_cross"] == p_cross]
        n = len(runs)
        mean_p_ab = sum(r["p_A_and_B_archive"] for r in runs) / n
        mean_nodes = sum(r["mean_n_nodes_elite"] for r in runs) / n
        level_stats[p_cross] = {
            "n_seeds": n,
            "mean_p_A_and_B": round(mean_p_ab, 4),
            "mean_p_A": round(sum(r["p_A_archive"] for r in runs) / n, 4),
            "mean_p_B": round(sum(r["p_B_archive"] for r in runs) / n, 4),
            "mean_coverage_atoms": round(sum(r["coverage_atoms"] for r in runs) / n, 4),
            "mean_coverage_pairs": round(sum(r["coverage_pairs"] for r in runs) / n, 4),
            "mean_n_nodes_elite": round(mean_nodes, 4),
            "mean_best_r_archive": round(sum(r["best_r_archive"] for r in runs) / n, 4),
            "mean_best_c_archive": round(sum(r["best_c_archive"] for r in runs) / n, 4),
            "mean_best_r_snap": round(sum(r["best_r_snap"] for r in runs) / n, 4),
            "v1_archive_rate": round(sum(r["v1_archive"] for r in runs) / n, 4),
            "v2_archive_rate": round(sum(r["v2_archive"] for r in runs) / n, 4),
            "v3_archive_rate": round(sum(r["v3_archive"] for r in runs) / n, 4),
            "v1_snap_rate": round(sum(r["v1_snap"] for r in runs) / n, 4),
            "v2_snap_rate": round(sum(r["v2_snap"] for r in runs) / n, 4),
            "v3_snap_rate": round(sum(r["v3_snap"] for r in runs) / n, 4),
            "J": round(compute_j(mean_p_ab, mean_nodes), 6),
        }

    m_star = select_m_star(level_stats)
    plateau = check_plateau(level_stats)
    snr = check_snr(level_stats, m_star["p_cross_star"])
    v3_verdict = "success" if level_stats[m_star["p_cross_star"]]["v3_archive_rate"] > 0 else "failure"

    curve_m = {
        "p_cross_levels": list(P_CROSS_LEVELS), "seeds": list(SEEDS),
        "level_stats": {str(k): v for k, v in level_stats.items()},
        "m_star": m_star, "plateau": plateau, "snr": snr,
        "v3_at_m_star": v3_verdict,
    }
    with open(METRICS_DIR / "curve_m.json", "w", encoding="utf-8") as f:
        json.dump(curve_m, f, ensure_ascii=False, indent=2)
    print(f"[run_all] m* = {m_star['p_cross_star']}  (J={m_star['J_star']:.6f})")
    print(f"[run_all] плато: {plateau['plateau']}")
    print(f"[run_all] SNR-эффект: {snr['snr_limit']} (archive={snr['v3_archive_rate']:.2f} vs "
          f"snap={snr['v3_snap_rate']:.2f})")
    print(f"[run_all] V3 на m*: {v3_verdict}")
    print(f"[run_all] записан -> {METRICS_DIR / 'curve_m.json'}")

    print("\n=== ПАКЕТ A3: ПРОГОН ЗАВЕРШЁН (отчёт пишется отдельно) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
