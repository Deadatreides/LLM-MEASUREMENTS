"""verdict.py — PROTOCOL.md §8: ровно одна корзина LIVE_ROUTE_WINS /
LIVE_PARTIAL / LIVE_NULL, механически, по TEST, агрегат по 5 seed. Плюс
обязательная строка про B3 (не корзина) с запретом overclaim, если CI
включает 0.
"""

from __future__ import annotations


def compute_basket(per_seed: list) -> dict:
    """`per_seed` — список результатов `orchestrator.run_seed_campaign` (по
    одному на seed). `top3[0]` каждого seed'а — "best_complex" этого seed'а
    (топ архива по full-TRAIN r, тай-брейк по c) — единственный кандидат от
    каждого seed'а для формулы PROTOCOL.md §8 ("best_complex" в единственном
    числе за seed)."""
    n_seeds = len(per_seed)
    wins_b1 = 0
    partial_b1 = 0
    b2_ok = 0
    seed_rows = []

    for res in per_seed:
        top3 = res.get("top3") or []
        best = top3[0] if top3 else None
        if best is None:
            seed_rows.append({"seed": res["seed"], "best_complex": None,
                              "win_b1": False, "partial_b1": False, "b2_ok": False})
            continue
        d = best["delta_r_vs_b1"]
        point, ci = d["point"], d["ci_95"]
        win = point > 0 and ci[0] >= 0
        partial = point > 0 and not win
        b2_ok_seed = best["r_test"] >= (res["b2_r_test"] - 0.02)
        wins_b1 += int(win)
        partial_b1 += int(partial)
        b2_ok += int(b2_ok_seed)
        seed_rows.append({
            "seed": res["seed"], "best_complex": best["complex_id"],
            "win_b1": win, "partial_b1": partial, "b2_ok": b2_ok_seed,
            "delta_r_vs_b1_point": point, "delta_r_vs_b1_ci95": ci,
            "r_test": best["r_test"], "b2_r_test": res["b2_r_test"],
        })

    route_wins = wins_b1 >= 4 and b2_ok >= 4
    live_partial = (not route_wins) and (wins_b1 + partial_b1) >= 3
    basket = "LIVE_ROUTE_WINS" if route_wins else ("LIVE_PARTIAL" if live_partial else "LIVE_NULL")

    # -- обязательная строка про B3 (не корзина, PROTOCOL.md §8) --
    b3_points, b3_cis = [], []
    for res in per_seed:
        top3 = res.get("top3") or []
        if top3:
            d3 = top3[0]["delta_r_vs_b3"]
            b3_points.append(d3["point"])
            b3_cis.append(d3["ci_95"])
    mean_delta_r_vs_b3 = sum(b3_points) / len(b3_points) if b3_points else None
    # Консервативно: claim "beat B3" допустим только если ВСЕ 5 seed по
    # отдельности показывают CI95, целиком исключающий ноль в плюс -- не
    # достаточно, чтобы среднее было положительным.
    beat_b3_claim_allowed = bool(b3_cis) and all(ci[0] > 0 for ci in b3_cis)

    return {
        "basket": basket, "n_seeds": n_seeds,
        "n_wins_b1_of_5": wins_b1, "n_partial_b1_of_5": partial_b1, "n_b2_ok_of_5": b2_ok,
        "seed_rows": seed_rows,
        "mean_delta_r_vs_b3": mean_delta_r_vs_b3,
        "delta_r_vs_b3_per_seed": list(zip([r["seed"] for r in per_seed], b3_points, b3_cis)),
        "beat_b3_claim_allowed": beat_b3_claim_allowed,
    }
