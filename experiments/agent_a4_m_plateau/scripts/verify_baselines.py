"""verify_baselines.py — Проверка базовых линий B1, B2, B3 на HETEROSTEP train grid.

Независимая копия гейта `agent_a3_mstar/scripts/verify_baselines.py` (не
импорт -- пакеты агентов изолированы друг от друга), тот же принцип, что и
во всём проекте: |Δr| > 1e-4 -> СТОП, эталоны не подгоняются. Дополнительно
пишет `metrics/verify.json` (PROTOCOL.md §1, шаг 2 плана).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH2 = ROOT / "arch2"
AGENT_DIR = ROOT / "agent_a4_m_plateau"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ARCH2) not in sys.path:
    sys.path.insert(0, str(ARCH2))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import heterostep as HSTEP        # noqa: E402
import heterostep_seeds as HSEED  # noqa: E402
import runner as R                # noqa: E402
import genotype as G              # noqa: E402

EXPECTED = {
    "REF_B1_route": (0.4600, 667.4),
    "REF_B2_greedy2": (0.5800, 771.2),
    "REF_B3_greedy_cover": (0.7000, 917.4),
}

TOLERANCE = 1e-4


def main() -> int:
    ds = HSTEP.default_dataset()
    reg = HSTEP.default_registry()
    be = HSTEP.HeterostepBackend(ds)
    train_ids = ds.split["train"]

    cov = ds.coverage(train_ids)
    print(f"[verify_baselines] покрытие train-сетки: {cov['found']}/{cov['total']} = {cov['coverage']:.4f}")
    if cov["coverage"] < 1.0:
        print("[verify_baselines] ОШИБКА: покрытие сетки < 1.0 -- СТОП")
        return 1

    seeds = HSEED.seed_complexes(ds, train_ids)
    passed = True
    results = {}

    print("\n[verify_baselines] сверка эталонов B1/B2/B3:")
    print("  эталон                    ожид. r  измер. r  Δr        ожид. c  измер. c  статус")
    for name, (exp_r, exp_c) in EXPECTED.items():
        g = seeds[name]
        valid = G.validate(g, reg) == []
        n_resolved = 0
        total_cost = 0
        for tid in train_ids:
            res = R.run(g, ds.initial_state(tid), reg, be, budget_tokens=3000)
            if res.outcome == R.RESOLVED:
                n_resolved += 1
            total_cost += res.cost
        r = n_resolved / len(train_ids)
        c = total_cost / len(train_ids)
        diff = abs(r - exp_r)
        status = "OK" if diff <= TOLERANCE and valid else "FAIL"
        if status != "OK":
            passed = False
        results[name] = {"expected_r": exp_r, "measured_r": r, "diff": diff,
                         "expected_c": exp_c, "measured_c": c, "valid": valid, "status": status}
        print(f"  {name:24s}  {exp_r:.4f}   {r:.4f}   {diff:.6f}  {exp_c:7.1f}  {c:7.1f}  {status}")

    METRICS_DIR = AGENT_DIR / "metrics"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "verify.json", "w", encoding="utf-8") as f:
        json.dump({"coverage": cov, "results": results, "passed": passed}, f,
                  ensure_ascii=False, indent=2)

    if not passed:
        print("\n[verify_baselines] СТОП: нарушен допуск по базовым линиям (>1e-4) -- не подгонять сетку")
        return 1

    print(f"\n[verify_baselines] УСПЕХ: все базовые линии сошлись с точностью {TOLERANCE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
