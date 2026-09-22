"""verify_baselines.py — Проверка базовых линий B1, B2, B3 на HETEROSTEP train grid."""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем корень репозитория и arch2 в sys.path для импорта
ROOT = Path(__file__).resolve().parents[2]
ARCH2 = ROOT / "arch2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ARCH2) not in sys.path:
    sys.path.insert(0, str(ARCH2))

import heterostep as HSTEP
import heterostep_seeds as HSEED
import runner as R
import genotype as G

EXPECTED = {
    "REF_B1_route": 0.4600,
    "REF_B2_greedy2": 0.5800,
    "REF_B3_greedy_cover": 0.7000,
}

TOLERANCE = 1e-4

def main() -> int:
    ds = HSTEP.default_dataset()
    reg = HSTEP.default_registry()
    be = HSTEP.HeterostepBackend(ds)
    train_ids = ds.split["train"]

    cov = ds.coverage(train_ids)
    print(f"[verify_baselines] Покрытие train-сетки: {cov['found']}/{cov['total']} = {cov['coverage']:.4f}")
    if cov["coverage"] < 1.0:
        print("[verify_baselines] ОШИБКА: Покрытие сетки < 1.0")
        return 1

    seeds = HSEED.seed_complexes(ds, train_ids)
    results = {}
    passed = True

    print("\n[verify_baselines] Сверка эталонов B1/B2/B3 vs a0:")
    print("  Эталон                   Ожидаемый r   Измеренный r   Расхождение   Статус")
    for name in HSEED.REFERENCE_NAMES:
        g = seeds[name]
        valid = G.validate(g, reg) == []
        n_resolved = 0
        for tid in train_ids:
            res = R.run(g, ds.initial_state(tid), reg, be, budget_tokens=3000)
            if res.outcome == R.RESOLVED:
                n_resolved += 1
        r = n_resolved / len(train_ids)
        exp_r = EXPECTED[name]
        diff = abs(r - exp_r)
        status = "OK" if diff <= TOLERANCE and valid else "FAIL"
        if status != "OK":
            passed = False
        print(f"  {name:23s}  {exp_r:.4f}        {r:.4f}         {diff:.6f}      {status}")
        results[name] = {"r": r, "expected_r": exp_r, "diff": diff, "valid": valid}

    if not passed:
        print("\n[verify_baselines] СТОП: Нарушен допуск по базовым линиям (>1e-4)!")
        return 1

    print("\n[verify_baselines] УСПЕХ: Все базовые линии сошлись с точностью 1e-4.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
