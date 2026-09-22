"""verify_seams.py — PROTOCOL.md §4.1: три независимые проверки ДО любой
evolve-работы, ни одна не требует нового generate() (grid=reuse, LIFE-4
PROTOCOL.md):

1. Швы (arch2.heterostep/experiment14.seams self-test) -- код-корректность,
   не зависит от сетки.
2. Целостность СКОПИРОВАННОЙ сетки (row-counts + train floor) против
   `agent_life3_block_live`'s уже подтверждённых чисел -- байт-в-байт
   копия, не пересборка.
3. Фикстура разворота cx. (Change 1) -- РЕАЛЬНЫЙ случай из LIFE-3
   (`cx-068fae` слот 2 = `cx.cx-23f285`, обёртка над `cx.cx-b3b0bb`),
   прочитанный напрямую из `agent_life3_block_live/runs/life3_with_
   s20261601/archive.json` (read-only, изоляция пакетов это разрешает) --
   не переписан вручную, чтобы исключить транскрипционную ошибку.

Провал любой -> BLOCKERS.md, стоп, без отката (PROTOCOL.md §0).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402
import metrics_lib as ML    # noqa: E402
import call_log             # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = AGENT_DIR.parent
LIFE3_FIXTURE_ARCHIVE = (ROOT / "agent_life3_block_live" / "runs" /
                        "life3_with_s20261601" / "archive.json")

EXPECTED_TRAIN_ROWS = 2400
EXPECTED_TEST_ROWS = 2400
EXPECTED_WHOLE_ROWS = 700
TRAIN_FLOOR = 1920   # int(0.8 * 6 models * 4 kinds * 100 train tasks)


def verify_seams() -> bool:
    r1 = LD.seams14.self_test()
    r2 = LD.HSTEP.self_test_seam()
    ok = r1["passed"] and r2["passed"]
    print(f"[verify] experiment14.seams.self_test: {r1['n_cases']} cases, "
          f"{'PASS' if r1['passed'] else 'FAIL'}")
    for f in r1["failures"]:
        print("  -", f)
    print(f"[verify] arch2.heterostep.self_test_seam: {r2['n_cases']} cases, "
          f"{'PASS' if r2['passed'] else 'FAIL'}")
    for f in r2["failures"]:
        print("  -", f)
    return ok


def verify_grid_copy() -> bool:
    """Целостность копии (не пересборка). Провал здесь -> BLOCKED-fallback
    `live_grid_builder.py` (не автоматически -- см. BLOCKERS.md)."""
    ok = True
    for path, expected, label in (
        (LD.LIVE_TRAIN_GRID, EXPECTED_TRAIN_ROWS, "train_grid"),
        (LD.LIVE_TEST_GRID, EXPECTED_TEST_ROWS, "test_grid"),
        (LD.LIVE_WHOLE_GRID, EXPECTED_WHOLE_ROWS, "whole_grid"),
    ):
        if not path.exists():
            print(f"[verify] !!! {label} missing at {path}")
            ok = False
            continue
        with open(path, encoding="utf-8") as f:
            n = len(json.load(f)["rows"])
        row_ok = (n == expected)
        print(f"[verify] {label}: {n} rows (expected {expected}) "
              f"{'OK' if row_ok else 'MISMATCH'}")
        ok = ok and row_ok

    train_ids = set(LD.default_dataset().split["train"])
    n_train_log = call_log.count_rows_for_tasks(train_ids)
    floor_ok = n_train_log >= TRAIN_FLOOR
    print(f"[verify] TRAIN-строк лога: {n_train_log} (пол {TRAIN_FLOOR}) "
          f"{'ВЫПОЛНЕН' if floor_ok else 'НЕ ВЫПОЛНЕН'}")
    return ok and floor_ok


def verify_fixture_unfold() -> bool:
    """LIFE-4 Change 1 sanity: реальный зарегистрированный блок из LIFE-3
    (`cx.cx-23f285`, carrier `cx-068fae`, слот 2) должен после разворота
    показать 4 модели (`gemma-3-it-1b-q5_k_s` напрямую + 3 через
    `cx.cx-b3b0bb`), не 1 (старое, cx.-слепое поведение, см. BLOCKERS.md)."""
    if not LIFE3_FIXTURE_ARCHIVE.exists():
        print(f"[verify] !!! фикстура недоступна: {LIFE3_FIXTURE_ARCHIVE} не найден")
        return False
    with open(LIFE3_FIXTURE_ARCHIVE, encoding="utf-8") as f:
        genotypes = {g["complex_id"]: g for g in json.load(f)["genotypes"]}

    carrier = genotypes.get("cx-068fae")
    inner = genotypes.get("cx-b3b0bb")
    if carrier is None or inner is None:
        print(f"[verify] !!! фикстура неполна: cx-068fae={carrier is not None} "
              f"cx-b3b0bb={inner is not None}")
        return False

    slot2 = carrier["root"]["children"][2]

    class _FixtureRegistry:
        def get(self, mid):
            if mid == "cx.cx-b3b0bb":
                return {"genotype": inner}
            return None

    models = set(ML.slot_models(slot2, _FixtureRegistry()))
    expected = {"gemma-3-it-1b-q5_k_s", "qwen3-1.7b-q4_0-unsloth",
                "qwen2.5-coder-1.5b-instruct-q4_0", "internvl3-2b-q4_k_m"}
    ok = (models == expected)
    print(f"[verify] unfold fixture (cx-068fae slot2 over cx-b3b0bb): "
          f"{sorted(models)} {'PASS' if ok else 'FAIL, expected ' + str(sorted(expected))}")

    # без registry (или с одной, не разворачивающей "cx.") -- старое поведение,
    # должно оставаться неполным (1 модель) -- доказывает, что фикс МЕНЯЕТ
    # результат, не просто не ломает его.
    class _NullRegistry:
        def get(self, mid):
            return None

    old_models = set(ML.slot_models(slot2, _NullRegistry()))
    regressed = (old_models == {"gemma-3-it-1b-q5_k_s"})
    print(f"[verify] pre-fix behavior reproduced (sanity): {sorted(old_models)} "
          f"{'OK (matches LIFE-3 bug)' if regressed else 'UNEXPECTED'}")
    return ok and regressed


if __name__ == "__main__":
    ok_seams = verify_seams()
    ok_grid = verify_grid_copy()
    ok_fixture = verify_fixture_unfold()
    all_ok = ok_seams and ok_grid and ok_fixture
    print(f"[verify] ИТОГО: seams={ok_seams} grid={ok_grid} fixture={ok_fixture} "
          f"-> {'OK' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
