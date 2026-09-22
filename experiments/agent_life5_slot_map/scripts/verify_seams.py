"""verify_seams.py — PROTOCOL.md: три проверки ДО любой карты/evolve:

1. Швы (arch2.heterostep/experiment14.seams self-test) -- код-корректность.
2. Целостность СКОПИРОВАННОЙ сетки (row-counts + train floor) -- нужна
   только для Phase B.2 (optional evolve); Phase B (карта) сетку вообще
   не открывает, читает готовые archive.json.
3. Фикстура разворота cx -- ТОТ ЖЕ реальный случай, что LIFE-4 (`cx-068fae`
   слот 2 = `cx.cx-23f285`, обёртка над `cx.cx-b3b0bb`), прочитанный
   напрямую из `agent_life3_block_live/runs/life3_with_s20261601/
   archive.json` (read-only) -- подтверждает, что `unfold_metrics.
   slot_models` (новый модуль, скопированная логика) работает идентично
   LIFE-4's версии до того, как на неё опирается вся карта слотов.

Провал любой -> BLOCKERS.md, стоп.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD        # noqa: E402
import unfold_metrics as ML      # noqa: E402
import call_log                  # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = AGENT_DIR.parent
LIFE3_FIXTURE_ARCHIVE = (ROOT / "agent_life3_block_live" / "runs" /
                        "life3_with_s20261601" / "archive.json")

EXPECTED_TRAIN_ROWS = 2400
EXPECTED_TEST_ROWS = 2400
EXPECTED_WHOLE_ROWS = 700
TRAIN_FLOOR = 1920


def verify_seams() -> bool:
    r1 = LD.seams14.self_test()
    r2 = LD.HSTEP.self_test_seam()
    ok = r1["passed"] and r2["passed"]
    print(f"[verify] experiment14.seams.self_test: {r1['n_cases']} cases, "
          f"{'PASS' if r1['passed'] else 'FAIL'}")
    print(f"[verify] arch2.heterostep.self_test_seam: {r2['n_cases']} cases, "
          f"{'PASS' if r2['passed'] else 'FAIL'}")
    return ok


def verify_grid_copy() -> bool:
    """Только для Phase B.2 -- Phase B (карта) не зависит от этого."""
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
    print(f"[verify] TRAIN log rows: {n_train_log} (floor {TRAIN_FLOOR}) "
          f"{'OK' if floor_ok else 'FAIL'}")
    return ok and floor_ok


def verify_fixture_unfold() -> bool:
    """Тот же фикстур-тест, что LIFE-4 -- подтверждает `unfold_metrics.
    slot_models` (новый модуль) даёт тот же результат, что LIFE-4's
    `metrics_lib.slot_models` на РЕАЛЬНОМ зарегистрированном блоке."""
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
    return ok


if __name__ == "__main__":
    ok_seams = verify_seams()
    ok_grid = verify_grid_copy()
    ok_fixture = verify_fixture_unfold()
    all_ok = ok_seams and ok_grid and ok_fixture
    print(f"[verify] TOTAL: seams={ok_seams} grid={ok_grid} fixture={ok_fixture} "
          f"-> {'OK' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
