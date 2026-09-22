"""verify_seams.py — PROTOCOL.md smoke section: grid-copy integrity +
stall-reset unit fixture (reused from LIFE-7 unchanged) + glue-donor P6
assertion (real 17 donors) + `ast`-based 0-registry-imports check +
(conditional, skipped this run -- candidates confirmed empty) 1-generate
new-model smoke. No new `generate()` calls for the original 6 models.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD        # noqa: E402
import orchestrator as O         # noqa: E402
import metrics_lib as ML         # noqa: E402
import stall_from_improvement as SFI  # noqa: E402
import glue_seeds as GS          # noqa: E402
import model_expansion as ME     # noqa: E402
import call_log                  # noqa: E402

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
        print(f"[verify] {label}: {n} rows (expected {expected}) {'OK' if row_ok else 'MISMATCH'}")
        ok = ok and row_ok

    train_ids = set(LD.default_dataset().split["train"])
    n_train_log = call_log.count_rows_for_tasks(train_ids)
    floor_ok = n_train_log >= TRAIN_FLOOR
    print(f"[verify] TRAIN log rows: {n_train_log} (floor {TRAIN_FLOOR}) {'OK' if floor_ok else 'FAIL'}")
    return ok and floor_ok


def verify_no_registry_imports() -> bool:
    dispatcher_src = (SRC / "default_channel.py").read_text(encoding="utf-8")
    tree = ast.parse(dispatcher_src)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    forbidden = {"block_registry", "block_insert"}
    has_registry_path = bool(imported_names & forbidden)
    print(f"[verify] imports in default_channel.py: {sorted(imported_names)}")
    print(f"[verify] 0 registry-adjacent code in default_channel.py: {not has_registry_path}")
    return not has_registry_path


def verify_stall_fixture() -> bool:
    class FakeEv:
        pass

    ev = FakeEv()
    ev.references = set()
    ev.population = ["X"]
    ev.genotypes = {"X": {"complex_id": "X"}}

    def imp_fn(g):
        return frozenset({0}) if g["complex_id"] == "X" else frozenset()

    tracker = SFI.StallFromImprovement(g_stall=20)

    ev.generations = [{"generation": 0, "scores": {"X": {"r": 0.3}}}]
    tracker.update(ev, imp_fn)
    assert tracker.stall_count == 0, "baseline call unexpectedly did not reset stall_count"

    ev.generations.append({"generation": 1, "scores": {"X": {"r": 0.3}}})
    tracker.update(ev, imp_fn)
    ev.generations.append({"generation": 2, "scores": {"X": {"r": 0.3}}})
    tracker.update(ev, imp_fn)
    stall_after_flat = tracker.stall_count
    print(f"[verify] stall fixture: 2 flat generations after baseline -> "
          f"stall_count={stall_after_flat} (expect 2)")

    ev.generations.append({"generation": 3, "scores": {"X": {"r": 0.3}}})

    def imp_fn2(g):
        return frozenset({0, 1}) if g["complex_id"] == "X" else frozenset()

    tracker.update(ev, imp_fn2)
    print(f"[verify] stall fixture: n_imp improved at same r -> "
          f"stall_count={tracker.stall_count} (expect 0)")

    ok = (stall_after_flat == 2) and (tracker.stall_count == 0)
    print(f"[verify] stall fixture result: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_glue_donors() -> bool:
    ds = LD.default_dataset()
    panel = O.build_panel(ds)["panel"]
    best_single = ML.best_single_per_slot(ds, panel)
    donors = GS.build_glue_donors(ds, panel, best_single)
    n = len(donors)
    print(f"[verify] glue donors built: {n} (expect 17: 3 READ + 4 LOOKUP + 10 COMPUTE)")
    ok_count = (n == 17)
    try:
        GS.assert_glue_donors_single_slot(donors, ds, panel, best_single)
        ok_assert = True
    except AssertionError as exc:
        print(f"[verify] !!! glue donor P6 assert failed: {exc}")
        ok_assert = False
    print(f"[verify] glue donors P6 (|ImpSet|==1 each): {'PASS' if ok_assert else 'FAIL'}")
    return ok_count and ok_assert


def verify_phase3_candidates() -> dict:
    candidates = ME.find_candidate_models(LD.HSTEP.MODEL_IDS)
    print(f"[verify] Phase 3 candidate models (sorted, K<=4): {candidates}")
    return {"candidates": candidates, "n": len(candidates)}


if __name__ == "__main__":
    ok_seams = verify_seams()
    ok_grid = verify_grid_copy()
    ok_registry = verify_no_registry_imports()
    ok_stall = verify_stall_fixture()
    ok_glue = verify_glue_donors()
    phase3_info = verify_phase3_candidates()
    all_ok = ok_seams and ok_grid and ok_registry and ok_stall and ok_glue
    print(f"[verify] TOTAL: seams={ok_seams} grid={ok_grid} registry={ok_registry} "
          f"stall={ok_stall} glue={ok_glue} phase3_candidates={phase3_info['n']} -> "
          f"{'OK' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
