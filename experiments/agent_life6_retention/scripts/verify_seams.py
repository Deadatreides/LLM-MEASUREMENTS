"""verify_seams.py — PROTOCOL.md §6: grid-copy integrity (md5-verified
this session, row-count re-checked here) + slot-safe-mutate fixtures
(happy path AND the whole-root-rewrite guard specifically) BEFORE any
evolve. No new `generate()` calls anywhere in this package.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD        # noqa: E402
import metrics_lib as ML         # noqa: E402
import slot_safe_mutate as SSM   # noqa: E402
import orchestrator as O         # noqa: E402
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


def _build_ctx(reg, be, ds, archive) -> dict:
    return {
        "generators": list(reg.generator_ids()),
        "observables": [],
        "feasible_params": be.feasible_params("heterostep"),
        "donors": list(archive.values()),
        "composites": list(reg.composite_ids()),
        "assemble_n_steps": len(LD.HSTEP.STEP_KINDS),
    }


def _fixture_parent(ds):
    """slot 0 (READ) = a known Imp=1 pair (confirmed by LIFE-5's slot
    map: gemma-3-it-1b-q5_k_s + internvl3-2b-q4_k_m, PAR); other 3 slots
    = single models (whatever they happen to Imp to doesn't matter for
    this fixture -- only READ's stability is checked)."""
    models = list(ds.models)
    route = {
        "READ": ["gemma-3-it-1b-q5_k_s", "internvl3-2b-q4_k_m"],
        "FORMAT": [models[0]], "LOOKUP": [models[1]], "COMPUTE": [models[2]],
    }
    return LD.G.genotype(LD.HSEED.route(route), gen=0, origin="fixture:slot_safe_read_pair")


def verify_slot_safe_fixture() -> bool:
    ds = LD.default_dataset()
    panel = O.build_panel(ds)["panel"]
    best_single = ML.best_single_per_slot(ds, panel)

    reg = LD.HSTEP.Registry()
    be = LD.HSTEP.HeterostepBackend(ds)
    parent = _fixture_parent(ds)

    def imp_fn(g):
        return ML.imp_set(ds, panel, best_single, g, reg)

    parent_imp = imp_fn(parent)
    print(f"[verify] fixture parent Imp = {sorted(parent_imp)} (expect 0 in it -- READ)")
    if 0 not in parent_imp:
        print("[verify] !!! fixture parent does not have Imp(READ)=1 -- fixture itself is invalid")
        return False

    archive = {parent["complex_id"]: parent}
    ctx = _build_ctx(reg, be, ds, archive)
    rng = random.Random(424242)

    n_trials = 50
    n_bypassed = 0
    n_read_changed_when_strict = 0
    n_none = 0
    read_slot_json = json.dumps(parent["root"]["children"][0], sort_keys=True, ensure_ascii=False)

    for _ in range(n_trials):
        child, report = SSM.slot_safe_mutate(parent, rng, ctx, reg, gen=1, imp_fn=imp_fn, p_safe=0.05)
        if child is None:
            n_none += 1
            continue
        bypassed = report.get("slot_safe_bypassed", False)
        if bypassed:
            n_bypassed += 1
            continue
        child_read_json = json.dumps(child["root"]["children"][0], sort_keys=True, ensure_ascii=False)
        if child_read_json != read_slot_json:
            n_read_changed_when_strict += 1

    print(f"[verify] slot-safe fixture: {n_trials} trials, bypassed={n_bypassed}, "
          f"none={n_none}, READ changed while STRICT={n_read_changed_when_strict} (expect 0)")
    happy_path_ok = (n_read_changed_when_strict == 0)

    # whole-root-rewrite guard, unit-level: a child whose root has a
    # DIFFERENT number of children than parent must be treated as
    # touching every protected slot, not silently compared per-index.
    fake_child_root = {"op": "SWITCH", "obs": "x", "cases": {}, "default": {"op": "STOP"}}
    guard_triggered = SSM._touches_protected_slot(parent["root"], fake_child_root, frozenset({0}))
    print(f"[verify] whole-root-rewrite guard triggers on non-ASSEMBLE child: {guard_triggered} (expect True)")

    fake_child_root2 = {"op": "ASSEMBLE", "children": parent["root"]["children"][:3]}   # 3 not 4
    guard_triggered2 = SSM._touches_protected_slot(parent["root"], fake_child_root2, frozenset({0}))
    print(f"[verify] whole-root-rewrite guard triggers on child-count mismatch: {guard_triggered2} (expect True)")

    return happy_path_ok and guard_triggered and guard_triggered2


if __name__ == "__main__":
    ok_seams = verify_seams()
    ok_grid = verify_grid_copy()
    ok_fixture = verify_slot_safe_fixture()
    all_ok = ok_seams and ok_grid and ok_fixture
    print(f"[verify] TOTAL: seams={ok_seams} grid={ok_grid} fixture={ok_fixture} -> {'OK' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
