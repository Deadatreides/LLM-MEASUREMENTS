"""verify_seams.py — PROTOCOL.md §5/§6: grid-copy integrity + unit
fixtures for `ParetoEvolution` (A1) and `StallFromImprovement` (A2)
BEFORE any evolve. No new `generate()` calls anywhere in this package.
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
import pareto_death as PD        # noqa: E402
import stall_from_improvement as SFI  # noqa: E402
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
    """`ast`-based, not substring search -- LIFE-6 found (and fixed in
    itself) that a naive substring check false-positives on a file's own
    explanatory docstring text. Reused directly here, not re-derived."""
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


class _FakeAccount:
    def __init__(self, subsidy=100, novelty_gen=None, born_gen=0):
        self.born_gen = born_gen
        self.novelty_gen = novelty_gen
        self.subsidy = subsidy
        self.subsidy_granted_total = 0
        self.overspend_total = 0


def verify_pareto_fixture() -> bool:
    """A/B/C: A has (r=0.5, n_imp=1), B has (r=0.5, n_imp=2) -- B weakly
    dominates A (r equal, n_imp strictly greater) -> A should die if
    economically flagged. C has (r=0.6, n_imp=0) -- nothing dominates C
    on this vector (A/B both have lower r... wait C's r is HIGHER, so C
    can't be dominated by A/B; A is dominated by B; B is dominated by
    nothing here (highest n_imp, and r tied with A, higher than nobody
    exceeds) -- constructed so exactly one of three economically-dead
    candidates is exempted."""
    class Bare(PD.ParetoEvolution):
        def __init__(self):
            pass

    ev = Bare()
    ev.references = set()
    ev.current_gen = 5
    ev.accounts = {"A": _FakeAccount(subsidy=-1), "B": _FakeAccount(subsidy=-1),
                  "C": _FakeAccount(subsidy=-1)}
    ev.genotypes = {"A": {"complex_id": "A"}, "B": {"complex_id": "B"}, "C": {"complex_id": "C"}}
    ev._imp_fn = lambda g: {"A": frozenset({0}), "B": frozenset({0, 1}),
                           "C": frozenset()}[g["complex_id"]]
    ev._objective_mode = PD.SCALAR_NIMP

    scored = {
        "A": {"delta_r": {"ci_95": [-0.5, -0.1]}, "unique_solves": set(), "r": 0.5},
        "B": {"delta_r": {"ci_95": [-0.5, -0.1]}, "unique_solves": set(), "r": 0.5},
        "C": {"delta_r": {"ci_95": [-0.5, -0.1]}, "unique_solves": set(), "r": 0.6},
    }
    # baseline: all 3 economically dead (forced via scored's delta_r), independent of D1/D2 internals --
    # exercise _filter_non_dominated directly, the piece this package actually adds.
    candidate_dead = [("A", "why"), ("B", "why"), ("C", "why")]
    kept = ev._filter_non_dominated(candidate_dead, ["A", "B", "C"], scored)
    kept_ids = sorted(c for c, _ in kept)
    # A=(r=0.5,n_imp=1), B=(r=0.5,n_imp=2), C=(r=0.6,n_imp=0).
    #   B dominates A (equal r, strictly higher n_imp) -> A stays dead.
    #   Nothing dominates B (A has lower n_imp; C has higher r but n_imp=0 < 2, fails "not less").
    #   Nothing dominates C (A/B both have strictly lower r than C).
    # -> only A should remain in the dead list; B and C get exempted.
    print(f"[verify] Pareto fixture: kept_dead={kept_ids} (expect ['A'])")
    ok = kept_ids == ["A"]
    print(f"[verify] Pareto fixture result: {'PASS' if ok else 'FAIL'}")
    return ok


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

    # Prime a baseline first -- the very FIRST ever call is trivially
    # "improved" against the tracker's initial -inf/-1 sentinels (there's
    # nothing to have stalled against yet), so it must not be counted as
    # one of the "flat" generations below.
    ev.generations = [{"generation": 0, "scores": {"X": {"r": 0.3}}}]
    tracker.update(ev, imp_fn)
    assert tracker.stall_count == 0, "baseline call unexpectedly did not reset stall_count"

    ev.generations.append({"generation": 1, "scores": {"X": {"r": 0.3}}})
    tracker.update(ev, imp_fn)
    ev.generations.append({"generation": 2, "scores": {"X": {"r": 0.3}}})
    tracker.update(ev, imp_fn)
    stall_after_flat = tracker.stall_count
    print(f"[verify] stall fixture: 2 flat generations AFTER baseline (same r, same n_imp) -> "
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


if __name__ == "__main__":
    ok_seams = verify_seams()
    ok_grid = verify_grid_copy()
    ok_registry = verify_no_registry_imports()
    ok_pareto = verify_pareto_fixture()
    ok_stall = verify_stall_fixture()
    all_ok = ok_seams and ok_grid and ok_registry and ok_pareto and ok_stall
    print(f"[verify] TOTAL: seams={ok_seams} grid={ok_grid} registry={ok_registry} "
          f"pareto={ok_pareto} stall={ok_stall} -> {'OK' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
