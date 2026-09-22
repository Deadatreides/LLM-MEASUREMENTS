"""verify_seams.py — PROTOCOL.md §4-6 pre-flight checks: grid-copy md5 +
seam self-test (reused) + UNION ±1 cross-check vs LIFE-8's own ceiling +
the 89/200 reparse-recovery fixture (regression guard on the P1/§2
correction itself) + planner-JSON-parse fixture. Zero new generate()
calls -- everything here is CPU-only, reading already-existing data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402
import lean_seeds as LS     # noqa: E402
import ceiling as CEIL      # noqa: E402
import residual as RES      # noqa: E402
import whole_alpha as WA    # noqa: E402
import decomp_pipeline as DP  # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = AGENT_DIR.parent
LIFE8_GRID_DIR = ROOT / "agent_life8_close_b3" / "metrics" / "live_grid"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def verify_grid_copy() -> bool:
    ok = True
    for name in ("train_grid.json", "test_grid.json", "whole_grid.json"):
        a = _md5(LIFE8_GRID_DIR / name)
        b = _md5(LD.LIVE_GRID_DIR / name)
        match = a == b
        print(f"[verify] {name}: md5 {'MATCH' if match else 'MISMATCH'} ({a[:8]} vs {b[:8]})")
        ok = ok and match
    return ok


def verify_seams14() -> bool:
    r = LD.seams14.self_test()
    print(f"[verify] experiment14.seams.self_test: {r['n_cases']} cases, "
          f"{'PASS' if r['passed'] else 'FAIL'}")
    return r["passed"]


def verify_union_cross_check() -> tuple:
    ds = LD.default_dataset()
    test_ids = ds.split["test"]
    panel = sorted(ds.split["train"])[:80]
    ceiling = CEIL.compute_ceiling(ds, panel, test_ids)
    anatomy = RES.compute_residual_anatomy(ds, test_ids)
    expected = ceiling["UNION"]["r_test"] * len(test_ids)
    diff = abs(anatomy["n_u_ok"] - round(expected))
    ok = diff <= 1
    print(f"[verify] U_ok={anatomy['n_u_ok']} vs UNION*n={expected:.1f} "
          f"diff={diff} (tolerance 1) {'OK' if ok else 'FAIL'}")
    return ok, ceiling, ds


def verify_reparse_fixture() -> bool:
    ds = LD.default_dataset()
    rows = LD.load_whole_grid_rows()
    winner_rows = [r for r in rows if r["model"] == WA.B2_MODEL]
    n_raw = sum(1 for r in winner_rows if WA.whole_resolved(ds, r, reparsed=False))
    n_reparsed = sum(1 for r in winner_rows if WA.whole_resolved(ds, r, reparsed=True))
    ok = (n_raw == 59) and (n_reparsed == 148) and (n_reparsed - n_raw == 89)
    print(f"[verify] reparse fixture: raw={n_raw} (expect 59) reparsed={n_reparsed} "
          f"(expect 148, +89) {'PASS' if ok else 'FAIL'}")
    return ok


_PLANNER_FIXTURE_CASES = [
    ('[{"kind":"READ","name":"a","brief":"x"},{"kind":"COMPUTE","name":"b","brief":"y"}]', True),
    ('```json\n[{"kind":"READ","name":"a","brief":"x"},'
     '{"kind":"LOOKUP","name":"b","brief":"y"},'
     '{"kind":"COMPUTE","name":"c","brief":"z"}]\n```', True),
    ("not json at all", False),
    ('[{"kind":"READ","name":"a","brief":"x"}]', False),                  # only 1, need >=2
    ('[{"kind":"FORMAT","name":"a","brief":"x"},{"kind":"COMPUTE","name":"b","brief":"y"}]', False),
    ('[{"kind":"READ","brief":"x"},{"kind":"COMPUTE","name":"b","brief":"y"}]', True),  # name auto-generated
    ('[{},{},{},{},{}]', False),                                          # 5 items, out of range
]


def verify_list_prefix_regex_fixture() -> bool:
    """Regression guard: `_LIST_PREFIX_RE` must strip a real list marker
    ("1. foo" -> "foo") WITHOUT eating the integer part of a decimal
    response ("13.1" -> "13.1", not "1"). Caught this session via a smoke
    run where every chained COMPUTE value was corrupted -- see
    BLOCKERS.md."""
    cases = [
        ("13.1", "13.1"), ("33.9", "33.9"), ("1. parcels = 5", "parcels = 5"),
        ("12) foo", "foo"), ("42", "42"), ("2. 7", "7"),
    ]
    ok = True
    for text, expected in cases:
        got = WA._LIST_PREFIX_RE.sub("", text)
        if got != expected:
            print(f"[verify] !!! list-prefix regex FAILED for {text!r}: expected {expected!r}, got {got!r}")
            ok = False
        got_dp = DP._LIST_PREFIX_RE.sub("", text)
        if got_dp != expected:
            print(f"[verify] !!! decomp_pipeline's list-prefix regex FAILED for {text!r}: "
                  f"expected {expected!r}, got {got_dp!r}")
            ok = False
    print(f"[verify] list-prefix regex fixture: {'PASS' if ok else 'FAIL'} ({len(cases)} cases)")
    return ok


def verify_planner_parse_fixture() -> bool:
    ok = True
    for text, expect_valid in _PLANNER_FIXTURE_CASES:
        plan = DP.parse_plan(text)
        got_valid = plan is not None
        if got_valid != expect_valid:
            print(f"[verify] !!! planner parse fixture FAILED for {text!r}: "
                  f"expected valid={expect_valid}, got {got_valid}")
            ok = False
    print(f"[verify] planner parse fixture: {'PASS' if ok else 'FAIL'} "
          f"({len(_PLANNER_FIXTURE_CASES)} cases)")
    return ok


if __name__ == "__main__":
    ok_grid = verify_grid_copy()
    ok_seams = verify_seams14()
    ok_union, ceiling, ds = verify_union_cross_check()
    ok_reparse = verify_reparse_fixture()
    ok_list_prefix = verify_list_prefix_regex_fixture()
    ok_planner = verify_planner_parse_fixture()
    all_ok = ok_grid and ok_seams and ok_union and ok_reparse and ok_list_prefix and ok_planner
    print(f"[verify] TOTAL: grid={ok_grid} seams={ok_seams} union={ok_union} "
          f"reparse={ok_reparse} list_prefix={ok_list_prefix} planner={ok_planner} -> "
          f"{'OK' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
