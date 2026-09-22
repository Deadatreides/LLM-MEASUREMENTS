"""verify_seams.py — FORK-1 pre-flight checks (no live calls): Path A's
own self-test (40/40 filter_det matches DELTA-0's golden F2 -- HARD_STOP
if not, PROTOCOL.md §1.4), T_hard generator invariants, and oracle
checker fixtures (reused pattern from DELTA-0, negative-lookahead
regression guard included).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_f2 as TGF2      # noqa: E402
import task_generator_hard as TGH     # noqa: E402
import oracles as OR                  # noqa: E402
import det_atoms as DA                # noqa: E402


def verify_path_a_self_test() -> bool:
    """PROTOCOL.md §1.4: filter_det must match golden F2 matched_ids
    40/40 exactly -- a mismatch is a CODE BUG (HARD_STOP), not a scored
    result. Also checks the full det chain (filter->aggregate->derive)
    against final_oracle as an r_A preview/consistency check."""
    d = TGF2.build_tasks()
    n_filter_ok = n_final_ok = 0
    mismatches = []
    for t_id in d["test"]:
        task = d["tasks"][t_id]
        ids = DA.filter_det(task["records"], task["target_category"], task["target_region"])
        fcheck = OR.check_filter(", ".join(sorted(ids)), task["matched_ids"])
        if fcheck["v"] == 1:
            n_filter_ok += 1
        else:
            mismatches.append(t_id)
        agg = DA.aggregate_det(ids, task["op"], task["id_to_amount"])
        final = DA.derive_det(agg, task["threshold"], task["comparator"])
        if final == task["final_oracle"]:
            n_final_ok += 1
    n = len(d["test"])
    ok = (n_filter_ok == n) and (n_final_ok == n)
    print(f"[verify] Path A self-test: filter_det {n_filter_ok}/{n} exact, "
          f"full chain {n_final_ok}/{n} matches final_oracle {'PASS' if ok else 'FAIL'}")
    if mismatches:
        print(f"[verify] !!! mismatches: {mismatches}")
    return ok


def verify_task_hard_generator() -> bool:
    d = TGH.build_tasks()
    ok = True
    lens = []
    for t_id in d["train"] + d["test"]:
        t = d["tasks"][t_id]
        n_match = len(t["matched_ids"])
        if not (2 <= n_match <= 5):
            print(f"[verify] !!! T_hard {t_id}: n_true_match={n_match} out of [2,5]")
            ok = False
        lens.append(len(t["db_text"]))
        # sum_value must equal independent recomputation from records
        recomputed = round(sum(r["amount"] for r in t["records"]
                               if r["category"] == t["target_category"]
                               and r["region"] == t["target_region"]), 2)
        if abs(recomputed - t["sum_value"]) > 0.01:
            print(f"[verify] !!! T_hard {t_id}: sum_value mismatch {t['sum_value']} vs {recomputed}")
            ok = False
        if t["final_oracle"] != t["sum_value"]:
            print(f"[verify] !!! T_hard {t_id}: final_oracle != sum_value")
            ok = False
    if min(lens) < 1500:
        print(f"[verify] !!! T_hard min db_text chars {min(lens)} < 1500 (B5)")
        ok = False
    n_test_ok = len(d["test"]) >= 30 and len(d["train"]) >= 30
    if not n_test_ok:
        print(f"[verify] !!! T_hard N_TRAIN={len(d['train'])} N_TEST={len(d['test'])} < 30 (B4)")
        ok = False
    print(f"[verify] T_hard generator: n_tasks_ok={ok}, chars=[{min(lens)},{max(lens)}], "
          f"N_TRAIN={len(d['train'])} N_TEST={len(d['test'])} {'PASS' if ok else 'FAIL'}")
    return ok


def verify_list_prefix_regex_fixture() -> bool:
    cases = [
        ("13.1", "13.1"), ("33.9", "33.9"), ("1. parcels = 5", "parcels = 5"),
        ("12) foo", "foo"), ("42", "42"), ("2. 7", "7"),
    ]
    ok = True
    for text, expected in cases:
        got = OR._LIST_PREFIX_RE.sub("", text)
        if got != expected:
            print(f"[verify] !!! list-prefix regex FAILED for {text!r}: expected {expected!r}, got {got!r}")
            ok = False
    print(f"[verify] list-prefix regex fixture: {'PASS' if ok else 'FAIL'} ({len(cases)} cases)")
    return ok


def verify_id_regex_both_families() -> bool:
    ok = True
    if OR.extract_ids("TXN-0001, TXN-0005") != frozenset({"TXN-0001", "TXN-0005"}):
        print("[verify] !!! id regex FAILED for TXN- family")
        ok = False
    if OR.extract_ids("REC-0001, REC-0012") != frozenset({"REC-0001", "REC-0012"}):
        print("[verify] !!! id regex FAILED for REC- family")
        ok = False
    print(f"[verify] id regex (both TXN-/REC- families): {'PASS' if ok else 'FAIL'}")
    return ok


def verify_oracle_fixtures() -> bool:
    """Reused fixture shapes from DELTA-0: golden roundtrip, empty-match,
    garbled response -> v=0."""
    ok = True
    f1 = OR.check_filter("REC-0001, REC-0002", ["REC-0001", "REC-0002"])
    if f1["v"] != 1:
        print(f"[verify] !!! oracle fixture FAILED: filter golden -> {f1}")
        ok = False
    f2 = OR.check_filter("", [])
    if f2["v"] != 1:
        print(f"[verify] !!! oracle fixture FAILED: filter empty-match -> {f2}")
        ok = False
    a1 = OR.check_aggregate("1203.18", 1203.18)
    if a1["v"] != 1:
        print(f"[verify] !!! oracle fixture FAILED: aggregate golden -> {a1}")
        ok = False
    a2 = OR.check_aggregate("примерно 100 или, может, 105", 100.0)
    if a2["v"] != 0:
        print(f"[verify] !!! oracle fixture FAILED: aggregate garbled should be v=0 -> {a2}")
        ok = False
    d1 = OR.check_derive("ДА", "ДА")
    if d1["v"] != 1:
        print(f"[verify] !!! oracle fixture FAILED: derive golden -> {d1}")
        ok = False
    print(f"[verify] oracle fixtures (golden/empty/garbled): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r1 = verify_path_a_self_test()
    r2 = verify_task_hard_generator()
    r3 = verify_list_prefix_regex_fixture()
    r4 = verify_id_regex_both_families()
    r5 = verify_oracle_fixtures()
    all_ok = r1 and r2 and r3 and r4 and r5
    print(f"[verify] TOTAL: path_a={r1} t_hard_gen={r2} list_prefix={r3} id_regex={r4} "
          f"oracles={r5} -> {'OK' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
