"""verify_seams.py — DELTA-0 C4: 5 self-test fixtures for the generator +
oracles (no live calls). Also exposes `roundtrip_check` (CPU-only, used
by C10's smoke first half: 5 tasks, generator -> intermediate v -> final
oracle roundtrip).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator as TG   # noqa: E402
import oracles as OR          # noqa: E402


def fixture_1_golden_roundtrip() -> bool:
    """Golden inputs -> v=1 on all 3 checkpoints."""
    d = TG.build_tasks(n_train=4, n_test=4, seed=99001)
    ok = True
    for tid in d["train"] + d["test"]:
        t = d["tasks"][tid]
        f = OR.check_filter(", ".join(t["matched_ids"]), t["matched_ids"])
        a = OR.check_aggregate(f"{t['aggregate_value']:g}", t["aggregate_value"])
        fin = OR.check_derive(t["final_oracle"], t["final_oracle"])
        if not (f["v"] == 1 and a["v"] == 1 and fin["v"] == 1):
            print(f"[verify] !!! fixture_1 FAILED on {tid}: filter_v={f['v']} agg_v={a['v']} final_v={fin['v']}")
            ok = False
    print(f"[verify] fixture 1 (golden roundtrip): {'PASS' if ok else 'FAIL'}")
    return ok


def fixture_2_empty_match() -> bool:
    """Degenerate 0-match case does not crash and scores correctly --
    generator itself never produces this (2-5 guaranteed), but the
    ORACLE code must still be robust to it."""
    ok = True
    f = OR.check_filter("", [])
    if not (f["v"] == 1 and f["extracted"] == []):
        print(f"[verify] !!! fixture_2 FAILED: empty filter -> {f}")
        ok = False
    a_sum = OR.check_aggregate("0", 0.0)
    a_count = OR.check_aggregate("0", 0.0)
    if not (a_sum["v"] == 1 and a_count["v"] == 1):
        print(f"[verify] !!! fixture_2 FAILED: empty aggregate -> sum={a_sum} count={a_count}")
        ok = False
    print(f"[verify] fixture 2 (empty-match, no crash): {'PASS' if ok else 'FAIL'}")
    return ok


def fixture_3_comparator_boundary() -> bool:
    """Aggregate exactly equals threshold: >= is True, < is False."""
    ok = True
    if TG.evaluate_comparator(100.0, 100.0, ">=") is not True:
        print("[verify] !!! fixture_3 FAILED: 100>=100 should be True")
        ok = False
    if TG.evaluate_comparator(100.0, 100.0, "<") is not False:
        print("[verify] !!! fixture_3 FAILED: 100<100 should be False")
        ok = False
    if TG.evaluate_comparator(99.99, 100.0, ">=") is not False:
        print("[verify] !!! fixture_3 FAILED: 99.99>=100 should be False")
        ok = False
    print(f"[verify] fixture 3 (comparator boundary): {'PASS' if ok else 'FAIL'}")
    return ok


def fixture_4_wellformed_response() -> bool:
    """A plausible, well-formed synthetic model response per atom parses
    correctly."""
    ok = True
    f = OR.check_filter("Подходящие транзакции: TXN-0001, TXN-0005, TXN-0009",
                        ["TXN-0001", "TXN-0005", "TXN-0009"])
    if f["v"] != 1:
        print(f"[verify] !!! fixture_4 FAILED: filter well-formed -> {f}")
        ok = False
    a = OR.check_aggregate("Считаем сумму:\n245.10", 245.10)
    if a["v"] != 1:
        print(f"[verify] !!! fixture_4 FAILED: aggregate well-formed (tier2) -> {a}")
        ok = False
    a2 = OR.check_aggregate("13.1", 13.1)   # regression guard: must NOT become 1.0
    if a2["v"] != 1 or a2["extracted"] != 13.1:
        print(f"[verify] !!! fixture_4 FAILED: decimal-eating regression -> {a2}")
        ok = False
    d1 = OR.check_derive("ДА", "ДА")
    d2 = OR.check_derive("Ответ:\nНЕТ.", "НЕТ")   # tier-2: answer on its own line, trailing period
    if d1["v"] != 1 or d2["v"] != 1:
        print(f"[verify] !!! fixture_4 FAILED: derive well-formed -> {d1} {d2}")
        ok = False
    print(f"[verify] fixture 4 (well-formed responses parse): {'PASS' if ok else 'FAIL'}")
    return ok


def fixture_5_garbled_response() -> bool:
    """A garbled/ambiguous response correctly yields v=0, not a crash and
    not a guess."""
    ok = True
    f = OR.check_filter("не могу определить подходящие записи", ["TXN-0001"])
    if f["v"] != 0:
        print(f"[verify] !!! fixture_5 FAILED: filter garbled -> {f}")
        ok = False
    a = OR.check_aggregate("примерно 100 или, может, 105, точно не скажу", 100.0)
    if a["v"] != 0:
        print(f"[verify] !!! fixture_5 FAILED: aggregate garbled (ambiguous, 2 numbers) -> {a}")
        ok = False
    d = OR.check_derive("возможно", "ДА")
    if d["v"] != 0:
        print(f"[verify] !!! fixture_5 FAILED: derive garbled -> {d}")
        ok = False
    d2 = OR.check_derive("ДА, наверное, хотя НЕТ, не уверен", "ДА")   # both tokens present -> ambiguous
    if d2["v"] != 0:
        print(f"[verify] !!! fixture_5 FAILED: derive both-tokens-present should be v=0 -> {d2}")
        ok = False
    print(f"[verify] fixture 5 (garbled response -> v=0, no crash): {'PASS' if ok else 'FAIL'}")
    return ok


def roundtrip_check(n: int = 5, seed: int = 99002) -> bool:
    """C10 smoke, first half: N tasks, generator -> intermediate v ->
    final oracle roundtrip, CPU-only."""
    d = TG.build_tasks(n_train=0, n_test=n, seed=seed)
    ok = True
    for tid in d["test"]:
        t = d["tasks"][tid]
        if not (2 <= len(t["matched_ids"]) <= 5):
            print(f"[verify] !!! roundtrip FAILED {tid}: matched_ids count out of range")
            ok = False
        if len(t["db_text"]) < 2000:
            print(f"[verify] !!! roundtrip FAILED {tid}: db_text only {len(t['db_text'])} chars (T4)")
            ok = False
        f = OR.check_filter(", ".join(t["matched_ids"]), t["matched_ids"])
        a = OR.check_aggregate(f"{t['aggregate_value']:g}", t["aggregate_value"])
        fin = OR.check_derive(t["final_oracle"], t["final_oracle"])
        if not (f["v"] == 1 and a["v"] == 1 and fin["v"] == 1):
            print(f"[verify] !!! roundtrip FAILED {tid}: v's = {f['v']},{a['v']},{fin['v']}")
            ok = False
    print(f"[verify] roundtrip ({n} tasks, CPU-only): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r1 = fixture_1_golden_roundtrip()
    r2 = fixture_2_empty_match()
    r3 = fixture_3_comparator_boundary()
    r4 = fixture_4_wellformed_response()
    r5 = fixture_5_garbled_response()
    all_ok = r1 and r2 and r3 and r4 and r5
    print(f"[verify] TOTAL C4 (5 fixtures): {'OK' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
