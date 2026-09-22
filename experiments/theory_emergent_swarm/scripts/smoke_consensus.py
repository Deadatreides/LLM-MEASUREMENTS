"""smoke_consensus.py — gate-before-budget for E1/E2. Runs the consensus
mechanism on the FIRST 3 test tasks x 6 models per family (36 calls
total) and verifies: no crashes, `extract_ids` parses non-trivially on at
least some responses, LOO-reliability code runs without dividing by zero
on the tiny slice. Does NOT judge the actual rate (n=3 is not meaningful
for that) -- only that the pipeline is sound before the full 480-call
campaign.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_f2 as TF2       # noqa: E402
import task_generator_hard as THARD   # noqa: E402
import atoms_hard as AH               # noqa: E402
import consensus_filter as CF         # noqa: E402
import model_registry_11 as MR        # noqa: E402

N_SMOKE = 3


def smoke_family(name: str, tasks_by_id: dict, test_ids: list, prompt_builder, max_tokens: int) -> bool:
    slice_ids = test_ids[:N_SMOKE]
    print(f"\n=== smoke [{name}] tasks={slice_ids} models={len(MR.MODEL_IDS)} "
         f"calls={len(slice_ids) * len(MR.MODEL_IDS)} ===", flush=True)
    t0 = time.time()
    votes = CF.collect_votes(MR.MODEL_IDS, tasks_by_id, slice_ids, prompt_builder, max_tokens,
                             family=f"smoke-{name}")
    print(f"  collect_votes done in {time.time() - t0:.1f}s", flush=True)

    any_nonempty = any(v["extracted"] for tid in slice_ids for v in votes[tid].values())
    any_failed = any(v["failed"] for tid in slice_ids for v in votes[tid].values())
    if not any_nonempty:
        print("  FAIL: every single extracted id-set was empty across all 3x6 calls "
             "-- prompt or parser is broken.", flush=True)
        return False
    if any_failed:
        n_fail = sum(1 for tid in slice_ids for v in votes[tid].values() if v["failed"])
        print(f"  WARNING: {n_fail}/{N_SMOKE * len(MR.MODEL_IDS)} calls failed "
             "(generation_failed=True) -- inspect filter_calls.jsonl before full run.", flush=True)

    try:
        cons = CF.run_consensus(tasks_by_id, slice_ids, MR.MODEL_IDS, votes)
    except ZeroDivisionError as exc:
        print(f"  FAIL: ZeroDivisionError in run_consensus: {exc}", flush=True)
        return False
    for tid in slice_ids:
        c = cons[tid]
        print(f"  {tid}: admitted={c['admitted']} consensus_ids={sorted(c['consensus_ids'])} "
             f"golden={tasks_by_id[tid]['matched_ids']}", flush=True)
    print(f"  OK [{name}]", flush=True)
    return True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    f2 = TF2.build_tasks()
    ok_f2 = smoke_family("F2", f2["tasks"], f2["test"], TF2.build_filter_prompt, TF2.MAX_TOKENS_FILTER)

    hard = THARD.build_tasks()
    ok_hard = smoke_family("T_hard", hard["tasks"], hard["test"], AH.build_filter_prompt,
                           AH.MAX_TOKENS_FILTER)

    print(f"\n=== SMOKE {'PASS' if (ok_f2 and ok_hard) else 'FAIL'} "
         f"(F2={'ok' if ok_f2 else 'FAIL'}, T_hard={'ok' if ok_hard else 'FAIL'}) ===", flush=True)
    if not (ok_f2 and ok_hard):
        sys.exit(1)


if __name__ == "__main__":
    main()
