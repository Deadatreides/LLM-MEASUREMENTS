"""verify_seams.py — PROTOCOL.md §3.2: швы + РОВНО один смоук generate().
Обязателен ДО live_grid_builder.py. Провал -> BLOCKERS.md, стоп, без
отката на offline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402
import call_log             # noqa: E402


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


def smoke_generate() -> dict:
    ds = LD.default_dataset()
    task_id = ds.split["train"][0]
    task = ds.tasks[task_id]
    kind = LD.HSTEP.STEP_KINDS[0]
    step_idx = 0
    step = task["steps"][step_idx]
    model_id = LD.HSTEP.MODEL_IDS[0]

    prompt = LD.tasks14.step_prompt(task, step_idx, {})
    seed = LD.draw_seed(task_id, kind, model_id)

    llm, load_t = LD.mr11.load_model(model_id)
    t0 = time.time()
    raw = LD.mr11.generate(llm, model_id, prompt, temperature=LD.TEMPERATURE, top_p=1.0,
                           seed=seed, max_tokens=LD.MAX_TOKENS_STEP)
    ms = (time.time() - t0) * 1000.0
    del llm

    if raw.get("generation_failed"):
        raise RuntimeError(f"smoke generate() failed: {raw.get('generation_error')}")

    res = LD.seams14.check_step(step, raw.get("raw_text", ""))
    n_in = raw.get("input_tokens") or 0
    n_out = raw.get("output_tokens") or 0
    call_log.append_call_log(
        model=model_id, task_id=task_id, step=kind, seed=seed,
        n_in=n_in, n_out=n_out, ms=ms,
        path_to_raw=f"smoke#{task_id}|{kind}|{model_id}",
    )
    print(f"[smoke] model={model_id} load={load_t:.2f}s generate={ms:.0f}ms "
          f"in={n_in} out={n_out} status={res['status']}")
    print(f"[smoke] raw_text[:120] = {raw.get('raw_text', '')[:120]!r}")
    return {"model": model_id, "task_id": task_id, "seed": seed, "status": res["status"]}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok = verify_seams()
    if not ok:
        print("[verify] ШВЫ НЕ ПРОШЛИ — СТОП (PROTOCOL.md §3.2: без offline-отката)")
        sys.exit(1)
    ds = LD.default_dataset()
    LD.write_split_json(ds)
    try:
        smoke_generate()
    except Exception as exc:                                      # noqa: BLE001
        print(f"[smoke] ПРОВАЛ: {type(exc).__name__}: {exc}")
        print("[smoke] -> BLOCKERS.md, остановиться. НЕ откатываться на offline.")
        sys.exit(1)
    print(f"[verify] live_call_log.jsonl строк: {call_log.count_rows()}")
    print("[verify] OK -- можно переходить к live_grid_builder.py")
