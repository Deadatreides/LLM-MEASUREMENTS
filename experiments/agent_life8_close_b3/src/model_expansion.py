"""model_expansion.py — LIFE-8 Phase 3 (PROTOCOL.md P7, correction #5).
Every function here is written to be genuinely correct if `experiment11`'s
registry ever grows — NOT a hardcoded "always empty" shortcut — even
though this session's direct read of that registry (recorded in
PROTOCOL.md/BLOCKERS.md) confirms it currently contains exactly the same
6 models already in use, so `find_candidate_models` is expected to return
`[]` when this actually runs.

**Correction #5, load-bearing**: `patch_model_ids` is the ONLY place in
this entire package that touches `arch2.heterostep.MODEL_IDS`. It must
only ever be called from Phase 3's own branch in `run_all.py`, after
Phase 1 and Phase 2 have both already run to completion in the same
process — never at import time, never reachable from any Phase 1/2 code
path. Confirmed safe this session (direct read of `arch2/heterostep.py`):
`MODEL_IDS` is read as a bare global name inside `Registry.__init__`'s
`for mid in MODEL_IDS:` loop, resolved live at call time, so patching the
module attribute before constructing a `Registry()` is sufficient and
correctly scoped — no `arch2/` file edit, no import-time binding to work
around.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402
import call_log             # noqa: E402

STEP_KINDS = LD.HSTEP.STEP_KINDS
K_MAX = 4
GRID_FLOOR_FRAC = 0.8


def _fresh_registry_module():
    """Re-`exec_module`s `experiment11/configs/model_registry.py` fresh
    on every call — never cached at this package's import time — so a
    registry change between sessions is picked up without a restart."""
    path = LD.ROOT / "experiment11" / "configs" / "model_registry.py"
    spec = _ilu.spec_from_file_location("life8_model_registry11_fresh", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_candidate_models(current_models: tuple) -> list:
    """P7: sorted by id, first `K_MAX`, never "best by intuition". Reads
    the registry fresh (see `_fresh_registry_module`) — this is a real
    computation, not a hardcoded `[]`."""
    mod = _fresh_registry_module()
    all_ids = sorted(mod.MODEL_REGISTRY.keys())
    current = set(current_models)
    candidates = [m for m in all_ids if m not in current]
    return candidates[:K_MAX]


def patch_model_ids(new_ids: tuple) -> tuple:
    """Correction #5 — see module docstring. Returns the expanded tuple
    for logging/confirmation; also mutates `LD.HSTEP.MODEL_IDS` in place
    (the actual effect `Registry()` construction depends on)."""
    expanded = tuple(LD.HSTEP.MODEL_IDS) + tuple(new_ids)
    LD.HSTEP.MODEL_IDS = expanded
    return expanded


def build_new_models_grid(new_ids: list, task_ids: list, tasks: dict, out_path: Path) -> dict:
    """Model-major loop, restricted to ONLY `new_ids` (never the original
    6 — their grid files are never reopened here at all, read or write).
    One combined file (`out_path`, typically `metrics/live_grid/new_
    models_grid.json`) covering train∪test in one pass — resumable
    (skips cells already on disk), every call logged to `live_call_log.
    jsonl` (same anti-fraud mechanism as A5 onward)."""
    cells: dict = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for r in json.load(f)["rows"]:
                cells[(r["task_id"], r["kind"], r["model"])] = r
        print(f"[l8-expand] {len(cells)} ячеек уже на диске (резюме)", flush=True)

    for model_id in new_ids:
        n_todo = sum(1 for t in task_ids for k in STEP_KINDS if (t, k, model_id) not in cells)
        if n_todo == 0:
            print(f"[l8-expand] {model_id} уже полностью на диске, пропуск", flush=True)
            continue
        llm, load_t = LD.mr11.load_model(model_id)
        print(f"[l8-expand] {model_id} загружена за {load_t:.1f}с — {n_todo} ячеек в очереди", flush=True)
        t0 = time.time()
        n_done = 0
        for task_id in task_ids:
            task = tasks[task_id]
            oracle_upstream = {s["name"]: s["oracle"] for s in task["steps"]}
            for step_idx, kind in enumerate(STEP_KINDS):
                key = (task_id, kind, model_id)
                if key in cells:
                    continue
                step = task["steps"][step_idx]
                upstream = {n: oracle_upstream[n] for n in step["depends_on"]}
                prompt = LD.tasks14.step_prompt(task, step_idx, upstream)
                seed = LD.draw_seed(task_id, kind, model_id)

                t_call = time.time()
                raw = LD.mr11.generate(llm, model_id, prompt, temperature=LD.TEMPERATURE,
                                       top_p=1.0, seed=seed, max_tokens=LD.MAX_TOKENS_STEP)
                ms = (time.time() - t_call) * 1000.0
                res = LD.seams14.check_step(step, raw.get("raw_text", ""))
                n_in = raw.get("input_tokens") or 0
                n_out = raw.get("output_tokens") or 0

                row = {
                    "experiment_id": "life8_close_b3_expand", "phase": "expand", "task_id": task_id,
                    "model": model_id, "step_index": step_idx, "step_name": step["name"],
                    "kind": kind, "status": res["status"], "extracted": res.get("extracted"),
                    "oracle": res.get("oracle"), "reason": res.get("reason"),
                    "raw_output": raw.get("raw_text", ""), "tokens": n_in + n_out,
                    "latency": raw.get("generation_time_sec"), "seed": seed,
                    "timestamp": time.time(),
                }
                cells[key] = row
                call_log.append_call_log(
                    model=model_id, task_id=task_id, step=kind, seed=seed,
                    n_in=n_in, n_out=n_out, ms=ms,
                    path_to_raw=f"live_grid/new_models_grid.json#{task_id}|{kind}|{model_id}",
                )
                n_done += 1
                if n_done % 200 == 0:
                    print(f"[l8-expand]   {model_id}: {n_done}/{n_todo}", flush=True)
        del llm
        print(f"[l8-expand] {model_id} готова за {time.time() - t0:.0f}с ({n_done} новых вызовов)", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"rows": list(cells.values())}, f, ensure_ascii=False, indent=2)
        tmp.replace(out_path)

    return cells


def expand_dataset(new_ids: list, task_ids: list, tasks: dict) -> "LD.LiveDataset":
    """Full Phase 3 wiring: build the new-models grid, patch `MODEL_IDS`
    (correction #5 -- called here, nowhere else), construct a FRESH
    `LiveDataset` (never through `default_dataset()`'s cache — a
    different alphabet needs a genuinely separate instance)."""
    grid_ok = build_grid_and_check_floor(new_ids, task_ids, tasks)
    patch_model_ids(tuple(new_ids))
    ds_expanded = LD.LiveDataset(extra_paths=(LD.NEW_MODELS_GRID,))
    return ds_expanded, grid_ok


def build_grid_and_check_floor(new_ids: list, task_ids: list, tasks: dict) -> bool:
    cells = build_new_models_grid(new_ids, task_ids, tasks, LD.NEW_MODELS_GRID)
    target = len(task_ids) * len(STEP_KINDS) * len(new_ids)
    floor = int(GRID_FLOOR_FRAC * target)
    n_rows = len(cells)
    print(f"[l8-expand] ИТОГО: {n_rows}/{target} ячеек (пол {GRID_FLOOR_FRAC:.0%} = {floor})", flush=True)
    return n_rows >= floor
