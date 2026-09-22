"""live_grid_builder.py — PROTOCOL.md §3.1: единственная тяжёлая фаза
generate() в этом пакете. Модель-мажорный обход (как `experiment14/
harness.py::run_train_grid`, `agent_a5_live_m/src/live_grid_builder.py`),
резюмируемо (ячейки, уже на диске, не перевызываются), каждый вызов --
строка `live_call_log.jsonl`.
"""

from __future__ import annotations

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
MODEL_IDS = LD.HSTEP.MODEL_IDS


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _load_existing_cells() -> dict:
    cells: dict = {}
    for path in (LD.LIVE_TRAIN_GRID, LD.LIVE_TEST_GRID):
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for r in data["rows"]:
                cells[(r["task_id"], r["kind"], r["model"])] = r
    return cells


def _flush_grid(cells: dict, train_ids: set) -> None:
    train_rows = [r for (t, k, m), r in cells.items() if t in train_ids]
    test_rows = [r for (t, k, m), r in cells.items() if t not in train_ids]
    _atomic_write(LD.LIVE_TRAIN_GRID, {"rows": train_rows})
    _atomic_write(LD.LIVE_TEST_GRID, {"rows": test_rows})


def build_live_grid(task_ids: list, tasks: dict, split: dict) -> dict:
    cells = _load_existing_cells()
    train_ids = set(split["train"])
    n_before = len(cells)
    print(f"[l3-grid] на диске уже {n_before} ячеек (резюме)", flush=True)

    for model_id in MODEL_IDS:
        n_todo = sum(1 for t in task_ids for k in STEP_KINDS if (t, k, model_id) not in cells)
        if n_todo == 0:
            print(f"[l3-grid] {model_id} уже полностью на диске, пропуск загрузки", flush=True)
            continue
        llm, load_t = LD.mr11.load_model(model_id)
        print(f"[l3-grid] {model_id} загружена за {load_t:.1f}с — {n_todo} ячеек в очереди", flush=True)
        t0 = time.time()
        n_done_this_model = 0
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
                grid_file = "train_grid.json" if task_id in train_ids else "test_grid.json"

                row = {
                    "experiment_id": "life3_block_live", "phase": "grid", "task_id": task_id,
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
                    path_to_raw=f"live_grid/{grid_file}#{task_id}|{kind}|{model_id}",
                )
                n_done_this_model += 1
                if n_done_this_model % 200 == 0:
                    print(f"[l3-grid]   {model_id}: {n_done_this_model}/{n_todo}", flush=True)
        del llm
        print(f"[l3-grid] {model_id} готова за {time.time() - t0:.0f}с "
              f"({n_done_this_model} новых вызовов)", flush=True)
        _flush_grid(cells, train_ids)

    return cells


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ds = LD.default_dataset()
    all_ids = sorted(ds.tasks)
    t_start = time.time()
    cells = build_live_grid(all_ids, ds.tasks, ds.split)
    n_rows = len(cells)
    target = len(all_ids) * len(STEP_KINDS) * len(MODEL_IDS)
    train_floor = int(0.8 * 6 * 4 * 100)
    n_train_rows = sum(1 for (t, k, m) in cells if t in set(ds.split["train"]))
    print(f"[l3-grid] ИТОГО: {n_rows}/{target} ячеек за {time.time() - t_start:.0f}с "
          f"(train-строк {n_train_rows}, приёмочный минимум {train_floor})")
    if n_train_rows < train_floor:
        print("[l3-grid] НИЖЕ ПРИЁМОЧНОГО МИНИМУМА -- писать в BLOCKERS.md", flush=True)
        sys.exit(1)
    print("[l3-grid] OK", flush=True)
