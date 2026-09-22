"""baselines.py — PROTOCOL.md §5: B0-B3.

B1/B2/B3 — ЧИСТАЯ конструкция генотипа из уже построенной live_grid
(`arch2.heterostep_seeds`, импорт без изменений — читает только
`ds.models`/`ds.cells[...]["status"]`) — новых вызовов `generate()` здесь
нет. Их r/c на full TRAIN/TEST считает `orchestrator.py` той же машинерией
(`ev.evaluate` + `arch2.fitness.metrics`), что и эволюционные генотипы —
не отдельным путём.

B0 — лучшая ОДИНОЧНАЯ модель на целостном плече, решено ДАННЫМИ (не
зафиксированная заранее qwen3): живые вызовы `whole_prompt` для всех 6
моделей на TRAIN (выбор победителя по доле задач, где ВСЕ шаги верны — та
же строгая метрика, что `experiment14/harness.py::all_steps_rates`),
затем только победитель — на TEST. Кэш — `metrics/live_grid/whole_grid.json`
(отдельно от шаговой сетки: другая форма промпта/бюджет токенов).
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

MODEL_IDS = LD.HSTEP.MODEL_IDS


def _atomic_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _load_whole_grid() -> dict:
    if LD.LIVE_WHOLE_GRID.exists():
        with open(LD.LIVE_WHOLE_GRID, encoding="utf-8") as f:
            data = json.load(f)
        return {(r["task_id"], r["model"]): r for r in data["rows"]}
    return {}


def _flush_whole_grid(rows: dict) -> None:
    _atomic_write(LD.LIVE_WHOLE_GRID, {"rows": list(rows.values())})


def run_whole_live(ds, task_id: str, model_id: str, llm, rows: dict) -> dict:
    key = (task_id, model_id)
    if key in rows:
        return rows[key]
    task = ds.tasks[task_id]
    prompt = LD.tasks14.whole_prompt(task)
    seed = LD.draw_seed(task_id, "WHOLE", model_id)
    t0 = time.time()
    raw = LD.mr11.generate(llm, model_id, prompt, temperature=LD.TEMPERATURE, top_p=1.0,
                           seed=seed, max_tokens=LD.MAX_TOKENS_WHOLE)
    ms = (time.time() - t0) * 1000.0
    seam = LD.seams14.whole_seam(raw.get("raw_text", ""), task["steps"])
    final = seam["step_results"][-1] if seam["step_results"] else {"status": LD.seams14.INAPPLICABLE}
    n_in = raw.get("input_tokens") or 0
    n_out = raw.get("output_tokens") or 0
    row = {
        "task_id": task_id, "model": model_id, "raw_output": raw.get("raw_text", ""),
        "step_results": seam["step_results"], "tokens": n_in + n_out,
        "latency": raw.get("generation_time_sec"), "seed": seed,
        "final_status": final["status"],
        "resolved_all_steps": all(s["status"] == LD.seams14.PASS for s in seam["step_results"]),
        "timestamp": time.time(),
    }
    rows[key] = row
    call_log.append_call_log(
        model=model_id, task_id=task_id, step="WHOLE", seed=seed, n_in=n_in, n_out=n_out, ms=ms,
        path_to_raw=f"live_grid/whole_grid.json#{task_id}|WHOLE|{model_id}",
    )
    return row


def build_b0(ds) -> dict:
    """-> {"model": winner_id, "train_rate": ..., "test_rate": ...,
    "train_rates_all_models": {...}}."""
    rows = _load_whole_grid()
    train_ids = ds.split["train"]
    test_ids = ds.split["test"]

    for model_id in MODEL_IDS:
        n_todo = sum(1 for t in train_ids if (t, model_id) not in rows)
        if n_todo == 0:
            continue
        llm, load_t = LD.mr11.load_model(model_id)
        print(f"[a5-b0] {model_id} загружена за {load_t:.1f}с — {n_todo} TRAIN-вызовов", flush=True)
        t0 = time.time()
        for task_id in train_ids:
            run_whole_live(ds, task_id, model_id, llm, rows)
        del llm
        print(f"[a5-b0] {model_id} TRAIN готов за {time.time() - t0:.0f}с", flush=True)
        _flush_whole_grid(rows)

    def all_steps_rate(model_id: str, ids: list) -> float:
        hits = sum(1 for t in ids if (rows.get((t, model_id)) or {}).get("resolved_all_steps"))
        return hits / len(ids) if ids else 0.0

    train_rates = {m: all_steps_rate(m, train_ids) for m in MODEL_IDS}
    winner = max(train_rates, key=train_rates.get)
    print(f"[a5-b0] TRAIN all-steps rates: {train_rates}")
    print(f"[a5-b0] победитель B0 = {winner} (train_rate={train_rates[winner]:.4f})")

    n_todo_test = sum(1 for t in test_ids if (t, winner) not in rows)
    if n_todo_test:
        llm, load_t = LD.mr11.load_model(winner)
        print(f"[a5-b0] {winner} загружена за {load_t:.1f}с — {n_todo_test} TEST-вызовов", flush=True)
        t0 = time.time()
        for task_id in test_ids:
            run_whole_live(ds, task_id, winner, llm, rows)
        del llm
        print(f"[a5-b0] {winner} TEST готов за {time.time() - t0:.0f}с", flush=True)
        _flush_whole_grid(rows)

    test_rate = all_steps_rate(winner, test_ids)
    return {"model": winner, "train_rate": train_rates[winner], "test_rate": test_rate,
           "train_rates_all_models": train_rates}


def build_b1_b2_b3(ds, train_ids: list) -> dict:
    """PROTOCOL.md §5: B1/B2/B3 — функция от `live_grid`. Маршрут учится
    ТОЛЬКО на TRAIN (без hindsight на TEST)."""
    b1 = LD.HSEED.b1_route(ds, train_ids)
    b2 = LD.HSEED.b2_route(ds, train_ids)
    b3 = LD.HSEED.b3_route(ds, train_ids)
    return {
        "REF_B1_route": LD.G.genotype(LD.HSEED.route(b1), gen=0, origin="seed:REF_B1_route"),
        "REF_B2_greedy2": LD.G.genotype(LD.HSEED.route(b2), gen=0, origin="seed:REF_B2_greedy2"),
        "REF_B3_greedy_cover": LD.G.genotype(LD.HSEED.route(b3), gen=0, origin="seed:REF_B3_greedy_cover"),
        "_routes": {"b1": b1, "b2": b2, "b3": b3},
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ds = LD.default_dataset(fresh=True)
    cov = ds.coverage(list(ds.tasks.keys()))
    print(f"[a5-b0] live_grid покрытие: {cov}")
    if cov["coverage"] < 1.0:
        print("[a5-b0] ПОКРЫТИЕ < 100% — live_grid_builder.py должен закончить первым. Стоп.")
        sys.exit(1)
    b0 = build_b0(ds)
    b123 = build_b1_b2_b3(ds, ds.split["train"])
    summary_path = SRC.parent / "metrics" / "baselines.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"b0": b0, "routes": b123["_routes"]}, f, ensure_ascii=False, indent=2)
    print(f"[a5-b0] записано {summary_path}")
