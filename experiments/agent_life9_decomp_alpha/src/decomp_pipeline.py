"""decomp_pipeline.py — LIFE-9 Phase C (PROTOCOL.md P4/P5).

Planner call (`B2_MODEL`, see `whole_alpha.py` P1) decomposes a task into
2-4 substeps (JSON: kind in {READ,LOOKUP,COMPUTE} + name + brief — FORMAT
excluded, task's own instruction). Each substep is executed by that
kind's B3-greedy rank-1 model (`orders[kind][0]`, from `lean_seeds.py` on
the panel). Intermediate substeps have NO ground truth of their own (the
plan is LLM-invented, not the task's real named steps) — their extracted
values are only ever forwarded as "already established" context for later
substeps, mirroring `tasks14.step_prompt`'s own upstream-value pattern.
The FINAL check is the last substep's raw text against the task's REAL
final oracle (`task["steps"][-1]`, always numeric, always "shipping_cost"
— every task shares one template, confirmed by reading
`experiment14/tasks/heterostep.py::_build_task`), via the unmodified
`numeric_seam` at the standard 0.011 tolerance — this IS `alpha_decomp`'s/
`r_D_test`'s PASS criterion, nothing else.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD    # noqa: E402
import call_log              # noqa: E402

DECOMP_KINDS = LD.DECOMP_KINDS   # ("READ", "LOOKUP", "COMPUTE")
MIN_SUBSTEPS = 2
MAX_SUBSTEPS = 4

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_LIST_PREFIX_RE = re.compile(r"^\s*\d+[.)](?!\d)\s*")   # same correction as whole_alpha.py, applied defensively here too;
                                                          # (?!\d): don't eat "13.1"'s "13." as if it were list-marker "13. "
                                                          # (real bug caught this session -- see BLOCKERS.md)


def _fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def build_planner_prompt(task: dict) -> str:
    final_instruction = task["steps"][-1]["instruction"]
    return (
        "Запись:\n" + task["record"] + "\n\n"
        f"Итоговая задача: {final_instruction}\n\n"
        "Разбей путь к ответу на 2-4 маленьких подшага. Каждый подшаг — "
        "одного из трёх родов: READ (найти число или факт прямо в записи), "
        "LOOKUP (найти значение в таблице/списке, приведённых в записи), "
        "COMPUTE (вычислить что-то из уже найденных значений). "
        "Не используй под-шаг рода FORMAT.\n\n"
        "Ответь СТРОГО списком JSON, без пояснений и без текста до или "
        "после списка. Формат одного элемента: "
        '{"kind": "READ"|"LOOKUP"|"COMPUTE", "name": "короткий_id", '
        '"brief": "что именно найти или посчитать на этом подшаге"}.'
    )


def parse_plan(raw_text: str):
    """-> list of {"kind","name","brief"} dicts, or None if malformed/
    out-of-range (a planner failure — logged by the caller, not raised;
    still a spent generate() call for the floor check, still not-PASS for
    alpha_decomp/r_D_test)."""
    text = (raw_text or "").strip()
    text = _CODE_FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list) or not (MIN_SUBSTEPS <= len(data) <= MAX_SUBSTEPS):
        return None
    out = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return None
        kind = item.get("kind")
        brief = item.get("brief")
        if kind not in DECOMP_KINDS or not isinstance(brief, str) or not brief.strip():
            return None
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            name = f"step{i}"
        out.append({"kind": kind, "name": name.strip(), "brief": brief.strip()})
    return out


def extract_value(text: str):
    """Best-effort numeric extraction, NO oracle — reuses `numeric_seam`'s
    own tier-1/tier-2 extraction logic via a dummy oracle (`.extracted` is
    populated independent of PASS/FAIL against that dummy), so
    intermediate substeps are extracted-but-never-scored."""
    text = _LIST_PREFIX_RE.sub("", text or "")
    res = LD.seams14.numeric_seam(text, 0.0)
    return res.get("extracted")


def build_executor_prompt(task: dict, substep: dict, established: list) -> str:
    parts = ["Запись:\n" + task["record"], ""]
    if established:
        known = "; ".join(f"{n} = {_fmt(v)}" for n, v in established)
        parts.append("Уже установлено: " + known)
        parts.append("")
    parts.append(substep["brief"])
    parts.append("Выведи ТОЛЬКО одно значение, без пояснений и без имени поля.")
    return "\n".join(parts)


def needed_models(planner_model: str, executor_by_kind: dict) -> list:
    return sorted({planner_model, *executor_by_kind.values()})


def run_decomp_for_task(task: dict, planner_model: str, executor_by_kind: dict, llms: dict) -> dict:
    """`llms`: {model_id: llm} — ALL needed models pre-loaded by the
    caller; this function never calls `load_model`."""
    task_id = task["task_id"]
    calls: list = []

    planner_prompt = build_planner_prompt(task)
    planner_seed = LD.draw_seed_9(task_id, "PLANNER", planner_model)
    t0 = time.time()
    raw = LD.mr11.generate(llms[planner_model], planner_model, planner_prompt,
                           temperature=LD.TEMPERATURE, top_p=1.0, seed=planner_seed,
                           max_tokens=LD.MAX_TOKENS_PLANNER)
    ms = (time.time() - t0) * 1000.0
    n_in, n_out = raw.get("input_tokens") or 0, raw.get("output_tokens") or 0
    calls.append({"role": "PLANNER", "model": planner_model, "seed": planner_seed,
                 "n_in": n_in, "n_out": n_out, "ms": ms,
                 "failed": bool(raw.get("generation_failed"))})
    call_log.append_call_log(model=planner_model, task_id=task_id, step="PLANNER",
                             seed=planner_seed, n_in=n_in, n_out=n_out, ms=ms,
                             path_to_raw=f"decomp:{task_id}:PLANNER")

    if raw.get("generation_failed"):
        return {"task_id": task_id, "status": "PLANNER_CALL_FAILED", "final_pass": False,
               "calls": calls, "plan": None, "substeps": [], "final_extracted": None}

    plan = parse_plan(raw.get("raw_text", ""))
    if plan is None:
        return {"task_id": task_id, "status": "PLANNER_PARSE_FAILED", "final_pass": False,
               "calls": calls, "plan": None, "substeps": [], "final_extracted": None,
               "planner_raw": raw.get("raw_text", "")}

    established: list = []
    substep_records = []
    for i, sub in enumerate(plan):
        model_id = executor_by_kind[sub["kind"]]
        prompt = build_executor_prompt(task, sub, established)
        seed = LD.draw_seed_9(task_id, f"EXEC{i}", model_id)
        t0 = time.time()
        eraw = LD.mr11.generate(llms[model_id], model_id, prompt, temperature=LD.TEMPERATURE,
                                top_p=1.0, seed=seed, max_tokens=LD.MAX_TOKENS_STEP)
        ms = (time.time() - t0) * 1000.0
        en_in, en_out = eraw.get("input_tokens") or 0, eraw.get("output_tokens") or 0
        calls.append({"role": f"EXEC{i}", "model": model_id, "seed": seed,
                     "n_in": en_in, "n_out": en_out, "ms": ms,
                     "failed": bool(eraw.get("generation_failed"))})
        call_log.append_call_log(model=model_id, task_id=task_id, step=f"EXEC{i}", seed=seed,
                                 n_in=en_in, n_out=en_out, ms=ms,
                                 path_to_raw=f"decomp:{task_id}:EXEC{i}")
        raw_text = "" if eraw.get("generation_failed") else eraw.get("raw_text", "")
        value = extract_value(raw_text) if raw_text else None
        substep_records.append({"kind": sub["kind"], "name": sub["name"], "brief": sub["brief"],
                               "raw_output": raw_text, "extracted": value})
        if value is not None:
            established.append((sub["name"], value))

    true_oracle = task["steps"][-1]["oracle"]
    if substep_records:
        last_text = _LIST_PREFIX_RE.sub("", substep_records[-1]["raw_output"] or "")
        final_check = LD.seams14.numeric_seam(last_text, true_oracle)
    else:
        final_check = {"status": "INAPPLICABLE", "extracted": None}
    final_pass = final_check["status"] == "PASS"

    return {"task_id": task_id, "status": "OK", "final_pass": final_pass, "calls": calls,
           "plan": plan, "substeps": substep_records, "final_extracted": final_check.get("extracted")}


def run_decomp_over_tasks(ds, task_ids: list, planner_model: str, executor_by_kind: dict,
                          out_path: Path, *, resume: bool = True) -> dict:
    """Model-major load (each distinct model loaded exactly once), one
    combined pass over ALL `task_ids` (PROTOCOL.md P5 — no separate
    U_dead-only pass). Resumable: results already on disk are kept,
    matching the project's live-call convention."""
    results: dict = {}
    if resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"[decomp] {len(results)} результатов уже на диске (резюме)", flush=True)

    todo = [t for t in task_ids if t not in results]
    if not todo:
        return results

    models = needed_models(planner_model, executor_by_kind)
    llms = {}
    for m in models:
        llm, load_t = LD.mr11.load_model(m)
        llms[m] = llm
        print(f"[decomp] {m} загружена за {load_t:.1f}с", flush=True)

    t0 = time.time()
    for i, t in enumerate(todo):
        task = ds.tasks[t]
        results[t] = run_decomp_for_task(task, planner_model, executor_by_kind, llms)
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            print(f"[decomp] {i + 1}/{len(todo)} задач ({time.time() - t0:.0f}с)", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            tmp.replace(out_path)

    for m in models:
        del llms[m]
    return results
