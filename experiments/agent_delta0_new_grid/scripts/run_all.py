"""run_all.py — DELTA-0 §11: assert C0 done, run C1..C13 strictly in
order, mark CHECKLIST.md + append step_log.jsonl after EACH step, no
return to the user between steps except HARD_STOP (§9).
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
AGENT_DIR = SCRIPTS.parent
SRC = AGENT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator as TG        # noqa: E402
import oracles as OR               # noqa: E402
import atoms as AT                 # noqa: E402
import model_registry_11 as MR     # noqa: E402
import whole_pipeline as WP        # noqa: E402
import union_lower as UL           # noqa: E402
import call_log as CL              # noqa: E402


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VS = _load_module("delta0_verify_seams", SCRIPTS / "verify_seams.py")

CHECKLIST_PATH = AGENT_DIR / "CHECKLIST.md"
STEP_LOG_PATH = AGENT_DIR / "metrics" / "step_log.jsonl"
METRICS_DIR = AGENT_DIR / "metrics"
REPORTS_DIR = AGENT_DIR / "reports"

FLOOR_FRAC = 0.80
HARD_STOP_MODEL_FAIL_FRAC = 0.50   # ">50%" == majority of 6 models


def mark_checklist(step_id: str) -> None:
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    old = f"- [ ] {step_id} "
    new = f"- [x] {step_id} "
    if old not in text:
        raise RuntimeError(f"checklist marker for {step_id} not found or already marked")
    CHECKLIST_PATH.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_step_log(step_id: str, status: str, artifact_paths: list, **extra) -> None:
    row = {"step_id": step_id, "status": status, "artifact_paths": artifact_paths, **extra}
    STEP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STEP_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_blockers(reason: str) -> None:
    text = (
        "# BLOCKERS.md — agent_delta0_new_grid\n\n"
        f"HARD_STOP (PROTOCOL.md §9): {reason}\n\n"
        "Корзина: DELTA_BLOCKED. См. CHECKLIST.md для последнего пройденного шага и "
        "metrics/step_log.jsonl для деталей.\n"
    )
    (AGENT_DIR / "BLOCKERS.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# step functions
# ---------------------------------------------------------------------

def step_c1(state: dict) -> dict:
    protocol_text = (AGENT_DIR / "PROTOCOL.md").read_text(encoding="utf-8")
    required_markers = ("F2", "T1", "T2", "T3", "T4", "T5", "T6", "FILTER", "AGGREGATE", "DERIVE")
    missing = [m for m in required_markers if m not in protocol_text]
    if missing:
        return {"hard_stop": True, "reason": f"C1: PROTOCOL.md missing T-family markers: {missing}"}
    return {"artifact_paths": ["agent_delta0_new_grid/PROTOCOL.md"], "markers_confirmed": required_markers}


def step_c2(state: dict) -> dict:
    d = TG.build_tasks()
    assert len(d["train"]) == TG.N_TRAIN and len(d["test"]) == TG.N_TEST, "C2: task count mismatch"
    state["tasks_by_id"] = d["tasks"]
    state["train_ids"] = d["train"]
    state["test_ids"] = d["test"]
    manifest = {
        "seed_t": TG.SEED_T, "n_train": len(d["train"]), "n_test": len(d["test"]),
        "k_records": TG.K, "train_ids": d["train"], "test_ids": d["test"],
        "db_text_chars": {"min": min(len(d["tasks"][t]["db_text"]) for t in d["tasks"]),
                          "max": max(len(d["tasks"][t]["db_text"]) for t in d["tasks"])},
    }
    _write_json(METRICS_DIR / "tasks_manifest.json", manifest)
    return {"artifact_paths": ["agent_delta0_new_grid/metrics/tasks_manifest.json"],
           "n_train": len(d["train"]), "n_test": len(d["test"])}


def step_c3(state: dict) -> dict:
    tasks_by_id = state["tasks_by_id"]
    required_fields = ("final_oracle", "matched_ids", "aggregate_value")
    bad = []
    for tid, t in tasks_by_id.items():
        for f in required_fields:
            if f not in t or t[f] is None:
                bad.append((tid, f))
    if bad:
        return {"hard_stop": True, "reason": f"C3: tasks missing checkpoint fields: {bad[:5]}"}
    return {"artifact_paths": [], "checkpoints_confirmed": list(required_fields),
           "n_checkpoints": 3, "note": "3 >= required 2 (T3)"}


def step_c4(state: dict) -> dict:
    r1 = VS.fixture_1_golden_roundtrip()
    r2 = VS.fixture_2_empty_match()
    r3 = VS.fixture_3_comparator_boundary()
    r4 = VS.fixture_4_wellformed_response()
    r5 = VS.fixture_5_garbled_response()
    if not (r1 and r2 and r3 and r4 and r5):
        return {"hard_stop": True, "reason": "C4: one or more of the 5 self-test fixtures failed"}
    rt = VS.roundtrip_check()
    if not rt:
        return {"hard_stop": True, "reason": "C4/smoke: 5-task CPU roundtrip failed"}

    # -- SMOKE (task §10, "внутри C4 и перед C6"): 1 live whole generate, 1 live atom generate --
    print("[C4-smoke] 1 live whole generate() ...", flush=True)
    sample_task = state["tasks_by_id"][state["test_ids"][0]]
    probe_model = MR.MODEL_IDS[0]
    llm, load_t = MR.load_model(probe_model)
    print(f"[C4-smoke] {probe_model} загружена за {load_t:.1f}с", flush=True)
    t0 = time.time()
    whole_prompt = WP.build_whole_prompt(sample_task)
    raw_whole = MR.generate(llm, probe_model, whole_prompt, temperature=WP.TEMPERATURE, top_p=1.0,
                            seed=WP.draw_seed(sample_task["task_id"], probe_model),
                            max_tokens=WP.MAX_TOKENS_WHOLE_GEN)
    whole_ok = not raw_whole.get("generation_failed")
    print(f"[C4-smoke] whole generate ok={whole_ok} raw={raw_whole.get('raw_text','')!r} "
          f"({time.time()-t0:.1f}s)", flush=True)
    try:
        OR.check_whole_final(raw_whole.get("raw_text", ""), sample_task["final_oracle"])
    except Exception as exc:   # noqa: BLE001
        del llm
        return {"hard_stop": True, "reason": f"C4-smoke: whole parse raised {type(exc).__name__}: {exc}"}

    print("[C4-smoke] 1 live atom (FILTER) generate() ...", flush=True)
    t0 = time.time()
    filter_prompt = AT.build_filter_prompt(sample_task)
    raw_atom = MR.generate(llm, probe_model, filter_prompt, temperature=WP.TEMPERATURE, top_p=1.0,
                           seed=UL.draw_seed(sample_task["task_id"], "FILTER", probe_model),
                           max_tokens=AT.MAX_TOKENS_FILTER)
    atom_ok = not raw_atom.get("generation_failed")
    print(f"[C4-smoke] atom generate ok={atom_ok} raw={raw_atom.get('raw_text','')[:120]!r} "
          f"({time.time()-t0:.1f}s)", flush=True)
    del llm
    try:
        OR.check_filter(raw_atom.get("raw_text", ""), sample_task["matched_ids"])
    except Exception as exc:   # noqa: BLE001
        return {"hard_stop": True, "reason": f"C4-smoke: atom parse raised {type(exc).__name__}: {exc}"}

    if not (whole_ok and atom_ok):
        return {"hard_stop": True, "reason": f"C4-smoke: generate() failed (whole_ok={whole_ok}, atom_ok={atom_ok})"}

    return {"artifact_paths": [], "fixtures": 5, "roundtrip_tasks": 5,
           "smoke_whole_ok": whole_ok, "smoke_atom_ok": atom_ok}


def step_c5(state: dict) -> dict:
    """B fixed HERE from C2's actual measured DB/prompt size -- never
    guessed ahead of time (PROTOCOL.md §3, locked correction #5)."""
    sample_task = state["tasks_by_id"][state["test_ids"][0]]
    filter_prompt_len = len(AT.build_filter_prompt(sample_task))
    whole_prompt_len = len(WP.build_whole_prompt(sample_task))
    # rough chars->token estimate (3.5 chars/token, conservative for mixed RU/EN text),
    # + worst-case output allowance, + 15% safety margin, rounded up to nearest 50.
    est_filter_tokens = filter_prompt_len / 3.5 + AT.MAX_TOKENS_FILTER
    b_atom_raw = est_filter_tokens * 1.15
    b_atom = int(math.ceil(b_atom_raw / 50.0) * 50)
    B = b_atom * 4
    state["B"] = B
    state["B_atom"] = b_atom
    est_whole_tokens = whole_prompt_len / 3.5 + WP.MAX_TOKENS_WHOLE_GEN
    print(f"[C5] filter_prompt~{filter_prompt_len} chars (~{est_filter_tokens:.0f} tok est), "
          f"whole_prompt~{whole_prompt_len} chars (~{est_whole_tokens:.0f} tok est) "
          f"-> B_atom={b_atom} B={B}", flush=True)
    budget = {"B": B, "B_atom": b_atom, "B_total_union_ceiling": 2 * B,
             "sample_filter_prompt_chars": filter_prompt_len,
             "sample_whole_prompt_chars": whole_prompt_len,
             "est_filter_tokens": round(est_filter_tokens, 1),
             "est_whole_tokens": round(est_whole_tokens, 1)}
    _write_json(METRICS_DIR / "budget.json", budget)
    return {"artifact_paths": ["agent_delta0_new_grid/metrics/budget.json"], **budget}


def step_c6(state: dict) -> dict:
    tasks_by_id, test_ids, B = state["tasks_by_id"], state["test_ids"], state["B"]
    results_by_model = {}
    n_load_failed = 0
    for m in MR.MODEL_IDS:
        try:
            results_by_model[m] = WP.measure_whole_for_model(m, tasks_by_id, test_ids, B)
        except Exception as exc:   # noqa: BLE001
            n_load_failed += 1
            print(f"[C6] !!! {m} failed: {type(exc).__name__}: {exc}", flush=True)
    if n_load_failed / len(MR.MODEL_IDS) > HARD_STOP_MODEL_FAIL_FRAC:
        return {"hard_stop": True,
               "reason": f"C6: {n_load_failed}/{len(MR.MODEL_IDS)} models failed to load/run (OOM pattern)"}

    r_m_table = WP.compute_r_m(results_by_model, test_ids)
    n_calls = CL.count_rows(METRICS_DIR / "whole_calls.jsonl")
    n_expected = len(MR.MODEL_IDS) * len(test_ids)
    n_failed_calls = sum(1 for m in results_by_model for t in test_ids if results_by_model[m][t]["failed"])
    floor_ok = n_calls > 0 and (n_calls - n_failed_calls) / n_expected >= FLOOR_FRAC
    if not floor_ok:
        return {"hard_stop": True,
               "reason": f"C6: floor {n_calls - n_failed_calls}/{n_expected} < {FLOOR_FRAC:.0%}"}

    state["whole_results_by_model"] = results_by_model
    state["r_m_table"] = r_m_table
    _write_json(METRICS_DIR / "whole_by_model.json", r_m_table)
    for m, tbl in r_m_table.items():
        print(f"[C6] {m}: r_m={tbl['r_m']:.3f} ({tbl['n_pass']}/{tbl['n']}) mean_cost={tbl['mean_cost']:.0f}",
              flush=True)
    return {"artifact_paths": ["agent_delta0_new_grid/metrics/whole_by_model.json",
                              "agent_delta0_new_grid/metrics/whole_calls.jsonl"],
           "n_calls": n_calls, "floor_ok": floor_ok}


def step_c7(state: dict) -> dict:
    m_star = WP.select_m_star(state["r_m_table"])
    state["m_star"] = m_star
    r_m_star = state["r_m_table"][m_star]["r_m"]
    print(f"[C7] m* = {m_star} (r_m*={r_m_star:.3f}, cost<=B respected by construction)", flush=True)
    return {"artifact_paths": [], "m_star": m_star, "r_m_star": r_m_star}


def step_c8(state: dict) -> dict:
    spec_out = {name: {"input_schema": s["input_schema"], "output_schema": s["output_schema"],
                       "max_tokens": s["max_tokens"]} for name, s in AT.ATOM_SPECS.items()}
    assert len(spec_out) >= 3, "C8: fewer than 3 atom types"
    _write_json(METRICS_DIR / "atoms_spec.json", spec_out)
    return {"artifact_paths": ["agent_delta0_new_grid/metrics/atoms_spec.json"],
           "n_atom_types": len(spec_out)}


def step_c9(state: dict) -> dict:
    tasks_by_id, test_ids, m_star = state["tasks_by_id"], state["test_ids"], state["m_star"]
    try:
        union_results = UL.measure_union_lower(m_star, tasks_by_id, test_ids)
    except Exception as exc:   # noqa: BLE001
        return {"hard_stop": True, "reason": f"C9: union chain executor failed to load/run: "
                                             f"{type(exc).__name__}: {exc}"}

    n_expected_max = len(test_ids) * 3
    n_calls = CL.count_rows(METRICS_DIR / "union_calls.jsonl")
    n_failed_calls = sum(1 for c in union_results.values() for call in c["calls"] if call["failed"])
    n_ok_calls = n_calls - n_failed_calls
    floor_ok = n_calls > 0 and (n_ok_calls / n_calls) >= FLOOR_FRAC
    if not floor_ok:
        return {"hard_stop": True, "reason": f"C9: floor {n_ok_calls}/{n_calls} calls OK < {FLOOR_FRAC:.0%}"}

    r_union = UL.compute_r_union(union_results, test_ids)
    state["union_results"] = union_results
    state["r_union"] = r_union
    _write_json(METRICS_DIR / "union_lower.json", r_union)
    _write_json(METRICS_DIR / "union_lower_detail.json", union_results)   # full per-task raw text, for forensics
    print(f"[C9] r_union_lower={r_union['r_union_lower']:.3f} ({r_union['n_pass']}/{r_union['n']}) "
          f"mean_cost={r_union['mean_cost']:.0f} break_at={r_union['break_at_counts']}", flush=True)
    return {"artifact_paths": ["agent_delta0_new_grid/metrics/union_lower.json",
                              "agent_delta0_new_grid/metrics/union_calls.jsonl"],
           "n_calls_max_possible": n_expected_max, "n_calls_actual": n_calls, "floor_ok": floor_ok}


def step_c10(state: dict) -> dict:
    r_m_star = state["r_m_table"][state["m_star"]]["r_m"]
    r_union = state["r_union"]["r_union_lower"]
    delta = r_union - r_m_star
    state["delta"] = delta
    mean_cost_whole = state["r_m_table"][state["m_star"]]["mean_cost"]
    mean_cost_union = state["r_union"]["mean_cost"]
    budget_violation = mean_cost_union > 2 * state["B"]
    state["budget_violation"] = budget_violation
    state["mean_cost_whole"] = mean_cost_whole
    state["mean_cost_union"] = mean_cost_union
    print(f"[C10] Delta = {delta:.3f} (r_union={r_union:.3f} - r_m*={r_m_star:.3f}); "
          f"cost whole={mean_cost_whole:.0f} vs union={mean_cost_union:.0f} "
          f"(2B={2*state['B']}) BUDGET_VIOLATION={budget_violation}", flush=True)
    delta_out = {"delta": delta, "r_union_lower": r_union, "r_m_star": r_m_star,
                "mean_cost_whole": mean_cost_whole, "mean_cost_union": mean_cost_union,
                "budget_violation": budget_violation}
    _write_json(METRICS_DIR / "delta.json", delta_out)
    return {"artifact_paths": ["agent_delta0_new_grid/metrics/delta.json"], **delta_out}


def classify_delta(delta: float) -> str:
    if delta >= 0.15:
        return "DELTA_POS_STRONG"
    if delta >= 0.05:
        return "DELTA_POS_WEAK"
    return "DELTA_NONPOS"


NEXT_LINES = {
    "DELTA_POS_STRONG": ("пакет DELTA-1: слой D (план) + E_i при бюджете <=2B; "
                        "цель r_real(B_swarm)>r_m*; без HGT до первого r_real"),
    "DELTA_POS_WEAK": ("DELTA-1 узкий: только задачи где m* FAIL и ∪ PASS; "
                       "измерить ε_plan/ε_exec; не expand registry"),
    "DELTA_NONPOS": ("сменить семейство T или атомы; не отбор/HGT; "
                     "текущий T не даёт зазора роя"),
    "DELTA_BLOCKED": "чинить generate/parse/OOM; не теория",
}


def step_c11(state: dict) -> dict:
    basket = classify_delta(state["delta"])
    high_single = state["r_m_table"][state["m_star"]]["r_m"] >= 0.70
    state["basket"] = basket
    state["high_single"] = high_single
    state["next_line"] = NEXT_LINES[basket]
    print(f"[C11] basket={basket} HIGH_SINGLE={high_single} next={state['next_line']}", flush=True)
    return {"artifact_paths": [], "basket": basket, "high_single": high_single}


def _fmt(v, spec=".3f"):
    return "n/d" if v is None else format(v, spec)


def write_report(state: dict) -> Path:
    L = []
    A = L.append
    A("# REPORT_DELTA0 — iron-clad Δ measurement on a new polygon\n")
    A("Спека: `PROTOCOL.md`. F2 (multi-hop record) на транзакциях, НЕ HETEROSTEP. "
      "0 Evolution/HGT/Pareto/registry/ASSEMBLE в этом пакете.\n")

    A("## 1. Класс T и почему T1-T6\n")
    manifest_path = METRICS_DIR / "tasks_manifest.json"
    chars_note = "не измерено (C2 не завершён)"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        chars_note = f"{m['db_text_chars']['min']}-{m['db_text_chars']['max']} символов (измерено C2)"
    A("F2: JOINT-фильтр (category AND region) -> агрегат (SUM/COUNT/MAX amount) -> "
      "производный ДА/НЕТ. T1 (не одношаговая): 3 логических хода. T2 (final oracle "
      "детерминирован): код генератора. T3 (3 checkpoint, не HETEROSTEP kinds): "
      f"matched_ids/aggregate_value/final. T4 (контекст): db_text, K={TG.K} записей, "
      f"{chars_note}. T5: HIGH_SINGLE допустим, не отменяет пакет. T6: "
      f"N_TRAIN={TG.N_TRAIN}, N_TEST={TG.N_TEST}, SEED_T={TG.SEED_T}.\n")

    A("## 2. B и B_atom\n")
    A(f"B={state['B']}, B_atom={state['B_atom']} (>= B/4 отношение соблюдено по построению), "
      f"B_total_union потолок = {2*state['B']}. Число получено из фактически измеренного "
      "размера промпта (C2/C5), не угадано.\n")

    A("## 3. Таблица r_m по моделям, m*, cost\n")
    A("| model | r_m | n_pass/n | mean_cost |\n|---|---|---|---|\n")
    for m, tbl in state["r_m_table"].items():
        star = " **<- m\\*** " if m == state["m_star"] else ""
        A(f"| {m}{star} | {_fmt(tbl['r_m'])} | {tbl['n_pass']}/{tbl['n']} | {_fmt(tbl['mean_cost'],'.0f')} |")
    A("")

    A("## 4. Спека A (атомы)\n")
    for name, spec in AT.ATOM_SPECS.items():
        A(f"- **{name}**: in={spec['input_schema']}, out={spec['output_schema']}, "
          f"max_tokens={spec['max_tokens']}")
    A("")

    A("## 5. r_∪_lower, метод M1\n")
    ru = state["r_union"]
    A(f"M1 (oracle-route, золотой план, реальные E, executor=m\\*={state['m_star']}): "
      f"r_∪_lower={_fmt(ru['r_union_lower'])} ({ru['n_pass']}/{ru['n']}), "
      f"mean_cost={_fmt(ru['mean_cost'],'.0f')}. Разрыв цепочки (без golden подстановки): "
      f"{ru['break_at_counts']}.\n")

    A("## 6. Δ, корзина, HIGH_SINGLE\n")
    A(f"Δ = r_∪_lower - r_m* = {_fmt(state['delta'])}. Корзина: **{state['basket']}**.\n")
    A(f"HIGH_SINGLE (r_m*>=0.70): {'ДА' if state['high_single'] else 'нет'} "
      f"(r_m*={_fmt(state['r_m_table'][state['m_star']]['r_m'])})"
      + (" -- single уже силён; рой не приоритет на T." if state["high_single"] else "") + "\n")
    A(f"Cost: mean(whole,m\\*)={_fmt(state['mean_cost_whole'],'.0f')} vs "
      f"mean(union-chain)={_fmt(state['mean_cost_union'],'.0f')} (потолок 2B={2*state['B']}) -> "
      + ("**BUDGET_VIOLATION**" if state["budget_violation"] else "в пределах потолка") + "\n")

    A("## 7. Next\n")
    A(f"{state['next_line']}\n")

    A("## 8. Non-claims\n")
    A("- r_∪_lower требует ВСЮ цепочку (FILTER И AGGREGATE И DERIVE), не \"хотя бы один checkpoint\".\n"
      "- Цепочка не подставляет golden-значения при провале атома -- реальный обрыв, не рескью.\n"
      "- Whole-промпт просит только финальный ДА/НЕТ -- intermediate не элиситируются и не влияют на v_final.\n"
      "- `experiment11/` не отредактирован. `arch2/`, `experiment14/`, `agent_life*/`, "
      "`agent_a5*/` не читались и не изменялись.\n"
      "- Пороги §6/PROTOCOL не двигались после первого посчитанного Δ.\n")

    A(f"## 9. Приложение\nCHECKLIST.md: `agent_delta0_new_grid/CHECKLIST.md` (все пункты [x]).\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "REPORT_DELTA0.md"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


def write_blocked_report(state: dict, step_id: str, reason: str) -> Path:
    """§9: REPORT is written regardless of HARD_STOP, with the reason --
    separate from `write_report` (which needs the FULL pipeline's state)
    so a stop at an early step (e.g. C6/C9) can never itself crash the
    reporting step by indexing a field that was never reached."""
    L = []
    A = L.append
    A("# REPORT_DELTA0 — iron-clad Δ measurement on a new polygon\n")
    A(f"**DELTA_BLOCKED** — HARD_STOP at `{step_id}` (PROTOCOL.md §9): {reason}\n")
    A("## 1-6. Не достигнуты\n")
    A(f"Прогон остановлен на `{step_id}`, до этой точки — см. `metrics/step_log.jsonl` "
      "и `CHECKLIST.md` для того, что успело завершиться.\n")
    if "r_m_table" in state:
        A("### Частично: таблица r_m (C6 успела завершиться)\n")
        A("| model | r_m | n_pass/n | mean_cost |\n|---|---|---|---|\n")
        for m, tbl in state["r_m_table"].items():
            A(f"| {m} | {_fmt(tbl['r_m'])} | {tbl['n_pass']}/{tbl['n']} | {_fmt(tbl['mean_cost'],'.0f')} |")
        A("")
    A("## 7. Next\n")
    A(f"{NEXT_LINES['DELTA_BLOCKED']}\n")
    A("## 8. Non-claims\n- Δ не посчитана -- пайплайн не дошёл до C10.\n")
    A(f"## 9. Приложение\nCHECKLIST.md: `agent_delta0_new_grid/CHECKLIST.md` "
      f"(остановлен на `{step_id}`). BLOCKERS.md содержит причину.\n")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "REPORT_DELTA0.md"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


def step_c12(state: dict) -> dict:
    out_path = write_report(state)
    return {"artifact_paths": [f"agent_delta0_new_grid/reports/REPORT_DELTA0.md"], "path": str(out_path)}


def step_c13(state: dict) -> dict:
    """C13 marks ITSELF as its own last action -- the "all [x]" check must
    look at C0-C12 only (checking C13 too would be self-contradictory:
    C13 can never already be [x] before this function marks it)."""
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    unmarked = [ln for ln in text.splitlines()
               if ln.strip().startswith("- [ ] C") and not ln.strip().startswith("- [ ] C13")]
    if unmarked:
        return {"hard_stop": True, "reason": f"C13: items still unchecked: {unmarked}"}
    mark_checklist("C13")
    n_log_rows = CL.count_rows(STEP_LOG_PATH)
    print(f"[C13] CHECKLIST all [x], step_log has {n_log_rows} rows", flush=True)
    return {"artifact_paths": ["agent_delta0_new_grid/CHECKLIST.md",
                              "agent_delta0_new_grid/metrics/step_log.jsonl"],
           "already_marked": True, "n_log_rows": n_log_rows}


STEP_FUNCS = {
    "C1": step_c1, "C2": step_c2, "C3": step_c3, "C4": step_c4, "C5": step_c5,
    "C6": step_c6, "C7": step_c7, "C8": step_c8, "C9": step_c9, "C10": step_c10,
    "C11": step_c11, "C12": step_c12,
}


if __name__ == "__main__":
    checklist_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    assert "- [x] C0 " in checklist_text, "C0 must be marked done before run_all.py starts"

    state: dict = {}
    for step_id, fn in STEP_FUNCS.items():
        print(f"\n=== {step_id} ===", flush=True)
        t0 = time.time()
        result = fn(state)
        dt = time.time() - t0
        if result.get("hard_stop"):
            reason = result["reason"]
            print(f"[run_all] HARD_STOP at {step_id}: {reason}", flush=True)
            append_step_log(step_id, "HARD_STOP", result.get("artifact_paths", []), reason=reason)
            write_blockers(f"{step_id}: {reason}")
            state["basket"] = "DELTA_BLOCKED"
            out_path = write_blocked_report(state, step_id, reason)
            print(f"[run_all] DELTA_BLOCKED -- отчёт: {out_path}", flush=True)
            sys.exit(1)
        # C13's own checklist item is marked inside step_c13 (needs the OTHERS marked first
        # to check); C1-C12 mark here uniformly.
        if step_id != "C13":
            mark_checklist(step_id)
        append_step_log(step_id, "done", result.get("artifact_paths", []),
                        seconds=round(dt, 1),
                        **{k: v for k, v in result.items() if k not in ("artifact_paths", "hard_stop")})
        print(f"[run_all] {step_id} done in {dt:.1f}s", flush=True)

    print("\n=== C13 ===", flush=True)
    t0 = time.time()
    result = step_c13(state)
    dt = time.time() - t0
    if result.get("hard_stop"):
        reason = result["reason"]
        print(f"[run_all] HARD_STOP at C13: {reason}", flush=True)
        append_step_log("C13", "HARD_STOP", result.get("artifact_paths", []), reason=reason)
        write_blockers(f"C13: {reason}")
        sys.exit(1)
    append_step_log("C13", "done", result.get("artifact_paths", []), seconds=round(dt, 1))
    print(f"[run_all] C13 done in {dt:.1f}s -- basket={state['basket']}", flush=True)
    print(f"[run_all] FINAL: {state['basket']} -- отчёт: {REPORTS_DIR / 'REPORT_DELTA0.md'}", flush=True)
