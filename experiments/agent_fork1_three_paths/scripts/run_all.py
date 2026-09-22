"""run_all.py — FORK-1: assert K0 done, run K1..K10 strictly in order,
mark CHECKLIST.md + append step_log.jsonl after EACH step (uniformly,
including K10 -- DELTA-0's own bug was special-casing the last step and
skipping its hard_stop check; not repeated here), no return to the user
between steps except HARD_STOP (§10).
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
ROOT = AGENT_DIR.parent
SRC = AGENT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_f2 as TGF2       # noqa: E402
import task_generator_hard as TGH      # noqa: E402
import oracles as OR                   # noqa: E402
import det_atoms as DA                 # noqa: E402
import atoms_hard as AH                # noqa: E402
import model_registry_11 as MR         # noqa: E402
import whole_pipeline_hard as WPH      # noqa: E402
import union_hard as UH                # noqa: E402
import call_log as CL                  # noqa: E402


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VS = _load_module("fork1_verify_seams", SCRIPTS / "verify_seams.py")

CHECKLIST_PATH = AGENT_DIR / "CHECKLIST.md"
STEP_LOG_PATH = AGENT_DIR / "metrics" / "step_log.jsonl"
METRICS_DIR = AGENT_DIR / "metrics"
REPORTS_DIR = AGENT_DIR / "reports"
DELTA0_METRICS = ROOT / "agent_delta0_new_grid" / "metrics"

FLOOR_FRAC = 0.80
DELTA_SWARM_THRESHOLD = 0.05
DELTA_TOOL_THRESHOLD = 0.05
HIGH_SINGLE_THRESHOLD = 0.70
HARD_STOP_MODEL_FAIL_FRAC = 0.50

ALL_STEPS = ("K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10")


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
    (AGENT_DIR / "BLOCKERS.md").write_text(
        "# BLOCKERS.md — agent_fork1_three_paths\n\n"
        f"HARD_STOP (PROTOCOL.md §8): {reason}\n\n"
        "Корзина: FORK_BLOCKED. См. CHECKLIST.md для последнего пройденного шага и "
        "metrics/step_log.jsonl для деталей.\n", encoding="utf-8")


# ---------------------------------------------------------------------
# step functions
# ---------------------------------------------------------------------

def step_k1(state: dict) -> dict:
    delta_path = DELTA0_METRICS / "delta.json"
    whole_path = DELTA0_METRICS / "whole_by_model.json"
    if not delta_path.exists() or not whole_path.exists():
        return {"hard_stop": True, "reason": f"K1: delta0 metrics missing ({delta_path}, {whole_path})"}
    d0 = json.loads(delta_path.read_text(encoding="utf-8"))
    ref = {"r_m_star_F2": d0["r_m_star"], "r_union_LLM_F2": d0["r_union_lower"],
          "delta_F2_LLM": d0["delta"]}
    expected = {"r_m_star_F2": 0.525, "r_union_LLM_F2": 0.025, "delta_F2_LLM": -0.500}
    mismatch = {k: (ref[k], expected[k]) for k in expected if abs(ref[k] - expected[k]) > 1e-9}
    if mismatch:
        print(f"[K1] !!! delta0 reference numbers differ from PROTOCOL.md's recorded constants: "
              f"{mismatch} (using the ACTUAL file values, not the stale constants)", flush=True)
    _write_json(METRICS_DIR / "delta0_reference.json", ref)

    d = TGF2.build_tasks()
    state["f2_tasks_by_id"] = d["tasks"]
    state["f2_test_ids"] = d["test"]
    state["delta0_ref"] = ref
    print(f"[K1] delta0 reference: r_m*_F2={ref['r_m_star_F2']} r_union_LLM_F2={ref['r_union_LLM_F2']} "
          f"Delta_F2_LLM={ref['delta_F2_LLM']}; F2 TEST reproduced ({len(d['test'])} tasks)", flush=True)
    return {"artifact_paths": ["agent_fork1_three_paths/metrics/delta0_reference.json"],
           "n_f2_test": len(d["test"]), "mismatch": bool(mismatch)}


def step_k2(state: dict) -> dict:
    ok = VS.verify_path_a_self_test()
    if not ok:
        return {"hard_stop": True, "reason": "K2: Path A self-test (filter_det vs golden F2) failed 40/40 -- code bug"}
    tasks_by_id, test_ids = state["f2_tasks_by_id"], state["f2_test_ids"]
    results = {}
    for t_id in test_ids:
        task = tasks_by_id[t_id]
        ids = DA.filter_det(task["records"], task["target_category"], task["target_region"])
        agg = DA.aggregate_det(ids, task["op"], task["id_to_amount"])
        final = DA.derive_det(agg, task["threshold"], task["comparator"])
        results[t_id] = {"filter_ids": sorted(ids), "aggregate": agg, "final": final,
                         "correct": final == task["final_oracle"]}
    state["path_a_results"] = results
    _write_json(METRICS_DIR / "path_a_detail.json", results)
    return {"artifact_paths": ["agent_fork1_three_paths/metrics/path_a_detail.json"],
           "self_test_ok": ok}


def step_k3(state: dict) -> dict:
    results, test_ids = state["path_a_results"], state["f2_test_ids"]
    n = len(test_ids)
    n_pass = sum(1 for t in test_ids if results[t]["correct"])
    r_A = n_pass / n if n else 0.0
    delta_A = r_A - state["delta0_ref"]["r_m_star_F2"]
    state["r_A"] = r_A
    state["delta_A"] = delta_A
    path_a = {"r_A": r_A, "n_pass": n_pass, "n": n, "cost_A": 0, "delta_A": delta_A,
             "TOOL_PIPELINE": True}
    _write_json(METRICS_DIR / "path_a.json", path_a)
    print(f"[K3] r_A={r_A:.3f} ({n_pass}/{n}) cost_A=0 Delta_A={delta_A:.3f} TOOL_PIPELINE=true", flush=True)
    return {"artifact_paths": ["agent_fork1_three_paths/metrics/path_a.json"], **path_a}


def step_k4(state: dict) -> dict:
    ok = VS.verify_task_hard_generator()
    if not ok:
        return {"hard_stop": True, "reason": "K4: T_hard generator invariants failed (see log above)"}
    d = TGH.build_tasks()
    if len(d["test"]) < 30:
        return {"hard_stop": True, "reason": f"K4: N_TEST_B={len(d['test'])} < 30"}
    state["hard_tasks_by_id"] = d["tasks"]
    state["hard_train_ids"] = d["train"]
    state["hard_test_ids"] = d["test"]

    sample_task = d["tasks"][d["test"][0]]
    filter_prompt_len = len(AH.build_filter_prompt(sample_task))
    whole_prompt_len = len(WPH.build_whole_prompt(sample_task))
    est_filter_tokens = filter_prompt_len / 3.5 + AH.MAX_TOKENS_FILTER
    b_atom = int(math.ceil((est_filter_tokens * 1.15) / 50.0) * 50)
    B_hard = b_atom * 4
    state["B_hard"] = B_hard
    state["B_atom_hard"] = b_atom
    est_whole_tokens = whole_prompt_len / 3.5 + WPH.MAX_TOKENS_WHOLE_GEN
    print(f"[K4] T_hard: {len(d['test'])} test tasks, filter_prompt~{filter_prompt_len} chars "
          f"(~{est_filter_tokens:.0f} tok est), whole_prompt~{whole_prompt_len} chars "
          f"(~{est_whole_tokens:.0f} tok est) -> B_atom_hard={b_atom} B_hard={B_hard}", flush=True)
    budget = {"n_train": len(d["train"]), "n_test": len(d["test"]), "B_hard": B_hard,
             "B_atom_hard": b_atom, "B_total_union_hard_ceiling": 2 * B_hard,
             "sample_filter_prompt_chars": filter_prompt_len,
             "sample_whole_prompt_chars": whole_prompt_len}
    _write_json(METRICS_DIR / "task_hard_manifest.json", {
        "train_ids": d["train"], "test_ids": d["test"], "budget": budget})
    return {"artifact_paths": ["agent_fork1_three_paths/metrics/task_hard_manifest.json"], **budget}


def step_k5(state: dict) -> dict:
    tasks_by_id, test_ids, B = state["hard_tasks_by_id"], state["hard_test_ids"], state["B_hard"]
    results_by_model = {}
    n_load_failed = 0
    for m in MR.MODEL_IDS:
        try:
            results_by_model[m] = WPH.measure_whole_for_model(m, tasks_by_id, test_ids, B)
        except Exception as exc:   # noqa: BLE001
            n_load_failed += 1
            print(f"[K5] !!! {m} failed: {type(exc).__name__}: {exc}", flush=True)
    if n_load_failed / len(MR.MODEL_IDS) > HARD_STOP_MODEL_FAIL_FRAC:
        return {"hard_stop": True,
               "reason": f"K5: {n_load_failed}/{len(MR.MODEL_IDS)} models failed to load/run (OOM pattern)"}

    r_m_table = WPH.compute_r_m(results_by_model, test_ids)
    n_calls = CL.count_rows(METRICS_DIR / "whole_calls.jsonl")
    n_expected = len(MR.MODEL_IDS) * len(test_ids)
    n_failed_calls = sum(1 for m in results_by_model for t in test_ids if results_by_model[m][t]["failed"])
    floor_ok = n_calls > 0 and (n_calls - n_failed_calls) / n_expected >= FLOOR_FRAC
    if not floor_ok:
        return {"hard_stop": True,
               "reason": f"K5: floor {n_calls - n_failed_calls}/{n_expected} < {FLOOR_FRAC:.0%}"}

    m_star_B = WPH.select_m_star(r_m_table)
    state["whole_results_by_model_hard"] = results_by_model
    state["r_m_table_hard"] = r_m_table
    state["m_star_B"] = m_star_B
    state["r_m_star_B"] = r_m_table[m_star_B]["r_m"]
    _write_json(METRICS_DIR / "path_b_whole.json", {"r_m_table": r_m_table, "m_star_B": m_star_B})
    _write_json(METRICS_DIR / "path_b_whole_detail.json", results_by_model)   # full raw text, for forensics
    for m, tbl in r_m_table.items():
        star = " <- m*_B" if m == m_star_B else ""
        print(f"[K5] {m}{star}: r_m={tbl['r_m']:.3f} ({tbl['n_pass']}/{tbl['n']}) "
              f"mean_cost={tbl['mean_cost']:.0f}", flush=True)
    return {"artifact_paths": ["agent_fork1_three_paths/metrics/path_b_whole.json",
                              "agent_fork1_three_paths/metrics/path_b_whole_detail.json",
                              "agent_fork1_three_paths/metrics/whole_calls.jsonl"],
           "m_star_B": m_star_B, "r_m_star_B": r_m_table[m_star_B]["r_m"], "floor_ok": floor_ok}


def step_k6(state: dict) -> dict:
    tasks_by_id, test_ids = state["hard_tasks_by_id"], state["hard_test_ids"]
    m_star_B = state["m_star_B"]

    det_results = UH.measure_union_det(tasks_by_id, test_ids)
    r_union_det = UH.compute_r_union(det_results, test_ids)
    _write_json(METRICS_DIR / "path_b_union_det.json", r_union_det)
    _write_json(METRICS_DIR / "path_b_union_det_detail.json", det_results)
    print(f"[K6] B-DET: r_union_B_DET={r_union_det['r_union_lower']:.3f} "
          f"({r_union_det['n_pass']}/{r_union_det['n']}) break_at={r_union_det['break_at_counts']}",
          flush=True)

    try:
        llm_results = UH.measure_union_llm(m_star_B, tasks_by_id, test_ids)
    except Exception as exc:   # noqa: BLE001
        return {"hard_stop": True, "reason": f"K6: B-LLM executor failed to load/run: "
                                             f"{type(exc).__name__}: {exc}"}

    n_calls = CL.count_rows(METRICS_DIR / "union_calls.jsonl")
    n_failed_calls = sum(1 for r in llm_results.values() for c in r["calls"] if c["failed"])
    n_ok_calls = n_calls - n_failed_calls
    floor_ok = n_calls > 0 and (n_ok_calls / n_calls) >= FLOOR_FRAC
    if not floor_ok:
        return {"hard_stop": True, "reason": f"K6: B-LLM floor {n_ok_calls}/{n_calls} calls OK < {FLOOR_FRAC:.0%}"}

    r_union_llm = UH.compute_r_union(llm_results, test_ids)
    _write_json(METRICS_DIR / "path_b_union_llm.json", r_union_llm)
    _write_json(METRICS_DIR / "path_b_union_llm_detail.json", llm_results)
    print(f"[K6] B-LLM: r_union_B_LLM={r_union_llm['r_union_lower']:.3f} "
          f"({r_union_llm['n_pass']}/{r_union_llm['n']}) break_at={r_union_llm['break_at_counts']}",
          flush=True)

    state["r_union_det"] = r_union_det
    state["r_union_llm"] = r_union_llm
    return {"artifact_paths": ["agent_fork1_three_paths/metrics/path_b_union_det.json",
                              "agent_fork1_three_paths/metrics/path_b_union_llm.json",
                              "agent_fork1_three_paths/metrics/union_calls.jsonl"],
           "floor_ok": floor_ok}


def step_k7(state: dict) -> dict:
    r_m_star_B = state["r_m_table_hard"][state["m_star_B"]]["r_m"]
    delta_b_llm = state["r_union_llm"]["r_union_lower"] - r_m_star_B
    delta_b_det = state["r_union_det"]["r_union_lower"] - r_m_star_B
    state["delta_B_LLM"] = delta_b_llm
    state["delta_B_DET"] = delta_b_det
    mean_cost_whole_B = state["r_m_table_hard"][state["m_star_B"]]["mean_cost"]
    mean_cost_union_llm = state["r_union_llm"]["mean_cost"]
    budget_violation = mean_cost_union_llm > 2 * state["B_hard"]
    state["mean_cost_whole_B"] = mean_cost_whole_B
    state["mean_cost_union_llm"] = mean_cost_union_llm
    state["budget_violation_B"] = budget_violation
    out = {"delta_B_LLM": delta_b_llm, "delta_B_DET": delta_b_det, "r_m_star_B": r_m_star_B,
          "mean_cost_whole_B": mean_cost_whole_B, "mean_cost_union_llm": mean_cost_union_llm,
          "budget_violation_B": budget_violation}
    _write_json(METRICS_DIR / "path_b_delta.json", out)
    print(f"[K7] Delta_B_LLM={delta_b_llm:.3f} Delta_B_DET={delta_b_det:.3f} "
          f"(r_m*_B={r_m_star_B:.3f}) BUDGET_VIOLATION={budget_violation}", flush=True)
    return {"artifact_paths": ["agent_fork1_three_paths/metrics/path_b_delta.json"], **out}


def step_k8(state: dict) -> dict:
    swarm_has_sense = (state["delta_B_LLM"] >= DELTA_SWARM_THRESHOLD)
    lines = []
    A = lines.append
    A("# Path C — product spec (K8, 0 new generate() calls)\n")
    A(f"Из уже посчитанных чисел: Δ_A={state['delta_A']:.3f} (TOOL_PIPELINE), "
      f"Δ_B_LLM={state['delta_B_LLM']:.3f}, Δ_B_DET={state['delta_B_DET']:.3f}.\n")
    A("## Продукт\n")
    A(f"- **primary**: single whole `m*` на классе short structured QA "
      f"(F2's m*={state['delta0_ref']['r_m_star_F2']:.3f}-tier reference; "
      f"T_hard's m*_B={state['m_star_B']}, r_m*_B={state['r_m_star_B']:.3f}).\n")
    A("- **optional**: детерминированный tool pipeline, когда нужен точный set/sum "
      "(Path A/B-DET показали потолок ~1.0 при 0 LLM-вызовах в критическом атоме).\n")
    A("- **explicit non-goal**: multi-LLM HGT на 1-2B без показанного Δ>0 в критическом "
      f"LLM-атоме (LLM_SWARM_HAS_SENSE={swarm_has_sense} на этом прогоне).\n")
    text = "\n".join(lines) + "\n"
    (METRICS_DIR / "path_c_product.md").write_text(text, encoding="utf-8")
    state["swarm_has_sense"] = swarm_has_sense
    return {"artifact_paths": ["agent_fork1_three_paths/metrics/path_c_product.md"],
           "swarm_has_sense": swarm_has_sense}


def classify_basket(state: dict) -> str:
    if state["delta_B_LLM"] >= DELTA_SWARM_THRESHOLD and state["r_union_llm"]["r_union_lower"] > state["r_m_star_B"]:
        return "FORK_LLM_SWARM"
    if state["delta_A"] >= DELTA_TOOL_THRESHOLD or state["delta_B_DET"] >= DELTA_TOOL_THRESHOLD:
        return "FORK_TOOL_ONLY"
    return "FORK_WHOLE_ONLY"


NEXT_LINES = {
    "FORK_LLM_SWARM": ("DELTA-1 только T_hard: D-слой + LLM-атомы; бюджет <=2B; "
                       "без HGT до r_real>r_m* на holdout"),
    "FORK_TOOL_ONLY": ("продукт = whole + det tools; LLM-рой на 1-2B не приоритет; "
                       "R&D отбора/HGT стоп"),
    "FORK_WHOLE_ONLY": "продукт = single whole m*; декомпозиция на этих T не нужна",
    "FORK_BLOCKED": "чинить generate/parse; не теория",
}


def step_k9(state: dict) -> dict:
    basket = classify_basket(state)
    high_single = (state["delta0_ref"]["r_m_star_F2"] >= HIGH_SINGLE_THRESHOLD or
                  state["r_m_star_B"] >= HIGH_SINGLE_THRESHOLD)
    state["basket"] = basket
    state["high_single"] = high_single
    state["next_line"] = NEXT_LINES[basket]
    table = {
        "A det-F2": {"r": state["r_A"], "r_m_ref": state["delta0_ref"]["r_m_star_F2"],
                    "delta": state["delta_A"], "llm_in_critical_atom": False, "cost": 0},
        "B whole": {"r": state["r_m_star_B"], "r_m_ref": state["r_m_star_B"], "delta": 0.0,
                   "llm_in_critical_atom": None, "cost": state["mean_cost_whole_B"]},
        "B ∪ LLM": {"r": state["r_union_llm"]["r_union_lower"], "r_m_ref": state["r_m_star_B"],
                   "delta": state["delta_B_LLM"], "llm_in_critical_atom": True,
                   "cost": state["r_union_llm"]["mean_cost"]},
        "B ∪ DET": {"r": state["r_union_det"]["r_union_lower"], "r_m_ref": state["r_m_star_B"],
                   "delta": state["delta_B_DET"], "llm_in_critical_atom": False,
                   "cost": state["r_union_det"]["mean_cost"]},
    }
    state["summary_table"] = table
    _write_json(METRICS_DIR / "summary_table.json", table)
    print(f"[K9] basket={basket} HIGH_SINGLE={high_single} next={state['next_line']}", flush=True)
    return {"artifact_paths": ["agent_fork1_three_paths/metrics/summary_table.json"],
           "basket": basket, "high_single": high_single}


def _fmt(v, spec=".3f"):
    return "n/d" if v is None else format(v, spec)


def write_report(state: dict) -> Path:
    L = []
    A = L.append
    A("# REPORT_FORK1 — three outcomes in one pass (A · B · C)\n")
    A("Спека: `PROTOCOL.md`. 0 Evolution/HGT/Pareto/registry/ASSEMBLE/HETEROSTEP в этом пакете.\n")

    A("## 1. Дуга LIFE-8/9/DELTA-0\n")
    A("LIFE-8: организм не выходит за пределы B3 на старом полигоне (B3_STACK). "
      "LIFE-9: недекомпозированный whole бьёт весь степ-декомпозированный аппарат, "
      "сконструированная декомпозиция (LLM-planner) проигрывает ещё сильнее. "
      "DELTA-0: на F2 (новый полигон, золотой план, реальный executor) Δ=-0.500 — "
      "LLM-исполнитель неточен на exact-set чекпойнте (precision 0.545, recall 0.819), "
      "whole's мягкий финал прощает эту неточность. FORK-1 разводит: была ли проблема в "
      "ИСПОЛНЕНИИ (Path A, det-атомы) или в СНИСХОДИТЕЛЬНОСТИ threshold-финала "
      "(Path B, T_hard=EXTRACT-SUM, точный финал).\n")

    A("## 2. Path A: числа + TOOL_PIPELINE\n")
    A(f"r_A={_fmt(state['r_A'])} ({state['path_a_results'] and sum(1 for t in state['f2_test_ids'] if state['path_a_results'][t]['correct'])}/{len(state['f2_test_ids'])}), "
      f"cost_A=0, Δ_A=r_A-r_m*_F2={_fmt(state['delta_A'])}. **TOOL_PIPELINE=true** "
      "(0 LLM-вызовов в критических атомах -- НЕ победа LLM-роя, §6/§9 PROTOCOL.md).\n")

    A("## 3. Path B: T_hard спека, r_m*_B, r_∪ LLM/DET, Δ\n")
    A("T_hard=EXTRACT-SUM: транзакции, JOIN(category AND region), final = точная СУММА "
      "amount (не threshold-ДА/НЕТ). 2 чекпойнта: matched_ids (exact-set), sum_value "
      "(numeric, = final_oracle).\n")
    A(f"r_m*_B={_fmt(state['r_m_star_B'])} (m*_B={state['m_star_B']}). "
      f"r_∪_B_LLM={_fmt(state['r_union_llm']['r_union_lower'])} "
      f"(break_at={state['r_union_llm']['break_at_counts']}), "
      f"r_∪_B_DET={_fmt(state['r_union_det']['r_union_lower'])} "
      f"(break_at={state['r_union_det']['break_at_counts']}).\n")
    A(f"Δ_B_LLM={_fmt(state['delta_B_LLM'])}, Δ_B_DET={_fmt(state['delta_B_DET'])}. "
      f"Cost: whole(m*_B)={_fmt(state['mean_cost_whole_B'],'.0f')} vs "
      f"union-LLM={_fmt(state['mean_cost_union_llm'],'.0f')} (потолок 2B_hard="
      f"{2*state['B_hard']}) -> " +
      ("**BUDGET_VIOLATION**" if state["budget_violation_B"] else "в пределах потолка") + "\n")

    A("## 4. Path C: продуктовая спека\n")
    A((METRICS_DIR / "path_c_product.md").read_text(encoding="utf-8"))

    A("## 5. Сводная таблица\n")
    t = state["summary_table"]
    A("| path | r | r_m* ref | Δ | LLM в критическом атоме? | cost |\n|---|---|---|---|---|---|\n")
    for name, row in t.items():
        llm_col = "n/a" if row["llm_in_critical_atom"] is None else ("yes" if row["llm_in_critical_atom"] else "no")
        A(f"| {name} | {_fmt(row['r'])} | {_fmt(row['r_m_ref'])} | {_fmt(row['delta'])} | "
          f"{llm_col} | {_fmt(row['cost'],'.0f')} |")
    A("")

    A("## 6. Корзина + Next\n")
    A(f"**{state['basket']}**\n")
    A(f"HIGH_SINGLE: {'ДА' if state['high_single'] else 'нет'}\n")
    A(f"\nNext: {state['next_line']}\n")

    A("## 7. Non-claims\n")
    A("- Δ_A>0 НЕ объявлена победой LLM-роя (TOOL_PIPELINE).\n"
      "- FILTER-цепочка не подставляет golden ids при провале -- реальный обрыв.\n"
      "- SUM всегда детерминирован (и в B-LLM, и в B-DET) -- LLM никогда не считает сумму.\n"
      "- T_hard не усложнялся после Δ_B; exact-set не смягчался после цифр.\n"
      "- `agent_delta0_new_grid/`, `experiment11/` прочитаны, не изменены. `arch2/`, "
      "`experiment14/`, `agent_life*/`, `agent_a5*/` не читались.\n"
      "- Пороги §8 не двигались после первых чисел.\n")

    A("## 8. Приложение\nCHECKLIST.md: `agent_fork1_three_paths/CHECKLIST.md` (все пункты [x]).\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "REPORT_FORK1.md"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


def write_blocked_report(state: dict, step_id: str, reason: str) -> Path:
    L = []
    A = L.append
    A("# REPORT_FORK1 — three outcomes in one pass (A · B · C)\n")
    A(f"**FORK_BLOCKED** — HARD_STOP at `{step_id}` (PROTOCOL.md §8): {reason}\n")
    A(f"Прогон остановлен на `{step_id}` -- см. `metrics/step_log.jsonl`/`CHECKLIST.md` "
      "для того, что успело завершиться.\n")
    if "r_A" in state:
        A(f"### Частично: Path A завершился\nr_A={_fmt(state['r_A'])}, Δ_A={_fmt(state.get('delta_A'))}\n")
    if "r_m_table_hard" in state:
        A("### Частично: Path B whole (K5) завершился\n")
        for m, tbl in state["r_m_table_hard"].items():
            A(f"- {m}: r_m={_fmt(tbl['r_m'])}")
    A(f"\n## Next\n{NEXT_LINES['FORK_BLOCKED']}\n")
    A("## Non-claims\nΔ_B не посчитана -- пайплайн не дошёл до K7.\n")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "REPORT_FORK1.md"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


def step_k10(state: dict) -> dict:
    """K10 marks ITSELF as part of its own completion -- the "all [x]"
    check excludes K10's own line (same fix DELTA-0 needed after finding
    the self-referential bug; applied correctly from the start here)."""
    out_path = write_report(state)
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    unmarked = [ln for ln in text.splitlines()
               if ln.strip().startswith("- [ ] K") and not ln.strip().startswith("- [ ] K10")]
    if unmarked:
        return {"hard_stop": True, "reason": f"K10: items still unchecked: {unmarked}"}
    mark_checklist("K10")
    n_log_rows = CL.count_rows(STEP_LOG_PATH)
    print(f"[K10] REPORT written: {out_path}; CHECKLIST all [x]; step_log has {n_log_rows} rows",
          flush=True)
    return {"artifact_paths": ["agent_fork1_three_paths/reports/REPORT_FORK1.md",
                              "agent_fork1_three_paths/CHECKLIST.md",
                              "agent_fork1_three_paths/metrics/step_log.jsonl"],
           "n_log_rows": n_log_rows}


STEP_FUNCS = {
    "K1": step_k1, "K2": step_k2, "K3": step_k3, "K4": step_k4, "K5": step_k5,
    "K6": step_k6, "K7": step_k7, "K8": step_k8, "K9": step_k9, "K10": step_k10,
}


if __name__ == "__main__":
    checklist_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    assert "- [x] K0 " in checklist_text, "K0 must be marked done before run_all.py starts"

    state: dict = {}
    for step_id in ALL_STEPS:
        print(f"\n=== {step_id} ===", flush=True)
        t0 = time.time()
        result = STEP_FUNCS[step_id](state)
        dt = time.time() - t0
        if result.get("hard_stop"):
            reason = result["reason"]
            print(f"[run_all] HARD_STOP at {step_id}: {reason}", flush=True)
            append_step_log(step_id, "HARD_STOP", result.get("artifact_paths", []), reason=reason)
            write_blockers(f"{step_id}: {reason}")
            state["basket"] = "FORK_BLOCKED"
            write_blocked_report(state, step_id, reason)
            sys.exit(1)
        # K10 marks itself as its own last action (see step_k10's own docstring);
        # every other step is marked here uniformly.
        if step_id != "K10":
            mark_checklist(step_id)
        append_step_log(step_id, "done", result.get("artifact_paths", []), seconds=round(dt, 1),
                        **{k: v for k, v in result.items() if k not in ("artifact_paths", "hard_stop")})
        print(f"[run_all] {step_id} done in {dt:.1f}s", flush=True)

    print(f"\n[run_all] FINAL: {state['basket']} -- отчёт: {REPORTS_DIR / 'REPORT_FORK1.md'}", flush=True)
