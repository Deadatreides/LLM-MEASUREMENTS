"""run_all.py — PROTOCOL.md: verify -> ceiling -> Phase 1 (8) -> classify
-> [Phase 2 (6)] -> [Phase 3: expand+6] -> REPORT, all in one pass, per
the 5 locked corrections from review. No return to the user mid-run.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
AGENT_DIR = SCRIPTS.parent
SRC = AGENT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD          # noqa: E402
import orchestrator as O           # noqa: E402
import metrics_lib as ML           # noqa: E402
import glue_seeds as GS            # noqa: E402
import model_expansion as ME       # noqa: E402


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# `ceiling` collides with the pre-existing, unrelated `arch2/ceiling.py`:
# `live_dataset.py`'s own module-level code inserts ARCH2/ROOT at sys.path[0]
# (needed so it can reach arch2's own modules), which shadows this package's
# `src/ceiling.py` for any later bare `import ceiling`. Load it by direct
# file path instead -- immune to sys.path order (see BLOCKERS.md).
CEIL = _load_module("life8_ceiling", SRC / "ceiling.py")
VS = _load_module("life8_verify_seams", SCRIPTS / "verify_seams.py")

RUNS_DIR = AGENT_DIR / "runs"
REPORTS_DIR = AGENT_DIR / "reports"
METRICS_DIR = AGENT_DIR / "metrics"

SEEDS_F1 = tuple(range(20262201, 20262209))    # 8
SEEDS_F2 = tuple(range(20262211, 20262217))    # 6
SEEDS_F3 = tuple(range(20262221, 20262227))    # 6

SMOKE_SEED = 999901
SMOKE_N_GEN = 8


# ---------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------

def run_smoke() -> dict:
    ds = LD.default_dataset()
    panel_info = O.build_panel(ds)
    panel, screen = panel_info["panel"], panel_info["screen"]

    out_dir = RUNS_DIR / "smoke_throwaway"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_n_gen = O.N_GENERATIONS
    O.N_GENERATIONS = SMOKE_N_GEN
    try:
        summary = O.run_cell(ds, panel, screen, SMOKE_SEED, out_dir, phase="SMOKE")
    finally:
        O.N_GENERATIONS = old_n_gen

    births_path = out_dir / "births.jsonl"
    births = []
    if births_path.exists():
        births = [json.loads(l) for l in births_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_transfer_complementary = sum(1 for b in births
                                   if b.get("operator") == "TRANSFER_SLOT" and b.get("complementary"))
    print(f"[smoke] TRANSFER_SLOT complementary: {n_transfer_complementary}", flush=True)
    print(f"[smoke] best_panel_r={summary['best_panel_r']:.3f} "
          f"best_test_r={summary['best_test_metrics']['r']:.3f}", flush=True)

    ok_stall = VS.verify_stall_fixture()
    ok_glue = VS.verify_glue_donors()
    ok_registry = VS.verify_no_registry_imports()
    phase3_info = VS.verify_phase3_candidates()

    ok = (n_transfer_complementary >= 1) and ok_stall and ok_glue and ok_registry
    return {"ok": ok, "n_transfer_complementary": n_transfer_complementary,
           "stall_fixture_ok": ok_stall, "glue_fixture_ok": ok_glue,
           "no_registry_path": ok_registry, "phase3_candidates": phase3_info["candidates"],
           "n_generations_run": summary["n_generations"]}


# ---------------------------------------------------------------------
# cell running + phase metrics
# ---------------------------------------------------------------------

def run_cells(ds, panel, screen, seeds: tuple, phase: str, label: str, extra_donors=None) -> list:
    out = []
    for seed in seeds:
        out_dir = RUNS_DIR / f"life8_{phase.lower()}_s{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        summary = O.run_cell(ds, panel, screen, seed, out_dir, phase=phase, extra_donors=extra_donors)
        dt = time.time() - t0
        print(f"[run_all] {label} seed={seed} done in {dt:.0f}s n_generations={summary['n_generations']} "
              f"extinct={summary['extinct']} best_test_r={summary['best_test_metrics']['r']:.3f}", flush=True)
        out.append({"seed": seed, "out_dir": out_dir, "summary": summary, "wall_s": dt})
    return out


def phase_metrics(cells: list) -> dict:
    best_test_rs = [c["summary"]["best_test_metrics"]["r"] for c in cells]
    return {"R": statistics.mean(best_test_rs), "best_test_rs": best_test_rs,
           "n_seeds": len(cells)}


def _sign_count(best_test_rs: list, reference_r: float) -> int:
    return sum(1 for r in best_test_rs if r > reference_r)


def classify_f1(pm: dict, ceiling: dict) -> str:
    b1, b2, b3 = ceiling["B1"]["r_test"], ceiling["B2"]["r_test"], ceiling["B3"]["r_test"]
    R, rs = pm["R"], pm["best_test_rs"]
    if R > b3 and _sign_count(rs, b3) >= 6:
        return "F1_B3"
    if (b3 - R) <= 0.03 and R > b2:
        return "F1_NEAR"
    if R > b1:
        return "F1_MID"
    return "F1_FAIL"


def classify_f2(pm: dict, ceiling: dict) -> str:
    b3 = ceiling["B3"]["r_test"]
    if pm["R"] > b3 and _sign_count(pm["best_test_rs"], b3) >= 5:
        return "F2_B3"
    return "F2_NO"


def classify_f3(pm: dict, ceiling_expanded: dict, old_b3: float) -> str:
    b3p = ceiling_expanded["B3"]["r_test"]
    if pm["R"] > b3p and _sign_count(pm["best_test_rs"], b3p) >= 5:
        return "F3_B3"
    if pm["R"] > old_b3:
        return "F3_OLD"
    return "F3_NO"


def classify_final(state: dict) -> tuple:
    if state.get("f1") == "F1_B3" or state.get("f2") == "F2_B3" or state.get("f3") == "F3_B3":
        return "B3_DONE", state.get("f1") or state.get("f2") or state.get("f3")
    if state.get("f3") == "F3_OLD":
        return "B3_LANG", "F3_OLD"
    if state.get("f1") == "F1_FAIL":
        return "B3_FAIL", "F1_FAIL"
    lang_poor = state["ceiling"]["LANG_POOR"]
    phase3_empty_or_failed = state.get("phase3_outcome") in ("CLOSE_NEED_MODELS", "F3_NO", None)
    if lang_poor and phase3_empty_or_failed:
        return "B3_CEILING", "LANG_POOR + no rescue"
    if (not lang_poor) and state.get("f2") == "F2_NO" and phase3_empty_or_failed:
        return "B3_STACK", "LANG_RICH + F2_NO + no atoms"
    return "B3_BLOCKED", "unclassified state -- see BLOCKERS.md"


# ---------------------------------------------------------------------
# forensic (task's own §6)
# ---------------------------------------------------------------------

def forensic_best_vs_baselines(best_cell: dict, ceiling: dict, ds, test_ids: list) -> dict:
    summary = best_cell["summary"]
    art = ML.load_cell_artifacts(best_cell["out_dir"])
    genotypes = art["genotypes"]
    best_g = genotypes.get(summary["best_panel_cid"])

    def _route_root(route):
        return LD.HSEED.route(route)

    b2_root = _route_root(ceiling["B2"]["route"])
    b3_root = _route_root(ceiling["B3"]["route"])
    identical_to_b2 = (best_g is not None and
                       json.dumps(best_g["root"], sort_keys=True) == json.dumps(b2_root, sort_keys=True))
    identical_to_b3 = (best_g is not None and
                       json.dumps(best_g["root"], sort_keys=True) == json.dumps(b3_root, sort_keys=True))

    # Per-task win breakdown: reuse the REAL trace-based outcomes both sides
    # already have on disk (`F.resolved_set`, the same function `metrics()`
    # itself is built on) -- NOT a from-scratch replay via `ds.cells`. A
    # static per-slot-model replay is unsound here: a slot can hold a
    # `CALL_COMPOSITE` reference to a WHOLE OTHER archived genotype (routine
    # arch2 behavior -- any admitted genotype is auto-composite-referenceable
    # by complex_id, unrelated to the block registry P5 excludes), and that
    # inner genotype's own PASS/FAIL depends on its OWN 4-slot resolution,
    # not on "any model in a flattened list passing this one slot's kind" --
    # confirmed by direct trace inspection this session (BLOCKERS.md).
    # `REF_B3_greedy_cover` was already evaluated on `test_ids` by this same
    # cell's own `"final:test_refs"` pass -- its complex_id is in `seed_names`.
    exp_id = f"life8_{summary['phase'].lower()}_s{summary['seed']}"
    traces_results = LD.F.load_results(best_cell["out_dir"] / "traces", exp_id)
    ref_b3_cid = summary["seed_names"]["REF_B3_greedy_cover"]
    org_resolved = LD.F.resolved_set(traces_results.get(summary["best_panel_cid"], {}), test_ids)
    b3_resolved = LD.F.resolved_set(traces_results.get(ref_b3_cid, {}), test_ids)
    both = len(org_resolved & b3_resolved)
    only_b3 = len(b3_resolved - org_resolved)
    only_org = len(org_resolved - b3_resolved)
    neither = len(test_ids) - len(org_resolved | b3_resolved)

    return {"best_complex_id": summary["best_panel_cid"], "identical_to_B2": identical_to_b2,
           "identical_to_B3": identical_to_b3, "test_both_win": both, "test_only_B3_win": only_b3,
           "test_only_organism_win": only_org, "test_neither_win": neither}


# ---------------------------------------------------------------------
# report
# ---------------------------------------------------------------------

def _fmt(v, spec=".3f"):
    return "n/d" if v is None else format(v, spec)


def _phase_table(cells: list) -> list:
    lines = ["| seed | n_gen | extinct | best_panel_r | best_test_r |", "|---|---|---|---|---|"]
    for c in cells:
        s = c["summary"]
        lines.append(f"| {c['seed']} | {s['n_generations']} | {s['extinct']} | "
                     f"{_fmt(s['best_panel_r'])} | {_fmt(s['best_test_metrics']['r'])} |")
    return lines


def write_report(state: dict) -> Path:
    L = []
    A = L.append
    A("# REPORT_LIFE8 — closing B3: ceiling -> organism -> glue -> language\n")
    A("Спека: `PROTOCOL.md` (P1-P7). Организм зафиксирован (P5): HGT (`p=0.35`) + "
      "stall-from-improvement, обычные D1/D2 -- Pareto/GATE/registry не сравнивались. "
      "Живая сетка ПЕРЕИСПОЛЬЗОВАНА byte-identical из `agent_life7_selection` (md5 подтверждён), "
      "0 новых generate() на исходных 6 моделях.\n")

    A("## 1. P1-P7 и что не гоняли\n")
    A("P1-P7 -- см. PROTOCOL.md (самоопределены из структуры задания, не найден готовый черновик). "
      "Не гоняли: сетку p_slot_hgt, Pareto-survival, gate-based stall, slot-safe mutate, "
      "carrier immunity, block registry, brute-force 1296-комбинаций.\n")

    A("## 2. Потолок (Phase 0)\n")
    c = state["ceiling"]
    A("| | r_panel | r_test |")
    A("|---|---|---|")
    for name in ("B1", "B2", "B3", "UNION"):
        A(f"| {name} | {_fmt(c[name]['r_panel'])} | {_fmt(c[name]['r_test'])} |")
    A(f"\nGAP_LANG = {_fmt(c['GAP_LANG'])}, GAP_STACK = {_fmt(c['GAP_STACK'])} "
      f"(порог {c['gap_lang_threshold']}) -> **{'LANG_POOR' if c['LANG_POOR'] else 'LANG_RICH'}**\n")

    A("## 3. Phase 1: организм на 6 моделях, 8 seed\n")
    A("\n".join(_phase_table(state["cells_f1"])))
    pm1 = state["pm_f1"]
    A(f"\nR = {_fmt(pm1['R'])}. Знаки vs B1/B2/B3/UNION: "
      f"{_sign_count(pm1['best_test_rs'], c['B1']['r_test'])}/{pm1['n_seeds']} vs B1, "
      f"{_sign_count(pm1['best_test_rs'], c['B2']['r_test'])}/{pm1['n_seeds']} vs B2, "
      f"{_sign_count(pm1['best_test_rs'], c['B3']['r_test'])}/{pm1['n_seeds']} vs B3, "
      f"{_sign_count(pm1['best_test_rs'], c['UNION']['r_test'])}/{pm1['n_seeds']} vs UNION.\n")
    A(f"\n**F1: {state['f1']}**\n")

    if "cells_f2" in state:
        A("## 4. Phase 2: склейка (17 archive-only донора), 6 seed\n")
        A("\n".join(_phase_table(state["cells_f2"])))
        pm2 = state["pm_f2"]
        A(f"\nR2 = {_fmt(pm2['R'])}. Знак vs B3: {_sign_count(pm2['best_test_rs'], c['B3']['r_test'])}/"
          f"{pm2['n_seeds']}.\n")
        A(f"\n**F2: {state['f2']}**\n")

    if "phase3_outcome" in state:
        A("## 5. Phase 3: добор языка\n")
        A(f"Кандидаты (sorted, K<=4): `{state['phase3_candidates']}`.\n")
        if state["phase3_outcome"] == "CLOSE_NEED_MODELS":
            A("**CLOSE_NEED_MODELS** -- реестр `experiment11` не содержит моделей сверх текущих 6 "
              "(подтверждено прямым чтением, не предположено). Grid-расширение/пересчёт B3'/evolve "
              "не запускались -- нет кандидатов.\n")
        else:
            A(f"**F3: {state.get('f3')}**\n")

    if "forensic" in state:
        A("## 6. Forensic\n")
        fz = state["forensic"]
        A(f"Лучший генотип (`{fz['best_complex_id']}`) идентичен B2 bit-for-bit: "
          f"{fz['identical_to_B2']}; идентичен B3: {fz['identical_to_B3']}.")
        A(f"На test: оба (B3 и организм) решают {fz['test_both_win']}, только B3 -- "
          f"{fz['test_only_B3_win']}, только организм -- {fz['test_only_organism_win']}, "
          f"ни один -- {fz['test_neither_win']}.\n")

    A("## 7. Корзина\n")
    A(f"**{state['basket']}** ({state['basket_reason']})\n")

    A("\n## 8. Non-claims\n")
    A("- Не заявляется «победили B3» вне правила §5 PROTOCOL.md.\n"
      "- Нет block registry/Pareto/GATE-сравнения нигде в этом пакете.\n"
      "- LANG_POOR/LANG_RICH из Phase 0 не пересчитывались задним числом.\n"
      "- `arch2/`, `agent_a5_live_m/`, `agent_life1..7` не изменялись.")

    A("\n## 9. Дальше (ровно одна строка)\n")
    A(state["next_line"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "REPORT_LIFE8.md"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


NEXT_LINES = {
    "B3_DONE": "спека организма + демо, не новые прогоны отбора.",
    "B3_LANG": "следующий бриф: ещё атомы ИЛИ принять B3'-gap.",
    "B3_CEILING": "только новые типы шагов/операций, не 7-я LLM того же класса.",
    "B3_STACK": "подключить модели в registry, не крутить stall.",
    "B3_FAIL": "чинить reproduce/grid, не алфавит.",
    "B3_BLOCKED": "см. BLOCKERS.md для инфраструктурной причины.",
}


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=== [1/6] verify_seams ===", flush=True)
    ok_seams = VS.verify_seams()
    ok_grid = VS.verify_grid_copy()
    ok_registry0 = VS.verify_no_registry_imports()
    if not (ok_seams and ok_grid and ok_registry0):
        print("[run_all] verify_seams FAILED -> BLOCKERS.md, стоп", flush=True)
        sys.exit(1)

    print("\n=== [2/6] smoke ===", flush=True)
    smoke = run_smoke()
    if not smoke["ok"]:
        print(f"[run_all] smoke FAILED: {smoke} -> BLOCKERS.md, стоп", flush=True)
        sys.exit(1)

    print("\n=== [3/6] Phase 0: ceiling ===", flush=True)
    ds = LD.default_dataset()
    panel_info = O.build_panel(ds)
    panel, screen = panel_info["panel"], panel_info["screen"]
    test_ids = ds.split["test"]
    ceiling = CEIL.compute_ceiling(ds, panel, test_ids)
    (METRICS_DIR / "ceiling.json").write_text(json.dumps(ceiling, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_all] B1={ceiling['B1']['r_test']:.3f} B2={ceiling['B2']['r_test']:.3f} "
          f"B3={ceiling['B3']['r_test']:.3f} UNION={ceiling['UNION']['r_test']:.3f} "
          f"GAP_LANG={ceiling['GAP_LANG']:.3f} -> {'LANG_POOR' if ceiling['LANG_POOR'] else 'LANG_RICH'}",
          flush=True)

    state = {"smoke": smoke, "ceiling": ceiling}

    print("\n=== [4/6] Phase 1: organism, 8 seeds ===", flush=True)
    t0 = time.time()
    cells_f1 = run_cells(ds, panel, screen, SEEDS_F1, "F1", "F1")
    print(f"[run_all] Phase 1: {time.time()-t0:.0f}s for 8 cells", flush=True)
    pm_f1 = phase_metrics(cells_f1)
    f1 = classify_f1(pm_f1, ceiling)
    print(f"[run_all] F1 = {f1}: R={pm_f1['R']:.3f}", flush=True)
    state.update({"cells_f1": cells_f1, "pm_f1": pm_f1, "f1": f1})

    best_cell_f1 = max(cells_f1, key=lambda c: c["summary"]["best_test_metrics"]["r"])
    state["forensic"] = forensic_best_vs_baselines(best_cell_f1, ceiling, ds, test_ids)

    if f1 == "F1_B3":
        state["basket"], state["basket_reason"] = "B3_DONE", "F1_B3"
        state["next_line"] = NEXT_LINES["B3_DONE"]
        out_path = write_report(state)
        print(f"[run_all] B3_DONE at Phase 1 -- отчёт: {out_path}", flush=True)
        sys.exit(0)

    if f1 == "F1_FAIL":
        print("\n=== F1_FAIL -> B3_FAIL immediately (correction #1) ===", flush=True)
        phase3_info = VS.verify_phase3_candidates()   # for the record only, per correction #1
        state["phase3_candidates"] = phase3_info["candidates"]
        state["basket"], state["basket_reason"] = "B3_FAIL", "F1_FAIL"
        state["next_line"] = NEXT_LINES["B3_FAIL"]
        out_path = write_report(state)
        print(f"[run_all] B3_FAIL -- отчёт: {out_path}", flush=True)
        sys.exit(0)

    # F1_NEAR or F1_MID from here on
    if not ceiling["LANG_POOR"]:
        print("\n=== [5/6] Phase 2: glue (LANG_RICH, mandatory per correction #3) ===", flush=True)
        best_single = ML.best_single_per_slot(ds, panel)
        extra_donors = GS.build_glue_donors(ds, panel, best_single)
        GS.assert_glue_donors_single_slot(extra_donors, ds, panel, best_single)
        t0 = time.time()
        cells_f2 = run_cells(ds, panel, screen, SEEDS_F2, "F2", "F2", extra_donors=extra_donors)
        print(f"[run_all] Phase 2: {time.time()-t0:.0f}s for 6 cells", flush=True)
        pm_f2 = phase_metrics(cells_f2)
        f2 = classify_f2(pm_f2, ceiling)
        print(f"[run_all] F2 = {f2}: R2={pm_f2['R']:.3f}", flush=True)
        state.update({"cells_f2": cells_f2, "pm_f2": pm_f2, "f2": f2})

        if f2 == "F2_B3":
            state["basket"], state["basket_reason"] = "B3_DONE", "F2_B3"
            state["next_line"] = NEXT_LINES["B3_DONE"]
            out_path = write_report(state)
            print(f"[run_all] B3_DONE at Phase 2 -- отчёт: {out_path}", flush=True)
            sys.exit(0)

    print("\n=== [6/6] Phase 3: language expansion ===", flush=True)
    phase3_info = VS.verify_phase3_candidates()
    state["phase3_candidates"] = phase3_info["candidates"]
    if not phase3_info["candidates"]:
        state["phase3_outcome"] = "CLOSE_NEED_MODELS"
        print("[run_all] Phase 3: CLOSE_NEED_MODELS (registry has 0 candidates beyond current 6)", flush=True)
    else:
        # Not expected to fire this session (registry confirmed empty) -- implemented for correctness.
        ds3, grid_ok = ME.expand_dataset(phase3_info["candidates"], sorted(ds.tasks), ds.tasks)
        if not grid_ok:
            state["basket"], state["basket_reason"] = "B3_FAIL", "Phase 3 generate floor not met"
            state["next_line"] = NEXT_LINES["B3_FAIL"]
            out_path = write_report(state)
            print(f"[run_all] B3_FAIL (Phase 3 floor) -- отчёт: {out_path}", flush=True)
            sys.exit(0)
        panel3_info = O.build_panel(ds3)
        panel3, screen3 = panel3_info["panel"], panel3_info["screen"]
        test3_ids = ds3.split["test"]
        ceiling3 = CEIL.compute_ceiling(ds3, panel3, test3_ids)
        (METRICS_DIR / "ceiling_expanded.json").write_text(
            json.dumps(ceiling3, ensure_ascii=False, indent=2), encoding="utf-8")
        cells_f3 = run_cells(ds3, panel3, screen3, SEEDS_F3, "F3", "F3")
        pm_f3 = phase_metrics(cells_f3)
        f3 = classify_f3(pm_f3, ceiling3, ceiling["B3"]["r_test"])
        state.update({"cells_f3": cells_f3, "pm_f3": pm_f3, "f3": f3, "ceiling_expanded": ceiling3})
        state["phase3_outcome"] = f3
        if f3 == "F3_B3":
            state["basket"], state["basket_reason"] = "B3_DONE", "F3_B3"
            state["next_line"] = NEXT_LINES["B3_DONE"]
            out_path = write_report(state)
            print(f"[run_all] B3_DONE at Phase 3 -- отчёт: {out_path}", flush=True)
            sys.exit(0)
        if f3 == "F3_OLD":
            state["basket"], state["basket_reason"] = "B3_LANG", "F3_OLD"
            state["next_line"] = NEXT_LINES["B3_LANG"]
            out_path = write_report(state)
            print(f"[run_all] B3_LANG -- отчёт: {out_path}", flush=True)
            sys.exit(0)

    basket, reason = classify_final(state)
    state["basket"], state["basket_reason"] = basket, reason
    state["next_line"] = NEXT_LINES[basket]
    out_path = write_report(state)
    print(f"[run_all] {basket} -- отчёт: {out_path}", flush=True)
