"""run_all.py — PROTOCOL.md: verify -> Phase A (residual) -> Phase B
(alpha_whole) -> [C-skip | smoke -> full Phase C] -> Phase D -> REPORT.
One pass, no return to the user after any phase.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
AGENT_DIR = SCRIPTS.parent
ROOT = AGENT_DIR.parent
SRC = AGENT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
ARCH2 = ROOT / "arch2"
if str(ARCH2) not in sys.path:
    sys.path.insert(0, str(ARCH2))

import live_dataset as LD          # noqa: E402
import lean_seeds as LS            # noqa: E402
import residual as RES             # noqa: E402
import whole_alpha as WA           # noqa: E402
import decomp_pipeline as DP       # noqa: E402


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# `ceiling` collides with the pre-existing, unrelated `arch2/ceiling.py`
# once ARCH2 is on sys.path (needed below for `fitness.py`) -- same bug
# class LIFE-8 hit and fixed; load by direct file path, immune to sys.path
# order (see BLOCKERS.md).
CEIL = _load_module("life9_ceiling", SRC / "ceiling.py")
import fitness as ARCH2_F          # noqa: E402  -- paired_delta_r only, no genotype dependency

VS = _load_module("life9_verify_seams", SCRIPTS / "verify_seams.py")

METRICS_DIR = AGENT_DIR / "metrics"
REPORTS_DIR = AGENT_DIR / "reports"
RUNS_DIR = AGENT_DIR / "runs"

SMOKE_N = 4
BEATS_ORG_MARGIN = 0.03


def build_panel(ds) -> list:
    return sorted(ds.split["train"])[:80]


def _fmt(v, spec=".3f"):
    return "n/d" if v is None else format(v, spec)


NEXT_LINES = {
    "S0_CLOSED": ("S0+6 моделей исчерпаны; следующий пакет = новая задача/шаги "
                 "ИЛИ ортогональный атом с проверкой P(PASS|all FAIL)>=tau; "
                 "не registry ради cardinality, не HGT"),
    "D_PRODUCT": "слой D + E_i = default; HGT на S0 не развивать",
    "ALPHA_BLOCKED": "починить generate/parse, не теория",
}


def write_report(state: dict) -> Path:
    L = []
    A = L.append
    A("# REPORT_LIFE9 — alpha diagnostics + immediate continuation\n")
    A("Спека: `PROTOCOL.md` (P0-P5). 0 evolve/HGT/Pareto/registry в этом пакете -- "
      "чистый CPU-анализ (Phase A/B) + один живой decomposition-пайплайн (Phase C).\n")

    A("## 1. Anatomy residual (пол / kill-step / ρ)\n")
    at = state["anatomy_test"]
    c = state["ceiling"]
    A(f"Test: n={at['n_tasks']}, U_dead={at['n_u_dead']}, U_ok={at['n_u_ok']} "
      f"(UNION_r={at['union_r']:.3f}, LIFE-8 UNION_r_test={c['UNION']['r_test']:.3f}, "
      f"diff={abs(at['n_u_ok']-round(c['UNION']['r_test']*at['n_tasks']))} задач, допуск 1).\n")
    A(f"fail_s (кинды-убийцы среди U_dead, счёт с повтором): "
      f"{', '.join(f'{k}={v}' for k, v in at['fail_s'].items())}.\n")
    A(f"ρ̄_FAIL (доля НЕ-PASS среди всех 24 ячеек U_dead-задачи, усреднено) = "
      f"{_fmt(at['rho_bar_fail'])}; ρ̄ для U_ok = {_fmt(at['rho_bar_ok'])} (для контраста).\n")
    A(f"\n| | r_panel | r_test |\n|---|---|---|\n"
      f"| B1 | {_fmt(c['B1']['r_panel'])} | {_fmt(c['B1']['r_test'])} |\n"
      f"| B2 | {_fmt(c['B2']['r_panel'])} | {_fmt(c['B2']['r_test'])} |\n"
      f"| B3 | {_fmt(c['B3']['r_panel'])} | {_fmt(c['B3']['r_test'])} |\n"
      f"| UNION | {_fmt(c['UNION']['r_panel'])} | {_fmt(c['UNION']['r_test'])} |\n")

    A("## 2. α̂_B2, α̂_decomp, r_D_test\n")
    ab = state["alpha_b2_info"]
    aa = state["alpha_any_info"]
    A(f"α̂_B2_raw = {_fmt(ab['alpha_B2_raw'])}, α̂_B2_reparsed = {_fmt(ab['alpha_B2_reparsed'])} "
      f"(n_U_dead={ab['n_u_dead']}, missing_whole_row={ab['n_missing_whole_row']}) -- "
      f"**{state['classification']}** (порог по α̂_B2_reparsed, не двигался).\n")
    A(f"α̂_any_reparsed (диагностика, PANEL, не гейтит) = {_fmt(aa.get('alpha_any_reparsed'))} "
      f"(n_U_dead_panel={aa.get('n_u_dead_panel')}).\n")
    if state.get("alpha_decomp") is not None:
        A(f"α̂_decomp = {_fmt(state['alpha_decomp'])}, r_D_test = {_fmt(state['r_d_test'])} "
          f"(один прогон пайплайна на всех {len(state['test_ids'])} test-задачах, P5).\n")
    else:
        A("Phase C не запускалась (C-skip: α̂_B2 < 0.05).\n")

    A("## 3. Таблица vs B1/B2/B3/UNION/org\n")
    A("| | r_test |\n|---|---|\n"
      f"| B1 | {_fmt(c['B1']['r_test'])} |\n"
      f"| B2 (route) | {_fmt(c['B2']['r_test'])} |\n"
      f"| B3 | {_fmt(c['B3']['r_test'])} |\n"
      f"| UNION | {_fmt(c['UNION']['r_test'])} |\n"
      f"| org (LIFE-8, R) | {_fmt(state.get('r_org'))} |\n"
      f"| whole B2 (reparsed, n=100) | {_fmt(state.get('r_whole_b2'))} |\n"
      f"| D (decomp, r_D_test) | {_fmt(state.get('r_d_test'))} |\n")

    A("## 4. Корзина\n")
    A(f"**{state['basket']}**")
    if state.get("basket_detail"):
        A(f" -- {state['basket_detail']}")
    A(f" (механически, по PROTOCOL.md §7 D3 -- r_D_test здесь буквально decomp-пайплайн, "
      f"P4/§3, а не выбранный D1-кандидат-продукт -- см. §4.1).\n")

    if "interpretation" in state:
        ip = state["interpretation"]
        A("\n### 4.1 Интерпретация (раскрыта рядом, вердикт §4 не переписан)\n")
        A(f"D1 выбрал **{ip['chosen']}** (r_test={_fmt(ip['chosen_r_test'])}) как лучший "
          f"кандидат-продукт -- если применить ТЕ ЖЕ формулы D_BEATS_B3/D_BEATS_ORG к НЕМУ "
          f"(не к литеральному r_D_test decomp-пайплайна), результат другой:\n")
        A(f"- vs B3: point={_fmt(ip['vs_b3']['point'])}, "
          f"CI95=[{_fmt(ip['vs_b3']['ci_95'][0])}, {_fmt(ip['vs_b3']['ci_95'][1])}], "
          f"p={_fmt(ip['vs_b3']['p_value'], '.4f')}, n={ip['vs_b3']['n_pairs']} -> "
          f"**{'D_BEATS_B3 устойчиво' if ip['chosen_beats_b3_stable'] else 'не устойчиво выше B3'}**\n")
        A(f"- vs UNION: point={_fmt(ip['vs_union']['point'])}, "
          f"CI95=[{_fmt(ip['vs_union']['ci_95'][0])}, {_fmt(ip['vs_union']['ci_95'][1])}], "
          f"p={_fmt(ip['vs_union']['p_value'], '.4f')}\n")
        A(f"- vs org+0.03: {'да' if ip['chosen_beats_org'] else 'нет'}\n")
        if ip["chosen"] == "whole_B2":
            A("\nПрактический вывод: **не декомпозиция оживляет мёртвых, а сам факт "
              "недекомпозированного запроса одной достаточно сильной моделью** -- "
              "decomp-пайплайн (planner+executor'ы) на этих же 6 моделях эту силу теряет, "
              "не приумножает (см. §2 α̂_decomp/r_D_test и BLOCKERS.md §forensic).\n")

    if "revived" in state:
        A(f"\n### D4. U_dead: ожили / не ожили (по выбранному D1-кандидату: {state['interpretation']['chosen']})\n")
        A(f"Ожили ({len(state['revived'])}): `{state['revived']}`\n\n")
        A(f"Не ожили ({len(state['not_revived'])}): `{state['not_revived']}`\n")

    A("## 5. Next\n")
    A(f"{state['next_line']}\n")
    if state.get("interpretation", {}).get("chosen") == "whole_B2":
        A("\n(Уточнение к строке выше, не замена: конкретный D-продукт этого пакета -- "
          "недекомпозированный whole-task запрос B2_model, БЕЗ E_i -- см. §4.1.)\n")

    A("## 6. Non-claims\n")
    A("- α̂_B2 гейтится по reparsed-числу (локальная, раскрытая поправка), не по raw -- "
      "оба числа приведены выше и в BLOCKERS.md.\n"
      "- Не сравнивались Pareto/GATE/G_STALL/registry -- этот пакет не пересматривает отбор LIFE-8.\n"
      "- `experiment14/seams.py` НЕ отредактирован -- поправка парсинга живёт только в этом "
      "пакете (`whole_alpha.py`/`decomp_pipeline.py`).\n"
      "- `arch2/`, `agent_a5_live_m/`, `agent_life1..8` не изменялись.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "REPORT_LIFE9.md"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


def finalize(state: dict) -> Path:
    return write_report(state)


if __name__ == "__main__":
    print("=== [1/5] verify_seams ===", flush=True)
    ok_grid = VS.verify_grid_copy()
    ok_seams = VS.verify_seams14()
    ok_reparse = VS.verify_reparse_fixture()
    ok_list_prefix = VS.verify_list_prefix_regex_fixture()
    ok_planner = VS.verify_planner_parse_fixture()
    if not (ok_grid and ok_seams and ok_reparse and ok_list_prefix and ok_planner):
        print("[run_all] verify FAILED -> stop", flush=True)
        sys.exit(1)

    print("\n=== [2/5] Phase A: residual anatomy ===", flush=True)
    ds = LD.default_dataset()
    test_ids = ds.split["test"]
    panel = build_panel(ds)
    ceiling = CEIL.compute_ceiling(ds, panel, test_ids)
    anatomy_test = RES.compute_residual_anatomy(ds, test_ids)
    anatomy_panel = RES.compute_residual_anatomy(ds, panel)

    diff = abs(anatomy_test["n_u_ok"] - round(ceiling["UNION"]["r_test"] * len(test_ids)))
    print(f"[run_all] U_dead={anatomy_test['n_u_dead']} U_ok={anatomy_test['n_u_ok']} "
          f"(UNION cross-check diff={diff}, tolerance 1)", flush=True)
    if diff > 1:
        print("[run_all] UNION cross-check FAILED -> stop (PROTOCOL.md §4)", flush=True)
        sys.exit(1)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "residual_anatomy.json").write_text(
        json.dumps({"test": anatomy_test, "panel": anatomy_panel}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("\n=== [3/5] Phase B: alpha_whole ===", flush=True)
    whole_rows = LD.load_whole_grid_rows()
    whole_by_key = WA.index_whole_rows(whole_rows)
    alpha_b2_info = WA.compute_alpha_b2(ds, whole_by_key, anatomy_test["u_dead"])
    alpha_any_info = WA.compute_alpha_any_panel(ds, whole_by_key, anatomy_panel["u_dead"], ds.models)
    alpha_b2 = alpha_b2_info["alpha_B2_reparsed"]
    classification = WA.classify_alpha_b2(alpha_b2)
    print(f"[run_all] alpha_B2_raw={alpha_b2_info['alpha_B2_raw']:.3f} "
          f"alpha_B2_reparsed={alpha_b2:.3f} -> {classification}", flush=True)
    print(f"[run_all] alpha_any_reparsed (panel, diagnostic) = "
          f"{alpha_any_info.get('alpha_any_reparsed')}", flush=True)

    state = {"ceiling": ceiling, "anatomy_test": anatomy_test, "anatomy_panel": anatomy_panel,
            "alpha_b2_info": alpha_b2_info, "alpha_any_info": alpha_any_info,
            "classification": classification, "test_ids": test_ids, "r_org": 0.456}

    c_skip = alpha_b2 is not None and alpha_b2 < 0.05
    if c_skip:
        print("\n=== C-skip: alpha_B2 < 0.05 -> S0_CLOSED, Phase C not run ===", flush=True)
        state["alpha_decomp"] = None
        state["alpha_star"] = alpha_b2
        state["basket"] = "S0_CLOSED"
        state["basket_detail"] = "1-UNION доминирует, B2 whole не поднимает пол (C-skip, PROTOCOL.md §5)"
        state["next_line"] = NEXT_LINES["S0_CLOSED"]
        out_path = finalize(state)
        print(f"[run_all] S0_CLOSED -- отчёт: {out_path}", flush=True)
        sys.exit(0)

    print("\n=== [4/5] Phase C: smoke -> full decomp ===", flush=True)
    executor_by_kind = {kind: ceiling["orders"][kind][0] for kind in LD.DECOMP_KINDS}
    print(f"[run_all] planner={WA.B2_MODEL} executor_by_kind={executor_by_kind}", flush=True)

    smoke_tasks = sorted(anatomy_test["u_dead"])[:SMOKE_N]
    t0 = time.time()
    smoke_results = DP.run_decomp_over_tasks(ds, smoke_tasks, WA.B2_MODEL, executor_by_kind,
                                             RUNS_DIR / "decomp_smoke.json", resume=False)
    print(f"[run_all] smoke: {time.time()-t0:.0f}s for {len(smoke_tasks)} tasks", flush=True)
    n_smoke_json_ok = sum(1 for r in smoke_results.values() if r["status"] == "OK")
    n_smoke_calls = sum(len(r["calls"]) for r in smoke_results.values())
    n_smoke_failed = sum(1 for r in smoke_results.values() for c in r["calls"] if c.get("failed"))
    for t, r in smoke_results.items():
        print(f"[run_all]   smoke {t}: status={r['status']} final_pass={r['final_pass']} "
              f"n_calls={len(r['calls'])}", flush=True)
    print(f"[run_all] smoke: {n_smoke_json_ok}/{len(smoke_tasks)} planner-JSON parsed OK, "
          f"{n_smoke_failed}/{n_smoke_calls} generate() calls failed", flush=True)

    if n_smoke_calls == 0 or (n_smoke_failed / n_smoke_calls) > 0.20:
        print("[run_all] smoke infra check FAILED -> ALPHA_BLOCKED", flush=True)
        state["basket"] = "ALPHA_BLOCKED"
        state["basket_detail"] = f"smoke: {n_smoke_failed}/{n_smoke_calls} generate() calls failed"
        state["next_line"] = NEXT_LINES["ALPHA_BLOCKED"]
        out_path = finalize(state)
        print(f"[run_all] ALPHA_BLOCKED -- отчёт: {out_path}", flush=True)
        sys.exit(1)

    t0 = time.time()
    decomp_results = DP.run_decomp_over_tasks(ds, test_ids, WA.B2_MODEL, executor_by_kind,
                                              RUNS_DIR / "decomp_full.json", resume=True)
    print(f"[run_all] full decomp: {time.time()-t0:.0f}s for {len(test_ids)} tasks", flush=True)

    all_calls = [c for r in decomp_results.values() for c in r["calls"]]
    n_calls_total = len(all_calls)
    n_calls_failed = sum(1 for c in all_calls if c.get("failed"))
    floor_ok = n_calls_total > 0 and (n_calls_failed / n_calls_total) <= 0.20
    print(f"[run_all] floor: {n_calls_total - n_calls_failed}/{n_calls_total} generate() calls OK "
          f"({'OK' if floor_ok else 'FAIL'}, threshold 80%)", flush=True)
    if not floor_ok:
        state["basket"] = "ALPHA_BLOCKED"
        state["basket_detail"] = f"full run: {n_calls_failed}/{n_calls_total} generate() calls failed"
        state["next_line"] = NEXT_LINES["ALPHA_BLOCKED"]
        out_path = finalize(state)
        print(f"[run_all] ALPHA_BLOCKED -- отчёт: {out_path}", flush=True)
        sys.exit(1)

    n_dead_pass = sum(1 for t in anatomy_test["u_dead"] if decomp_results[t]["final_pass"])
    alpha_decomp = n_dead_pass / len(anatomy_test["u_dead"]) if anatomy_test["u_dead"] else None
    n_all_pass = sum(1 for t in test_ids if decomp_results[t]["final_pass"])
    r_d_test = n_all_pass / len(test_ids)
    print(f"[run_all] alpha_decomp={alpha_decomp:.3f} r_D_test={r_d_test:.3f}", flush=True)

    state["alpha_decomp"] = alpha_decomp
    state["r_d_test"] = r_d_test
    state["r_whole_b2"] = WA.whole_b2_test_rate(ds, whole_by_key, test_ids, reparsed=True)

    print("\n=== [5/5] Phase D ===", flush=True)
    alpha_star = max(x for x in (alpha_b2, alpha_decomp) if x is not None)
    state["alpha_star"] = alpha_star
    print(f"[run_all] alpha_star={alpha_star:.3f}", flush=True)

    if alpha_star < 0.05:
        state["basket"] = "S0_CLOSED"
        state["basket_detail"] = "1-UNION доминирует, B2 whole/decomp не поднимает пол"
        state["next_line"] = NEXT_LINES["S0_CLOSED"]
        out_path = finalize(state)
        print(f"[run_all] S0_CLOSED -- отчёт: {out_path}", flush=True)
        sys.exit(0)

    # D1: candidate D-product = best of {whole B2, decomp} by test-r
    if state["r_whole_b2"] >= r_d_test:
        d_route = {"chosen": "whole_B2", "model": WA.B2_MODEL, "r_test": state["r_whole_b2"]}
    else:
        d_route = {"chosen": "decomp_pipeline", "planner_model": WA.B2_MODEL,
                  "executor_by_kind": executor_by_kind,
                  "planner_prompt_template": DP.build_planner_prompt.__doc__ or "see decomp_pipeline.py",
                  "r_test": r_d_test}
    (METRICS_DIR / "d_route.json").write_text(json.dumps(d_route, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"[run_all] D1 candidate product: {d_route['chosen']} (r_test={d_route['r_test']:.3f})",
          flush=True)

    # D3: basket, checked in order -- literally on r_D_test (PROTOCOL.md §3 P4's own
    # named metric = the DECOMP PIPELINE specifically), exactly as pre-registered.
    # D1 may have chosen a DIFFERENT candidate (whole_B2) as the actual best D-product --
    # that is a disclosed INTERPRETATION alongside this mechanical basket, never a
    # silent substitution into it (see PROTOCOL.md/BLOCKERS.md).
    d_outcomes = {t: {"outcome": "RESOLVED" if decomp_results[t]["final_pass"] else "UNRESOLVED"}
                 for t in test_ids}
    b3_outcomes = {t: {"outcome": "RESOLVED" if CEIL.task_solved_by_route(ds, t, ceiling["B3"]["route"])
                       else "UNRESOLVED"} for t in test_ids}
    union_outcomes = {t: {"outcome": "RESOLVED" if CEIL.task_solved_by_route(ds, t, ceiling["UNION"]["route"])
                          else "UNRESOLVED"} for t in test_ids}
    delta = ARCH2_F.paired_delta_r(d_outcomes, b3_outcomes, test_ids)
    beats_b3_stable = r_d_test > ceiling["B3"]["r_test"] and delta["ci_95"][0] > 0
    print(f"[run_all] paired_delta_r(D,B3): point={delta['point']:.3f} "
          f"ci_95={delta['ci_95']} p={delta['p_value']:.4f}", flush=True)

    if beats_b3_stable:
        basket, detail = "D_BEATS_B3", f"r_D_test={r_d_test:.3f} > B3={ceiling['B3']['r_test']:.3f}, CI>0"
    elif r_d_test > state["r_org"] + BEATS_ORG_MARGIN:
        basket, detail = "D_BEATS_ORG", f"r_D_test={r_d_test:.3f} > org+0.03={state['r_org']+0.03:.3f}"
    else:
        basket, detail = "D_ONLY_ALPHA", f"alpha_star={alpha_star:.3f}>=0.05, r_D_test<=B3"

    state["basket"] = basket
    state["basket_detail"] = detail
    state["next_line"] = NEXT_LINES["D_PRODUCT"]

    # -- Interpretation layer (disclosed, alongside, never replacing the mechanical
    # basket above): the same D_BEATS_B3/D_BEATS_ORG check applied to the ACTUALLY
    # CHOSEN D1 candidate product (here, whole_B2 -- see d_route["chosen"]), not to
    # the literal r_D_test name. Zero new live calls -- whole_B2's rate is already-
    # existing whole_grid.json data, B3/UNION are from the already-computed ceiling.
    whole_outcomes = WA.whole_b2_per_task_outcomes(ds, whole_by_key, test_ids, reparsed=True)
    chosen_outcomes = whole_outcomes if d_route["chosen"] == "whole_B2" else d_outcomes
    chosen_r = d_route["r_test"]
    chosen_vs_b3 = ARCH2_F.paired_delta_r(chosen_outcomes, b3_outcomes, test_ids)
    chosen_vs_union = ARCH2_F.paired_delta_r(chosen_outcomes, union_outcomes, test_ids)
    chosen_beats_b3_stable = chosen_r > ceiling["B3"]["r_test"] and chosen_vs_b3["ci_95"][0] > 0
    state["interpretation"] = {
        "chosen": d_route["chosen"], "chosen_r_test": chosen_r,
        "vs_b3": chosen_vs_b3, "vs_union": chosen_vs_union,
        "chosen_beats_b3_stable": chosen_beats_b3_stable,
        "chosen_beats_org": chosen_r > state["r_org"] + BEATS_ORG_MARGIN,
    }
    print(f"[run_all] INTERPRETATION: chosen={d_route['chosen']} r={chosen_r:.3f} "
          f"vs_B3 point={chosen_vs_b3['point']:.3f} ci_95={chosen_vs_b3['ci_95']} "
          f"beats_b3_stable={chosen_beats_b3_stable}", flush=True)

    state["revived"] = sorted(t for t in anatomy_test["u_dead"]
                              if chosen_outcomes[t]["outcome"] == "RESOLVED")
    state["not_revived"] = sorted(t for t in anatomy_test["u_dead"]
                                  if chosen_outcomes[t]["outcome"] != "RESOLVED")

    out_path = finalize(state)
    print(f"[run_all] {basket} -- отчёт: {out_path}", flush=True)
