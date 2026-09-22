"""run_all.py — PROTOCOL.md §4: smoke -> Branch A (12 cells) -> classify
-> BRANCH (A2 | A3[->maybe B] | B), all in one pass, no return to the
user mid-run. One REPORT_LIFE7.md at the end reflecting whichever path
actually executed.
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
import baselines as BASE           # noqa: E402


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VS = _load_module("life7_verify_seams", SCRIPTS / "verify_seams.py")

RUNS_DIR = AGENT_DIR / "runs"
REPORTS_DIR = AGENT_DIR / "reports"
METRICS_DIR = AGENT_DIR / "metrics"

SLOT_NAMES = {0: "READ", 2: "LOOKUP", 3: "COMPUTE"}
RETENTION_SLOTS = (0, 2, 3)

SEEDS_A = (20262001, 20262002, 20262003, 20262004, 20262005, 20262006)
SEEDS_A2 = (20262011, 20262012, 20262013, 20262014)
SEEDS_A3 = (20262021, 20262022, 20262023, 20262024)
SEEDS_B = (20262031, 20262032, 20262033, 20262034)

SMOKE_SEED = 999901
SMOKE_N_GEN = 8


# ---------------------------------------------------------------------
# smoke -- exercise the REAL orchestrator.run_cell path (WITH_A), just
# with a temporarily shortened generation count, rather than hand-rolling
# a separate mini-loop that could drift from the real code path.
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
        summary = O.run_cell("WITH_A", SMOKE_SEED, ds, panel, screen, out_dir)
    finally:
        O.N_GENERATIONS = old_n_gen

    births_path = out_dir / "births.jsonl"
    births = []
    if births_path.exists():
        births = [json.loads(l) for l in births_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_transfer_complementary = sum(1 for b in births
                                   if b.get("operator") == "TRANSFER_SLOT" and b.get("complementary"))

    pareto_forensic = summary.get("pareto_forensic", [])
    n_pareto_candidates = sum(f["n_candidates"] for f in pareto_forensic)
    n_pareto_exempted = sum(f["n_exempted"] for f in pareto_forensic)
    a2_log = summary.get("a2_improved_log", [])
    n_a2_improved = sum(1 for e in a2_log if e["improved"])

    print(f"[smoke] TRANSFER_SLOT complementary: {n_transfer_complementary}", flush=True)
    print(f"[smoke] Pareto candidates={n_pareto_candidates} exempted={n_pareto_exempted} "
          f"(over {len(pareto_forensic)} generations)", flush=True)
    print(f"[smoke] A2 improved-resets: {n_a2_improved}/{len(a2_log)} generations", flush=True)

    ok_pareto_fixture = VS.verify_pareto_fixture()
    ok_stall_fixture = VS.verify_stall_fixture()
    ok_registry = VS.verify_no_registry_imports()

    ok = (n_transfer_complementary >= 1) and ok_pareto_fixture and ok_stall_fixture and ok_registry
    return {"ok": ok, "n_transfer_complementary": n_transfer_complementary,
           "n_pareto_candidates": n_pareto_candidates, "n_pareto_exempted": n_pareto_exempted,
           "n_a2_improved": n_a2_improved, "n_a2_generations": len(a2_log),
           "pareto_fixture_ok": ok_pareto_fixture, "stall_fixture_ok": ok_stall_fixture,
           "no_registry_path": ok_registry, "n_generations_run": summary["n_generations"]}


# ---------------------------------------------------------------------
# cell running + per-cell metrics
# ---------------------------------------------------------------------

def run_cells_for_variant(variant: str, seeds: tuple, ds, panel: list, screen: list, label: str) -> list:
    out = []
    for seed in seeds:
        out_dir = RUNS_DIR / f"life7_{variant.lower()}_s{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        summary = O.run_cell(variant, seed, ds, panel, screen, out_dir)
        dt = time.time() - t0
        print(f"[run_all] {label} {variant} seed={seed} done in {dt:.0f}s "
              f"n_generations={summary['n_generations']} extinct={summary['extinct']}", flush=True)
        out.append({"variant": variant, "seed": seed, "out_dir": out_dir, "summary": summary, "wall_s": dt})
    return out


def compute_cell_metrics(cell_result: dict, run_id: str, ds, panel: list, imp_cache: dict, best_single: dict) -> dict:
    run_result = ML.process_run(cell_result["out_dir"], run_id, ds, panel, imp_cache, best_single)
    slot_metrics = ML.per_cell_slot_metrics(run_result, min_gen=1)
    fa_lg = ML.first_assembly_and_loss_gain(run_result)
    occ = ML.per_generation_occupancy(cell_result["out_dir"], ds, panel, best_single)
    return {"run_id": run_id, "variant": cell_result["variant"], "seed": cell_result["seed"],
           "slots": slot_metrics, "occ": occ, **fa_lg}


def metrics_for_cells(cells: list, ds, panel: list, imp_cache: dict, best_single: dict) -> list:
    out = []
    for c in cells:
        run_id = f"{c['variant']}_s{c['seed']}"
        out.append(compute_cell_metrics(c, run_id, ds, panel, imp_cache, best_single))
    return out


# ---------------------------------------------------------------------
# classify_sel -- ONE function, PROTOCOL.md §3, reused verbatim for A/A3/B
# ---------------------------------------------------------------------

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def classify_sel(with_metrics: list, ctrl_metrics: list) -> tuple:
    per_slot = {}
    n_meanL_grew = n_p90_grew = n_occ_grew = 0

    for s in RETENTION_SLOTS:
        mean_with_L = _mean([m["slots"][s]["mean_L_imp"] for m in with_metrics])
        mean_ctrl_L = _mean([m["slots"][s]["mean_L_imp"] for m in ctrl_metrics])
        mean_with_p90 = _mean([m["slots"][s]["p90_L_imp"] for m in with_metrics])
        mean_ctrl_p90 = _mean([m["slots"][s]["p90_L_imp"] for m in ctrl_metrics])
        mean_with_occ = _mean([m["occ"]["occ_imp"][s] for m in with_metrics])
        mean_ctrl_occ = _mean([m["occ"]["occ_imp"][s] for m in ctrl_metrics])

        meanL_grew = (mean_with_L is not None and mean_ctrl_L is not None
                     and mean_with_L >= mean_ctrl_L + 0.3)
        p90_grew = (mean_with_p90 is not None and mean_ctrl_p90 is not None
                   and mean_with_p90 >= mean_ctrl_p90 + 1.0)
        occ_grew = (mean_with_occ is not None and mean_ctrl_occ is not None
                   and mean_with_occ >= mean_ctrl_occ + 0.10)

        per_slot[SLOT_NAMES[s]] = {
            "mean_with_L": mean_with_L, "mean_ctrl_L": mean_ctrl_L, "meanL_grew": meanL_grew,
            "mean_with_p90": mean_with_p90, "mean_ctrl_p90": mean_ctrl_p90, "p90_grew": p90_grew,
            "mean_with_occ": mean_with_occ, "mean_ctrl_occ": mean_ctrl_occ, "occ_grew": occ_grew,
        }
        n_meanL_grew += int(meanL_grew)
        n_p90_grew += int(p90_grew)
        n_occ_grew += int(occ_grew)

    growth_signal = (n_meanL_grew >= 2) or (n_p90_grew >= 1) or (n_occ_grew >= 2)

    mean_with_ab_alive = _mean([m["occ"]["frac_gen_AB_alive"] for m in with_metrics])
    mean_ctrl_ab_alive = _mean([m["occ"]["frac_gen_AB_alive"] for m in ctrl_metrics])
    ab_alive_ok = (mean_with_ab_alive is not None and mean_ctrl_ab_alive is not None
                  and mean_with_ab_alive >= mean_ctrl_ab_alive)

    mean_with_fa = _mean([m["N_AB_first_assembly"] for m in with_metrics])
    mean_ctrl_fa = _mean([m["N_AB_first_assembly"] for m in ctrl_metrics])
    assembly_ok_works = (mean_with_fa is not None and mean_ctrl_fa is not None
                        and mean_with_fa >= mean_ctrl_fa - 1)
    assembly_fail = (mean_with_fa is not None and mean_ctrl_fa is not None
                    and mean_with_fa <= mean_ctrl_fa - 3)

    with_extinct = [m["occ"]["extinct_gen"] for m in with_metrics]
    ctrl_extinct = [m["occ"]["extinct_gen"] for m in ctrl_metrics]
    mean_with_extinct = _mean(with_extinct)
    mean_ctrl_extinct = _mean(ctrl_extinct)
    extinct_mean_ok = (mean_with_extinct is not None and mean_ctrl_extinct is not None
                      and mean_with_extinct >= mean_ctrl_extinct)
    n_pairs = min(len(with_extinct), len(ctrl_extinct))
    n_extinct_earlier = sum(1 for i in range(n_pairs) if with_extinct[i] < ctrl_extinct[i])
    # PROTOCOL.md's literal "≥4 of 6" is Branch A's 6-seed design; classify_sel is reused
    # verbatim for A3/B (4 seeds each), where an absolute count of 4 could never fire --
    # generalized as the same PROPORTION (4/6 ≈ 2/3), applied to whatever n_pairs actually is.
    extinct_consistently_earlier = (n_pairs > 0 and n_extinct_earlier / n_pairs >= 4 / 6)

    detail = {"per_slot": per_slot, "n_meanL_grew": n_meanL_grew, "n_p90_grew": n_p90_grew,
             "n_occ_grew": n_occ_grew, "growth_signal": growth_signal,
             "mean_with_frac_gen_AB_alive": mean_with_ab_alive, "mean_ctrl_frac_gen_AB_alive": mean_ctrl_ab_alive,
             "mean_with_first_assembly": mean_with_fa, "mean_ctrl_first_assembly": mean_ctrl_fa,
             "mean_with_extinct_gen": mean_with_extinct, "mean_ctrl_extinct_gen": mean_ctrl_extinct,
             "n_extinct_earlier_of_pairs": f"{n_extinct_earlier}/{n_pairs}"}

    if growth_signal and ab_alive_ok and assembly_ok_works and extinct_mean_ok:
        return "WORKS", detail
    if (not growth_signal) or assembly_fail or extinct_consistently_earlier:
        return "FAIL", detail
    return "PARTIAL", detail


# ---------------------------------------------------------------------
# report
# ---------------------------------------------------------------------

def _fmt(v, spec=".2f"):
    return "n/d" if v is None else format(v, spec)


def _slot_table(per_slot: dict) -> list:
    lines = ["| slot | mean_L WITH | CTRL | grew? | p90_L WITH | CTRL | grew? | occ_imp WITH | CTRL | grew? |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("READ", "LOOKUP", "COMPUTE"):
        p = per_slot[name]
        lines.append(f"| {name} | {_fmt(p['mean_with_L'])} | {_fmt(p['mean_ctrl_L'])} | {p['meanL_grew']} | "
                     f"{_fmt(p['mean_with_p90'])} | {_fmt(p['mean_ctrl_p90'])} | {p['p90_grew']} | "
                     f"{_fmt(p['mean_with_occ'])} | {_fmt(p['mean_ctrl_occ'])} | {p['occ_grew']} |")
    return lines


def _forensic_lines(cells_with: list) -> list:
    if not cells_with:
        return []
    first = cells_with[0]["summary"]
    pareto_forensic = first.get("pareto_forensic", [])
    n_candidates = sum(f["n_candidates"] for f in pareto_forensic)
    n_exempted = sum(f["n_exempted"] for f in pareto_forensic)
    a2_log = first.get("a2_improved_log", [])
    n_improved = sum(1 for e in a2_log if e["improved"])
    lines = [
        f"На первом WITH-seed ({cells_with[0]['seed']}): Pareto-кандидатов на смерть="
        f"{n_candidates}, исключено (недоминированы)={n_exempted}.",
        f"A2: {n_improved}/{len(a2_log)} поколений сбросили stall по улучшению r/n_imp.",
    ]
    return lines


def write_report(state: dict) -> Path:
    L = []
    A = L.append
    A("# REPORT_LIFE7 — selection timescale: Pareto-survival + stall-from-improvement\n")
    A("Спека: `PROTOCOL.md`. Default-канал = ASSEMBLE + complementary slot-HGT (`p=0.35`), "
      "БЕЗ slot-safe mutate, БЕЗ M=2-youngest immunity (оба — LIFE-6, оба FAIL по median_L). "
      "Живая сетка ПЕРЕИСПОЛЬЗОВАНА byte-identical из `agent_life6_retention` (md5 подтверждён), "
      "0 новых generate() вызовов.\n")

    A("## 1. Рамка\n")
    A("LIFE-6: `RET_A_FAIL`/`RET_B_FAIL` по median_L, но mean_L/occ показывали слабый сигнал. "
      "LIFE-7 меняет слой ПОД retention-рычагами: КТО умирает (Pareto-доминирование по (r,n_imp) "
      "поверх обычных D1/D2) и КОГДА объявляется stall (улучшение best_r/best_n_imp, не gate).\n")

    A("## 2. Смоук\n")
    smoke = state["smoke"]
    A(f"seed={SMOKE_SEED}, {smoke['n_generations_run']} поколений: TRANSFER_SLOT complementary="
      f"{smoke['n_transfer_complementary']}, Pareto candidates={smoke['n_pareto_candidates']} "
      f"exempted={smoke['n_pareto_exempted']}, A2 improved={smoke['n_a2_improved']}/"
      f"{smoke['n_a2_generations']}, юнит-фикстуры Pareto={smoke['pareto_fixture_ok']} "
      f"stall={smoke['stall_fixture_ok']}, 0 registry-путей={smoke['no_registry_path']}. "
      f"Смоук: {'PASS' if smoke['ok'] else 'FAIL'}.\n")

    A("## 3. Branch A: 6 WITH_A + 6 CTRL_A\n")
    A(", ".join(f"{c['variant']}/{c['seed']}: n_gen={c['summary']['n_generations']} "
                f"extinct={c['summary']['extinct']}" for c in state["cells_A"]))
    A("")
    A("\n".join(_slot_table(state["detail_A"]["per_slot"])))
    d = state["detail_A"]
    A(f"\nmean frac_gen_AB_alive: WITH={_fmt(d['mean_with_frac_gen_AB_alive'])}, "
      f"CTRL={_fmt(d['mean_ctrl_frac_gen_AB_alive'])}")
    A(f"mean N_AB_first_assembly: WITH={_fmt(d['mean_with_first_assembly'])}, "
      f"CTRL={_fmt(d['mean_ctrl_first_assembly'])}")
    A(f"mean extinct_gen: WITH={_fmt(d['mean_with_extinct_gen'])}, CTRL={_fmt(d['mean_ctrl_extinct_gen'])}, "
      f"seeds where WITH extinct earlier: {d['n_extinct_earlier_of_pairs']}\n")
    A("\n".join(_forensic_lines(state["cells_with_A"])))
    A(f"\n\n**Корзина A: {state['basket_A']}**\n")

    if state.get("phase") == "A2":
        A("## 4. Phase A2 (SEL_A_WORKS): устойчивость на 4 новых WITH_A-only seed\n")
        A("\n".join(_slot_table(state["detail_A2_vs_A_ctrl"]["per_slot"])))
        A("\n(сравнение против Branch A's уже посчитанного CTRL_A — новый CTRL не строился)\n")

    elif state.get("phase") in ("A3", "A3_then_B"):
        A("## 4. Phase A3 (SEL_A_PARTIAL): vector-Pareto (per-slot Imp), 4 WITH_A3 + 4 CTRL_A3\n")
        A("\n".join(_slot_table(state["detail_A3"]["per_slot"])))
        A(f"\n**Корзина A3: {state['basket_A3']}**\n")
        if state.get("phase") == "A3_then_B":
            A("\nA3 не WORKS -> переход к Branch B в этом же прогоне.\n")

    if state.get("phase") in ("B", "A3_then_B"):
        A("## 5. Branch B: stall-from-improvement ONLY (без Pareto-смерти), 4 WITH_B + 4 CTRL_B\n")
        A("\n".join(_slot_table(state["detail_B"]["per_slot"])))
        db = state["detail_B"]
        A(f"\nmean N_AB_first_assembly: WITH={_fmt(db['mean_with_first_assembly'])}, "
          f"CTRL={_fmt(db['mean_ctrl_first_assembly'])}")
        A(f"mean extinct_gen: WITH={_fmt(db['mean_with_extinct_gen'])}, CTRL={_fmt(db['mean_ctrl_extinct_gen'])}\n")
        A(f"\n**Корзина B: {state['basket_B']}**\n")

    A("## 6. Итоговый default\n")
    A(state["final_default_line"])

    A("\n## 7. Non-claims\n")
    A("- Не заявляется достижение/сравнение с B3 в заголовке.\n"
      "- Нет block registry/BLOCK_INSERT/R1-R4/A1-A5/slot-safe-mutate/M=2-youngest-immunity нигде здесь.\n"
      "- Нет новых атомов/моделей.\n"
      "- Победа НЕ объявлена по одной лишь `median_L_imp` (secondary here, per PROTOCOL.md/LIFE-6 lesson).\n"
      "- `arch2/`, `agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`, "
      "`agent_life3_block_live/`, `agent_life4_block_fix/`, `agent_life5_slot_map/`, "
      "`agent_life6_retention/` не изменялись.")

    A("\n## 8. Дальше (ровно одна строка)\n")
    A(state["next_line"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "REPORT_LIFE7.md"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=== [1/5] verify_seams ===", flush=True)
    ok_seams = VS.verify_seams()
    ok_grid = VS.verify_grid_copy()
    ok_registry0 = VS.verify_no_registry_imports()
    ok_pareto0 = VS.verify_pareto_fixture()
    ok_stall0 = VS.verify_stall_fixture()
    if not (ok_seams and ok_grid and ok_registry0 and ok_pareto0 and ok_stall0):
        print("[run_all] verify_seams FAILED -> BLOCKERS.md, стоп", flush=True)
        sys.exit(1)

    print("\n=== [2/5] smoke ===", flush=True)
    smoke = run_smoke()
    if not smoke["ok"]:
        print(f"[run_all] smoke FAILED: {smoke} -> BLOCKERS.md, стоп", flush=True)
        sys.exit(1)

    ds = LD.default_dataset()
    panel_info = O.build_panel(ds)
    panel, screen = panel_info["panel"], panel_info["screen"]
    imp_cache: dict = {}
    best_single = ML.best_single_per_slot(ds, panel)

    print("\n=== [3/5] Branch A: 6 WITH_A + 6 CTRL_A ===", flush=True)
    t0 = time.time()
    cells_with_A = run_cells_for_variant("WITH_A", SEEDS_A, ds, panel, screen, "A")
    cells_ctrl_A = run_cells_for_variant("CTRL_A", SEEDS_A, ds, panel, screen, "A")
    print(f"[run_all] Branch A: {time.time()-t0:.0f}s for 12 cells", flush=True)

    metrics_with_A = metrics_for_cells(cells_with_A, ds, panel, imp_cache, best_single)
    metrics_ctrl_A = metrics_for_cells(cells_ctrl_A, ds, panel, imp_cache, best_single)
    basket_A, detail_A = classify_sel(metrics_with_A, metrics_ctrl_A)
    print(f"[run_all] basket_A = {basket_A}: {detail_A}", flush=True)

    state = {"smoke": smoke, "cells_A": cells_with_A + cells_ctrl_A, "cells_with_A": cells_with_A,
            "basket_A": basket_A, "detail_A": detail_A}

    print(f"\n=== [4/5] branch dispatch on SEL_A_{basket_A} ===", flush=True)
    if basket_A == "WORKS":
        state["phase"] = "A2"
        cells_A2 = run_cells_for_variant("WITH_A", SEEDS_A2, ds, panel, screen, "A2")
        metrics_A2 = metrics_for_cells(cells_A2, ds, panel, imp_cache, best_single)
        _, detail_A2_vs_A_ctrl = classify_sel(metrics_A2, metrics_ctrl_A)
        state["detail_A2_vs_A_ctrl"] = detail_A2_vs_A_ctrl
        state["final_default_line"] = "default = HGT + Pareto-survival (scalar n_imp) + stall-from-improvement."
        state["next_line"] = ("default = HGT + Pareto(scalar n_imp) + stall-from-improvement; следующий "
                             "крупный пакет = r vs B2/B3 на этом отборе.")

    elif basket_A == "PARTIAL":
        state["phase"] = "A3"
        cells_with_A3 = run_cells_for_variant("WITH_A3", SEEDS_A3, ds, panel, screen, "A3")
        cells_ctrl_A3 = run_cells_for_variant("CTRL_A3", SEEDS_A3, ds, panel, screen, "A3")
        metrics_with_A3 = metrics_for_cells(cells_with_A3, ds, panel, imp_cache, best_single)
        metrics_ctrl_A3 = metrics_for_cells(cells_ctrl_A3, ds, panel, imp_cache, best_single)
        basket_A3, detail_A3 = classify_sel(metrics_with_A3, metrics_ctrl_A3)
        state["basket_A3"] = basket_A3
        state["detail_A3"] = detail_A3
        print(f"[run_all] basket_A3 = {basket_A3}: {detail_A3}", flush=True)

        if basket_A3 == "WORKS":
            state["final_default_line"] = "default = HGT + Pareto-survival (vector per-slot Imp) + stall-from-improvement."
            state["next_line"] = ("default = HGT + Pareto(vector per-slot) + stall-from-improvement; "
                                 "следующий крупный пакет = r vs B2/B3 на этом отборе.")
        else:
            state["phase"] = "A3_then_B"
            cells_with_B = run_cells_for_variant("WITH_B", SEEDS_B, ds, panel, screen, "B")
            cells_ctrl_B = run_cells_for_variant("CTRL_B", SEEDS_B, ds, panel, screen, "B")
            metrics_with_B = metrics_for_cells(cells_with_B, ds, panel, imp_cache, best_single)
            metrics_ctrl_B = metrics_for_cells(cells_ctrl_B, ds, panel, imp_cache, best_single)
            basket_B, detail_B = classify_sel(metrics_with_B, metrics_ctrl_B)
            state["basket_B"] = basket_B
            state["detail_B"] = detail_B
            print(f"[run_all] basket_B = {basket_B}: {detail_B}", flush=True)
            if basket_B == "WORKS":
                state["final_default_line"] = "default = HGT + stall-from-improvement (смерть не тронута)."
                state["next_line"] = "default = HGT + stall-from-improvement; следующий крупный пакет = r vs B2/B3."
            else:
                state["final_default_line"] = "default = HGT only -- ни один рычаг selection-timescale не сработал."
                state["next_line"] = ("ни Pareto-смерть, ни stall-без-B3 не держат слот во времени при "
                                     "pop=20; следующий пакет — смена размера pop/темпа replace ИЛИ "
                                     "измерение r vs B2/B3 на HGT-only; не атомы, не registry, не "
                                     "youngest-immunity.")

    else:  # FAIL
        state["phase"] = "B"
        cells_with_B = run_cells_for_variant("WITH_B", SEEDS_B, ds, panel, screen, "B")
        cells_ctrl_B = run_cells_for_variant("CTRL_B", SEEDS_B, ds, panel, screen, "B")
        metrics_with_B = metrics_for_cells(cells_with_B, ds, panel, imp_cache, best_single)
        metrics_ctrl_B = metrics_for_cells(cells_ctrl_B, ds, panel, imp_cache, best_single)
        basket_B, detail_B = classify_sel(metrics_with_B, metrics_ctrl_B)
        state["basket_B"] = basket_B
        state["detail_B"] = detail_B
        print(f"[run_all] basket_B = {basket_B}: {detail_B}", flush=True)
        if basket_B == "WORKS":
            state["final_default_line"] = "default = HGT + stall-from-improvement (смерть не тронута)."
            state["next_line"] = "default = HGT + stall-from-improvement; следующий крупный пакет = r vs B2/B3."
        else:
            state["final_default_line"] = "default = HGT only -- ни один рычаг selection-timescale не сработал."
            state["next_line"] = ("ни Pareto-смерть, ни stall-без-B3 не держат слот во времени при pop=20; "
                                 "следующий пакет — смена размера pop/темпа replace ИЛИ измерение r vs "
                                 "B2/B3 на HGT-only; не атомы, не registry, не youngest-immunity.")

    print("\n=== [5/5] report ===", flush=True)
    out_path = write_report(state)
    (METRICS_DIR / "final_state.json").write_text(
        json.dumps({k: v for k, v in state.items() if k not in ("cells_A", "cells_with_A")},
                  ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"[run_all] отчёт записан: {out_path}", flush=True)
