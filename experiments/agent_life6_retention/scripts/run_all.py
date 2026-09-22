"""run_all.py — PROTOCOL.md §4: smoke -> Branch A (12 cells) -> classify
-> BRANCH (A2 | A3[->maybe B] | B), all in one pass, no return to the
user mid-run. One REPORT_LIFE6.md at the end reflecting whichever path
actually executed.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import random
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
import lean_seeds as LS            # noqa: E402
import slot_safe_mutate as SSM     # noqa: E402
import default_channel as DC       # noqa: E402
import baselines as BASE           # noqa: E402


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VS = _load_module("life6_verify_seams", SCRIPTS / "verify_seams.py")

RUNS_DIR = AGENT_DIR / "runs"
REPORTS_DIR = AGENT_DIR / "reports"
METRICS_DIR = AGENT_DIR / "metrics"

SLOT_NAMES = {0: "READ", 2: "LOOKUP", 3: "COMPUTE"}
RETENTION_SLOTS = (0, 2, 3)

SEEDS_A = (20261901, 20261902, 20261903, 20261904, 20261905, 20261906)
SEEDS_A2 = (20261911, 20261912, 20261913, 20261914)
SEEDS_A3 = (20261921, 20261922, 20261923, 20261924)
SEEDS_B = (20261931, 20261932, 20261933, 20261934)

SMOKE_SEED = 999901
SMOKE_N_GEN = 8


# ---------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------

def run_smoke() -> dict:
    ds = LD.default_dataset()
    panel_info = O.build_panel(ds)
    panel, screen = panel_info["panel"], panel_info["screen"]
    best_single = ML.best_single_per_slot(ds, panel)

    out_dir = RUNS_DIR / "smoke_throwaway"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    births_path = out_dir / "births.jsonl"

    reg = LD.HSTEP.Registry()
    be = LD.HSTEP.HeterostepBackend(ds)

    seeds_pop = LS.build_seeds_pop20_lean(ds, panel, best_single)
    LS.assert_lean_population(seeds_pop, ds, panel, best_single, reg)
    print("[smoke] lean-20-distinct + |ImpSet|<2 assert: PASS", flush=True)

    imp_cache: dict = {}

    def imp_fn(genotype: dict) -> frozenset:
        cid = genotype.get("complex_id")
        if cid is not None and cid in imp_cache:
            return imp_cache[cid]
        val = ML.imp_set(ds, panel, best_single, genotype, reg)
        if cid is not None:
            imp_cache[cid] = val
        return val

    rng = random.Random(SMOKE_SEED)
    exp_id = f"life6_smoke_s{SMOKE_SEED}"
    ev = LD.E.Evolution(
        registry=reg, backend=be, dataset=ds, experiment_id=exp_id,
        runs_dir=out_dir / "traces", rng=rng, screen_ids=screen,
        control_slices=[panel], holdout_ids=[],
        assemble_n_steps=len(LD.HSTEP.STEP_KINDS), budget_per_task=O.BUDGET_PER_TASK,
        g_stall=O.G_STALL,
    )
    ev.current_gen = 0
    ev.seed(seeds_pop, reference_names=LD.HSEED.REFERENCE_NAMES,
            baseline_name=LD.HSEED.BASELINE_NAME, gate_reference_name=LD.HSEED.GATE_REFERENCE_NAME)

    # `slot_safe_mutate`'s extra report fields (`slot_safe_bypassed` etc.)
    # are INTENTIONALLY not part of `births.jsonl`'s frozen cross-variant
    # schema (`default_channel._write_births_row` only captures the fixed
    # LIFE-3..5 field set, on purpose -- see its docstring) -- so this
    # smoke check counts directly via a wrapping closure, not by reading
    # them back out of births.jsonl afterwards (that would always read 0,
    # which is not itself a bug in the schema, only in a naive read-back).
    counts = {"strict": 0, "bypassed": 0}

    def counting_mutate_fn(parent, rng_, ctx, registry_, gen_):
        child, report = SSM.slot_safe_mutate(parent, rng_, ctx, registry_, gen_, imp_fn, p_safe=O.P_SAFE_A)
        if report.get("slot_safe_bypassed"):
            counts["bypassed"] += 1
        elif "slot_safe_bypassed" in report:
            counts["strict"] += 1
        return child, report

    custom_reproduce = DC.make_default_reproduce(births_path, imp_fn, mutate_fn=counting_mutate_fn)

    for gen in range(SMOKE_N_GEN):
        ev.run_generation(gen)
        if ev.extinct:
            print(f"[smoke] extinct at gen={gen}", flush=True)
            break
        if gen < SMOKE_N_GEN - 1:
            res_data = LD.F.load_results(out_dir / "traces", exp_id)
            custom_reproduce(ev, gen + 1, res_data, panel)

    births = []
    if births_path.exists():
        births = [json.loads(l) for l in births_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_transfer_complementary = sum(1 for b in births
                                   if b.get("operator") == "TRANSFER_SLOT" and b.get("complementary"))
    n_slot_safe_strict = counts["strict"]
    n_slot_safe_bypassed = counts["bypassed"]
    print(f"[smoke] TRANSFER_SLOT complementary: {n_transfer_complementary}, "
          f"slot_safe strict={n_slot_safe_strict} bypassed={n_slot_safe_bypassed}", flush=True)

    fixture_ok = VS.verify_slot_safe_fixture()

    # Confirm 0 registry-adjacent CODE via `ast` (import statements only) --
    # NOT a substring search over the raw text, which would false-positive
    # on this very file's own docstring (it legitimately explains the
    # ABSENCE of registry code using the words "block registry"/
    # "BLOCK_INSERT" in prose -- a naive `"BLOCK_INSERT" in text` check
    # already caught itself doing this once, see BLOCKERS.md).
    import ast
    dispatcher_src = (SRC / "default_channel.py").read_text(encoding="utf-8")
    tree = ast.parse(dispatcher_src)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    forbidden = {"block_registry", "block_insert"}
    has_registry_path = bool(imported_names & forbidden)
    print(f"[smoke] imports in default_channel.py: {sorted(imported_names)}", flush=True)
    print(f"[smoke] 0 registry-adjacent code in default_channel.py: {not has_registry_path}", flush=True)

    ok = (n_transfer_complementary >= 1) and fixture_ok and not has_registry_path
    return {"ok": ok, "n_transfer_complementary": n_transfer_complementary,
           "n_slot_safe_strict": n_slot_safe_strict, "n_slot_safe_bypassed": n_slot_safe_bypassed,
           "fixture_ok": fixture_ok, "no_registry_path": not has_registry_path,
           "n_generations_run": gen + 1}


# ---------------------------------------------------------------------
# cell running + per-cell metrics
# ---------------------------------------------------------------------

def run_cells_for_variant(variant: str, seeds: tuple, ds, panel: list, screen: list, label: str) -> list:
    out = []
    for seed in seeds:
        out_dir = RUNS_DIR / f"life6_{variant.lower()}_s{seed}"
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
    return {"run_id": run_id, "variant": cell_result["variant"], "seed": cell_result["seed"],
           "slots": slot_metrics, **fa_lg}


def metrics_for_cells(cells: list, ds, panel: list, imp_cache: dict, best_single: dict) -> list:
    out = []
    for c in cells:
        run_id = f"{c['variant']}_s{c['seed']}"
        out.append(compute_cell_metrics(c, run_id, ds, panel, imp_cache, best_single))
    return out


# ---------------------------------------------------------------------
# classify_ret -- ONE function, PROTOCOL.md §3, reused verbatim for A/A3/B
# ---------------------------------------------------------------------

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def classify_ret(with_metrics: list, ctrl_metrics: list) -> tuple:
    per_slot = {}
    n_L_grew = n_frac_grew = 0
    all_coverage_ok = True
    all_delta_L_small = True

    for s in RETENTION_SLOTS:
        mean_with_L = _mean([m["slots"][s]["median_L_imp"] for m in with_metrics])
        mean_ctrl_L = _mean([m["slots"][s]["median_L_imp"] for m in ctrl_metrics])
        mean_with_frac = _mean([m["slots"][s]["frac_L_ge3"] for m in with_metrics])
        mean_ctrl_frac = _mean([m["slots"][s]["frac_L_ge3"] for m in ctrl_metrics])
        mean_with_nsig = _mean([m["slots"][s]["n_imp_sig"] for m in with_metrics]) or 0.0
        mean_ctrl_nsig = _mean([m["slots"][s]["n_imp_sig"] for m in ctrl_metrics]) or 0.0

        L_grew = (mean_with_L is not None and mean_ctrl_L is not None
                 and mean_with_L >= mean_ctrl_L + 0.5)
        frac_grew = (mean_with_frac is not None and mean_ctrl_frac is not None
                    and mean_with_frac >= mean_ctrl_frac + 0.05)
        coverage_ok = (mean_ctrl_nsig == 0) or (mean_with_nsig >= 0.7 * mean_ctrl_nsig)
        delta_L_small = (mean_with_L is None or mean_ctrl_L is None
                        or (mean_with_L - mean_ctrl_L) < 0.5)

        per_slot[SLOT_NAMES[s]] = {
            "mean_with_L": mean_with_L, "mean_ctrl_L": mean_ctrl_L, "L_grew": L_grew,
            "mean_with_frac": mean_with_frac, "mean_ctrl_frac": mean_ctrl_frac, "frac_grew": frac_grew,
            "mean_with_nsig": mean_with_nsig, "mean_ctrl_nsig": mean_ctrl_nsig, "coverage_ok": coverage_ok,
        }
        n_L_grew += int(L_grew)
        n_frac_grew += int(frac_grew)
        all_coverage_ok = all_coverage_ok and coverage_ok
        all_delta_L_small = all_delta_L_small and delta_L_small

    mean_with_fa = _mean([m["N_AB_first_assembly"] for m in with_metrics])
    mean_ctrl_fa = _mean([m["N_AB_first_assembly"] for m in ctrl_metrics])
    assembly_ok_works = (mean_with_fa is not None and mean_ctrl_fa is not None
                        and mean_with_fa >= mean_ctrl_fa - 1)
    assembly_fail = (mean_with_fa is not None and mean_ctrl_fa is not None
                    and mean_with_fa <= mean_ctrl_fa - 3)

    detail = {"per_slot": per_slot, "mean_with_first_assembly": mean_with_fa,
             "mean_ctrl_first_assembly": mean_ctrl_fa,
             "n_slots_L_grew": n_L_grew, "n_slots_frac_grew": n_frac_grew,
             "all_coverage_ok": all_coverage_ok}

    if n_L_grew >= 2 and n_frac_grew >= 1 and assembly_ok_works and all_coverage_ok:
        return "WORKS", detail
    if all_delta_L_small or assembly_fail:
        return "FAIL", detail
    return "PARTIAL", detail


# ---------------------------------------------------------------------
# report
# ---------------------------------------------------------------------

def _fmt(v, spec=".2f"):
    return "n/d" if v is None else format(v, spec)


def _slot_table(per_slot: dict) -> list:
    lines = ["| slot | mean_L WITH | CTRL | grew? | frac_ge3 WITH | CTRL | grew? | n_imp_sig WITH | CTRL | cov_ok? |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("READ", "LOOKUP", "COMPUTE"):
        p = per_slot[name]
        lines.append(f"| {name} | {_fmt(p['mean_with_L'])} | {_fmt(p['mean_ctrl_L'])} | {p['L_grew']} | "
                     f"{_fmt(p['mean_with_frac'])} | {_fmt(p['mean_ctrl_frac'])} | {p['frac_grew']} | "
                     f"{_fmt(p['mean_with_nsig'],'.1f')} | {_fmt(p['mean_ctrl_nsig'],'.1f')} | {p['coverage_ok']} |")
    return lines


def write_report(state: dict) -> Path:
    L = []
    A = L.append
    A("# REPORT_LIFE6 — slot retention, single-pass branching\n")
    A("Спека: `PROTOCOL.md`. Default-канал = ASSEMBLE + complementary slot-HGT (`p=0.35`), "
      "БЕЗ block registry/BLOCK_INSERT/R1-R4/A1-A5 -- этаж «блок» не возобновлялся "
      "(LIFE-3/4: `UNIT_LIVE_FAIL`/`UNIT4_FAIL`). Живая сетка ПЕРЕИСПОЛЬЗОВАНА byte-identical "
      "из `agent_life5_slot_map` (md5 подтверждён), 0 новых generate() вызовов.\n")

    A("## 1. Рамка\n")
    A("LIFE-5: `MAP_STABILITY_GAP` на READ/LOOKUP/COMPUTE -- Imp=1-сигнатур много, "
      "`median_L≈1` почти везде. LIFE-6 проверяет, поднимает ли удержание (а) slot-safe "
      "mutate (Branch A) или, если нет, (б) carrier soft-immunity (Branch B).\n")

    A("## 2. Смоук\n")
    smoke = state["smoke"]
    A(f"seed={SMOKE_SEED}, {smoke['n_generations_run']} поколений: TRANSFER_SLOT complementary="
      f"{smoke['n_transfer_complementary']}, slot-safe strict={smoke['n_slot_safe_strict']} "
      f"bypassed={smoke['n_slot_safe_bypassed']}, fixture={smoke['fixture_ok']}, "
      f"0 registry-путей={smoke['no_registry_path']}. Смоук: {'PASS' if smoke['ok'] else 'FAIL'}.\n")

    A("## 3. Branch A: 6 WITH_A + 6 CTRL\n")
    A(", ".join(f"{c['variant']}/{c['seed']}: n_gen={c['summary']['n_generations']} "
                f"extinct={c['summary']['extinct']}" for c in state["cells_A"]))
    A("")
    A("\n".join(_slot_table(state["detail_A"]["per_slot"])))
    A(f"\nmean N_AB_first_assembly: WITH={_fmt(state['detail_A']['mean_with_first_assembly'])}, "
      f"CTRL={_fmt(state['detail_A']['mean_ctrl_first_assembly'])}\n")
    A(f"\n**Корзина A: {state['basket_A']}**\n")

    if state.get("phase") == "A2":
        A("## 4. Phase A2 (RET_A_WORKS): устойчивость на 4 новых WITH_A-only seed\n")
        A("\n".join(_slot_table(state["detail_A2_vs_A_ctrl"]["per_slot"])))
        A(f"\n(сравнение против Branch A's уже посчитанного CTRL — новый CTRL не строился)\n")

    elif state.get("phase") in ("A3", "A3_then_B"):
        A("## 4. Phase A3 (RET_A_PARTIAL): p_safe=0.0, 4 WITH_A_hard + 4 CTRL_A3\n")
        A("\n".join(_slot_table(state["detail_A3"]["per_slot"])))
        A(f"\n**Корзина A3: {state['basket_A3']}**\n")
        if state.get("phase") == "A3_then_B":
            A("\nA3 не WORKS -> переход к Branch B в этом же прогоне.\n")

    if state.get("phase") in ("B", "A3_then_B"):
        A("## 5. Branch B: carrier soft-immunity, 4 WITH_B + 4 CTRL_B\n")
        A("\n".join(_slot_table(state["detail_B"]["per_slot"])))
        A(f"\nmean N_AB_first_assembly: WITH={_fmt(state['detail_B']['mean_with_first_assembly'])}, "
          f"CTRL={_fmt(state['detail_B']['mean_ctrl_first_assembly'])}\n")
        A(f"\n**Корзина B: {state['basket_B']}**\n")

    A("## 6. Итоговый default\n")
    A(state["final_default_line"])

    A("\n## 7. Non-claims\n")
    A("- Не заявляется достижение/сравнение с B3 в заголовке.\n"
      "- Нет block registry/BLOCK_INSERT/R1-R4/A1-A5 нигде в этом пакете.\n"
      "- Нет новых атомов/моделей.\n"
      "- `arch2/`, `agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`, "
      "`agent_life3_block_live/`, `agent_life4_block_fix/`, `agent_life5_slot_map/` не изменялись.")

    A("\n## 8. Дальше (ровно одна строка)\n")
    A(state["next_line"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "REPORT_LIFE6.md"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=== [1/5] verify_seams ===", flush=True)
    ok_seams = VS.verify_seams()
    ok_grid = VS.verify_grid_copy()
    ok_fixture = VS.verify_slot_safe_fixture()
    if not (ok_seams and ok_grid and ok_fixture):
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

    print("\n=== [3/5] Branch A: 6 WITH_A + 6 CTRL ===", flush=True)
    t0 = time.time()
    cells_with_A = run_cells_for_variant("WITH_A", SEEDS_A, ds, panel, screen, "A")
    cells_ctrl_A = run_cells_for_variant("CTRL", SEEDS_A, ds, panel, screen, "A")
    print(f"[run_all] Branch A: {time.time()-t0:.0f}s for 12 cells", flush=True)

    metrics_with_A = metrics_for_cells(cells_with_A, ds, panel, imp_cache, best_single)
    metrics_ctrl_A = metrics_for_cells(cells_ctrl_A, ds, panel, imp_cache, best_single)
    basket_A, detail_A = classify_ret(metrics_with_A, metrics_ctrl_A)
    print(f"[run_all] basket_A = {basket_A}: {detail_A}", flush=True)

    state = {"smoke": smoke, "cells_A": cells_with_A + cells_ctrl_A,
            "basket_A": basket_A, "detail_A": detail_A}

    print(f"\n=== [4/5] branch dispatch on RET_A_{basket_A} ===", flush=True)
    if basket_A == "WORKS":
        state["phase"] = "A2"
        cells_A2 = run_cells_for_variant("WITH_A", SEEDS_A2, ds, panel, screen, "A2")
        metrics_A2 = metrics_for_cells(cells_A2, ds, panel, imp_cache, best_single)
        _, detail_A2_vs_A_ctrl = classify_ret(metrics_A2, metrics_ctrl_A)
        state["detail_A2_vs_A_ctrl"] = detail_A2_vs_A_ctrl
        state["final_default_line"] = "default = HGT + slot-safe mutate (p_safe=0.05)."
        state["next_line"] = ("default = HGT + slot-safe; следующий крупный пакет = r vs B2/B3 "
                             "на этом канале ИЛИ multi-slot overlap lifetimes; не registry.")

    elif basket_A == "PARTIAL":
        state["phase"] = "A3"
        cells_with_A3 = run_cells_for_variant("WITH_A_hard", SEEDS_A3, ds, panel, screen, "A3")
        cells_ctrl_A3 = run_cells_for_variant("CTRL_A3", SEEDS_A3, ds, panel, screen, "A3")
        metrics_with_A3 = metrics_for_cells(cells_with_A3, ds, panel, imp_cache, best_single)
        metrics_ctrl_A3 = metrics_for_cells(cells_ctrl_A3, ds, panel, imp_cache, best_single)
        basket_A3, detail_A3 = classify_ret(metrics_with_A3, metrics_ctrl_A3)
        state["basket_A3"] = basket_A3
        state["detail_A3"] = detail_A3
        print(f"[run_all] basket_A3 = {basket_A3}: {detail_A3}", flush=True)

        if basket_A3 == "WORKS":
            state["final_default_line"] = "default = HGT + slot-safe mutate (p_safe=0.0, полный запрет)."
            state["next_line"] = ("default = HGT + жёсткий slot-safe (p_safe=0.0); следующий крупный "
                                 "пакет = r vs B2/B3 на этом канале ИЛИ multi-slot overlap lifetimes; "
                                 "не registry.")
        else:
            state["phase"] = "A3_then_B"
            cells_with_B = run_cells_for_variant("WITH_B", SEEDS_B, ds, panel, screen, "B")
            cells_ctrl_B = run_cells_for_variant("CTRL_B", SEEDS_B, ds, panel, screen, "B")
            metrics_with_B = metrics_for_cells(cells_with_B, ds, panel, imp_cache, best_single)
            metrics_ctrl_B = metrics_for_cells(cells_ctrl_B, ds, panel, imp_cache, best_single)
            basket_B, detail_B = classify_ret(metrics_with_B, metrics_ctrl_B)
            state["basket_B"] = basket_B
            state["detail_B"] = detail_B
            print(f"[run_all] basket_B = {basket_B}: {detail_B}", flush=True)
            if basket_B == "WORKS":
                state["final_default_line"] = "default = HGT + carrier soft-immunity (M=2)."
                state["next_line"] = "default = HGT + immunity; следующий крупный пакет = r vs B2/B3 на этом канале."
            else:
                state["final_default_line"] = "default = HGT only (без slot-safe, без immunity) -- ни один рычаг retention не сработал."
                state["next_line"] = ("удержание не берётся ни mutate-safe, ни immunity при pop=20/gate; "
                                     "следующий пакет -- только смена selection timescale (Pareto Imp-slots "
                                     "vs r ИЛИ ослабление gate), не атомы, не registry.")

    else:  # FAIL
        state["phase"] = "B"
        cells_with_B = run_cells_for_variant("WITH_B", SEEDS_B, ds, panel, screen, "B")
        cells_ctrl_B = run_cells_for_variant("CTRL_B", SEEDS_B, ds, panel, screen, "B")
        metrics_with_B = metrics_for_cells(cells_with_B, ds, panel, imp_cache, best_single)
        metrics_ctrl_B = metrics_for_cells(cells_ctrl_B, ds, panel, imp_cache, best_single)
        basket_B, detail_B = classify_ret(metrics_with_B, metrics_ctrl_B)
        state["basket_B"] = basket_B
        state["detail_B"] = detail_B
        print(f"[run_all] basket_B = {basket_B}: {detail_B}", flush=True)
        if basket_B == "WORKS":
            state["final_default_line"] = "default = HGT + carrier soft-immunity (M=2)."
            state["next_line"] = "default = HGT + immunity; следующий крупный пакет = r vs B2/B3 на этом канале."
        else:
            state["final_default_line"] = "default = HGT only (без slot-safe, без immunity) -- ни один рычаг retention не сработал."
            state["next_line"] = ("удержание не берётся ни mutate-safe, ни immunity при pop=20/gate; "
                                 "следующий пакет -- только смена selection timescale (Pareto Imp-slots "
                                 "vs r ИЛИ ослабление gate), не атомы, не registry.")

    print("\n=== [5/5] report ===", flush=True)
    out_path = write_report(state)
    (METRICS_DIR / "final_state.json").write_text(
        json.dumps({k: v for k, v in state.items() if k not in ("cells_A",)}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"[run_all] отчёт записан: {out_path}", flush=True)
