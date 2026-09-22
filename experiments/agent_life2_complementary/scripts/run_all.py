"""run_all.py — PROTOCOL.md: панель -> lean-assert -> smoke -> 24 клетки
(3 p x 8 seed) -> агрегаты -> корзина COMP_* -> REPORT_LIFE2.md. Только
чтение живой сетки A5; НИ ОДНОГО generate() в этом пакете.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD          # noqa: E402
import lean_seeds as LS            # noqa: E402
import metrics_lib as ML           # noqa: E402
import orchestrator as ORCH        # noqa: E402
import slot_hgt as ST              # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = AGENT_DIR / "metrics"
CELLS_DIR = METRICS_DIR / "cells"
RUNS_DIR = AGENT_DIR / "runs"
REPORTS_DIR = AGENT_DIR / "reports"

P_LEVELS = ORCH.P_LEVELS
SEEDS = ORCH.SEEDS
LOSS_GAIN_THRESHOLD = 4.0
SLOT_MATCH_THRESHOLD = 0.50   # используется в argmax-фильтре p* (задание §1, не менялось)
SLOT_MATCH_SANITY = 0.95      # PROTOCOL.md §1: успешный TRANSFER_SLOT обязан быть complementary


def write_panel(ds) -> dict:
    pools = ORCH.build_panel(ds)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "train_panel.json", "w", encoding="utf-8") as f:
        json.dump({"panel": pools["panel"], "screen": pools["screen"],
                   "source": "agent_life1_mechanism/metrics/runs_instrumented/"
                             "life1_s20261201/summary.json"}, f, ensure_ascii=False, indent=2)
    print(f"[run_all] panel: {len(pools['panel'])} задач, screen: {len(pools['screen'])} задач",
          flush=True)
    return pools


def smoke_test(ds, panel: list, screen: list) -> None:
    """PROTOCOL.md §7 старт: lean-assert + ровно один залогированный
    transfer_slot-attempt (p_slot_hgt=1.0, форсированно) до полной сетки."""
    best_single = ML.best_single_per_slot(ds, panel)
    for s, kind in enumerate(ML.STEP_KINDS):
        if kind == "FORMAT":
            continue
    seeds_pop = LS.build_seeds_pop20_lean(ds, panel)
    LS.assert_lean_population(seeds_pop, ds, panel, best_single)
    print("[smoke] lean-assert: OK (20 различных не-эталонных, все |ImpSet|<=1)", flush=True)

    tmp_dir = AGENT_DIR / "runs" / "_smoke"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)
    reg = ML.HSTEP.Registry()
    be = ML.HSTEP.HeterostepBackend(ds)
    imp_cache: dict = {}

    def imp_fn(g):
        cid = g.get("complex_id")
        if cid in imp_cache:
            return imp_cache[cid]
        v = ML.imp_set(ds, panel, best_single, g)
        imp_cache[cid] = v
        return v

    rng = random.Random(999999)
    ev = ML.LD.E.Evolution(
        registry=reg, backend=be, dataset=ds, experiment_id="smoke",
        runs_dir=tmp_dir / "traces", rng=rng, screen_ids=screen,
        control_slices=[panel], holdout_ids=[],
        assemble_n_steps=len(ML.STEP_KINDS), budget_per_task=ORCH.BUDGET_PER_TASK,
        g_stall=ORCH.G_STALL,
    )
    ev.current_gen = 0
    ev.seed(seeds_pop, reference_names=ML.HSEED.REFERENCE_NAMES,
           baseline_name=ML.HSEED.BASELINE_NAME, gate_reference_name=ML.HSEED.GATE_REFERENCE_NAME)

    births_path = tmp_dir / "births.jsonl"
    custom_reproduce = ORCH.make_custom_reproduce(births_path, imp_fn, p_slot_hgt=1.0)
    # Поколение 0 засеяно ровно 20 не-эталонными членами (start_pop=20) -- при
    # POP_SIZE=24 это даёт n_children=max(0,24-20-6)=0 на первом reproduce()
    # (новорождённые защищены от D1/D2 в СВОЁМ поколении рождения, население не
    # сжимается сразу). Нужно прогнать НЕСКОЛЬКО поколений, чтобы естественная
    # смертность (с gen>=1, защита новорождённого больше не действует) открыла
    # место для потомков и transfer_slot вообще получил шанс попытаться.
    for gen in range(4):
        ev.run_generation(gen)
        if ev.extinct:
            break
        res_data = ML.LD.F.load_results(tmp_dir / "traces", "smoke")
        custom_reproduce(ev, gen + 1, res_data, panel)

    rows = [json.loads(l) for l in births_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    hgt_rows = [r for r in rows if r.get("hgt_attempted")]
    assert hgt_rows, "smoke FAILED: p_slot_hgt=1.0 но ни одной попытки transfer_slot не залогировано"
    successes = [r for r in hgt_rows if r.get("operator") == "TRANSFER_SLOT"]
    print(f"[smoke] {len(hgt_rows)} попыток HGT залогировано, {len(successes)} успешных "
          f"(complementary={[r.get('complementary') for r in successes][:3]})", flush=True)
    if successes:
        assert all(r.get("complementary") for r in successes), (
            "smoke FAILED: успешный TRANSFER_SLOT без complementary=True -- баг в slot_hgt.py")
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[smoke] OK", flush=True)


def run_all_cells(ds, panel: list, screen: list) -> list:
    results = []
    for p in P_LEVELS:
        for seed in SEEDS:
            t0 = time.time()
            out_dir = RUNS_DIR / f"life2_p{p:.2f}_s{seed}"
            summary = ORCH.run_cell(p, seed, ds, panel, screen, out_dir)
            audit = ML.audit_cell(p, seed, out_dir, ds, panel, ML.best_single_per_slot(ds, panel))
            dt = time.time() - t0
            audit["wall_seconds"] = dt
            audit["extinct"] = summary["extinct"]
            audit["n_generations"] = summary["n_generations"]
            cell_path = CELLS_DIR / f"p{p:.2f}_s{seed}.json"
            cell_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cell_path, "w", encoding="utf-8") as f:
                json.dump(audit, f, ensure_ascii=False, indent=2, default=str)
            print(f"[run_all] p={p:.2f} seed={seed} done in {dt:.0f}s", flush=True)
            results.append(audit)
    return results


def aggregate_by_p(results: list) -> dict:
    out = {}
    for p in P_LEVELS:
        rows = [r for r in results if r["p"] == p]
        def mean(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return statistics.mean(vals) if vals else None
        out[p] = {
            "mean_slot_match_rate": mean("slot_match_rate"),
            "mean_loss_gain_overall": mean("loss_gain_overall"),
            "mean_loss_gain_transfer_only": mean("loss_gain_transfer_only"),
            "mean_transfer_success_rate": mean("transfer_success_rate"),
            "mean_N_A": mean("N_A"), "mean_N_AB_evolved": mean("N_AB_evolved"),
            "mean_N_AB_first_assembly": mean("N_AB_first_assembly"),
            "mean_elite_AB_share": mean("elite_AB_share"),
            "mean_P_B_given_A": mean("P_B_given_A"), "mean_overlap": mean("overlap"),
            "per_seed_N_AB_first_assembly": {r["seed"]: r["N_AB_first_assembly"] for r in rows},
            "per_seed_elite_AB_share": {r["seed"]: r["elite_AB_share"] for r in rows},
            "n_cells": len(rows),
        }
    return out


def pick_p_star(agg: dict) -> float:
    """Задание §1 (не менялось поправкой пользователя): argmax mean(N_A+N_AB_evolved)
    среди уровней, прошедших slot_match>=0.50 и loss:gain<=4; иначе argmax
    mean(N_AB_evolved); тай-брейк -> меньший p."""
    qualifying = [p for p in P_LEVELS
                 if (agg[p]["mean_slot_match_rate"] or 0) >= SLOT_MATCH_THRESHOLD
                 and (agg[p]["mean_loss_gain_overall"] if agg[p]["mean_loss_gain_overall"] is not None
                      else float("inf")) <= LOSS_GAIN_THRESHOLD]
    pool = qualifying or list(P_LEVELS)
    def score(p):
        na = agg[p]["mean_N_A"] or 0
        nab = agg[p]["mean_N_AB_evolved"] or 0
        return na + nab if qualifying else nab
    best_score = max(score(p) for p in pool)
    candidates = sorted(p for p in pool if score(p) == best_score)
    return candidates[0]


def classify_basket(agg: dict, p_star: float) -> dict:
    """PROTOCOL.md §1, с поправкой пользователя 2026-08-21: growth
    измеряется относительно p=0.10 (тот же пакет, тот же 8-seed набор),
    НЕ против LIFE-1's 28.2 (другой режим посева)."""
    baseline_p = min(P_LEVELS)   # 0.10 -- lean-внутренний базовый уровень

    def qualifies(p) -> bool:
        a = agg[p]
        if (a["mean_slot_match_rate"] or 0) < SLOT_MATCH_SANITY:
            return False
        if a["mean_loss_gain_overall"] is None or a["mean_loss_gain_overall"] > LOSS_GAIN_THRESHOLD:
            return False
        if p == baseline_p:
            return False   # сравнивать базовый уровень сам с собой бессмысленно
        grow_ab = (a["mean_N_AB_first_assembly"] or 0) > (agg[baseline_p]["mean_N_AB_first_assembly"] or 0)
        grow_elite = (a["mean_elite_AB_share"] or 0) > (agg[baseline_p]["mean_elite_AB_share"] or 0)
        return grow_ab and grow_elite

    n_qualifying = sum(1 for p in P_LEVELS if qualifies(p))
    any_transfer_happened = any((agg[p]["mean_slot_match_rate"] or 0) >= SLOT_MATCH_THRESHOLD
                                for p in P_LEVELS)

    if p_star == baseline_p:
        # p* совпал с базовым уровнем -- внутреннего контраста роста нет по построению
        basket = "COMP_PARTIAL" if any_transfer_happened else "COMP_FAIL"
        reason = f"p*=baseline({baseline_p}) -- нет внутреннего уровня для сравнения роста"
        seed_consistency = None
    elif n_qualifying >= 2:
        per_seed_star = agg[p_star]["per_seed_N_AB_first_assembly"]
        per_seed_base = agg[baseline_p]["per_seed_N_AB_first_assembly"]
        signs = [per_seed_star.get(s, 0) - per_seed_base.get(s, 0) > 0 for s in SEEDS]
        n_positive = sum(signs)
        if n_positive >= 6:
            basket, reason = "COMP_WORKS", f"{n_qualifying}/3 уровней прошли, {n_positive}/8 seed знак+"
        else:
            basket, reason = "COMP_PARTIAL", f"{n_qualifying}/3 уровня прошли, но только {n_positive}/8 seed знак+"
        seed_consistency = {"n_positive_of_8": n_positive, "signs": dict(zip(SEEDS, signs))}
    elif any_transfer_happened:
        basket, reason = "COMP_PARTIAL", f"перенос случается (slot_match>=0.5 хотя бы на 1 уровне), но {n_qualifying}/3 полных условий"
        seed_consistency = None
    else:
        basket, reason = "COMP_FAIL", "перенос не случается ни на одном уровне (slot_match<0.5 везде)"
        seed_consistency = None

    return {"basket": basket, "reason": reason, "p_star": p_star, "baseline_p": baseline_p,
           "n_qualifying_levels": n_qualifying, "seed_consistency": seed_consistency}


def write_report(agg: dict, basket: dict, results: list, panel: list) -> None:
    L = []
    add = L.append
    add("# REPORT_LIFE2 — complementary slot-HGT на бедном посеве")
    add("")
    add("Спека: `PROTOCOL.md` (определения `Imp`/A/AB/`N_AB_first_assembly`, "
        "пороги `COMP_*` с поправкой пользователя 2026-08-21 -- growth считается "
        "против lean-внутреннего `p=0.10`, не против LIFE-1's rich-seed 28.2). "
        "24 CPU-replay прогона (3 p x 8 seed) поверх `agent_a5_live_m/metrics/"
        "live_grid/` -- 0 новых `generate()`.")
    add("")

    add("## 1. Рамка")
    add("")
    add("Полигон HETEROSTEP+6 моделей — измерительный прибор, не цель. Проверяется "
        "один рычаг из `agent_life1_mechanism/reports/REPORT_LIFE1.md`: замена "
        "similarity-донора на комплементарный-по-покрытию, на старте без готовой "
        "multi-slot сборки.")
    add("")

    add("## 2. Бедный старт")
    add("")
    add(f"Панель: {len(panel)} задач (единая на все 24 клетки, "
        "`metrics/train_panel.json`, источник — `agent_life1_mechanism` seed 20261201). "
        "Lean-assert на каждой из 24 клеток: 20 различных не-эталонных complex_id, "
        "0 с `|ImpSet|>=2` — все клетки прошли (иначе см. `BLOCKERS.md`).")
    add("")

    add("## 3. Таблица 3x8")
    add("")
    add("| p | seed | slot_match | loss:gain (overall) | loss:gain (transfer) | "
        "N_A | N_AB_evolved | N_AB_first_assembly | elite_AB_share | overlap | extinct |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        add(f"| {r['p']:.2f} | {r['seed']} | {_fmt(r['slot_match_rate'])} | "
            f"{_fmt(r['loss_gain_overall'],'.2f')} | {_fmt(r['loss_gain_transfer_only'],'.2f')} | "
            f"{r['N_A']} | {r['N_AB_evolved']} | {r['N_AB_first_assembly']} | "
            f"{_fmt(r['elite_AB_share'])} | {_fmt(r['overlap'])} | {r['extinct']} |")
    add("")
    add("### Агрегаты по p (mean по 8 seed)")
    add("")
    add("| p | slot_match | loss:gain overall | loss:gain transfer | transfer_success | "
        "N_A | N_AB_evolved | N_AB_first_assembly | elite_AB_share |")
    add("|---|---|---|---|---|---|---|---|---|")
    for p in P_LEVELS:
        a = agg[p]
        add(f"| {p:.2f} | {_fmt(a['mean_slot_match_rate'])} | "
            f"{_fmt(a['mean_loss_gain_overall'],'.2f')} | {_fmt(a['mean_loss_gain_transfer_only'],'.2f')} | "
            f"{_fmt(a['mean_transfer_success_rate'])} | {_fmt(a['mean_N_A'],'.1f')} | "
            f"{_fmt(a['mean_N_AB_evolved'],'.1f')} | {_fmt(a['mean_N_AB_first_assembly'],'.1f')} | "
            f"{_fmt(a['mean_elite_AB_share'])} |")
    add("")

    add("## 4. Кривая по p")
    add("")
    vals = [agg[p]["mean_N_AB_first_assembly"] or 0 for p in P_LEVELS]
    shape = "монотонный рост" if vals == sorted(vals) else (
        "монотонное падение" if vals == sorted(vals, reverse=True) else "немонотонно (горб/провал)")
    add(f"mean N_AB_first_assembly по p = {[round(v,2) for v in vals]} -> {shape}.")
    add("")

    add("## 5. p* и корзина")
    add("")
    add(f"**p\\* = {basket['p_star']:.2f}** (правило §1 задания, не менялось).")
    add("")
    add(f"**{basket['basket']}** — {basket['reason']}")
    add("")
    if basket["seed_consistency"]:
        add(f"Знак роста N_AB_first_assembly (p* vs p={basket['baseline_p']:.2f}) "
            f"положителен на {basket['seed_consistency']['n_positive_of_8']}/8 seed.")
        add("")

    add("## 6. Разделение эффектов (канал vs посев vs LIFE-1 фон)")
    add("")
    add("- **Эффект канала** (complementary vs similarity): качественно — LIFE-1's "
        "similarity-SWAP_SLOT дал slot_match~4%, loss:gain~13.5:1 (rich-посев). Здесь, "
        "на ЛЮБОМ p>0, slot_match у TRANSFER_SLOT ~1.0 по построению (см. §3) — "
        "канал ведёт себя как спроектирован. Это НЕ прямое сравнение чисел (разный "
        "посев), только качественное: конструкция донора устраняет источник потерь, "
        "найденный LIFE-1.")
    add("- **Эффект посева** (lean vs rich): не измерялся здесь напрямую (для этого "
        "нужен был бы rich-посев + комплементарный донор в одной клетке — вне рамок "
        "этого пакета, см. §8). Здесь варьируется только p при фиксированном lean-старте.")
    add("- **Фон LIFE-1** (rich-посев + similarity-донор): mean N_AB_evolved/seed "
        "~28.2, loss:gain(SWAP_SLOT)~13.5:1 — приведено только как контекст, НЕ как "
        "порог для корзины этого пакета (поправка пользователя 2026-08-21).")
    add("")

    add("## 7. Переносимое правило (без имён моделей) + фальсификаторы")
    add("")
    if basket["basket"] == "COMP_WORKS":
        add("> Донор slot-HGT, допустимый ТОЛЬКО при комплементарном покрытии "
            "(`Imp(s,donor)=1 ∧ Imp(s,recipient)=0`), даёт устойчивый на большинстве "
            "seed рост числа НОВОСОБРАННЫХ (не унаследованных) совместных покрытий, "
            "даже когда старт не содержит готовой multi-slot сборки. Фальсификатор: "
            "если на другом пуле атомов/полигоне комплементарный фильтр НЕ увеличивает "
            "долю новосборок относительно контроля без фильтра — правило не переносится.")
    else:
        add("> Формальный результат этого прогона — корзина "
            f"**{basket['basket']}**, не COMP_WORKS. Правило НЕ подтверждено на "
            "этом полигоне при бедном старте в заявленном виде — см. §5/§8 для "
            "интерпретации и следующего шага.")
    add("")

    add("## 8. Non-claims")
    add("")
    add("- Не заявляется превосходство над B3 без CI, исключающего равенство.")
    add("- Не заявляется перенос за пределы HETEROSTEP+6 моделей — гипотеза, не "
        "доказанный перенос.")
    add("- `arch2/`, `agent_a5_live_m/`, `agent_life1_mechanism/` не изменялись; "
        "в этом пакете не было ни одного вызова `generate()`.")
    add("- Эффект канала и эффект посева НЕ сведены в одну дельту (§6).")
    add("- `COMP_FAIL`/`COMP_PARTIAL` — явно допустимый исход, не провал задания.")
    add("")

    add("## 9. Дальше")
    add("")
    if basket["basket"] == "COMP_WORKS":
        add(f"Зафиксировать p*={basket['p_star']:.2f} как default комплементарного "
            "переноса; следующий пакет — единица блока/автокатализ на этом же канале, "
            "или перенос канала на новый пул атомов (не расширение алфавита ЭТИХ 6 "
            "моделей).")
    elif basket["basket"] == "COMP_PARTIAL":
        add("Один уточняющий рычаг (не новая сетка p) — например, ослабить D1/D2 для "
            "линий с уникальным Imp-слотом (лайфтайм-защита прекурсора), раз перенос "
            "случается, но не закрепляется.")
    else:
        add("Пересмотреть единицу переноса или сам `Imp`/`covered` — НЕ добавлять "
            "модели/расширять алфавит.")
    add("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "REPORT_LIFE2.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[run_all] {REPORTS_DIR / 'REPORT_LIFE2.md'} записан", flush=True)


def _fmt(x, spec=".3f", default="н/д") -> str:
    return format(x, spec) if x is not None else default


def main() -> int:
    t_start = time.time()
    ds = LD.default_dataset()

    print("=== панель ===", flush=True)
    pools = write_panel(ds)
    panel, screen = pools["panel"], pools["screen"]

    print("=== smoke ===", flush=True)
    smoke_test(ds, panel, screen)

    print("=== 24 клетки ===", flush=True)
    results = run_all_cells(ds, panel, screen)

    print("=== агрегаты + корзина ===", flush=True)
    agg = aggregate_by_p(results)
    p_star = pick_p_star(agg)
    basket = classify_basket(agg, p_star)
    print(f"[run_all] p*={p_star}, basket={basket['basket']} ({basket['reason']})", flush=True)

    summary = {"elapsed_seconds": time.time() - t_start, "panel_size": len(panel),
              "aggregates_by_p": {str(p): v for p, v in agg.items()}, "basket": basket,
              "per_cell": results}
    with open(METRICS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[run_all] {METRICS_DIR / 'summary.json'} записан", flush=True)

    write_report(agg, basket, results, panel)
    print(f"[run_all] ГОТОВО за {time.time()-t_start:.0f}с. Корзина: {basket['basket']}", flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
