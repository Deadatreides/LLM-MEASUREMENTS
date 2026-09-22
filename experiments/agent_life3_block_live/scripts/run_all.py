"""run_all.py — PROTOCOL.md: grid проверка -> baselines B0-B3 -> lean-assert
smoke -> 6 WITH + 6 CTRL -> агрегаты -> корзина UNIT_LIVE_* ->
REPORT_LIFE3_LIVE.md.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD          # noqa: E402
import live_grid_builder as GRID   # noqa: E402
import lean_seeds as LS            # noqa: E402
import metrics_lib as ML           # noqa: E402
import orchestrator as ORCH        # noqa: E402
import baselines as BASE           # noqa: E402
import call_log                    # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = AGENT_DIR / "metrics"
RUNS_DIR = AGENT_DIR / "runs"
REPORTS_DIR = AGENT_DIR / "reports"

SEEDS = ORCH.SEEDS
TRAIN_LOG_FLOOR = int(0.8 * 6 * 4 * 100)   # PROTOCOL.md §1


def _fmt(x, spec=".3f", default="н/д") -> str:
    return format(x, spec) if x is not None else default


def ensure_live_grid(ds) -> dict:
    all_ids = sorted(ds.tasks)
    cells = GRID.build_live_grid(all_ids, ds.tasks, ds.split)
    ds.reload_grid()
    train_ids = set(ds.split["train"])
    n_train_rows = sum(1 for (t, k, m) in cells if t in train_ids)
    n_rows_log_train = call_log.count_rows_for_tasks(train_ids)
    print(f"[run_all] live_grid: {len(cells)} ячеек всего, train-ячеек {n_train_rows}; "
          f"live_call_log train-строк {n_rows_log_train} (пол {TRAIN_LOG_FLOOR})", flush=True)
    return {"n_cells": len(cells), "n_train_cells": n_train_rows,
           "n_train_log_rows": n_rows_log_train, "floor": TRAIN_LOG_FLOOR,
           "floor_met": n_rows_log_train >= TRAIN_LOG_FLOOR}


def smoke_lean_assert(ds, panel: list, screen: list) -> None:
    best_single = ML.best_single_per_slot(ds, panel)
    seeds_pop = LS.build_seeds_pop20_lean(ds, panel, best_single)
    LS.assert_lean_population(seeds_pop, ds, panel, best_single)
    print("[smoke] lean-assert: OK (20 различных не-эталонных, все |ImpSet|<=1)", flush=True)


def run_all_cells(ds, panel: list, screen: list) -> dict:
    out = {"WITH": [], "CTRL": []}
    for mode in ("WITH", "CTRL"):
        for seed in SEEDS:
            t0 = time.time()
            out_dir = RUNS_DIR / f"life3_{mode.lower()}_s{seed}"
            summary = ORCH.run_cell(mode, seed, ds, panel, screen, out_dir)
            best_single = ML.best_single_per_slot(ds, panel)
            audit = ML.audit_cell(mode, seed, out_dir, ds, panel, best_single)
            dt = time.time() - t0
            audit["wall_seconds"] = dt
            audit["extinct"] = summary["extinct"]
            audit["n_generations"] = summary["n_generations"]
            audit["ref_test_metrics"] = summary.get("ref_test_metrics")
            cell_path = METRICS_DIR / "cells" / f"{mode.lower()}_s{seed}.json"
            cell_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cell_path, "w", encoding="utf-8") as f:
                json.dump(audit, f, ensure_ascii=False, indent=2, default=str)
            print(f"[run_all] {mode} seed={seed} done in {dt:.0f}s", flush=True)
            out[mode].append(audit)
    return out


def classify_basket(with_results: list, ctrl_results: list, grid_status: dict) -> dict:
    """PROTOCOL.md §2. Порядок проверки (не в задании явно, но нужен для
    детерминизма при пересечении определений корзин): UNIT_LIVE_BLOCKED ->
    UNIT_LIVE_WORKS -> UNIT_LIVE_FAIL -> иначе UNIT_LIVE_PARTIAL."""
    if not grid_status.get("floor_met"):
        return {"basket": "UNIT_LIVE_BLOCKED",
               "reason": f"live_call_log train floor не выполнен: "
                         f"{grid_status['n_train_log_rows']} < {grid_status['floor']}"}

    n_a1a5 = sum(1 for r in with_results if r.get("block_a1_a5") and r["block_a1_a5"]["all_A1_A5_pass"])
    fa_with = [r["N_AB_first_assembly"] for r in with_results]
    fa_ctrl = [r["N_AB_first_assembly"] for r in ctrl_results]
    lg_with = [r["loss_gain_overall"] for r in with_results]
    lg_ctrl = [r["loss_gain_overall"] for r in ctrl_results]
    mean_fa_with, mean_fa_ctrl = statistics.mean(fa_with), statistics.mean(fa_ctrl)
    mean_lg_with, mean_lg_ctrl = statistics.mean(lg_with), statistics.mean(lg_ctrl)

    # парное сравнение по seed (одинаковый номер seed в WITH и CTRL, PROTOCOL.md §2)
    with_by_seed = {r["seed"]: r["N_AB_first_assembly"] for r in with_results}
    ctrl_by_seed = {r["seed"]: r["N_AB_first_assembly"] for r in ctrl_results}
    signs = [with_by_seed[s] - ctrl_by_seed[s] for s in SEEDS if s in with_by_seed and s in ctrl_by_seed]
    n_negative = sum(1 for d in signs if d < 0)
    consistently_lower = n_negative >= 4   # большинство из 6

    slot_match_alive = any((r.get("slot_match_rate") or 0) >= 0.95 for r in with_results)
    transfer_exists = any((r.get("n_transfer_success") or 0) >= 1 for r in with_results)

    detail = {
        "n_A1_A5_pass_of_6": n_a1a5,
        "mean_N_AB_first_assembly_WITH": mean_fa_with, "mean_N_AB_first_assembly_CTRL": mean_fa_ctrl,
        "mean_loss_gain_overall_WITH": mean_lg_with, "mean_loss_gain_overall_CTRL": mean_lg_ctrl,
        "n_negative_of_6": n_negative, "channel_alive": bool(slot_match_alive and transfer_exists),
    }

    works = (n_a1a5 >= 5 and mean_fa_with >= mean_fa_ctrl
            and mean_lg_with <= mean_lg_ctrl + 0.5)
    if works:
        return {"basket": "UNIT_LIVE_WORKS", "reason": f"{n_a1a5}/6 A1-A5, "
               f"first_assembly {mean_fa_with:.2f}>={mean_fa_ctrl:.2f}, "
               f"loss:gain {mean_lg_with:.2f}<={mean_lg_ctrl:.2f}+0.5", **detail}

    fails = (n_a1a5 <= 2 or consistently_lower or mean_lg_with > mean_lg_ctrl + 0.5)
    if fails:
        reasons = []
        if n_a1a5 <= 2:
            reasons.append(f"A1-A5 держится только на {n_a1a5}/6")
        if consistently_lower:
            reasons.append(f"first_assembly устойчиво ниже control ({n_negative}/6 seed отрицательны)")
        if mean_lg_with > mean_lg_ctrl + 0.5:
            reasons.append(f"деградация loss:gain ({mean_lg_with:.2f} > {mean_lg_ctrl:.2f}+0.5)")
        return {"basket": "UNIT_LIVE_FAIL", "reason": "; ".join(reasons), **detail}

    return {"basket": "UNIT_LIVE_PARTIAL",
           "reason": f"{n_a1a5}/6 A1-A5 (не >=5) либо first_assembly не выше control, "
                     f"но канал жив (transfer_exists={transfer_exists}, slot_match~1={slot_match_alive})",
           **detail}


def write_report(with_results, ctrl_results, basket, grid_status, b0, panel) -> None:
    L = []
    add = L.append
    add("# REPORT_LIFE3_LIVE — единица блока + автокатализ + комплементарный HGT, LIVE")
    add("")
    add("Спека: `PROTOCOL.md` (что считается live, R1-R4, корзины `UNIT_LIVE_*` -- "
        "не двигались после чисел). Живая сетка построена ЭТИМ пакетом с нуля "
        "(новый сплит, GEN_SEED=150002/SPLIT_SEED=20260821), НЕ прочитана из "
        "`agent_a5_live_m`.")
    add("")

    add("## 1. Рамка")
    add("")
    add("Три проверяемых live-утверждения (PROTOCOL.md §0): (1) комплементарный "
        "slot-HGT (`p=0.35`, из LIFE-2) собирает joint coverage на свежем сплите; "
        "(2) регистрация блока + `BLOCK_INSERT` замыкает автокатализ A1-A5 (для "
        "БЛОКОВ, не целых комплексов -- отличие от LIFE-1's REPORT_HEREDITY); "
        "(3) `N_AB_first_assembly` — не артефакт старой сетки A5.")
    add("")

    add("## 2. Пол лога + покрытие сетки + смоук")
    add("")
    add(f"- `live_call_log.jsonl`: {call_log.count_rows()} строк всего.")
    add(f"- TRAIN-ячеек в сетке: {grid_status['n_train_cells']}; TRAIN-строк лога: "
        f"{grid_status['n_train_log_rows']} (пол {grid_status['floor']}, "
        f"{'ВЫПОЛНЕН' if grid_status['floor_met'] else 'НЕ ВЫПОЛНЕН'}).")
    add(f"- Всего ячеек сетки: {grid_status['n_cells']}.")
    add(f"- Панель: {len(panel)} задач (`metrics/train_panel.json`).")
    add(f"- B0 (лучшая одиночная модель, живой целостный вызов, решено данными): "
        f"{b0['model']} (train_rate={_fmt(b0['train_rate'])}, test_rate={_fmt(b0['test_rate'])}).")
    add("")

    add("## 3. Таблица 6 WITH + 6 CTRL")
    add("")
    add("| mode | seed | A1-A5 | N_blocks | N_AB_first_assembly | elite_AB_share | "
        "loss:gain overall | slot_match | extinct |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in with_results + ctrl_results:
        a1a5 = "н/д"
        n_blocks = "—"
        if r.get("block_a1_a5"):
            a1a5 = "PASS" if r["block_a1_a5"]["all_A1_A5_pass"] else str(r["block_a1_a5"]["passes"])
            n_blocks = r["block_a1_a5"]["N_blocks_registered"]
        add(f"| {r['mode']} | {r['seed']} | {a1a5} | {n_blocks} | {r['N_AB_first_assembly']} | "
            f"{_fmt(r['elite_AB_share'])} | {_fmt(r['loss_gain_overall'],'.2f')} | "
            f"{_fmt(r['slot_match_rate'])} | {r['extinct']} |")
    add("")

    add("## 4. Парное WITH-CTRL")
    add("")
    with_by_seed = {r["seed"]: r for r in with_results}
    ctrl_by_seed = {r["seed"]: r for r in ctrl_results}
    add("| seed | N_AB_first_assembly WITH | CTRL | Δ | loss:gain WITH | CTRL |")
    add("|---|---|---|---|---|---|")
    for s in SEEDS:
        w, c = with_by_seed.get(s), ctrl_by_seed.get(s)
        if not w or not c:
            continue
        d = w["N_AB_first_assembly"] - c["N_AB_first_assembly"]
        add(f"| {s} | {w['N_AB_first_assembly']} | {c['N_AB_first_assembly']} | {d:+d} | "
            f"{_fmt(w['loss_gain_overall'],'.2f')} | {_fmt(c['loss_gain_overall'],'.2f')} |")
    add("")

    add("## 5. Корзина")
    add("")
    add(f"**{basket['basket']}** — {basket['reason']}")
    add("")
    add(f"- A1-A5 держатся одновременно на {basket.get('n_A1_A5_pass_of_6','?')}/6 WITH-seed.")
    add(f"- mean N_AB_first_assembly: WITH={_fmt(basket.get('mean_N_AB_first_assembly_WITH'),'.2f')}, "
        f"CTRL={_fmt(basket.get('mean_N_AB_first_assembly_CTRL'),'.2f')}.")
    add(f"- mean loss:gain(overall): WITH={_fmt(basket.get('mean_loss_gain_overall_WITH'),'.2f')}, "
        f"CTRL={_fmt(basket.get('mean_loss_gain_overall_CTRL'),'.2f')}.")
    add("")

    add("## 6. Сравнение с LIFE-2")
    add("")
    if basket["basket"] == "UNIT_LIVE_WORKS":
        add("Гипотеза LIFE-2 (`p=0.35` комплементарного переноса собирает joint "
            "coverage) **подтвердилась на живых данных**: та же качественная "
            "картина (рост совместной сборки, канал структурно корректен) "
            "воспроизводится на свежем сплите/свежих ответах моделей, не только "
            "на CPU-replay старой сетки A5.")
    else:
        add(f"Формальный результат на живых данных — **{basket['basket']}**, не "
            "прямое подтверждение LIFE-2 гипотезы в заявленном виде. См. §5/§8 "
            "для интерпретации.")
    add("")

    add("## 7. Переносимое правило (без имён моделей)")
    add("")
    if basket["basket"] == "UNIT_LIVE_WORKS":
        add("> Единица (блок), зарегистрированная по правилу «использована в "
            "≥3 независимых линиях, устойчива ≥2 поколений», и вставляемая "
            "предпочтительно в комплементарные слоты, даёт замкнутый цикл "
            "автокатализа (регистрация → повторное использование → отбор → "
            "исполнение) на ЖИВЫХ данных, без потери способности собирать "
            "совместное покрытие независимых блоков относительно контроля без "
            "реестра.")
    else:
        add(f"> Формальный результат — **{basket['basket']}**; правило не "
            "подтверждено в заявленном виде на этом прогоне.")
    add("")

    add("## 8. Non-claims")
    add("")
    add("- Не заявляется превосходство над B3 без CI, исключающего равенство.")
    add("- Не заявляется перенос за пределы HETEROSTEP+6 моделей.")
    add("- `arch2/`, `agent_a5_live_m/`, `agent_life1_mechanism/`, "
        "`agent_life2_complementary/` не изменялись.")
    add("- R1-R4 (PROTOCOL.md §4.2) — одна конкретная операционализация "
        "«блока», не единственно возможная.")
    add("- `UNIT_LIVE_PARTIAL`/`UNIT_LIVE_FAIL`/`UNIT_LIVE_BLOCKED` — явно "
        "допустимый исход, не провал задания.")
    add("")

    add("## 9. Дальше")
    add("")
    if basket["basket"] == "UNIT_LIVE_WORKS":
        add("Зафиксировать связку (комплементарный HGT + регистрация блока + "
            "BLOCK_INSERT) как default для joint-coverage сборки; следующий шаг "
            "— перенос на новый пул атомов, не расширение алфавита ЭТИХ 6 моделей.")
    else:
        add("Один уточняющий рычаг (не новая сетка факторов) — см. §5 для того, "
            "какое именно условие корзины не выполнилось; либо `BLOCKERS.md`, "
            "если причина инфраструктурная (grid/regstry/seam).")
    add("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "REPORT_LIFE3_LIVE.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[run_all] {REPORTS_DIR / 'REPORT_LIFE3_LIVE.md'} записан", flush=True)


def main() -> int:
    t_start = time.time()
    ds = LD.default_dataset(fresh=True)
    split_path = LD.write_split_json(ds)
    print(f"[run_all] {split_path} записан", flush=True)

    print("[run_all] --- живая сетка ---", flush=True)
    grid_status = ensure_live_grid(ds)
    if not grid_status["floor_met"]:
        print("[run_all] !!! пол лога НЕ выполнен -- дальше идём, но корзина будет UNIT_LIVE_BLOCKED",
              flush=True)

    pools = ORCH.build_panel(ds)
    panel, screen = pools["panel"], pools["screen"]
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "train_panel.json", "w", encoding="utf-8") as f:
        json.dump({"panel": panel, "screen": screen}, f, ensure_ascii=False, indent=2)
    print(f"[run_all] панель: {len(panel)} задач, screen: {len(screen)} задач", flush=True)

    print("[run_all] --- смоук lean-assert ---", flush=True)
    smoke_lean_assert(ds, panel, screen)

    print("[run_all] --- baselines B0-B3 ---", flush=True)
    b0 = BASE.build_b0(ds)
    ds.reload_grid()

    print("[run_all] --- 6 WITH + 6 CTRL ---", flush=True)
    results = run_all_cells(ds, panel, screen)

    print("[run_all] --- корзина ---", flush=True)
    basket = classify_basket(results["WITH"], results["CTRL"], grid_status)
    print(f"[run_all] basket={basket['basket']} ({basket['reason']})", flush=True)

    summary = {"elapsed_seconds": time.time() - t_start, "grid_status": grid_status,
              "b0": b0, "basket": basket, "with": results["WITH"], "ctrl": results["CTRL"]}
    with open(METRICS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[run_all] {METRICS_DIR / 'summary.json'} записан", flush=True)

    write_report(results["WITH"], results["CTRL"], basket, grid_status, b0, panel)
    print(f"[run_all] ГОТОВО за {time.time()-t_start:.0f}с. Корзина: {basket['basket']}", flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
