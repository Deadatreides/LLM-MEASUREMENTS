"""run_all.py — PROTOCOL.md шаги 2-6, конец в конец. Требует, чтобы шаг 0-1
(`scripts/verify_seams.py`) уже прошёл (по конвенции проекта — не
проверяется здесь программно, чтобы не задваивать смоук-вызов).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD          # noqa: E402
import live_grid_builder as GRID   # noqa: E402
import baselines as BASE           # noqa: E402
import orchestrator as ORCH        # noqa: E402
import verdict as VERDICT          # noqa: E402
import call_log                    # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = AGENT_DIR / "runs"
METRICS_DIR = AGENT_DIR / "metrics"
REPORTS_DIR = AGENT_DIR / "reports"

SEEDS = [20261201, 20261202, 20261203, 20261204, 20261205]


def _fmt(x, spec=".4f", default="н/д") -> str:
    return format(x, spec) if x is not None else default


def _fmt_ci(ci) -> str:
    if ci is None:
        return "н/д"
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def ensure_live_grid(ds) -> dict:
    all_ids = sorted(ds.tasks)
    cells = GRID.build_live_grid(all_ids, ds.tasks, ds.split)
    ds.reload_grid()
    target = len(all_ids) * 4 * 6
    floor = 6 * 4 * 120
    n_rows = len(cells)
    print(f"[run_all] live_grid: {n_rows}/{target} ячеек (минимум {floor})", flush=True)
    if n_rows < floor:
        raise RuntimeError(f"live_grid ниже приёмочного минимума: {n_rows} < {floor} -- см. BLOCKERS.md")
    return {"n_rows": n_rows, "target": target, "floor": floor}


def main() -> int:
    t_start = time.time()

    ds = LD.default_dataset(fresh=True)
    split_path = LD.write_split_json(ds)
    print(f"[run_all] {split_path} записан (train={len(ds.split['train'])}, test={len(ds.split['test'])})",
          flush=True)
    grid_status = ensure_live_grid(ds)

    print("[run_all] --- шаг 3: baselines B0-B3 ---", flush=True)
    b0 = BASE.build_b0(ds)
    BASE.build_b1_b2_b3(ds, ds.split["train"])
    ds.reload_grid()

    print("[run_all] --- шаг 4-5: 5x эволюция + финальная оценка ---", flush=True)
    per_seed = []
    for seed in SEEDS:
        print(f"[run_all] seed {seed} -- старт", flush=True)
        t0 = time.time()
        res = ORCH.run_seed_campaign(seed, ds, RUNS_DIR)
        dt = time.time() - t0
        res["wall_seconds"] = dt
        per_seed.append(res)
        per_seed_path = METRICS_DIR / f"per_seed_{seed}.json"
        with open(per_seed_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2, default=str)
        print(f"[run_all] seed {seed} -- готово за {dt:.0f}с, "
              f"extinct={res['extinct']}, archive_excl_refs={res['archive_size_excl_refs']}, "
              f"gens={res['n_generations_run']}", flush=True)

    print("[run_all] --- шаг 6: вердикт + отчёт ---", flush=True)
    basket = VERDICT.compute_basket(per_seed)

    summary = {
        "elapsed_seconds": time.time() - t_start,
        "live_grid": grid_status,
        "b0": b0,
        "seeds": SEEDS,
        "per_seed": per_seed,
        "basket": basket,
        "live_call_log_rows": call_log.count_rows(),
    }
    summary_path = METRICS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[run_all] {summary_path} записан", flush=True)

    write_report(summary)
    print(f"[run_all] ГОТОВО за {time.time() - t_start:.0f}с. Вердикт: {basket['basket']}", flush=True)
    return 0


def write_report(summary: dict) -> None:
    basket = summary["basket"]
    per_seed = summary["per_seed"]
    b0 = summary["b0"]

    L = []
    add = L.append
    add("# REPORT_LA5.md — agent_a5_live_m")
    add("")
    add(f"Суммарное время: {summary['elapsed_seconds']:.0f}с. "
        f"Строк live_call_log.jsonl: {summary['live_call_log_rows']}.")
    add("")

    add("## 1. Изоляция, сплит, панель, seeds, швы")
    add("")
    add(f"- live_grid: {summary['live_grid']['n_rows']}/{summary['live_grid']['target']} "
        f"ячеек (приёмочный минимум {summary['live_grid']['floor']}).")
    add(f"- Сплит: GEN_SEED={LD.GEN_SEED}, SPLIT_SEED={LD.SPLIT_SEED}, "
        f"LIVE_SEED_BASE={LD.LIVE_SEED_BASE}.")
    add(f"- 5 evolutionary seeds: {summary['seeds']}.")
    if per_seed:
        add(f"- Панель (control): {per_seed[0]['panel_size']} задач, "
            f"screen: {per_seed[0]['screen_size']} задач — обе фиксированы на весь "
            "прогон (PROTOCOL.md §6), не ротируют между поколениями.")
    add(f"- B0 (лучшая одиночная модель на целостном плече, решено данными): "
        f"{b0['model']} (train_rate={_fmt(b0['train_rate'])}, test_rate={_fmt(b0['test_rate'])}).")
    add("- Швы и смоук: `scripts/verify_seams.py` прогнан до шага 2 (PROTOCOL.md §1).")
    add("")

    add("## 2. Train: динамика по seed")
    add("")
    add("| seed | поколений | extinct | archive (искл. эталоны) | effective_p_cross |")
    add("|---|---|---|---|---|")
    for res in per_seed:
        add(f"| {res['seed']} | {res['n_generations_run']} | {res['extinct']} | "
            f"{res['archive_size_excl_refs']} | {_fmt(res['effective_p_cross'], '.2f')} |")
    add("")

    add("## 3. Test: complex vs B1/B2/B3 (r, c, ΔR, E, CI)")
    add("")
    for res in per_seed:
        add(f"### seed {res['seed']}")
        add("")
        add(f"B1: r={_fmt(res['b1_r_test'])} c={_fmt(res['b1_c_test'], '.1f')} | "
            f"B2: r={_fmt(res['b2_r_test'])} c={_fmt(res['b2_c_test'], '.1f')} | "
            f"B3: r={_fmt(res['b3_r_test'])} c={_fmt(res['b3_c_test'], '.1f')}")
        add("")
        add("| top-N complex_id | r_test | c_test | ΔR vs B1 (CI95) | ΔR vs B3 (CI95) | "
            "E vs B1 (CI95) | r_train archive | r_train snapshot |")
        add("|---|---|---|---|---|---|---|---|")
        for t in res["top3"]:
            e = t["efficiency_vs_b1"]
            e_str = f"{_fmt(e.get('point'), '.3f')} {_fmt_ci(e.get('ci_95'))}"
            add(f"| {t['complex_id']} | {_fmt(t['r_test'])} | {_fmt(t['c_test'], '.1f')} | "
                f"{t['delta_r_vs_b1']['point']:+.4f} {_fmt_ci(t['delta_r_vs_b1']['ci_95'])} | "
                f"{t['delta_r_vs_b3']['point']:+.4f} {_fmt_ci(t['delta_r_vs_b3']['ci_95'])} | "
                f"{e_str} | {_fmt(t['r_train_archive'])} | {_fmt(t['r_train_snapshot'])} |")
        add("")

    add("## 4. Вердикт-корзина (PROTOCOL.md §8)")
    add("")
    add(f"**{basket['basket']}**")
    add("")
    add(f"- ΔR vs B1 > 0 И CI95 нижняя ≥ 0: {basket['n_wins_b1_of_5']}/5 seed.")
    add(f"- r(complex) ≥ r(B2) − 0.02: {basket['n_b2_ok_of_5']}/5 seed.")
    add(f"- ΔR vs B1 > 0, но CI пересекает 0 (partial): {basket['n_partial_b1_of_5']}/5 seed.")
    add("")
    add("| seed | best_complex | ΔR vs B1 | CI95 | win | partial | b2_ok |")
    add("|---|---|---|---|---|---|---|")
    for row in basket["seed_rows"]:
        if row["best_complex"] is None:
            add(f"| {row['seed']} | (архив пуст) | — | — | — | — | — |")
        else:
            add(f"| {row['seed']} | {row['best_complex']} | "
                f"{row['delta_r_vs_b1_point']:+.4f} | {_fmt_ci(row['delta_r_vs_b1_ci95'])} | "
                f"{row['win_b1']} | {row['partial_b1']} | {row['b2_ok']} |")
    add("")
    add(f"**Обязательная строка про B3**: mean ΔR vs B3 по 5 seed = "
        f"{_fmt(basket['mean_delta_r_vs_b3'], '+.4f')}.")
    claim = "ДОПУСТИМ" if basket["beat_b3_claim_allowed"] else "НЕ ДОПУСТИМ"
    add(f"Claim «победили B3» {claim} (допустим только если CI95 всех 5 seed целиком "
        "исключает ноль в плюс).")
    add("")

    add("## 5. Snapshot vs archive на TRAIN (закрытие находки A3)")
    add("")
    add("| seed | complex_id | r_train archive (full) | r_train snapshot (панель, посл. поколение) |")
    add("|---|---|---|---|")
    for res in per_seed:
        for t in res["top3"]:
            add(f"| {res['seed']} | {t['complex_id']} | {_fmt(t['r_train_archive'])} | "
                f"{_fmt(t['r_train_snapshot'])} |")
    add("")

    add("## 6. Не заявляется (Non-claims)")
    add("")
    add("- Превосходство над B3 не заявляется, если CI включает 0 (см. §4).")
    add("- Перенос за пределы этого конкретного сплита/seed'ов не заявляется.")
    add("- `arch2/` и соседние `agent_aN` пакеты не изменялись (изоляция).")
    add("- Расширение алфавита (DECOMPOSE, R1/R2, новые модели) вне области этого пакета.")
    add("- LIVE_NULL — явно допустимый, не проваленный исход (PROTOCOL.md §8/приёмка).")
    add("")

    add("## 7. Стоимость")
    add("")
    add(f"- Строк live_call_log.jsonl всего: {summary['live_call_log_rows']} "
        "(живая сетка шаг 2 + B0-калибровка шаг 3; шаги 4-5 — 0 новых вызовов, "
        "чистый replay поверх live_grid, как и ожидалось по PROTOCOL.md §1).")
    add(f"- Общее время (сборка сетки + baselines + 5 seed + отчёт): "
        f"{summary['elapsed_seconds']:.0f}с.")
    add("")

    add("## 8. Следующий шаг")
    add("")
    if basket["basket"] == "LIVE_NULL":
        add("Живая проверка не нашла устойчивого превосходства над B1 на held-out — "
            "следующий пакет должен разбираться с отбором на full-grid внутри цикла "
            "эволюции (не только на финальной оценке) или с явным оператором слот→слот, "
            "прежде чем расширять алфавит.")
    else:
        add("Результат воспроизводится живыми вызовами на независимом сплите — следующий "
            "шаг: held-out перенос на ДРУГОЙ независимый сплит/генератор задач, прежде чем "
            "заявлять общий результат.")
    add("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "REPORT_LA5.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[run_all] {report_path} записан", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
