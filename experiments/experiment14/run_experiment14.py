"""run_experiment14.py — CLI: --phase gen|pilot|train|g0|test|analyze.

Чекпоинты атомарные (tmp + replace), кампания резюмируема — та же схема, что
run_experiment12.py / run_experiment13.py.

Порядок с точками остановки (TASK_EXPERIMENT14.md §6):
  gen    -> задачи и швы, 0 GPU
  pilot  -> 20 задач, плечи A и B; проверка G4 и пригодности полигона
  train  -> сетка «6 моделей x 3 шага» на train-половине
  g0     -> гейт полигона: есть ли специализация вообще. ОСТАНОВ, если нет
  test   -> четыре плеча на test-половине
  analyze-> G1-G4 с CI
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import harness as H                                    # noqa: E402
import seams                                           # noqa: E402
from configs.model_registry import MODEL_IDS, load_model  # noqa: E402
from tasks.heterostep import (N_TASKS, STEP_KINDS,     # noqa: E402
                              build_tasks, split)

RUNS = ROOT / "runs14"
METRICS = ROOT / "metrics14"
N_BOOT = 10000
PILOT_N = 20
EASY_THRESHOLD = 0.90       # PASS(A) выше -> полигон слишком лёгкий
HARD_THRESHOLD = 0.05       # PASS(A) ниже -> слишком тяжёлый


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def _read(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tasks_and_split():
    tasks = build_tasks()
    sp = split(list(tasks))
    return tasks, sp


# -- bootstrap (тот же метод, что во всех кампаниях проекта) --------------------------------


def paired_diff(a: list, b: list, n_boot: int = N_BOOT, seed: int = 20260819) -> dict:
    """a, b — парные булевы списки по задачам. -> точка, CI95, знаковый тест."""
    import math
    import random

    n = len(a)
    if n == 0:
        return {"point": 0.0, "ci_95": [0.0, 0.0], "n": 0, "p_value": 1.0}
    point = sum(a) / n - sum(b) / n
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    m = n10 + n01
    if m == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        p = min(1.0, sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m * 2)
    return {"point": point, "ci_95": [diffs[int(0.025 * n_boot)],
                                      diffs[min(int(0.975 * n_boot), n_boot - 1)]],
            "n": n, "p_value": p, "a_wins": n10, "b_wins": n01}


# -- фазы ------------------------------------------------------------------------------------


def cmd_gen(args) -> int:
    rep = seams.self_test()
    print(f"контрольные случаи швов: {rep['n_cases']} — "
          f"{'ВСЕ ПРОЙДЕНЫ' if rep['passed'] else 'ЕСТЬ ПРОВАЛЫ'}")
    for f in rep["failures"]:
        print("  -", f)
    if not rep["passed"]:
        print("швы не прошли контрольные случаи — генерация задач не запускается")
        return 1

    tasks, sp = _tasks_and_split()
    _write(RUNS / "tasks.json", {"experiment_id": H.EXPERIMENT_ID, "n": len(tasks),
                                 "split": sp, "tasks": tasks})
    kinds = [s["kind"] for s in next(iter(tasks.values()))["steps"]]
    print(f"задач: {len(tasks)} (train {len(sp['train'])}, test {len(sp['test'])}), "
          f"роды шагов: {kinds}")
    print(f"сохранено -> {RUNS / 'tasks.json'}")
    return 0


def cmd_pilot(args) -> int:
    """20 задач, плечи A и B, все модели. Проверяет G4 и пригодность полигона."""
    tasks, sp = _tasks_and_split()
    ids = sp["train"][:PILOT_N]
    rows = []
    for mid in MODEL_IDS:
        llm, load_t = load_model(mid)
        print(f"[pilot] {mid} за {load_t:.1f}с", flush=True)
        for tid in ids:
            rows.append(H.run_whole(tasks[tid], mid, llm))
            rows.append(H.run_steps(tasks[tid], [mid] * len(tasks[tid]["steps"]), {mid: llm},
                                    "B_STEP_SAME", use_oracle_upstream=True))
        del llm
        _write(RUNS / "pilot.json", {"experiment_id": H.EXPERIMENT_ID, "rows": rows})

    print("\n=== ПИЛОТ: плечо A против B, по моделям ===")
    print("  модель                              A:PASS  A:INAPPL  B:PASS  B:INAPPL")
    best_a = 0.0
    for mid in MODEL_IDS:
        a = [r for r in rows if r["model"] == mid and r["arm"] == "A_WHOLE"]
        b = [r for r in rows if r["arm"] == "B_STEP_SAME" and r["model_by_step"][0] == mid]
        ap = sum(r["resolved"] for r in a) / len(a)
        ai = sum(r["final_status"] == seams.INAPPLICABLE for r in a) / len(a)
        bp = sum(r["resolved"] for r in b) / len(b)
        bi = sum(r["final_status"] == seams.INAPPLICABLE for r in b) / len(b)
        best_a = max(best_a, ap)
        print(f"  {mid:34s} {ap:6.3f}  {ai:8.3f}  {bp:6.3f}  {bi:8.3f}")

    # G4: доля INAPPLICABLE на ВСЕХ шагах, а не только на финальном
    for arm, key in (("A_WHOLE", "step_results"), ("B_STEP_SAME", "step_results")):
        cells = [s for r in rows if r["arm"] == arm for s in r[key]]
        inap = sum(s["status"] == seams.INAPPLICABLE for s in cells) / len(cells)
        print(f"  G4 {arm:14s}: INAPPLICABLE по всем ячейкам шагов = {inap:.3f}")

    print(f"\n  пригодность полигона: лучший A:PASS = {best_a:.3f}")
    if best_a > EASY_THRESHOLD:
        print(f"  ОСТАНОВ: полигон слишком ЛЁГКИЙ (>{EASY_THRESHOLD}), переделать генератор")
        return 1
    if best_a < HARD_THRESHOLD:
        print(f"  ОСТАНОВ: полигон слишком ТЯЖЁЛЫЙ (<{HARD_THRESHOLD}), переделать генератор")
        return 1
    print("  полигон пригоден, можно продолжать")
    return 0


def cmd_train(args) -> int:
    tasks, sp = _tasks_and_split()
    path = RUNS / "train_grid.json"

    def ckpt(rows):
        _write(path, {"experiment_id": H.EXPERIMENT_ID, "rows": rows})

    t0 = time.time()
    rows = H.run_train_grid(tasks, sp["train"], on_checkpoint=ckpt)
    ckpt(rows)
    print(f"[train] {len(rows)} записей за {time.time() - t0:.0f}с -> {path}")
    return 0


def cmd_g0(args) -> int:
    data = _read(RUNS / "train_grid.json")
    if data is None:
        print("[g0] нет runs14/train_grid.json — сначала --phase train")
        return 1
    rows = data["rows"]
    rates = H.per_kind_rates(rows)
    gate = H.gate_g0(rates)
    best = H.best_single_model(rows)

    print("=== G0: гейт полигона — доля PASS по (род шага x модель), train ===")
    header = "  род        " + "".join(f"{m.split('-')[0][:9]:>11s}" for m in MODEL_IDS)
    print(header)
    for kind in STEP_KINDS:
        line = f"  {kind:10s} " + "".join(f"{rates[kind][m]:11.3f}" for m in MODEL_IDS)
        print(line)
    print(f"\n  таблица маршрутизации: "
          + ", ".join(f"{k} -> {v.split('-')[0]}" for k, v in gate["table"].items()))
    print(f"  различных победителей: {gate['distinct_winners']}")
    chain = H.all_steps_rates(rows)
    print("\nВСЯ ЦЕПОЧКА верна (все шаги), train:")
    for m, v in sorted(chain.items(), key=lambda kv: -kv[1]):
        print(f"     {v:.3f}  {m}")
    print(f"  лучшая ОДИНОЧНАЯ модель по цепочке: {best}")

    out = {"experiment_id": H.EXPERIMENT_ID, **gate, "best_single_model": best,
           "all_steps_rates_train": chain}
    _write(METRICS / "g0.json", out)

    if not gate["passed"]:
        print("\n  G0 НЕ ПРОЙДЕН: одна модель лучшая для всех родов шагов.")
        print("  Гетерогенности негде взяться — тезис на этом пуле опровергнут.")
        print("  Кампания останавливается, плечо C не гоняется (предрегистрировано).")
        return 1
    print("\n  G0 ПРОЙДЕН: специализация есть, плечо C имеет смысл")
    return 0


def cmd_test(args) -> int:
    g0 = _read(METRICS / "g0.json")
    if g0 is None:
        print("[test] нет metrics14/g0.json — сначала --phase g0")
        return 1
    if not g0["passed"]:
        print("[test] G0 не пройден — кампания остановлена предрегистрированно")
        return 1

    tasks, sp = _tasks_and_split()
    ids = sp["test"]
    table, best = g0["table"], g0["best_single_model"]
    routed = [table[k] for k in STEP_KINDS]
    needed = sorted(set(routed) | {best})
    print(f"[test] {len(ids)} задач; лучшая одиночная = {best}; маршрут = "
          + ", ".join(f"{k}->{table[k].split('-')[0]}" for k in STEP_KINDS))

    llms = {}
    for mid in needed:
        llms[mid], lt = load_model(mid)
        print(f"[test] загружена {mid} за {lt:.1f}с", flush=True)

    rows, t0 = [], time.time()
    for n, tid in enumerate(ids, 1):
        task = tasks[tid]
        rows.append(H.run_whole(task, best, llms[best]))
        rows.append(H.run_steps(task, [best] * len(task["steps"]), llms, "B_STEP_SAME", True))
        rows.append(H.run_steps(task, routed, llms, "C_STEP_ROUTED", True))
        rows.append(H.run_steps(task, [best] * len(task["steps"]), llms, "D_STEP_NOORACLE", False))
        if n % 20 == 0:
            _write(RUNS / "test.json", {"experiment_id": H.EXPERIMENT_ID, "rows": rows})
            print(f"[test] {n}/{len(ids)} задач, {time.time() - t0:.0f}с", flush=True)
    _write(RUNS / "test.json", {"experiment_id": H.EXPERIMENT_ID, "rows": rows,
                                "best_single_model": best, "routing_table": table})
    print(f"[test] готово: {len(rows)} записей за {time.time() - t0:.0f}с")
    return 0


def cmd_awhole(args) -> int:
    """Дополнительная фаза, добавленная ПОСЛЕ теста и до отчёта.

    Дефект, который она чинит: плечо A гонялось моделью, выбранной по ВСЕЙ ЦЕПОЧКЕ
    (gemma), тогда как у целостного протокола лучшая модель своя — пилот показал у
    qwen3 0.550 по финальному шагу. Сравнивать декомпозицию с искусственно ослабленным
    целостным плечом нечестно. Здесь лучший для A выбирается на TRAIN тем же способом,
    что и для B/C, и прогоняется на TEST.
    """
    tasks, sp = _tasks_and_split()
    path = RUNS / "a_whole.json"
    data = _read(path, {"train": [], "test": []})

    if not data["train"]:
        rows = []
        for mid in MODEL_IDS:
            llm, _ = load_model(mid)
            print(f"[awhole:train] {mid}", flush=True)
            for tid in sp["train"]:
                rows.append(H.run_whole(tasks[tid], mid, llm))
            del llm
            data["train"] = rows
            _write(path, data)

    def rate(rows, mid, key):
        rr = [r for r in rows if r["model"] == mid]
        return sum(r[key] for r in rr) / len(rr) if rr else 0.0

    best_chain = max(MODEL_IDS, key=lambda m: rate(data["train"], m, "resolved_all_steps"))
    best_final = max(MODEL_IDS, key=lambda m: rate(data["train"], m, "resolved"))
    print("\n=== плечо A на train, по моделям ===")
    for mid in MODEL_IDS:
        print(f"  {mid:34s} цепочка={rate(data['train'], mid, 'resolved_all_steps'):.3f}  "
              f"фин.шаг={rate(data['train'], mid, 'resolved'):.3f}")
    print(f"  лучший для A по цепочке: {best_chain}; по финальному шагу: {best_final}")

    if not data["test"]:
        rows = []
        for mid in sorted({best_chain, best_final}):
            llm, _ = load_model(mid)
            print(f"[awhole:test] {mid}", flush=True)
            for tid in sp["test"]:
                rows.append(H.run_whole(tasks[tid], mid, llm))
            del llm
        data["test"] = rows
    data["best_chain"], data["best_final"] = best_chain, best_final
    _write(path, data)
    print("\n=== плечо A на test ===")
    for mid in sorted({best_chain, best_final}):
        print(f"  {mid:34s} цепочка={rate(data['test'], mid, 'resolved_all_steps'):.3f}  "
              f"фин.шаг={rate(data['test'], mid, 'resolved'):.3f}")
    return 0


def cmd_analyze(args) -> int:
    data = _read(RUNS / "test.json")
    if data is None:
        print("[analyze] нет runs14/test.json — сначала --phase test")
        return 1
    rows = data["rows"]
    by_arm = defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["task_id"]] = r
    ids = sorted(set.intersection(*(set(v) for v in by_arm.values())))
    # Основная метрика — ВСЯ цепочка (дефект, найденный на G0 и исправленный до
    # измерения: при подтверждённом оракулом upstream «только финальный шаг» вырожден
    # для плеч B/C). Прежняя метрика публикуется рядом как вторичная.
    res = lambda arm: [by_arm[arm][t]["resolved_all_steps"] for t in ids]      # noqa: E731
    res_final = lambda arm: [by_arm[arm][t]["resolved"] for t in ids]          # noqa: E731

    def econ(arm):
        rr = [by_arm[arm][t] for t in ids]
        tok = sum(x["tokens"] for x in rr)
        n = sum(x["resolved_all_steps"] for x in rr)
        return {"resolve_rate": n / len(rr),
                "resolve_rate_final_step_only": sum(x["resolved"] for x in rr) / len(rr),
                "tokens": tok, "calls": sum(x["n_calls"] for x in rr),
                "tokens_per_resolved": (tok / n) if n else None}

    economics = {a: econ(a) for a in H.ARMS if a in by_arm}
    print(f"=== ЭКОНОМИКА ПЛЕЧ, test n={len(ids)} ===")
    print("  плечо            r(цепочка) r(фин.шаг)  вызовов  токенов  ток/решённую")
    for a, e in economics.items():
        tpr = f"{e['tokens_per_resolved']:8.1f}" if e["tokens_per_resolved"] else "     inf"
        print(f"  {a:16s} {e['resolve_rate']:10.3f} {e['resolve_rate_final_step_only']:10.3f}"
              f"  {e['calls']:7d}  {e['tokens']:7d}  {tpr}")

    g1 = paired_diff(res("C_STEP_ROUTED"), res("B_STEP_SAME"))
    g1_final = paired_diff(res_final("C_STEP_ROUTED"), res_final("B_STEP_SAME"))
    g2 = paired_diff(res("B_STEP_SAME"), res("A_WHOLE"))
    g3 = paired_diff(res("B_STEP_SAME"), res("D_STEP_NOORACLE"))

    print(f"\n=== ГЕЙТЫ (парный bootstrap, {N_BOOT} ресэмплов) ===")
    for label, g, note in (("G1  C-B  (тезис: разнообразие на шагах)", g1, "равное число вызовов"),
                           ("G2  B-A  (окупаемость декомпозиции)", g2, "B стоит 3x вызовов"),
                           ("G3  B-D  (цена оракула)", g3, "")):
        verdict = "УСПЕХ" if g["ci_95"][0] > 0 else "не значимо"
        print(f"  {label:44s} {g['point']:+.4f} CI[{g['ci_95'][0]:+.4f},{g['ci_95'][1]:+.4f}] "
              f"p={g['p_value']:.4f}  {verdict}  {note}")

    print("\n=== G4: доля INAPPLICABLE по ячейкам шагов ===")
    g4 = {}
    for a in H.ARMS:
        if a not in by_arm:
            continue
        cells = [s for t in ids for s in by_arm[a][t]["step_results"]]
        g4[a] = sum(s["status"] == seams.INAPPLICABLE for s in cells) / len(cells)
        print(f"  {a:16s} {g4[a]:.4f}")

    print("\n=== по родам шагов (доля PASS) ===")
    for a in H.ARMS:
        if a not in by_arm:
            continue
        per = defaultdict(lambda: [0, 0])
        for t in ids:
            for s in by_arm[a][t]["step_results"]:
                per[s["kind"]][0] += 1
                per[s["kind"]][1] += s["status"] == seams.PASS
        line = "  ".join(f"{k}={per[k][1] / per[k][0]:.3f}" for k in STEP_KINDS)
        print(f"  {a:16s} {line}")

    out = {"experiment_id": H.EXPERIMENT_ID, "n_test": len(ids),
           "best_single_model": data.get("best_single_model"),
           "routing_table": data.get("routing_table"),
           "economics": economics, "G1_C_minus_B": g1, "G2_B_minus_A": g2,
           "G3_B_minus_D": g3, "G4_inapplicable": g4,
           "G1_secondary_final_step_only": g1_final,
           "verdict_G1": "success" if g1["ci_95"][0] > 0 else "not significant"}
    _write(METRICS / "analysis.json", out)
    print(f"\nсохранено -> {METRICS / 'analysis.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Эксперимент 14: декомпозиция и швы")
    p.add_argument("--phase", required=True,
                   choices=["gen", "pilot", "train", "g0", "test", "awhole", "analyze"])
    args = p.parse_args()
    return {"gen": cmd_gen, "pilot": cmd_pilot, "train": cmd_train,
            "g0": cmd_g0, "test": cmd_test, "awhole": cmd_awhole,
            "analyze": cmd_analyze}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
