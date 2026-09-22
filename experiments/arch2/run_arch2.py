"""run_arch2.py — CLI кампаний Complex Layer.

  --phase e4   барьеры и критерий слабого сигнала по сохранённому сырью (0 GPU)
  --phase e0   офлайн-эволюция комплексов по сохранённому сырью        (0 GPU)
  --phase e2   динамика смерти: 8 поколений + проверка механизмов D1/D2/D3 (0 GPU)
  --phase e3   критерий novelty с бюджетным матчингом                  (0 GPU)
  --phase e1   живой перенос топ-комплексов на held-out задачи         (GPU)

  Пятый цикл (ASSEMBLE, SPEC.md §30-34) -- полигон HETEROSTEP, план в
  C:\\Users\\user\\.claude\\plans\\cached-hopping-aho.md:
  --phase a0   базовые линии B1/B2/B3 + покрытие train-сетки            (0 GPU)
  --phase a1   эволюция на train-сетке, вердикты V1-V3                  (0 GPU)

Чекпоинты атомарны, кампания резюмируема (та же схема, что run_experiment12/13.py).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import evolve as E            # noqa: E402
import fitness as F           # noqa: E402
import genotype as G          # noqa: E402
import heredity as H          # noqa: E402
import library                # noqa: E402
import registry as REG        # noqa: E402
import runner as R           # noqa: E402
import replay                 # noqa: E402
import heterostep as HSTEP    # noqa: E402
import heterostep_seeds as HSEED  # noqa: E402

RUNS = ROOT / "runs"
METRICS = ROOT / "metrics"

SEED_E0 = 20260818
# Пятый цикл: seed не пересекается с SEED_E0/+7(e2)/+101(h0b) — тот же принцип, что
# LIVE_SEED_BASE в live.py не пересекается с offline-фазами.
SEED_A1 = SEED_E0 + 500

# HETEROSTEP-конкретные пулы (SPEC.md §32): train-сетка -- 100 задач, а не 355
# коллизий MSARITH, поэтому `evolve.split_pools`'s SCREEN_SIZE=40/HOLDOUT_SIZE=98
# исчерпали бы пул до последнего control-среза (100-40-98 < 0). Пропорции
# сохранены (скрин ~15%, контроль остальное), holdout не резервируется -- реальный
# held-out для HETEROSTEP есть ОТДЕЛЬНЫЙ test-сплит (experiment14 --phase testgrid,
# план §5), а не срез train.
SCREEN_SIZE_A1 = 15
CONTROL_SIZE_A1 = 20

# Измерено (heterostep_seeds.py, REF_B3_greedy_cover): максимум фактической
# стоимости по 100 train-задачам -- 1552 ток. Общий `evolve.BUDGET_PER_TASK=1200`
# (подобран под MSARITH) обрезал бы 23/100 запусков ПЛАНКИ в EXHAUSTED, занижая
# r эталона. 3000 -- запас поверх измеренного максимума, не произвольный подбор
# под конкретное число (SPEC.md §32).
BUDGET_PER_TASK_A1 = 3000

# Пилотный прогон (2 gen) и первый полный прогон (10 gen, G_STALL=3 общий) показали
# вымирание на поколении 2: 5 стартовых членов популяции (против 13 именованных
# генотипов library.py для MSARITH) не успевают за 3 поколения СКРЕСТИТЬ независимо
# открытые отклонения в разных слотах (LOOKUP отдельно, COMPUTE отдельно) в одно
# составное улучшение. Зафиксировано ДО итогового прогона (не подгонка под число):
# холоднее старт -> больше терпения до диагноза вымирания.
G_STALL_A1 = 8


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def _name_of(cid: str, names: dict) -> str:
    return names.get(cid, cid)


def _print_generation(rep: dict, names: dict) -> None:
    print(f"\n--- поколение {rep['generation']} "
          f"(контроль: {rep['control_slice_size']} состояний, популяция {len(rep['population'])}) ---")
    print("  ЭТАЛОНЫ (вне конкуренции):")
    for cid, s in sorted((rep.get("references") or {}).items(),
                         key=lambda kv: _name_of(kv[0], names)):
        c = s["c"]
        c_s = "  inf" if c == float("inf") else f"{c:6.1f}"
        print(f"    r={s['r']:.3f} c={c_s} ток/resolved   {_name_of(cid, names)}")
    rows = []
    for cid, s in rep["scores"].items():
        rows.append((s["r"], s["c"], s["u"], s["delta_r"]["point"], s["delta_r"]["ci_95"],
                     s["efficiency_gain"], cid))
    rows.sort(key=lambda r: (-r[0], r[1]))
    gate = rep.get("vs_gate") or {}
    print("   r      c      u     dR(B0)  CI95           E(B0)  бьёт план  комплекс")
    for r, c, u, d, ci, e, cid in rows[:12]:
        c_s = "  inf" if c == float("inf") else f"{c:6.1f}"
        g = gate.get(cid) or {}
        mark = "ДА " if g.get("beats_gate") else "   "
        print(f"  {r:.3f} {c_s} {u:.3f} {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {e:5.2f}  {mark}    "
              f"{_name_of(cid, names)}")
    if rep["deaths"]:
        print(f"  СМЕРТИ ({len(rep['deaths'])}):")
        for d in rep["deaths"][:8]:
            print(f"    - {_name_of(d['complex_id'], names)}: {d['reason']}")
    if rep["elite_dropouts_D3"]:
        print(f"  ВЫБЫЛИ ИЗ ЭЛИТЫ (D3): {len(rep['elite_dropouts_D3'])}")
        for d in rep["elite_dropouts_D3"][:5]:
            print(f"    - {_name_of(d['complex_id'], names)}: {d['reason']}")
    print(f"  элита: {[_name_of(c, names) for c in rep['elite']]}")
    print(f"  кто-то бьёт сильнейший эталон ({_name_of(rep['gate_reference_id'], names)}, "
          f"r={rep.get('gate_r') or 0:.3f}): {rep['any_complex_beats_gate']}  "
          f"stall={rep['stall_count']}  вымирание={rep['extinct']}")


# -- e0 / e2 ------------------------------------------------------------------------------


def _run_evolution(experiment_id: str, n_generations: int, seed: int) -> dict:
    ds = replay.default_dataset()
    reg = REG.default_registry()
    be = replay.ReplayBackend(ds)
    ev = E.build(reg, be, ds, experiment_id, RUNS, seed)

    names = {}
    for name, g in library.seed_complexes(ds.models).items():
        names[g["complex_id"]] = name

    print(f"[{experiment_id}] пулы: скрининг={len(ev.screen_ids)}, "
          f"контроль={len(ev.control_slices)} срезов по {len(ev.control_slices[0])}, "
          f"held-out={len(ev.holdout_ids)}")
    print(f"[{experiment_id}] стартовая популяция: {len(ev.population)} комплексов, "
          f"эталон = {_name_of(ev.baseline_id, names)}")

    for gen in range(n_generations):
        rep = ev.run_generation(gen)
        for cid in ev.population:
            names.setdefault(cid, cid)
        _print_generation(rep, names)
        if ev.extinct:
            print(f"\n[{experiment_id}] ПОПУЛЯЦИЯ ОБЪЯВЛЕНА ВЫМЕРШЕЙ (§3.3): "
                  f"{E.G_STALL} поколения подряд ни один комплекс не бьёт эталон.")
            break
        if gen < n_generations - 1:
            ev.reproduce(gen + 1,
                         results=F.load_results(RUNS, experiment_id),
                         control_ids=ev.control_slices[gen % len(ev.control_slices)])

    summary = ev.summary()
    summary["names"] = names
    summary["replay_backend"] = {
        "served": be.n_served, "unavailable": be.n_unavailable,
        "unavailable_reasons": be.unavailable_reasons,
    }
    _write(METRICS / f"{experiment_id}_evolution.json", summary)
    _write(RUNS / experiment_id / "archive.json",
           {"experiment_id": experiment_id, "genotypes": list(ev.archive.values())})
    ev.edges.to_jsonl(RUNS / experiment_id / "heredity.jsonl")
    return summary


def cmd_e0(args) -> int:
    experiment_id = "e0-pilot" if args.pilot else "e0"
    n_gen = 2 if args.pilot else (args.generations or 6)
    summary = _run_evolution(experiment_id, n_gen, SEED_E0)

    print(f"\n[{experiment_id}] ИТОГ: поколений {summary['n_generations']}, "
          f"архив {summary['archive_size']}, вымирание={summary['extinct']}")
    rb = summary["replay_backend"]
    print(f"  replay: обслужено {rb['served']} вызовов, недоступно {rb['unavailable']}")
    for reason, n in sorted(rb["unavailable_reasons"].items(), key=lambda kv: -kv[1])[:5]:
        print(f"    {n:6d}  {reason}")
    return 0


def cmd_e2(args) -> int:
    """Динамика смерти: сработал ли КАЖДЫЙ механизм хотя бы раз (норма проекта:
    непройденная ветка не считается проверенной), и держится ли граница субсидии."""
    experiment_id = "e2"
    summary = _run_evolution(experiment_id, args.generations or 8, SEED_E0 + 7)

    fired = {"D1": 0, "D2": 0, "D3": 0}
    for rep in summary["generations"]:
        for d in rep["deaths"]:
            fired[d["reason"].split(":")[0]] = fired.get(d["reason"].split(":")[0], 0) + 1
        fired["D3"] += len(rep["elite_dropouts_D3"])

    bound = E.S_0 / (1 - E.LAMBDA)
    worst = 0.0
    for rep in summary["generations"]:
        for cid, acc in rep["accounts"].items():
            worst = max(worst, acc.get("subsidy_granted_total", 0.0))

    families = set()
    if summary["generations"]:
        last = summary["generations"][-1]
        for cid in last["elite"]:
            families.add(last["family_of"].get(cid, "?"))

    checks = {
        "D1_fired": fired.get("D1", 0) > 0,
        "D2_fired": fired.get("D2", 0) > 0,
        "D3_fired": fired.get("D3", 0) > 0,
        "subsidy_bound_holds": worst <= bound + 1e-6,
        "elite_has_3plus_root_families": len(families) >= 3,
    }
    print("\nE2 проверки динамики смерти:")
    for k, v in checks.items():
        print(f"  [{'OK  ' if v else 'НЕТ '}] {k}")
    print(f"  максимально выданная субсидия по линии: {worst:.0f} при границе {bound:.0f}")
    print(f"  корневых семейств в финальной элите: {len(families)} {sorted(families)}")
    summary["e2_checks"] = checks
    _write(METRICS / "e2_evolution.json", summary)
    return 0


# -- e3 ------------------------------------------------------------------------------------


def _shift_seed_slots(node: dict, shift: int, feasible: tuple) -> dict:
    """Тот же генотип, но с другим набором розыгрышей — операционализация «свежего
    seed-вектора» внутри offline-данных (пункт 2 критерия novelty)."""
    node = json.loads(json.dumps(node))
    for _, n in G.walk(node):
        if n.get("op") == "CALL" and n.get("params"):
            slot = n["params"].get("seed_slot")
            if slot is not None:
                n["params"]["seed_slot"] = feasible[(feasible.index(slot) + shift) % len(feasible)]
    return node


def _run_on(genotype: dict, task_ids: list, ds, reg, be, budget: int) -> dict:
    out = {}
    for tid in task_ids:
        res = R.run(genotype, ds.collision_state(tid), reg, be, budget_tokens=budget)
        out[tid] = {"outcome": res.outcome, "cost": res.cost}
    return out


def cmd_e3(args) -> int:
    """Критерий «открыт новый режим» (SPEC.md §5) — все три пункта.

    Первая версия этой команды содержала дефект измерения: «уникальные решения»
    считались по ОБЪЕДИНЕНИЮ задач, на которых оценивались разные члены элиты, и
    комплекс получал уникальность просто за то, что конкурент этой задачи не видел.
    Здесь сравнение идёт строго по ПЕРЕСЕЧЕНИЮ оценённых задач.
    """
    experiment_id = args.source
    summary_path = METRICS / f"{experiment_id}_evolution.json"
    archive_path = RUNS / experiment_id / "archive.json"
    if not summary_path.exists() or not archive_path.exists():
        print(f"[e3] нет данных кампании {experiment_id} — сначала прогоните --phase e0")
        return 1
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    with open(archive_path, encoding="utf-8") as f:
        archive = {g["complex_id"]: g for g in json.load(f)["genotypes"]}

    ds = replay.default_dataset()
    reg = REG.default_registry()
    be = replay.ReplayBackend(ds)
    results = F.load_results(RUNS, experiment_id)
    names = summary.get("names", {})
    last = summary["generations"][-1]
    elite = [c for c in last["elite"] if c in results]

    # ПЕРЕСЕЧЕНИЕ оценённых задач — иначе «уникальность» достаётся за невиданные задачи
    common = None
    for cid in elite:
        seen = set(results.get(cid, {}))
        common = seen if common is None else (common & seen)
    common = sorted(common or [])
    print(f"[e3] элита {len(elite)} комплексов, общих оценённых состояний: {len(common)}")
    if not common:
        print("  пересечение пусто — критерий неприменим к этой кампании")
        return 0

    feasible = be.feasible_params("arithmetic")["seed_slot"]
    report = {"experiment_id": experiment_id, "u_min": E.U_MIN,
              "n_common_tasks": len(common), "candidates": []}

    for cid in elite:
        others = [results.get(e, {}) for e in elite if e != cid]
        uniq = F.unique_solves(results.get(cid, {}), others, common)
        entry = {"complex_id": cid, "name": names.get(cid, cid), "n_unique": len(uniq),
                 "unique_solves": uniq, "criterion_1_uniqueness": len(uniq) >= E.U_MIN,
                 "criterion_2_reproducible": None, "criterion_3_not_budget": None,
                 "new_regime": False}

        if entry["criterion_1_uniqueness"]:
            # -- пункт 2: сохраняется ли >= 50% U(G) на другом наборе розыгрышей --
            shifted = G.genotype(_shift_seed_slots(archive[cid]["root"], 1, feasible),
                                 gen=archive[cid]["gen"], origin="e3:seed_shift")
            re_res = _run_on(shifted, uniq, ds, reg, be, E.BUDGET_PER_TASK)
            kept = sum(1 for t in uniq if re_res[t]["outcome"] == "RESOLVED")
            entry["reproduced_kept"] = kept
            entry["criterion_2_reproducible"] = kept >= 0.5 * len(uniq)

            # -- пункт 3: бюджетный матчинг лучшего конкурента --
            my_cost = sum(results[cid][t]["cost"] for t in uniq)
            best_other = max((c for c in elite if c != cid),
                             key=lambda c: F.metrics(results[c], common)["r"], default=None)
            if best_other is None:
                entry["criterion_3_not_budget"] = True
            else:
                root = archive[best_other]["root"]
                best_k, best_res, best_cost, blocked = 1, None, 0, None
                for k in (1, 2, 3, 4):
                    cand = G.genotype(G.PAR(*[_shift_seed_slots(root, i, feasible) for i in range(k)]),
                                      origin=f"e3:budget_match_x{k}")
                    violations = G.validate(cand, reg)
                    if violations:
                        blocked = violations[0]
                        break
                    res_k = _run_on(cand, uniq, ds, reg, be, E.BUDGET_PER_TASK)
                    cost_k = sum(v["cost"] for v in res_k.values())
                    if best_res is None or cost_k <= my_cost:
                        best_k, best_res, best_cost = k, res_k, cost_k
                    if cost_k > my_cost:
                        break
                solved_by_matched = sum(1 for t in uniq if best_res[t]["outcome"] == "RESOLVED")
                # Бюджет действительно сматчен только если конкурент получил
                # сопоставимые деньги. Иначе критерий НЕ ПРОВЕРЕН (None), а не пройден:
                # объявить его пройденным, не выдав конкуренту бюджет, значило бы
                # вынести суждение в обход собственной проверки.
                ratio = best_cost / my_cost if my_cost else 0.0
                matched = ratio >= 0.9
                entry["budget_match"] = {
                    "competitor": names.get(best_other, best_other), "k": best_k,
                    "candidate_cost_on_U": my_cost, "competitor_cost_on_U": best_cost,
                    "budget_ratio": ratio, "budget_actually_matched": matched,
                    "blocked_by": blocked,
                    "competitor_solved_on_U": solved_by_matched, "n_U": len(uniq),
                }
                entry["criterion_3_not_budget"] = (solved_by_matched == 0) if matched else None

            entry["new_regime"] = bool(entry["criterion_2_reproducible"] and
                                       entry["criterion_3_not_budget"])
        report["candidates"].append(entry)

    report["any_new_regime"] = any(c["new_regime"] for c in report["candidates"])
    _write(METRICS / f"{experiment_id}_novelty.json", report)

    print(f"\n[e3] критерий novelty (u_min={E.U_MIN}), пул {len(common)} общих состояний:")
    print("  |U|  п.1  п.2  п.3  новый режим  комплекс")
    for c in sorted(report["candidates"], key=lambda d: -d["n_unique"]):
        f1 = "да " if c["criterion_1_uniqueness"] else "нет"
        f2 = {True: "да ", False: "нет", None: " - "}[c["criterion_2_reproducible"]]
        f3 = {True: "да ", False: "нет", None: " - "}[c["criterion_3_not_budget"]]
        print(f"  {c['n_unique']:3d}  {f1}  {f2}  {f3}  {'ДА' if c['new_regime'] else '--':11s}  "
              f"{c['name']}")
        bm = c.get("budget_match")
        if bm:
            print(f"       бюджетный матчинг: {bm['competitor']} x{bm['k']} "
                  f"({bm['competitor_cost_on_U']} против {bm['candidate_cost_on_U']} токенов, "
                  f"{bm['budget_ratio']:.0%}) решил {bm['competitor_solved_on_U']} из {bm['n_U']}")
            if not bm["budget_actually_matched"]:
                print(f"       ВНИМАНИЕ: бюджет НЕ сматчен ({bm['blocked_by']}) -> "
                      f"п.3 считается НЕ ПРОВЕРЕННЫМ, а не пройденным")
    if not report["any_new_regime"]:
        print("\n  Новый режим НЕ обнаружен. Это предрегистрированный допустимый исход "
              "(SPEC.md §5), а не сбой харнесса.")
    return 0


# -- h0 / h0b (наследственность) ---------------------------------------------------------


def _print_heredity(summary: dict) -> None:
    names = summary.get("names", {})
    last = summary["generations"][-1]
    print(f"\n[{summary['experiment_id']}] наследственность:")
    print(f"  рёбра: {summary['edge_counts']}")
    print(f"  архив по поколениям: {summary['archive_sizes_by_gen']}  merge_events={summary['merge_events']}")
    print(f"  shadow в финале: {len(summary['final_shadow'])}  "
          f"популяция: {len(summary['final_population'])}")
    revivals = sum(a.get("revivals", 0) for a in (last.get("accounts") or {}).values())
    print(f"  воскрешений из shadow за кампанию: {revivals}")
    for r in summary["registered_molecules"]:
        print(f"  ЗАРЕГИСТРИРОВАНА молекула {r['molecule_id']} из {_name_of(r['complex_id'], names)} "
              f"(поколение {r['gen']}, r={r['r']:.3f}, c={r['c']:.1f}, "
              f"профиль {r['cost_profile']['median_tokens']} ток / "
              f"{r['cost_profile']['median_calls']} вызовов)")
    if not summary["registered_molecules"]:
        print("  молекул не зарегистрировано")
    print("  рождений с composite CALL по поколениям: "
          + str([g.get("births_with_composite", 0) for g in summary["generations"]]))


def cmd_h0(args) -> int:
    experiment_id = "h0"
    summary = _run_evolution(experiment_id, args.generations or H.H0_GENERATIONS, SEED_E0)
    _print_heredity(summary)
    _write(METRICS / f"{experiment_id}_evolution.json", summary)
    return 0


def cmd_h0b(args) -> int:
    """Тот же протокол на СВЕЖЕМ seed — проверка, что наблюдаемое в H0 не является
    эффектом конкретного розыгрыша."""
    experiment_id = "h0b"
    summary = _run_evolution(experiment_id, args.generations or H.H0_GENERATIONS, SEED_E0 + 101)
    _print_heredity(summary)
    _write(METRICS / f"{experiment_id}_evolution.json", summary)
    return 0


def _print_selection(summary: dict) -> None:
    names = summary.get("names", {})
    print(f"\n[{summary['experiment_id']}] усиление отбора (SPEC.md §21-25):")
    for rep in summary["generations"]:
        n_niches = len(set(rep.get("niches", {})))
        rej = rep.get("parsimony_rejections") or []
        d3 = rep.get("elite_dropouts_D3") or []
        ds = rep.get("dual_slice_entrants") or {}
        two_slice = sum(1 for v in ds.values() if v.get("evaluated_on_previous_slice"))
        print(f"  gen{rep['generation']} [{rep.get('control_set_id')}]: ниш в элите={n_niches}  "
              f"D3 изгнано={len(d3)}  парсимония-отказов={len(rej)}  "
              f"новых в элите с данными за 2 среза={two_slice}/{len(ds)}")
        for r in rej:
            print(f"      отклонён (парсимония): {_name_of(r['complex_id'], names)} "
                  f"доминируется {r['dominated_by']}")


def _pick_best(rep: dict) -> Optional[str]:
    sc = rep.get("scores") or {}
    if not sc:
        return None
    return max(sc, key=lambda c: (sc[c]["r"], -sc[c]["c"]))


def cmd_s0(args) -> int:
    """Регрессионный контроль (SPEC.md §26, V1). Тот же SEED_E0, что у H0 -> те же
    пулы/срезы по построению (`split_pools` детерминирован от seed), поэтому сравнение
    ведётся buквально поколение-к-поколению на ОБЩИХ задачах. Сравнивается с УЖЕ
    ЗАПИСАННЫМ metrics/h0_evolution.json (кампания ДО усиления отбора) -- это и есть
    смысл регрессионного контроля: не ухудшило ли усиление результат H0."""
    experiment_id = "s0"
    summary = _run_evolution(experiment_id, args.generations or 8, SEED_E0)
    _print_selection(summary)
    _write(METRICS / f"{experiment_id}_evolution.json", summary)

    h0_path = METRICS / "h0_evolution.json"
    if not h0_path.exists():
        print("\n[s0] нет metrics/h0_evolution.json -- V1 не может быть посчитан "
              "(прогоните --phase h0 в отдельной, более ранней кампании)")
        return 1
    with open(h0_path, encoding="utf-8") as f:
        h0 = json.load(f)

    g = min(len(summary["generations"]), len(h0["generations"])) - 1
    if g < 0:
        print("\n[s0] нет ни одного общего поколения -- V1 не может быть посчитан")
        return 1
    rep_s0, rep_h0 = summary["generations"][g], h0["generations"][g]
    best_s0, gate_s0 = _pick_best(rep_s0), rep_s0.get("gate_reference_id")
    best_h0, gate_h0 = _pick_best(rep_h0), rep_h0.get("gate_reference_id")

    names = summary.get("names", {})
    if best_s0 is None or best_h0 is None:
        print("\n[s0] пустая scores на общем поколении -- V1 не может быть посчитан")
        return 1

    res_s0 = F.load_results(RUNS, "s0")
    res_h0 = F.load_results(RUNS, "h0")
    common = sorted(set(res_s0.get(best_s0, {})) & set(res_s0.get(gate_s0, {})) &
                    set(res_h0.get(best_h0, {})) & set(res_h0.get(gate_h0, {})))

    v1 = F.paired_ratio_ci(res_s0.get(best_s0, {}), res_s0.get(gate_s0, {}),
                           res_h0.get(best_h0, {}), res_h0.get(gate_h0, {}), common, n_boot=10000)
    report = {
        "experiment_id": "s0", "compared_generation": g, "n_common_tasks": len(common),
        "s0_best": best_s0, "s0_gate": gate_s0, "h0_best": best_h0, "h0_gate": gate_h0,
        "s0_r": rep_s0["scores"].get(best_s0, {}).get("r"),
        "h0_r": rep_h0["scores"].get(best_h0, {}).get("r"),
        "v1": v1,
    }
    _write(METRICS / "s0_regression.json", report)

    print(f"\n[s0] V1 регрессионный контроль, поколение {g}, {len(common)} общих задач:")
    print(f"  S0 лучший: {_name_of(best_s0, names)}  r={report['s0_r']}")
    print(f"  H0 лучший: {_name_of(best_h0, names)}  r={report['h0_r']}")
    if v1["point"] is None:
        print("  V1: недостаточно данных для расчёта (нет решённых задач ни в одной кампании)")
        return 1
    print(f"  E_S0/E_H0 = {v1['point']:.3f}  CI95 [{v1['ci_95'][0]:.3f}, {v1['ci_95'][1]:.3f}]")
    verdict = "ПРОВАЛ (регрессия)" if v1["regression"] else "УСПЕХ (не хуже H0)"
    print(f"  ВЕРДИКТ V1: {verdict}")
    return 1 if v1["regression"] else 0


def cmd_s1(args) -> int:
    """Основной прогон усиленного отбора на СВЕЖЕМ seed (не пересекается ни с H0
    (SEED_E0), ни с H0b (SEED_E0+101), ни с S0 (SEED_E0))."""
    experiment_id = "s1"
    summary = _run_evolution(experiment_id, args.generations or 10, SEED_E0 + 303)
    _print_selection(summary)
    _write(METRICS / f"{experiment_id}_evolution.json", summary)
    return 0


# -- autocat (A1-A5) -----------------------------------------------------------------------


def _composite_execution_ok(trace: list) -> tuple:
    """A4: у каждого CALL_COMPOSITE обязано быть вложенное исполнение.

    Мёртвая ссылка (узел в трассе есть, внутри ничего не исполнилось) — это провал
    критерия, а не деталь: именно она отличает библиотеку от автокатализа.
    """
    total = alive = 0
    for e in trace:
        if e.get("op") != "CALL_COMPOSITE":
            continue
        total += 1
        prefix = e.get("node_path", "") + "/composite"
        if any(o.get("node_path", "").startswith(prefix) for o in trace):
            alive += 1
    return total, alive


def cmd_autocat(args) -> int:
    experiment_id = args.source
    summary_path = METRICS / f"{experiment_id}_evolution.json"
    archive_path = RUNS / experiment_id / "archive.json"
    if not summary_path.exists() or not archive_path.exists():
        print(f"[autocat] нет данных кампании {experiment_id}")
        return 1
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    with open(archive_path, encoding="utf-8") as f:
        archive = {g["complex_id"]: g for g in json.load(f)["genotypes"]}

    names = summary.get("names", {})
    registered = summary.get("registered_molecules", [])
    edges = summary.get("edges", [])
    results = F.load_results(RUNS, experiment_id)

    reg_by_mid = {r["molecule_id"]: r for r in registered}
    inst = [e for e in edges if e["type"] == "instantiated_from" and e["dst"] in reg_by_mid]
    # A2 требует рождения ПОСЛЕ регистрации
    inst_after = [e for e in inst if e["gen"] >= reg_by_mid[e["dst"]]["gen"]]
    users = sorted({e["src"] for e in inst_after})

    # A3: попал ли такой потомок в элиту или побил ли планку
    successful = []
    for g in summary["generations"]:
        elite = set(g.get("elite") or [])
        beats = {cid for cid, d in (g.get("vs_gate") or {}).items() if d.get("beats_gate")}
        for cid in users:
            if cid in elite or cid in beats:
                successful.append({"complex_id": cid, "gen": g["generation"],
                                   "in_elite": cid in elite, "beats_gate": cid in beats})
    successful_ids = sorted({s["complex_id"] for s in successful})

    # A4: реальное исполнение композита в трассах потомков
    total_calls = alive_calls = 0
    for cid in users:
        for rec in results.get(cid, {}).values():
            tot, alive = _composite_execution_ok(rec.get("trace") or [])
            total_calls += tot
            alive_calls += alive

    sizes = summary.get("archive_sizes_by_gen") or []
    monotone = all(b >= a for a, b in zip(sizes, sizes[1:]))

    A = {
        "A1_registered": len(registered),
        "A2_users_after_registration": len(users),
        "A3_successful_users": len(successful_ids),
        "A4_composite_calls_total": total_calls,
        "A4_composite_calls_executed": alive_calls,
        "A4_ratio": (alive_calls / total_calls) if total_calls else None,
        "A5_merge_events": summary.get("merge_events", 0),
        "A5_archive_monotone": monotone,
    }
    verdict = {
        "A1": A["A1_registered"] >= 1,
        "A2": A["A2_users_after_registration"] >= 1,
        "A3": A["A3_successful_users"] >= 1,
        "A4": (total_calls > 0 and alive_calls == total_calls),
        "A5": (A["A5_merge_events"] == 0 and monotone),
    }
    autocatalysis = all(verdict.values())
    library_only = verdict["A1"] and verdict["A2"] and not verdict["A3"]

    # -- сопутствующее: фитнес потомков с композитом против без, на одном срезе --
    last = summary["generations"][-1]
    control = sorted({t for cid in last.get("survivors", []) for t in results.get(cid, {})})
    with_c, without_c = [], []
    for cid in last.get("survivors", []):
        g = archive.get(cid)
        if g is None:
            continue
        m = F.metrics(results.get(cid, {}), control)
        (with_c if H.uses_composite(g) else without_c).append(m)

    def _avg(rows, key):
        vals = [r[key] for r in rows if r[key] not in (None, float("inf"))]
        return sum(vals) / len(vals) if vals else None

    comparison = {
        "n_with_composite": len(with_c), "n_without_composite": len(without_c),
        "mean_r_with": _avg(with_c, "r"), "mean_r_without": _avg(without_c, "r"),
        "mean_c_with": _avg(with_c, "c"), "mean_c_without": _avg(without_c, "c"),
    }

    report = {
        "experiment_id": experiment_id, "criteria": A, "verdict": verdict,
        "autocatalysis": autocatalysis, "library_only": library_only,
        "registered_molecules": registered,
        "users": [{"complex_id": c, "name": names.get(c, c)} for c in users],
        "successful_users": successful,
        "births_with_composite_by_gen": [g.get("births_with_composite", 0)
                                         for g in summary["generations"]],
        "descendant_comparison": comparison,
        "edge_counts": summary.get("edge_counts", {}),
    }
    _write(METRICS / f"{experiment_id}_autocatalysis.json", report)

    print(f"[autocat:{experiment_id}] критерии автокатализа (предрегистрация SPEC.md §19):")
    labels = {
        "A1": f"зарегистрировано молекул: {A['A1_registered']}",
        "A2": f"потомков с CALL на них после регистрации: {A['A2_users_after_registration']}",
        "A3": f"из них достигли элиты/побили планку: {A['A3_successful_users']}",
        "A4": (f"исполнено composite-вызовов: {A['A4_composite_calls_executed']}"
               f"/{A['A4_composite_calls_total']}"),
        "A5": (f"merge_events={A['A5_merge_events']}, "
               f"архив монотонен={A['A5_archive_monotone']}"),
    }
    for k in ("A1", "A2", "A3", "A4", "A5"):
        print(f"  [{'ДА ' if verdict[k] else 'НЕТ'}] {k}: {labels[k]}")
    if autocatalysis:
        print("\n  ВЕРДИКТ: автокатализ ПОДТВЕРЖДЁН (все пять критериев).")
    elif library_only:
        print("\n  ВЕРДИКТ: это БИБЛИОТЕКА, а не автокатализ — молекула зарегистрирована и "
              "используется, но ни один потомок с ней не достиг элиты и не побил планку (A3).")
    else:
        print("\n  ВЕРДИКТ: автокатализ НЕ подтверждён.")
    c = comparison
    if c["mean_r_with"] is not None and c["mean_r_without"] is not None:
        print(f"\n  потомки с композитом (n={c['n_with_composite']}): r={c['mean_r_with']:.3f}, "
              f"c={c['mean_c_with']:.1f}")
        print(f"  без композита    (n={c['n_without_composite']}): r={c['mean_r_without']:.3f}, "
              f"c={c['mean_c_without']:.1f}")
    print(f"  рождений с композитом по поколениям: {report['births_with_composite_by_gen']}")
    return 0


# -- e4 ------------------------------------------------------------------------------------


def cmd_e4(args) -> int:
    import barriers
    return barriers.main()


# -- inspect ---------------------------------------------------------------------------------


def cmd_inspect(args) -> int:
    """Кто выжил и КАК он устроен — генотип победителей в читаемом виде."""
    summary_path = METRICS / f"{args.source}_evolution.json"
    archive_path = RUNS / args.source / "archive.json"
    if not summary_path.exists() or not archive_path.exists():
        print(f"[inspect] нет данных кампании {args.source}")
        return 1
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    with open(archive_path, encoding="utf-8") as f:
        archive = {g["complex_id"]: g for g in json.load(f)["genotypes"]}

    names = summary.get("names", {})
    last = summary["generations"][-1]
    scores = last["scores"]
    refs = last.get("references", {})

    print(f"=== {args.source}: поколение {last['generation']}, эталоны ===")
    for cid, s in sorted(refs.items(), key=lambda kv: _name_of(kv[0], names)):
        print(f"  {_name_of(cid, names):26s} r={s['r']:.3f}  c={s['c']:.1f}")

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1]["r"], kv[1]["c"]))
    print(f"\n=== топ-{args.top} выживших ===")
    for cid, s in ranked[:args.top]:
        vg = (last.get("vs_gate") or {}).get(cid, {})
        d = vg.get("delta_r_vs_gate", {})
        e = vg.get("efficiency_vs_gate", {})
        print(f"\n[{_name_of(cid, names)}]  r={s['r']:.3f}  c={s['c']:.1f}  u={s['u']:.3f}  "
              f"gen={archive[cid]['gen']}  origin={archive[cid]['origin']}")
        if d:
            print(f"  против B2: dR={d['point']:+.3f} CI[{d['ci_95'][0]:+.3f},{d['ci_95'][1]:+.3f}]  "
                  f"E={(e.get('point') or 0):.2f} CI{[round(x, 2) for x in (e.get('ci_95') or [])]}  "
                  f"бьёт={vg.get('beats_gate')}")
        print(G.describe(archive[cid]["root"], indent=1))
    return 0


# -- e1 ------------------------------------------------------------------------------------


def cmd_e1(args) -> int:
    import live
    return live.main(args)


# -- пятый цикл: ASSEMBLE / HETEROSTEP (SPEC.md §30-34) --------------------------------------


def _split_pools_a1(all_ids: list, rng: random.Random) -> dict:
    """Тот же принцип, что `evolve.split_pools`, но с пропорциями под 100-задачный
    пул HETEROSTEP вместо 355-collision пула MSARITH. Holdout не резервируется --
    реальный held-out для HETEROSTEP есть ОТДЕЛЬНЫЙ test-сплит (`experiment14
    --phase testgrid`, план §5), а не срез train."""
    ids = list(all_ids)
    rng.shuffle(ids)
    screen = ids[:SCREEN_SIZE_A1]
    control_pool = ids[SCREEN_SIZE_A1:]
    slices = [control_pool[i:i + CONTROL_SIZE_A1] for i in range(0, len(control_pool), CONTROL_SIZE_A1)]
    slices = [s for s in slices if len(s) >= CONTROL_SIZE_A1 // 2]
    return {"screen": screen, "holdout": [], "control_slices": slices}


def _build_a1(experiment_id: str, seed: int):
    ds = HSTEP.default_dataset()
    reg = HSTEP.default_registry()
    be = HSTEP.HeterostepBackend(ds)
    train_ids = ds.split["train"]

    cov = ds.coverage(train_ids)
    if cov["coverage"] < 1.0:
        raise RuntimeError(f"[a1] покрытие train-сетки {cov['coverage']:.3f} < 1.0 -- "
                           f"офлайн-результат недостоверен, кампания не начинается (план §6)")

    rng = random.Random(seed)
    pools = _split_pools_a1(train_ids, random.Random(seed + 1))
    ev = E.Evolution(
        registry=reg, backend=be, dataset=ds, experiment_id=experiment_id,
        runs_dir=RUNS, rng=rng, screen_ids=pools["screen"],
        control_slices=pools["control_slices"], holdout_ids=pools["holdout"],
        assemble_n_steps=len(HSTEP.STEP_KINDS), budget_per_task=BUDGET_PER_TASK_A1,
        g_stall=G_STALL_A1,
    )
    ev.current_gen = 0
    seeds = HSEED.seed_complexes(ds, train_ids)
    ev.seed(seeds, reference_names=HSEED.REFERENCE_NAMES, baseline_name=HSEED.BASELINE_NAME,
            gate_reference_name=HSEED.GATE_REFERENCE_NAME)
    return ev, ds, reg, be


def cmd_a0(args) -> int:
    """Гейт полигона (план §6, шаг 2): покрытие train-сетки ДОЛЖНО быть 100%, и три
    предрегистрированные базовые линии обязаны воспроизвестись Runner'ом точно так
    же, как независимым Python-подсчётом (REPORT_ASSEMBLE.md §2) -- ДО единой минуты
    эволюции. 0 GPU: сетка уже записана в experiment14/runs14/train_grid.json."""
    ds = HSTEP.default_dataset()
    reg = HSTEP.default_registry()
    be = HSTEP.HeterostepBackend(ds)
    train_ids = ds.split["train"]

    cov = ds.coverage(train_ids)
    print(f"[a0] покрытие train-сетки: {cov['found']}/{cov['total']} = {cov['coverage']:.4f}")
    seam_rep = reg.self_test_report()
    for sid, rep in seam_rep.items():
        print(f"[a0] шов {sid}: {'OK' if rep['passed'] else 'FAIL'} ({rep['n_cases']} случаев)")

    seeds = HSEED.seed_complexes(ds, train_ids)
    results = {}
    print("\n[a0] базовые линии (train, 100 задач):")
    print("  имя                      r       ток/задачу  n_nodes  валиден")
    for name in HSEED.REFERENCE_NAMES:
        g = seeds[name]
        valid = G.validate(g, reg) == []
        n = 0
        cost = 0
        for tid in train_ids:
            res = R.run(g, ds.initial_state(tid), reg, be, budget_tokens=BUDGET_PER_TASK_A1)
            n += res.outcome == R.RESOLVED
            cost += res.cost
        r, tok = n / len(train_ids), cost / len(train_ids)
        results[name] = {"r": r, "tokens_per_task": tok, "n_nodes": G.n_nodes(g["root"]), "valid": valid}
        print(f"  {name:24s}  {r:.4f}  {tok:9.1f}  {G.n_nodes(g['root']):7d}  {valid}")

    ok = (cov["coverage"] == 1.0 and all(rep["passed"] for rep in seam_rep.values())
         and all(v["valid"] for v in results.values()))
    print(f"\n[a0] {'ПРОЙДЕН' if ok else 'ОСТАНОВ'}: покрытие=100%, швы зелёные, "
          f"базовые линии валидны -- {ok}")
    _write(METRICS / "a0_baselines.json", {"coverage": cov, "seam_self_test": seam_rep,
                                           "baselines": results, "passed": ok})
    return 0 if ok else 1


def _slot_has_par_fallback(g: dict) -> bool:
    """Структурный факт по ДЕРЕВУ генотипа (не по трассе исполнения): есть ли внутри
    какого-либо слота ASSEMBLE узел PAR с >=2 ветвями -- сигнатура ОТКРЫТОГО отбором
    отката, а не подсказанного затравкой (план §5, V2)."""
    for path, node in G.walk(g["root"]):
        if node.get("op") == "PAR" and len(node.get("children") or []) >= 2 and "/assemble[" in path:
            return True
    return False


def cmd_a1(args) -> int:
    """Эволюция на train-сетке HETEROSTEP. Вердикты V1 (обяз.), V2, V3 (план §5).

    Не переиспользует `_run_evolution` (MSARITH-специфична: `replay.default_dataset`,
    `library.seed_complexes`, SWITCH("task_family",...)) -- зеркалит её структуру на
    `heterostep`/`heterostep_seeds`."""
    experiment_id = "a1-pilot" if args.pilot else "a1"
    n_gen = 2 if args.pilot else (args.generations or 20)   # запас над G_STALL_A1=8

    ev, ds, reg, be = _build_a1(experiment_id, SEED_A1)
    names = {g["complex_id"]: name for name, g in
            HSEED.seed_complexes(ds, ds.split["train"]).items()}

    print(f"[{experiment_id}] пулы: скрин={len(ev.screen_ids)}, "
          f"контроль={len(ev.control_slices)} срезов по {len(ev.control_slices[0])}, "
          f"holdout={len(ev.holdout_ids)} (held-out -- отдельный test-сплит, не срез train)")
    print(f"[{experiment_id}] стартовая популяция: {len(ev.population)}, "
          f"эталонов: {len(ev.references)}, эталон-планка: "
          f"{names.get(ev.gate_reference_id, ev.gate_reference_id)}")

    for gen in range(n_gen):
        rep = ev.run_generation(gen)
        for cid in ev.population:
            names.setdefault(cid, cid)
        _print_generation(rep, names)
        if ev.extinct:
            print(f"\n[{experiment_id}] ПОПУЛЯЦИЯ ОБЪЯВЛЕНА ВЫМЕРШЕЙ (§3.3): "
                  f"{E.G_STALL} поколения подряд ни один комплекс не бьёт эталон.")
            break
        if gen < n_gen - 1:
            ev.reproduce(gen + 1,
                        results=F.load_results(RUNS, experiment_id),
                        control_ids=ev.control_slices[gen % len(ev.control_slices)])

    # -- вердикты V1-V3 (план §5), по последнему поколению -----------------------------
    last = ev.generations[-1] if ev.generations else {}
    scored = last.get("scores", {})
    refs = last.get("references", {})
    b1 = refs.get(ev.baseline_id, {})
    gate_r = last.get("gate_r")
    gate_cid = last.get("gate_reference_id")

    v1_hits = [cid for cid, s in scored.items()
              if s["r"] > (b1.get("r") or 0.46) and s["c"] <= (b1.get("c") or float("inf"))]
    v2_hits = [cid for cid in (ev.elite or ev.population)
              if scored.get(cid, {}).get("r", 0) >= 0.58
              and cid in ev.genotypes and _slot_has_par_fallback(ev.genotypes[cid])]
    vs_gate = last.get("vs_gate") or {}
    v3_hits = [cid for cid in scored if (vs_gate.get(cid) or {}).get("beats_gate")]

    verdicts = {
        "V1_beats_B1_depth1": {"passed": bool(v1_hits), "hits": [names.get(c, c) for c in v1_hits]},
        "V2_discovers_fallback": {"passed": bool(v2_hits), "hits": [names.get(c, c) for c in v2_hits]},
        "V3_beats_gate_B3": {"passed": bool(v3_hits), "hits": [names.get(c, c) for c in v3_hits],
                            "gate_was": names.get(gate_cid, gate_cid), "gate_r": gate_r},
    }
    print(f"\n=== [{experiment_id}] ВЕРДИКТЫ (план §5) ===")
    for vid, v in verdicts.items():
        mark = "УСПЕХ" if v["passed"] else "ПРОВАЛ"
        print(f"  {vid}: {mark}  {v}")

    summary = ev.summary()
    summary["names"] = names
    summary["verdicts"] = verdicts
    summary["backend"] = {"served": be.n_served, "unavailable": be.n_unavailable,
                          "unavailable_reasons": be.unavailable_reasons}
    _write(METRICS / f"{experiment_id}_evolution.json", summary)
    _write(RUNS / experiment_id / "archive.json",
          {"experiment_id": experiment_id, "genotypes": list(ev.archive.values())})
    ev.edges.to_jsonl(RUNS / experiment_id / "heredity.jsonl")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Complex Layer (arch2) campaigns")
    p.add_argument("--phase",
                   choices=["e0", "e1", "e2", "e3", "e4", "inspect", "h0", "h0b", "autocat",
                            "s0", "s1", "a0", "a1"],
                   required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--generations", type=int, default=0,
                   help="0 = по умолчанию фазы (e0/e2: 6, h0/h0b: SPEC §18, a1: 10)")
    p.add_argument("--source", default="e0", help="e3/e1: из какой кампании брать элиту")
    p.add_argument("--top", type=int, default=5, help="e1: сколько комплексов проверять живьём")
    p.add_argument("--n-tasks", type=int, default=40, help="e1: сколько held-out состояний")
    args = p.parse_args()

    return {"e0": cmd_e0, "e1": cmd_e1, "e2": cmd_e2, "e3": cmd_e3, "e4": cmd_e4,
            "inspect": cmd_inspect, "h0": cmd_h0, "h0b": cmd_h0b,
            "autocat": cmd_autocat, "s0": cmd_s0, "s1": cmd_s1,
            "a0": cmd_a0, "a1": cmd_a1}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
