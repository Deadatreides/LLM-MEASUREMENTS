"""orchestrator.py — PROTOCOL.md §6/§7: per-seed эволюция на ФИКСИРОВАННОЙ
80-задачной панели (вместо ротации control_slices), затем финальная full
TRAIN+TEST оценка архива (эталоны исключены — урок A2→A3→A4) и bootstrap
против B1/B3. Каждая оценка генотипа в этом файле — попадание в
`HeterostepBackend` поверх уже построенного `live_grid` (dict lookup),
НИКАКИХ новых `generate()`-вызовов здесь нет (PROTOCOL.md §1/§6).

Отличие от `agent_a4_m_plateau/src/orchestrator.py` (откуда скопированы
`build_seeds_pop20`/`custom_reproduce` — изоляция пакетов, копирование не
импорт): здесь конструируется СВЕЖИЙ `HSTEP.Registry()` на каждый seed, не
`HSTEP.default_registry()` (общий на процесс синглтон). A4 переиспользовал
синглтон между всеми 12 своими ячейками в одном процессе — для A4 это не
меняло вердикт (`self.registered` — поле САМОГО Evolution, не реестра, и
у реестра нет отдельного состояния, кроме зарегистрированных композитов),
но здесь предрегистрирована независимость именно 5 ЭВОЛЮЦИОННЫХ прогонов
друг от друга (PROTOCOL.md §6) — риск, что композит, зарегистрированный в
seed N, протечёт в `registry.generator_ids()` seed'а N+1 через общий
Registry, не должен допускаться неявно."""

from __future__ import annotations

import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD                      # noqa: E402
from recombination import swap_assemble_slot    # noqa: E402

E = LD.E
F = LD.F
G = LD.G
M = LD.M
HSTEP = LD.HSTEP
HSEED = LD.HSEED

SCREEN_SIZE_A5 = 20
PANEL_SIZE_A5 = 80
BUDGET_PER_TASK_A5 = 3000
N_GENERATIONS_A5 = 24
G_STALL_A5 = 12
P_CROSS_A5 = 0.60
MIN_FRAC_LOCAL = 0.25
SEED_SEEDS_RNG = 20260820          # A5's own seeding constant for build_seeds_pop20's random fill


def _greedy_order_local(ds, task_ids: list, kind: str) -> list:
    covered: set = set()
    chosen: list = []
    rest = set(ds.models)

    def gain(m):
        return len({t for t in task_ids if ds.cells[(t, kind, m)]["status"] == "PASS"} - covered)

    while rest:
        best = max(rest, key=gain)
        g = gain(best)
        if g == 0 and chosen:
            break
        chosen.append(best)
        covered |= {t for t in task_ids if ds.cells[(t, kind, best)]["status"] == "PASS"}
        rest.discard(best)
    return chosen


def build_seeds_pop20(ds, task_ids: list) -> dict:
    """Копия `agent_a4_m_plateau/src/orchestrator.py::build_seeds_pop20`
    (изоляция пакетов agent_aN — копирование, не импорт), состав без
    изменений: 3 эталона + 17 структурно заданных + 3 случайных ASSEMBLE
    (до ровно 20 не-эталонных)."""
    b1 = HSEED.b1_route(ds, task_ids)
    b2 = HSEED.b2_route(ds, task_ids)
    models = list(ds.models)

    out: dict = {}
    out["REF_B1_route"] = HSEED.route(b1)
    out["REF_B2_greedy2"] = HSEED.route(b2)
    out["REF_B3_greedy_cover"] = HSEED.route(HSEED.b3_route(ds, task_ids))

    out["P_single_model_x4"] = HSEED.route({k: [models[0]] for k in HSTEP.STEP_KINDS})
    out["P_single_model_x4_alt"] = HSEED.route({k: [models[-1]] for k in HSTEP.STEP_KINDS})
    out["P_second_best_route"] = HSEED.route({
        k: [_greedy_order_local(ds, task_ids, k)[1] if len(_greedy_order_local(ds, task_ids, k)) > 1
            else _greedy_order_local(ds, task_ids, k)[0]]
        for k in HSTEP.STEP_KINDS
    })
    out["P_par_lookup_only"] = HSEED.route({**b1, "LOOKUP": b2["LOOKUP"]})
    out["P_par_compute_only"] = HSEED.route({**b1, "COMPUTE": b2["COMPUTE"]})

    for m in models[1:-1]:
        out[f"P_single_{m[:8]}"] = HSEED.route({k: [m] for k in HSTEP.STEP_KINDS})

    out["P_par_read_only"] = HSEED.route({**b1, "READ": b2["READ"]})
    out["P_par_format_only"] = HSEED.route({**b1, "FORMAT": b2["FORMAT"]})

    out["P_par_read_format"] = HSEED.route({**b1, "READ": b2["READ"], "FORMAT": b2["FORMAT"]})
    out["P_par_read_lookup"] = HSEED.route({**b1, "READ": b2["READ"], "LOOKUP": b2["LOOKUP"]})
    out["P_par_read_compute"] = HSEED.route({**b1, "READ": b2["READ"], "COMPUTE": b2["COMPUTE"]})
    out["P_par_format_lookup"] = HSEED.route({**b1, "FORMAT": b2["FORMAT"], "LOOKUP": b2["LOOKUP"]})
    out["P_par_format_compute"] = HSEED.route({**b1, "FORMAT": b2["FORMAT"], "COMPUTE": b2["COMPUTE"]})
    out["P_par_lookup_compute"] = HSEED.route({**b1, "LOOKUP": b2["LOOKUP"], "COMPUTE": b2["COMPUTE"]})

    rng = random.Random(SEED_SEEDS_RNG)
    reg = HSTEP.Registry()
    be = HSTEP.HeterostepBackend(ds)
    ctx = {
        "generators": list(reg.generator_ids()), "observables": [],
        "feasible_params": be.feasible_params("heterostep"), "donors": [], "composites": [],
        "assemble_n_steps": len(HSTEP.STEP_KINDS),
    }
    idx = 1
    while len(out) - 3 < 20:
        g, _ = M.random_genotype(rng, ctx, reg, gen=0)
        if g:
            out[f"P_rand_assemble_{idx}"] = g["root"]
            idx += 1

    return {name: G.genotype(root, gen=0, origin=f"seed:{name}") for name, root in out.items()}


def split_pools_a5(train_ids: list, rng: random.Random) -> dict:
    """PROTOCOL.md §6: панель ФИКСИРОВАНА (не ротирует между поколениями —
    `control_slices=[panel]`, длина 1, `gen % 1 == 0` всегда); screen —
    непересекающийся остаток TRAIN."""
    ids = list(train_ids)
    rng.shuffle(ids)
    panel = sorted(ids[:PANEL_SIZE_A5])
    screen = sorted(ids[PANEL_SIZE_A5:])
    return {"screen": screen, "panel": panel}


def custom_reproduce(ev, gen: int, results: dict, control_ids: list, p_cross: float) -> float:
    """Копия `agent_a4_m_plateau/src/orchestrator.py::custom_reproduce`
    (изоляция пакетов): донор-пул исключает `ev.references` (урок A3→A4 —
    найдено смоук-тестом на p_cross=0.90 в A4: клон эталона получал
    r_full=0.7000 побитово). Здесь `p_cross=0.60 < 1-MIN_FRAC_LOCAL=0.75` —
    клэмп ниже НИКОГДА не включается (в отличие от A4's 0.80/0.90, которые
    оба клэмпились к 0.75 и стали побитово идентичны) — зафиксировано явно
    в PROTOCOL.md §6, чтобы не удивлять при чтении отчёта."""
    effective_p_cross = min(p_cross, 1.0 - MIN_FRAC_LOCAL)

    reference_cids_live = set(ev.references)
    obs_fn = getattr(ev.registry, "observable_names", None)
    ctx = {
        "generators": [m for m in ev.registry.generator_ids()],
        "observables": list(obs_fn()) if obs_fn else list(G.OBSERVABLES),
        "feasible_params": ev.backend.feasible_params("heterostep"),
        "donors": [ev.archive[c] for c in ev.archive if c not in reference_cids_live],
        "composites": list(ev.registry.composite_ids()),
        "donor_score_fn": ev._donor_score_fn(results, control_ids or []),
        "assemble_n_steps": ev.assemble_n_steps,
    }

    n_fresh = int(E.POP_SIZE * E.FRESH_FRACTION)
    n_children = max(0, E.POP_SIZE - len(ev.population) - n_fresh)

    parents = ev.elite or ev.population
    parent_w = [max(1e-6, ev.accounts[c].w) for c in parents] if parents else []
    added = 0

    for _ in range(n_children * 4):
        if added >= n_children or not parents:
            break
        parent_id = ev.rng.choices(parents, weights=parent_w, k=1)[0]
        parent = ev.genotypes[parent_id]

        if ev.rng.random() < effective_p_cross:
            child, report = swap_assemble_slot(parent, ev.rng, ctx, ev.registry, gen)
        else:
            child, report = M.mutate(parent, ev.rng, ctx, ev.registry, gen)

        if child and child["complex_id"] not in ev.genotypes:
            ev._admit(child, gen)
            ev._emit_birth_edges(child, gen, report, parent_id)
            added += 1

    fresh = 0
    for _ in range(n_fresh * 6):
        if fresh >= n_fresh:
            break
        g, report = M.random_genotype(ev.rng, ctx, ev.registry, gen)
        if g and g["complex_id"] not in ev.genotypes:
            ev._admit(g, gen)
            ev._emit_birth_edges(g, gen, report, None)
            fresh += 1

    return effective_p_cross


def run_seed_campaign(seed: int, ds, runs_dir: Path) -> dict:
    """PROTOCOL.md §6/§7: один seed — эволюция на фиксированной панели,
    затем финальная full TRAIN+TEST оценка архива (эталоны исключены)."""
    exp_id = f"a5_live_s{seed}"
    train_ids = ds.split["train"]
    test_ids = ds.split["test"]

    reg = HSTEP.Registry()                 # свежий реестр -- см. докстринг модуля
    be = HSTEP.HeterostepBackend(ds)

    rng = random.Random(seed)
    pools = split_pools_a5(train_ids, random.Random(seed + 1))
    panel = pools["panel"]

    ev = E.Evolution(
        registry=reg, backend=be, dataset=ds, experiment_id=exp_id,
        runs_dir=runs_dir, rng=rng, screen_ids=pools["screen"],
        control_slices=[panel], holdout_ids=[],
        assemble_n_steps=len(HSTEP.STEP_KINDS), budget_per_task=BUDGET_PER_TASK_A5,
        g_stall=G_STALL_A5,
    )
    ev.current_gen = 0

    seeds_pop = build_seeds_pop20(ds, train_ids)
    ev.seed(seeds_pop, reference_names=HSEED.REFERENCE_NAMES, baseline_name=HSEED.BASELINE_NAME,
            gate_reference_name=HSEED.GATE_REFERENCE_NAME)

    effective_p_cross = min(P_CROSS_A5, 1.0 - MIN_FRAC_LOCAL)
    for gen in range(N_GENERATIONS_A5):
        ev.run_generation(gen)
        if ev.extinct:
            break
        if gen < N_GENERATIONS_A5 - 1:
            res_data = F.load_results(runs_dir, exp_id)
            effective_p_cross = custom_reproduce(ev, gen + 1, res_data, panel, P_CROSS_A5)

    # -- snapshot (панель, ПОСЛЕДНЕЕ поколение) -- для истории, не для вердикта --
    last_rep = ev.generations[-1] if ev.generations else {}
    snapshot_scores = last_rep.get("scores", {})

    # -- archive (эталоны исключены -- урок A2→A3→A4) --
    reference_cids = set(ev.references)
    archive_genotypes = {cid: g for cid, g in ev.archive.items() if cid not in reference_cids}

    # -- full TRAIN оценка ВСЕГО архива (не только выживших/элиты); попадания в
    # HeterostepBackend -- dict lookup поверх live_grid, НЕ generate() --
    ev.evaluate(list(archive_genotypes.values()), train_ids, "final:train")
    train_results = F.load_results(runs_dir, exp_id)

    ranked = []
    for cid in archive_genotypes:
        m = F.metrics(train_results.get(cid, {}), train_ids)
        ranked.append({"complex_id": cid, "r_train": m["r"], "c_train": m["c"]})
    ranked.sort(key=lambda d: (-d["r_train"], d["c_train"] if d["c_train"] != float("inf") else 1e18))
    top3 = ranked[:3]
    top3_ids = [d["complex_id"] for d in top3]

    # -- subjects на full TEST: B1/B2/B3 + top-3 (B0 живёт отдельно, baselines.py --
    # у него нет ASSEMBLE-генотипа, целостное плечо не проходит через Runner) --
    subjects = {name: seeds_pop[name] for name in HSEED.REFERENCE_NAMES}
    for cid in top3_ids:
        subjects[cid] = archive_genotypes[cid]
    ev.evaluate(list(subjects.values()), test_ids, "final:test")
    test_results = F.load_results(runs_dir, exp_id)

    ref_cid = {name: seeds_pop[name]["complex_id"] for name in HSEED.REFERENCE_NAMES}
    b1_cid, b3_cid = ref_cid["REF_B1_route"], ref_cid["REF_B3_greedy_cover"]
    ref_test_metrics = {name: F.metrics(test_results.get(cid, {}), test_ids)
                        for name, cid in ref_cid.items()}

    top3_report = []
    for d in top3:
        cid = d["complex_id"]
        m_test = F.metrics(test_results.get(cid, {}), test_ids)
        top3_report.append({
            "complex_id": cid,
            "r_train_archive": d["r_train"], "c_train_archive": d["c_train"],
            "r_train_snapshot": (snapshot_scores.get(cid) or {}).get("r"),
            "r_test": m_test["r"], "c_test": m_test["c"],
            "delta_r_vs_b1": F.paired_delta_r(test_results.get(cid, {}), test_results.get(b1_cid, {}),
                                              test_ids, n_boot=10000),
            "delta_r_vs_b3": F.paired_delta_r(test_results.get(cid, {}), test_results.get(b3_cid, {}),
                                              test_ids, n_boot=10000),
            "efficiency_vs_b1": F.bootstrap_efficiency(test_results.get(cid, {}), test_results.get(b1_cid, {}),
                                                       test_ids, n_boot=10000),
        })

    return {
        "seed": seed, "n_generations_run": len(ev.generations), "extinct": ev.extinct,
        "archive_size": len(ev.archive), "archive_size_excl_refs": len(archive_genotypes),
        "panel_size": len(panel), "screen_size": len(pools["screen"]),
        "effective_p_cross": effective_p_cross,
        "b1_r_test": ref_test_metrics["REF_B1_route"]["r"], "b1_c_test": ref_test_metrics["REF_B1_route"]["c"],
        "b2_r_test": ref_test_metrics["REF_B2_greedy2"]["r"], "b2_c_test": ref_test_metrics["REF_B2_greedy2"]["c"],
        "b3_r_test": ref_test_metrics["REF_B3_greedy_cover"]["r"], "b3_c_test": ref_test_metrics["REF_B3_greedy_cover"]["c"],
        "top3": top3_report,
    }
