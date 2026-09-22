"""orchestrator.py — Исполнитель прогонов 2x2x2 сетки без изменения файлов arch2/."""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
ARCH2 = ROOT / "arch2"
AGENT_DIR = ROOT / "agent_a2_assemble_budget"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ARCH2) not in sys.path:
    sys.path.insert(0, str(ARCH2))

import evolve as E
import fitness as F
import genotype as G
import heterostep as HSTEP
import heterostep_seeds as HSEED
import mutate as M

from src.coverage import build_oracle_space, analyze_archive_coverage
from src.recombination import enhanced_m5_cross_slot

SCREEN_SIZE_A2 = 15
CONTROL_SIZE_A2 = 20
BUDGET_PER_TASK_A2 = 3000

def _split_pools_a2(all_ids: list, rng: random.Random) -> dict:
    ids = list(all_ids)
    rng.shuffle(ids)
    screen = ids[:SCREEN_SIZE_A2]
    control_pool = ids[SCREEN_SIZE_A2:]
    slices = [control_pool[i:i + CONTROL_SIZE_A2] for i in range(0, len(control_pool), CONTROL_SIZE_A2)]
    slices = [s for s in slices if len(s) >= CONTROL_SIZE_A2 // 2]
    return {"screen": screen, "holdout": [], "control_slices": slices}

def build_seeds_pop20(ds: HSTEP.Dataset, task_ids: list) -> dict:
    """Формирует стартовую популяцию из 20 генотипов (без вставки B3)."""
    b1 = HSEED.b1_route(ds, task_ids)
    b2 = HSEED.b2_route(ds, task_ids)
    b3 = HSEED.b3_route(ds, task_ids)
    models = list(ds.models)

    out: dict = {}
    # Эталоны (references)
    out["REF_B1_route"] = HSEED.route(b1)
    out["REF_B2_greedy2"] = HSEED.route(b2)
    out["REF_B3_greedy_cover"] = HSEED.route(b3)

    # 5 базовых из a1
    out["P_single_model_x4"] = HSEED.route({k: [models[0]] for k in HSTEP.STEP_KINDS})
    out["P_single_model_x4_alt"] = HSEED.route({k: [models[-1]] for k in HSTEP.STEP_KINDS})
    out["P_second_best_route"] = HSEED.route({
        k: [_greedy_order_local(ds, task_ids, k)[1] if len(_greedy_order_local(ds, task_ids, k)) > 1
            else _greedy_order_local(ds, task_ids, k)[0]]
        for k in HSTEP.STEP_KINDS
    })
    out["P_par_lookup_only"] = HSEED.route({**b1, "LOOKUP": b2["LOOKUP"]})
    out["P_par_compute_only"] = HSEED.route({**b1, "COMPUTE": b2["COMPUTE"]})

    # Дополнительные одиночные маршруты (4)
    for m in models[1:-1]:
        out[f"P_single_{m[:8]}"] = HSEED.route({k: [m] for k in HSTEP.STEP_KINDS})

    # Однослотовые PAR stubs (2)
    out["P_par_read_only"] = HSEED.route({**b1, "READ": b2["READ"]})
    out["P_par_format_only"] = HSEED.route({**b1, "FORMAT": b2["FORMAT"]})

    # Двухслотовые PAR stubs (6)
    out["P_par_read_format"] = HSEED.route({**b1, "READ": b2["READ"], "FORMAT": b2["FORMAT"]})
    out["P_par_read_lookup"] = HSEED.route({**b1, "READ": b2["READ"], "LOOKUP": b2["LOOKUP"]})
    out["P_par_read_compute"] = HSEED.route({**b1, "READ": b2["READ"], "COMPUTE": b2["COMPUTE"]})
    out["P_par_format_lookup"] = HSEED.route({**b1, "FORMAT": b2["FORMAT"], "LOOKUP": b2["LOOKUP"]})
    out["P_par_format_compute"] = HSEED.route({**b1, "FORMAT": b2["FORMAT"], "COMPUTE": b2["COMPUTE"]})
    out["P_par_lookup_compute"] = HSEED.route({**b1, "LOOKUP": b2["LOOKUP"], "COMPUTE": b2["COMPUTE"]})

    # Случайные ASSEMBLE до 20 штук
    rng = random.Random(20260999)
    reg = HSTEP.default_registry()
    be = HSTEP.HeterostepBackend(ds)
    ctx = {
        "generators": list(reg.generator_ids()),
        "observables": [],
        "feasible_params": be.feasible_params("heterostep"),
        "donors": [],
        "composites": [],
        "assemble_n_steps": len(HSTEP.STEP_KINDS),
    }
    idx = 1
    while len(out) - 3 < 20:
        g, _ = M.random_genotype(rng, ctx, reg, gen=0)
        if g:
            name = f"P_rand_assemble_{idx}"
            out[name] = g["root"]
            idx += 1

    res = {}
    for name, root in out.items():
        res[name] = G.genotype(root, gen=0, origin=f"seed:{name}")
    return res

def _greedy_order_local(ds: HSTEP.Dataset, task_ids: list, kind: str) -> list:
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

def custom_reproduce_high_m5(ev: E.Evolution, gen: int, results: dict, control_ids: list) -> None:
    """Кастомное размножение с усиленным M5 (cross-slot)."""
    obs_fn = getattr(ev.registry, "observable_names", None)
    ctx = {
        "generators": [m for m in ev.registry.generator_ids()],
        "observables": list(obs_fn()) if obs_fn else list(G.OBSERVABLES),
        "feasible_params": ev.backend.feasible_params("heterostep"),
        "donors": [ev.archive[c] for c in ev.archive],
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

        # С вероятностью 40% используем усиленный cross-slot M5
        if ev.rng.random() < 0.40:
            child, report = enhanced_m5_cross_slot(parent, ev.rng, ctx, ev.registry, gen)
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

def run_cell_experiment(cell_name: str, config: dict, seed: int) -> dict:
    """Запускает один ячеечный прогон (cell + seed)."""
    start_pop_size = config["start_pop"]
    n_generations = config["n_generations"]
    g_stall = config["g_stall"]
    rec_level = config["recombination"]

    exp_id = f"{cell_name}_s{seed}"
    runs_dir = AGENT_DIR / "runs"
    
    ds = HSTEP.default_dataset()
    reg = HSTEP.default_registry()
    be = HSTEP.HeterostepBackend(ds)
    train_ids = ds.split["train"]

    rng = random.Random(seed)
    pools = _split_pools_a2(train_ids, random.Random(seed + 1))
    ev = E.Evolution(
        registry=reg, backend=be, dataset=ds, experiment_id=exp_id,
        runs_dir=runs_dir, rng=rng, screen_ids=pools["screen"],
        control_slices=pools["control_slices"], holdout_ids=pools["holdout"],
        assemble_n_steps=len(HSTEP.STEP_KINDS), budget_per_task=BUDGET_PER_TASK_A2,
        g_stall=g_stall,
    )
    ev.current_gen = 0

    if start_pop_size == 5:
        seeds = HSEED.seed_complexes(ds, train_ids)
    else:
        seeds = build_seeds_pop20(ds, train_ids)

    ev.seed(seeds, reference_names=HSEED.REFERENCE_NAMES, baseline_name=HSEED.BASELINE_NAME,
            gate_reference_name=HSEED.GATE_REFERENCE_NAME)

    names = {g["complex_id"]: name for name, g in seeds.items()}

    for gen in range(n_generations):
        rep = ev.run_generation(gen)
        for cid in ev.population:
            names.setdefault(cid, cid)
        if ev.extinct:
            break
        if gen < n_generations - 1:
            res_data = F.load_results(runs_dir, exp_id)
            ctrl_ids = ev.control_slices[gen % len(ev.control_slices)]
            if rec_level == "high":
                custom_reproduce_high_m5(ev, gen + 1, res_data, ctrl_ids)
            else:
                ev.reproduce(gen + 1, results=res_data, control_ids=ctrl_ids)

    # Вычисление итоговых метрик
    summary = ev.summary()
    archive_genotypes = list(ev.archive.values())
    
    atoms, pairs = build_oracle_space()
    cov_metrics = analyze_archive_coverage(archive_genotypes, atoms, pairs)

    # Сверка V1, V2, V3 по последнему поколению
    last_rep = ev.generations[-1] if ev.generations else {}
    scored = last_rep.get("scores", {})
    refs = last_rep.get("references", {})
    b1_ref = refs.get("REF_B1_route", {})
    b2_ref = refs.get("REF_B2_greedy2", {})
    b3_ref = refs.get("REF_B3_greedy_cover", {})

    b1_r, b1_c = b1_ref.get("r", 0.46), b1_ref.get("c", 667.4)
    b2_r = b2_ref.get("r", 0.58)

    # V1: exists complex with r > r(B1) and c <= c(B1)
    v1_pass = any(s["r"] > b1_r and s["c"] <= b1_c for s in scored.values())

    # V2: elite contains ASSEMBLE with COMPUTE PAR >= 2 and r >= r(B2)
    v2_pass = False
    for cid in last_rep.get("elite", []):
        g = ev.genotypes.get(cid)
        if not g:
            continue
        s_score = scored.get(cid, {})
        if s_score.get("r", 0) >= b2_r:
            root = g.get("root", {})
            if root.get("op") == "ASSEMBLE" and len(root.get("children", [])) >= 4:
                compute_slot = root["children"][3]
                if compute_slot.get("op") == "PAR" and len(compute_slot.get("children", [])) >= 2:
                    v2_pass = True
                    break

    # V3 (устойчивый): any_complex_beats_gate vs B3 планки
    v3_pass = any(rep.get("any_complex_beats_gate", False) for rep in ev.generations)

    best_r = max((s["r"] for s in scored.values()), default=0.0)
    best_c = min((s["c"] for s in scored.values() if s["r"] == best_r), default=float("inf"))

    result = {
        "cell": cell_name,
        "seed": seed,
        "config": config,
        "n_generations_run": len(ev.generations),
        "extinct": ev.extinct,
        "archive_size": len(ev.archive),
        "best_r": best_r,
        "best_c": best_c if best_c != float("inf") else -1.0,
        "v1_pass": v1_pass,
        "v2_pass": v2_pass,
        "v3_pass": v3_pass,
        "coverage_atoms": cov_metrics["coverage_atoms"],
        "coverage_pairs": cov_metrics["coverage_pairs"],
        "P_A": cov_metrics["P_A"],
        "P_B": cov_metrics["P_B"],
        "P_A_and_B": cov_metrics["P_A_and_B"],
    }

    return result
