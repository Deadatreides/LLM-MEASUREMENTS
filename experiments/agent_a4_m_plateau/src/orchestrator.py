"""orchestrator.py — Исполнитель 4x3 прогонов (p_cross x seed) без изменения arch2/.

Копия `agent_a3_mstar/src/orchestrator.py` со следующими изменениями
(PROTOCOL.md §3/§4):
  - N_GENERATIONS 24->32, G_STALL 12->16 (константы A4);
  - `custom_reproduce` применяет `effective_p_cross = min(p_cross,
    1 - MIN_FRAC_LOCAL)` вместо голого `p_cross` -- клэмп нового
    `min_frac_local=0.25` (swap только переставляет уже существующие атомы,
    не производит новые, поэтому доля обычной мутации не должна падать
    ниже 25% даже при p_cross=0.90);
  - snapshot-метрики по-прежнему считаются (дёшево, тот же код, что A3), но
    результат явно помечен как "для истории" -- verdict-функции
    (scripts/run_all.py) их не читают;
  - `best_nodes` из archive_verdicts прокинут в результат (нужен формуле J).

`build_seeds_pop20` -- тот же состав стартовой популяции без изменений.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH2 = ROOT / "arch2"
AGENT_DIR = ROOT / "agent_a4_m_plateau"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ARCH2) not in sys.path:
    sys.path.insert(0, str(ARCH2))

import evolve as E            # noqa: E402
import fitness as F           # noqa: E402
import genotype as G          # noqa: E402
import heterostep as HSTEP    # noqa: E402
import heterostep_seeds as HSEED  # noqa: E402
import mutate as M            # noqa: E402

from src.metrics_lib import archive_verdicts, analyze_archive_coverage, build_oracle_space  # noqa: E402
from src.recombination import swap_assemble_slot                                             # noqa: E402

SCREEN_SIZE_A4 = 15
CONTROL_SIZE_A4 = 20
BUDGET_PER_TASK_A4 = 3000
N_GENERATIONS_A4 = 32
G_STALL_A4 = 16
MIN_FRAC_LOCAL = 0.25


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
    """Идентично agent_a3_mstar/src/orchestrator.py::build_seeds_pop20 --
    скопировано без изменений, полный B3 не вставляется."""
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

    rng = random.Random(20260999)
    reg = HSTEP.default_registry()
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


def _split_pools_a4(all_ids: list, rng: random.Random) -> dict:
    ids = list(all_ids)
    rng.shuffle(ids)
    screen = ids[:SCREEN_SIZE_A4]
    control_pool = ids[SCREEN_SIZE_A4:]
    slices = [control_pool[i:i + CONTROL_SIZE_A4] for i in range(0, len(control_pool), CONTROL_SIZE_A4)]
    slices = [s for s in slices if len(s) >= CONTROL_SIZE_A4 // 2]
    return {"screen": screen, "holdout": [], "control_slices": slices}


def custom_reproduce(ev: "E.Evolution", gen: int, results: dict, control_ids: list, p_cross: float) -> float:
    """PROTOCOL.md §4: с вероятностью effective_p_cross = min(p_cross,
    1-MIN_FRAC_LOCAL) -- swap_assemble_slot, иначе немодифицированный
    arch2.mutate.mutate(...). Возвращает effective_p_cross фактически
    применённое (для явной публикации в отчёте)."""
    effective_p_cross = min(p_cross, 1.0 - MIN_FRAC_LOCAL)

    # ВАЖНО (найдено смоук-тестом ДО кампании A4, BLOCKERS.md): эталоны (в т.ч.
    # REF_B3_greedy_cover) НЕ должны быть донорами -- иначе swap_assemble_slot
    # может ПОСЛЕДОВАТЕЛЬНО скопировать все 4 слота B3 через несколько раундов
    # рекомбинации, и потомок-клон окажется в архиве с r_full=0.7000/c_full=917.37
    # -- буквальным повтором эталона, а не открытием поиска. PROTOCOL.md A3 §4
    # уже ДЕКЛАРИРОВАЛ этот запрет («ни как донора не производится ни при каком
    # уровне»), но код A3 (`custom_reproduce`) его не применял -- расхождение
    # между предрегистрацией и реализацией, обнаруженное здесь. Исправлено в A4:
    # референсы исключены из donors тем же способом, что и из архивных метрик.
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


def run_cell_experiment(p_cross: float, seed: int, runs_dir: Path) -> dict:
    """Один прогон: эволюция на p_cross/seed, затем archive-метрики (решают)
    И snapshot-метрики (только для таблицы "для истории", PROTOCOL.md §5)."""
    exp_id = f"pcross{p_cross:.2f}_s{seed}"

    ds = HSTEP.default_dataset()
    reg = HSTEP.default_registry()
    be = HSTEP.HeterostepBackend(ds)
    train_ids = ds.split["train"]

    rng = random.Random(seed)
    pools = _split_pools_a4(train_ids, random.Random(seed + 1))
    ev = E.Evolution(
        registry=reg, backend=be, dataset=ds, experiment_id=exp_id,
        runs_dir=runs_dir, rng=rng, screen_ids=pools["screen"],
        control_slices=pools["control_slices"], holdout_ids=pools["holdout"],
        assemble_n_steps=len(HSTEP.STEP_KINDS), budget_per_task=BUDGET_PER_TASK_A4,
        g_stall=G_STALL_A4,
    )
    ev.current_gen = 0

    seeds = build_seeds_pop20(ds, train_ids)
    ev.seed(seeds, reference_names=HSEED.REFERENCE_NAMES, baseline_name=HSEED.BASELINE_NAME,
            gate_reference_name=HSEED.GATE_REFERENCE_NAME)

    effective_p_cross = min(p_cross, 1.0 - MIN_FRAC_LOCAL)
    for gen in range(N_GENERATIONS_A4):
        ev.run_generation(gen)
        if ev.extinct:
            break
        if gen < N_GENERATIONS_A4 - 1:
            res_data = F.load_results(runs_dir, exp_id)
            ctrl_ids = ev.control_slices[gen % len(ev.control_slices)]
            effective_p_cross = custom_reproduce(ev, gen + 1, res_data, ctrl_ids, p_cross)

    # -- snapshot-метрики: считаются, но НЕ используются ни в одном вердикте (PROTOCOL.md §5) --
    last_rep = ev.generations[-1] if ev.generations else {}
    scored = last_rep.get("scores", {})
    refs = last_rep.get("references", {})
    b1_ref = refs.get(ev.baseline_id, {})
    b1_r, b1_c = b1_ref.get("r", 0.46), b1_ref.get("c", 667.4)
    b2_cid = seeds["REF_B2_greedy2"]["complex_id"]
    b2_ref = refs.get(b2_cid, {})
    b2_r = b2_ref.get("r", 0.58)

    v1_snap = any(s["r"] > b1_r and s["c"] <= b1_c for s in scored.values())
    v2_snap = False
    for cid in last_rep.get("elite", []):
        g = ev.genotypes.get(cid)
        if not g:
            continue
        if scored.get(cid, {}).get("r", 0) >= b2_r:
            root = g.get("root", {})
            if root.get("op") == "ASSEMBLE" and len(root.get("children", [])) >= 4:
                cs = root["children"][3]
                if cs.get("op") == "PAR" and len(cs.get("children", [])) >= 2:
                    v2_snap = True
                    break
    v3_snap = any(rep.get("any_complex_beats_gate", False) for rep in ev.generations)
    best_r_snap = max((s["r"] for s in scored.values()), default=0.0)
    best_c_snap = min((s["c"] for s in scored.values() if s["r"] == best_r_snap), default=-1.0)

    # -- archive-метрики (PROTOCOL.md §5: ЭТИ РЕШАЮТ) -- исключая посеянные эталоны,
    # тот же смоук-тестом найденный принцип, что в A3 (BLOCKERS.md) --
    reference_cids = set(ev.references)
    archive_genotypes = [g for cid, g in ev.archive.items() if cid not in reference_cids]
    atoms, pairs = build_oracle_space()
    cov = analyze_archive_coverage(archive_genotypes, atoms, pairs)
    verdicts = archive_verdicts(archive_genotypes, ds, reg, be, train_ids)

    return {
        "p_cross_nominal": p_cross, "p_cross_effective": effective_p_cross, "seed": seed,
        "n_generations_run": len(ev.generations), "extinct": ev.extinct,
        "archive_size": len(ev.archive), "archive_size_excl_refs": len(archive_genotypes),
        "v1_archive": verdicts["v1_archive"], "v2_archive": verdicts["v2_archive"],
        "v3_archive": verdicts["v3_archive"],
        "best_r_archive": verdicts["best_r_archive"], "best_c_archive": verdicts["best_c_archive"],
        "best_nodes_archive": verdicts["best_nodes"],
        "p_A_archive": cov["P_A"], "p_B_archive": cov["P_B"], "p_A_and_B_archive": cov["P_A_and_B"],
        "coverage_atoms": cov["coverage_atoms"], "coverage_pairs": cov["coverage_pairs"],
        # -- для истории, не для вердикта --
        "v1_snap": v1_snap, "v2_snap": v2_snap, "v3_snap": v3_snap,
        "best_r_snap": best_r_snap, "best_c_snap": best_c_snap,
    }
