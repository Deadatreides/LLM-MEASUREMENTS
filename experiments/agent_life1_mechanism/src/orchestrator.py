"""orchestrator.py — инструментированный прогон (PROTOCOL.md §3 Gate R,
затем Phase A/B). Копия `agent_a5_live_m/src/orchestrator.py`'s
`build_seeds_pop20`/`custom_reproduce`/`run_seed_campaign` (изоляция
пакетов -- копирование, не импорт), с двумя изменениями:

1. b1/b2/b3 строятся из ФИКСИРОВАННОГО `orders` (найден Gate R1 в
   `deterministic_route.py`), а не через `heterostep_seeds._greedy_order`
   (хэш-порядок-зависимый дефект, PROTOCOL.md §5) -- это единственная
   правка, нужная для побитовой воспроизводимости архивного прогона A5.
2. КАЖДОЕ рождение дописывает строку в `births.jsonl` с ПОЛНЫМ report
   мутатора (operator, donor_id, slot если есть) -- `arch2.evolve.
   _emit_birth_edges` этого не делает (роняет `slot`, PROTOCOL.md
   раздел "Instrumentation"), и это единственный способ атрибутировать
   M5/SWAP_SLOT рекомбинацию по слотам постфактум.

Плюс: после эволюции дампятся `archive.json` (генотипы) и
`heredity.jsonl` (`ev.edges.to_jsonl`) -- тройка, которую
`arch2/run_arch2.py:159-168` пишет всегда, а `agent_a5_live_m` пропустил.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD                  # noqa: E402
import deterministic_route as DR           # noqa: E402
from recombination import swap_assemble_slot   # noqa: E402

E = LD.E
F = LD.F
G = LD.G
M = LD.M
H = LD.H
HSTEP = LD.HSTEP
HSEED = LD.HSEED

SCREEN_SIZE = 20
PANEL_SIZE = 80
BUDGET_PER_TASK = 3000
N_GENERATIONS = 24
G_STALL = 12
P_CROSS_DEFAULT = 0.60           # A5's operator, used for the Phase-A/baseline reproduction
MIN_FRAC_LOCAL = 0.25
SEED_SEEDS_RNG = 20260820        # same constant A5 used


def build_seeds_pop20(ds, orders: dict) -> dict:
    """Копия `agent_a5_live_m::build_seeds_pop20`, но b1/b2/(2-й элемент)
    берутся из ФИКСИРОВАННОГО `orders` (Gate R1), не из
    `heterostep_seeds.b1_route`/`b2_route`/`_greedy_order`. Состав
    популяции (какие P_* строятся) не менялся."""
    r = DR.routes_from_orders(orders)
    b1, b2 = r["b1"], r["b2"]
    models = list(ds.models)

    out: dict = {}
    out["REF_B1_route"] = HSEED.route(b1)
    out["REF_B2_greedy2"] = HSEED.route(b2)
    out["REF_B3_greedy_cover"] = HSEED.route(r["b3"])

    out["P_single_model_x4"] = HSEED.route({k: [models[0]] for k in HSTEP.STEP_KINDS})
    out["P_single_model_x4_alt"] = HSEED.route({k: [models[-1]] for k in HSTEP.STEP_KINDS})
    out["P_second_best_route"] = HSEED.route({
        k: [orders[k][1] if len(orders[k]) > 1 else orders[k][0]] for k in HSTEP.STEP_KINDS
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


def split_pools(train_ids: list, rng: random.Random) -> dict:
    ids = list(train_ids)
    rng.shuffle(ids)
    panel = sorted(ids[:PANEL_SIZE])
    screen = sorted(ids[PANEL_SIZE:])
    return {"screen": screen, "panel": panel}


def _write_births_row(births_path: Path, *, gen: int, child_id: str, parent_id, report: dict) -> None:
    births_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"gen": gen, "child_id": child_id, "parent_id": parent_id,
          "operator": report.get("operator"), "donor_id": report.get("donor_id"),
          "slot": report.get("slot"), "composite_id": report.get("composite_id")}
    with open(births_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_custom_reproduce(births_path: Path, recomb_fn=swap_assemble_slot,
                          p_cross: float = P_CROSS_DEFAULT):
    """Возвращает `custom_reproduce(ev, gen, results, control_ids)` замкнутую
    на `births_path`/`recomb_fn`/`p_cross`.

    По умолчанию `recomb_fn = swap_assemble_slot` — ЭТО ОПЕРАТОР АРХИВНОГО
    ПРОГОНА A5, используется для Gate R2 и Phase A audit (PROTOCOL.md §4:
    audit измеряет динамику, КАК ОНА ЕСТЬ у A5, не гипотетическую "чистую"
    без рекомбинации — иначе Gate R2 не смог бы дать точное совпадение
    complex_id с архивом). Phase B передаёт СВОЙ рычаг явно (`slot_hgt.
    transfer_slot` для MECH_RECOMB) — ровно одна замена на пакет; для
    MECH_LIFETIME/MECH_SELECT рычаг НЕ здесь, а в `evolution_factory`
    (переопределённый метод `Evolution`), `recomb_fn` остаётся тем же
    `swap_assemble_slot`."""

    def custom_reproduce(ev, gen: int, results: dict, control_ids: list) -> float:
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
                child, report = recomb_fn(parent, ev.rng, ctx, ev.registry, gen)
            else:
                child, report = M.mutate(parent, ev.rng, ctx, ev.registry, gen)

            if child and child["complex_id"] not in ev.genotypes:
                ev._admit(child, gen)
                ev._emit_birth_edges(child, gen, report, parent_id)
                _write_births_row(births_path, gen=gen, child_id=child["complex_id"],
                                  parent_id=parent_id, report=report or {})
                added += 1

        fresh = 0
        for _ in range(n_fresh * 6):
            if fresh >= n_fresh:
                break
            g, report = M.random_genotype(ev.rng, ctx, ev.registry, gen)
            if g and g["complex_id"] not in ev.genotypes:
                ev._admit(g, gen)
                ev._emit_birth_edges(g, gen, report, None)
                _write_births_row(births_path, gen=gen, child_id=g["complex_id"],
                                  parent_id=None, report=report or {})
                fresh += 1

        return effective_p_cross

    return custom_reproduce


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def run_seed_campaign(seed: int, ds, orders: dict, out_dir: Path, *,
                      recomb_fn=swap_assemble_slot, p_cross: float = P_CROSS_DEFAULT,
                      evolution_factory=None,
                      n_generations: int = N_GENERATIONS, g_stall: int = G_STALL) -> dict:
    """Полный инструментированный прогон одного seed'а. Пишет в `out_dir`:
    `archive.json`, `heredity.jsonl`, `summary.json`, `births.jsonl`.
    Возвращает `ev.summary()` (плюс `orders`/`p_cross` для протокола).

    `recomb_fn` — единственный рычаг для ветки MECH_RECOMB (Phase B),
    по умолчанию `swap_assemble_slot` (=A5's оператор, нужен для Gate R2 /
    Phase A). `evolution_factory` — единственный рычаг для веток
    MECH_LIFETIME/MECH_SELECT: callable с ТОЙ ЖЕ сигнатурой, что
    `E.Evolution(...)`, обычно подкласс, переопределяющий
    `_d1_screen_deaths` или `select_elite`. `None` -> `E.Evolution`
    (PROTOCOL.md §4: ровно один рычаг на прогон, никогда оба сразу)."""
    exp_id = f"life1_s{seed}"
    train_ids = ds.split["train"]
    runs_dir = out_dir / "traces"          # per-complex-id JSONL (Evolution.evaluate), отдельно от аудит-артефактов
    births_path = out_dir / "births.jsonl"
    if births_path.exists():
        births_path.unlink()

    ev_cls = evolution_factory or E.Evolution
    reg = HSTEP.Registry()
    be = HSTEP.HeterostepBackend(ds)

    rng = random.Random(seed)
    pools = split_pools(train_ids, random.Random(seed + 1))
    panel = pools["panel"]

    ev = ev_cls(
        registry=reg, backend=be, dataset=ds, experiment_id=exp_id,
        runs_dir=runs_dir, rng=rng, screen_ids=pools["screen"],
        control_slices=[panel], holdout_ids=[],
        assemble_n_steps=len(HSTEP.STEP_KINDS), budget_per_task=BUDGET_PER_TASK,
        g_stall=g_stall,
    )
    ev.current_gen = 0

    seeds_pop = build_seeds_pop20(ds, orders)
    ev.seed(seeds_pop, reference_names=HSEED.REFERENCE_NAMES, baseline_name=HSEED.BASELINE_NAME,
            gate_reference_name=HSEED.GATE_REFERENCE_NAME)

    custom_reproduce = make_custom_reproduce(births_path, recomb_fn=recomb_fn, p_cross=p_cross)

    for gen in range(n_generations):
        ev.run_generation(gen)
        if ev.extinct:
            break
        if gen < n_generations - 1:
            res_data = F.load_results(runs_dir, exp_id)
            custom_reproduce(ev, gen + 1, res_data, panel)

    summary = ev.summary()
    summary["orders"] = {k: list(v) for k, v in orders.items()}
    summary["p_cross"] = p_cross
    summary["recomb_operator"] = getattr(recomb_fn, "__name__", None)
    summary["evolution_class"] = ev_cls.__name__
    summary["panel"] = panel
    summary["screen"] = pools["screen"]
    summary["seed_names"] = {name: g["complex_id"] for name, g in seeds_pop.items()}

    _atomic_write_json(out_dir / "archive.json",
                       {"experiment_id": exp_id, "genotypes": list(ev.archive.values())})
    ev.edges.to_jsonl(out_dir / "heredity.jsonl")
    _atomic_write_json(out_dir / "summary.json", summary)

    return summary
