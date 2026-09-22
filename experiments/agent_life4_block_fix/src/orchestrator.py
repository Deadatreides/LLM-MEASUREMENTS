"""orchestrator.py — копия (не импорт) `agent_life3_block_live/src/
orchestrator.py`. Диспетчер размножения: ОДИН код-путь для WITH и CTRL,
один флаг `enable_blocks` -- расхождение запускается только первой
реальной регистрацией блока в WITH, не заранее. `roll<0.35` ->
`transfer_slot`; `0.35<=roll<0.50` -> `block_insert` (или откат на
`mutate`, если реестр пуст -- CTRL всегда здесь); иначе -> `mutate`.
РОВНО один `rng.random()` на попытку в ОБЕИХ ветках (PROTOCOL.md §5) --
эта RNG-lockstep-инвариант НЕ менялась (не Change 1/2).

LIFE-4 Change 2 (PROTOCOL.md §0, Рычаг B): `N_GENERATIONS` 24->32,
`G_STALL` 12->20 (Вариант B1) -- зафиксировано ДО кампании, не подкручено
по цифрам. `g_stall` -- обычное поле `Evolution` (`arch2/evolve.py`,
`Optional[int]=None`), уже переопределялось LIFE-3 тем же способом (12
вместо arch2's дефолта 3) -- здесь просто другое число, ноль правок
arch2. R1-R4 (`block_registry.py`) НЕ меняются вместе с окном (см.
PROTOCOL.md §4.2 -- спека явно требует держать их как в LIFE-3, если
выбран Вариант B1).

LIFE-4 Change 1 threading: `imp_fn` и `LS.assert_lean_population` теперь
передают `reg` (сигнатуры `ML.imp_set`/`LS.assert_lean_population`
изменились, см. `metrics_lib.py`/`lean_seeds.py`) -- та же живая
`HSTEP.Registry()` инстанция, что `ev.registry`.

Seeds: 20261701..20261706 -- НОВЫЕ, не LIFE-3's 20261601..20261606
(PROTOCOL.md §0 -- нет случайного повторного использования RNG-потока
между пакетами).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD          # noqa: E402
import lean_seeds as LS            # noqa: E402
import metrics_lib as ML           # noqa: E402
import slot_hgt as ST              # noqa: E402
import block_insert as BI          # noqa: E402
import block_registry as BR        # noqa: E402

E = LD.E
F = LD.F
G = LD.G
M = LD.M
HSTEP = LD.HSTEP
HSEED = LD.HSEED

BUDGET_PER_TASK = 3000
N_GENERATIONS = 32          # LIFE-4 Change 2 (было 24 в LIFE-3)
G_STALL = 20                # LIFE-4 Change 2, Вариант B1 (было 12 в LIFE-3)
P_SLOT_HGT = 0.35
P_BLOCK_INSERT = 0.15
SEEDS = (20261701, 20261702, 20261703, 20261704, 20261705, 20261706)


def build_panel(ds) -> dict:
    """PROTOCOL.md §3.1: 80 из TRAIN, детерминированно (префикс сортированного
    списка -- нет причины предпочитать случайную подвыборку прямому срезу,
    сам список задач уже пришёл из детерминированного сплита)."""
    train_ids = sorted(ds.split["train"])
    panel = train_ids[:80]
    screen = train_ids[80:]
    return {"panel": panel, "screen": screen}


def _write_births_row(births_path: Path, *, gen: int, child_id, parent_id, report: dict,
                      admitted: bool, delta_impset) -> None:
    births_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "gen": gen, "child_id": child_id, "parent_id": parent_id,
        "operator": report.get("operator"),
        "donor_id": report.get("donor_id"), "molecule_id": report.get("molecule_id"),
        "slot": report.get("slot"), "complementary": report.get("complementary"),
        "hgt_attempted": bool(report.get("hgt_attempted", False)),
        "hgt_fallback_reason": report.get("hgt_fallback_reason"),
        "block_insert_attempted": bool(report.get("block_insert_attempted", False)),
        "block_fallback_reason": report.get("block_fallback_reason"),
        "delta_impset": delta_impset, "admitted": admitted,
    }
    with open(births_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_custom_reproduce(births_path: Path, imp_fn, block_registry: dict, enable_blocks: bool,
                          p_slot_hgt: float = P_SLOT_HGT, p_block_insert: float = P_BLOCK_INSERT):
    def custom_reproduce(ev, gen: int, results: dict, control_ids: list) -> None:
        reference_cids_live = set(ev.references)
        ctx = {
            "generators": [m for m in ev.registry.generator_ids()],
            "observables": [],
            "feasible_params": ev.backend.feasible_params("heterostep"),
            "donors": [ev.archive[c] for c in ev.archive if c not in reference_cids_live],
            "composites": list(ev.registry.composite_ids()),
            "assemble_n_steps": ev.assemble_n_steps,
            # ПРЕДНАМЕРЕННО нет "donor_score_fn" -- ни здесь, ни в откате на M.mutate.
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

            roll = ev.rng.random()               # РОВНО один вызов, WITH и CTRL одинаково
            if roll < p_slot_hgt:
                child, report = ST.transfer_slot(parent, ev.rng, ctx, ev.registry, gen, imp_fn)
            elif roll < p_slot_hgt + p_block_insert:
                if enable_blocks:
                    child, report = BI.block_insert(parent, ev.rng, ctx, ev.registry, gen,
                                                    block_registry, imp_fn)
                else:
                    child, report = M.mutate(parent, ev.rng, ctx, ev.registry, gen)
                    report = dict(report or {})
                    report.setdefault("block_insert_attempted", False)
            else:
                child, report = M.mutate(parent, ev.rng, ctx, ev.registry, gen)
                report = dict(report or {})
                report.setdefault("hgt_attempted", False)
                report.setdefault("block_insert_attempted", False)

            delta = None
            if child is not None:
                delta = len(imp_fn(child)) - len(imp_fn(parent))
            admitted = bool(child and child["complex_id"] not in ev.genotypes)
            _write_births_row(births_path, gen=gen, child_id=(child["complex_id"] if child else None),
                              parent_id=parent_id, report=report or {}, admitted=admitted,
                              delta_impset=delta)
            if admitted:
                ev._admit(child, gen)
                ev._emit_birth_edges(child, gen, report, parent_id)
                added += 1

        fresh = 0
        for _ in range(n_fresh * 6):
            if fresh >= n_fresh:
                break
            g, report = M.random_genotype(ev.rng, ctx, ev.registry, gen)
            delta = None
            if g is not None:
                delta = len(imp_fn(g)) - 0
            admitted = bool(g and g["complex_id"] not in ev.genotypes)
            _write_births_row(births_path, gen=gen, child_id=(g["complex_id"] if g else None),
                              parent_id=None,
                              report={**(report or {}), "hgt_attempted": False,
                                     "block_insert_attempted": False},
                              admitted=admitted, delta_impset=delta)
            if admitted:
                ev._admit(g, gen)
                ev._emit_birth_edges(g, gen, report, None)
                fresh += 1

    return custom_reproduce


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def run_cell(mode: str, seed: int, ds, panel: list, screen: list, out_dir: Path) -> dict:
    """`mode` ∈ {"WITH","CTRL"}. Пишет `archive.json`, `heredity.jsonl`,
    `births.jsonl`, `blocks.jsonl` (пустой при CTRL), `summary.json`."""
    exp_id = f"life4_{mode.lower()}_s{seed}"
    births_path = out_dir / "births.jsonl"
    blocks_path = out_dir / "blocks.jsonl"
    for p in (births_path, blocks_path):
        if p.exists():
            p.unlink()
    traces_dir = out_dir / "traces"

    reg = HSTEP.Registry()
    be = HSTEP.HeterostepBackend(ds)
    best_single = ML.best_single_per_slot(ds, panel)

    imp_cache: dict = {}

    def imp_fn(genotype: dict) -> frozenset:
        cid = genotype.get("complex_id")
        if cid is not None and cid in imp_cache:
            return imp_cache[cid]
        val = ML.imp_set(ds, panel, best_single, genotype, reg)   # LIFE-4 Change 1: registry threaded
        if cid is not None:
            imp_cache[cid] = val
        return val

    seeds_pop = LS.build_seeds_pop20_lean(ds, panel, best_single)
    LS.assert_lean_population(seeds_pop, ds, panel, best_single, reg)   # LIFE-4 Change 1: registry threaded

    rng = random.Random(seed)
    ev = E.Evolution(
        registry=reg, backend=be, dataset=ds, experiment_id=exp_id,
        runs_dir=traces_dir, rng=rng, screen_ids=screen,
        control_slices=[panel], holdout_ids=[],
        assemble_n_steps=len(HSTEP.STEP_KINDS), budget_per_task=BUDGET_PER_TASK,
        g_stall=G_STALL,
    )
    ev.current_gen = 0
    ev.seed(seeds_pop, reference_names=HSEED.REFERENCE_NAMES, baseline_name=HSEED.BASELINE_NAME,
            gate_reference_name=HSEED.GATE_REFERENCE_NAME)

    enable_blocks = (mode == "WITH")
    block_registry: dict = {}
    custom_reproduce = make_custom_reproduce(births_path, imp_fn, block_registry, enable_blocks)

    for gen in range(N_GENERATIONS):
        ev.run_generation(gen)
        if ev.extinct:
            break
        if enable_blocks:
            deaths_at = ML.death_gen_index({"generations": ev.generations})
            BR.maybe_register_one_block(ev, block_registry, blocks_path, gen, deaths_at, gen)
        if gen < N_GENERATIONS - 1:
            res_data = F.load_results(traces_dir, exp_id)
            custom_reproduce(ev, gen + 1, res_data, panel)

    # -- вторичные метрики (PROTOCOL.md §3.3/§7): r эталонов на held-out TEST,
    # чистый lookup поверх уже собранной живой сетки, НОВЫХ generate() нет --
    test_ids = ds.split["test"]
    ref_genotypes = [seeds_pop[n] for n in HSEED.REFERENCE_NAMES]
    ev.evaluate(ref_genotypes, test_ids, "final:test")
    test_results = F.load_results(traces_dir, exp_id)
    ref_test_metrics = {name: F.metrics(test_results.get(seeds_pop[name]["complex_id"], {}), test_ids)
                        for name in HSEED.REFERENCE_NAMES}

    summary = ev.summary()
    summary["mode"] = mode
    summary["seed"] = seed
    summary["panel"] = panel
    summary["screen"] = screen
    summary["seed_names"] = {name: g["complex_id"] for name, g in seeds_pop.items()}
    summary["n_blocks_registered"] = len(block_registry)
    summary["ref_test_metrics"] = {name: {"r": m["r"], "c": m["c"]} for name, m in ref_test_metrics.items()}

    _atomic_write_json(out_dir / "archive.json",
                       {"experiment_id": exp_id, "genotypes": list(ev.archive.values())})
    ev.edges.to_jsonl(out_dir / "heredity.jsonl")
    _atomic_write_json(out_dir / "summary.json", summary)
    if not blocks_path.exists():
        blocks_path.parent.mkdir(parents=True, exist_ok=True)
        blocks_path.touch()

    return summary
