"""orchestrator.py — PROTOCOL.md §5/§6: единая панель на все 24 клетки,
`custom_reproduce`, логирующий КАЖДУЮ попытку размножения (не только
принятые -- отличие от LIFE-1), единственный рычаг -- `p_slot_hgt`
(вероятность попытки `transfer_slot` вместо обычного `arch2.mutate.mutate`).
`ctx["donor_score_fn"]` НЕ кладётся нигде (PROTOCOL.md §5) -- единственный
способ выбора донора и слота во всём пакете -- равномерный.
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

E = LD.E
F = LD.F
G = LD.G
M = LD.M
HSTEP = LD.HSTEP
HSEED = LD.HSEED

BUDGET_PER_TASK = 3000
N_GENERATIONS = 24
G_STALL = 12
P_LEVELS = (0.10, 0.20, 0.35)
SEEDS = (20261401, 20261402, 20261403, 20261404, 20261405, 20261406, 20261407, 20261408)


def build_panel(ds) -> dict:
    """PROTOCOL.md §3: одна панель на все 24 клетки. Берётся из
    `agent_life1_mechanism`'s seed 20261201 (80 id, уже существует),
    записывается в `metrics/train_panel.json`."""
    life1_summary_path = (LD.ROOT / "agent_life1_mechanism" / "metrics" / "runs_instrumented"
                          / "life1_s20261201" / "summary.json")
    with open(life1_summary_path, encoding="utf-8") as f:
        life1_summary = json.load(f)
    panel = sorted(life1_summary["panel"])
    train_ids = ds.split["train"]
    screen = sorted(set(train_ids) - set(panel))
    assert set(panel).issubset(set(train_ids)), "panel must be a subset of this dataset's TRAIN split"
    return {"panel": panel, "screen": screen}


def _write_births_row(births_path: Path, *, gen: int, child_id, parent_id, report: dict,
                      admitted: bool, delta_impset) -> None:
    births_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "gen": gen, "child_id": child_id, "parent_id": parent_id,
        "operator": report.get("operator"), "donor_id": report.get("donor_id"),
        "slot": report.get("slot"), "complementary": report.get("complementary"),
        "hgt_attempted": bool(report.get("hgt_attempted", False)),
        "hgt_fallback_reason": report.get("hgt_fallback_reason"),
        "delta_impset": delta_impset, "admitted": admitted,
    }
    with open(births_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_custom_reproduce(births_path: Path, imp_fn, p_slot_hgt: float):
    """`imp_fn(genotype_dict) -> frozenset[int]`, кэширована вызывающей
    стороной по `complex_id` (см. `run_cell`). `p_slot_hgt` -- доля актов
    размножения, где ПЫТАЕМСЯ `transfer_slot` вместо обычной мутации
    (PROTOCOL.md §6) -- попытка логируется независимо от успеха."""

    def custom_reproduce(ev, gen: int, results: dict, control_ids: list) -> float:
        reference_cids_live = set(ev.references)
        ctx = {
            "generators": [m for m in ev.registry.generator_ids()],
            "observables": [],   # HETEROSTEP: observable_names() пуст, SWITCH недостижим
            "feasible_params": ev.backend.feasible_params("heterostep"),
            "donors": [ev.archive[c] for c in ev.archive if c not in reference_cids_live],
            "composites": list(ev.registry.composite_ids()),
            "assemble_n_steps": ev.assemble_n_steps,
            # ПРЕДНАМЕРЕННО нет "donor_score_fn" -- PROTOCOL.md §5: единственный
            # способ выбора донора -- равномерный, ни здесь, ни в откате на M.mutate.
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

            if ev.rng.random() < p_slot_hgt:
                child, report = ST.transfer_slot(parent, ev.rng, ctx, ev.registry, gen, imp_fn)
            else:
                child, report = M.mutate(parent, ev.rng, ctx, ev.registry, gen)
                report = dict(report or {})
                report.setdefault("hgt_attempted", False)

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
                delta = len(imp_fn(g)) - 0   # у fresh нет родителя -- "прирост" от нуля
            admitted = bool(g and g["complex_id"] not in ev.genotypes)
            _write_births_row(births_path, gen=gen, child_id=(g["complex_id"] if g else None),
                              parent_id=None, report={**(report or {}), "hgt_attempted": False},
                              admitted=admitted, delta_impset=delta)
            if admitted:
                ev._admit(g, gen)
                ev._emit_birth_edges(g, gen, report, None)
                fresh += 1

        return p_slot_hgt

    return custom_reproduce


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def run_cell(p: float, seed: int, ds, panel: list, screen: list, out_dir: Path) -> dict:
    """Один прогон (p, seed). Пишет `archive.json`, `heredity.jsonl`,
    `births.jsonl`, `summary.json` в `out_dir` (обязательный дамп,
    PROTOCOL.md §5 -- не повторять дыру A5)."""
    exp_id = f"life2_p{p:.2f}_s{seed}"
    births_path = out_dir / "births.jsonl"
    if births_path.exists():
        births_path.unlink()
    traces_dir = out_dir / "traces"

    reg = HSTEP.Registry()
    be = HSTEP.HeterostepBackend(ds)
    best_single = ML.best_single_per_slot(ds, panel)

    imp_cache: dict = {}

    def imp_fn(genotype: dict) -> frozenset:
        cid = genotype.get("complex_id")
        if cid is not None and cid in imp_cache:
            return imp_cache[cid]
        val = ML.imp_set(ds, panel, best_single, genotype)
        if cid is not None:
            imp_cache[cid] = val
        return val

    seeds_pop = LS.build_seeds_pop20_lean(ds, panel)
    LS.assert_lean_population(seeds_pop, ds, panel, best_single)

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

    custom_reproduce = make_custom_reproduce(births_path, imp_fn, p)

    for gen in range(N_GENERATIONS):
        ev.run_generation(gen)
        if ev.extinct:
            break
        if gen < N_GENERATIONS - 1:
            res_data = F.load_results(traces_dir, exp_id)
            custom_reproduce(ev, gen + 1, res_data, panel)

    summary = ev.summary()
    summary["p_slot_hgt"] = p
    summary["seed"] = seed
    summary["panel"] = panel
    summary["screen"] = screen
    summary["seed_names"] = {name: g["complex_id"] for name, g in seeds_pop.items()}

    _atomic_write_json(out_dir / "archive.json",
                       {"experiment_id": exp_id, "genotypes": list(ev.archive.values())})
    ev.edges.to_jsonl(out_dir / "heredity.jsonl")
    _atomic_write_json(out_dir / "summary.json", summary)

    return summary
