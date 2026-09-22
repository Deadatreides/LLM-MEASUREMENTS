"""orchestrator.py — LIFE-7 PROTOCOL.md: one `run_cell(variant, ...)`
covering every cell. `variant` selects (a) the `Evolution` class (plain
vs `pareto_death.ParetoEvolution`) and its objective mode, and (b)
whether the `stall_from_improvement.StallFromImprovement` override is
applied each generation:

  CTRL_A / CTRL_A3 / CTRL_B  -> plain `E.Evolution`, no A2 override
                                (arch2's own gate-based stall, unchanged)
  WITH_A                     -> ParetoEvolution(SCALAR_NIMP) + A2 override
  WITH_A3                    -> ParetoEvolution(VECTOR_SLOTS) + A2 override
  WITH_B                     -> plain `E.Evolution` (Pareto death OFF)
                                + A2 override only

All variants share the identical default-channel dispatcher
(`default_channel.make_default_reproduce`, `mutate_fn` never overridden
in LIFE-7 -- see that file's docstring), identical lean-seed
construction, identical `budget_per_task=3000`/`max_gen=32`/`G_STALL=20`
-- the ONLY things that differ between CTRL and any WITH_* cell are
exactly the selection-timescale levers under test.
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
import default_channel as DC       # noqa: E402
import pareto_death as PD          # noqa: E402
import stall_from_improvement as SFI  # noqa: E402

E = LD.E
F = LD.F
HSTEP = LD.HSTEP
HSEED = LD.HSEED

BUDGET_PER_TASK = 3000
N_GENERATIONS = 32
G_STALL = 20

VARIANT_CONFIG = {
    "CTRL_A":  {"cls": "plain", "objective": None, "a2": False},
    "WITH_A":  {"cls": "pareto", "objective": PD.SCALAR_NIMP, "a2": True},
    "CTRL_A3": {"cls": "plain", "objective": None, "a2": False},
    "WITH_A3": {"cls": "pareto", "objective": PD.VECTOR_SLOTS, "a2": True},
    "WITH_B":  {"cls": "plain", "objective": None, "a2": True},
    "CTRL_B":  {"cls": "plain", "objective": None, "a2": False},
}


def build_panel(ds) -> dict:
    train_ids = sorted(ds.split["train"])
    panel = train_ids[:80]
    screen = train_ids[80:]
    return {"panel": panel, "screen": screen}


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def run_cell(variant: str, seed: int, ds, panel: list, screen: list, out_dir: Path) -> dict:
    cfg = VARIANT_CONFIG[variant]
    exp_id = f"life7_{variant.lower()}_s{seed}"
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
        val = ML.imp_set(ds, panel, best_single, genotype, reg)
        if cid is not None:
            imp_cache[cid] = val
        return val

    seeds_pop = LS.build_seeds_pop20_lean(ds, panel, best_single)
    LS.assert_lean_population(seeds_pop, ds, panel, best_single, reg)

    rng = random.Random(seed)
    evolution_cls = PD.ParetoEvolution if cfg["cls"] == "pareto" else E.Evolution
    ev = evolution_cls(
        registry=reg, backend=be, dataset=ds, experiment_id=exp_id,
        runs_dir=traces_dir, rng=rng, screen_ids=screen,
        control_slices=[panel], holdout_ids=[],
        assemble_n_steps=len(HSTEP.STEP_KINDS), budget_per_task=BUDGET_PER_TASK,
        g_stall=G_STALL,
    )
    if cfg["cls"] == "pareto":
        ev._imp_fn = imp_fn
        ev._objective_mode = cfg["objective"]

    ev.current_gen = 0
    ev.seed(seeds_pop, reference_names=HSEED.REFERENCE_NAMES, baseline_name=HSEED.BASELINE_NAME,
            gate_reference_name=HSEED.GATE_REFERENCE_NAME)

    custom_reproduce = DC.make_default_reproduce(births_path, imp_fn)   # mutate_fn never overridden in LIFE-7

    stall_tracker = SFI.StallFromImprovement(G_STALL) if cfg["a2"] else None

    for gen in range(N_GENERATIONS):
        ev.run_generation(gen)   # sets self.current_gen = gen internally (arch2/evolve.py:370)
        if stall_tracker is not None:
            stall_tracker.update(ev, imp_fn)
        if ev.extinct:
            break
        if gen < N_GENERATIONS - 1:
            res_data = F.load_results(traces_dir, exp_id)
            custom_reproduce(ev, gen + 1, res_data, panel)

    test_ids = ds.split["test"]
    ref_genotypes = [seeds_pop[n] for n in HSEED.REFERENCE_NAMES]
    ev.evaluate(ref_genotypes, test_ids, "final:test")
    test_results = F.load_results(traces_dir, exp_id)
    ref_test_metrics = {name: F.metrics(test_results.get(seeds_pop[name]["complex_id"], {}), test_ids)
                        for name in HSEED.REFERENCE_NAMES}

    summary = ev.summary()
    summary["variant"] = variant
    summary["seed"] = seed
    summary["panel"] = panel
    summary["screen"] = screen
    summary["seed_names"] = {name: g["complex_id"] for name, g in seeds_pop.items()}
    summary["ref_test_metrics"] = {name: {"r": m["r"], "c": m["c"]} for name, m in ref_test_metrics.items()}
    if stall_tracker is not None:
        summary["a2_improved_log"] = stall_tracker.improved_log
    if cfg["cls"] == "pareto":
        summary["pareto_forensic"] = getattr(ev, "_pareto_forensic", [])

    _atomic_write_json(out_dir / "archive.json",
                       {"experiment_id": exp_id, "genotypes": list(ev.archive.values())})
    ev.edges.to_jsonl(out_dir / "heredity.jsonl")
    _atomic_write_json(out_dir / "summary.json", summary)
    blocks_path = out_dir / "blocks.jsonl"
    if not blocks_path.exists():
        blocks_path.parent.mkdir(parents=True, exist_ok=True)
        blocks_path.touch()   # always empty -- no block registry in LIFE-7

    return summary
