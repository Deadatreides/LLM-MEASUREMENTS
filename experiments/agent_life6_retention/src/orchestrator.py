"""orchestrator.py — LIFE-6 PROTOCOL.md: one `run_cell(variant, ...)`
covering every cell this package runs. `variant` selects (a) which
`mutate_fn` the default-channel dispatcher uses, and (b) whether the
`Evolution` instance is the plain class or `carrier_immunity.
ImmuneEvolution`:

  CTRL / CTRL_A3 / CTRL_B  -> plain `E.Evolution`, `mutate_fn=None` (=`M.mutate`)
  WITH_A                   -> plain `E.Evolution`, `mutate_fn=slot_safe_mutate(p_safe=0.05)`
  WITH_A_hard              -> plain `E.Evolution`, `mutate_fn=slot_safe_mutate(p_safe=0.0)`
  WITH_B                   -> `ImmuneEvolution`,    `mutate_fn=None` (=`M.mutate`)

All variants otherwise share the identical dispatcher (`default_channel.
make_default_reproduce`: `p_slot_hgt=0.35` complementary `transfer_slot`,
else `mutate_fn`), identical lean-seed construction, identical
`G_STALL=20`/`max_gen=32`/`budget_per_task=3000` -- the ONLY thing that
differs between CTRL and any WITH_* cell is exactly the one lever under
test, nothing else (PROTOCOL.md §1).
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
import slot_safe_mutate as SSM     # noqa: E402
import carrier_immunity as CI      # noqa: E402

E = LD.E
F = LD.F
HSTEP = LD.HSTEP
HSEED = LD.HSEED

BUDGET_PER_TASK = 3000
N_GENERATIONS = 32
G_STALL = 20
P_SAFE_A = 0.05
P_SAFE_HARD = 0.0
M_IMMUNITY = 2


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
    """`variant` in {CTRL, WITH_A, WITH_A_hard, CTRL_A3, WITH_B, CTRL_B}
    -- CTRL/CTRL_A3/CTRL_B are the identical plain-default dispatcher,
    kept as distinct labels only for run-directory/seed bookkeeping."""
    exp_id = f"life6_{variant.lower()}_s{seed}"
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
    evolution_cls = CI.ImmuneEvolution if variant == "WITH_B" else E.Evolution
    ev = evolution_cls(
        registry=reg, backend=be, dataset=ds, experiment_id=exp_id,
        runs_dir=traces_dir, rng=rng, screen_ids=screen,
        control_slices=[panel], holdout_ids=[],
        assemble_n_steps=len(HSTEP.STEP_KINDS), budget_per_task=BUDGET_PER_TASK,
        g_stall=G_STALL,
    )
    ev.current_gen = 0
    ev.seed(seeds_pop, reference_names=HSEED.REFERENCE_NAMES, baseline_name=HSEED.BASELINE_NAME,
            gate_reference_name=HSEED.GATE_REFERENCE_NAME)

    if variant == "WITH_A":
        mutate_fn = lambda parent, rng_, ctx, registry_, gen_: SSM.slot_safe_mutate(
            parent, rng_, ctx, registry_, gen_, imp_fn, p_safe=P_SAFE_A)
    elif variant == "WITH_A_hard":
        mutate_fn = lambda parent, rng_, ctx, registry_, gen_: SSM.slot_safe_mutate(
            parent, rng_, ctx, registry_, gen_, imp_fn, p_safe=P_SAFE_HARD)
    else:
        mutate_fn = None   # plain M.mutate -- CTRL/CTRL_A3/WITH_B/CTRL_B

    custom_reproduce = DC.make_default_reproduce(births_path, imp_fn, mutate_fn=mutate_fn)

    for gen in range(N_GENERATIONS):
        if variant == "WITH_B":
            ev.protected_ids = CI.compute_protected_ids(ev, gen, imp_fn, M=M_IMMUNITY)
        ev.run_generation(gen)
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

    _atomic_write_json(out_dir / "archive.json",
                       {"experiment_id": exp_id, "genotypes": list(ev.archive.values())})
    ev.edges.to_jsonl(out_dir / "heredity.jsonl")
    _atomic_write_json(out_dir / "summary.json", summary)
    blocks_path = out_dir / "blocks.jsonl"
    if not blocks_path.exists():
        blocks_path.parent.mkdir(parents=True, exist_ok=True)
        blocks_path.touch()   # always empty -- no block registry in LIFE-6

    return summary
