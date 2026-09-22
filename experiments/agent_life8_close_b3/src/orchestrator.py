"""orchestrator.py — LIFE-8 PROTOCOL.md P5: ONE `run_cell` for Phase 1,
2, and 3 alike — the organism (complementary HGT `p=0.35` + `stall_from_
improvement`, ordinary D1/D2, `Evolution` unmodified) never changes
between phases. Only two things vary, both optional keyword arguments:
`extra_donors` (Phase 2's 17 archive-only glue genotypes — see
`glue_seeds.py`; correction #4: inserted into `ev.archive`/`ev.genotypes`
only, asserted disjoint from `ev.population` every generation) and which
`ds`/`panel`/`screen` are passed in (Phase 3 passes the expanded dataset
— see `model_expansion.py` — everything else about the call is identical).

At the end of every cell: re-evaluate every non-reference ARCHIVE
genotype's PANEL r (the whole archive, not just final survivors — still
0 new `generate()`, pure grid lookup via `HeterostepBackend`), pick the
single best by panel r with a deterministic tie-break on `complex_id`
(selection never looks at test r — avoiding a leakage-shaped error),
then evaluate exactly that one genotype on `test_ids` once for the
reported `r_test`.
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
import stall_from_improvement as SFI  # noqa: E402

E = LD.E
F = LD.F
HSTEP = LD.HSTEP
HSEED = LD.HSEED

BUDGET_PER_TASK = 3000
N_GENERATIONS = 32
G_STALL = 20


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


def run_cell(ds, panel: list, screen: list, seed: int, out_dir: Path,
            phase: str = "F1", extra_donors: dict = None) -> dict:
    """`phase` only affects `experiment_id` naming/summary labeling — the
    organism itself is identical (P5). `extra_donors`: Phase 2 only, see
    module docstring."""
    exp_id = f"life8_{phase.lower()}_s{seed}"
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

    extra_donor_ids: set = set()
    if extra_donors:
        for name, g in extra_donors.items():
            cid = g["complex_id"]
            ev.archive[cid] = g
            ev.genotypes[cid] = g
            extra_donor_ids.add(cid)

    custom_reproduce = DC.make_default_reproduce(births_path, imp_fn)   # mutate_fn never overridden (P5)
    stall_tracker = SFI.StallFromImprovement(G_STALL)

    for gen in range(N_GENERATIONS):
        ev.run_generation(gen)
        stall_tracker.update(ev, imp_fn)
        if extra_donor_ids:
            leaked = extra_donor_ids & set(ev.population)
            assert not leaked, (
                f"correction #4 violated: glue donor id(s) {leaked} appeared in "
                f"ev.population at gen={gen} (must stay archive-only)")
        if ev.extinct:
            break
        if gen < N_GENERATIONS - 1:
            res_data = F.load_results(traces_dir, exp_id)
            custom_reproduce(ev, gen + 1, res_data, panel)

    # -- best non-ref ARCHIVE genotype by PANEL r (never test r), then one held-out test eval --
    reference_cids = set(ev.references)
    non_ref_ids = [cid for cid in ev.archive if cid not in reference_cids]
    ev.evaluate([ev.archive[cid] for cid in non_ref_ids], panel, "final:panel_rescan")
    panel_results = F.load_results(traces_dir, exp_id)
    panel_r = {cid: F.metrics(panel_results.get(cid, {}), panel)["r"] for cid in non_ref_ids}
    best_cid = max(non_ref_ids, key=lambda c: (panel_r[c], c))   # deterministic tie-break by complex_id

    test_ids = ds.split["test"]
    ev.evaluate([ev.archive[best_cid]], test_ids, "final:test_best")
    best_test_results = F.load_results(traces_dir, exp_id)
    best_test_metrics = F.metrics(best_test_results.get(best_cid, {}), test_ids)

    ref_genotypes = [seeds_pop[n] for n in HSEED.REFERENCE_NAMES]
    ev.evaluate(ref_genotypes, test_ids, "final:test_refs")
    ref_test_results = F.load_results(traces_dir, exp_id)
    ref_test_metrics = {name: F.metrics(ref_test_results.get(seeds_pop[name]["complex_id"], {}), test_ids)
                        for name in HSEED.REFERENCE_NAMES}

    summary = ev.summary()
    summary["phase"] = phase
    summary["seed"] = seed
    summary["panel"] = panel
    summary["screen"] = screen
    summary["seed_names"] = {name: g["complex_id"] for name, g in seeds_pop.items()}
    summary["extra_donor_ids"] = sorted(extra_donor_ids)
    summary["best_panel_cid"] = best_cid
    summary["best_panel_r"] = panel_r[best_cid]
    summary["best_test_metrics"] = {"r": best_test_metrics["r"], "c": best_test_metrics["c"]}
    summary["ref_test_metrics"] = {name: {"r": m["r"], "c": m["c"]} for name, m in ref_test_metrics.items()}
    summary["a2_improved_log"] = stall_tracker.improved_log

    _atomic_write_json(out_dir / "archive.json",
                       {"experiment_id": exp_id, "genotypes": list(ev.archive.values())})
    ev.edges.to_jsonl(out_dir / "heredity.jsonl")
    _atomic_write_json(out_dir / "summary.json", summary)
    blocks_path = out_dir / "blocks.jsonl"
    if not blocks_path.exists():
        blocks_path.parent.mkdir(parents=True, exist_ok=True)
        blocks_path.touch()   # always empty -- no block registry in LIFE-8 (P5)

    return summary
