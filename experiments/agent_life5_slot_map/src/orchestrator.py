"""orchestrator.py — LIFE-5 Phase B.2 (optional evolve, user-confirmed
this session): один рукав (без WITH/CTRL split — Phase A's дефолт это и
есть единственный канал), `default_channel.make_default_reproduce`
(комплементарный slot-HGT, БЕЗ регистрации блоков вообще). Те же
`max_gen=32`/`G_STALL=20`, что LIFE-4 (окно уже подтверждено рабочим --
все 6 WITH-клеток LIFE-4 доходили до gen=20 без искусственного
обрезания). 4 НОВЫХ seed (20261801..804) -- не пересекаются с LIFE-3's
2026160x или LIFE-4's 2026170x.
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
import unfold_metrics as ML        # noqa: E402
import default_channel as DC       # noqa: E402

E = LD.E
F = LD.F
G = LD.G
M = LD.M
HSTEP = LD.HSTEP
HSEED = LD.HSEED

BUDGET_PER_TASK = 3000
N_GENERATIONS = 32
G_STALL = 20
P_SLOT_HGT = 0.35
SEEDS = (20261801, 20261802, 20261803, 20261804)


def build_panel(ds) -> dict:
    """Идентично LIFE-3/4: 80 из TRAIN, детерминированный префикс
    отсортированного списка -- та же панель (проверено это сессией: все
    24 существующих прогона используют один и тот же 80-элементный
    список)."""
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


def run_cell(seed: int, ds, panel: list, screen: list, out_dir: Path) -> dict:
    """Пишет `archive.json`, `heredity.jsonl`, `births.jsonl`,
    `summary.json` -- НЕТ `blocks.jsonl` (Phase A не регистрирует
    ничего)."""
    exp_id = f"life5_default_s{seed}"
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

    default_reproduce = DC.make_default_reproduce(births_path, imp_fn, p_slot_hgt=P_SLOT_HGT)

    for gen in range(N_GENERATIONS):
        ev.run_generation(gen)
        if ev.extinct:
            break
        if gen < N_GENERATIONS - 1:
            res_data = F.load_results(traces_dir, exp_id)
            default_reproduce(ev, gen + 1, res_data, panel)

    test_ids = ds.split["test"]
    ref_genotypes = [seeds_pop[n] for n in HSEED.REFERENCE_NAMES]
    ev.evaluate(ref_genotypes, test_ids, "final:test")
    test_results = F.load_results(traces_dir, exp_id)
    ref_test_metrics = {name: F.metrics(test_results.get(seeds_pop[name]["complex_id"], {}), test_ids)
                        for name in HSEED.REFERENCE_NAMES}

    summary = ev.summary()
    summary["mode"] = "DEFAULT"
    summary["seed"] = seed
    summary["panel"] = panel
    summary["screen"] = screen
    summary["seed_names"] = {name: g["complex_id"] for name, g in seeds_pop.items()}
    summary["ref_test_metrics"] = {name: {"r": m["r"], "c": m["c"]} for name, m in ref_test_metrics.items()}

    _atomic_write_json(out_dir / "archive.json",
                       {"experiment_id": exp_id, "genotypes": list(ev.archive.values())})
    ev.edges.to_jsonl(out_dir / "heredity.jsonl")
    _atomic_write_json(out_dir / "summary.json", summary)

    return summary
