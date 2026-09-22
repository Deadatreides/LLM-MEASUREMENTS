"""glue_seeds.py — LIFE-8 Phase 2 (PROTOCOL.md P6, correction #4).

Verified this session, not assumed: `lean_seeds.find_robust_pairs` (run
directly against this package's own reused grid/panel) finds exactly the
same 10 two-model PAR Imp=1 signatures LIFE-5's map recorded, and
`lean_seeds.build_seeds_pop20_lean`'s existing "source 1" already
constructs a dedicated seed for every single one of them
(`P_single_slot_{kind}_{i}`) -- so literally re-adding those 10 as "glue"
would be a no-op. The genuine incremental material is LIFE-5's OTHER
PAR-structured Imp=1 signatures -- 3-to-5-model combinations `lean_
seeds.py`'s pair-only mechanism (exactly 2 models per injected slot)
cannot construct -- read directly from `agent_life5_slot_map/metrics/
per_slot_signatures.json` (already-computed, not re-derived).

Each becomes ONE single-slot-injection genotype over the `b1` baseline,
exactly the same construction `lean_seeds.py` already uses for its own
2-model pairs, just with a longer per-slot model list for the ONE
injected slot. P6: every glued genotype therefore has `|ImpSet|==1` by
construction (only the injected slot can exceed `best_single` -- the
other 3 slots keep `b1`'s own single-model choice, which by definition
cannot exceed `best_single` since `b1` IS that per-slot best) -- verified
by an explicit assert here, not just claimed.

These are NEVER added to a starting population (correction #4) -- see
`orchestrator.py`'s Phase 2 branch, which inserts them directly into
`ev.archive`/`ev.genotypes` only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402
import lean_seeds as LS     # noqa: E402
import metrics_lib as ML    # noqa: E402

ROOT = LD.ROOT
LIFE5_SIGNATURES_PATH = ROOT / "agent_life5_slot_map" / "metrics" / "per_slot_signatures.json"

STEP_KINDS = LD.HSTEP.STEP_KINDS
RETENTION_SLOTS = (0, 2, 3)   # READ, LOOKUP, COMPUTE -- FORMAT (1) never has Imp=1


def load_life5_par_signatures() -> dict:
    """{slot_index: [signature_entry, ...]} -- only `op=="PAR"` and
    `imp==True` entries, straight from LIFE-5's already-computed map."""
    with open(LIFE5_SIGNATURES_PATH, encoding="utf-8") as f:
        d = json.load(f)
    agg = d["mandatory"]["agg_gen1"]
    out = {}
    for s in RETENTION_SLOTS:
        sigs = agg[str(s)]["signatures"]
        out[s] = [e for e in sigs if e["imp"] and e["op"] == "PAR"]
    return out


def non_redundant_signatures() -> dict:
    """Filters out the 10 two-model pairs `lean_seeds.py`'s own
    construction already provides -- keeps only 3+-model PAR Imp=1
    combinations, the genuinely new material (confirmed 17 total: 3 READ,
    4 LOOKUP, 10 COMPUTE)."""
    par_sigs = load_life5_par_signatures()
    return {s: [e for e in par_sigs[s] if len(e["models"]) >= 3] for s in RETENTION_SLOTS}


def build_glue_donors(ds, panel: list, best_single: dict) -> dict:
    """-> {name: genotype}, one per non-redundant LIFE-5 signature."""
    routes = LS.build_b1_b2_b3(ds, panel)
    b1 = routes["b1"]
    extra = non_redundant_signatures()

    out: dict = {}
    for s in RETENTION_SLOTS:
        kind = STEP_KINDS[s]
        for i, sig in enumerate(extra[s]):
            models = sorted(sig["models"])
            per_kind = dict(b1)
            per_kind[kind] = models
            name = f"GLUE_{kind}_{i}_{len(models)}m"
            root = LD.HSEED.route(per_kind)
            out[name] = LD.G.genotype(root, gen=0, origin=f"glue:{name}")
    return out


def assert_glue_donors_single_slot(donors: dict, ds, panel: list, best_single: dict) -> None:
    """P6, verified not just claimed: every glued donor has `|ImpSet|==1`
    -- none is a ready-made multi-slot (let alone 3-slot, B3-equivalent)
    cover. `registry=None` is safe here: these fresh genotypes are pure
    `gen.*` CALLs, never `cx.*` composites, so `slot_models` never
    dereferences its `registry` argument for them."""
    violators = []
    for name, g in donors.items():
        imp = ML.imp_set(ds, panel, best_single, g, None)
        if len(imp) != 1:
            violators.append((name, g["complex_id"], sorted(imp)))
    assert not violators, (
        f"glue assert failed: {len(violators)} donor(s) with |ImpSet| != 1 "
        f"(P6 violation -- must never hand the organism a ready-made multi-slot cover): "
        f"{violators}")
