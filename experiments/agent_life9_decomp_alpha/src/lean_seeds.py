"""lean_seeds.py — LIFE-9 TRIMMED copy of `agent_life8_close_b3/src/
lean_seeds.py`: only `greedy_order_deterministic`/`build_b1_b2_b3` (route
construction), unchanged. The seed-population/Imp-pair functions
(`find_robust_pairs`, `build_seeds_pop20_lean`, `assert_lean_population`)
are deliberately NOT copied — this package builds no ASSEMBLE genotypes
and runs no `Evolution`, so there is nothing to seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402

STEP_KINDS = LD.STEP_KINDS


def _gain(ds, panel, kind, model, covered):
    return len({t for t in panel if ds.cells[(t, kind, model)]["status"] == "PASS"} - covered)


def greedy_order_deterministic(ds, panel: list, kind: str) -> list:
    """Alphabetical tie-break, not hash-order-dependent
    `heterostep_seeds._greedy_order`."""
    covered: set = set()
    chosen: list = []
    rest = set(ds.models)
    while rest:
        gains = {m: _gain(ds, panel, kind, m, covered) for m in rest}
        best = max(gains.values())
        if best == 0 and chosen:
            break
        tied = sorted(m for m, g in gains.items() if g == best)
        pick = tied[0]
        chosen.append(pick)
        covered |= {t for t in panel if ds.cells[(t, kind, pick)]["status"] == "PASS"}
        rest.discard(pick)
    return chosen


def build_b1_b2_b3(ds, panel: list) -> dict:
    orders = {kind: greedy_order_deterministic(ds, panel, kind) for kind in STEP_KINDS}
    return {
        "b1": {k: [orders[k][0]] for k in STEP_KINDS},
        "b2": {k: orders[k][:2] for k in STEP_KINDS},
        "b3": {k: list(orders[k]) for k in STEP_KINDS},
        "orders": orders,
    }
