"""ceiling.py — copy (unchanged logic) of `agent_life8_close_b3/src/
ceiling.py`: B1/B2/B3/UNION on `panel` and on `test`, `GAP_LANG`/
`GAP_STACK`, `LANG_POOR`/`LANG_RICH`. Reused here purely as LIFE-9's
Phase A cross-check (`len(U_ok)/100` must equal `UNION.r_test` here,
±1 task per PROTOCOL.md — expect exact, same grid, same logic).

UNION: a task counts as solved iff, independently per step, AT LEAST ONE
of the pool's models PASSes that step's cell — the same `_r_for_route`
routine used for B1/B2/B3, just with EVERY model offered at every slot.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402
import lean_seeds as LS     # noqa: E402

STEP_KINDS = LD.STEP_KINDS
GAP_LANG_THRESHOLD = 0.05


def _r_for_route(ds, task_ids: list, route: dict) -> float:
    """`route`: {kind: [model,...]}. Task solved iff EVERY step kind has
    >=1 model in `route[kind]` PASSing that (task,kind) cell."""
    if not task_ids:
        return 0.0
    n_solved = 0
    for t in task_ids:
        solved = True
        for kind in STEP_KINDS:
            models = route[kind]
            if not any(ds.cells[(t, kind, m)]["status"] == "PASS" for m in models):
                solved = False
                break
        if solved:
            n_solved += 1
    return n_solved / len(task_ids)


def compute_ceiling(ds, panel: list, test_ids: list) -> dict:
    routes = LS.build_b1_b2_b3(ds, panel)
    union_route = {kind: list(ds.models) for kind in STEP_KINDS}

    named_routes = {"B1": routes["b1"], "B2": routes["b2"], "B3": routes["b3"],
                    "UNION": union_route}
    out: dict = {}
    for name, route in named_routes.items():
        out[name] = {
            "route": route,
            "r_panel": _r_for_route(ds, panel, route),
            "r_test": _r_for_route(ds, test_ids, route),
        }

    gap_lang = out["UNION"]["r_test"] - out["B3"]["r_test"]
    gap_stack = out["B3"]["r_test"] - out["B2"]["r_test"]
    lang_poor = gap_lang < GAP_LANG_THRESHOLD

    out["GAP_LANG"] = gap_lang
    out["GAP_STACK"] = gap_stack
    out["LANG_POOR"] = lang_poor
    out["LANG_RICH"] = not lang_poor
    out["gap_lang_threshold"] = GAP_LANG_THRESHOLD
    out["orders"] = routes["orders"]
    return out


def task_solved_by_route(ds, t: str, route: dict) -> bool:
    """Per-task version of `_r_for_route`'s inner check -- used by
    `residual.py` (kill-step anatomy) and Phase D's per-task comparison."""
    for kind in STEP_KINDS:
        models = route[kind]
        if not any(ds.cells[(t, kind, m)]["status"] == "PASS" for m in models):
            return False
    return True
