"""residual.py — LIFE-9 Phase A (PROTOCOL.md P2/P3): residual anatomy of
UNION-dead tasks. NEW this package — no prior package computed a per-task
kill-step/density breakdown, only aggregate rates.

P2: kill-step(s) for task `t` = the STEP_KINDS with zero passing models
among all 6 on `t`. A task can have more than one kill-step. `fail_s[kind]`
counts how many U_dead tasks have `kind` as (one of) their kill-step(s) --
deliberately double-counts a task with 2+ kill-kinds (answers "how often
is this kind responsible", not a partition of U_dead).

P3: `rho(t)` = fraction of all `len(STEP_KINDS)*len(models)` (kind,model)
cells for task `t` that are NOT PASS (FAIL/INAPPLICABLE/ERROR pooled) --
a continuous "how hostile is this task overall" measure, deliberately NOT
restricted to just the kill step(s) (which would trivially always read
1.0 by construction, since a kill step's cells are ALL non-PASS by
definition -- see the docstring note in `compute_residual_anatomy`).
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402

STEP_KINDS = LD.STEP_KINDS


def compute_residual_anatomy(ds, task_ids: list) -> dict:
    """`rho(t)` is pooled over ALL 4 kinds x 6 models = 24 cells per task,
    not just the kill-step's own (trivially-all-non-PASS) cells -- this is
    what makes it an informative continuous measure rather than a
    constant 1.0 for every U_dead task."""
    kill_steps: dict = {}
    fail_s = {kind: 0 for kind in STEP_KINDS}
    rho_per_task: dict = {}
    u_dead: list = []
    u_ok: list = []

    n_models = len(ds.models)
    n_cells_per_task = len(STEP_KINDS) * n_models

    for t in task_ids:
        dead_kinds = []
        n_not_pass = 0
        for kind in STEP_KINDS:
            n_pass = sum(1 for m in ds.models if ds.cells[(t, kind, m)]["status"] == "PASS")
            n_not_pass += (n_models - n_pass)
            if n_pass == 0:
                dead_kinds.append(kind)
        rho_per_task[t] = n_not_pass / n_cells_per_task
        if dead_kinds:
            kill_steps[t] = dead_kinds
            u_dead.append(t)
            for k in dead_kinds:
                fail_s[k] += 1
        else:
            u_ok.append(t)

    rho_bar_fail = statistics.mean(rho_per_task[t] for t in u_dead) if u_dead else None
    rho_bar_ok = statistics.mean(rho_per_task[t] for t in u_ok) if u_ok else None

    return {
        "n_tasks": len(task_ids), "n_u_dead": len(u_dead), "n_u_ok": len(u_ok),
        "u_dead": sorted(u_dead), "u_ok": sorted(u_ok),
        "union_r": len(u_ok) / len(task_ids) if task_ids else 0.0,
        "kill_steps": {t: sorted(ks) for t, ks in kill_steps.items()},
        "fail_s": fail_s,
        "rho_per_task": rho_per_task,
        "rho_bar_fail": rho_bar_fail,
        "rho_bar_ok": rho_bar_ok,
    }
