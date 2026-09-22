"""whole_alpha.py — LIFE-9 Phase B (PROTOCOL.md P1, Key Discovery).

`whole_grid.json` (copied byte-identical from `agent_life8_close_b3`, itself
unchanged since LIFE-3) has full 6-model coverage on TRAIN only, plus TEST
coverage for exactly one model: `qwen2.5-coder-1.5b-instruct-q4_0` (the B0
train-winner) -- confirmed by direct inspection this session. That's why
P1 defines `B2_MODEL` as this model: it's both the best-motivated stand-in
for "a strong single whole-task solver" AND the only model Phase B can
score on TEST with zero new live calls.

`alpha_B2_raw` uses `experiment14.seams.whole_seam` completely unchanged.
`alpha_B2_reparsed` strips a leading numbered-list marker
(`"1. name = val"` -> `"name = val"`) from each line FIRST, then applies
the SAME unmodified `whole_seam` -- a package-local, disclosed correction
for a real defect in the shared, un-editable `experiment14/seams.py::
_ASSIGN_RE` (anchored at line start, so a numbered-list answer -- a common
LLM habit when asked to answer several questions at once, arguably
encouraged by `whole_prompt()`'s own stale "три строки" wording for what
is actually 4 steps -- makes every field register "missing" even when
every value is exactly correct). Verified this session against
`B2_MODEL`'s actual 200 whole_grid rows (built with the correct
`GEN_SEED=150002`, not `tasks14`'s own default): raw resolved rate 0.295,
reparsed 0.740, +89 recovered, 0 regressions -- reparse only ever adds a
pass the raw parser lost to formatting, never invents one.
`alpha_B2 := alpha_B2_reparsed` is the number PROTOCOL.md's fixed
STRONG/MODERATE/WEAK thresholds actually gate on; both are reported in
BLOCKERS.md/REPORT_LIFE9.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402

B2_MODEL = "qwen2.5-coder-1.5b-instruct-q4_0"   # PROTOCOL.md P1

STRONG_THRESHOLD = 0.20      # PROTOCOL.md -- fixed, never moved after being computed
MODERATE_THRESHOLD = 0.05

_LIST_PREFIX_RE = re.compile(r"^\s*\d+[.)](?!\d)\s*")   # (?!\d): don't eat "13.1"'s "13." as if it were list-marker "13. "


def _strip_list_prefix(text: str) -> str:
    return "\n".join(_LIST_PREFIX_RE.sub("", ln) for ln in (text or "").splitlines())


def index_whole_rows(rows: list) -> dict:
    return {(r["task_id"], r["model"]): r for r in rows}


def whole_resolved(ds, row: dict, reparsed: bool) -> bool:
    task = ds.tasks[row["task_id"]]
    text = row.get("raw_output", "")
    if reparsed:
        text = _strip_list_prefix(text)
    res = LD.seams14.whole_seam(text, task["steps"])
    return res["status"] == "PASS"


def compute_alpha_b2(ds, whole_by_key: dict, u_dead: list) -> dict:
    n = len(u_dead)
    if n == 0:
        return {"n_u_dead": 0, "alpha_B2_raw": None, "alpha_B2_reparsed": None,
               "n_missing_whole_row": 0}
    n_raw = n_reparsed = n_missing = 0
    for t in u_dead:
        row = whole_by_key.get((t, B2_MODEL))
        if row is None:
            n_missing += 1
            continue
        if whole_resolved(ds, row, reparsed=False):
            n_raw += 1
        if whole_resolved(ds, row, reparsed=True):
            n_reparsed += 1
    return {"n_u_dead": n, "n_missing_whole_row": n_missing,
           "alpha_B2_raw": n_raw / n, "alpha_B2_reparsed": n_reparsed / n}


def compute_alpha_any_panel(ds, whole_by_key: dict, u_dead_panel: list, models: list) -> dict:
    """Diagnostic only -- the task's own thresholds gate on alpha_B2 alone,
    never alpha_any. Only computable where FULL 6-model whole-task
    coverage exists (train/panel; whole_grid.json has just 1 model's rows
    on test), so this is reported as a labeled panel-only number, not
    conflated with the test-set alpha_B2."""
    n = len(u_dead_panel)
    if n == 0:
        return {"n_u_dead_panel": 0, "alpha_any_reparsed": None}
    n_any = 0
    for t in u_dead_panel:
        for m in models:
            row = whole_by_key.get((t, m))
            if row is not None and whole_resolved(ds, row, reparsed=True):
                n_any += 1
                break
    return {"n_u_dead_panel": n, "alpha_any_reparsed": n_any / n}


def classify_alpha_b2(alpha_b2) -> str:
    if alpha_b2 is None:
        return "UNKNOWN"
    if alpha_b2 >= STRONG_THRESHOLD:
        return "STRONG"
    if alpha_b2 >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "WEAK"


def whole_b2_per_task_outcomes(ds, whole_by_key: dict, task_ids: list, reparsed: bool = True) -> dict:
    """{task_id: {"outcome": "RESOLVED"|"UNRESOLVED"}} -- B2_MODEL's own
    whole-task result per task, shaped for `arch2.fitness.resolved_set`/
    `paired_delta_r` reuse in Phase D."""
    out = {}
    for t in task_ids:
        row = whole_by_key.get((t, B2_MODEL))
        resolved = row is not None and whole_resolved(ds, row, reparsed=reparsed)
        out[t] = {"outcome": "RESOLVED" if resolved else "UNRESOLVED"}
    return out


def whole_b2_test_rate(ds, whole_by_key: dict, test_ids: list, reparsed: bool = True) -> float:
    """r_whole_B2 (Phase D's table): B2_MODEL's whole-task PASS rate over
    ALL of test -- already-existing data, n=100, zero new calls."""
    outcomes = whole_b2_per_task_outcomes(ds, whole_by_key, test_ids, reparsed=reparsed)
    n = len(test_ids)
    if n == 0:
        return 0.0
    return sum(1 for t in test_ids if outcomes[t]["outcome"] == "RESOLVED") / n
