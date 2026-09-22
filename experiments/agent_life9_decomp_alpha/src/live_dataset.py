"""live_dataset.py — LIFE-9 bootstrap + LiveDataset over the REUSED live
grid (copied byte-identical from `agent_life8_close_b3/metrics/live_grid/`,
md5-verified — same `GEN_SEED=150002`/`SPLIT_SEED=20260821` chain every
package since LIFE-3 shares).

Unlike every LIFE-3..8 package, this one builds NO ASSEMBLE genotypes and
runs NO `Evolution` — so it does not import `arch2` at all here (Phase D's
one significance check imports `arch2/fitness.py::paired_delta_r`
separately, a pure-statistics function with no genotype dependency).
`STEP_KINDS`/`MODEL_IDS`/prompt constants come directly from
`experiment14`/`experiment11`, not via `arch2/heterostep.py`'s re-export
(that module is not needed here).
"""

from __future__ import annotations

import hashlib
import importlib.util as _ilu
import json
import sys
from pathlib import Path
from typing import Optional

AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = AGENT_DIR.parent
EXP14 = ROOT / "experiment14"
EXP11_CONFIGS = ROOT / "experiment11" / "configs"


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tasks14 = _load_module("life9_tasks14", EXP14 / "tasks" / "heterostep.py")
seams14 = _load_module("life9_seams14", EXP14 / "seams.py")
mr11 = _load_module("life9_model_registry11", EXP11_CONFIGS / "model_registry.py")

# -- PROTOCOL.md P0: grid=reuse -- same values as agent_life3..8, intentional --
GEN_SEED = 150002
SPLIT_SEED = 20260821
N_TASKS = 200

LIVE_SEED_BASE = 6000      # matches the LIFE-3..8 grid-building base (unused for lookups here, kept for provenance)
LIVE_SEED_BASE_9 = 9000    # LIFE-9's OWN new-live-call base (PROTOCOL.md P4/P5) -- distinct from LIVE_SEED_BASE
                            # so a fresh Phase C call is never seed-identical to an existing grid cell.

STEP_KINDS = tasks14.STEP_KINDS            # ("READ", "FORMAT", "LOOKUP", "COMPUTE")
DECOMP_KINDS = ("READ", "LOOKUP", "COMPUTE")  # FORMAT excluded from Phase C planning (task's own instruction)
MODEL_IDS = mr11.MODEL_IDS

TEMPERATURE = 0.5
MAX_TOKENS_STEP = 60
MAX_TOKENS_PLANNER = 400

LIVE_GRID_DIR = AGENT_DIR / "metrics" / "live_grid"
LIVE_TRAIN_GRID = LIVE_GRID_DIR / "train_grid.json"
LIVE_TEST_GRID = LIVE_GRID_DIR / "test_grid.json"
LIVE_WHOLE_GRID = LIVE_GRID_DIR / "whole_grid.json"


def stable_hash(*parts) -> int:
    """Deterministic across processes -- NOT builtin `hash()` (randomized
    per-process for strings)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest, 16) % 10 ** 6


def draw_seed(task_id: str, kind: str, model_id: str) -> int:
    return LIVE_SEED_BASE + stable_hash(task_id, kind, model_id)


def draw_seed_9(task_id: str, role: str, model_id: str) -> int:
    """Phase C's own seed draw. `role`: "PLANNER" or "EXEC{i}"."""
    return LIVE_SEED_BASE_9 + stable_hash(task_id, role, model_id)


class LiveDataset:
    def __init__(self, train_path: Path = LIVE_TRAIN_GRID, test_path: Path = LIVE_TEST_GRID):
        self.tasks = tasks14.build_tasks(n=N_TASKS, seed=GEN_SEED)
        self.split = tasks14.split(list(self.tasks), seed=SPLIT_SEED)
        self.cells: dict = {}
        self._load_grid(train_path)
        self._load_grid(test_path)
        self.models = MODEL_IDS
        self.kinds = STEP_KINDS

    def _load_grid(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data["rows"]:
            self.cells[(r["task_id"], r["kind"], r["model"])] = r


_DATASET: Optional[LiveDataset] = None


def default_dataset(*, fresh: bool = False) -> LiveDataset:
    global _DATASET
    if _DATASET is None or fresh:
        _DATASET = LiveDataset()
    return _DATASET


def load_whole_grid_rows() -> list:
    with open(LIVE_WHOLE_GRID, encoding="utf-8") as f:
        return json.load(f)["rows"]
