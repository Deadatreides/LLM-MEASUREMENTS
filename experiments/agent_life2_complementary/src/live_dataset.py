"""live_dataset.py — bootstrap пакета + LiveDataset поверх УЖЕ СОБРАННОЙ
живой сетки `agent_a5_live_m/` (только чтение, PROTOCOL.md §0). Копия (не
импорт — изоляция пакетов agent_aN) `agent_life1_mechanism/src/
live_dataset.py` — не потребовалось изменений: `AGENT_DIR` авторазрешается
относительно этого файла, `ROOT`/`A5_DIR` остаются корнем trace-probe и
`agent_a5_live_m/` независимо от того, в каком пакете лежит копия.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path
from typing import Optional

AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = AGENT_DIR.parent
ARCH2 = ROOT / "arch2"
EXP14 = ROOT / "experiment14"
A5_DIR = ROOT / "agent_a5_live_m"

for _p in (ROOT, ARCH2):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import runner as R              # noqa: E402  arch2/runner.py, unchanged
import genotype as G            # noqa: E402  arch2/genotype.py, unchanged
import mutate as M              # noqa: E402  arch2/mutate.py, unchanged
import evolve as E              # noqa: E402  arch2/evolve.py, unchanged
import fitness as F             # noqa: E402  arch2/fitness.py, unchanged
import heredity as H            # noqa: E402  arch2/heredity.py, unchanged
import heterostep as HSTEP      # noqa: E402  arch2/heterostep.py, unchanged
import heterostep_seeds as HSEED  # noqa: E402  arch2/heterostep_seeds.py -- .route/._slot/._fallback
                                   # reused; ._greedy_order NOT reused (hash-order defect) --
                                   # deterministic base route built locally in lean_seeds.py


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tasks14 = _load_module("life2_tasks14", EXP14 / "tasks" / "heterostep.py")
seams14 = _load_module("life2_seams14", EXP14 / "seams.py")

# -- те же seed'ы, что A5/LIFE-1 (PROTOCOL.md §0: тот же split) --
GEN_SEED = 150001
SPLIT_SEED = 20260820
N_TASKS = 200

TEMPERATURE = HSTEP.TEMPERATURE
MAX_TOKENS_STEP = HSTEP.MAX_TOKENS_STEP
PROMPT_VARIANT = HSTEP.PROMPT_VARIANT

# -- живая сетка A5, ТОЛЬКО ЧТЕНИЕ (PROTOCOL.md §0) --
LIVE_GRID_DIR = A5_DIR / "metrics" / "live_grid"
LIVE_TRAIN_GRID = LIVE_GRID_DIR / "train_grid.json"
LIVE_TEST_GRID = LIVE_GRID_DIR / "test_grid.json"


class LiveDataset:
    """Идентична `agent_a5_live_m.src.live_dataset.LiveDataset`/
    `agent_life1_mechanism.src.live_dataset.LiveDataset` по форме
    (`.tasks`/`.split`/`.cells`/`.models`/`.kinds`/`.initial_state`/
    `.coverage`) -- этот пакет тоже не строит сетку, только читает."""

    def __init__(self, train_path: Path = LIVE_TRAIN_GRID, test_path: Path = LIVE_TEST_GRID):
        self.tasks = tasks14.build_tasks(n=N_TASKS, seed=GEN_SEED)
        self.split = tasks14.split(list(self.tasks), seed=SPLIT_SEED)

        self.cells: dict = {}
        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} не найден -- agent_a5_live_m должен быть прогнан первым "
                "(PROTOCOL.md §0: этот пакет читает, не строит, живую сетку)")
        self._load_grid(train_path)
        self.has_test = test_path.exists()
        if self.has_test:
            self._load_grid(test_path)

        self.models = HSTEP.MODEL_IDS
        self.kinds = HSTEP.STEP_KINDS

    def _load_grid(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data["rows"]:
            self.cells[(r["task_id"], r["kind"], r["model"])] = r

    def initial_state(self, task_id: str) -> R.State:
        task = self.tasks[task_id]
        return R.State(task_id=task_id, task_family="heterostep", task=task,
                       origin_model=None, artifact="", evidence=[])

    state_for = initial_state

    def coverage(self, task_ids, kinds=None, models=None) -> dict:
        kinds = kinds or self.kinds
        models = models or self.models
        total = len(task_ids) * len(kinds) * len(models)
        found = sum(1 for t in task_ids for k in kinds for m in models
                   if (t, k, m) in self.cells)
        return {"total": total, "found": found,
                "coverage": found / total if total else 0.0}


_DATASET: Optional[LiveDataset] = None


def default_dataset(*, fresh: bool = False) -> LiveDataset:
    global _DATASET
    if _DATASET is None or fresh:
        _DATASET = LiveDataset()
    return _DATASET
