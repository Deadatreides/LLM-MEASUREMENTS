"""live_dataset.py — bootstrap пакета + LiveDataset поверх ПЕРЕИСПОЛЬЗОВАННОЙ
живой сетки (LIFE-7 PROTOCOL.md §0: grid=reuse -- скопирована byte-identical
из `agent_life6_retention/metrics/live_grid/` (сама переиспользована из
`agent_life5_slot_map` <- `agent_life4_block_fix` <- `agent_life3_block_live`),
md5 проверен до первого evolve, см. BLOCKERS.md). ТОТ ЖЕ сплит/seed, что
LIFE-3..6 (GEN_SEED=150002/SPLIT_SEED=20260821) -- намеренно: тот же грид,
не новый. Копия (не импорт — изоляция пакетов) `agent_life6_retention/src/
live_dataset.py`'s узора, путь к сетке указывает на СОБСТВЕННУЮ
(скопированную) директорию этого пакета.
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
ARCH2 = ROOT / "arch2"
EXP14 = ROOT / "experiment14"
EXP11_CONFIGS = ROOT / "experiment11" / "configs"

for _p in (ROOT, ARCH2):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import runner as R              # noqa: E402  arch2/runner.py, unchanged
import genotype as G            # noqa: E402  arch2/genotype.py, unchanged
import mutate as M              # noqa: E402  arch2/mutate.py, unchanged
import evolve as E              # noqa: E402  arch2/evolve.py, unchanged
import fitness as F             # noqa: E402  arch2/fitness.py, unchanged
import heredity as H            # noqa: E402  arch2/heredity.py, unchanged
import heterostep as HSTEP      # noqa: E402  arch2/heterostep.py -- MODEL_IDS/STEP_KINDS/
                                   # PROMPT_VARIANT/TEMPERATURE/MAX_TOKENS_STEP/HeterostepBackend/
                                   # Registry reused unchanged; .Dataset NOT reused (hardcodes
                                   # OLD seeds) -- see LiveDataset below.
import heterostep_seeds as HSEED  # noqa: E402  .route/._slot/._fallback reused;
                                   # ._greedy_order NOT reused (hash-order defect)


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tasks14 = _load_module("life7_tasks14", EXP14 / "tasks" / "heterostep.py")
seams14 = _load_module("life7_seams14", EXP14 / "seams.py")
mr11 = _load_module("life7_model_registry11", EXP11_CONFIGS / "model_registry.py")

# -- LIFE-7 PROTOCOL.md: grid=reuse -- ТЕ ЖЕ значения, что agent_life3..6,
# намеренно (тот же грид переиспользуется, не новый сплит) --
GEN_SEED = 150002
SPLIT_SEED = 20260821
N_TASKS = 200
LIVE_SEED_BASE = 6000

TEMPERATURE = HSTEP.TEMPERATURE            # 0.5, переиспользовано
MAX_TOKENS_STEP = HSTEP.MAX_TOKENS_STEP    # 60
MAX_TOKENS_WHOLE = 200
PROMPT_VARIANT = HSTEP.PROMPT_VARIANT      # "step"

LIVE_GRID_DIR = AGENT_DIR / "metrics" / "live_grid"
LIVE_TRAIN_GRID = LIVE_GRID_DIR / "train_grid.json"
LIVE_TEST_GRID = LIVE_GRID_DIR / "test_grid.json"
LIVE_WHOLE_GRID = LIVE_GRID_DIR / "whole_grid.json"


def stable_hash(task_id: str, kind: str, model_id: str) -> int:
    """Детерминировано между прогонами -- НЕ встроенный Python hash()
    (рандомизирован по процессу для строк)."""
    digest = hashlib.sha256(f"{task_id}|{kind}|{model_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 10 ** 6


def draw_seed(task_id: str, kind: str, model_id: str) -> int:
    return LIVE_SEED_BASE + stable_hash(task_id, kind, model_id)


class LiveDataset:
    """Аналог `agent_life3_block_live.src.live_dataset.LiveDataset`, тот
    же сплит/сетка (скопирована, не перестроена -- grid=reuse)."""

    def __init__(self, train_path: Path = LIVE_TRAIN_GRID, test_path: Path = LIVE_TEST_GRID):
        self.tasks = tasks14.build_tasks(n=N_TASKS, seed=GEN_SEED)
        self.split = tasks14.split(list(self.tasks), seed=SPLIT_SEED)

        self.cells: dict = {}
        self._train_path, self._test_path = train_path, test_path
        if train_path.exists():
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

    def reload_grid(self) -> None:
        self.cells = {}
        if self._train_path.exists():
            self._load_grid(self._train_path)
        self.has_test = self._test_path.exists()
        if self.has_test:
            self._load_grid(self._test_path)

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


SPLIT_JSON_PATH = AGENT_DIR / "metrics" / "split.json"


def write_split_json(ds: "LiveDataset", path: Path = SPLIT_JSON_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "n_tasks": N_TASKS, "gen_seed": GEN_SEED, "split_seed": SPLIT_SEED,
            "live_seed_base": LIVE_SEED_BASE,
            "train": ds.split["train"], "test": ds.split["test"],
        }, f, ensure_ascii=False, indent=2)
    return path
