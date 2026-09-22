"""live_dataset.py — общий bootstrap пакета + LiveDataset (PROTOCOL.md §2/§3).

Единственное место, где загружаются: arch2 (импортом, без правок),
experiment14/tasks/heterostep.py + experiment14/seams.py + experiment11/
configs/model_registry.py (собственной загрузкой файла, тем же приёмом
importlib, что уже применяет arch2/heterostep.py -- НЕ импортом experiment14
как пакета, там нет __init__.py). Остальные модули этого пакета делают
`import live_dataset as LD` и используют `LD.tasks14`/`LD.seams14`/`LD.mr11`/
`LD.HSTEP`/`LD.HSEED`, а не грузят файлы повторно.

LiveDataset — единственное необходимое отличие от `arch2.heterostep.Dataset`:
тот вызывает `_tasks14.build_tasks()`/`split(...)` БЕЗ аргументов (проверено
чтением arch2/heterostep.py:84-85), то есть всегда восстанавливает
ОРИГИНАЛЬНЫЙ пул 140001/20260819 независимо от того, какой grid-файл ему
передан. Здесь -- НОВЫЙ пул/сплит (GEN_SEED=150001, SPLIT_SEED=20260820).
Всё остальное (форма `.cells`/`.models`/`.kinds`/`.initial_state`/
`.coverage`) идентично `arch2.heterostep.Dataset`, поэтому
`HeterostepBackend`/`heterostep_seeds` (импортируются БЕЗ изменений)
работают с этим объектом как с обычным Dataset -- они читают только
`.models`/`.cells`, никогда не проверяют тип.
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
import heterostep as HSTEP      # noqa: E402  arch2/heterostep.py, unchanged
import heterostep_seeds as HSEED  # noqa: E402  arch2/heterostep_seeds.py, unchanged


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Собственная загрузка -- experiment14 не пакет (нет __init__.py), тот же
# приём importlib, что arch2/heterostep.py уже применяет к тем же файлам.
tasks14 = _load_module("a5_tasks14", EXP14 / "tasks" / "heterostep.py")
seams14 = _load_module("a5_seams14", EXP14 / "seams.py")
mr11 = _load_module("a5_model_registry11", EXP11_CONFIGS / "model_registry.py")

# -- PROTOCOL.md §2 -----------------------------------------------------------------
GEN_SEED = 150001
SPLIT_SEED = 20260820
N_TASKS = 200

# -- PROTOCOL.md §2/§3 -- переиспользованы из arch2.heterostep, чтобы CALL-параметры
# генотипов (heterostep_seeds._slot использует H.TEMPERATURE и т.д.) СОВПАДАЛИ с тем,
# что live_grid_builder.py реально запрашивает у generate() -- иначе HeterostepBackend
# ._lookup молча возвращал бы "not measured" на каждой ячейке.
TEMPERATURE = HSTEP.TEMPERATURE            # 0.5
MAX_TOKENS_STEP = HSTEP.MAX_TOKENS_STEP    # 60
PROMPT_VARIANT = HSTEP.PROMPT_VARIANT      # "step"
MAX_TOKENS_WHOLE = 200
LIVE_SEED_BASE = 5000

LIVE_GRID_DIR = AGENT_DIR / "metrics" / "live_grid"
LIVE_TRAIN_GRID = LIVE_GRID_DIR / "train_grid.json"
LIVE_TEST_GRID = LIVE_GRID_DIR / "test_grid.json"
LIVE_WHOLE_GRID = LIVE_GRID_DIR / "whole_grid.json"


def stable_hash(task_id: str, kind: str, model_id: str) -> int:
    """PROTOCOL.md §3: детерминировано между прогонами -- НЕ встроенный Python
    hash() (рандомизирован по процессу для строк, PYTHONHASHSEED)."""
    digest = hashlib.sha256(f"{task_id}|{kind}|{model_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 10 ** 6


def draw_seed(task_id: str, kind: str, model_id: str) -> int:
    return LIVE_SEED_BASE + stable_hash(task_id, kind, model_id)


class LiveDataset:
    """Аналог arch2.heterostep.Dataset поверх НОВОГО пула/сплита и
    agent_a5_live_m/metrics/live_grid/ (не experiment14/runs14/)."""

    def __init__(self, train_path: Path = LIVE_TRAIN_GRID, test_path: Path = LIVE_TEST_GRID):
        self.tasks = tasks14.build_tasks(n=N_TASKS, seed=GEN_SEED)
        self.split = tasks14.split(list(self.tasks), seed=SPLIT_SEED)

        self.cells: dict = {}
        self._train_path = train_path
        self._test_path = test_path
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
        """Перечитать live_grid/*.json с диска -- вызывается после того, как
        live_grid_builder.py дописал ячейки В ТЕКУЩЕМ процессе построения."""
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
    """PROTOCOL.md §0/§2: split.json зафиксирован до генерации живой сетки.
    Идемпотентно -- безопасно вызывать повторно (train/test не меняются,
    т.к. GEN_SEED/SPLIT_SEED фиксированы)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "n_tasks": N_TASKS, "gen_seed": GEN_SEED, "split_seed": SPLIT_SEED,
            "live_seed_base": LIVE_SEED_BASE,
            "train": ds.split["train"], "test": ds.split["test"],
        }, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ds = default_dataset()
    print(f"задач: {len(ds.tasks)} (train {len(ds.split['train'])}, test {len(ds.split['test'])})")
    print(f"ячеек live_grid уже на диске: {len(ds.cells)}")
    path = write_split_json(ds)
    print(f"split.json записан: {path}")
