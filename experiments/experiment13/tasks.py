"""tasks.py — переиспользует датасет задач Эксперимента 12 без изменений
(задание: "использовать преимущественно те же многошаговые задачи... не
менять датасет без необходимости"). Явная importlib-загрузка по пути --
тот же приём, что уже применён для `seams.py`/`configs/model_registry.py`
между экспериментами 11/12/13 (оба модуля физически лежат в каталогах
`experiment12/tasks/`, обычный `import tasks` после добавления обоих
каталогов в sys.path нашёл бы не тот пакет)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_EXP12_TASKS = Path(__file__).resolve().parents[1] / "experiment12" / "tasks"

_spec_arith = importlib.util.spec_from_file_location("exp12_arithmetic_multistep_tasks_for_exp13", _EXP12_TASKS / "arithmetic_multistep_tasks.py")
_arith_mod = importlib.util.module_from_spec(_spec_arith)
_spec_arith.loader.exec_module(_arith_mod)

_spec_code = importlib.util.spec_from_file_location("exp12_code_tasks_for_exp13", _EXP12_TASKS / "code_tasks.py")
_code_mod = importlib.util.module_from_spec(_spec_code)
_spec_code.loader.exec_module(_code_mod)

ARITH_TASKS = _arith_mod.TASKS
ARITH_TASK_IDS = _arith_mod.TASK_IDS
CODE_TASKS = _code_mod.CODE_TASKS
CODE_TASK_IDS = _code_mod.CODE_TASK_IDS
