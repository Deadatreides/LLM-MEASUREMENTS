"""model_registry.py — тонкая обёртка над `experiment11/configs/model_registry.py`.

Реестр моделей и низкоуровневые `load_model()`/`generate()` полностью
переиспользуются (TASK_EXPERIMENT12.md §14: «переиспользовать
проверенный стенд Эксперимента 11» -- 2790 вызовов без единой ошибки
конвейера). Загрузка по явному пути, не через `sys.path` + обычный
`import configs.model_registry` -- оба каталога (`experiment11/configs`,
`experiment12/configs`) называют свой подпакет `configs`, обычный импорт
после добавления обоих в sys.path мог бы получить не тот модуль (та же
коллизия имён, что решена в `experiment12/seams.py`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "experiment11" / "configs" / "model_registry.py"
_spec = importlib.util.spec_from_file_location("experiment11_model_registry", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MODEL_REGISTRY = _mod.MODEL_REGISTRY
MODEL_IDS = _mod.MODEL_IDS
N_CTX = _mod.N_CTX
load_model = _mod.load_model
generate = _mod.generate
