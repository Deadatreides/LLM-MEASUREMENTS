"""MODEL ADAPTER — уровень 4 (API_BOUNDARIES.md §6).

Тонкая обёртка над experiment9/configs/llm_client.py. Единственный
модуль, которому позволено знать о конкретной модели — но и он не
выносит суждений, только генерирует.

llm_client.py/env_fix.py/chat_render.py ИМПОРТИРУЮТСЯ, а не копируются
(в отличие от ast_checks.py на этапе 3): это не горстка чистых функций,
а работающая, машинно-специфичная инфраструктура (фикс CUDA DLL-путей,
per-family jinja2-шаблоны) — WORK_PLAN.md §0.3: «ничего из этого
переписывать не нужно». Импорт experiment9/configs в sys.path происходит
ЛЕНИВО, внутри конструктора ModelAdapter — просто `import
src.model_adapter` не требует установленного llama_cpp/CUDA.

Гарантии:
- generate() возвращает СЫРЬЁ, не парсит;
- не выносит суждений;
- сохранение RawResult обязательно на стороне вызывающего (ACTION
  EXECUTOR), ДО попытки разбора.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .storage import Sampling, now_iso

_EXPERIMENT9_CONFIGS = Path(__file__).resolve().parents[2] / "experiment9" / "configs"


def _llm_client():
    """Ленивый импорт experiment9/configs/llm_client.py (и его собственных
    зависимостей env_fix/chat_render) через вставку в sys.path."""
    path_str = str(_EXPERIMENT9_CONFIGS)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    import llm_client  # noqa: PLC0415 (намеренно ленивый импорт)

    return llm_client


@dataclass(frozen=True)
class Capabilities:
    model_id: str
    family: str
    context_capacity: int


@dataclass(frozen=True)
class RawResult:
    """Сырьё вызова (DATA_MODEL.md §12, поля RUN, относящиеся к самому
    вызову — без run_id/action_id, которые проставляет ACTION EXECUTOR)."""

    rendered_prompt: Optional[str]
    raw_output: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    wall_time_sec: float
    failed: bool
    error: Optional[str]
    timestamp: str = field(default_factory=now_iso)


class ModelAdapter:
    """MODEL ADAPTER (API_BOUNDARIES.md §6)."""

    def __init__(self, model_id: str) -> None:
        llm_client = _llm_client()
        self._model_id = model_id
        self._llm, self._model_entry, self._load_time_sec = llm_client.load_model(model_id)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            model_id=self._model_id,
            family=self._model_entry["family"],
            context_capacity=self._model_entry.get("context_length_train") or 4096,
        )

    def generate(self, content: str, sampling: Sampling) -> RawResult:
        """content — то, что собрал CONTEXT_BUILDER (ещё не прогнано через
        чат-шаблон). llm_client.generate() сам рендерит шаблон конкретной
        модели и делает low-level completion; итоговая ОТПРАВЛЕННАЯ строка
        попадает в RawResult.rendered_prompt (DATA_MODEL §12: «полный, как
        отправлено»)."""
        llm_client = _llm_client()
        raw = llm_client.generate(
            self._llm,
            self._model_id,
            self._model_entry,
            content,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            seed=sampling.seed,
            max_tokens=sampling.max_tokens,
        )
        return RawResult(
            rendered_prompt=raw["rendered_prompt"],
            raw_output=raw["raw_text"],
            input_tokens=raw["input_tokens"],
            output_tokens=raw["output_tokens"],
            wall_time_sec=raw["generation_time_sec"],
            failed=raw["generation_failed"],
            error=raw["generation_error"],
        )
