"""POLICY — уровень 4, заменяемая (API_BOUNDARIES.md §7).

В ARCH-1 — реализация RuleBasedPolicy по правилам REPAIR_MODEL.md §4.2.
Роутер, bandit и обучаемые политики сознательно не входят в ARCH-1
(WORK_PLAN.md §8).

Жёсткие ограничения (соблюдены буквально):
- нет доступа на запись в граф, состояние, evidence — select() и
  preferred_action_for_collision() ничего не пишут, только читают
  переданные аргументы;
- не может расширить множество кандидатов — select() всегда выбирает
  ИЗ candidate_actions, никогда не добавляет к нему.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .storage import Cost

# -- REPAIR_MODEL.md §4.2: тип коллизии -> предпочтительное действие -------
# Сила этих строк MODERATE/WEAK (n от 1 до 4 в исходных данных) -- это
# НАЧАЛЬНЫЕ правила, не законы (см. докстринг REPAIR_MODEL §4.2).

CHANGE_MODEL = "CHANGE_MODEL"
LOCAL_RETRY_AVOID = "AVOID_LOCAL_RETRY"
REGENERATE_NARROW = "REGENERATE_NARROW"
REGENERATE_STRICT_FORMAT = "REGENERATE_STRICT_FORMAT"
CHANGE_PROMPT = "CHANGE_PROMPT"
REQUEST_EXTERNAL_EVIDENCE = "REQUEST_EXTERNAL_EVIDENCE"
VERIFY = "VERIFY"

_COLLISION_TYPE_PREFERRED_ACTION = {
    "MECHANICAL_FAILURE": REGENERATE_NARROW,  # "внутреннее самопротиворечие" -- локальный REGENERATE (F24)
    "INTERNAL_INCONSISTENCY": REGENERATE_NARROW,
    "CROSS_ARTIFACT_INCONSISTENCY": REGENERATE_NARROW,
    "FORMAT_VIOLATION": REGENERATE_STRICT_FORMAT,  # F27: формат стабилен, сбои редки и локальны
    "SOURCE_DISAGREEMENT": CHANGE_PROMPT,  # слепая зона нескольких моделей -- F23
    "EVIDENCE_CONFLICT": LOCAL_RETRY_AVOID,  # "не локальный retry" -- F24: 4/4 ухудшал результат
    "DEPENDENCY_VIOLATION": VERIFY,  # STALE без собственного дефекта -- F37+F38
}


class RuleBasedPolicy:
    """RepairPolicy (REPAIR_MODEL.md §4.4)."""

    def preferred_action_for_collision(
        self, collision_type: str, reproducible_error: bool = False
    ) -> str:
        """REPAIR_MODEL.md §4.2, таблица.

        reproducible_error -- «устойчивая воспроизводимая ошибка (та же
        ошибка при повторе той же конфигурации)» -- перевешивает
        табличное правило по типу коллизии (CHANGE_MODEL, F22).
        """
        if reproducible_error:
            return CHANGE_MODEL
        return _COLLISION_TYPE_PREFERRED_ACTION.get(collision_type, REGENERATE_NARROW)

    def select(
        self,
        state_snapshot: Any,
        task: Any,
        candidate_actions: Iterable[dict],
        model_profiles: Optional[dict] = None,
        budget: Optional[Cost] = None,
    ) -> str:
        """REPAIR_MODEL.md §4.4.

        candidate_actions -- [{"strategy": ..., "cost": Cost, ...}, ...],
        обычно построенное RepairPlanner.build_plan(). Выбор: среди
        влезающих в budget -- дешевейшая по (tokens, time_sec); если
        budget не передан или ничего не влезает -- дешевейшая из всех
        предложенных (обязательное сравнение стратегий по-прежнему
        видно вызывающему через alternatives_considered, это не
        отбрасывание информации, а только выбор).
        """
        candidates = list(candidate_actions)
        if not candidates:
            raise ValueError("select: candidate_actions must not be empty")

        if budget is not None:
            feasible = [
                a for a in candidates if a.get("cost", Cost()).fits_in(budget)
            ]
        else:
            feasible = candidates

        pool = feasible or candidates
        best = min(pool, key=lambda a: (a.get("cost", Cost()).tokens, a.get("cost", Cost()).time_sec))
        return best["strategy"]
