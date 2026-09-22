"""BUDGET MANAGER — уровень 5 (API_BOUNDARIES.md §10).

Бюджет как объект (BUDGET_MODEL.md); обязательное сравнение стратегий.

Интерфейс (API_BOUNDARIES.md §10):
    estimate(action, model_profile, context) -> Cost
    reserve(cost) -> ok | DENIED
    commit(actual_cost)
    release(reserved)
    remaining() -> Cost
    compare_strategies([strategy]) -> [{strategy, cost, feasible}]

Заготовка на этапе 0. Реализация распределена по этапам, где
требуется учёт стоимости (начиная с этапа 4).
"""
