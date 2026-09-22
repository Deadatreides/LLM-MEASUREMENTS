"""ARCH-1: вертикальный срез архитектуры Мицелия (arch0/ARCHITECTURE.md).

Пакет собран по слоям API_BOUNDARIES.md §1, зависимости строго
однонаправленны сверху вниз:

    6 api / orchestrator
    5 repair_planner | action_executor | budget_manager
    4 policy | model_adapter
    3 seam_engine | evidence_engine
    2 state_engine
    1 graph_core
    0 storage

Модуль уровня N зависит только от уровней < N. Ни один модуль здесь
не импортирует конкретную реализацию LLM (см. arch1/CLAUDE.md).
"""
