# agent_a5_live_m

Четвёртый пакет поверх ASSEMBLE (`arch2/`, 5-й цикл). В отличие от
`agent_a2_assemble_budget/`, `agent_a3_mstar/`, `agent_a4_m_plateau/` (все —
offline-replay поверх `experiment14/runs14/train_grid.json`), этот пакет
делает НАСТОЯЩИЕ живые вызовы моделей через `experiment11/configs/
model_registry.py` на свежем независимом сплите задач.

Полная спецификация — `PROTOCOL.md`. Порядок работы:

```
python scripts/verify_seams.py    # шаг 0-1: швы + ровно один смоук-вызов generate()
python scripts/run_all.py         # шаги 2-6: живая сетка -> baselines -> 5x evolve -> отчёт
```

Изоляция: пишет только в `agent_a5_live_m/`. Читает `arch2/*`/
`experiment14/*`/`experiment11/*` только импортом (без правок) или как
образец для копирования (правило изоляции `agent_aN` пакетов друг от
друга — копирование, не импорт).

Итог — `reports/REPORT_LA5.md`.
