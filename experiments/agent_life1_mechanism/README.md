# agent_life1_mechanism

Пятый пакет поверх ASSEMBLE. Вопрос — не «какой маршрут лучше на 6
моделях HETEROSTEP», а какой универсальный механизм (сборка/перенос/
отбор) стабильно производит совместное покрытие независимых блоков.
Полный вопрос, определения `Imp`/A/AB, пороги корзин `MECH_*` и ветка
Phase B — в `PROTOCOL.md`.

Работает целиком на CPU поверх уже собранной живой сетки
`agent_a5_live_m/metrics/live_grid/` (только чтение, новых `generate()`
вызовов нет).

```
python scripts/run_audit.py          # Gate R1/R2 -> lineage_audit.json -> корзина MECH_*
python scripts/run_intervention.py   # только если корзина требует Phase B
```

Итог — `reports/REPORT_LIFE1.md`.

Изоляция: пишет только в `agent_life1_mechanism/`. `arch2/` — импортом без
правок; `agent_a5_live_m/` и прочие `agent_aN` — чтением/копированием кода
(изоляция пакетов), без правок.
