# agent_life2_complementary

Шестой пакет поверх ASSEMBLE. Проверяет ОДИН рычаг, найденный
`agent_life1_mechanism/reports/REPORT_LIFE1.md`: заменяет
similarity-донора у cross-slot переноса на комплементарный-по-покрытию, на
бедном (не заготовленном под совместное покрытие) старте. Полный вопрос,
определения `Imp`/A/AB/`N_AB_first_assembly`, пороги `COMP_*` — в
`PROTOCOL.md`.

Работает целиком на CPU поверх уже собранной живой сетки
`agent_a5_live_m/metrics/live_grid/` (только чтение, новых `generate()`
вызовов нет).

```
python scripts/run_all.py   # lean-assert -> smoke -> 24 клетки (3 p x 8 seed) -> отчёт
```

Итог — `reports/REPORT_LIFE2.md`.

Изоляция: пишет только в `agent_life2_complementary/`. `arch2/` — импортом
без правок; `agent_a5_live_m/`, `agent_life1_mechanism/` — чтением/копированием
кода (изоляция пакетов), без правок.
