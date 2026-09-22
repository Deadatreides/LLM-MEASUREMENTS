# agent_a2_assemble_budget

Пакет A2: бюджет эволюции × рекомбинация × Coverage на ASSEMBLE/HETEROSTEP.

## Инструкция по запуску единой командой

```bash
python agent_a2_assemble_budget/scripts/run_all.py
```

### Порядок работы:
1. `scripts/verify_baselines.py`: проверка базовых линий B1=0.4600, B2=0.5800, B3=0.7000 на HETEROSTEP train grid.
2. Построение `metrics/oracle_atoms.json` (атомы $U$ и пары $P$).
3. Прогон 8 ячеек факторов (2 seeds на ячейку = 16 прогонов).
4. Сводка `metrics/summary.json`.
5. Генерация отчёта `reports/REPORT_A2.md`.

## Честная перепроверка (3-й seed) и разбивка по факторам

После первого прохода проведена личная перепроверка на независимом seed
(20260903, не пересекается с 20260901/20260902 — гарантированно свежее
вычисление, не резюмируемый кэш) и разбивка результатов по факторам.
Детали и находки — `INVENTORY.md` §0, `reports/REPORT_A2.md` §2/§5.2.

```bash
python agent_a2_assemble_budget/scripts/verify_run_seed3.py   # ~4 мин, 8 ячеек, seed 20260903
python agent_a2_assemble_budget/src/robustness_analysis.py     # секунды, слияние 24 прогонов + разбивка по факторам
```

Первая команда пишет в `runs/C*_s20260903/` и `metrics/seed3_memory_log.json`
(включает живые замеры памяти/времени). Вторая читает `metrics/summary.json`
+ `metrics/seed3_memory_log.json` и пишет `metrics/factor_breakdown.json` —
чистый анализ уже посчитанных данных, эволюция не перезапускается.
