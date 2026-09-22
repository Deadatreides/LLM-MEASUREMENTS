# INVENTORY.md — что здесь есть

Пакет A3 — чистый старт (в отличие от A2, здесь не было предшественника):
`agent_a3_mstar/` не существовал до этой сессии.

## Что скопировано из A2 (не импортировано — изоляция пакетов, PROTOCOL.md §1)

- `src/metrics_lib.py::build_oracle_space/extract_genotype_models_per_slot/
  genotype_has_atom/analyze_archive_coverage` — дословная копия формулы
  `agent_a2_assemble_budget/src/coverage.py`, определение НЕ изменено.
- `src/orchestrator.py::build_seeds_pop20/_greedy_order_local` — дословная
  копия состава стартовой популяции `agent_a2_assemble_budget/src/
  orchestrator.py::build_seeds_pop20` (полный B3 не вставляется, тот же
  запрет).
- `scripts/verify_baselines.py` — тот же гейт 0.4600/0.5800/0.7000
  (независимая копия, а не импорт из `agent_a2_assemble_budget/`).

## Что написано заново для A3

- `src/recombination.py::swap_assemble_slot` — в отличие от A2's
  `enhanced_m5_cross_slot` (равномерный выбор слота, фиксированная
  вероятность 0.40), здесь слот выбирается с уклоном `[1,1,2,2]`
  (READ/FORMAT/LOOKUP/COMPUTE), а вероятность применения — внешний параметр
  `p_cross`, не константа.
- `src/orchestrator.py::custom_reproduce` — параметризованная версия A2's
  `custom_reproduce_high_m5` (тот же цикл, `p_cross` вместо хардкода).
- `src/metrics_lib.py::evaluate_full_train/archive_verdicts` — новая часть,
  не имеющая аналога в A2: переоценка ВСЕГО архива на ПОЛНОЙ 100-задачной
  train-сетке против ФИКСИРОВАННЫХ канонических чисел B1/B2/B3, вместо
  снимка последнего поколения на ротирующемся control-срезе. Обоснование —
  `PROTOCOL.md` §5.1: это прямой ответ на находку честной перепроверки A2
  (V2/V3 сильно расходились между seed при снимковой методике).

## Дефект, найденный смоук-тестом ДО кампании и исправленный

См. `BLOCKERS.md` — `ev.archive` содержит посеянные эталоны, без исключения
archive-вердикты были бы тривиально истинны всегда. Исправлено до 10
реальных прогонов.

## Статус артефактов

```
agent_a3_mstar/
  PROTOCOL.md          complete — написан до первых цифр, дополнен поправкой
                        об исключении эталонов (найдено смоук-тестом, тоже
                        до кампании — не задним числом)
  README.md            complete
  BLOCKERS.md           complete
  INVENTORY.md           complete [этот файл]
  src/
    metrics_lib.py       complete — копия формулы A2 + archive_verdicts (новое)
    recombination.py      complete — swap_assemble_slot с уклоном по слоту
    orchestrator.py        complete — 10 прогонов, snapshot+archive метрики
  scripts/
    verify_baselines.py   complete — три эталона, 0.000000 отклонение (трижды подряд)
    run_all.py              complete — единая точка входа, живой прогресс (psutil)
  metrics/
    summary.json           complete — 10 прогонов
    curve_m.json             complete — J(p_cross), m*, плато, SNR
  reports/
    REPORT_A3.md             complete — написан отдельно (не автогенерируется
                              run_all.py, в отличие от A2 — все 8 разделов
                              задания требуют содержательной интерпретации
                              находки о SNR, которую скрипт сам не формулирует)
  runs/
    pcross0.05_s20261001/ ... pcross0.60_s20261002/   10 директорий, полные трассы
```

**Переиспользовать as-is**: всё содержимое `src/`, `scripts/` — прогнано от
начала до конца, дало осмысленные, не тривиальные числа (после исправления
дефекта с эталонами).

**Не осталось незавершённого/сломанного/мусора.**

`configs/` не создавалась — те же основания, что в A2 (5 уровней `p_cross`
инлайн в `scripts/run_all.py`, отдельный файл избыточен).
