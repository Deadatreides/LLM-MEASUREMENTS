# agent_a3_mstar

Пакет A3: калибровка силы cross-slot рекомбинации (`p_cross`, операционально
= m*) и проверка SNR отбора (снимок последнего поколения vs полный архив на
полной train-сетке) на ASSEMBLE/HETEROSTEP.

## Запуск единой командой

```bash
python agent_a3_mstar/scripts/run_all.py
```

### Порядок работы

1. `scripts/verify_baselines.py`: проверка B1=0.4600, B2=0.5800, B3=0.7000
   (r и c) на HETEROSTEP train grid.
2. 10 эволюционных прогонов: 5 уровней `p_cross` ∈ {0.05, 0.15, 0.30, 0.45,
   0.60} × 2 seed (20261001, 20261002), `start_pop=20`, до 24 поколений,
   `G_STALL=12`.
3. Для каждого прогона — snapshot-метрики (последнее поколение, control-срез,
   как в `agent_a2_assemble_budget`) И archive-метрики (переоценка всего
   архива на ПОЛНОЙ 100-задачной train-сетке против канонических чисел B1/
   B2/B3, исключая посеянные эталоны — см. `PROTOCOL.md` §5.1).
4. `metrics/summary.json` (сырые 10 прогонов), `metrics/curve_m.json`
   (J(p_cross), m*, плато, SNR-сравнение).
5. Отчёт — `reports/REPORT_A3.md` (пишется отдельно после `run_all.py`, не
   автогенерируется скриптом, в отличие от A2 — см. `INVENTORY.md`).

Полный прогон занимает ~5-6 минут (0 GPU, offline replay).
