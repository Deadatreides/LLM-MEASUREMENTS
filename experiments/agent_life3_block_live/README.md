# agent_life3_block_live

Седьмой пакет поверх ASSEMBLE. Проверяет на ЖИВЫХ вызовах моделей (не
CPU-replay чужой сетки): (1) комплементарный slot-HGT (`agent_life2_
complementary`) на свежем сплите, (2) новую единицу — регистрируемый
БЛОК (слот+сигнатура) + `BLOCK_INSERT`, замыкающий автокатализ A1-A5.
Полный вопрос, R1-R4, корзины `UNIT_LIVE_*` — в `PROTOCOL.md`.

```
python scripts/verify_seams.py   # швы + ровно один смоук generate()
python scripts/build_grid.py     # живая сетка на новом сплите (~15-40 мин GPU)
python scripts/run_all.py        # lean-assert -> 6 WITH + 6 CTRL -> отчёт
```

Итог — `reports/REPORT_LIFE3_LIVE.md`.

Изоляция: пишет только в `agent_life3_block_live/`. `arch2/`,
`agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`
— чтением/копированием кода, без правок.
