# agent_life8_close_b3

Двенадцатый пакет поверх ASSEMBLE. LIFE-3..7 подтвердили комплементарный
slot-HGT + stall-from-improvement как default, но ни разу не сверили
итоговую силу маршрута `r` с каноническими B1/B2/B3 (самый повторяемый
открытый пункт памяти проекта после LIFE-7). LIFE-8 закрывает это одним
проходом: потолок (B1/B2/B3/UNION, до какого-либо evolve) → организм
(фиксированный по P5, 8 seed) → склейка (17 archive-only донора из карты
LIFE-5, если язык богат) → добор моделей (если язык беден или склейка не
помогла). Одно ветвление внутри `run_all.py`, без возврата к пользователю
после Phase 1. Полная спека, пороги корзин — в `PROTOCOL.md`.

```
python scripts/verify_seams.py   # целостность копии сетки + glue/stall-фикстуры + AST-проверка
python scripts/run_all.py        # ceiling -> F1(8) -> [F2(6)] -> [F3: expand] -> REPORT_LIFE8.md
```

Итог — `reports/REPORT_LIFE8.md`: потолок **LANG_RICH** (`GAP_LANG=0.060`)
→ Phase 1 **`F1_MID`** (`R=0.456`, никогда не достиг B3) → Phase 2
(склейка) **`F2_NO`** (`R2=0.447`, склейка не подняла результат) → Phase 3
**`CLOSE_NEED_MODELS`** (реестр `experiment11` подтверждён без кандидатов
сверх текущих 6) → корзина **`B3_STACK`**. Forensic: лучший генотип
Phase 1 решает на test строгое ПОДМНОЖЕСТВО того, что решает B3 (49 общих,
0 уникальных для организма) — организм структурно не выходит за пределы
простого greedy-cover, даже находя иные сборки.

Изоляция: пишет только в `agent_life8_close_b3/`. `arch2/`,
`agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`,
`agent_life3_block_live/`, `agent_life4_block_fix/`,
`agent_life5_slot_map/`, `agent_life6_retention/`, `agent_life7_selection/`
— чтением/копированием кода и данных (сетка md5-проверена,
`agent_life5_slot_map/metrics/per_slot_signatures.json` только читается
для Phase 2), без правок.
