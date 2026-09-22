# agent_life7_selection

Одиннадцатый пакет поверх ASSEMBLE. LIFE-6 (slot-safe mutate, carrier
immunity) — оба `FAIL` по `median_L`, но с forensic-объяснением, а не
поломкой. LIFE-7 меняет слой ПОД retention-рычагами: (A1) КТО умирает
(Pareto-доминирование по `(r, n_imp)` поверх обычных D1/D2, вместо
чисто экономической смерти) и (A2) КОГДА объявляется stall (улучшение
best_r/best_n_imp за прогон, не победа над B1/B2/B3 gate). Одно
ветвление внутри `run_all.py`, без возврата к пользователю. Полная
спека, пороги корзин — в `PROTOCOL.md`.

```
python scripts/verify_seams.py   # целостность копии сетки + Pareto/stall юнит-фикстуры
python scripts/run_all.py        # smoke -> Branch A -> ветвление (A2|A3[->B]|B) -> REPORT_LIFE7.md
```

Итог — `reports/REPORT_LIFE7.md`: **`SEL_A_PARTIAL`** → Phase A3
(vector-Pareto) → **`SEL_A3_FAIL`** → Branch B → **`SEL_B_WORKS`**.
Pareto-смерть дала самый сильный retention-эффект во всей линии
LIFE-6/7 (mean_L до 2.7x, p90_L до 6.4x, проверено согласованным по
всем 6 seed), но ценой first_assembly — тем сильнее, чем строже защита
(scalar → мягкая просадка → PARTIAL; vector → резкая просадка → FAIL).
Stall-from-improvement БЕЗ Pareto-смерти дал меньший, но "бесплатный"
рост (first_assembly даже вырос) — другим механизмом (продлевает весь
прогон, не защищает конкретных носителей) — итоговый default пакета.

Изоляция: пишет только в `agent_life7_selection/`. `arch2/`,
`agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`,
`agent_life3_block_live/`, `agent_life4_block_fix/`,
`agent_life5_slot_map/`, `agent_life6_retention/` — чтением/копированием
кода и данных сетки (md5-проверена), без правок.
