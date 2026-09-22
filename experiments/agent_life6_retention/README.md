# agent_life6_retention

Десятый пакет поверх ASSEMBLE. LIFE-5 нашла `MAP_STABILITY_GAP` на
READ/LOOKUP/COMPUTE: Imp=1-сигнатур много, но `median_L≈1` почти везде
— носитель редко живёт дольше одного поколения. LIFE-6 проверяет два
рычага удержания за ОДИН проход, с ветвлением внутри `run_all.py` и без
возврата к пользователю: Branch A (slot-safe mutate — не дать мутации
портить уже-хороший слот) первым; если он не сработал — Branch B
(carrier soft-immunity — на время освободить нескольких молодых
Imp=1-носителей от смерти популяции). Ни block registry, ни
BLOCK_INSERT, ни R1-R4, ни A1-A5 здесь нет вообще — этаж «блок» не
возобновлялся. Полная спека, пороги корзин — в `PROTOCOL.md`.

```
python scripts/verify_seams.py   # целостность копии сетки + slot-safe фикстуры
python scripts/run_all.py        # smoke -> Branch A -> ветвление (A2|A3[->B]|B) -> REPORT_LIFE6.md
```

Итог — `reports/REPORT_LIFE6.md`: **`RET_A_FAIL`** -> Branch B ->
**`RET_B_FAIL`** — ни один рычаг не сдвинул `median_L_imp`. Оба
механизма независимо подтверждены рабочими на уровне юнит-фикстур ДО
кампании; forensic-разбор после кампании (§3.1/§5.1 отчёта,
BLOCKERS.md) нашёл правдоподобные причины для обоих null-результатов:
Branch A защищает межпоколенческое наследование слота, не жизнь самого
носителя (метрика и рычаг — о разном); Branch B защищал ≈86% всех
Imp=1-носителей хотя бы раз, поднял СРЕДНИЙ лайфспан (2.41 vs 1.57), но
не медиану — правило "самый молодой" делает защиту временной, теряемой
за 1-2 поколения.

Изоляция: пишет только в `agent_life6_retention/`. `arch2/`,
`agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`,
`agent_life3_block_live/`, `agent_life4_block_fix/`,
`agent_life5_slot_map/` — чтением/копированием кода и данных сетки
(md5-проверена), без правок.
