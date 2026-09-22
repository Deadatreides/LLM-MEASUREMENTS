# agent_life4_block_fix

Восьмой пакет поверх ASSEMBLE. Не новый эксперимент — точечный фикс двух
конкретных, раскрытых в LIFE-3 дефектов: (1) `metrics_lib.slot_models`
не видела содержимое за `cx.`-composite-CALL (единственный
зарегистрированный блок в LIFE-3 имел неполную сигнатуру), (2) окно
вымирания (~gen 12) вероятно не давало R1 накопить 3 носителя. Тот же
WITH/CTRL/6-seed дизайн, что LIFE-3, на ПЕРЕИСПОЛЬЗОВАННОЙ (byte-identical
скопированной, 0 новых `generate()`) живой сетке LIFE-3. Полный вопрос,
R1-R4 (не изменены), корзины `UNIT4_*` — в `PROTOCOL.md`.

```
python scripts/verify_seams.py   # швы + целостность копии сетки + фикстура разворота cx
python scripts/run_all.py        # verify -> smoke -> 6 WITH + 6 CTRL -> форензик -> отчёт
```

Итог — `reports/REPORT_LIFE4.md`: вердикт `UNIT4_FAIL` (A1-A5 на 1/6,
порог ≥5/6 не достигнут), плюс важная forensic-находка (§4.2/BLOCKERS.md):
единственный успех LIFE-4 повторно нашёл байт-в-байт тот же композит,
что LIFE-3 нашла один раз — не независимая удача, а воспроизводимый
аттрактор на LOOKUP-слоте.

Изоляция: пишет только в `agent_life4_block_fix/`. `arch2/`,
`agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`,
`agent_life3_block_live/` — чтением/копированием кода и данных сетки/
фикстуры (md5-проверено), без правок.
