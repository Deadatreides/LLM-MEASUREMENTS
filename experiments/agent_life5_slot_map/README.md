# agent_life5_slot_map

Девятый пакет поверх ASSEMBLE. Не чинит block registry снова (LIFE-3/4:
`UNIT_LIVE_FAIL`/`UNIT4_FAIL`, 1/6 оба раза). Вместо этого: (Phase A)
фиксирует операционный дефолт — ASSEMBLE + комплементарный slot-HGT,
БЕЗ registry вообще — и (Phase B) строит карту Imp-сигнатур по слоту
(READ/FORMAT/LOOKUP/COMPUTE) поверх 24 уже существующих архивов
LIFE-3/LIFE-4, плюс (Phase B.2, подтверждено пользователем) 4 новых
seed чистого default-канала как независимый кросс-чек. Полный вопрос,
корзины `MAP_*` — в `PROTOCOL.md`.

```
python scripts/verify_seams.py       # швы + целостность копии сетки + unfold-фикстура
python scripts/run_all.py            # verify -> smoke -> optional evolve (4 seed) ->
                                      # карта (24+4) -> корзины -> REPORT_LIFE5.md
```

Итог — `reports/REPORT_LIFE5.md`: вердикт **`MAP_STABILITY_GAP`** на
ВСЕХ трёх слотах (READ/LOOKUP/COMPUTE), не `MAP_LOOKUP_ONLY`, вопреки
ожиданию из LIFE-3/4 — карта показала обильную, широко воспроизводимую
Imp-структуру на каждом слоте (не дефицит покрытия), но `median_L≈1`
почти везде: хорошие структуры создаются легко, носители просто не
задерживаются в популяции. Независимо подтверждено 4-seed кросс-чеком с
нулём registry-кода.

Изоляция: пишет только в `agent_life5_slot_map/`. `arch2/`,
`agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`,
`agent_life3_block_live/`, `agent_life4_block_fix/` — чтением/
копированием кода и данных (архивы life3/4 read-only; сетка
скопирована, md5-проверена), без правок.
