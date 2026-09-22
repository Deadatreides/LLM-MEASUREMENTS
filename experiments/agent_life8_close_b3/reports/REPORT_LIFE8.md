# REPORT_LIFE8 — closing B3: ceiling -> organism -> glue -> language

Спека: `PROTOCOL.md` (P1-P7). Организм зафиксирован (P5): HGT (`p=0.35`) + stall-from-improvement, обычные D1/D2 -- Pareto/GATE/registry не сравнивались. Живая сетка ПЕРЕИСПОЛЬЗОВАНА byte-identical из `agent_life7_selection` (md5 подтверждён), 0 новых generate() на исходных 6 моделях.

## 1. P1-P7 и что не гоняли

P1-P7 -- см. PROTOCOL.md (самоопределены из структуры задания, не найден готовый черновик). Не гоняли: сетку p_slot_hgt, Pareto-survival, gate-based stall, slot-safe mutate, carrier immunity, block registry, brute-force 1296-комбинаций.

## 2. Потолок (Phase 0)

| | r_panel | r_test |
|---|---|---|
| B1 | 0.312 | 0.270 |
| B2 | 0.525 | 0.460 |
| B3 | 0.613 | 0.510 |
| UNION | 0.613 | 0.570 |

GAP_LANG = 0.060, GAP_STACK = 0.050 (порог 0.05) -> **LANG_RICH**

## 3. Phase 1: организм на 6 моделях, 8 seed

| seed | n_gen | extinct | best_panel_r | best_test_r |
|---|---|---|---|---|
| 20262201 | 26 | True | 0.525 | 0.440 |
| 20262202 | 32 | False | 0.613 | 0.490 |
| 20262203 | 32 | False | 0.562 | 0.470 |
| 20262204 | 28 | True | 0.525 | 0.440 |
| 20262205 | 32 | False | 0.575 | 0.470 |
| 20262206 | 32 | False | 0.525 | 0.440 |
| 20262207 | 29 | True | 0.525 | 0.440 |
| 20262208 | 29 | True | 0.525 | 0.460 |

R = 0.456. Знаки vs B1/B2/B3/UNION: 8/8 vs B1, 3/8 vs B2, 0/8 vs B3, 0/8 vs UNION.


**F1: F1_MID**

## 4. Phase 2: склейка (17 archive-only донора), 6 seed

| seed | n_gen | extinct | best_panel_r | best_test_r |
|---|---|---|---|---|
| 20262211 | 25 | True | 0.537 | 0.430 |
| 20262212 | 25 | True | 0.537 | 0.430 |
| 20262213 | 26 | True | 0.588 | 0.510 |
| 20262214 | 25 | True | 0.537 | 0.430 |
| 20262215 | 32 | False | 0.550 | 0.450 |
| 20262216 | 26 | True | 0.537 | 0.430 |

R2 = 0.447. Знак vs B3: 0/6.


**F2: F2_NO**

## 5. Phase 3: добор языка

Кандидаты (sorted, K<=4): `[]`.

**CLOSE_NEED_MODELS** -- реестр `experiment11` не содержит моделей сверх текущих 6 (подтверждено прямым чтением, не предположено). Grid-расширение/пересчёт B3'/evolve не запускались -- нет кандидатов.

## 6. Forensic

Лучший генотип (`cx-e7b946`) идентичен B2 bit-for-bit: False; идентичен B3: False.
На test: оба (B3 и организм) решают 49, только B3 -- 2, только организм -- 0, ни один -- 49.

## 7. Корзина

**B3_STACK** (LANG_RICH + F2_NO + no atoms)


## 8. Non-claims

- Не заявляется «победили B3» вне правила §5 PROTOCOL.md.
- Нет block registry/Pareto/GATE-сравнения нигде в этом пакете.
- LANG_POOR/LANG_RICH из Phase 0 не пересчитывались задним числом.
- `arch2/`, `agent_a5_live_m/`, `agent_life1..7` не изменялись.

## 9. Дальше (ровно одна строка)

подключить модели в registry, не крутить stall.
