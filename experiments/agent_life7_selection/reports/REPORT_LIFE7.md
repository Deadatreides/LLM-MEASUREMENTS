# REPORT_LIFE7 — selection timescale: Pareto-survival + stall-from-improvement

Спека: `PROTOCOL.md`. Default-канал = ASSEMBLE + complementary slot-HGT (`p=0.35`), БЕЗ slot-safe mutate, БЕЗ M=2-youngest immunity (оба — LIFE-6, оба FAIL по median_L). Живая сетка ПЕРЕИСПОЛЬЗОВАНА byte-identical из `agent_life6_retention` (md5 подтверждён), 0 новых generate() вызовов.

## 1. Рамка

LIFE-6: `RET_A_FAIL`/`RET_B_FAIL` по median_L, но mean_L/occ показывали слабый сигнал. LIFE-7 меняет слой ПОД retention-рычагами: КТО умирает (Pareto-доминирование по (r,n_imp) поверх обычных D1/D2) и КОГДА объявляется stall (улучшение best_r/best_n_imp, не gate).

## 2. Смоук

seed=999901, 8 поколений: TRANSFER_SLOT complementary=8, Pareto candidates=109 exempted=4, A2 improved=4/8, юнит-фикстуры Pareto=True stall=True, 0 registry-путей=True. Смоук: PASS.

## 3. Branch A: 6 WITH_A + 6 CTRL_A

WITH_A/20262001: n_gen=24 extinct=True, WITH_A/20262002: n_gen=25 extinct=True, WITH_A/20262003: n_gen=25 extinct=True, WITH_A/20262004: n_gen=27 extinct=True, WITH_A/20262005: n_gen=32 extinct=True, WITH_A/20262006: n_gen=25 extinct=True, CTRL_A/20262001: n_gen=20 extinct=True, CTRL_A/20262002: n_gen=20 extinct=True, CTRL_A/20262003: n_gen=20 extinct=True, CTRL_A/20262004: n_gen=20 extinct=True, CTRL_A/20262005: n_gen=20 extinct=True, CTRL_A/20262006: n_gen=20 extinct=True

| slot | mean_L WITH | CTRL | grew? | p90_L WITH | CTRL | grew? | occ_imp WITH | CTRL | grew? |
|---|---|---|---|---|---|---|---|---|---|
| READ | 3.79 | 1.38 | True | 11.87 | 1.85 | True | 0.94 | 0.88 | False |
| LOOKUP | 4.34 | 1.73 | True | 12.77 | 3.78 | True | 0.96 | 0.93 | False |
| COMPUTE | 3.34 | 1.63 | True | 10.67 | 2.82 | True | 1.00 | 1.00 | False |

mean frac_gen_AB_alive: WITH=0.92, CTRL=0.88
mean N_AB_first_assembly: WITH=9.17, CTRL=11.67
mean extinct_gen: WITH=25.33, CTRL=19.00, seeds where WITH extinct earlier: 0/6

На первом WITH-seed (20262001): Pareto-кандидатов на смерть=235, исключено (недоминированы)=97.
A2: 3/24 поколений сбросили stall по улучшению r/n_imp.


**Корзина A: PARTIAL**

### 3.1 Интерпретация (не меняет корзину, добавлена рядом)

Рост `mean_L`/`p90_L` — САМЫЙ сильный из всех retention-рычагов этой линии
(LIFE-6's slot-safe/immunity двигали mean_L максимум в разы меньше):
`mean_L` вырос в ~2.4-2.7 раза на каждом слоте (READ 1.38→3.79, LOOKUP
1.73→4.34, COMPUTE 1.63→3.34), `p90_L` — в 3.5-6.4 раза. **Проверено, не
взято на веру**: рост НЕ артефакт одного долгого seed — по каждому из 6
WITH_A seed'ов отдельно `mean_L(READ)` лежит в диапазоне 3.07-4.43 (все
seed'ы согласованно выше CTRL's ~1.38), рост стабилен, не вызван
единственным seed 20262005 (32 поколения из 32 возможных).

Пареto-фильтр реально и часто срабатывает: на первом WITH-seed (20262001,
24 поколения) — 235 кандидатов на смерть, 97 (≈41%) исключены как
недоминированные. Это НЕ узкий эффект.

**Почему всё же PARTIAL, не WORKS**: единственное несработавшее условие
— `mean_N_AB_first_assembly`(WITH=9.17) ниже порога `CTRL-1`(=10.67)
CTRL=11.67. Защита существующих носителей от смерти буквально означает
меньше освобождающихся мест в популяции — меньше циклов
смерть→размножение, меньше попыток СОБРАТЬ НОВОЕ совместное покрытие с
нуля. Рычаг реально продлевает то, что уже хорошо, ценой скорости
появления нового хорошего — предсказуемый, не ошибочный компромисс.

## 4. Phase A3 (SEL_A_PARTIAL): vector-Pareto (per-slot Imp), 4 WITH_A3 + 4 CTRL_A3

| slot | mean_L WITH | CTRL | grew? | p90_L WITH | CTRL | grew? | occ_imp WITH | CTRL | grew? |
|---|---|---|---|---|---|---|---|---|---|
| READ | 2.99 | 1.38 | True | 8.90 | 2.15 | True | 1.00 | 0.81 | True |
| LOOKUP | 3.47 | 1.90 | True | 9.40 | 3.53 | True | 1.00 | 0.85 | True |
| COMPUTE | 2.92 | 1.63 | True | 9.00 | 2.50 | True | 1.00 | 1.00 | False |

**Корзина A3: FAIL**

### 4.1 Интерпретация (не меняет корзину)

Ужесточение Pareto (вектор per-slot Imp вместо скалярного `n_imp`) дало
ЕЩЁ более сильный рост retention (occ_imp вырос уже на 2/3 слотов, не 0/3,
как в Branch A) — но именно поэтому first_assembly просел СИЛЬНЕЕ, а не
слабее: `mean_N_AB_first_assembly` WITH=8.25 против CTRL=13.25 — падение
ровно −5.0, пересекает явный порог FAIL (`<= CTRL-3`). Это согласуется с
§3.1's объяснением: более строгая защита (недоминирование по ВСЕМ 4
координатам сразу, а не по одной агрегированной) оставляет ЕЩЁ меньше
оборота популяции. Дозозависимость (больше защиты -> больше retention,
но и больше цены по assembly) видна прямо в цифрах между Branch A и
Phase A3, не постулируется.

A3 не WORKS -> переход к Branch B в этом же прогоне.

## 5. Branch B: stall-from-improvement ONLY (без Pareto-смерти), 4 WITH_B + 4 CTRL_B

| slot | mean_L WITH | CTRL | grew? | p90_L WITH | CTRL | grew? | occ_imp WITH | CTRL | grew? |
|---|---|---|---|---|---|---|---|---|---|
| READ | 1.47 | 1.14 | True | 2.90 | 1.78 | True | 0.65 | 0.62 | False |
| LOOKUP | 3.06 | 2.24 | True | 9.50 | 4.83 | True | 0.86 | 0.88 | False |
| COMPUTE | 2.49 | 1.90 | True | 6.40 | 4.73 | True | 0.99 | 1.00 | False |

mean N_AB_first_assembly: WITH=13.25, CTRL=11.75
mean extinct_gen: WITH=26.25, CTRL=19.00


**Корзина B: WORKS**

### 5.1 Интерпретация: почему stall-only выигрывает там, где Pareto-смерть — нет

Тот же рост retention виден и здесь (mean_L/p90_L выросли на всех
слотах), но МЕНЬШЕГО масштаба, чем в Pareto-death ветках (READ 1.14→1.47,
не →3.79) — а `first_assembly` при этом НЕ просел, а ВЫРОС (WITH=13.25
против CTRL=11.75, ветка A3 показала 8.25). Правдоподобный механизм: A2
не защищает НИ ОДИН конкретный генотип от смерти (обычные D1/D2 работают
как всегда) — он лишь меняет, когда популяция объявляется "застрявшей",
на более терпеливый критерий (улучшение best_r/best_n_imp, а не победа
над gate). Это продлевает ВЕСЬ прогон (`mean_extinct_gen` WITH=26.25
против CTRL=19 — почти как у Pareto-веток), давая больше поколений и
на retention, и на assembly одновременно, БЕЗ прямого удержания каких-то
конкретных носителей ценой оборота популяции. Ретеншн и стоимость сборки
в Pareto-ветках оказались двумя сторонами ОДНОГО и того же механизма
(меньше оборота); в Branch B они РАЗДЕЛЕНЫ (больше времени всему прогону
— не то же самое, что меньше оборота конкретных слотов), поэтому только
здесь удалось получить рост без компромисса.

## 6. Итоговый default

default = HGT + stall-from-improvement (смерть не тронута).

## 7. Non-claims

- Не заявляется достижение/сравнение с B3 в заголовке.
- Нет block registry/BLOCK_INSERT/R1-R4/A1-A5/slot-safe-mutate/M=2-youngest-immunity нигде здесь.
- Нет новых атомов/моделей.
- Победа НЕ объявлена по одной лишь `median_L_imp` (secondary here, per PROTOCOL.md/LIFE-6 lesson).
- `arch2/`, `agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`, `agent_life3_block_live/`, `agent_life4_block_fix/`, `agent_life5_slot_map/`, `agent_life6_retention/` не изменялись.

## 8. Дальше (ровно одна строка)

default = HGT + stall-from-improvement; следующий крупный пакет = r vs B2/B3.

**Добавлено рядом, не вместо** (§3.1/§4.1/§5.1): Pareto-смерть (Branch
A/A3) дала САМЫЙ сильный retention-эффект во всей линии LIFE-6/7 (mean_L
до 2.7x, p90_L до 6.4x), но ценой первой сборки (first_assembly), тем
сильнее, чем строже защита — это НЕ отброшенный тупик, а количественно
охарактеризованный компромисс (retention vs скорость обновления
популяции), который стоит иметь в виду, если следующий пакет когда-либо
станет измерять r vs B2/B3: если итоговый маршрут окажется чувствителен
к разнообразию (не только к удержанию), Pareto-смерть с более мягким
объективом (не vector, возможно между scalar и vector) — нерассмотренная
здесь промежуточная точка дозозависимости.
