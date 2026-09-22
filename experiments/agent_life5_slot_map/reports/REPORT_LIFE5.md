# REPORT_LIFE5 — карта слотов + default без registry

Спека: `PROTOCOL.md`. Блок-этаж (LIFE-3/4, `UNIT_LIVE_FAIL`/`UNIT4_FAIL`, 1/6 оба раза) отложен, не возобновляется. Default = ASSEMBLE + комплементарный slot-HGT (`p=0.35`), БЕЗ block registry / BLOCK_INSERT вообще.

## 1. Рамка

Не чиним registry снова. Вопрос: дефицит ПОКРЫТИЯ (мало полезных Imp-сигнатур на слоте) или дефицит УСТОЙЧИВОСТИ (сигнатуры есть, не копятся носители)? Карта строится по уже существующим архивам LIFE-3/4, не по новому эксперименту.

## 2. Источники данных

- Обязательный пул: 24 прогонов (`agent_life3_block_live`/`agent_life4_block_fix`, WITH+CTRL). FORMAT sanity (Imp=0 везде): OK.
  - Из них 12 CTRL-прогонов УЖЕ являются точным экземпляром Phase-A дефолта (registry всегда был выключен там) — обязательный пул не чисто «старый код с мёртвым registry».
- Опциональный кросс-чек (подтверждено пользователем): 4 новых seed, чистый default-канал (0 registry-кода вообще). Смоук (seed=999801, 6 поколений): admitted TRANSFER_SLOT=8, операторы=['M1', 'M2', 'M3', 'M5', 'TRANSFER_SLOT', 'random'], block-поля просочились=False. Смоук: PASS.
  - FORMAT sanity (кросс-чек): OK.

## 3. Таблицы по слотам (обязательный пул, gen>=1)

### READ (slot 0)

N_distinct_signatures=43, N_signatures_with_Imp=1=15

| models | op | n_carriers | n_seed_appeared | median_L | max_L |
|---|---|---|---|---|---|
| gemma-3-it-1b-q5_k_s, internvl3-2b-q4_k_m | PAR | 108 | 24 | 1.0 | 15 |
| internvl3-2b-q4_k_m, qwen2.5-coder-1.5b-instruct-q4_0 | PAR | 72 | 22 | 1.0 | 7 |
| qwen2.5-coder-1.5b-instruct-q4_0, qwen3-1.7b-q4_0-unsloth | PAR | 60 | 20 | 1 | 5 |
| internvl3-2b-q4_k_m, qwen3-1.7b-q4_0-unsloth | PAR | 19 | 16 | 1.0 | 1 |
| gemma-3-it-1b-q5_k_s, internvl3-2b-q4_k_m, qwen2.5-coder-1.5b-instruct-q4_0, qwen3-1.7b-q4_0-unsloth | CALL | 17 | 18 | 1 | 5 |

ΔImp на рождение (admitted, весь пул): +1=314, -1=70, 0=1722

### FORMAT (slot 1)

N_distinct_signatures=25, N_signatures_with_Imp=1=0

Ни одной Imp=1 сигнатуры не найдено на этом слоте (gen>=1, обязательный пул).

ΔImp на рождение (admitted, весь пул): +1=0, -1=0, 0=2106

### LOOKUP (slot 2)

N_distinct_signatures=33, N_signatures_with_Imp=1=10

| models | op | n_carriers | n_seed_appeared | median_L | max_L |
|---|---|---|---|---|---|
| gemma-3-it-1b-q5_k_s, qwen3-1.7b-q4_0-unsloth | PAR | 236 | 22 | 1 | 17 |
| gemma-3-it-1b-q5_k_s, internvl3-2b-q4_k_m, qwen2.5-coder-1.5b-instruct-q4_0, qwen3-1.7b-q4_0-unsloth | PAR | 66 | 20 | 1 | 7 |
| qwen3-1.7b-q4_0-unsloth, smollm2-1.7b-instruct-q4_k_m | PAR | 59 | 20 | 1.0 | 9 |
| gemma-3-it-1b-q5_k_s, internvl3-2b-q4_k_m, qwen2.5-coder-1.5b-instruct-q4_0, qwen3-1.7b-q4_0-unsloth | SEQ | 9 | 14 | 1.0 | 1 |
| gemma-3-it-1b-q5_k_s, internvl3-2b-q4_k_m, qwen2.5-coder-1.5b-instruct-q4_0, qwen3-1.7b-q4_0-unsloth | CALL | 7 | 6 | 1.0 | 2 |

ΔImp на рождение (admitted, весь пул): +1=122, -1=126, 0=1858

### COMPUTE (slot 3)

N_distinct_signatures=34, N_signatures_with_Imp=1=16

| models | op | n_carriers | n_seed_appeared | median_L | max_L |
|---|---|---|---|---|---|
| internvl3-2b-q4_k_m, qwen2.5-coder-1.5b-instruct-q4_0 | PAR | 285 | 24 | 1 | 17 |
| qwen2.5-coder-1.5b-instruct-q4_0, smollm2-1.7b-instruct-q4_k_m | PAR | 102 | 24 | 1.0 | 7 |
| gemma-3-it-1b-q5_k_s, internvl3-2b-q4_k_m, qwen2.5-coder-1.5b-instruct-q4_0, qwen3-1.7b-q4_0-unsloth | PAR | 48 | 20 | 1.0 | 11 |
| gemma-3-it-1b-q5_k_s, qwen2.5-coder-1.5b-instruct-q4_0 | PAR | 28 | 10 | 1.0 | 15 |
| qwen2.5-coder-1.5b-instruct-q4_0, qwen3-1.7b-q4_0-unsloth | PAR | 20 | 10 | 1.0 | 5 |

ΔImp на рождение (admitted, весь пул): +1=14, -1=202, 0=1890

### 3.1 Кросс-чек: опциональный default-only пул (4 seed, 0 registry-кода)

Не смешан с обязательным пулом (§4 считает корзины только по нему), но
именно ради этого запрошен пользователем — подтвердить, что находка не
артефакт «мёртвого» registry-кода LIFE-3/4. Тот же паттерн, независимо:

| slot | N_sig | N_imp | top-1 (models, op) | carriers | seeds/4 | median_L |
|---|---|---|---|---|---|---|
| READ | 27 | 10 | internvl3-2b-q4_k_m, qwen2.5-coder-1.5b-instruct-q4_0 (PAR) | 42 | 4 | 1 |
| FORMAT | 16 | 0 | — | — | — | — |
| LOOKUP | 31 | 7 | gemma-3-it-1b-q5_k_s, qwen3-1.7b-q4_0-unsloth (PAR) | 118 | 4 | 1.0 |
| COMPUTE | 26 | 9 | internvl3-2b-q4_k_m, qwen2.5-coder-1.5b-instruct-q4_0 (PAR) | 219 | 4 | 1 |

Топ-сигнатуры LOOKUP и COMPUTE **совпадают** с обязательным пулом
(`gemma+qwen3` и `internvl3+qwen2.5-coder` соответственно — те же пары,
что доминируют в §3), при полностью независимом коде (никакого
registry-пути вообще) и независимых seed (1801-1804, не пересекаются с
1601-1706). `median_L=1` везде — паттерн воспроизводится чисто, не
объясняется остаточным registry-кодом обязательного пула.

## 4. Корзины MAP_*

- **READ**: MAP_STABILITY_GAP — {'n_signatures_with_imp': 15, 'median_Ls': [1.0, 1.0, 1, 1.0, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1], 'rest_all_thin': False, 'outlier': ['gemma-3-it-1b-q5_k_s', 'internvl3-2b-q4_k_m']}
- **LOOKUP**: MAP_STABILITY_GAP — {'n_signatures_with_imp': 10, 'median_Ls': [1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], 'rest_all_thin': False, 'outlier': ['gemma-3-it-1b-q5_k_s', 'qwen3-1.7b-q4_0-unsloth']}
- **COMPUTE**: MAP_STABILITY_GAP — {'n_signatures_with_imp': 16, 'median_Ls': [1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0], 'rest_all_thin': False, 'outlier': ['internvl3-2b-q4_k_m', 'qwen2.5-coder-1.5b-instruct-q4_0']}

**Сводная корзина: MAP_STABILITY_GAP**
- LOOKUP воспроизводима на >=2 seed/пакета: True
- READ воспроизводима на >=2 seed/пакета: True
- COMPUTE воспроизводима на >=2 seed/пакета: True

## 5. Следствие

**Не дефицит покрытия.** На всех трёх слотах десятки Imp=1-сигнатур с
carrier-счётом от единиц до 285 и воспроизводимостью до 24/24 seed —
LIFE-3/4's "только LOOKUP" был артефактом ОДНОГО зарегистрированного
блока на 12 прогонов, не свойством содержимого. При полном обзоре
карты COMPUTE (285 носителей на топ-сигнатуру) и LOOKUP (236) богаче
READ (108) по этой метрике — LOOKUP не выделяется структурным
изобилием.

**Дефицит устойчивости, равномерно по слотам.** `median_L=1` почти
везде на всех трёх слотах (обязательный пул И независимый кросс-чек
§3.1) — хорошая Imp-структура создаётся легко и часто, но конкретный
носитель (генотип) не задерживается в популяции дольше одного
поколения в большинстве случаев. R1 (3 различных носителя + медиана
L>=2 ОДНОВРЕМЕННО) — это ставка на то, что структуры накопятся, пока
живы; при `median_L=1` это структурно маловероятно на ЛЮБОМ слоте, не
только на READ/COMPUTE — LOOKUP's единственная регистрация в LIFE-3/4
была статистической удачей на редком длинном хвосте (`max_L` доходит
до 15-17), не признаком того, что LOOKUP как-то стабильнее.

**ΔImp-асимметрия по слотам (§3) — вторичный, не решающий момент.**
READ теряет Imp редко относительно приобретений (70 против 314),
COMPUTE — наоборот, теряет Imp гораздо чаще, чем приобретает (202
против 14, при этом всё ещё самый богатый по carrier-счёту слот — Imp
там, видимо, уже присутствует у большинства предков и жертва мутаций
чаще ломает уже-хорошее, чем создаёт новое). LOOKUP — единственный
слот с примерно равным балансом (122/126). Ни это, ни `MAP_STABILITY_GAP`
не объясняют друг друга полностью — оставлено как отдельное наблюдение,
не как причинный механизм (не проверялось глубже, не заявляется).

## 6. Non-claims

- Не заявляется достижение/сравнение с B3 в заголовке.
- Не заявляется дефицит покрытия ни на одном слоте — §5 явно опровергает эту гипотезу (обратное тому, что предполагал `MAP_COVERAGE_GAP`/`MAP_LOOKUP_ONLY` до подсчёта): READ/LOOKUP/COMPUTE все показали десятки Imp=1-сигнатур с высокой carrier/seed-воспроизводимостью, независимо подтверждённой в §3.1.
- Не заявляется причинный механизм ΔImp-асимметрии (READ/COMPUTE, §5) — отмечено как наблюдение, не объяснено и не проверено глубже.
- `arch2/`, `agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`, `agent_life3_block_live/`, `agent_life4_block_fix/` не изменялись (только прочитаны).
- Ни A1-A5, ни block_registry, ни BLOCK_INSERT, ни R1-R4 здесь не пересматривались.

## 7. Дальше (ровно одна ветка)

Все три слота (READ/LOOKUP/COMPUTE) — `MAP_STABILITY_GAP`, не смешанный исход: следующий пакет — один рычаг survival/мутационной нагрузки (не per-slot, а общесистемный — например, `elite`-удержание или частота замены популяции), применённый ко ВСЕМ слотам разом, не point-fix на LOOKUP; не новый R1-тюнинг и не новый пул атомов (§5 закрывает вопрос покрытия).
