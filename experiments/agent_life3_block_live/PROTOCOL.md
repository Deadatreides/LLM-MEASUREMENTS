# PROTOCOL.md — agent_life3_block_live

Написан и зафиксирован **2026-08-21, до первого `generate()`**. Не
переписывается задним числом — вердикт-корзина `UNIT_LIVE_*` публикуется
КАК ВЫЧИСЛЕНА, интерпретация добавляется рядом, явно.

## §0. Рамка

Цель — НЕ «набить r=B3 на шести моделях». Три проверяемых утверждения,
все — только на ЖИВЫХ данных этого пакета:

1. комплементарный slot-HGT (`p=0.35`, из `agent_life2_complementary`)
   всё ещё собирает joint coverage на СВЕЖЕМ сплите/свежих ответах;
2. регистрация устойчивого БЛОКА как единицы + `BLOCK_INSERT` замыкает
   автокатализ (A1–A5) без мёртвых ссылок;
3. `N_AB_first_assembly` — не артефакт конкретной сетки `agent_a5_live_m`.

Изоляция: пишет только в `agent_life3_block_live/`. `arch2/`,
`agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`
— чтение кода/паттернов, без правок. Без pip/venv. Модели — только через
`experiment11/configs/model_registry.py`, как `agent_a5_live_m`.

## §1. Что считается LIVE (жёстко, не переоткрывается)

Live = каждый PASS/`Imp`/r/c, попавший в ЛЮБОЙ вердикт этого пакета,
трассируется к строке `metrics/live_call_log.jsonl` (поля: `ts, model,
task_id, step/kind, seed, temperature, max_tokens, prompt_hash, raw_status,
tokens, ms`).

Запрещено: читать `experiment14/runs14/*`/любую offline-сетку `arch2` как
fitness; брать PASS/`Imp` из `agent_a5_live_m/metrics/live_grid` как
ЕДИНСТВЕННЫЙ источник истины для вердикта (разрешено только как smoke-
ориентир ДО построения своей сетки, см. §3.2).

Порядок: (1) пустой кэш пакета → (2) build live_grid на НОВОМ сплите
(§3.1) → (3) evolve/registry — только lookup в ЭТУ сетку → (4) на этапе
evolve `live_call_log` МОЖЕТ не расти — это ожидаемо и корректно (весь
живой сигнал уже собран на шаге 2), не провал → (5) если evolve идёт, а
`live_grid`/`live_call_log` не существует — провал, `BLOCKERS.md`, стоп.

**Пол (floor) минимума строк лога** после grid-build:
`n_models × n_kinds × n_tasks_train = 6×4×100 = 2400` (панель+screen —
это и есть весь TRAIN). Приёмка: ≥ 80% ориентира = **1920** строк по
TRAIN-клеткам; ниже — `BLOCKERS.md` с честной причиной, не маскировка.

## §2. Научный вопрос и корзины (не двигать после чисел)

На живой сетке этого пакета: комплементарный HGT + регистрация блока +
`BLOCK_INSERT` дают замкнутый автокатализ единиц и не убивают сборку
joint coverage?

- **UNIT_LIVE_WORKS** ⟺
  - A1–A5 (§4.3) держатся ОДНОВРЕМЕННО на ≥5 из 6 WITH-seed **И**
  - mean `N_AB_first_assembly`(WITH) ≥ mean `N_AB_first_assembly`(CTRL)
    (парно по номеру seed — WITH и CTRL используют ОДИН И ТОТ ЖЕ seed,
    расходятся только включённостью реестра блоков, см. §5) **И**
  - mean `loss:gain(overall)`(WITH) ≤ mean `loss:gain(overall)`(CTRL) + 0.5
    **И**
  - `live_call_log` пол (§1) выполнен; 0 обращений к чужой сетке как
    fitness (проверяется grep'ом, §6).

- **UNIT_LIVE_PARTIAL** ⟺ цикл A1–A5 держится на 3–4 из 6 WITH-seed, ИЛИ
  `N_AB_first_assembly`(WITH) не выше CTRL, но канал живой (были реальные
  `TRANSFER_SLOT`, `slot_match`≈1.0 — корректность реализации, не находка).

- **UNIT_LIVE_FAIL** ⟺ цикл A1–A5 не замыкается (≤2 из 6), ИЛИ
  `N_AB_first_assembly` устойчиво ниже CTRL, ИЛИ деградация `loss:gain`
  (WITH хуже CTRL больше чем на 0.5).

- **UNIT_LIVE_BLOCKED** ⟺ registry/GPU/шов/пол лога сломаны — только
  `BLOCKERS.md`, без корзины.

Числа `6.8`/`28.2` из LIFE-1/LIFE-2 НЕ используются как порог здесь
(другой режим — LIFE-2 была гипотеза `p=0.35`-комплементарности на CPU-
replay старой сетки; подтверждение только через живые данные этого
пакета).

## §3. Дизайн прогона

### 3.1 Сплит и сетка

```
GEN_SEED   = 150002   # новый, ≠ A5 (150001), ≠ LIFE-1/2
SPLIT_SEED = 20260821
LIVE_SEED_BASE = 6000  # не пересекается с A5's 5000/exp14's 3000/arch2 E1's 2000
N_TASKS = 200          # heterostep.split() -- строго 50/50, n_train=100/n_test=50
                       # из задания технически недостижимо без правки
                       # experiment14 (запрещено); 200→100/100 -- то же
                       # соотношение, что A5, задание явно разрешает
                       # "или как heterostep generator позволяет"
panel  = 80 id из TRAIN (зафиксирован в metrics/train_panel.json ДО evolve)
screen = оставшиеся 20 TRAIN id
```

Клетки сетки: ВСЕ пары `(model ∈ 6, kind ∈ 4 шага)` на всех 200 задач
(TRAIN∪TEST) — модель-мажорный обход (`experiment14/harness.py::
run_train_grid`, доказанно быстрый паттерн, ~517с/2400 ячеек на этой
машине). `temperature=0.5, top_p=1.0, max_tokens_step=60`. Seed ячейки:
`LIVE_SEED_BASE + stable_hash(task_id,kind,model) % 10**6` (sha256-based,
не встроенный `hash()` — тот же приём, что `agent_a5_live_m/PROTOCOL.md`
§3, детерминирован между прогонами).

### 3.2 Смоук ДО сетки

`scripts/verify_seams.py`: швы `experiment14/seams.py` self-test + `arch2/
heterostep.py::self_test_seam` + РОВНО один живой `generate()`, одна
строка в `live_call_log.jsonl`. Провал → `BLOCKERS.md`, стоп, без отката
на offline. `agent_a5_live_m`'s сетка МОЖЕТ использоваться здесь только
как visual sanity-ориентир до построения своей (например, "какие модели
вообще что-то отвечают") — не как источник чисел ни в одном отчёте.

### 3.3 Базовые линии

B0/B1/B2/B3 строятся НА ЭТОЙ живой сетке (та же логика, что `agent_a5_
live_m/src/baselines.py`: B1/B2/B3 — детерминированный greedy-порядок
[`agent_life2_complementary/src/lean_seeds.py::greedy_order_deterministic`,
скопирован, не `heterostep_seeds._greedy_order` — тот хэш-порядок-
зависим], B0 — лучшая одиночная модель на целостном плече, решено
данными). Вторичные метрики (§7), не заголовок вердикта.

## §4. Канал и единица блока

### 4.1 Комплементарный перенос (фиксировано, не сетка)

`p_slot_hgt = 0.35`, `transfer_slot` — копия логики `agent_life2_
complementary/src/slot_hgt.py` (донор допустим для слота `s` только если
`Imp(s,donor)=1 ∧ Imp(s,recipient)=0`, равномерный выбор слота и донора,
`donor_score_fn` нигде не в `ctx`, откат на `arch2.mutate.mutate`).

### 4.2 Единица «блок» и регистрация (R1–R4)

**Черновик LIFE-3 с готовыми R1–R4 в репозитории не найден** — правила
ниже определены здесь, ЗАФИКСИРОВАНЫ ДО ПРОГОНА, из численных параметров
задания (`N_MIN_USE=3`, `median L≥2`, `≤1 reg/gen`, `MAX_BLOCKS=12`), по
образцу уже существующего в `arch2/heredity.py`/`evolve.py`
молекула-регистрационного гейта (`registration_gate`,
`_try_register_molecule`, `MAX_REGISTER_PER_GEN`), применённого на
уровне СЛОТА, а не целого генотипа.

**Блок** = `(slot_index ∈ {0..3}, signature)`, где `signature = (op ∈
{SEQ,PAR}, sorted(model_id атомов в слоте))` — та же `signature_s`, что
LIFE-1/2.

**R1 (минимум использования)**: кандидат на регистрацию только если его
`(slot, signature)` встречается СТРУКТУРНО (`children[slot]` совпадает по
`signature`) в ≥ `N_MIN_USE=3` РАЗЛИЧНЫХ генотипов текущего архива
(эталоны исключены).

**R2 (минимальная устойчивость)**: медиана `L` (поколений от рождения до
смерти/конца прогона — тот же расчёт, что LIFE-1/2's `L(A)`) по всем
носителям сигнатуры ≥ 2.

**R3 (темп)**: не более 1 НОВОЙ регистрации за поколение.

**R4 (потолок)**: не более `MAX_BLOCKS=12` регистраций за весь прогон.

Если в одном поколении квалифицируется НЕСКОЛЬКО `(slot,signature)` —
берётся с наибольшим числом носителей; тай-брейк — меньший `slot_index`,
затем алфавитно по множеству моделей. Правило парсимонии (arch2 SPEC §24,
"не регистрировать доминируемое уже зарегистрированным") НЕ добавлено —
не запрошено заданием, не изобретается дополнительно.

**Механизм регистрации**: переиспользуется `arch2.heterostep.Registry.
register_composite` БЕЗ ИЗМЕНЕНИЙ — слот-поддерево оборачивается в
самостоятельный генотип (`G.genotype(slot_subtree_root, gen=g,
origin=f"block:slot{s}")`), регистрируется под СОБСТВЕННЫМ `complex_id`
(не carrier'а — идентичность блока = свойство сигнатуры, не того, кто её
первым принёс). `H.composite_calls`/`instantiated_from`-рёбра работают
для блоков БЕСПЛАТНО (молекулы блоков именуются `cx.<id>`, тот же
префикс, что и обычные композиты — существующий код их не различает,
и не должен).

### 4.3 A1–A5 (для БЛОКОВ, не для целых комплексов — отличие от LIFE-1)

| | критерий | проходит если |
|---|---|---|
| A1 | зарегистрировано блоков | ≥1 |
| A2 | потомков с `CALL` на блок ПОСЛЕ его регистрации | ≥1 |
| A3 | из них — дошли до элиты / побили планку хотя бы раз | ≥1 |
| A4 | доля исполненных block-CALL (по трассам) | =100% (структурная проверка, не находка) |
| A5 | `merge_events=0` и архив монотонен | оба верны (унаследованный инвариант arch2) |

Seed «проходит» A1–A5, если ВСЕ пять верны одновременно.

### 4.4 `BLOCK_INSERT`

```
BLOCK_INSERT(parent, rng, ctx, registry, gen, block_registry, imp_fn):
  candidates = [(slot,mid) for slot,mid in block_registry, slot < len(parent.children)]
  если candidates пуст -> M.mutate(...)                      # блоков ещё нет
  complementary = [(s,m) for s,m in candidates if Imp(s,parent)=0]
  pool = complementary if complementary else candidates       # "предпочтительно", не жёстко
  (s, mid) = rng.choice(pool)
  child.children[s] = CALL(mid, params={})                    # автокатализ: молекула = композит
  retry до MAX_RETRIES при невалидности/дубликате, иначе M.mutate
  report = {"operator":"BLOCK_INSERT","molecule_id":mid,"slot":s,"complementary":bool}
```

## §5. Факторы и объём (не сетка — фиксировано)

Не сканируется `p`. Два режима × 6 seed = 12 evolve, ОДНА живая сетка на
всё (строится 1 раз):

```
seeds = 20261601 .. 20261606
WITH:  transfer_slot + регистрация блока + BLOCK_INSERT включены
CTRL:  transfer_slot включён; регистрация/insert ВЫКЛЮЧЕНЫ (пустой реестр
       навсегда -> BLOCK_INSERT всегда падает в M.mutate, тот же код,
       один флаг)
```

Диспетчер размножения (общий код для WITH/CTRL, один `rng.random()` на
попытку, число обращений к RNG одинаково в обеих ветках — расхождение
запускается только ПЕРВОЙ реальной регистрацией в WITH, честно, не
скрыто): `roll<0.35` → `transfer_slot`; `0.35≤roll<0.50` →
`BLOCK_INSERT` (или откат, если реестр пуст — CTRL всегда здесь); иначе →
`M.mutate`. `p_block_insert=0.15` — абсолютная вероятность попытки, не
условная.

Константы: `pop=20` (бедный старт — 0 не-эталонных `|ImpSet|≥2` до
gen 0, `assert`, как LIFE-2 §4), `max_gen=24`, `G_STALL=12`,
`budget_per_task=3000`. `M.random_genotype`, как выяснено в LIFE-2, не
производит валидных ASSEMBLE на реестре HETEROSTEP — бедный посев строится
явно через `heterostep_seeds.route`, копия `agent_life2_complementary/
src/lean_seeds.py`, но Imp-пары для однослотовых затравок ищутся ЗАНОВО
на этой (новой) сетке, не копируются с LIFE-2's старых панелей (панель
другая, PASS-паттерны другие).

## §6. Анти-фрод

- Смоук (§3.2) — до сетки, обязателен.
- `len(live_call_log)` ≥ 1920 после grid-build (§1) — иначе `BLOCKERS.md`.
- Grep пакета перед публикацией отчёта: ни одного `open(...)` на
  `experiment14/runs14/*`/`arch2` offline-grid/`agent_a5_live_m/metrics/
  live_grid` внутри кода, участвующего в подсчёте fitness/`Imp`/вердикта.
- `archive.json`/`heredity.jsonl`/`births.jsonl`/`blocks.jsonl` — дамп на
  КАЖДЫЙ из 12 прогонов (не повторять дыру A5 — урок `agent_life1_
  mechanism/BLOCKERS.md`).
- Эталоны исключены из donor-пула HGT И из `block_registry` (регистрация
  блока — только из НЕ-эталонных archive-генотипов, R1 явно проверяет
  `not in ev.references`).

## §7. Метрики

Первичные: строки `live_call_log`, покрытие сетки (train/test отдельно),
A1–A5 по WITH-seed (§4.3), `N_AB_first_assembly`, `elite_AB_share`,
`loss:gain(overall)` — WITH vs CTRL парно, `N_blocks_registered`,
`N_BLOCK_INSERT` (попыток/успешных/дошедших до элиты).

Вторично, без claim про B3: `r_test` относительно живых B1/B2 с CI
(bootstrap, `n_boot=10000`, `arch2.fitness.paired_delta_r`).

## §8. Не заявляется

- Не заявляется превосходство над B3 без CI, исключающего равенство.
- Не заявляется перенос за пределы HETEROSTEP+6 моделей.
- `arch2/`, `agent_a5_live_m/`, `agent_life1_mechanism/`,
  `agent_life2_complementary/` не изменялись.
- `UNIT_LIVE_PARTIAL`/`UNIT_LIVE_FAIL`/`UNIT_LIVE_BLOCKED` — явно
  допустимый исход, не провал задания.
- Не заявляется, что R1–R4 (§4.2) — единственно возможное определение
  «блока как единицы»; они зафиксированы здесь как ОДНА конкретная,
  явно обоснованная операционализация, не переоткрываемая после чисел.
