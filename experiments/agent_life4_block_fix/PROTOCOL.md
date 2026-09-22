# PROTOCOL.md — agent_life4_block_fix

Написано ДО запуска `scripts/verify_seams.py` (т.е. до первого числа этого
пакета). Не переписывается после того, как числа увидены — правки/
интерпретации добавляются рядом (BLOCKERS.md / REPORT_LIFE4.md), не сюда.

## 0. Рамка (не сужать)

LIFE-3-LIVE (`agent_life3_block_live/`, завершён) дал **`UNIT_LIVE_FAIL`**:
complementary-slot-HGT жив на живых данных (`slot_match=1`, положительный
`N_AB_first_assembly` во всех 6 WITH-seed), но автокатализ блока (A1-A5)
закрылся только на 1/6 seed при пороге ≥5/6. Раскрытый форензик
(`agent_life3_block_live/BLOCKERS.md`, `reports/REPORT_LIFE3_LIVE.md`
§5.1) нашёл ДВЕ конкретные, задокументированные, но НЕ исправленные там
причины:

1. **Измерение**: `metrics_lib.slot_models()` распознаёт только
   `gen.`-префиксные `CALL`; единственный зарегистрированный блок
   (`cx.cx-23f285`) содержал `CALL(cx.cx-b3b0bb)` (вложенный композит) —
   невидим для старого кода, записанная сигнатура (1 модель) была
   неполной (истина — 4 модели, см. §4.1 ниже).
2. **Окно вымирания**: популяция гаснет (`ev.extinct=True`) около
   поколения ~12 — вероятно, до того как сигнатура слота успевает
   накопить R1's порог (3 различных носителя, медиана L≥2).

Этот пакет исправляет РОВНО эти два места и ничего больше — тот же
WITH/CTRL/6-seed дизайн, только (1) `slot_models` разворачивает `cx.`-
композиты и (2) окно вымирания расширено. Не цель: r≥B3, новые модели,
сетка по `p`, similarity-HGT, ещё один общий audit.

Пишем только в `agent_life4_block_fix/`. Читаем `arch2/` (импорт,
неизменно), `agent_life3_block_live/` (копия кода + чтение данных —
изоляция пакетов это разрешает), `agent_life2_complementary/`,
`agent_life1_mechanism/`, `agent_a5_live_m/` — без правок чужих папок.
Без pip/venv. Модели — только `experiment11` registry (не понадобится
здесь: см. §1, grid=reuse).

## 1. Что считается live (как LIFE-3, не ослаблено)

- Fitness/Imp/PASS для вердикта — только `generate()` этого пакета ИЛИ
  живая сетка, которую этот пакет сам построил/подтвердил. Запрет
  fitness из `experiment14/runs14`, `arch2` offline, чужого сырого grid
  без верификации.
- **Решение (зафиксировано здесь, до чисел): grid = REUSE**, не
  пересборка. Основание: `agent_life3_block_live/metrics/live_grid/`
  проверен напрямую (эта сессия, до записи этого файла) — `train_grid.
  json`/`test_grid.json` по 2400 строк, `whole_grid.json` 700 строк
  (ровно 6 моделей × 4 шага × 100 train / 100 test, плюс B0's 6×100
  train + 100 test-победителя = 700). Финальная строка build-лога
  LIFE-3 подтверждает `4800/4800 ячеек за 1031с (train-строк 2400,
  приёмочный минимум 1920)`. `live_call_log.jsonl` (5501 строк) содержит
  реальные timestamp/latency/seed — не синтетика. Файлы
  СКОПИРОВАНЫ (не импортированы, не прочитаны на лету из чужой папки) в
  `agent_life4_block_fix/metrics/live_grid/` + `split.json` +
  `live_call_log.jsonl` + `train_panel.json`; побайтовое совпадение
  (md5) с оригиналом подтверждено до записи этого файла (см.
  BLOCKERS.md). `live_grid_builder.py` скопирован НЕ используемым —
  только как `UNIT4_BLOCKED`-путь, если целостность копии когда-либо не
  подтвердится (`scripts/verify_seams.py::verify_grid_copy`).
- Раз grid=reuse: **0 новых `generate()` вызовов** — ни для сетки, ни
  для B0 (тот же `whole_grid.json` тоже скопирован). Evolve — чистый
  CPU-replay поверх скопированной сетки, как в LIFE-2/части LIFE-3.
- TRAIN floor ≥1920 проверяется заново на КОПИИ (не берётся на веру из
  LIFE-3) — `scripts/verify_seams.py`, до эволюции.

## 2. Корзины (предрегистрировано, не двигать после чисел)

Не использовать 6.8/28.2 (LIFE-1/2, другой режим посева) как порог здесь
— confirmed правило проекта. LIFE-3's собственные числа (WITH=CTRL на
5/6 seed) тоже НЕ порог сами по себе — сравнение только против CTRL этого
же прогона, парно по seed.

- **`UNIT4_WORKS`**:
  A1-A5 (§4.3, идентично LIFE-3) одновременно на **≥5/6** WITH-seed
  **И** (mean `N_AB_first_assembly`(WITH) > mean(CTRL) **ИЛИ** парный
  знак `+` на ≥4/6 seed) **И** mean `loss:gain`(overall, WITH) ≤
  mean(CTRL)+0.5 **И** live floor выполнен **И** фикстура разворота cx
  (§4.1) проходит.
- **`UNIT4_PARTIAL`**: A1-A5 на 3-4/6 **ИЛИ** (A1-A5 ≥5/6, но нет Δ
  first_assembly по правилу выше).
- **`UNIT4_FAIL`**: A1-A5 ≤2/6 **ИЛИ** `N_AB_first_assembly` устойчиво
  (≥4/6 seed) ниже CTRL **ИЛИ** фикстура разворота cx проваливается.
- **`UNIT4_BLOCKED`**: инфраструктура сломана (grid-копия не проходит
  целостность и rebuild-fallback тоже не проходит; либо смоук §5 не
  проходит) — вердикт не выносится.

Precedence при вычислении: BLOCKED → WORKS → FAIL → иначе PARTIAL
(зеркалирует `agent_life3_block_live/scripts/run_all.py::classify_basket`).

## 3. Design: 6 WITH + 6 CTRL, одна сетка

```
seeds = 20261701 .. 20261706   # НОВЫЕ, не LIFE-3's 20261601..06
WITH: complementary transfer_slot + block registry + BLOCK_INSERT
CTRL: complementary transfer_slot only, registry навсегда пуст
pop=20, budget_per_task=3000
Диспетчер: roll<0.35 -> transfer_slot; 0.35<=roll<0.50 -> block_insert
           (или откат на mutate, если реестр пуст -- CTRL всегда);
           иначе -> mutate. РОВНО один rng.random() на попытку, обе ветки.
```

## 4. Ровно два изменения против `agent_life3_block_live/`

### 4.1 Рычаг A — измерение (`metrics_lib.py`, `block_registry.py`)

`slot_models(slot_node, registry, _depth=0, _seen=frozenset())`: при
`CALL` с `molecule` начинающимся на `cx.` — резолвится через
`registry.get(mid)["genotype"]["root"]` и обходится рекурсивно (ВЕСЬ
`root`, т.к. вложенный композит сам `ASSEMBLE` — это семантика
исполнения в `arch2/runner.py`, композит-CALL выполняет ВЕСЬ свой root,
не один слот). `MAX_UNFOLD_DEPTH=6` + `_seen` — защита от глубины/циклов
(циклов не ожидается: id композита детерминирован по содержимому, но
проверка дешёвая и явная).

`registry` — теперь обязательный параметр `slot_models`/`imp_set`
(`metrics_lib.py`) и `block_signature` (`block_registry.py`). Живой код
(`orchestrator.py`) передаёт настоящую `HSTEP.Registry()`-инстанцию
(`ev.registry`/`reg`). Пост-хок аудит (`metrics_lib.audit_cell`)
передаёт `build_audit_registry(genotypes, blocks)` — read-only резолвер
поверх уже сохранённых `archive.json`+`blocks.jsonl` (реальный in-memory
`registry` никогда не сериализуется — подтверждено на LIFE-3's
`cx-b3b0bb`: он существует ТОЛЬКО как обычная запись `archive.json`, не
в отдельном registry-дампе; свои же блоки восстанавливаются через
`carrier_cid`+`slot`, т.к. `register_block` не добавляет обёрнутое
поддерево в `archive` отдельно).

**Sanity fixture** (before any evolve, `scripts/verify_seams.py::
verify_fixture_unfold`): читает `cx-068fae` и `cx-b3b0bb` НАПРЯМУЮ из
`agent_life3_block_live/runs/life3_with_s20261601/archive.json`
(read-only, реальные данные, не переписаны вручную) — `cx-068fae`'s
слот 2 (`= cx.cx-23f285`, единственный реально зарегистрированный блок
в LIFE-3) есть `PAR(SEQ(CALL(cx.cx-b3b0bb),CHECK), SEQ(CALL(gen.gemma-
3-it-1b-q5_k_s),CHECK))`. После фикса `slot_models` на этом узле должен
вернуть **4** различных модели: `gemma-3-it-1b-q5_k_s` (прямая ветка) +
`qwen3-1.7b-q4_0-unsloth`, `qwen2.5-coder-1.5b-instruct-q4_0`,
`internvl3-2b-q4_k_m` (все три — через разворот `cx-b3b0bb`, у него
самого все листья `gen.`-префиксные, разворот не рекурсирует дальше).
Проверка также подтверждает, что БЕЗ разворота (null-registry) результат
воспроизводит СТАРОЕ (1 модель) поведение — доказывает, что фикс меняет
результат, не просто безвредно ничего не делает.

### 4.2 Рычаг B — окно вымирания (`orchestrator.py`)

**Вариант B1** (не B2): `G_STALL = 20`, `N_GENERATIONS(max_gen) = 32`
(было 12/24 в LIFE-3). Основание: `g_stall` — обычное поле `Evolution`
(`arch2/evolve.py`, dataclass, `Optional[int]=None`, не frozen),
разрешается как `self.g_stall if self.g_stall is not None else
G_STALL(=3, модульный дефолт)` в единственном месте, где проверяется
(`arch2/evolve.py:441`). Экстинция срабатывает после `stall_count >=
g_stall` ПОДРЯД идущих поколений без единого генотипа, побившего gate.
`agent_life3_block_live/src/orchestrator.py` УЖЕ переопределяет это
(`g_stall=12`) нулём правок `arch2/` — прямой прецедент; здесь просто
другое число. R1-R4 (`N_MIN_USE=3`, `MEDIAN_L_MIN=2`, `MAX_BLOCKS=12`,
≤1 регистрация/поколение) **НЕ меняются** вместе с окном — задание явно
требует держать их как в LIFE-3 при выборе B1.

Ничего больше в диспетчере/RNG-потреблении не меняется — WITH/CTRL
RNG-lockstep-при-пустом-реестре инвариант (LIFE-3 BLOCKERS.md "Находка
1") остаётся верным здесь тоже: `block_insert`'s откат на `M.mutate`
byte-идентичен CTRL's собственному пути, ровно один `rng.random()` на
попытку в обеих ветках.

Стоимость: LIFE-3's 12 клеток — 1642с суммарно (~137с/клетка), многие
гасли задолго до старого max_gen=24. Расширенное окно позволит более
медленным seed'ам идти дольше (до 32 поколений вместо 24) — ожидается
~1.5-3x время evolve-фазы, всё ещё CPU-only replay поверх
скопированной сетки, комфортно в пределах бюджета 2-4ч.

### 4.3 A1-A5 (НЕ меняется)

Идентично LIFE-3 (`metrics_lib.compute_block_a1_a5`): A1=≥1 блок
зарегистрирован; A2=≥1 потомок (gen≥1) использует его через CALL,
отфильтровано СТРОГО по `block_ids` из `blocks.jsonl` этого пакета (не
по любому `cx.`-префиксу — arch2's собственная `_try_register_molecule`
безусловно активна в обеих ветках, независимо от блоков этого пакета,
LIFE-3's находка, здесь не переисследуется); A3=≥1 из них дошёл до элиты
или побил gate; A4=100% исполнения (CALL_COMPOSITE с molecule∈block_ids
в трассе); A5=merge_events=0 и архив монотонен.

## 5. Обязанности по ходу прогона (в `scripts/run_all.py`, не отдельно)

1. Копирование+проверка сетки (§1) — уже выполнено при подготовке
   пакета, `scripts/verify_seams.py` повторно проверяет на КОПИИ.
2. Смоук (throwaway или seed 20261701, ≤8 поколений): lean-20-distinct
   assert; ≥1 `transfer_slot` complementary попытка в births; фикстура
   разворота cx (§4.1); искусственно посеянный один dummy-блок в
   `block_registry` → ≥1 РЕАЛЬНАЯ (не fallback) попытка `BLOCK_INSERT` в
   births. Провал любого → BLOCKERS.md, стоп, без 12 клеток.
3. 12 клеток (6 WITH + 6 CTRL) — дамп archive/heredity/births/blocks/
   summary на клетку, как в LIFE-3.
4. Форензик автоматически (не отдельным шагом после): на seed с
   `n_blocks>0` — diff `complex_id` множеств WITH/CTRL + поколение
   первого расхождения; если агрегаты всё равно равны при `n_blocks>0`
   — явная строка в отчёте; список сигнатур блоков ПОСЛЕ разворота
   (`metrics_lib.unfolded_block_signatures`).
5. Корзина (§2) вычисляется механически; НЕ пересматривается после
   того, как числа видны — расхождения/интерпретация добавляются рядом
   (REPORT_LIFE4.md), не через смену вердикта.

## 6. Non-goals (см. также задание)

Алфавит моделей, DECOMPOSE, similarity-donor, сетка по `p`, правки
`arch2/`, отключение `_try_register_molecule` — не трогаются. После
REPORT: без «ещё пары seed», без «ослабим N_MIN_USE и пересчитаем»,
без «быстрый offline вместо live» — без нового явного задания.

## 7. Файлы

```
agent_life4_block_fix/
  PROTOCOL.md README.md BLOCKERS.md
  src/  scripts/  metrics/  runs/  reports/REPORT_LIFE4.md
```
См. README.md для полного списка модулей и того, что в каждом изменено
против `agent_life3_block_live/`.
