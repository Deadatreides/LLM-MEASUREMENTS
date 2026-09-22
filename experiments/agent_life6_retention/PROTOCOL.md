# PROTOCOL.md — agent_life6_retention

Написано ДО первого evolve этого пакета. Не переписывается после того,
как числа увидены — интерпретация/расхождения добавляются в BLOCKERS.md/
REPORT_LIFE6.md, не сюда.

## 0. Рамка

LIFE-5 (`agent_life5_slot_map/`, завершён): `MAP_STABILITY_GAP` на
READ/LOOKUP/COMPUTE — Imp=1-сигнатур много (не дефицит покрытия), но
`median_L≈1` почти везде: носитель редко живёт дольше одного поколения.
Complementary slot-HGT (`transfer_slot`) НЕ считается сломанным — это
подтверждённый default-канал (LIFE-2..5), в этом пакете не трогается.

Цель: за ОДИН проход `scripts/run_all.py` — smoke → кампания Branch A →
механическая корзина → ветвление БЕЗ возврата к пользователю → REPORT.
Не чинить registry снова (его здесь вообще нет — ни block registry, ни
BLOCK_INSERT, ни R1-R4, ни A1-A5). Не новые атомы/модели. Один REPORT в
конце, ровно одна next-строка.

Пишем только в `agent_life6_retention/`. Читаем `arch2/` (импорт,
неизменно), `agent_life2_complementary/`, `agent_life3_block_live/`,
`agent_life4_block_fix/`, `agent_life5_slot_map/`, `agent_a5_live_m/` —
без правок. Grid: REUSE (скопирован из `agent_life5_slot_map`, md5
проверен до первого evolve — см. BLOCKERS.md), 0 новых `generate()`.

## 1. Default-канал (всегда, во всех вариантах)

`p_slot_hgt=0.35`, `transfer_slot` — комплементарный-only (копия
LIFE-2..5's `slot_hgt.py`, без изменений), НЕТ block_registry/
BLOCK_INSERT/R1/A1-A5, lean-старт assert (20 различных не-эталонных, 0 с
`|ImpSet|>=2`), `references ∉ donors`, `donor_score_fn` отсутствует,
`pop=20`, `budget_per_task=3000`, `G_STALL=20`, `max_gen=32` (как
LIFE-4/5's optional evolve — уже подтверждено, что клетки реально
доходят до этого окна). Панель/грид — копия LIFE-5 (сама копия LIFE-4
← LIFE-3), md5-проверена перед первым evolve.

**Уже верно по конструкции, без нового кода** (перепроверено этой
сессией чтением `slot_hgt.py`): `transfer_slot` никогда не трогает
слот, где `Imp(s, parent)=1` — `valid_by_slot` строится циклом
`for s in range(n_slots): if s in parent_imp: continue`, т.е. слоты с
Imp=1 у родителя исключаются ДО выбора донора. Половина рычага Branch A
("TRANSFER не ломает Imp=1") не требует изменений вообще.

## 2. Primary метрики (предрегистрация)

На каждом seed, gen≥1, по слотам READ/LOOKUP/COMPUTE (FORMAT — только
sanity, Imp(FORMAT) всегда False по построению):

- `median_L_imp` — медиана лайфспана (`death_gen - birth_gen`) среди
  ВСЕХ носителей (по всем Imp=1-сигнатурам слота вместе, не по одной
  сигнатуре — вопрос "как долго живут хорошие структуры на этом слоте
  вообще", не "какая сигнатура самая частая").
- `frac_L_ge3` — доля этих носителей с `L≥3`.
- `n_imp_sig` — число различных Imp=1-сигнатур на слоте (контроль: не
  обвалить покрытие рычагом retention).
- `N_AB_first_assembly`, `loss:gain(overall)` — та же формула, что
  LIFE-1..5 (`imp_set`-based, admitted births).
- (вторично, не входит в корзину) best `r` на panel/test — без claim B3.

Агрегаты — mean по seed (WITH vs CTRL, парно по позиции в списке seed).

## 3. Branch A (сначала всегда): slot-safe mutate

**Рычаг — один непрерывный параметр `p_safe`, не два механизма.**
Задание предлагает "запрещены ИЛИ p_safe=0.05" — реализовано как ОДНА
функция `slot_safe_mutate(..., p_safe)`: с вероятностью `p_safe` пропуск
защиты (принять что угодно от `M.mutate` — не даёт слоту быть
ЗАМОРОЖЕННЫМ навсегда, что само по себе было бы патологией); иначе —
повтор (`MAX_RETRIES=10`, как везде в кодовой базе) до кандидата,
ДОКАЗУЕМО не трогающего ни один защищённый слот. `p_safe=0.05` — Branch
A; `p_safe=0.0` (полный запрет — частный случай того же параметра) —
Phase A3.

**Структурная защита ПЕРЕД диффом (найдено чтением `arch2/mutate.py`
этой сессией, не предполагалось заранее).** `addresses(root)` всегда
включает пустой адрес самого узла, и `set_at(root, [], new)` заменяет
ВЕСЬ root. M3 (`wrap_budget`/`wrap_par`), M4 (`refine`), M5, M6 могут
переписать `child["root"]` целиком (может стать `SWITCH`/`BUDGET`/голым
`CALL`/произвольным поддеревом донора, без `"children"` вовсе, или с
ДРУГИМ числом children) — `mutate()`'s report никогда не содержит
слот/адрес (проверено исчерпывающе: только `{"operator", "donor_id"?,
"composite_id"?}`), так что доказать безопасность можно только диффом
результата. Правило: если `child["root"]["op"] != "ASSEMBLE"` или число
children изменилось — считать, что мутация коснулась ВСЕХ защищённых
слотов (нельзя атрибутировать одному — "не доказано безопасно" должно
значить "небезопасно" для этого рычага). Только при сохранённой
структуре сравнивается JSON защищённых `children[s]` попарно.

Дизайн: `agent_life3/4/5`'s `MAX_RETRIES`-циклы (`transfer_slot`/
`block_insert`) КОНСТРУКТИВНЫ — знают слот ДО построения кандидата;
здесь принципиально другой паттерн — построить кандидат ЧЕРЕЗ
`M.mutate()`, затем инспектировать/отклонять. Такого паттерна в кодовой
базе раньше не было — новый код, не копия.

Исчерпание `max_retries` → `(None, {"operator": None, ...})`, как у
`mutate()` самого — НЕ откат на небезопасную мутацию (это свело бы
рычаг на нет).

### Дизайн: WITH_A vs CTRL

WITH_A = default-канал + `slot_safe_mutate(p_safe=0.05)` вместо
`M.mutate` в "иначе"-ветке диспетчера. CTRL = default-канал как есть
(plain `M.mutate`). seeds_A = 20261901..20261906 (6+6=12 клеток).

### Корзина A (не двигать после чисел)

```
RET_A_WORKS:
  mean median_L_imp(WITH) >= mean(CTRL) + 0.5   на >=2 из {READ,LOOKUP,COMPUTE}
  AND mean frac_L_ge3(WITH) >= mean(CTRL) + 0.05  на >=1 из этих слотов
  AND mean N_AB_first_assembly(WITH) >= mean(CTRL) - 1   (не развалить сборку)
  AND n_imp_sig не просел катастрофически: mean(WITH) >= 0.7 * mean(CTRL) по каждому слоту

RET_A_FAIL:
  Delta median_L < 0.5 на ВСЕХ слотах  ИЛИ  mean N_AB_first_assembly(WITH) <= mean(CTRL) - 3

RET_A_PARTIAL: иначе (рост слабый/на 1 слоте/сборка слегка просела)
```
Precedence вычисления: WORKS → FAIL → иначе PARTIAL (проверяется в этом
порядке; та же прецедентность, что во всех прежних пакетах).

## 4. Ветвление внутри run_all.py (без нового брифа, PROTOCOL §0)

```
RET_A_WORKS  -> Phase A2: 4 seed WITH_A-only (20261911..20261914),
                устойчивость median_L (сравнение с Branch A's CTRL-базой,
                уже посчитанной — новый CTRL не нужен).
                Next: default = HGT + slot-safe; следующий крупный —
                r vs B2/B3 на канале ИЛИ multi-slot overlap lifetimes.

RET_A_PARTIAL -> Phase A3: p_safe=0.0 (жёсткий запрет), 4 WITH_A_hard +
                4 CTRL_A3 (20261921..20261924). Reclassify той же
                classify_ret(). Если результат WORKS-подобен -> default
                = жёсткий запрет, done. Иначе -> Phase B в ЭТОМ ЖЕ
                прогоне (время/CPU позволяют по бюджету ниже); если
                вдруг не позволяют — BLOCKERS явно, REPORT всё равно с
                A(+A3) и next-строкой "Branch B по PROTOCOL §5, не
                переписывая".

RET_A_FAIL   -> Phase B сразу (carrier soft-immunity, §5), 4 WITH_B +
                4 CTRL_B (20261931..20261934). Reclassify той же
                classify_ret(). RET_B_WORKS -> default = HGT + immunity.
                RET_B_FAIL -> "удержание не берётся ни mutate-safe, ни
                immunity при pop=20/gate; следующий пакет — только смена
                selection timescale (Pareto Imp-slots vs r ИЛИ
                ослабление gate), не атомы, не registry".
```

`classify_ret(with_metrics, ctrl_metrics)` — ОДНА функция, буквально те
же пороги §3, вызывается для A, A3 и B без модификации (задание: "те же
пороги, что RET_A_WORKS").

## 5. Branch B: carrier soft-immunity (только если A/A3 не сработали)

Confirmed чтением `arch2/evolve.py` этой сессией: "смерть", которая
реально убирает генотип из `self.population` (и это именно то, что
измеряет `death_gen_index`/lifespan — `summary["generations"][i]
["deaths"]` строится РОВНО из `d1 + d2`) — это D1
(`_d1_screen_deaths`) и D2 (`_d2_subsidy_deaths`), обычные
переопределяемые методы без побочных эффектов. D3 (`_d3_elite_dropouts`)
НЕ убирает из population (SPEC.md: остаётся жив на субсидии) — защищать
там нечего, поэтому НЕ переопределяется.

**"не D1-kill" прочитано как "исключение из смерти популяции широко"
(D1 ∪ D2), не буквально только D1** — иначе D2 убивал бы того же
носителя в то же поколение, и защита была бы no-op примерно в половине
случаев. Это единственное прочтение, реально двигающее `median_L`/
`frac_L_ge3`.

Архитектурно безопасно (проверено, не предположено): `self.archive`
только пополняется (`_admit`/`seed`, никогда не удаляется) — A5-подобные
инварианты монотонности архива, на которые опирались прежние пакеты,
остаются верны. `merge_events` не затронут. `stall_count`/`extinct`
могут только выиграть от бОльшего числа выживших (больше шансов на
`any_win=True`), не проиграть.

`ImmuneEvolution(E.Evolution)` — переопределяет `_d1_screen_deaths`/
`_d2_subsidy_deaths`, отфильтровывая `self.protected_ids` из
возвращаемого списка "мёртвых". Конструируется НАПРЯМУЮ (`ImmuneEvolution
(registry=..., ...)`), как и везде в этой линии пакетов — `arch2.evolve.
build()` не используется ни в одном orchestrator.py этой линии, так что
подкласс не требует monkeypatch.

`M=2` носителя на слот (READ/LOOKUP/COMPUTE), пересчитывается КАЖДОЕ
поколение из текущих alive non-reference генотипов с `Imp(s)=1`,
наименьший возраст (`gen - birth_gen`, т.е. САМЫЕ МОЛОДЫЕ) первыми, тай-брейк
через `ev.rng` (не Python-порядок множества/словаря).

WITH_B = default-канал + immunity (plain `M.mutate`, БЕЗ slot-safe —
рычаги не комбинируются в одном прогоне, PROTOCOL §0 says один рычаг за
раз). CTRL_B = default-канал как есть. seeds_B = 20261931..20261934.

## 6. Смоук (до 12 клеток A, обязателен)

- lean 20 различных, `|ImpSet|<2` у не-эталонов.
- ≥1 `TRANSFER_SLOT` complementary попытка.
- Фикстура slot-safe: синтетический родитель с ИЗВЕСТНЫМ Imp(READ)=1
  (пара `gemma-3-it-1b-q5_k_s`+`internvl3-2b-q4_k_m`, PAR — подтверждена
  LIFE-5's картой), `slot_safe_mutate` вызывается ~50 раз с `p_safe=0.05`
  — слот READ не меняется в подавляющем большинстве принятых исходов
  (только `p_safe`-пропуски могут его менять — залогировано, не просто
  assert вслепую). Отдельно: юнит-проверка whole-root-rewrite ветки
  (искусственный child с другим числом children) действительно
  трактуется как "трогает всё защищённое".
- grep-подтверждение: 0 registry-путей в default-диспетчере.

Провал → BLOCKERS.md, стоп, без 12 клеток.

## 7. Бюджет

Худший случай A(12)+A3(8)+B(8)=28 клеток при ~170-190с/клетка
(max_gen=32/G_STALL=20, наблюдение LIFE-4/5) ≈ 80-90 мин CPU в фоне;
лучший случай (WORKS сразу) A(12)+A2(4)=16 клеток ≈ 45-50 мин. Оба
укладываются в "один сеанс" — не дробить на LIFE-6a/6b без явного
BLOCKERS о нехватке времени.

## 8. Файлы

```
agent_life6_retention/
  PROTOCOL.md README.md BLOCKERS.md
  src/  scripts/  metrics/  runs/  reports/REPORT_LIFE6.md
```
