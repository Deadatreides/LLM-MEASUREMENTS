# PROTOCOL.md — agent_fork1_three_paths (FORK-1)

Написан на шаге K0, ДО любого числа прогона. Пороги §8 не двигаются
после первых чисел (K3/K7). Интерпретация — только в BLOCKERS.md/
REPORT_FORK1.md, рядом с механическим вердиктом, никогда вместо него.

## 0. Рамка

DELTA-0 (2026-08-23): Δ=-0.500 на F2 (транзакции/JOIN-фильтр →
threshold ДА/НЕТ) — LLM-исполнитель фильтровал неточно (recall 0.82,
precision 0.55), exact-set чекпойнт не прощает эту неточность, whole's
мягкий финальный ДА/НЕТ — прощает. Два раздельных вопроса, один прогон:

- **Path A**: была ли проблема именно в ИСПОЛНЕНИИ (LLM), а не в самой
  структуре декомпозиции? Те же атомы на детерминированном коде.
- **Path B**: было ли DELTA-0's threshold-ДА/НЕТ необычно снисходительным
  к неточности? На T с точным (не boolean) финалом — тот же LLM-
  исполнитель в критическом атоме, тот же честный golden-план.
- **Path C**: продуктовая спека из фактов A/B, без новых вызовов.

Закон (§0 задания) соблюдается буквально: пишем только в
`agent_fork1_three_paths/`; 0 Evolution/HGT/mutate/Pareto/stall/block
registry/ASSEMBLE/HETEROSTEP-шаблонов где-либо в этом пакете; `delta0`/
`experiment11` — read-only (данные читаем, код копируем, не импортируем).

## 1. Path A — детерминированные атомы на F2 TEST (K1-K3)

### 1.1 Входные константы (K1, из delta0, не выдумываются)

Источник: `agent_delta0_new_grid/metrics/delta.json` (прочитан и
сверен этой сессией — совпадает буквально):
```
r_m*_F2      = 0.525   # qwen3-1.7b-q4_0-unsloth, whole, F2 TEST
r_∪_LLM_F2   = 0.025   # LLM-FILTER chain (DELTA-0's own r_∪_lower)
Δ_F2_LLM     = -0.500
```
Если `delta.json`/`whole_by_model.json` отсутствуют в дереве —
HARD_STOP (§10), не пересчитывать F2 whole заново (240 вызовов уже
потрачены в DELTA-0).

F2 TEST (те же 40 задач) переиспользуется КОДОМ generator'а delta0,
скопированным (не импортированным) в `src/task_generator_f2.py`,
вызванным с тем же `SEED_T=20260823` (см. §1.2 — подтверждено сверкой
с delta0's собственным манифестом, не другим seed из соседней линии
LIFE-3..9, который использует похожее, но другое число 20260821 для
СПЛИТА, не для этого генератора).

### 1.2 Уточнение: seed константа

`agent_delta0_new_grid/src/task_generator.py::SEED_T = 20260823` —
именно это значение используется для воспроизведения тех же 40 F2
TEST-задач (подтверждено сверкой при K1: тот же список id, что в
`agent_delta0_new_grid/metrics/tasks_manifest.json`).

### 1.3 Детерминированные атомы (K2)

`src/det_atoms.py`, чистый Python, 0 LLM-вызовов по умолчанию:
- `filter_det(records, category, region) -> set(ids)` — НЕЗАВИСИМАЯ
  переимплементация того же предиката category AND region, что
  использовал generator (не passthrough `task["matched_ids"]` — само-тест
  §1.4 имеет смысл, только если код независим).
- `aggregate_det(ids, op, id_to_amount) -> float` — SUM/COUNT/MAX.
- `derive_det(aggregate_value, threshold, comparator) -> "ДА"|"НЕТ"`.

DERIVE: **default = det** (весь pipeline без LLM, задание §3 A3). LLM-
DERIVE не реализуется в этом прогоне (не запрошено, не включается).

### 1.4 Само-тест (обязателен, HARD_STOP при провале)

`filter_det` на всех 40 F2 TEST задач ДОЛЖЕН дать v=1 (точное совпадение
с `task["matched_ids"]`) 40/40 — иначе это баг КОДА (не находка), HARD_STOP
до K3.

### 1.5 Метрики Path A

```
r_A = mean(final correct) на F2 TEST (40)
cost_A = 0 LLM-вызовов (весь pipeline det)
Δ_A = r_A - r_m*_F2
```
Ожидание (задание, не гадаем): r_A≈1.0, Δ_A≈+0.475. **Флаг
`TOOL_PIPELINE=true`** в `metrics/path_a.json` — это НЕ победа LLM-роя
(§6/§9 задания, буквально).

## 2. Path B — T_hard = EXTRACT-SUM (K4-K7)

### 2.1 T_hard, зафиксирован (не выбор из трёх — задание даёт один дизайн)

Тот же дух multi-record/JOIN-фильтр, что F2 (category AND region по
транзакциям — переиспользуется ДОМЕН, не threshold-финал F2, что и
запрещено B3), но final = **точная СУММА** amount по JOIN — голое число,
толеранс 0.01, никогда не threshold-сравнение.

- `K_HARD=14` записей на задачу (клирит B5's ≥1500 символов с запасом —
  F2's формат строки ~118 символов/запись, `14*118≈1652`, подтверждено
  измерением в K4, не угадано).
- `SEED_B=20260824` (буквальная константа задания, B4).
- `N_TRAIN=40`, `N_TEST=40` (≥30 по B4, тот же масштаб, что F2, для
  сравнимости).
- Та же дистракторная конструкция, что F2: 2-5 истинных совпадений,
  часть записей — только category ИЛИ только region (гарантировано
  построением).

### 2.2 Два чекпойнта (не три) — самоопределённое чтение §4

Задание называет ровно два: "matched set (exact), partial sum
(numeric)". Читаем буквально:
- **matched_ids** — вывод FILTER, exact-set.
- **sum_value** — вывод SUM-атома, numeric-tolerance. `sum_value` И ЕСТЬ
  `final_oracle` (в EXTRACT-SUM нет отдельного DERIVE-преобразования —
  сумма САМА является ответом). Причинность B2 ("ошибка mid ⇒ final
  почти наверняка 0") соблюдена: неверный matched_ids почти всегда даёт
  неверную сумму. Третий искусственный чекпойнт не вводится — задание
  §9 явно запрещает "не третий boolean-порог" и в целом усложнение
  постфактум; вводить чекпойнт ради счёта было бы тем же духом
  нарушения.

### 2.3 Атомы Path B

- FILTER: тот же `filter_det` (для B-DET) ИЛИ LLM (для B-LLM, промпт в
  `src/atoms_hard.py`) — по конструкции задачи и K_HARD (14 записей),
  промпт аналогичен F2's FILTER (без изменений в подходе).
- SUM: **ВСЕГДА `aggregate_det(op="SUM")`** — и в B-LLM, и в B-DET (K6:
  "SUM=det по извлечённым id"). Ни в одном варианте SUM не спрашивается
  у LLM.

### 2.4 K5 — r_m*_B (обязательный live, C6-style)

Whole-промпт T_hard просит ТОЛЬКО финальное число (не intermediate) —
тот же принцип, что DELTA-0's whole (§4 P4 delta0's PROTOCOL, тут
переприменён, не заново обоснован). Все 6 моделей registry, N_TEST_B=40.
`r_m*_B = argmax r_m` (T_hard's СОБСТВЕННЫЙ, не переиспользован от F2 —
T_hard другое семейство задач, K5 сам его измеряет как обязательный шаг).

### 2.5 K6 — r_∪_B, два под-замера

- **B-DET**: FILTER=`filter_det`, SUM=`aggregate_det`. 0 LLM-вызовов.
  Ожидание ~1.0 (тот же потолок-по-построению, что Path A) — тоже
  `TOOL_PIPELINE=true`, если `Δ_B_DET>0`.
- **B-LLM**: FILTER=LLM. **Executor = m\*_B из K5** (самоопределённый
  выбор между двумя вариантами задания "m* из path A/delta0 ИЛИ argmax
  whole на T_hard" — второй строже: T_hard — другое семейство, K5 уже
  обязателен и даёт этот m* бесплатно, без отдельного замера). SUM —
  det (§2.3).

Обрыв цепочки — идентично DELTA-0 (перенесённые правила, не заново
выведенные): FILTER v=0 ⟹ `union_pass=False` немедленно, без golden
подстановки, SUM не вызывается для этой задачи.

### 2.6 K7 — Δ_B

```
Δ_B_LLM = r_∪_B_LLM - r_m*_B
Δ_B_DET = r_∪_B_DET - r_m*_B
```

### 2.7 Бюджет T_hard

Формула как в delta0: `B_hard` = потолок (input+output) на whole-вызов,
фиксируется числом ПОСЛЕ измерения фактического размера промпта (K4),
`B_atom_hard <= B_hard/4`, `B_total_union_hard <= 2*B_hard`. Cost в
отчёт; если measured mean union cost > 2B_hard — флаг `BUDGET_VIOLATION`
(тот же принцип, что delta0).

## 3. Path C — продукт (K8, 0 новых вызовов)

`metrics/path_c_product.md`, написан из УЖЕ посчитанных чисел A/B/K5-7,
по шаблону задания §5: primary = whole m*; optional = det tool pipeline
для точного set/sum; explicit non-goal = multi-LLM HGT на 1-2B без
показанного Δ>0. Не запускает модели. Не оспаривает корзину §8.

## 4. Критерий "LLM-рой имеет смысл" (§6 задания, буквально)

```
LLM_SWARM_HAS_SENSE ⟺ Δ_B_LLM >= 0.05 И r_∪_B_LLM > r_m*_B И FILTER в
                       этой цепочке — LLM (не det)
```
`Δ_A>0` или `Δ_B_DET>0` при det-FILTER → флаг `TOOL_WIN`, никогда
`LLM_SWARM_WIN`.

## 5. Сводная таблица (K9, обязательна, формат §7 задания)

| path | r | r_m* ref | Δ | LLM в критическом атоме? | cost |
|---|---|---|---|---|---|
| A det-F2 | r_A | 0.525 | Δ_A | no | 0 |
| B whole | r_m*_B | = | 0 | n/a | |
| B ∪ LLM | r_∪_B_LLM | r_m*_B | Δ_B_LLM | yes FILTER | |
| B ∪ DET | r_∪_B_DET | r_m*_B | Δ_B_DET | no | 0 |

## 6. Корзина (K9-K10, механически, порядок сверху вниз — не пересматривается)

```
FORK_LLM_SWARM : Δ_B_LLM >= 0.05
                 Next: "DELTA-1 только T_hard: D-слой + LLM-атомы;
                        бюджет <=2B; без HGT до r_real>r_m* на holdout"
FORK_TOOL_ONLY : (Δ_A>=0.05 или Δ_B_DET>=0.05) и не FORK_LLM_SWARM
                 Next: "продукт = whole + det tools; LLM-рой на 1-2B
                        не приоритет; R&D отбора/HGT стоп"
FORK_WHOLE_ONLY: ни один из вышеперечисленных порогов не пройден
                 Next: "продукт = single whole m*; декомпозиция на
                        этих T не нужна"
FORK_BLOCKED   : HARD_STOP / floor < 80%
                 Next: "чинить generate/parse; не теория"
```
При равенстве интересов: LLM_SWARM > TOOL_ONLY > WHOLE_ONLY (не
применимо здесь — корзины взаимоисключающие по условию, оговорено
заданием на случай будущей неоднозначности).

`HIGH_SINGLE` (`r_m* >= 0.70` на F2-референсе ИЛИ на T_hard) — отдельная
строка REPORT, корзину не отменяет.

## 7. Запреты на импровизацию (§9 задания, буквально)

Не усложнять T_hard после Δ_B; не смягчать exact-set после цифр; не
подставлять golden ids при FILTER fail; не объявлять Δ_A>0 победой роя;
не запускать HGT/evolve; не вводить третий boolean-порог в T_hard.

## 8. HARD_STOP (§10)

- нет delta0 metrics для K1
- generate floor < 80% на K5 (или K6)
- det FILTER не даёт 40/40 на golden F2 (баг Path A)
- N_TEST_B < 30 (не должно случиться — B4 фиксирует 40)
- OOM на ≥ половине registry при whole B

При срабатывании: BLOCKERS.md, корзина `FORK_BLOCKED`, CHECKLIST с `[ ]`
на сорванном шаге, REPORT всё равно пишется, с причиной.
