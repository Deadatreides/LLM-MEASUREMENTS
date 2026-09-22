# DATA_MODEL.md — сущности и схемы данных

Обозначения: `I` = immutable после создания, `M` = mutable, `A` = append-only (можно только добавлять, нельзя удалять/перезаписывать).

**Глобальное правило:** ни одно поле с пометкой `I` или `A` не перезаписывается никогда. Изменение = создание новой версии сущности с ссылкой на предыдущую (`INVARIANTS: NO_SILENT_OVERWRITE`, `EVERY_STATE_CHANGE_IS_VERSIONED`).

---

## 0. Общие типы

```
ID          := строка вида "<kind>:<uuid>"        # kind ∈ {task, art, claim, dep, seam, ev, col, act, run, plan, budget, ...}
VERSION     := целое ≥ 1, монотонно растёт в пределах одного логического объекта
TIMESTAMP   := ISO-8601 UTC
HASH        := sha256 содержимого (для детекции повторяющегося состояния)

PROVENANCE  := {
  created_by       : "SYSTEM" | "MODEL" | "HUMAN" | "EXTERNAL"   # I
  action_id        : ID | null                                    # I  какое действие породило
  run_id           : ID | null                                    # I
  model_id         : string | null                                # I
  prompt_profile_id: ID | null                                    # I
  sampling_config  : SAMPLING | null                              # I
  input_refs       : [ID]                                         # I  что подавалось на вход
  parent_version   : ID | null                                    # I  предыдущая версия этого объекта
  timestamp        : TIMESTAMP                                    # I
}

SAMPLING    := { temperature, top_p, top_k, seed, max_tokens }     # I целиком
CONFIDENCE  := { value: [0..1] | null, basis: "MECHANICAL"|"STATISTICAL"|"MODEL_REPORTED"|"UNSET" }
```

**Про CONFIDENCE.** Отдельно оговаривается: `basis="MODEL_REPORTED"` (модель сама сказала «уверен на 90%») **никогда** не участвует в решениях уровня STATE и EVIDENCE. Оно хранится только как диагностика. Основание: F9–F14 — самооценка модели в этих данных не коррелировала с корректностью, а сгенерированные моделью «проверки» ошибались в большинстве случаев.

---

## 1. TASK

Корневая единица работы.

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `task_id` | ID | I | |
| `task_version` | VERSION | I | новая версия при изменении формулировки/требований |
| `statement` | text | I | исходная формулировка |
| `task_class` | enum | I | `MATH`/`CODE`/`LOGIC`/`EXPLANATION`/`MIXED`/`OTHER` — используется для профилирования моделей |
| `risk_level` | enum | M | `LOW`/`MEDIUM`/`HIGH` — влияет на policy для UNKNOWN (см. §21 задания) |
| `oracle_availability` | enum | M | `MECHANICAL`/`PARTIAL`/`NONE` — известно ли, что задачу можно проверить |
| `budget_id` | ID | I | |
| `root_artifacts` | [ID] | A | |
| `graph_version` | VERSION | M | текущая версия графа |
| `state` | enum | M | `OPEN`/`STOPPED`/`RESOLVED`/`ABANDONED` |
| `stop_reason` | enum \| null | M | см. REPAIR_MODEL.md §7 |
| `provenance` | PROVENANCE | I | |

`oracle_availability` — не косметика: это единственное поле, по которому система заранее знает, что она **в принципе** не сможет перевести claims в `CORRECT` механически, и должна работать в режиме «graph consistency only» (см. EVIDENCE_MODEL.md §6).

---

## 2. ARTIFACT

Самостоятельный результат вычисления, передаваемый между стадиями. **Не только код.**

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `artifact_id` | ID | I | идентификатор логического артефакта |
| `artifact_version` | VERSION | I | конкретная версия; `(artifact_id, artifact_version)` уникальны |
| `artifact_type` | enum | I | см. ниже |
| `task_id` | ID | I | |
| `content` | text/blob | I | **никогда не редактируется**; правка = новая версия |
| `content_hash` | HASH | I | |
| `parents` | [ID] | I | артефакты-входы (версионные ссылки) |
| `claims` | [ID] | A | claims, извлечённые из этого артефакта |
| `producer` | enum | I | `MODEL`/`SYSTEM`/`HUMAN`/`EXTERNAL` |
| `run_id` | ID \| null | I | |
| `state` | enum | M | см. STATE_MACHINE.md |
| `supersedes` | ID \| null | I | предыдущая версия |
| `superseded_by` | ID \| null | A→I | заполняется один раз при появлении новой версии |
| `provenance` | PROVENANCE | I | |
| `creation_time` | TIMESTAMP | I | |

**ARTIFACT_TYPE** (открытое перечисление, расширяемое):
`SPECIFICATION`, `TESTS`, `IMPLEMENTATION`, `CALCULATION`, `FACT_SET`, `DERIVATION`, `ARGUMENT`, `PLAN`, `EXPLANATION`, `PATCH`, `VERIFICATION_RESULT`, `OTHER`.

**Критично (F9):** артефакт типа `TESTS`, произведённый моделью, — это **обычный артефакт со своим состоянием и своими claims**, а не оракул. Он сам подлежит проверке и сам может быть `INCORRECT`. В эксп. 2 именно неверные `TESTS` были главным источником ложных отказов; архитектура обязана допускать состояние «тесты неверны» как первоклассное.

**Immutable versioning.** Изменение артефакта создаёт `artifact_version = N+1` с `supersedes = <N>`. Версия N остаётся доступной. Откат = переключение указателя `current_version` в графе, а не удаление (аудит J).

---

## 3. CLAIM

Минимальная смысловая единица, о которой можно **независимо** рассуждать, проверять зависимость и фиксировать состояние.

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `claim_id` | ID | I | |
| `claim_version` | VERSION | I | |
| `artifact_id` | ID | I | артефакт-носитель (версионная ссылка) |
| `content` | text | I | формулировка утверждения |
| `claim_type` | enum | I | `SIGNATURE`/`TYPE`/`VALUE`/`SEMANTICS`/`EDGE_CASE`/`METHOD`/`DERIVATION_STEP`/`FORMAT`/`OTHER` |
| `atomicity` | enum | I | `ATOMIC` / `COMPOSITE` / `INDIVISIBLE_BLOCK` (см. §3.3) |
| `parents` | [ID] | A | upstream claims (через DEPENDENCY) |
| `children` | [ID] | A | downstream claims |
| `evidence` | [ID] | A | **только добавление**; противоречащие evidence сосуществуют |
| `state` | enum | M | см. STATE_MACHINE.md |
| `verification_status` | enum | M | `NOT_ATTEMPTED`/`MECHANICALLY_VERIFIED`/`MODEL_VERIFIED`/`UNVERIFIABLE`/`FAILED` |
| `confidence` | CONFIDENCE | M | |
| `staleness_source` | ID \| null | M | какой upstream-claim вызвал STALE |
| `provenance` | PROVENANCE | I | |

**Разделение `state` и `verification_status` обязательно** (F44): `verification_status = UNVERIFIABLE` означает «наши правила проверки не применимы», а не «утверждение неизвестно по существу». В эксп. 9 за-строгие правила проверки давали ложный `UNKNOWN` в 32% случаев при полностью корректных ответах — то есть смешение этих двух вещей напрямую искажает картину состояния графа.

### 3.1 Критерии хорошего claim

Claim **хорош**, если выполняются все три:

1. **Отдельная проверяемость.** Существует (или может существовать) seam, проверяющий именно его, не затрагивая соседей.
2. **Отдельная ремонтируемость.** Его можно перегенерировать, не переписывая соседние claims того же артефакта.
3. **Осмысленная зависимость.** Его связи с другими claims выражают конкретное отношение, а не соседство в тексте.

**Claim слишком крупный**, если:
- одна коллизия в нём не позволяет отличить, какая его часть неверна (нет локализации внутри — теряется главная выгода декомпозиции, F29);
- его нельзя починить, не переписав заведомо корректные части.

**Claim слишком мелкий**, если:
- у него нет собственного seam и он всегда проверяется только вместе с соседом;
- его отдельная регенерация невозможна без полного контекста соседа (тогда это не независимая единица, а фрагмент);
- дробление увеличивает число рёбер и стоимость обхода, не давая новой локализации (см. DECOMPOSITION_MODEL.md §4).

**Явный запрет:** размер claim в токенах **не является** критерием. Критерий — вычислительная независимость и отдельная проверяемость.

### 3.2 Когда два claims нельзя безопасно разделить

Если корректность A определима только совместно с B (например, «i < j» и «i, j — индексы одной пары»), их разделение создаёт два claim, каждый из которых по отдельности не имеет истинностного значения. Такие claims объединяются в один с `atomicity = COMPOSITE` и сохранённой внутренней структурой в `content`.

### 3.3 Вычислительно неразделимый claim

`atomicity = INDIVISIBLE_BLOCK` — единица, внутри которой локализация невозможна имеющимися средствами (например, схлопнутая SCC из циклической зависимости, см. DEPENDENCY_MODEL.md §5, или монолитное рассуждение без внутренних проверяемых границ).

Для такого claim:
- affected region внутрь не заходит;
- единственный вид ремонта — `REGENERATE` целиком;
- он **явно помечен** как точка потери локализации, чтобы это было видно в диагностике, а не выглядело как обычный claim.

### 3.4 Защита от бессмысленного дробления

Дробление разрешено только если `DECOMPOSITION_GATE` (DECOMPOSITION_MODEL.md §4) даёт `benefit > cost`. Основание — F28: декомпозиция сама по себе стоит дороже и **не улучшает** качество ответа; её оправдывает только локализация и экономия ремонта.

---

## 4. DEPENDENCY

Направленное ребро `source_claim → target_claim`. Подробности — DEPENDENCY_MODEL.md.

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `dependency_id` | ID | I | |
| `source_claim` | ID | I | |
| `target_claim` | ID | I | |
| `dependency_type` | enum | I | `LOGICAL`/`COMPUTATIONAL`/`DATA`/`CONTRACT`/`CAUSAL`/`DERIVATION`/`FORMAT`/`OTHER`/`UNKNOWN_DEPENDENCY` |
| `evidence` | [ID] | A | чем обосновано существование ребра |
| `confidence` | CONFIDENCE | M | |
| `status` | enum | M | `HYPOTHESIS`/`CONFIRMED`/`REFUTED`/`DISPUTED` |
| `created_by` | enum | I | `DECOMPOSITION`/`INFERENCE`/`HUMAN`/`SEAM_OBSERVATION` |
| `provenance` | PROVENANCE | I | |

**`REFUTED` ребро не удаляется** — оно остаётся в графе со статусом `REFUTED` (иначе теряется история и невозможно понять, почему ремонт вёл себя так, а не иначе).

---

## 5. SEAM

Наблюдаемая граница, на которой возможна проверка. Подробности — SEAM_MODEL.md.

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `seam_id` | ID | I | |
| `seam_type` | enum | I | `SEAM_CONTRACT`/`SEAM_EXECUTION`/`SEAM_CONSISTENCY`/`SEAM_DEPENDENCY`/`SEAM_EVIDENCE`/`SEAM_FORMAT`/`SEAM_DRIFT`/`SEAM_CONFLICT` |
| `seam_class` | enum | I | **`HARD`/`SOFT`/`UNKNOWN`** — ключевое разделение |
| `participants` | [ID] | I | claims/artifacts, между которыми проходит шов |
| `check_ref` | ID \| null | I | ссылка на процедуру проверки (детерминированную) |
| `last_result` | enum \| null | M | `PASS`/`FAIL`/`INAPPLICABLE`/`ERROR` |
| `last_evidence` | ID \| null | M | |
| `provenance` | PROVENANCE | I | |

---

## 6. EVIDENCE

Единица знания о состоянии. Подробности и иерархия доверия — EVIDENCE_MODEL.md.

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `evidence_id` | ID | I | |
| `evidence_type` | enum | I | `MECHANICAL`/`EXECUTION`/`EXACT_MATCH`/`STATIC_ANALYSIS`/`EXTERNAL_DATA`/`MULTI_SOURCE_AGREEMENT`/`MODEL_JUDGEMENT`/`SELF_JUDGEMENT`/`HUMAN`/`UNKNOWN` |
| `subject` | ID | I | claim / artifact / dependency, о котором говорит |
| `assertion` | enum | I | `SUPPORTS`/`REFUTES`/`INCONCLUSIVE` |
| `source` | struct | I | кто/что породило (модель, тест-раннер, внешний источник, человек) |
| `method` | string | I | как получено |
| `strength` | enum | I | `HARD`/`MODERATE`/`WEAK`/`NONE` (вычисляется по типу, см. EVIDENCE_MODEL.md §3) |
| `independence_axis` | enum \| null | I | `NONE`/`SEED`/`TEMPERATURE`/`PROMPT`/`MODEL`/`FAMILY`/`EXTERNAL` — **обязательно для MULTI_SOURCE_AGREEMENT** |
| `reproducibility` | enum | I | `DETERMINISTIC`/`STOCHASTIC`/`ONE_SHOT` |
| `timestamp` | TIMESTAMP | I | |
| `provenance` | PROVENANCE | I | |

**Поле `independence_axis` — прямое следствие F1/F2/F5/F6/F7.** Без него система не может отличить «два независимых источника согласились» от «одна модель дважды повторила свою же систематическую ошибку» (что в этих данных происходило в ~75% случаев). Согласие по оси `SEED` — почти не evidence; согласие по оси `MODEL`/`FAMILY` — существенно более сильное.

**Evidence никогда не удаляется и не «обновляется».** Противоречащие evidence по одному subject сосуществуют; их совместное наличие — это и есть коллизия типа `EVIDENCE_CONFLICT`.

---

## 7. STATE (снимок состояния узла)

`state` живёт на claim/artifact как поле, но каждое его изменение материализуется как отдельная запись:

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `state_record_id` | ID | I | |
| `subject` | ID | I | claim или artifact |
| `subject_version` | VERSION | I | |
| `from_state` | enum \| null | I | |
| `to_state` | enum | I | |
| `trigger` | enum | I | `EVIDENCE_ADDED`/`UPSTREAM_CHANGED`/`ACTION_COMPLETED`/`COLLISION_DETECTED`/`REPAIR_APPLIED`/`MANUAL` |
| `trigger_ref` | ID | I | |
| `graph_version` | VERSION | I | версия графа на момент перехода |
| `timestamp` | TIMESTAMP | I | |

Это append-only журнал. Он даёт откат (аудит J) и детекцию повторяющегося состояния (STOP-условие).

---

## 8. COLLISION

Подробности — COLLISION_MODEL.md.

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `collision_id` | ID | I | |
| `collision_type` | enum | I | см. COLLISION_MODEL.md §2 |
| `participants` | [ID] | I | ≥1 claim/artifact/evidence |
| `evidence` | [ID] | I | что именно наблюдалось |
| `origin` | ID \| null | M | вычисляется; `null` пока не локализовано |
| `origin_confidence` | CONFIDENCE | M | |
| `severity` | enum | M | `BLOCKING`/`MAJOR`/`MINOR`/`INFORMATIONAL` |
| `status` | enum | M | `OPEN`/`LOCALIZED`/`PLANNED`/`REPAIRING`/`RESOLVED`/`UNRESOLVABLE`/`ACCEPTED` |
| `iteration_count` | int | M | защита от бесконечного ремонта |
| `timestamp` | TIMESTAMP | I | |

**`ACCEPTED`** — легитимный терминальный статус: коллизия признана, но не устраняется (нет средств проверки, или бюджет исчерпан, или риск задачи низкий). Это не «ошибка системы», это честное состояние (см. STOP-условия).

---

## 9. AFFECTED_REGION

Материализуется как объект, потому что три множества внутри него — разные и все три нужны для аудита.

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `region_id` | ID | I | |
| `collision_id` | ID | I | |
| `origin` | ID | I | |
| `candidate_affected_set` | [ID] | I | транзитивное замыкание по DAG — **безопасная верхняя граница** (F34: recall=1.0) |
| `verified_affected_set` | [ID] | M | подмножество, где проверка подтвердила фактическое влияние |
| `unaffected_confirmed_set` | [ID] | M | подмножество, где проверка показала, что влияния нет (F38: те случаи, где downstream остался корректен) |
| `unknown_set` | [ID] | M | проверка не применима/не проведена |
| `graph_version` | VERSION | I | |

Разделение `candidate` / `verified` / `unknown` — прямое следствие F37 (граф шире истины) + F38 (сужать по умолчанию нельзя). Без явного `unknown_set` система была бы вынуждена относить непроверяемое либо к «затронуто» (лишний ремонт), либо к «не затронуто» (риск пропуска).

---

## 10. REPAIR_PLAN

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `plan_id` | ID | I | |
| `collision_id` | ID | I | |
| `region_id` | ID | I | |
| `entries` | [PLAN_ENTRY] | I | по одной на claim |
| `estimated_cost` | COST | I | |
| `alternatives_considered` | [{strategy, estimated_cost}] | I | **обязательно**: минимум `FULL_RETRY` и `LOCAL_REPAIR` |
| `selected_strategy` | enum | I | |
| `selection_basis` | enum | I | `COST`/`RISK`/`POLICY_RULE`/`BUDGET_CONSTRAINT` |
| `plan_hash` | HASH | I | для детекции повторяющегося плана (осцилляция) |
| `provenance` | PROVENANCE | I | |

```
PLAN_ENTRY := {
  claim_id       : ID
  necessity      : "MUST_REGENERATE"|"MUST_VERIFY"|"MAY_REUSE"|"DO_NOT_TOUCH"|"UNKNOWN"
  action         : ACTION_TYPE | null
  rationale      : enum          # почему именно так (структурная причина, не свободный текст)
  estimated_cost : COST
}
```

`alternatives_considered` обязателен, потому что F47 прямо показал: локальный ремонт **не всегда** дешевле полного. План, не сравнивший стоимости, архитектурно некорректен.

---

## 11. ACTION

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `action_id` | ID | I | |
| `action_type` | enum | I | см. ACTION_MODEL.md |
| `target` | ID | I | claim/artifact/task |
| `preconditions_checked` | [enum] | I | |
| `input_context_ref` | ID | I | что реально ушло в модель (CONTEXT_MODEL.md) |
| `model_id` | ID \| null | I | |
| `prompt_profile_id` | ID \| null | I | |
| `sampling_config` | SAMPLING \| null | I | |
| `estimated_cost` | COST | I | до выполнения |
| `actual_cost` | COST \| null | M→I | после выполнения, далее immutable |
| `result_refs` | [ID] | A | произведённые артефакты/evidence |
| `status` | enum | M | `PLANNED`/`RUNNING`/`COMPLETED`/`FAILED`/`ABORTED` |
| `failure_mode` | enum \| null | M | см. FAILURE_MODES.md |
| `provenance` | PROVENANCE | I | |

---

## 12. RUN

Один физический вызов вычислительного оператора (обычно LLM). Отделён от ACTION, потому что одно действие может потребовать нескольких вызовов (ретраи формата и т.п.), и все они должны сохраняться.

| Поле | Тип | Изм. | Описание |
|---|---|---|---|
| `run_id` | ID | I | |
| `action_id` | ID | I | |
| `model_id` | ID | I | |
| `prompt_profile_id` | ID | I | |
| `sampling_config` | SAMPLING | I | |
| `rendered_prompt` | text | I | **полный, как отправлено** |
| `raw_output` | text | I | **сырой, до любого парсинга** |
| `input_tokens` / `output_tokens` | int | I | |
| `wall_time_sec` | float | I | |
| `failed` | bool | I | |
| `error` | text \| null | I | |
| `timestamp` | TIMESTAMP | I | |

**`raw_output` сохраняется всегда и никогда не удаляется**, даже если парсинг провалился и артефакт не создан. Обоснование прямое: за девять экспериментов не менее шести дефектов проверяющих правил были найдены **пересчётом по сохранённым сырым данным** (F45). Без сырых данных эти дефекты были бы неотличимы от свойств моделей.

---

## 13. MODEL_PROFILE

Наблюдаемые характеристики, **не оценочные ярлыки**. Подробности — MODEL_INTERFACE.md.

| Поле | Тип | Изм. |
|---|---|---|
| `profile_id` | ID | I |
| `model_id` | ID | I |
| `scope` | {artifact_type, task_class, prompt_profile_id, sampling_config} | I |
| `observations` | [OBSERVATION] | A |
| `artifact_success_rate` | float \| null | M |
| `error_classes` | {class: count} | M |
| `verification_success_rate` | float \| null | M |
| `repair_success_rate` | float \| null | M |
| `format_reliability` | float \| null | M |
| `mean_cost` / `mean_latency` | COST/float | M |
| `context_capacity` | int | I |
| `known_failure_modes` | [enum] | A |
| `sample_size` | int | M |

**`scope` обязателен.** Профиль «модель X хорошая» бессмысленен: F3 и F4 показали, что и температура, и промпт меняют результат разнонаправленно и по-разному у разных моделей, а F25 — что даже экономика локального ремонта у одной модели положительна, у другой отрицательна. Профиль без scope усредняет то, что усреднять нельзя.

**`sample_size` обязателен** — чтобы отличить «модель плоха на этом типе» от «мы видели её один раз».

---

## 14. PROMPT_PROFILE

| Поле | Тип | Изм. |
|---|---|---|
| `prompt_profile_id` | ID | I |
| `template` | text | I |
| `expected_artifact_type` | enum | I |
| `output_contract` | struct | I | ожидаемая структура (секции/схема), для SEAM_FORMAT |
| `version` | VERSION | I |

Промпт — **не свойство модели**, а отдельный оператор конфигурации. Один и тот же чекпоинт с разными `prompt_profile` — это разные вычислительные операторы с разными профилями (F4).

---

## 15. BUDGET

| Поле | Тип | Изм. |
|---|---|---|
| `budget_id` | ID | I |
| `token_budget` / `token_spent` | int | I / M |
| `time_budget` / `time_spent` | float | I / M |
| `call_budget` / `call_spent` | int | I / M |
| `model_budget` | {model_id: int} | I |
| `retry_budget` | int | I |
| `retry_spent` | int | M |
| `policy_on_exhaustion` | enum | I | `STOP`/`DEGRADE`/`ASK_HUMAN` |

```
COST := { tokens: int, time_sec: float, calls: int, model_id: ID|null }
```

Отдельная строка про graph-анализ: он **тоже** имеет стоимость (F41 — она мала для малых графов, но растёт экспоненциально для точного перебора MRS). `COST` для внутренних вычислений заполняется полем `time_sec` при нулевых `tokens`.

---

## 16. Сводная схема связей

```
TASK ──1:N──► ARTIFACT ──1:N──► CLAIM
 │                │                │
 │                │                ├──N:M──► DEPENDENCY ──► CLAIM
 │                │                ├──1:N──► EVIDENCE
 │                │                └──1:N──► STATE_RECORD
 │                │
 │                └──N:M──► SEAM ──► EVIDENCE
 │
 ├──1:N──► COLLISION ──1:1──► AFFECTED_REGION ──1:1──► REPAIR_PLAN
 │                                                          │
 │                                                          └──1:N──► ACTION ──1:N──► RUN
 └──1:1──► BUDGET

MODEL_PROFILE ◄── (обновляется наблюдениями из) ── RUN + EVIDENCE
PROMPT_PROFILE ◄── (ссылается) ── ACTION / RUN
```
