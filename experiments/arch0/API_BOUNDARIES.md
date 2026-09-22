# API_BOUNDARIES.md — границы модулей

## 1. Слои и направление зависимостей

Зависимости строго однонаправленны (сверху вниз). Циклы между модулями запрещены.

```
уровень 6   API / ORCHESTRATOR
uровень 5   REPAIR PLANNER  │  ACTION EXECUTOR  │  BUDGET MANAGER
уровень 4   POLICY (заменяемая)   │   MODEL ADAPTER
уровень 3   SEAM ENGINE  │  EVIDENCE ENGINE
уровень 2   STATE ENGINE
уровень 1   GRAPH CORE
уровень 0   STORAGE (append-only)
```

Правило: модуль уровня N может зависеть только от уровней < N. Обратные вызовы — только через явные интерфейсы, передаваемые сверху (инверсия зависимостей).

---

## 2. GRAPH CORE (уровень 1)

**Знает:** структуру. Артефакты, claims, зависимости, версии, обходы.
**Не знает:** ничего об LLM, о промптах, о стоимости, о политике.

```
interface GraphCore:
    # чтение
    get_claim(claim_id, version?) → Claim
    get_artifact(artifact_id, version?) → Artifact
    downstream_closure(claim_id) → Set[claim_id]
    upstream_closure(claim_id) → Set[claim_id]
    direct_parents(claim_id) → [claim_id]
    direct_children(claim_id) → [claim_id]
    topological_order(claim_set) → [claim_id]
    root_origins(defective_set) → [claim_id]

    # запись (создаёт версии, не перезаписывает)
    add_artifact_version(artifact) → artifact_version_id
    add_claim_version(claim) → claim_version_id
    add_dependency(dep) → dependency_id | CYCLE_DETECTED
    collapse_scc(claim_set) → composite_claim_id

    # версионирование
    graph_snapshot(graph_version) → GraphView
    graph_hash(graph_version) → HASH
```

**Гарантии:**
- `add_dependency` **никогда** не создаёт цикл: при обнаружении возвращает `CYCLE_DETECTED` (обработка — DEPENDENCY_MODEL.md §5);
- ни один метод не удаляет данные;
- `downstream_closure` детерминирован и полон (свойство, подтверждённое F34).

**Критично:** этот модуль не импортирует ничего, связанного с моделями. Проверка L аудита («заменить модель без изменения core») выполняется тем, что здесь нечего менять.

---

## 3. STATE ENGINE (уровень 2)

**Знает:** таблицу переходов, правила распространения STALE.
**Не знает:** откуда взялось evidence, сколько оно стоило.

```
interface StateEngine:
    apply_evidence(subject_id, evidence) → StateTransition | REJECTED
    propagate_stale(changed_claim_id) → Set[claim_id]
    current_state(subject_id) → State
    state_history(subject_id) → [StateRecord]
    validate_invariants(graph_version) → [Violation]
```

**Гарантии:**
- каждый переход порождает `STATE_RECORD` (инвариант `EVERY_STATE_CHANGE_IS_VERSIONED`);
- запрещённые переходы (STATE_MACHINE.md §2.1) отклоняются с `REJECTED`, а не выполняются молча;
- `apply_evidence` **не решает**, доверять ли evidence — сила уже проставлена EVIDENCE ENGINE.

---

## 4. EVIDENCE ENGINE (уровень 3)

**Знает:** типы evidence, иерархию силы, оси независимости.
**Не знает:** структуру графа (кроме идентификаторов subject).

```
interface EvidenceEngine:
    record(evidence) → evidence_id
    strength_of(evidence_type, source, independence_axis, reproducibility) → Strength
    evidence_for(subject_id) → [Evidence]
    resolve(subject_id) → {CORRECT|INCORRECT|UNKNOWN|CONFLICTED}
```

**Гарантии:**
- `strength_of` детерминирована и не зависит от того, какой ответ «хотелось бы» получить;
- evidence никогда не удаляется и не изменяется;
- `resolve` реализует §3.2 EVIDENCE_MODEL.md, включая запрет на накопление слабых до сильного.

---

## 5. SEAM ENGINE (уровень 3)

**Знает:** какие швы объявлены, как их исполнять, их контрольные случаи.
**Не знает:** состояние графа, стоимость.

```
interface SeamEngine:
    applicable_seams(subject_id, graph_view) → [Seam]
    evaluate(seam_id, inputs) → SeamResult{PASS|FAIL|INAPPLICABLE|ERROR}
    seam_class(seam_id) → HARD | SOFT | UNKNOWN
    self_test(seam_id) → {passed, failed_cases}   # контрольные случаи
    is_trusted(seam_id) → bool
```

**Гарантии:**
- шов, не прошедший `self_test`, имеет `is_trusted = false` и **не может** производить evidence силы `HARD` (SEAM_MODEL.md §5);
- `ERROR` (сбой самой проверки) никогда не транслируется в `FAIL`;
- SOFT-шов, вызывающий модель, делает это через MODEL ADAPTER — то есть SEAM ENGINE **зависит от интерфейса адаптера, но не от конкретной модели**.

---

## 6. MODEL ADAPTER (уровень 4)

```
interface ModelAdapter:
    capabilities() → Capabilities
    generate(rendered_prompt, sampling_config) → RawResult
```

**Гарантии:**
- возвращает **сырьё**, не парсит;
- не выносит суждений;
- сохранение `RawResult` обязательно на стороне вызывающего, **до** попытки разбора.

---

## 7. POLICY (уровень 4, заменяемая)

```
interface Policy:
    select(state_snapshot, task, candidate_actions, model_profiles, budget)
        → selected_action
```

**Жёсткие ограничения:**
- нет доступа на запись в граф, состояние, evidence;
- не может расширить множество кандидатов;
- заменяется целиком без изменений в уровнях 0–3.

В ARCH-0 — реализация `RuleBasedPolicy` по правилам REPAIR_MODEL.md §4.2.

---

## 8. REPAIR PLANNER (уровень 5)

```
interface RepairPlanner:
    localize(collision, graph_view) → Origin
    affected_region(collision, graph_view) → AffectedRegion
    narrow(region, graph_view, budget) → AffectedRegion   # с verified/unaffected/unknown
    build_plan(region, collision, task, budget) → RepairPlan
```

**Гарантии:**
- `build_plan` всегда заполняет `alternatives_considered` включая `FULL_RETRY` (инвариант `EVERY_PLAN_COMPARES_ALTERNATIVES`);
- `narrow` никогда не переводит узел из `candidate` в «не затронуто» без evidence (F38);
- планировщик формирует **множество допустимых** действий; выбор делает POLICY.

---

## 9. ACTION EXECUTOR (уровень 5)

```
interface ActionExecutor:
    check_preconditions(action, graph_view) → ok | [violations]
    execute(action, context) → ActionResult
```

**Гарантии:**
- сырьё сохраняется до разбора и при любом исходе;
- `actual_cost` фиксируется всегда;
- новая версия создаётся, старая помечается `superseded_by` — не удаляется.

---

## 10. BUDGET MANAGER (уровень 5)

```
interface BudgetManager:
    estimate(action, model_profile, context) → Cost
    reserve(cost) → ok | DENIED
    commit(actual_cost)
    release(reserved)
    remaining() → Cost
    compare_strategies([strategy]) → [{strategy, cost, feasible}]
```

---

## 11. STORAGE (уровень 0)

```
interface Storage:
    append(record) → record_id       # ЕДИНСТВЕННЫЙ метод записи
    get(record_id) → Record
    query(filter) → [Record]
    snapshot_at(graph_version) → GraphView
```

**Гарантии:**
- **нет** методов `update` и `delete`;
- сырые выводы моделей (`RUN.raw_output`) хранятся всегда и полностью;
- любое состояние графа восстановимо по `graph_version` (аудит J).

Обоснование append-only: F45 — все найденные за проект дефекты проверяющих правил обнаружены пересчётом по сохранённому сырью. Возможность перезаписи уничтожила бы этот механизм.

---

## 12. Запрещённые зависимости (проверяемые статически)

```
GRAPH CORE   →  ничего, кроме STORAGE
STATE ENGINE →  GRAPH CORE, STORAGE
SEAM ENGINE  →  STORAGE, ModelAdapter (интерфейс, не реализация)
EVIDENCE ENG →  STORAGE
POLICY       →  только чтение снимков; НЕТ записи никуда
любой модуль →  НЕ импортирует конкретную реализацию LLM
```

Проверка: линтер импортов в CI. Нарушение — ошибка сборки, а не замечание.

---

## 13. Внешний API

```
interface MyceliumAPI:
    submit_task(statement, task_class, risk_level, budget) → task_id
    get_state(task_id) → {
        result_artifacts, claim_states, open_collisions,
        unknown_claims, stale_claims, stop_reason, budget_spent
    }
    get_provenance(subject_id) → LineageTree
    rollback(task_id, graph_version) → new_graph_version
    inject_evidence(subject_id, evidence)     # внешний источник / человек
```

**`get_state` обязан возвращать разметку состояний, а не «ответ».** Инвариант `UNKNOWN_IS_NOT_ERROR` действует на границе API: непроверенное и неизвестное доезжает до потребителя явно (см. ARCHITECTURE.md §6 — траектория «привет» заканчивается результатом с маркером `UNVERIFIED`, а не тихим «готово»).

**`rollback` не удаляет** — переключает указатель текущей версии. Все версии остаются.
