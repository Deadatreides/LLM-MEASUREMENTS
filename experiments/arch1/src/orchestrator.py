"""API / ORCHESTRATOR — уровень 6 (API_BOUNDARIES.md §1, §13).

Главный цикл (ARCHITECTURE.md §2) на одной задаче: единственный модуль,
которому позволено видеть все нижележащие уровни целиком.

Не привязан к конкретной задачной схеме (tasks/branch_tasks.py) —
run_task() принимает структуру claims/зависимостей/промптов/oracle как
параметры, чтобы orchestrator.py оставался переиспользуемым для любой
задачи с фиксированным (не выводимым) графом и механическим oracle.

Область этого этапа (WORK_PLAN.md, Этап 6):
- DECOMPOSE не вызывается: граф для этого класса задач зафиксирован
  протоколом, не выводится моделью (WORK_PLAN §8 — DECOMPOSE как
  действие модели вне ARCH-1 целиком).
- Механическая проверка — только для claims с известным oracle
  (обычно VALUE). Остальные остаются честно UNVERIFIED — прямая
  реализация траектории «hello» (ARCHITECTURE.md §6): непроверяемое
  доезжает до результата как явный статус, не как ошибка.
- Ремонт использует тот же оператор (без CHANGE_MODEL) — переключение
  GGUF-модели на лету не нужно для критерия «один таск проходит цикл
  целиком»; policy.preferred_action_for_collision существует, но здесь
  не вызывается — задокументированное сужение, не молчаливое упущение.

Внешний API (submit_task/get_state/get_provenance/rollback/
inject_evidence, API_BOUNDARIES §13) в этом этапе не реализован —
только run_task() для одной задачи; полный MyceliumAPI не входит в
критерий готовности этапа 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .action_executor import COMPLETED, GENERATE, REGENERATE, VERIFY, ActionExecutor
from .collision import Collision, localize
from .context_builder import build_context
from .repair_planner import (
    COLLISION_RESOLVED,
    DO_NOT_TOUCH,
    EXTERNAL_EVIDENCE_REQUIRED,
    MAX_ITERATIONS,
    MAY_REUSE,
    MUST_REGENERATE,
    RepairPlanner,
)
from .state_engine import INCORRECT, StateEngine
from .storage import Provenance, Sampling, Storage, new_id, now_iso


@dataclass(frozen=True)
class Task:
    """TASK (DATA_MODEL.md §1), обрезано до полей, которые действительно
    заполняет и использует run_task()."""

    task_id: str
    statement: str
    task_class: str
    risk_level: str
    oracle_availability: str
    state: str
    stop_reason: Optional[str]
    provenance: Provenance = field(default_factory=lambda: Provenance(created_by="SYSTEM"))
    timestamp: str = field(default_factory=now_iso)


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    claim_states: dict
    collisions_detected: int
    collisions_resolved: int
    iterations: int
    stop_reason: str
    graph_version: int


def _topological_order(claims: tuple, dependency_edges: tuple) -> list:
    indegree = {c: 0 for c in claims}
    children: dict[str, list] = {c: [] for c in claims}
    for source, target, _dep_type in dependency_edges:
        children[source].append(target)
        indegree[target] += 1

    frontier = sorted(c for c in claims if indegree[c] == 0)
    order = []
    while frontier:
        cur = frontier.pop(0)
        order.append(cur)
        for child in sorted(children[cur]):
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)
        frontier.sort()
    return order


def _verify_against_oracle(executor: ActionExecutor, graph: Any, claim_id: str, expected: float):
    """Передаёт шву СЫРОЙ текст claim'а, а не заранее извлечённое число.

    До этапа 8 извлечение («последнее число в тексте») делалось здесь и
    потому не покрывалось ни self_test шва, ни гейтом доверия — и дало
    59.6% ложных FAIL на кампании этапа 7. Теперь извлечение живёт
    внутри numeric_seam, под его контрольными случаями."""
    content = graph.get_claim(claim_id).content
    action = executor.build_action(VERIFY, claim_id)
    return executor.execute(
        action, seam_id="seam:numeric", seam_inputs={"text": content, "expected": float(expected)}
    )


def run_task(
    task_id: str,
    task: dict,
    claims: tuple,
    claim_type: dict,
    dependency_edges: tuple,
    prompt_fn: Callable[[str, dict], str],
    oracle_fn: Callable[[dict], dict],
    graph: Any,
    state: StateEngine,
    seams: Any,
    evidence: Any,
    planner: RepairPlanner,
    executor: ActionExecutor,
    storage: Storage,
    max_collision_iterations: int = 5,
    plan_collector: Optional[list] = None,
    sampling: Optional[Sampling] = None,
) -> TaskResult:
    """ARCHITECTURE.md §2 — главный цикл, на одной задаче.

    plan_collector — если передан список, каждый построенный RepairPlan
    добавляется в него. RepairPlanner не хранит планы (нет ссылки на
    Storage, см. STATUS.md решение Р2 этапа 4) — без этого крючка
    сравнение стоимости LOCAL_REPAIR/FULL_RETRY (этап 7) было бы
    невозможно восстановить после run_task(). Необязательный параметр,
    по умолчанию ничего не собирает — обратно совместимо.

    sampling — если передан, используется для ВСЕХ GENERATE/REGENERATE
    этого прогона (нужно этапу 7: разные seed/temperature по кампании).
    По умолчанию None => ActionExecutor берёт свой _DEFAULT_SAMPLING
    (temperature=0.0) — поведение этапов 5-6 не меняется.
    """

    artifact_version_id = graph.add_artifact_version(
        artifact_type="DERIVATION", task_id=task_id, content=task["question"]
    )
    artifact_id = storage.get(artifact_version_id).payload.artifact_id

    order = _topological_order(claims, dependency_edges)
    edges_by_target: dict[str, list] = {}
    for source, target, dep_type in dependency_edges:
        edges_by_target.setdefault(target, []).append((source, dep_type))

    # 1-2. GENERATE в топологическом порядке (без DECOMPOSE — структура
    # фиксирована заранее). Рёбра добавляются СРАЗУ ПОСЛЕ генерации
    # claim'а: add_dependency требует существования обоих концов, а
    # DEPENDENCY_CONTEXT для генерации передаётся явно (parent'ы уже
    # известны из dependency_edges, до появления самого ребра в графе).
    for claim_id in order:
        parent_ids = tuple(source for source, _dep_type in edges_by_target.get(claim_id, []))
        context = build_context(
            GENERATE,
            claim_id,
            graph,
            task_statement=prompt_fn(claim_id, task),
            dependency_claim_ids=parent_ids,
        )
        action = executor.build_action(GENERATE, claim_id, sampling=sampling)
        result = executor.execute(action, context, artifact_id=artifact_id, claim_type=claim_type[claim_id])
        if result.status != COMPLETED:
            return _finalize(
                task_id, task, claims, state, storage, graph,
                f"GENERATE_FAILED:{claim_id}:{result.failure_mode}", 0, 0, 0,
            )

        for source, dep_type in edges_by_target.get(claim_id, []):
            graph.add_dependency(
                source_claim=source, target_claim=claim_id,
                dependency_type=dep_type, status="CONFIRMED", created_by="HUMAN",
            )

    # 3. EVALUATE SEAMS -> EVIDENCE, только для claims с oracle (§ решение 2)
    oracle = oracle_fn(task)
    for claim_id, expected in oracle.items():
        _verify_against_oracle(executor, graph, claim_id, expected)

    # 4-7. DETECT COLLISIONS -> LOCALIZE -> PLAN REPAIR -> назад в EXECUTE
    iterations = 0
    collisions_detected = 0
    collisions_resolved = 0

    while True:
        defective = {cid for cid in oracle if state.current_state(cid) == INCORRECT}
        if not defective:
            collisions_resolved += 1 if collisions_detected else 0
            return _finalize(
                task_id, task, claims, state, storage, graph,
                COLLISION_RESOLVED, iterations, collisions_detected, collisions_resolved,
            )

        if iterations >= max_collision_iterations:
            return _finalize(
                task_id, task, claims, state, storage, graph,
                MAX_ITERATIONS, iterations, collisions_detected, collisions_resolved,
            )

        iterations += 1
        collisions_detected += 1

        col = Collision(
            collision_id=new_id("col"),
            collision_type="MECHANICAL_FAILURE",
            participants=tuple(sorted(defective)),
        )
        localize(col, graph, defective_claims=defective)

        if col.origin is None:
            return _finalize(
                task_id, task, claims, state, storage, graph,
                EXTERNAL_EVIDENCE_REQUIRED, iterations, collisions_detected, collisions_resolved,
            )

        region = planner.affected_region(col)

        def seam_inputs_provider(claim_id, seam, _oracle=oracle):
            # СЫРОЙ текст шву -- извлечение внутри numeric_seam, см.
            # _verify_against_oracle выше (исправление этапа 8).
            if seam.seam_id != "seam:numeric" or claim_id not in _oracle:
                return None
            return {
                "text": graph.get_claim(claim_id).content,
                "expected": float(_oracle[claim_id]),
            }

        narrowed = planner.narrow(region, seam_inputs_provider)

        # STATE_MACHINE §4.1 / REPAIR_MODEL §8: origin -> INCORRECT
        # распространяет STALE на прежде-CORRECT потомков. Наш цикл ниже
        # всё равно заново VERIFY-ит весь region с oracle, поэтому здесь
        # это в первую очередь про честную историю состояний (аудит), не
        # про то, от чего зависит корректность.
        for origin_claim in col.origin:
            state.propagate_stale(origin_claim)

        plan = planner.build_plan(narrowed, col, task={"risk_level": "MEDIUM"})
        if plan_collector is not None:
            plan_collector.append(plan)

        stop_reason = planner.check_stop_conditions(plan, col)
        if stop_reason:
            return _finalize(
                task_id, task, claims, state, storage, graph,
                stop_reason, iterations, collisions_detected, collisions_resolved,
            )
        planner.record_iteration(col, plan)

        do_not_touch = tuple(
            e.claim_id for e in plan.entries if e.necessity in (MAY_REUSE, DO_NOT_TOUCH)
        )

        for entry in plan.entries:
            if entry.necessity != MUST_REGENERATE:
                continue
            parent_ids = tuple(source for source, _dep_type in edges_by_target.get(entry.claim_id, []))
            context = build_context(
                REGENERATE, entry.claim_id, graph,
                task_statement=prompt_fn(entry.claim_id, task),
                do_not_touch=do_not_touch,
                dependency_claim_ids=parent_ids,
            )
            action = executor.build_action(REGENERATE, entry.claim_id, sampling=sampling)
            executor.execute(action, context)

        # повторная проверка (§8: "назад в шаг 2") -- официальный канал
        # (ActionExecutor.VERIFY), а не внутренняя оценка narrow()
        # (которая не пишет evidence/state, см. STATUS.md этапа 4-5).
        for claim_id in region.candidate_affected_set:
            if claim_id in oracle:
                _verify_against_oracle(executor, graph, claim_id, oracle[claim_id])

        planner.post_repair_check(plan.repair_set, seam_inputs_provider)
        # цикл: следующая итерация while заново проверит defective


def _finalize(
    task_id: str,
    task: dict,
    claims: tuple,
    state: StateEngine,
    storage: Storage,
    graph: Any,
    stop_reason: str,
    iterations: int,
    collisions_detected: int,
    collisions_resolved: int,
) -> TaskResult:
    claim_states = {claim_id: state.current_state(claim_id) for claim_id in claims}
    task_state = "RESOLVED" if stop_reason == COLLISION_RESOLVED else "STOPPED"

    task_record = Task(
        task_id=task_id,
        statement=task["question"],
        task_class="MATH",
        risk_level="MEDIUM",
        oracle_availability="PARTIAL",  # часть claims (C1-C3) без механического oracle
        state=task_state,
        stop_reason=stop_reason,
    )
    storage.append(task_record.task_id, "task", task_record)

    return TaskResult(
        task_id=task_id,
        claim_states=claim_states,
        collisions_detected=collisions_detected,
        collisions_resolved=collisions_resolved,
        iterations=iterations,
        stop_reason=stop_reason,
        graph_version=graph.current_graph_version(),
    )
