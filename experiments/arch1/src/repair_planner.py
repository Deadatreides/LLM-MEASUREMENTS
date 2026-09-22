"""REPAIR PLANNER — уровень 5 (API_BOUNDARIES.md §8).

Центральный модуль архитектуры (REPAIR_MODEL.md). Три множества и
шаг сужения: candidate_affected -> narrow() -> repair_set.

Ни одного вызова LLM: model_adapter здесь не импортируется, вся логика
детерминирована (WORK_PLAN.md, Этап 4).

Гарантии:
- build_plan всегда заполняет alternatives_considered, включая
  FULL_RETRY (EVERY_PLAN_COMPARES_ALTERNATIVES);
- narrow никогда не переводит узел из candidate в «не затронуто» без
  evidence (F38) — см. docstring narrow();
- планировщик формирует множество допустимых действий; выбор делает
  POLICY (RuleBasedPolicy, уровень 4 — зависимость разрешена).

Решение по толкованию (утверждено пользователем, см. STATUS.md): §3
говорит «проверить c относительно НОВОГО состояния предков», но narrow()
работает ДО фактического ремонта — предки ещё не перегенерированы.
Реализована АБСОЛЮТНАЯ проверка: шов проверяет c против собственного
механического oracle, независимо от предка; PASS => c верен сам по
себе => unaffected => MAY_REUSE. «Новое состояние предков» из формулировки
реализуется топологическим порядком обхода: к моменту проверки c статус
всех его предков в candidate_set уже определён.

Поправка AMD-1 (REPAIR_MODEL.md §3.2, принята по итогам измерения
ARCH-1): порядок шагов в narrow() изменён — проверка идёт ПЕРВОЙ,
структурная эвристика («значение предка буквально присутствует в c»)
понижена до умолчания для узлов, которые проверить не удалось. В
исходной редакции эвристика стояла первой и перехватывала 87.2%
не-origin узлов, выводя валидность из структурного факта вопреки
ARCHITECTURE.md §1.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .graph_core import CYCLE_DETECTED, GraphCore
from .seam_engine import FAIL, PASS, Seam, SeamEngine
from .state_engine import CORRECT, StateEngine
from .storage import Cost, Provenance, new_id

# -- necessity (REPAIR_MODEL.md §4.1) ---------------------------------------

MUST_REGENERATE = "MUST_REGENERATE"
MUST_VERIFY = "MUST_VERIFY"
MAY_REUSE = "MAY_REUSE"
DO_NOT_TOUCH = "DO_NOT_TOUCH"
NECESSITY_UNKNOWN = "UNKNOWN"

# -- стратегии (REPAIR_MODEL.md §4.3) ---------------------------------------

FULL_RETRY = "FULL_RETRY"
LOCAL_REPAIR = "LOCAL_REPAIR"
VERIFY_FIRST = "VERIFY_FIRST"
MODEL_SWITCH = "MODEL_SWITCH"
PROMPT_SWITCH = "PROMPT_SWITCH"

# -- selection_basis (DATA_MODEL.md §10) ------------------------------------

BASIS_COST = "COST"
BASIS_RISK = "RISK"
BASIS_POLICY_RULE = "POLICY_RULE"
BASIS_BUDGET_CONSTRAINT = "BUDGET_CONSTRAINT"

# -- STOP-условия (REPAIR_MODEL.md §7) --------------------------------------

COLLISION_RESOLVED = "COLLISION_RESOLVED"
NO_COLLISION_AND_NO_VERIFICATION = "NO_COLLISION_AND_NO_VERIFICATION"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
REPEATED_STATE = "REPEATED_STATE"
REPAIR_OSCILLATION = "REPAIR_OSCILLATION"
MAX_ITERATIONS = "MAX_ITERATIONS"
MAX_DEPTH = "MAX_DEPTH"
NO_EXPECTED_GAIN = "NO_EXPECTED_GAIN"
EXTERNAL_EVIDENCE_REQUIRED = "EXTERNAL_EVIDENCE_REQUIRED"
UNCERTAINTY_TOO_HIGH = "UNCERTAINTY_TOO_HIGH"

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LITERAL_VALUE_DEPENDENCY_TYPES = ("DATA", "COMPUTATIONAL")


# -- сущности (DATA_MODEL.md §9-10) -----------------------------------------


@dataclass(frozen=True)
class AffectedRegion:
    region_id: str
    collision_id: str
    origin: tuple[str, ...]
    candidate_affected_set: frozenset
    verified_affected_set: frozenset = frozenset()
    unaffected_confirmed_set: frozenset = frozenset()
    unknown_set: frozenset = frozenset()
    graph_version: int = 0


@dataclass(frozen=True)
class PlanEntry:
    claim_id: str
    necessity: str
    action: Optional[str]
    rationale: str
    estimated_cost: Cost


@dataclass(frozen=True)
class RepairPlan:
    plan_id: str
    collision_id: str
    region_id: str
    entries: tuple[PlanEntry, ...]
    estimated_cost: Cost
    alternatives_considered: tuple[dict, ...]  # [{strategy, cost, feasible}]
    selected_strategy: str
    selection_basis: str
    plan_hash: str
    provenance: Provenance = field(default_factory=lambda: Provenance(created_by="SYSTEM"))

    @property
    def repair_set(self) -> frozenset:
        return frozenset(
            e.claim_id for e in self.entries if e.necessity != DO_NOT_TOUCH
        )


def _default_cost_estimator(action_type: str, claim_id: Optional[str] = None) -> Cost:
    """Заглушка-плейсхолдер: реальная стоимость зависит от MODEL_PROFILE
    (появится не раньше этапа 5). Отношения между действиями сохранены
    качественно (VERIFY дешевле REGENERATE дешевле FULL_RETRY по всей
    задаче) — этого достаточно для сравнения стратегий на этом этапе."""
    if action_type == "VERIFY":
        return Cost(tokens=0, time_sec=0.01, calls=1)
    if action_type == "REGENERATE":
        return Cost(tokens=200, time_sec=1.0, calls=1)
    if action_type == FULL_RETRY:
        return Cost(tokens=1200, time_sec=6.0, calls=6)
    return Cost()


def _sum_costs(costs: Iterable[Cost]) -> Cost:
    total = Cost()
    for c in costs:
        total = total + c
    return total


class RepairPlanner:
    """REPAIR PLANNER (API_BOUNDARIES.md §8)."""

    def __init__(
        self,
        graph: GraphCore,
        state_engine: StateEngine,
        seam_engine: SeamEngine,
        policy: Any = None,
        cost_estimator: Callable[[str, Optional[str]], Cost] = _default_cost_estimator,
        max_iterations: int = 10,
    ) -> None:
        self._graph = graph
        self._state = state_engine
        self._seams = seam_engine
        self._policy = policy
        self._cost_estimator = cost_estimator
        self._max_iterations = max_iterations
        self._plan_hash_history: dict[str, list[str]] = {}
        self._seen_states: set[tuple[str, str]] = set()

    # -- §2: candidate_affected_set -----------------------------------------

    def affected_region(self, collision: Any) -> AffectedRegion:
        graph_version = self._graph.current_graph_version()

        if not collision.origin:
            # origin ещё не локализован -- консервативно: участники ∪ их downstream
            candidate = set(collision.participants)
            for participant in collision.participants:
                candidate |= self._graph.downstream_closure(participant)
            origin_tuple: tuple[str, ...] = tuple(collision.participants)
        else:
            candidate = set()
            for origin_claim in collision.origin:
                candidate.add(origin_claim)
                candidate |= self._graph.downstream_closure(origin_claim)
            origin_tuple = tuple(collision.origin)

        return AffectedRegion(
            region_id=new_id("region"),
            collision_id=collision.collision_id,
            origin=origin_tuple,
            candidate_affected_set=frozenset(candidate),
            graph_version=graph_version,
        )

    # -- §3: шаг сужения ------------------------------------------------------

    def narrow(
        self,
        region: AffectedRegion,
        seam_inputs_provider: Callable[[str, Seam], Optional[dict]],
        budget: Optional[Cost] = None,
    ) -> AffectedRegion:
        """REPAIR_MODEL.md §3.

        seam_inputs_provider(claim_id, seam) -> inputs-словарь для
        evaluate() или None, если этот конкретный шов неприменим к этому
        claim'у (нет данных для оценки). Порядок обхода — топологический
        (§3.1.1): предки candidate_set обрабатываются раньше потомков.
        """
        origin_set = set(region.origin)
        verified: set[str] = set(origin_set)  # корни всегда затронуты
        unaffected: set[str] = set()
        unknown: set[str] = set()

        order = [
            c
            for c in self._graph.topological_order(region.candidate_affected_set)
            if c not in origin_set
        ]

        for claim_id in order:
            # ШАГ 1 (поправка AMD-1): СНАЧАЛА проверка -- evidence имеет
            # приоритет над структурой (ARCHITECTURE.md §1). До поправки
            # структурная эвристика стояла первой и перехватывала 87.2%
            # не-origin узлов, не давая проверке отработать.
            seam, inputs = self._find_applicable_seam(claim_id, seam_inputs_provider)

            check_cost = self._cost_estimator("VERIFY", claim_id)
            regen_cost = self._cost_estimator("REGENERATE", claim_id)
            too_expensive = (
                check_cost.tokens > regen_cost.tokens or check_cost.time_sec > regen_cost.time_sec
            )  # проверять дороже, чем чинить (F41, F47)

            if seam is not None and not too_expensive:
                result = self._seams.evaluate(seam.seam_id, inputs)
                # Недоверенный шов НЕ может сузить область, даже при
                # PASS/FAIL: перевод в unaffected эквивалентен
                # производству HARD-evidence (SEAM_MODEL.md §5, гейт
                # этапа 3). Такой результат -> в шаг 2, как непроверенный.
                if result.trusted and result.status == PASS:
                    unaffected.add(claim_id)  # F38: downstream уцелел
                    continue
                if result.trusted and result.status == FAIL:
                    verified.add(claim_id)
                    continue

            # ШАГ 2: проверки не было (нет шва / дороже ремонта / шов не
            # дал доверенного результата) -- только тогда структурная
            # эвристика, и только как КОНСЕРВАТИВНОЕ отнесение, не как
            # знание. Механически, без семантического сходства
            # (DEPENDENCY_MODEL.md §3.2).
            if self._parent_value_literally_present(claim_id, verified):
                verified.add(claim_id)
                continue

            unknown.add(claim_id)  # F43: честный unknown

        return AffectedRegion(
            region_id=region.region_id,
            collision_id=region.collision_id,
            origin=region.origin,
            candidate_affected_set=region.candidate_affected_set,
            verified_affected_set=frozenset(verified),
            unaffected_confirmed_set=frozenset(unaffected),
            unknown_set=frozenset(unknown),
            graph_version=region.graph_version,
        )

    def _find_applicable_seam(
        self, claim_id: str, seam_inputs_provider: Callable[[str, Seam], Optional[dict]]
    ) -> tuple[Optional[Seam], Optional[dict]]:
        for seam in self._seams.applicable_seams(claim_id, self._graph):
            inputs = seam_inputs_provider(claim_id, seam)
            if inputs is not None:
                return seam, inputs
        return None, None

    def _parent_value_literally_present(self, claim_id: str, verified_so_far: set[str]) -> bool:
        try:
            child_claim = self._graph.get_claim(claim_id)
        except KeyError:
            return False

        child_numbers = set(_NUMBER_RE.findall(child_claim.content))
        if not child_numbers:
            return False

        for parent_id, dependency_type in self._graph.direct_parent_edges(claim_id):
            if parent_id not in verified_so_far:
                continue
            if dependency_type not in _LITERAL_VALUE_DEPENDENCY_TYPES:
                continue
            try:
                parent_claim = self._graph.get_claim(parent_id)
            except KeyError:
                continue
            parent_numbers = set(_NUMBER_RE.findall(parent_claim.content))
            if parent_numbers & child_numbers:
                return True
        return False

    # -- §4.1: necessity per claim ----------------------------------------

    def assign_necessity(self, claim_id: str, region: AffectedRegion, task_risk_level: str = "MEDIUM") -> str:
        if claim_id in region.origin:
            return MUST_REGENERATE
        if claim_id in region.verified_affected_set:
            return MUST_REGENERATE
        if claim_id in region.unaffected_confirmed_set:
            return MAY_REUSE
        if claim_id not in region.candidate_affected_set:
            return DO_NOT_TOUCH  # F35: независимые ветви
        if claim_id in region.unknown_set:
            return MUST_REGENERATE if task_risk_level == "HIGH" else MUST_VERIFY
        return NECESSITY_UNKNOWN

    def _rationale_for(self, claim_id: str, region: AffectedRegion, necessity: str) -> str:
        if claim_id in region.origin:
            return "ORIGIN"
        if necessity == MUST_REGENERATE and claim_id in region.verified_affected_set:
            return "VERIFIED_BY_NARROW"
        if necessity == MAY_REUSE:
            return "UNAFFECTED_CONFIRMED"
        if necessity == DO_NOT_TOUCH:
            return "OUTSIDE_CANDIDATE_SET"
        if claim_id in region.unknown_set:
            return "UNKNOWN_HIGH_RISK" if necessity == MUST_REGENERATE else "UNKNOWN_LOW_RISK"
        return "OTHER"

    def _necessity_to_action(self, necessity: str) -> Optional[str]:
        return {
            MUST_REGENERATE: "REGENERATE",
            MUST_VERIFY: "VERIFY",
        }.get(necessity)

    # -- §4.3: план ремонта -------------------------------------------------

    def build_plan(self, region: AffectedRegion, collision: Any, task: Any = None, budget: Optional[Cost] = None) -> RepairPlan:
        risk_level = self._risk_level(task)

        entries = []
        for claim_id in region.candidate_affected_set:
            necessity = self.assign_necessity(claim_id, region, risk_level)
            action = self._necessity_to_action(necessity)
            cost = self._cost_estimator(action, claim_id) if action else Cost()
            entries.append(
                PlanEntry(
                    claim_id=claim_id,
                    necessity=necessity,
                    action=action,
                    rationale=self._rationale_for(claim_id, region, necessity),
                    estimated_cost=cost,
                )
            )
        entries.sort(key=lambda e: e.claim_id)

        local_repair_cost = _sum_costs(e.estimated_cost for e in entries if e.necessity != DO_NOT_TOUCH)
        full_retry_cost = self._cost_estimator(FULL_RETRY, None)
        verify_first_cost = _sum_costs(
            self._cost_estimator("VERIFY", e.claim_id)
            if e.necessity in (MUST_VERIFY, MUST_REGENERATE)
            else Cost()
            for e in entries
        )

        # MODEL_SWITCH/PROMPT_SWITCH: тот же объём работы, что LOCAL_REPAIR,
        # другой оператор/промпт -- реальная разница зависит от MODEL_PROFILE
        # (появится не раньше этапа 5), здесь не моделируется отдельно.
        strategies = (
            {"strategy": FULL_RETRY, "cost": full_retry_cost},
            {"strategy": LOCAL_REPAIR, "cost": local_repair_cost},
            {"strategy": VERIFY_FIRST, "cost": verify_first_cost},
            {"strategy": MODEL_SWITCH, "cost": local_repair_cost},
            {"strategy": PROMPT_SWITCH, "cost": local_repair_cost},
        )
        strategies_with_feasibility = tuple(
            {**s, "feasible": budget is None or s["cost"].fits_in(budget)} for s in strategies
        )

        if self._policy is not None:
            selected_strategy = self._policy.select(None, task, strategies_with_feasibility, budget=budget)
            basis = BASIS_POLICY_RULE
        else:
            feasible = [s for s in strategies_with_feasibility if s["feasible"]]
            pool = feasible or list(strategies_with_feasibility)
            selected_strategy = min(pool, key=lambda s: (s["cost"].tokens, s["cost"].time_sec))["strategy"]
            basis = BASIS_BUDGET_CONSTRAINT if budget is not None else BASIS_COST

        selected_cost = next(s["cost"] for s in strategies_with_feasibility if s["strategy"] == selected_strategy)
        plan_hash = self._compute_plan_hash(entries, selected_strategy)

        return RepairPlan(
            plan_id=new_id("plan"),
            collision_id=collision.collision_id,
            region_id=region.region_id,
            entries=tuple(entries),
            estimated_cost=selected_cost,
            alternatives_considered=strategies_with_feasibility,  # обязательно включает FULL_RETRY
            selected_strategy=selected_strategy,
            selection_basis=basis,
            plan_hash=plan_hash,
        )

    @staticmethod
    def _risk_level(task: Any) -> str:
        if task is None:
            return "MEDIUM"
        if isinstance(task, dict):
            return task.get("risk_level", "MEDIUM")
        return getattr(task, "risk_level", "MEDIUM")

    @staticmethod
    def _compute_plan_hash(entries: list[PlanEntry], selected_strategy: str) -> str:
        canonical = sorted((e.claim_id, e.necessity, e.action or "") for e in entries)
        blob = json.dumps({"entries": canonical, "strategy": selected_strategy}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- §7: STOP-условия -----------------------------------------------------

    def check_stop_conditions(self, plan: RepairPlan, collision: Any, budget: Optional[Cost] = None) -> Optional[str]:
        """REPAIR_MODEL.md §7.

        REPAIR_OSCILLATION и REPEATED_STATE — РАЗНЫЕ условия (решение,
        зафиксированное в STATUS.md): осцилляция сравнивает план N с
        планом N-2 (чередование двух планов); REPEATED_STATE смотрит на
        пару (graph_hash, plan_hash) когда-либо виденную для этой
        коллизии (шире осцилляции, ловит и не-чередующиеся повторы).
        """
        if collision.iteration_count >= self._max_iterations:
            return MAX_ITERATIONS

        history = self._plan_hash_history.get(collision.collision_id, [])
        if len(history) >= 2 and history[-2] == plan.plan_hash:
            return REPAIR_OSCILLATION

        graph_hash = self._graph.graph_hash(self._graph.current_graph_version())
        if (graph_hash, plan.plan_hash) in self._seen_states:
            return REPEATED_STATE

        if budget is not None and not plan.estimated_cost.fits_in(budget):
            return BUDGET_EXHAUSTED

        return None

    def record_iteration(self, collision: Any, plan: RepairPlan) -> None:
        """Зафиксировать план как рассмотренный для этой коллизии —
        после check_stop_conditions и до/вместо исполнения (REPAIR_MODEL
        §8: `seen_plan_hashes ← seen_plan_hashes ∪ {plan.plan_hash}`)."""
        graph_hash = self._graph.graph_hash(self._graph.current_graph_version())
        self._plan_hash_history.setdefault(collision.collision_id, []).append(plan.plan_hash)
        self._seen_states.add((graph_hash, plan.plan_hash))

    # -- §6: проверка после ремонта (hidden dependency) ------------------

    def post_repair_check(
        self,
        repair_set: Iterable[str],
        seam_inputs_provider: Callable[[str, Seam], Optional[dict]],
    ) -> list[tuple[str, str]]:
        """REPAIR_MODEL.md §6.

        Чистая функция от ТЕКУЩЕГО состояния графа/state — не требует
        фактического исполнения ремонта (ACTION EXECUTOR, этап 5):
        репарированные claims уже должны иметь новые версии/evidence к
        моменту вызова, post_repair_check лишь смотрит, не провалился ли
        шов у нетронутого claim'а, который раньше был CORRECT.

        Возвращает список найденных скрытых зависимостей (source, target).
        Каждая материализуется как ребро UNKNOWN_DEPENDENCY/HYPOTHESIS —
        по наблюдаемому факту (проверка провалилась), не по сходству.
        """
        repair_set = set(repair_set)
        downstream_of_repair: set[str] = set()
        for claim_id in repair_set:
            downstream_of_repair |= self._graph.downstream_closure(claim_id)
        candidates = downstream_of_repair - repair_set

        hidden: list[tuple[str, str]] = []
        for claim_id in sorted(candidates):
            if self._state.current_state(claim_id) != CORRECT:
                continue  # интересны только claims, БЫВШИЕ CORRECT

            seam, inputs = self._find_applicable_seam(claim_id, seam_inputs_provider)
            if seam is None:
                continue

            result = self._seams.evaluate(seam.seam_id, inputs)
            if not (result.trusted and result.status == FAIL):
                continue

            source = next(
                (r for r in repair_set if claim_id in self._graph.downstream_closure(r)), None
            )
            if source is None:
                continue

            dep_result = self._graph.add_dependency(
                source_claim=source,
                target_claim=claim_id,
                dependency_type="UNKNOWN_DEPENDENCY",
                status="HYPOTHESIS",
                created_by="SEAM_OBSERVATION",
            )
            if dep_result != CYCLE_DETECTED:
                hidden.append((source, claim_id))

        return hidden
