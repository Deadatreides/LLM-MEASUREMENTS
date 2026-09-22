"""ACTION EXECUTOR — уровень 5 (API_BOUNDARIES.md §9).

Исполняет действия (GENERATE / REGENERATE / VERIFY / STOP) через
MODEL ADAPTER, с обязательным сохранением сырья до разбора.

PATCH/DECOMPOSE/MERGE не реализуются — WORK_PLAN.md §8: сознательно вне
ARCH-1 целиком, не ограничение этого модуля.

Область этого этапа (важно для будущих этапов):
- PROPOSED как отдельное отслеживаемое состояние не материализуется
  (STATE ENGINE, этап 2, не реализует PROPOSED<->GENERATED — интерфейс
  API_BOUNDARIES §3 не даёт для этого метода). Раз в ARCH-1 нет DECOMPOSE,
  нет и пустых «предложенных» claims до генерации: GraphCore.add_claim_
  version() играет роль слияния PROPOSED+GENERATED. Precondition GENERATE
  поэтому не «state == PROPOSED», а «claim с этим id ещё не существует».
- VERIFY впервые в проекте связывает SEAM ENGINE -> EVIDENCE ENGINE ->
  STATE ENGINE (SeamResult -> Evidence -> apply_evidence). Недоверенный
  шов не может дать HARD-evidence и здесь — тот же гейт, что в
  RepairPlanner.narrow() (этап 4): strength=NONE, если not result.trusted.

Гарантии:
- сырьё (Run) сохраняется до разбора и при любом исходе;
- actual_cost фиксируется всегда;
- новая версия claim'а создаётся, старая не удаляется (GraphCore
  версионирует сам).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

from .context_builder import Context
from .graph_core import GraphCore
from .model_adapter import ModelAdapter
from .seam_engine import SeamEngine
from .state_engine import (
    CONFLICTED,
    INCORRECT,
    INCONCLUSIVE,
    REFUTES,
    REJECTED,
    REPAIRING,
    STALE,
    STRENGTH_HARD,
    STRENGTH_NONE,
    SUPPORTS,
    Evidence,
    StateEngine,
)
from .storage import Cost, Provenance, Sampling, Storage, new_id, now_iso

GENERATE = "GENERATE"
REGENERATE = "REGENERATE"
VERIFY = "VERIFY"
STOP = "STOP"

PLANNED = "PLANNED"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
ABORTED = "ABORTED"

# Частичный набор режимов отказа (ACTION_MODEL.md §2) -- полный список
# определён в FAILURE_MODES.md, которая не входит в чтение этого этапа
# по WORK_PLAN.md; расширяется по мере необходимости.
PRECONDITION_FAILED = "PRECONDITION_FAILED"
GENERATION_FAILED = "GENERATION_FAILED"
EMPTY_OUTPUT = "EMPTY_OUTPUT"
NO_APPLICABLE_SEAM = "NO_APPLICABLE_SEAM"
EVIDENCE_REJECTED = "EVIDENCE_REJECTED"

_DEFAULT_SAMPLING = Sampling(temperature=0.0, top_p=1.0, top_k=40, seed=1, max_tokens=700)

_PRECONDITION_NAMES = {
    GENERATE: ("CLAIM_NOT_EXISTS",),
    REGENERATE: ("STATE_IN_REPAIRABLE_SET", "NOT_REPAIRING"),
    VERIFY: ("SEAM_APPLICABLE_AND_TRUSTED",),
    STOP: (),
}


@dataclass(frozen=True)
class Run:
    """RUN (DATA_MODEL.md §12) — один физический вызов оператора."""

    run_id: str
    action_id: str
    model_id: str
    prompt_profile_id: Optional[str]
    sampling_config: Sampling
    rendered_prompt: Optional[str]
    raw_output: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    wall_time_sec: float
    failed: bool
    error: Optional[str]
    timestamp: str = field(default_factory=now_iso)


@dataclass(frozen=True)
class Action:
    """ACTION (DATA_MODEL.md §11)."""

    action_id: str
    action_type: str
    target: str
    preconditions_checked: tuple = ()
    input_context_ref: Optional[str] = None
    model_id: Optional[str] = None
    prompt_profile_id: Optional[str] = None
    sampling_config: Optional[Sampling] = None
    estimated_cost: Cost = field(default_factory=Cost)
    actual_cost: Optional[Cost] = None
    result_refs: tuple = ()
    status: str = PLANNED
    failure_mode: Optional[str] = None
    provenance: Provenance = field(default_factory=lambda: Provenance(created_by="SYSTEM"))


class ActionExecutor:
    """ACTION EXECUTOR (API_BOUNDARIES.md §9)."""

    def __init__(
        self,
        graph: GraphCore,
        state_engine: StateEngine,
        seam_engine: SeamEngine,
        evidence_engine: Any,
        storage: Storage,
        model_adapter: Optional[ModelAdapter] = None,
    ) -> None:
        self._graph = graph
        self._state = state_engine
        self._seams = seam_engine
        self._evidence = evidence_engine
        self._storage = storage
        self._model = model_adapter

    def build_action(
        self,
        action_type: str,
        target_claim_id: str,
        sampling: Optional[Sampling] = None,
        prompt_profile_id: Optional[str] = None,
    ) -> Action:
        return Action(
            action_id=new_id("act"),
            action_type=action_type,
            target=target_claim_id,
            model_id=self._model.capabilities().model_id
            if self._model and action_type in (GENERATE, REGENERATE)
            else None,
            prompt_profile_id=prompt_profile_id,
            sampling_config=sampling,
            estimated_cost=self._estimate_cost(action_type),
        )

    # -- предусловия (API_BOUNDARIES.md §9) --------------------------------

    def check_preconditions(
        self, action: Action, graph_view: Optional[GraphCore] = None
    ) -> tuple[bool, list[str]]:
        """Возвращает (ok, violations) -- явный кортеж вместо буквального
        объединения типов из спеки, ok=True <=> violations пуст."""
        graph_view = graph_view if graph_view is not None else self._graph
        violations: list[str] = []

        if action.action_type == GENERATE:
            if self._claim_exists(graph_view, action.target):
                violations.append("GENERATE: claim already exists (use REGENERATE)")

        elif action.action_type == REGENERATE:
            state = self._state.current_state(action.target)
            if state == REPAIRING:
                violations.append("REGENERATE: claim is already REPAIRING")
            elif state not in (INCORRECT, STALE, CONFLICTED):
                violations.append(
                    f"REGENERATE: claim state must be INCORRECT/STALE/CONFLICTED, got {state!r}"
                )
            # "все upstream в актуальной версии" -- structurally true:
            # GraphCore.get_claim() без явной версии всегда отдаёт latest,
            # отдельного трекинга "устаревших ссылок" в этой архитектуре нет.

        elif action.action_type == VERIFY:
            seams = self._seams.applicable_seams(action.target, graph_view)
            trusted = [s for s in seams if self._seams.is_trusted(s.seam_id)]
            if not trusted:
                violations.append("VERIFY: no applicable trusted seam for claim")

        elif action.action_type == STOP:
            pass  # STOP выполнимо всегда (ACTION_MODEL §2.8)

        else:
            violations.append(f"unknown action_type: {action.action_type!r}")

        return (not violations, violations)

    @staticmethod
    def _claim_exists(graph_view: GraphCore, claim_id: str) -> bool:
        try:
            graph_view.get_claim(claim_id)
        except KeyError:
            return False
        return True

    # -- исполнение ------------------------------------------------------

    def execute(
        self,
        action: Action,
        context: Optional[Context] = None,
        *,
        artifact_id: Optional[str] = None,
        claim_type: str = "OTHER",
        seam_id: Optional[str] = None,
        seam_inputs: Optional[dict] = None,
    ) -> Action:
        """ACTION_MODEL.md §3."""
        checked = _PRECONDITION_NAMES.get(action.action_type, ())
        ok, _violations = self.check_preconditions(action)
        if not ok:
            return self._persist_final(action, checked, FAILED, PRECONDITION_FAILED, Cost(), ())

        if action.action_type == STOP:
            return self._persist_final(action, checked, COMPLETED, None, Cost(), ())

        if action.action_type == VERIFY:
            return self._execute_verify(action, checked, seam_id, seam_inputs)

        return self._execute_generation(action, context, checked, artifact_id, claim_type)

    def _execute_generation(
        self,
        action: Action,
        context: Optional[Context],
        checked: tuple,
        artifact_id: Optional[str],
        claim_type: str,
    ) -> Action:
        if self._model is None:
            raise ValueError("execute: GENERATE/REGENERATE requires a model_adapter")
        if context is None:
            raise ValueError("execute: GENERATE/REGENERATE requires context")

        sampling = action.sampling_config or _DEFAULT_SAMPLING
        raw = self._model.generate(context.render(), sampling)

        run = Run(
            run_id=new_id("run"),
            action_id=action.action_id,
            model_id=self._model.capabilities().model_id,
            prompt_profile_id=action.prompt_profile_id,
            sampling_config=sampling,
            rendered_prompt=raw.rendered_prompt,
            raw_output=raw.raw_output,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            wall_time_sec=raw.wall_time_sec,
            failed=raw.failed,
            error=raw.error,
        )
        # Сырьё сохраняется БЕЗУСЛОВНО, до любой попытки разбора
        # (ACTION_MODEL §3 шаг 4 -- "run сохраняется ВСЕГДА, даже при
        # провале парсинга").
        self._storage.append(run.run_id, "run", run)

        actual_cost = Cost(
            tokens=(raw.input_tokens or 0) + (raw.output_tokens or 0),
            time_sec=raw.wall_time_sec,
            calls=1,
            model_id=self._model.capabilities().model_id,
        )

        if raw.failed:
            return self._persist_final(
                action, checked, FAILED, GENERATION_FAILED, actual_cost, (run.run_id,)
            )

        content = raw.raw_output.strip()
        if not content:
            return self._persist_final(
                action, checked, FAILED, EMPTY_OUTPUT, actual_cost, (run.run_id,)
            )

        if action.action_type == REGENERATE:
            existing = self._graph.get_claim(action.target)
            resolved_artifact_id = artifact_id or existing.artifact_id
            resolved_claim_type = existing.claim_type
        else:
            resolved_artifact_id = artifact_id
            resolved_claim_type = claim_type

        if resolved_artifact_id is None:
            raise ValueError("execute: GENERATE requires artifact_id")

        new_version_id = self._graph.add_claim_version(
            artifact_id=resolved_artifact_id,
            content=content,
            claim_type=resolved_claim_type,
            claim_id=action.target,
            provenance=Provenance(
                created_by="MODEL",
                action_id=action.action_id,
                run_id=run.run_id,
                model_id=self._model.capabilities().model_id,
                sampling_config=sampling,
            ),
        )

        if action.action_type == REGENERATE:
            # Новая версия -- старое evidence (о прежнем, возможно
            # неверном содержимом) не должно вечно конфликтовать с новым
            # (см. StateEngine.reset_evidence_window).
            self._state.reset_evidence_window(action.target)

        return self._persist_final(
            action, checked, COMPLETED, None, actual_cost, (new_version_id, run.run_id)
        )

    def _execute_verify(
        self,
        action: Action,
        checked: tuple,
        seam_id: Optional[str],
        seam_inputs: Optional[dict],
    ) -> Action:
        claim_id = action.target

        if seam_id is None:
            applicable = self._seams.applicable_seams(claim_id, self._graph)
            seam_id = applicable[0].seam_id if applicable else None

        if seam_id is None or seam_inputs is None:
            return self._persist_final(action, checked, FAILED, NO_APPLICABLE_SEAM, Cost(), ())

        result = self._seams.evaluate(seam_id, seam_inputs)

        if result.status == "PASS":
            assertion = SUPPORTS
        elif result.status == "FAIL":
            assertion = REFUTES
        else:
            assertion = INCONCLUSIVE

        # Недоверенный шов не может дать HARD-evidence -- тот же гейт,
        # что в RepairPlanner.narrow() (этап 4).
        strength = (
            STRENGTH_HARD if (result.trusted and result.status in ("PASS", "FAIL")) else STRENGTH_NONE
        )

        evidence = Evidence(
            evidence_id=new_id("ev"),
            subject=claim_id,
            assertion=assertion,
            evidence_type="MECHANICAL",
            strength=strength,
            source={"kind": "seam", "seam_id": seam_id},
            method=f"seam:{seam_id}",
            reproducibility="DETERMINISTIC",
            provenance=Provenance(created_by="SYSTEM", action_id=action.action_id),
        )
        self._evidence.record(evidence)
        transition = self._state.apply_evidence(claim_id, evidence)

        if transition == REJECTED:
            status, failure_mode = FAILED, EVIDENCE_REJECTED
        else:
            status, failure_mode = COMPLETED, None

        return self._persist_final(
            action, checked, status, failure_mode, Cost(calls=1), (evidence.evidence_id,)
        )

    # -- внутреннее ----------------------------------------------------------

    def _estimate_cost(self, action_type: str) -> Cost:
        if action_type in (GENERATE, REGENERATE):
            return Cost(tokens=900, time_sec=5.0, calls=1)
        if action_type == VERIFY:
            return Cost(tokens=0, time_sec=0.01, calls=1)
        return Cost()

    def _persist_final(
        self,
        action: Action,
        checked: tuple,
        status: str,
        failure_mode: Optional[str],
        actual_cost: Cost,
        result_refs: tuple,
    ) -> Action:
        finalized = replace(
            action,
            status=status,
            failure_mode=failure_mode,
            actual_cost=actual_cost,
            result_refs=result_refs,
            preconditions_checked=checked,
        )
        self._storage.append(finalized.action_id, "action", finalized)
        return finalized
