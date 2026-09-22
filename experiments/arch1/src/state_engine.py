"""STATE ENGINE — уровень 2 (API_BOUNDARIES.md §3).

Знает: таблицу переходов CLAIM (STATE_MACHINE.md §2), правила
распространения STALE (§4.1).
Не знает: откуда взялось evidence, сколько оно стоило — принимает
evidence с уже проставленной силой (strength) как готовый вход.

Зависит от graph_core (direct_children для propagate_stale) и storage.
Не импортирует ничего про LLM.

Здесь же — тип Evidence (DATA_MODEL.md §6) и связанные константы.
Это единственное место, где он может жить и оставаться доступным
STATE ENGINE без нарушения направления зависимостей: EVIDENCE ENGINE
(уровень 3) стоит выше STATE ENGINE (уровень 2) и поэтому не годится
как источник общего типа для нижнего слоя — импорт был бы вверх по
уровням. EVIDENCE ENGINE переиспользует Evidence и константы отсюда
(разрешено: уровень 3 может зависеть от уровня ≤2), не наоборот.

Область этого этапа (важно для API_BOUNDARIES §3):
- apply_evidence реализует только EVIDENCE_ADDED-переходы таблицы
  STATE_MACHINE §2. Переходы, управляемые ACTION_STARTED/ACTION_COMPLETED
  (PROPOSED<->GENERATED, REPAIRING) принадлежат ACTION EXECUTOR (этап 5)
  и здесь не реализованы — интерфейс API_BOUNDARIES §3 не даёт для них
  отдельного метода записи.
- Субъект без истории состояний по умолчанию считается GENERATED —
  это точка входа для evidence-driven переходов таблицы §2.
- CONFLICTED -> CORRECT/INCORRECT «по иерархии силы» (STATE_MACHINE §2)
  не реализован: спека не определяет порядок внутри HARD, необходимый
  для разрешения конфликта между двумя HARD-evidence. Текущее поведение
  консервативно: конфликт остаётся CONFLICTED, пока evidence не удаляется
  (оно и не может — append-only) — это и есть требуемая защита от
  автоматического INCORRECT (STATE_MACHINE §2.1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from .graph_core import GraphCore
from .storage import Provenance, Storage, new_id, now_iso

# -- состояния CLAIM (STATE_MACHINE.md §1) ---------------------------------

PROPOSED = "PROPOSED"
GENERATED = "GENERATED"
UNVERIFIED = "UNVERIFIED"
CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
UNKNOWN = "UNKNOWN"
STALE = "STALE"
CONFLICTED = "CONFLICTED"
REPAIRING = "REPAIRING"
SUPERSEDED = "SUPERSEDED"

# -- evidence: assertion / strength (DATA_MODEL.md §6) ---------------------

SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
INCONCLUSIVE = "INCONCLUSIVE"

STRENGTH_HARD = "HARD"
STRENGTH_MODERATE = "MODERATE"
STRENGTH_WEAK = "WEAK"
STRENGTH_NONE = "NONE"

EVIDENCE_TYPE_MODEL_JUDGEMENT = "MODEL_JUDGEMENT"
EVIDENCE_TYPE_SELF_JUDGEMENT = "SELF_JUDGEMENT"
_LLM_JUDGEMENT_TYPES = (EVIDENCE_TYPE_MODEL_JUDGEMENT, EVIDENCE_TYPE_SELF_JUDGEMENT)

# -- триггеры STATE_RECORD (DATA_MODEL.md §7) -------------------------------

TRIGGER_EVIDENCE_ADDED = "EVIDENCE_ADDED"
TRIGGER_UPSTREAM_CHANGED = "UPSTREAM_CHANGED"
TRIGGER_ACTION_COMPLETED = "ACTION_COMPLETED"
TRIGGER_COLLISION_DETECTED = "COLLISION_DETECTED"
TRIGGER_REPAIR_APPLIED = "REPAIR_APPLIED"
TRIGGER_MANUAL = "MANUAL"

REJECTED = "REJECTED"


@dataclass(frozen=True)
class Evidence:
    """EVIDENCE (DATA_MODEL.md §6). strength приходит уже вычисленной."""

    evidence_id: str
    subject: str
    assertion: str  # SUPPORTS | REFUTES | INCONCLUSIVE
    evidence_type: str
    strength: str  # HARD | MODERATE | WEAK | NONE
    source: dict
    method: str
    independence_axis: Optional[str] = None
    reproducibility: str = "ONE_SHOT"
    raw_ref: Optional[str] = None
    timestamp: str = field(default_factory=now_iso)
    provenance: Optional[Provenance] = None


@dataclass(frozen=True)
class StateRecord:
    """STATE_RECORD (DATA_MODEL.md §7) — append-only журнал переходов."""

    state_record_id: str
    subject: str
    from_state: Optional[str]
    to_state: str
    trigger: str
    trigger_ref: str
    graph_version: int
    timestamp: str = field(default_factory=now_iso)


class StateEngine:
    """STATE ENGINE (API_BOUNDARIES.md §3)."""

    def __init__(self, storage: Storage, graph: GraphCore) -> None:
        self._storage = storage
        self._graph = graph
        self._current: dict[str, str] = {}
        self._history: dict[str, list[str]] = {}
        self._evidence_by_subject: dict[str, list[Evidence]] = {}
        self._evidence_watermark: dict[str, int] = {}
        self._staleness_source: dict[str, set[str]] = {}

    # -- чтение (API_BOUNDARIES.md §3) ------------------------------------

    def current_state(self, subject_id: str) -> str:
        return self._current.get(subject_id, GENERATED)

    def state_history(self, subject_id: str) -> list[StateRecord]:
        return [
            self._storage.get(record_id).payload
            for record_id in self._history.get(subject_id, [])
        ]

    def validate_invariants(self, graph_version: Optional[int] = None) -> list[dict]:
        """INV-S1/S2/S3/S4 (STATE_MACHINE.md §6), над текущим накопленным состоянием."""
        violations: list[dict] = []
        for subject_id, state in self._current.items():
            evidence_list = self._evidence_by_subject.get(subject_id, [])
            hard_support = any(
                e.strength == STRENGTH_HARD and e.assertion == SUPPORTS for e in evidence_list
            )
            hard_refute = any(
                e.strength == STRENGTH_HARD and e.assertion == REFUTES for e in evidence_list
            )
            if state == CORRECT and not hard_support:
                violations.append({"invariant": "INV-S1", "subject": subject_id})
            if state == INCORRECT and not hard_refute:
                violations.append({"invariant": "INV-S2", "subject": subject_id})
            if state == STALE and subject_id not in self._staleness_source:
                violations.append({"invariant": "INV-S3", "subject": subject_id})
            if state == CONFLICTED and not (hard_support and hard_refute):
                violations.append({"invariant": "INV-S4", "subject": subject_id})
        return violations

    # -- запись: apply_evidence -------------------------------------------

    def apply_evidence(self, subject_id: str, evidence: Evidence) -> Any:
        """STATE_MACHINE.md §2, строки с триггером EVIDENCE_ADDED.

        Возвращает StateRecord при реальном переходе, тот же StateRecord
        без изменений при отсутствии перехода, либо константу REJECTED,
        если единственным основанием для CORRECT было бы evidence типа
        MODEL_JUDGEMENT/SELF_JUDGEMENT (INV-S5,
        LLM_JUDGEMENT_IS_NOT_HARD_ORACLE) — проверяется явно (assert),
        а не полагается только на то, что strength_of никогда не
        присвоит такому evidence HARD.

        Учитывается только evidence, накопленное с момента последнего
        входа в STALE (watermark) — «повторная проверка относительно
        нового upstream» (STATE_MACHINE.md §4.2) не может опираться на
        evidence, собранное до того, как предпосылка изменилась. Для
        остальных состояний используется вся история (append-only, ничего
        не теряется — просто не всё учитывается в КАЖДОМ конкретном
        вычислении цели).
        """
        current = self.current_state(subject_id)
        self._evidence_by_subject.setdefault(subject_id, []).append(evidence)
        full_history = self._evidence_by_subject[subject_id]
        watermark = self._evidence_watermark.get(subject_id, 0)
        window = full_history[watermark:]

        hard_refute = [e for e in window if e.strength == STRENGTH_HARD and e.assertion == REFUTES]
        hard_support_all = [
            e for e in window if e.strength == STRENGTH_HARD and e.assertion == SUPPORTS
        ]
        hard_support_eligible = [
            e for e in hard_support_all if e.evidence_type not in _LLM_JUDGEMENT_TYPES
        ]

        if hard_refute and hard_support_eligible:
            target = CONFLICTED
        elif hard_refute:
            target = INCORRECT
        elif hard_support_eligible:
            target = CORRECT
        elif current in (GENERATED, UNVERIFIED):
            # STATE_MACHINE §2: GENERATED -> UNKNOWN, если сила < HARD_ENOUGH для всех
            target = UNKNOWN
        else:
            # STALE/CONFLICTED/UNKNOWN без решающего evidence в окне -- не понижаем
            # информативность состояния произвольно (§4.1: "не меняем")
            target = current

        would_be_correct_without_filter = bool(hard_support_all) and not hard_refute
        if (
            evidence.strength == STRENGTH_HARD
            and evidence.assertion == SUPPORTS
            and evidence.evidence_type in _LLM_JUDGEMENT_TYPES
            and would_be_correct_without_filter
            and target != CORRECT
        ):
            return REJECTED

        # защитные ассерты §2.1 (не должны срабатывать при корректной
        # классификации выше — оставлены как явная защита, а не расчёт)
        assert not (current == UNKNOWN and target == INCORRECT and evidence.assertion != REFUTES)
        assert not (current == CONFLICTED and target == INCORRECT)

        if target == current:
            return StateRecord(
                state_record_id=new_id("strec"),
                subject=subject_id,
                from_state=current,
                to_state=target,
                trigger=TRIGGER_EVIDENCE_ADDED,
                trigger_ref=evidence.evidence_id,
                graph_version=self._graph.current_graph_version(),
            )

        return self._record_transition(
            subject_id, current, target, TRIGGER_EVIDENCE_ADDED, evidence.evidence_id
        )

    def reset_evidence_window(self, subject_id: str) -> None:
        """Тот же механизм watermark, что использует STALE (apply_evidence
        docstring выше), но с другим триггером: новая ВЕРСИЯ claim'а
        (REGENERATE) создана — старое evidence (о прежнем, возможно
        неверном содержимом) не должно вечно конфликтовать с новым.

        Без этого вызова claim, один раз получивший HARD REFUTES, после
        успешного REGENERATE и повторного HARD SUPPORTS навсегда
        застревал бы в CONFLICTED (append-only: старое REFUTES никуда не
        девается) — ровно тот сценарий, который выявила интеграция на
        этапе 6 (STATUS.md: «C4 после успешного REGENERATE — CONFLICTED,
        а не CORRECT»). Вызывается ACTION EXECUTOR сразу после успешного
        REGENERATE, не после GENERATE (у свежего claim'а и так нет
        предыдущего evidence).

        Старое evidence НЕ удаляется (append-only, INV-S8) — просто не
        учитывается заново в классификации, как и при STALE.
        """
        self._evidence_watermark[subject_id] = len(self._evidence_by_subject.get(subject_id, []))

    # -- запись: propagate_stale ------------------------------------------

    def propagate_stale(self, changed_claim_id: str) -> set[str]:
        """STATE_MACHINE.md §4.1.

        Возвращает множество claim'ов, которые этим вызовом ВПЕРВЫЕ
        перешли CORRECT -> STALE (не включает уже бывшие STALE — для них
        только добавляется источник, без нового STATE_RECORD/возврата).
        """
        newly_stale: set[str] = set()
        visited: set[str] = set()
        frontier = deque(self._graph.direct_children(changed_claim_id))
        visited.update(frontier)

        while frontier:
            node = frontier.popleft()
            state = self.current_state(node)

            if state == CORRECT:
                self._record_transition(
                    node, CORRECT, STALE, TRIGGER_UPSTREAM_CHANGED, changed_claim_id
                )
                self._staleness_source.setdefault(node, set()).add(changed_claim_id)
                self._evidence_watermark[node] = len(self._evidence_by_subject.get(node, []))
                newly_stale.add(node)
            elif state == STALE:
                self._staleness_source.setdefault(node, set()).add(changed_claim_id)
            # INCORRECT/CONFLICTED/UNKNOWN/UNVERIFIED/GENERATED/PROPOSED -- не меняем

            for child in self._graph.direct_children(node):
                if child not in visited:
                    visited.add(child)
                    frontier.append(child)

        return newly_stale

    # -- внутреннее ----------------------------------------------------------

    def _record_transition(
        self, subject_id: str, from_state: Optional[str], to_state: str, trigger: str, trigger_ref: str
    ) -> StateRecord:
        record = StateRecord(
            state_record_id=new_id("strec"),
            subject=subject_id,
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            trigger_ref=trigger_ref,
            graph_version=self._graph.current_graph_version(),
        )
        self._storage.append(record.state_record_id, "state_record", record)
        self._current[subject_id] = to_state
        self._history.setdefault(subject_id, []).append(record.state_record_id)
        return record
