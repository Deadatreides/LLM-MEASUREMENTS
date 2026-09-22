"""COLLISION — часть уровня 5, используется REPAIR PLANNER (COLLISION_MODEL.md).

Детекция коллизий и локализация origin через graph_core.root_origins.

Главное разграничение документа (COLLISION_MODEL.md §1):

    COLLISION != ERROR

Коллизия — факт наблюдения несовместимости, вход в процесс локализации,
а НЕ вердикт «этот элемент неверен». Поэтому здесь нет ни одной функции,
которая по факту коллизии помечала бы участника ошибочным: origin
вычисляется отдельным шагом и только по HARD-evidence(REFUTES).

Не импортирует ничего про LLM (весь этап 4 — без единого вызова модели).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .graph_core import GraphCore
from .storage import new_id, now_iso

# -- типы коллизий (COLLISION_MODEL.md §2) ---------------------------------

MECHANICAL_FAILURE = "MECHANICAL_FAILURE"
CONTRACT_VIOLATION = "CONTRACT_VIOLATION"
INTERNAL_INCONSISTENCY = "INTERNAL_INCONSISTENCY"
CROSS_ARTIFACT_INCONSISTENCY = "CROSS_ARTIFACT_INCONSISTENCY"
SOURCE_DISAGREEMENT = "SOURCE_DISAGREEMENT"
DEPENDENCY_VIOLATION = "DEPENDENCY_VIOLATION"
EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
FORMAT_VIOLATION = "FORMAT_VIOLATION"
DRIFT = "DRIFT"
UNVERIFIABLE_CONFLICT = "UNVERIFIABLE_CONFLICT"

# -- статусы (DATA_MODEL.md §8, STATE_MACHINE.md §5) -----------------------

OPEN = "OPEN"
LOCALIZED = "LOCALIZED"
LOCALIZED_AS_UNDECIDED = "LOCALIZED_AS_UNDECIDED"
PLANNED = "PLANNED"
REPAIRING = "REPAIRING"
RESOLVED = "RESOLVED"
UNRESOLVABLE = "UNRESOLVABLE"
ACCEPTED = "ACCEPTED"

# -- severity (COLLISION_MODEL.md §4) --------------------------------------

BLOCKING = "BLOCKING"
MAJOR = "MAJOR"
MINOR = "MINOR"
INFORMATIONAL = "INFORMATIONAL"

# -- уверенность в локализации ---------------------------------------------

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_NONE = "NONE"

# -- класс ошибки (DEPENDENCY_MODEL.md §6.3) --------------------------------

ERROR_LOCAL = "ERROR_LOCAL"
MULTIPLE_LOCAL = "MULTIPLE_LOCAL"
ERROR_GLOBAL = "ERROR_GLOBAL"

_SEAM_TYPE_TO_COLLISION = {
    "SEAM_EXECUTION": MECHANICAL_FAILURE,
    "SEAM_CONTRACT": CONTRACT_VIOLATION,
    "SEAM_FORMAT": FORMAT_VIOLATION,
    "SEAM_CONSISTENCY": INTERNAL_INCONSISTENCY,
    "SEAM_DEPENDENCY": DEPENDENCY_VIOLATION,
    "SEAM_CONFLICT": SOURCE_DISAGREEMENT,
    "SEAM_EVIDENCE": EVIDENCE_CONFLICT,
    "SEAM_DRIFT": DRIFT,
}


@dataclass
class Collision:
    """COLLISION (DATA_MODEL.md §8).

    Mutable по полям origin/status/severity/iteration_count — это
    единственная сущность проекта, где спека прямо помечает поля как M
    и где нет версионирования. Сама коллизия при этом фиксируется в
    Storage при создании (append-only журнал не теряет факт наблюдения).
    """

    collision_id: str
    collision_type: str
    participants: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    origin: Optional[tuple[str, ...]] = None
    origin_confidence: str = CONFIDENCE_NONE
    error_class: Optional[str] = None
    severity: str = INFORMATIONAL
    status: str = OPEN
    iteration_count: int = 0
    timestamp: str = field(default_factory=now_iso)


def detect_from_seam_result(seam_result, seam_type: str, participants: Iterable[str]) -> Optional[Collision]:
    """SeamResult(FAIL) -> коллизия соответствующего типа.

    PASS/INAPPLICABLE/ERROR коллизии НЕ порождают: INAPPLICABLE — не
    отказ, а ERROR — сбой самой проверки, а не свойство артефакта
    (SEAM_MODEL.md §4). Трактовать сбой проверки как дефект claim
    означало бы превращать баги инфраструктуры в ложные ошибки модели
    (F45).
    """
    if seam_result.status != "FAIL":
        return None
    return Collision(
        collision_id=new_id("col"),
        collision_type=_SEAM_TYPE_TO_COLLISION.get(seam_type, MECHANICAL_FAILURE),
        participants=tuple(participants),
    )


def localize(collision: Collision, graph: GraphCore, defective_claims: Iterable[str]) -> Collision:
    """COLLISION_MODEL.md §3.

    roots считается ЧЕРЕЗ ГРАФ ПРЕДКОВ (graph_core.root_origins), а не по
    признаку «у кого сработала проверка»: наивная версия была реализована
    в эксп. 8 и дала неверный результат — каскадный claim, согласованный с
    неверной предпосылкой, тоже не проходит проверку и выглядит как
    самостоятельный дефект (F31).
    """
    defective = {c for c in defective_claims if c in set(collision.participants)}

    if not defective:
        # никто не опровергнут жёстко -- коллизия есть, ошибки нет
        collision.origin = None
        collision.origin_confidence = CONFIDENCE_NONE
        collision.status = LOCALIZED_AS_UNDECIDED
        return collision

    roots = graph.root_origins(defective)

    if len(roots) == 1:
        collision.origin = (roots[0],)
        collision.origin_confidence = CONFIDENCE_HIGH
        collision.error_class = ERROR_LOCAL
        collision.status = LOCALIZED
    elif len(roots) > 1:
        # F39: MULTIPLE_LOCAL -- основной случай на реальных данных,
        # обрабатывается как норма, а не как исключение
        collision.origin = tuple(roots)
        collision.origin_confidence = CONFIDENCE_HIGH
        collision.error_class = MULTIPLE_LOCAL
        collision.status = LOCALIZED
    else:
        # ERROR_GLOBAL. При гарантиях уровня 1 (add_dependency не создаёт
        # циклов) непустое defective всегда даёт непустые roots, поэтому
        # ветка структурно недостижима. Сохранена для полноты класса
        # (DEPENDENCY_MODEL.md §6.3; F39: ни разу не наблюдался).
        collision.origin = None
        collision.origin_confidence = CONFIDENCE_NONE
        collision.error_class = ERROR_GLOBAL
        collision.status = LOCALIZED_AS_UNDECIDED

    return collision


def severity_of(
    collision: Collision, graph: GraphCore, final_claim_id: str, has_hard_refutes: bool
) -> str:
    """COLLISION_MODEL.md §4 — структурно, без эвристики «важности».

    Считается достижимость финального claim из участников коллизии плюс
    сила доступного evidence.
    """
    reaches_final = any(
        participant == final_claim_id or final_claim_id in graph.downstream_closure(participant)
        for participant in collision.participants
    )

    if not reaches_final:
        return MINOR if has_hard_refutes else INFORMATIONAL
    return BLOCKING if has_hard_refutes else MAJOR
