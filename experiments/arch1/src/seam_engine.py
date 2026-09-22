"""SEAM ENGINE — уровень 3 (API_BOUNDARIES.md §5).

Знает: какие швы объявлены, как их исполнять, их контрольные случаи.
Не знает: состояние графа, стоимость.

Зависит буквально только от Storage (API_BOUNDARIES.md §12: «SEAM
ENGINE → STORAGE, ModelAdapter»). `graph_view` в applicable_seams —
duck-typed параметр (нужен только `.get_claim(...)`), не импорт
graph_core — иначе появилась бы незадекларированная в §12 зависимость.

В ARCH-1 реализованы только HARD-швы: SOFT-швы (LLM в роли
проверяющего) сознательно не входят в этот этап (WORK_PLAN.md §8).
Поэтому ModelAdapter здесь не используется и не импортируется;
seam_class() для всех встроенных швов возвращает HARD.

Гарантии:
- шов, не прошедший self_test, имеет is_trusted = false и не может
  производить evidence силы HARD (SEAM_MODEL.md §5) — is_trusted по
  умолчанию false для только что зарегистрированного шва;
- ERROR (сбой самой процедуры проверки) никогда не транслируется в
  FAIL — evaluate() ловит любое исключение check() и превращает его
  в ERROR, а не пробрасывает и не путает с содержательным FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .storage import Storage, new_id

# -- статусы результата проверки (SEAM_MODEL.md §4) -------------------------

PASS = "PASS"
FAIL = "FAIL"
INAPPLICABLE = "INAPPLICABLE"
ERROR = "ERROR"

# -- классы швов (SEAM_MODEL.md §2) -----------------------------------------

HARD = "HARD"
SOFT = "SOFT"
SEAM_CLASS_UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ControlCase:
    """Контрольный случай с заранее известным ответом (SEAM_MODEL.md §5.1)."""

    name: str
    inputs: dict
    expected_status: str  # PASS | FAIL | INAPPLICABLE


@dataclass(frozen=True)
class SeamDefinition:
    """Реестровая запись: одна процедура проверки (check) + её контрольные случаи."""

    seam_id: str
    seam_type: str  # SEAM_CONTRACT | SEAM_EXECUTION | ... (DATA_MODEL.md §5)
    check: Callable[[dict], dict]  # inputs -> {"status": ..., "details": {...}}
    control_cases: tuple[ControlCase, ...]
    applicable_claim_types: Optional[frozenset[str]] = None  # None = любой claim_type


@dataclass(frozen=True)
class Seam:
    """SEAM (DATA_MODEL.md §5) — view-объект, возвращаемый applicable_seams."""

    seam_id: str
    seam_type: str
    seam_class: str
    participants: tuple[str, ...]


@dataclass(frozen=True)
class SeamResult:
    """Результат evaluate(). trusted отражает is_trusted() шва НА МОМЕНТ вызова."""

    seam_id: str
    status: str  # PASS | FAIL | INAPPLICABLE | ERROR
    trusted: bool
    details: dict


class SeamEngine:
    """SEAM ENGINE (API_BOUNDARIES.md §5)."""

    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._seams: dict[str, SeamDefinition] = {}
        self._trusted: dict[str, bool] = {}
        self._last_self_test: dict[str, dict] = {}

    def register(self, definition: SeamDefinition) -> None:
        self._seams[definition.seam_id] = definition
        self._trusted.setdefault(definition.seam_id, False)  # непроверенный != доверенный

    def seam_class(self, seam_id: str) -> str:
        self._require(seam_id)
        return HARD  # все зарегистрированные в ARCH-1 швы -- HARD (SOFT вне scope)

    def self_test(self, seam_id: str) -> dict:
        """SEAM_MODEL.md §5.1. Пересчитываемо (§5.3): можно вызывать повторно."""
        definition = self._require(seam_id)
        failed_cases: list[str] = []
        for case in definition.control_cases:
            try:
                raw = definition.check(case.inputs)
                status = raw.get("status", ERROR)
            except Exception:
                status = ERROR
            if status != case.expected_status:
                failed_cases.append(case.name)

        passed = not failed_cases
        self._trusted[seam_id] = passed
        result = {"passed": passed, "failed_cases": failed_cases}
        self._last_self_test[seam_id] = result
        return result

    def is_trusted(self, seam_id: str) -> bool:
        self._require(seam_id)
        return self._trusted.get(seam_id, False)

    def evaluate(self, seam_id: str, inputs: dict) -> SeamResult:
        definition = self._require(seam_id)
        try:
            raw = definition.check(inputs)
            status = raw.get("status", ERROR)
            details = raw.get("details", {})
        except Exception as exc:
            status = ERROR
            details = {"exception": f"{type(exc).__name__}: {exc}"}

        result = SeamResult(
            seam_id=seam_id, status=status, trusted=self.is_trusted(seam_id), details=details
        )
        self._storage.append(new_id("seamres"), "seam_result", result)
        return result

    def applicable_seams(self, subject_id: str, graph_view: Any) -> list[Seam]:
        """graph_view — любой объект с .get_claim(claim_id) (обычно GraphCore
        или его snapshot). Не импортируется как тип — см. докстринг модуля."""
        try:
            claim = graph_view.get_claim(subject_id)
        except KeyError:
            return []

        if getattr(claim, "atomicity", None) == "INDIVISIBLE_BLOCK":
            return []  # DEPENDENCY_MODEL.md §3.3: внутрь схлопнутого блока не заходим

        claim_type = getattr(claim, "claim_type", None)
        matches = []
        for definition in self._seams.values():
            if (
                definition.applicable_claim_types is None
                or claim_type in definition.applicable_claim_types
            ):
                matches.append(
                    Seam(
                        seam_id=definition.seam_id,
                        seam_type=definition.seam_type,
                        seam_class=self.seam_class(definition.seam_id),
                        participants=(subject_id,),
                    )
                )
        return matches

    # -- внутреннее ----------------------------------------------------------

    def _require(self, seam_id: str) -> SeamDefinition:
        if seam_id not in self._seams:
            raise KeyError(f"unknown seam_id: {seam_id!r}")
        return self._seams[seam_id]


def default_seam_engine(storage: Storage) -> SeamEngine:
    """Регистрирует встроенные HARD-швы ARCH-1.

    Импорт seams.* — намеренно ВНУТРИ функции (не на уровне модуля):
    seams/*.py импортируют типы ИЗ seam_engine.py (`from ..seam_engine
    import ...`), поэтому импорт seams.* на верхнем уровне здесь создал
    бы цикл на этапе загрузки модуля.
    """
    from .seams import exec_seam, format_seam, numeric_seam

    engine = SeamEngine(storage)
    engine.register(exec_seam.build_definition())
    engine.register(format_seam.build_definition())
    engine.register(numeric_seam.build_definition())
    return engine
