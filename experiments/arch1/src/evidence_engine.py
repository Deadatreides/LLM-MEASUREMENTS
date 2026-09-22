"""EVIDENCE ENGINE — уровень 3 (API_BOUNDARIES.md §4).

Знает: типы evidence, иерархию силы, оси независимости
(EVIDENCE_MODEL.md §2-4).
Не знает: структуру графа (кроме идентификаторов subject).

Зависит только от storage (API_BOUNDARIES.md §12: «EVIDENCE ENG →
STORAGE»). Тип Evidence и связанные с ним константы (assertion,
strength, evidence_type для LLM-суждений) переиспользуются из
state_engine.py — это только импорт готового типа данных (уровень ≤2
разрешён), а не функциональная зависимость от StateEngine: сам
класс StateEngine здесь не используется и не импортируется.

strength_of — чистая, детерминированная функция (EVIDENCE_MODEL.md
§2, §4.2). Она же гарантирует то, что STATE ENGINE защищает повторно
(defense in depth): MODEL_JUDGEMENT/SELF_JUDGEMENT никогда не
получают HARD, независимо от заявленной силы источника (F9-F14).
"""

from __future__ import annotations

from typing import Optional

from .state_engine import (
    CONFLICTED,
    CORRECT,
    EVIDENCE_TYPE_MODEL_JUDGEMENT,
    EVIDENCE_TYPE_SELF_JUDGEMENT,
    INCORRECT,
    REFUTES,
    STRENGTH_HARD,
    STRENGTH_MODERATE,
    STRENGTH_NONE,
    STRENGTH_WEAK,
    SUPPORTS,
    UNKNOWN,
    Evidence,
)
from .storage import Storage

# -- базовая сила по типу evidence (EVIDENCE_MODEL.md §2) ------------------

_BASE_STRENGTH = {
    "MECHANICAL": STRENGTH_HARD,
    "EXECUTION": STRENGTH_HARD,
    "EXACT_MATCH": STRENGTH_HARD,
    "STATIC_ANALYSIS": STRENGTH_HARD,
    "EXTERNAL_DATA": STRENGTH_HARD,
    "HUMAN": STRENGTH_HARD,
    EVIDENCE_TYPE_MODEL_JUDGEMENT: STRENGTH_WEAK,
    EVIDENCE_TYPE_SELF_JUDGEMENT: STRENGTH_WEAK,
    "UNKNOWN": STRENGTH_NONE,
}

# -- сила по оси независимости для MULTI_SOURCE_AGREEMENT (§4.2) -----------
# EXTERNAL официально «переходит в EXTERNAL_DATA» (HARD) -- сохранено здесь
# как максимум оси, но вызывающий код обязан фактически завести evidence
# типом EXTERNAL_DATA, а не MULTI_SOURCE_AGREEMENT, если действительно
# использует независимый неязыковой источник.

_AXIS_STRENGTH = {
    "NONE": STRENGTH_NONE,
    "SEED": STRENGTH_WEAK,
    "TEMPERATURE": STRENGTH_WEAK,
    "PROMPT": STRENGTH_WEAK,
    "MODEL": STRENGTH_MODERATE,
    "FAMILY": STRENGTH_MODERATE,
    "EXTERNAL": STRENGTH_HARD,
}


class EvidenceEngine:
    """EVIDENCE ENGINE (API_BOUNDARIES.md §4)."""

    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._by_subject: dict[str, list[Evidence]] = {}

    def strength_of(
        self,
        evidence_type: str,
        source: Optional[dict] = None,
        independence_axis: Optional[str] = None,
        reproducibility: Optional[str] = None,
    ) -> str:
        """EVIDENCE_MODEL.md §2-4. Детерминирована, без побочных эффектов."""
        if evidence_type in (EVIDENCE_TYPE_MODEL_JUDGEMENT, EVIDENCE_TYPE_SELF_JUDGEMENT):
            # F9-F14: никогда не HARD, вне зависимости от чего бы то ни было ещё.
            return STRENGTH_WEAK
        if evidence_type == "MULTI_SOURCE_AGREEMENT":
            if independence_axis is None:
                raise ValueError("MULTI_SOURCE_AGREEMENT requires independence_axis")
            return _AXIS_STRENGTH[independence_axis]
        return _BASE_STRENGTH.get(evidence_type, STRENGTH_NONE)

    def record(self, evidence: Evidence) -> str:
        """Evidence никогда не удаляется и не изменяется (DATA_MODEL.md §6)."""
        self._storage.append(evidence.evidence_id, "evidence", evidence)
        self._by_subject.setdefault(evidence.subject, []).append(evidence)
        return evidence.evidence_id

    def evidence_for(self, subject_id: str) -> list[Evidence]:
        return list(self._by_subject.get(subject_id, []))

    def resolve(self, subject_id: str) -> Optional[str]:
        """EVIDENCE_MODEL.md §3.2. Слабое evidence не накапливается до HARD."""
        evidence_list = self._by_subject.get(subject_id, [])
        hard_support = [e for e in evidence_list if e.strength == STRENGTH_HARD and e.assertion == SUPPORTS]
        hard_refute = [e for e in evidence_list if e.strength == STRENGTH_HARD and e.assertion == REFUTES]

        if hard_refute and hard_support:
            return CONFLICTED
        if hard_refute:
            return INCORRECT
        if hard_support:
            return CORRECT
        if not evidence_list:
            return None  # GENERATED/UNVERIFIED -- решает STATE ENGINE по применимости seam
        return UNKNOWN
