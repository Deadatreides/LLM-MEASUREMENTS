"""Контрольные случаи для src/evidence_engine.py.

Минимум (WORK_PLAN.md, Этап 2):
5. накопление 10 WEAK evidence НЕ даёт CORRECT
6. MULTI_SOURCE_AGREEMENT по оси SEED -> strength <= WEAK

Плюс: базовая сила по типу (§2), MODEL/SELF_JUDGEMENT никогда не HARD
(F9-F14, часть основания для случая 4 -- вторая половина в
test_state.py), resolve() по всей таблице §3.2, вся таблица осей §4.2.
"""

import unittest

from src.evidence_engine import EvidenceEngine
from src.state_engine import (
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
from src.storage import Storage, new_id


def make_evidence(subject: str, assertion: str, strength: str, evidence_type: str = "MECHANICAL") -> Evidence:
    return Evidence(
        evidence_id=new_id("ev"),
        subject=subject,
        assertion=assertion,
        evidence_type=evidence_type,
        strength=strength,
        source={"kind": "test"},
        method="test-fixture",
    )


class StrengthOfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = EvidenceEngine(Storage())

    def test_base_strength_by_type(self) -> None:
        hard_types = [
            "MECHANICAL", "EXECUTION", "EXACT_MATCH", "STATIC_ANALYSIS", "EXTERNAL_DATA", "HUMAN",
        ]
        for evidence_type in hard_types:
            self.assertEqual(self.engine.strength_of(evidence_type), STRENGTH_HARD, evidence_type)

    def test_model_and_self_judgement_never_hard(self) -> None:
        # F9-F14: базовая сила WEAK, и остаётся WEAK вне зависимости от того,
        # что бы источник ни заявлял (strength_of не принимает "желаемую" силу).
        self.assertEqual(self.engine.strength_of(EVIDENCE_TYPE_MODEL_JUDGEMENT), STRENGTH_WEAK)
        self.assertEqual(self.engine.strength_of(EVIDENCE_TYPE_SELF_JUDGEMENT), STRENGTH_WEAK)

    def test_multi_source_agreement_axis_table(self) -> None:
        expected = {
            "NONE": STRENGTH_NONE,
            "SEED": STRENGTH_WEAK,
            "TEMPERATURE": STRENGTH_WEAK,
            "PROMPT": STRENGTH_WEAK,
            "MODEL": STRENGTH_MODERATE,
            "FAMILY": STRENGTH_MODERATE,
            "EXTERNAL": STRENGTH_HARD,
        }
        for axis, strength in expected.items():
            self.assertEqual(
                self.engine.strength_of("MULTI_SOURCE_AGREEMENT", independence_axis=axis),
                strength,
                axis,
            )

    def test_multi_source_agreement_seed_capped_weak(self) -> None:
        # случай 6 из WORK_PLAN.md дословно
        strength = self.engine.strength_of("MULTI_SOURCE_AGREEMENT", independence_axis="SEED")
        self.assertLessEqual(
            ["NONE", "WEAK", "MODERATE", "HARD"].index(strength),
            ["NONE", "WEAK", "MODERATE", "HARD"].index(STRENGTH_WEAK),
        )
        self.assertEqual(strength, STRENGTH_WEAK)

    def test_multi_source_agreement_requires_independence_axis(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.strength_of("MULTI_SOURCE_AGREEMENT")

    def test_strength_of_is_deterministic(self) -> None:
        results = {self.engine.strength_of("MECHANICAL") for _ in range(5)}
        self.assertEqual(results, {STRENGTH_HARD})


class WeakAccumulationTests(unittest.TestCase):
    def test_ten_weak_evidence_never_reaches_correct(self) -> None:
        engine = EvidenceEngine(Storage())
        subject = "claim:X"
        for _ in range(10):
            strength = engine.strength_of(EVIDENCE_TYPE_MODEL_JUDGEMENT)
            engine.record(make_evidence(subject, SUPPORTS, strength, EVIDENCE_TYPE_MODEL_JUDGEMENT))
        self.assertEqual(len(engine.evidence_for(subject)), 10)
        self.assertEqual(engine.resolve(subject), UNKNOWN)
        self.assertNotEqual(engine.resolve(subject), CORRECT)


class ResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = EvidenceEngine(Storage())
        self.subject = "claim:Y"

    def test_no_evidence_resolves_to_none(self) -> None:
        self.assertIsNone(self.engine.resolve(self.subject))

    def test_only_weak_resolves_unknown(self) -> None:
        self.engine.record(make_evidence(self.subject, SUPPORTS, STRENGTH_WEAK))
        self.assertEqual(self.engine.resolve(self.subject), UNKNOWN)

    def test_hard_support_resolves_correct(self) -> None:
        self.engine.record(make_evidence(self.subject, SUPPORTS, STRENGTH_HARD))
        self.assertEqual(self.engine.resolve(self.subject), CORRECT)

    def test_hard_refute_resolves_incorrect(self) -> None:
        self.engine.record(make_evidence(self.subject, REFUTES, STRENGTH_HARD))
        self.assertEqual(self.engine.resolve(self.subject), INCORRECT)

    def test_hard_support_and_refute_resolves_conflicted(self) -> None:
        self.engine.record(make_evidence(self.subject, SUPPORTS, STRENGTH_HARD))
        self.engine.record(make_evidence(self.subject, REFUTES, STRENGTH_HARD))
        self.assertEqual(self.engine.resolve(self.subject), CONFLICTED)

    def test_evidence_never_removed_conflicting_coexist(self) -> None:
        self.engine.record(make_evidence(self.subject, SUPPORTS, STRENGTH_HARD))
        self.engine.record(make_evidence(self.subject, REFUTES, STRENGTH_HARD))
        self.assertEqual(len(self.engine.evidence_for(self.subject)), 2)


if __name__ == "__main__":
    unittest.main()
