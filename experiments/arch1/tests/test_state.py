"""Контрольные случаи для src/state_engine.py.

Минимум (WORK_PLAN.md, Этап 2):
1. UNKNOWN -> INCORRECT без evidence(REFUTES) -> отклонено
2. STALE -> INCORRECT без новой проверки -> отклонено
3. CONFLICTED -> INCORRECT автоматически -> отклонено
4. CORRECT только от HARD-evidence не-модельного типа (INV-S5, здесь --
   защита на стороне STATE ENGINE; версия для EVIDENCE ENGINE -- в
   test_evidence.py)
7. propagate_stale не затирает INCORRECT/CONFLICTED
8. propagate_stale не заходит в независимую ветвь
9. каждый переход порождает STATE_RECORD

(5, 6 -- в test_evidence.py, это про EVIDENCE ENGINE.)

Фикстура: A -> B, A -> C, B -> D (ветвление + цепочка) + независимая
ветвь E -> F.
"""

import unittest

from src.graph_core import STATUS_CONFIRMED, GraphCore
from src.state_engine import (
    CONFLICTED,
    CORRECT,
    EVIDENCE_TYPE_MODEL_JUDGEMENT,
    GENERATED,
    INCORRECT,
    REFUTES,
    REJECTED,
    STALE,
    STRENGTH_HARD,
    STRENGTH_WEAK,
    SUPPORTS,
    UNKNOWN,
    Evidence,
    StateEngine,
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


class StateEngineFixtureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = Storage()
        self.graph = GraphCore(self.storage)

        artifact_version_id = self.graph.add_artifact_version(
            artifact_type="DERIVATION", task_id="task:fixture", content="fixture artifact"
        )
        artifact_id = self.storage.get(artifact_version_id).payload.artifact_id

        for name in "ABCDEF":
            self.graph.add_claim_version(
                artifact_id=artifact_id,
                content=f"claim {name}",
                claim_type="OTHER",
                claim_id=self.c(name),
            )

        for source, target in [("A", "B"), ("A", "C"), ("B", "D"), ("E", "F")]:
            self.graph.add_dependency(
                source_claim=self.c(source),
                target_claim=self.c(target),
                dependency_type="DERIVATION",
                status=STATUS_CONFIRMED,
                created_by="HUMAN",
            )

        self.state = StateEngine(self.storage, self.graph)

    @staticmethod
    def c(name: str) -> str:
        return f"claim:{name}"


class ForbiddenTransitionTests(StateEngineFixtureTestCase):
    def test_unknown_never_becomes_incorrect_without_refutes(self) -> None:
        claim = self.c("B")
        self.state.apply_evidence(claim, make_evidence(claim, SUPPORTS, STRENGTH_WEAK))
        self.assertEqual(self.state.current_state(claim), UNKNOWN)

        # ещё поддерживающего/неинформативного evidence -- всё ещё не INCORRECT
        self.state.apply_evidence(claim, make_evidence(claim, SUPPORTS, STRENGTH_WEAK))
        self.assertEqual(self.state.current_state(claim), UNKNOWN)

    def test_stale_never_becomes_incorrect_without_fresh_hard_evidence(self) -> None:
        claim = self.c("D")  # ребёнок B, B -- ребёнок A
        self.state.apply_evidence(claim, make_evidence(claim, SUPPORTS, STRENGTH_HARD))
        self.assertEqual(self.state.current_state(claim), CORRECT)

        self.state.propagate_stale(self.c("B"))
        self.assertEqual(self.state.current_state(claim), STALE)

        # старое HARD-evidence "протухло" (watermark), а новое -- не HARD REFUTES:
        # STALE не должен автоматически стать INCORRECT
        self.state.apply_evidence(claim, make_evidence(claim, REFUTES, STRENGTH_WEAK))
        self.assertNotEqual(self.state.current_state(claim), INCORRECT)

    def test_conflicted_does_not_auto_become_incorrect(self) -> None:
        claim = self.c("C")
        self.state.apply_evidence(claim, make_evidence(claim, SUPPORTS, STRENGTH_HARD))
        self.assertEqual(self.state.current_state(claim), CORRECT)

        self.state.apply_evidence(claim, make_evidence(claim, REFUTES, STRENGTH_HARD))
        self.assertEqual(self.state.current_state(claim), CONFLICTED)

        # ещё одно жёсткое опровержение поверх уже конфликтующих -- не INCORRECT
        self.state.apply_evidence(claim, make_evidence(claim, REFUTES, STRENGTH_HARD))
        self.assertEqual(self.state.current_state(claim), CONFLICTED)

    def test_correct_rejected_when_sole_hard_support_is_model_judgement(self) -> None:
        claim = self.c("A")
        result = self.state.apply_evidence(
            claim,
            make_evidence(claim, SUPPORTS, STRENGTH_HARD, evidence_type=EVIDENCE_TYPE_MODEL_JUDGEMENT),
        )
        self.assertEqual(result, REJECTED)
        self.assertNotEqual(self.state.current_state(claim), CORRECT)


class PropagateStaleTests(StateEngineFixtureTestCase):
    def test_does_not_overwrite_incorrect_or_conflicted(self) -> None:
        self.state.apply_evidence(self.c("D"), make_evidence(self.c("D"), REFUTES, STRENGTH_HARD))
        self.assertEqual(self.state.current_state(self.c("D")), INCORRECT)

        self.state.apply_evidence(self.c("C"), make_evidence(self.c("C"), SUPPORTS, STRENGTH_HARD))
        self.state.apply_evidence(self.c("C"), make_evidence(self.c("C"), REFUTES, STRENGTH_HARD))
        self.assertEqual(self.state.current_state(self.c("C")), CONFLICTED)

        changed = self.state.propagate_stale(self.c("A"))

        self.assertEqual(self.state.current_state(self.c("D")), INCORRECT)
        self.assertEqual(self.state.current_state(self.c("C")), CONFLICTED)
        self.assertNotIn(self.c("D"), changed)
        self.assertNotIn(self.c("C"), changed)

    def test_does_not_leak_into_independent_branch(self) -> None:
        self.state.apply_evidence(self.c("F"), make_evidence(self.c("F"), SUPPORTS, STRENGTH_HARD))
        self.assertEqual(self.state.current_state(self.c("F")), CORRECT)

        changed = self.state.propagate_stale(self.c("A"))

        self.assertNotIn(self.c("F"), changed)
        self.assertEqual(self.state.current_state(self.c("F")), CORRECT)

    def test_marks_correct_descendants_stale(self) -> None:
        self.state.apply_evidence(self.c("B"), make_evidence(self.c("B"), SUPPORTS, STRENGTH_HARD))
        self.state.apply_evidence(self.c("D"), make_evidence(self.c("D"), SUPPORTS, STRENGTH_HARD))

        changed = self.state.propagate_stale(self.c("A"))

        self.assertEqual(changed, {self.c("B"), self.c("D")})
        self.assertEqual(self.state.current_state(self.c("B")), STALE)
        self.assertEqual(self.state.current_state(self.c("D")), STALE)


class StateRecordTests(StateEngineFixtureTestCase):
    def test_every_real_transition_creates_state_record(self) -> None:
        claim = self.c("A")
        self.assertEqual(self.state.state_history(claim), [])

        self.state.apply_evidence(claim, make_evidence(claim, SUPPORTS, STRENGTH_HARD))
        history = self.state.state_history(claim)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].from_state, GENERATED)
        self.assertEqual(history[0].to_state, CORRECT)
        self.assertIsNotNone(history[0].graph_version)

    def test_noop_apply_evidence_does_not_duplicate_state_record(self) -> None:
        claim = self.c("A")
        self.state.apply_evidence(claim, make_evidence(claim, SUPPORTS, STRENGTH_HARD))
        self.assertEqual(len(self.state.state_history(claim)), 1)

        # то же самое ещё раз -- состояние уже CORRECT, новой записи быть не должно
        self.state.apply_evidence(claim, make_evidence(claim, SUPPORTS, STRENGTH_HARD))
        self.assertEqual(len(self.state.state_history(claim)), 1)

    def test_propagate_stale_transition_also_recorded(self) -> None:
        self.state.apply_evidence(self.c("B"), make_evidence(self.c("B"), SUPPORTS, STRENGTH_HARD))
        self.assertEqual(len(self.state.state_history(self.c("B"))), 1)

        self.state.propagate_stale(self.c("A"))
        history = self.state.state_history(self.c("B"))
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1].to_state, STALE)
        self.assertEqual(history[-1].trigger, "UPSTREAM_CHANGED")


if __name__ == "__main__":
    unittest.main()
