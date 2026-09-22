"""Межмодульные гарантии SeamEngine (не привязаны к одному конкретному
шву): доверие по умолчанию, ERROR != FAIL, applicable_seams, реестр
встроенных швов default_seam_engine()."""

import unittest

from src.graph_core import GraphCore
from src.seam_engine import (
    ERROR,
    FAIL,
    HARD,
    PASS,
    ControlCase,
    SeamDefinition,
    SeamEngine,
    default_seam_engine,
)
from src.seams import exec_seam, format_seam, numeric_seam
from src.storage import Storage


def _broken_check(inputs: dict) -> dict:
    # всегда заявляет PASS -- контрольный случай ожидает FAIL, self_test должен это поймать
    return {"status": PASS, "details": {}}


def _crashing_check(inputs: dict) -> dict:
    raise RuntimeError("boom")


class UntrustedSeamGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SeamEngine(Storage())
        self.definition = SeamDefinition(
            seam_id="seam:broken",
            seam_type="SEAM_EXECUTION",
            check=_broken_check,
            control_cases=(ControlCase(name="c1", inputs={}, expected_status=FAIL),),
        )
        self.engine.register(self.definition)

    def test_freshly_registered_seam_is_untrusted_by_default(self) -> None:
        self.assertFalse(self.engine.is_trusted("seam:broken"))

    def test_failing_self_test_stays_untrusted(self) -> None:
        result = self.engine.self_test("seam:broken")
        self.assertFalse(result["passed"])
        self.assertIn("c1", result["failed_cases"])
        self.assertFalse(self.engine.is_trusted("seam:broken"))

    def test_evaluate_reflects_distrust_even_when_status_is_pass(self) -> None:
        self.engine.self_test("seam:broken")
        outcome = self.engine.evaluate("seam:broken", {})
        self.assertEqual(outcome.status, PASS)  # шов реально вернул PASS...
        self.assertFalse(outcome.trusted)  # ...но недоверенный шов не может дать HARD-evidence

    def test_seam_passing_self_test_becomes_trusted(self) -> None:
        engine = SeamEngine(Storage())
        engine.register(numeric_seam.build_definition())
        engine.self_test(numeric_seam.SEAM_ID)
        outcome = engine.evaluate(numeric_seam.SEAM_ID, {"actual": 1, "expected": 1})
        self.assertTrue(outcome.trusted)


class ErrorNeverBecomesFailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SeamEngine(Storage())
        self.engine.register(
            SeamDefinition(
                seam_id="seam:crash",
                seam_type="SEAM_EXECUTION",
                check=_crashing_check,
                control_cases=(ControlCase(name="c1", inputs={}, expected_status=PASS),),
            )
        )

    def test_crashing_check_returns_error_not_fail(self) -> None:
        outcome = self.engine.evaluate("seam:crash", {})
        self.assertEqual(outcome.status, ERROR)
        self.assertNotEqual(outcome.status, FAIL)

    def test_self_test_treats_crash_as_mismatch_not_silent_pass(self) -> None:
        result = self.engine.self_test("seam:crash")
        self.assertFalse(result["passed"])
        self.assertIn("c1", result["failed_cases"])


class DefaultSeamEngineTests(unittest.TestCase):
    def test_registers_three_builtin_hard_seams_all_self_test_passing(self) -> None:
        engine = default_seam_engine(Storage())
        for seam_id in (exec_seam.SEAM_ID, format_seam.SEAM_ID, numeric_seam.SEAM_ID):
            self.assertEqual(engine.seam_class(seam_id), HARD)
            result = engine.self_test(seam_id)
            self.assertTrue(result["passed"], f"{seam_id}: {result['failed_cases']}")
            self.assertTrue(engine.is_trusted(seam_id))


class ApplicableSeamsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = Storage()
        self.graph = GraphCore(self.storage)
        artifact_version_id = self.graph.add_artifact_version(
            artifact_type="DERIVATION", task_id="task:t", content="x"
        )
        self.artifact_id = self.storage.get(artifact_version_id).payload.artifact_id
        self.graph.add_claim_version(
            artifact_id=self.artifact_id, content="42", claim_type="VALUE", claim_id="claim:V"
        )
        self.engine = default_seam_engine(self.storage)

    def test_matches_by_claim_type(self) -> None:
        seam_ids = {s.seam_id for s in self.engine.applicable_seams("claim:V", self.graph)}
        self.assertIn(numeric_seam.SEAM_ID, seam_ids)
        self.assertIn(exec_seam.SEAM_ID, seam_ids)
        self.assertNotIn(format_seam.SEAM_ID, seam_ids)

    def test_unknown_subject_returns_empty(self) -> None:
        self.assertEqual(self.engine.applicable_seams("claim:missing", self.graph), [])

    def test_indivisible_block_has_no_applicable_seams(self) -> None:
        self.graph.add_claim_version(
            artifact_id=self.artifact_id, content="other", claim_type="VALUE", claim_id="claim:W"
        )
        composite_id = self.graph.collapse_scc({"claim:V", "claim:W"})
        self.assertEqual(self.engine.applicable_seams(composite_id, self.graph), [])


if __name__ == "__main__":
    unittest.main()
