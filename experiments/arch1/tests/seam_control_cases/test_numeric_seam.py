"""Контрольные случаи src/seams/numeric_seam.py через SeamEngine."""

import unittest

from src.seam_engine import FAIL, HARD, INAPPLICABLE, PASS, SeamEngine
from src.seams import numeric_seam
from src.storage import Storage


class NumericSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SeamEngine(Storage())
        self.engine.register(numeric_seam.build_definition())

    def test_self_test_passes_all_control_cases(self) -> None:
        result = self.engine.self_test(numeric_seam.SEAM_ID)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_cases"], [])
        self.assertTrue(self.engine.is_trusted(numeric_seam.SEAM_ID))

    def test_matching_values(self) -> None:
        outcome = self.engine.evaluate(numeric_seam.SEAM_ID, {"actual": 3.14, "expected": 3.14})
        self.assertEqual(outcome.status, PASS)

    def test_mismatching_values(self) -> None:
        outcome = self.engine.evaluate(numeric_seam.SEAM_ID, {"actual": 3.0, "expected": 3.14})
        self.assertEqual(outcome.status, FAIL)

    def test_within_tolerance_passes(self) -> None:
        outcome = self.engine.evaluate(
            numeric_seam.SEAM_ID, {"actual": 3.001, "expected": 3.0, "tolerance": 0.01}
        )
        self.assertEqual(outcome.status, PASS)

    def test_non_numeric_is_inapplicable(self) -> None:
        outcome = self.engine.evaluate(numeric_seam.SEAM_ID, {"actual": "42", "expected": 42})
        self.assertEqual(outcome.status, INAPPLICABLE)

    def test_seam_class_is_hard(self) -> None:
        self.assertEqual(self.engine.seam_class(numeric_seam.SEAM_ID), HARD)


if __name__ == "__main__":
    unittest.main()
