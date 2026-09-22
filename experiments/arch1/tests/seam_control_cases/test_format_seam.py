"""Контрольные случаи src/seams/format_seam.py через SeamEngine."""

import unittest

from src.seam_engine import FAIL, HARD, INAPPLICABLE, PASS, SeamEngine
from src.seams import format_seam
from src.storage import Storage


class FormatSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SeamEngine(Storage())
        self.engine.register(format_seam.build_definition())

    def test_self_test_passes_all_control_cases(self) -> None:
        result = self.engine.self_test(format_seam.SEAM_ID)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_cases"], [])
        self.assertTrue(self.engine.is_trusted(format_seam.SEAM_ID))

    def test_well_formed_matching_function(self) -> None:
        outcome = self.engine.evaluate(
            format_seam.SEAM_ID,
            {
                "text": "Here you go:\n```python\ndef square(x):\n    return x * x\n```",
                "expected_function_name": "square",
            },
        )
        self.assertEqual(outcome.status, PASS)

    def test_wrong_function_name_fails(self) -> None:
        outcome = self.engine.evaluate(
            format_seam.SEAM_ID,
            {
                "text": "```python\ndef cube(x):\n    return x ** 3\n```",
                "expected_function_name": "square",
            },
        )
        self.assertEqual(outcome.status, FAIL)

    def test_invalid_syntax_fails(self) -> None:
        outcome = self.engine.evaluate(
            format_seam.SEAM_ID,
            {"text": "```python\ndef square(x)\n    return x * x\n```", "expected_function_name": "square"},
        )
        self.assertEqual(outcome.status, FAIL)

    def test_no_code_is_inapplicable(self) -> None:
        outcome = self.engine.evaluate(
            format_seam.SEAM_ID,
            {"text": "The square of x is x times x.", "expected_function_name": "square"},
        )
        self.assertEqual(outcome.status, INAPPLICABLE)

    def test_seam_class_is_hard(self) -> None:
        self.assertEqual(self.engine.seam_class(format_seam.SEAM_ID), HARD)


if __name__ == "__main__":
    unittest.main()
