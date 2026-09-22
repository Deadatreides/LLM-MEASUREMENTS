"""Контрольные случаи src/seams/exec_seam.py через SeamEngine (не напрямую
check()) -- проверяем и саму процедуру, и self_test/evaluate вместе."""

import unittest

from src.seam_engine import FAIL, HARD, INAPPLICABLE, PASS, SeamEngine
from src.seams import exec_seam
from src.storage import Storage


class ExecSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SeamEngine(Storage())
        self.engine.register(exec_seam.build_definition())

    def test_self_test_passes_all_control_cases(self) -> None:
        result = self.engine.self_test(exec_seam.SEAM_ID)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_cases"], [])
        self.assertTrue(self.engine.is_trusted(exec_seam.SEAM_ID))

    def test_passing_implementation(self) -> None:
        outcome = self.engine.evaluate(
            exec_seam.SEAM_ID,
            {
                "impl_code": "def mul(a, b):\n    return a * b\n",
                "test_code": "def test_mul():\n    assert mul(3, 4) == 12\n",
            },
        )
        self.assertEqual(outcome.status, PASS)

    def test_failing_implementation(self) -> None:
        outcome = self.engine.evaluate(
            exec_seam.SEAM_ID,
            {
                "impl_code": "def mul(a, b):\n    return a + b\n",
                "test_code": "def test_mul():\n    assert mul(3, 4) == 12\n",
            },
        )
        self.assertEqual(outcome.status, FAIL)

    def test_inapplicable_when_no_tests_found(self) -> None:
        outcome = self.engine.evaluate(
            exec_seam.SEAM_ID,
            {"impl_code": "def mul(a, b):\n    return a * b\n", "test_code": "y = 2\n"},
        )
        self.assertEqual(outcome.status, INAPPLICABLE)

    def test_seam_class_is_hard(self) -> None:
        self.assertEqual(self.engine.seam_class(exec_seam.SEAM_ID), HARD)


if __name__ == "__main__":
    unittest.main()
