"""Контрольные случаи ХАРНЕССА эксперимента 11 (не моделей -- моделей
здесь нет, тесты не требуют GPU).

Дисциплина -- та же, что дала о себе знать на arch1 этапе 8: неоднозначное
извлечение должно давать INAPPLICABLE, не угаданный FAIL; общий стартовый
контекст для всех плеч должен быть буквально идентичен; подсчёт токенов
должен быть точным, не placeholder.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness  # noqa: E402
from harness import (  # noqa: E402
    ArmConfig,
    CollisionState,
    best_of_n,
    bootstrap_ratio_diff,
    build_arm_configs,
    classify_repeat,
    paired_sign_test,
    run_arm,
    run_seek_evidence,
    run_stop,
)
from seams import ERROR, FAIL, INAPPLICABLE, PASS, UNKNOWN, arithmetic_seam, code_seam  # noqa: E402
from configs.model_registry import MODEL_IDS  # noqa: E402


class ArithmeticSeamTests(unittest.TestCase):
    def test_single_equation_pass(self):
        r = arithmetic_seam("total = 26", 26)
        self.assertEqual(r["status"], PASS)

    def test_single_equation_fail(self):
        r = arithmetic_seam("total = 25", 26)
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(r["collision_type"], "numeric")

    def test_ambiguous_multi_equation_is_inapplicable_not_fail(self):
        text = "step1 = 10\nstep2 = 16\ntotal = 26"
        r = arithmetic_seam(text, 26)
        self.assertEqual(r["status"], INAPPLICABLE)
        self.assertNotEqual(r["status"], FAIL)

    def test_no_number_is_inapplicable(self):
        r = arithmetic_seam("I don't know the answer.", 26)
        self.assertEqual(r["status"], INAPPLICABLE)

    def test_single_bare_number_used(self):
        r = arithmetic_seam("26", 26)
        self.assertEqual(r["status"], PASS)


class CodeSeamTests(unittest.TestCase):
    def test_no_code_is_inapplicable(self):
        r = code_seam("I think the answer involves a loop.", "is_even", ["is_even(2) == True"])
        self.assertEqual(r["status"], INAPPLICABLE)
        self.assertEqual(r["collision_type"], "format")

    def test_syntax_error_is_fail_syntactic(self):
        text = "```python\ndef is_even(n)\n    return n % 2 == 0\n```"
        r = code_seam(text, "is_even", ["is_even(2) == True"])
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(r["collision_type"], "syntactic")

    def test_wrong_function_name_is_fail_structural(self):
        text = "```python\ndef check_even(n):\n    return n % 2 == 0\n```"
        r = code_seam(text, "is_even", ["is_even(2) == True"])
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(r["collision_type"], "structural")

    def test_wrong_logic_is_fail_logical(self):
        text = "```python\ndef is_even(n):\n    return True\n```"
        r = code_seam(text, "is_even", ["is_even(2) == True", "is_even(3) == False"])
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(r["collision_type"], "logical")

    def test_correct_code_is_pass(self):
        text = "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
        r = code_seam(text, "is_even", ["is_even(2) == True", "is_even(3) == False"])
        self.assertEqual(r["status"], PASS)
        self.assertEqual(r["details"]["n_tests_passed"], 2)

    def test_never_returns_unknown(self):
        # UNKNOWN зарезервирован спецификацией, но механические швы этого
        # стенда никогда его не производят -- INAPPLICABLE берёт на себя
        # эту роль. Тест фиксирует это как явный, а не случайный факт.
        cases = [
            code_seam("no code here", "f", ["f() == 1"]),
            code_seam("```python\ndef f(:\n```", "f", ["f() == 1"]),
            code_seam("```python\ndef g():\n    return 1\n```", "f", ["f() == 1"]),
        ]
        for r in cases:
            self.assertNotEqual(r["status"], UNKNOWN)


class ArmConfigTests(unittest.TestCase):
    def test_all_four_arm_signatures_distinct(self):
        import random

        rng = random.Random(1)
        configs = build_arm_configs("model-a", ["model-a", "model-b", "model-c"], rng)
        sigs = {c.signature() for c in configs.values()}
        self.assertEqual(len(sigs), 4, f"expected 4 distinct signatures, got {sigs}")

    def test_regenerate_same_does_not_reuse_initial_seed(self):
        import random

        rng = random.Random(1)
        configs = build_arm_configs("model-a", ["model-a", "model-b"], rng)
        control = configs["REGENERATE_SAME"]
        # initial generation used seeds 1..4 (harness.INITIAL_SEEDS) -- retry
        # must use a genuinely new seed, else it degenerates to a byte-identical
        # repeat (the exact defect found and fixed in arch1's Experiment 10).
        self.assertNotIn(control.seed, harness.INITIAL_SEEDS)

    def test_change_model_picks_a_different_model(self):
        import random

        rng = random.Random(1)
        configs = build_arm_configs("model-a", ["model-a", "model-b", "model-c"], rng)
        self.assertNotEqual(configs["CHANGE_MODEL"].model_id, "model-a")

    def test_change_temperature_only_differs_in_temperature(self):
        import random

        rng = random.Random(1)
        configs = build_arm_configs("model-a", ["model-a", "model-b"], rng)
        base, temp_arm = configs["REGENERATE_SAME"], configs["CHANGE_TEMPERATURE"]
        self.assertEqual(base.model_id, temp_arm.model_id)
        self.assertEqual(base.prompt_profile, temp_arm.prompt_profile)
        self.assertEqual(base.seed, temp_arm.seed)
        self.assertNotEqual(base.temperature, temp_arm.temperature)

    def test_change_prompt_only_differs_in_prompt_profile(self):
        import random

        rng = random.Random(1)
        configs = build_arm_configs("model-a", ["model-a", "model-b"], rng)
        base, prompt_arm = configs["REGENERATE_SAME"], configs["CHANGE_PROMPT"]
        self.assertEqual(base.model_id, prompt_arm.model_id)
        self.assertEqual(base.temperature, prompt_arm.temperature)
        self.assertEqual(base.seed, prompt_arm.seed)
        self.assertNotEqual(base.prompt_profile, prompt_arm.prompt_profile)


class CommonStartingPointTests(unittest.TestCase):
    """п.1 задания: ни одно плечо не должно получать более выгодный
    контекст, чем другое -- все retry-промпты строятся из ОДНОГО и того
    же state.initial_artifact / state.initial_seam_result."""

    def _state(self):
        seam = arithmetic_seam("100", 200)
        return CollisionState(
            collision_id="col:test:1", task_family="arithmetic", task_id="ARITH_01",
            model_id="model-a", seed=1, initial_artifact="100 = 100",
            initial_seam_result=seam, initial_record={},
        )

    def test_arms_receive_identical_initial_artifact_and_evidence(self):
        import random

        state = self._state()
        rng = random.Random(1)
        configs = build_arm_configs("model-a", ["model-a", "model-b"], rng)
        evidence = harness._describe_evidence(state.initial_seam_result)
        prompts = {
            arm: harness._retry_prompt(state.task_family, state.task_id, cfg.prompt_profile, state.initial_artifact, evidence)
            for arm, cfg in configs.items()
        }
        # A/B/D share the SAME prompt template (prompt_profile "v1") -> identical text
        self.assertEqual(prompts["REGENERATE_SAME"], prompts["CHANGE_MODEL"])
        self.assertEqual(prompts["REGENERATE_SAME"], prompts["CHANGE_TEMPERATURE"])
        # both v1 and v2 must embed the SAME prior artifact and evidence -- only wording differs
        self.assertIn(state.initial_artifact, prompts["REGENERATE_SAME"])
        self.assertIn(state.initial_artifact, prompts["CHANGE_PROMPT"])


class ClassifyRepeatTests(unittest.TestCase):
    def test_fixed(self):
        before = {"status": FAIL, "collision_type": "numeric", "details": {"actual": 5}}
        after = {"status": PASS, "collision_type": None, "details": {"actual": 26}}
        self.assertEqual(classify_repeat(before, after), "fixed")

    def test_same_failure_numeric(self):
        before = {"status": FAIL, "collision_type": "numeric", "details": {"actual": 5}}
        after = {"status": FAIL, "collision_type": "numeric", "details": {"actual": 5}}
        self.assertEqual(classify_repeat(before, after), "same_failure")

    def test_different_failure_value(self):
        before = {"status": FAIL, "collision_type": "numeric", "details": {"actual": 5}}
        after = {"status": FAIL, "collision_type": "numeric", "details": {"actual": 9}}
        self.assertEqual(classify_repeat(before, after), "different_failure_value")

    def test_different_failure_type(self):
        before = {"status": FAIL, "collision_type": "numeric", "details": {"actual": 5}}
        after = {"status": FAIL, "collision_type": "logical", "details": {"failing_tests": ["x"]}}
        self.assertEqual(classify_repeat(before, after), "different_failure_type")

    def test_became_inapplicable(self):
        before = {"status": FAIL, "collision_type": "numeric", "details": {"actual": 5}}
        after = {"status": INAPPLICABLE, "collision_type": "format", "details": {}}
        self.assertEqual(classify_repeat(before, after), "became_inapplicable")


class SeekEvidenceAndStopTests(unittest.TestCase):
    def test_seek_evidence_arithmetic_is_always_inapplicable(self):
        state = CollisionState(
            collision_id="c1", task_family="arithmetic", task_id="ARITH_01", model_id="m",
            seed=1, initial_artifact="1 = 1", initial_seam_result={"status": FAIL, "collision_type": "numeric", "details": {}},
            initial_record={},
        )
        r = run_seek_evidence(state)
        self.assertEqual(r["status"], INAPPLICABLE)
        self.assertEqual(r["tokens"], 0)

    def test_seek_evidence_code_runs_holdout_not_visible(self):
        # code passes visible-implied fix trivially wrong on visible, but this
        # artifact happens to satisfy CODE_01's holdout tests too -- exercises
        # that SEEK_EVIDENCE reads task holdout_tests, not visible_tests.
        state = CollisionState(
            collision_id="c2", task_family="code", task_id="CODE_01", model_id="m", seed=1,
            initial_artifact="```python\ndef is_even(n):\n    return n % 2 == 0\n```",
            initial_seam_result={"status": FAIL, "collision_type": "logical", "details": {}},
            initial_record={},
        )
        r = run_seek_evidence(state)
        self.assertIn(r["status"], (PASS, FAIL))
        self.assertEqual(r["tokens"], 0)
        self.assertIn("holdout_passes_despite_visible_fail", r)

    def test_stop_costs_zero_and_does_not_solve(self):
        state = CollisionState(
            collision_id="c3", task_family="arithmetic", task_id="ARITH_01", model_id="m", seed=1,
            initial_artifact="1 = 1", initial_seam_result={"status": FAIL, "collision_type": "numeric", "details": {}},
            initial_record={},
        )
        r = run_stop(state)
        self.assertEqual(r["tokens"], 0)
        self.assertFalse(r["solved"])


class RunArmNoLeakageTests(unittest.TestCase):
    """run_arm() -- с подменённым generate(), без GPU. Проверяет: токены
    считаются из ФАКТИЧЕСКОГО raw-ответа (не placeholder), а запись incorporates
    ровно тот initial_artifact, что был заморожен в состоянии."""

    def _state(self):
        seam = arithmetic_seam("total = 5", 26)
        return CollisionState(
            collision_id="col:x", task_family="arithmetic", task_id="ARITH_01", model_id=MODEL_IDS[0],
            seed=1, initial_artifact="total = 5", initial_seam_result=seam, initial_record={},
        )

    def test_run_arm_reports_actual_tokens_and_correct_status(self):
        state = self._state()
        config = ArmConfig("REGENERATE_SAME", MODEL_IDS[0], 0.5, "v1", 1000)
        # ARITH_01's actual oracle answer is 118 (3 * 18 + 2 * 32) -- must match
        # the real task, not an arbitrary number, or the seam legitimately FAILs.
        fake_raw = {
            "raw_text": "total = 118", "generation_failed": False, "generation_error": None,
            "input_tokens": 40, "output_tokens": 6, "generation_time_sec": 0.2,
        }
        with patch.object(harness, "generate", return_value=fake_raw):
            result = run_arm(state, config, llm=object())
        self.assertTrue(result["solved"])
        self.assertEqual(result["status"], PASS)
        self.assertEqual(result["tokens"], 46)
        self.assertEqual(result["record"]["initial_artifact"], "total = 5")

    def test_run_arm_generation_failure_is_error_not_fail(self):
        state = self._state()
        config = ArmConfig("REGENERATE_SAME", MODEL_IDS[0], 0.5, "v1", 1000)
        fake_raw = {
            "raw_text": "", "generation_failed": True, "generation_error": "boom",
            "input_tokens": None, "output_tokens": None, "generation_time_sec": 0.0,
        }
        with patch.object(harness, "generate", return_value=fake_raw):
            result = run_arm(state, config, llm=object())
        self.assertEqual(result["status"], ERROR)
        self.assertNotEqual(result["status"], FAIL)
        self.assertFalse(result["solved"])


class StatisticsTests(unittest.TestCase):
    def test_paired_sign_test_all_a_wins(self):
        a = [True, True, True, False]
        b = [False, False, False, False]
        r = paired_sign_test(a, b)
        self.assertEqual(r["n_a_wins"], 3)
        self.assertEqual(r["n_b_wins"], 0)
        self.assertLess(r["p_value"], 0.5)

    def test_paired_sign_test_no_difference(self):
        a = [True, False, True, False]
        b = [True, False, True, False]
        r = paired_sign_test(a, b)
        self.assertEqual(r["p_value"], 1.0)

    def test_bootstrap_ratio_diff_prefers_cheaper_arm(self):
        cheap = [{"solved": True, "tokens": 50} for _ in range(20)]
        expensive = [{"solved": True, "tokens": 200} for _ in range(20)]
        r = bootstrap_ratio_diff(cheap, expensive, n_resamples=500)
        self.assertLess(r["point_diff"], 0)
        self.assertLess(r["ci_95_diff"][1], 0)  # CI полностью ниже нуля -- cheap стабильно дешевле


class BestOfNTests(unittest.TestCase):
    def test_oracle_counts_any_pass_in_combo(self):
        records = [
            {"model": "m", "seam_type": "arithmetic", "task_id": "t1", "status": FAIL, "result": "a", "total_tokens": 10},
            {"model": "m", "seam_type": "arithmetic", "task_id": "t1", "status": PASS, "result": "b", "total_tokens": 10},
        ]
        out = best_of_n(records)
        self.assertEqual(out["N=2"]["oracle_success_rate"], 1.0)
        self.assertEqual(out["N=2"]["n_combinations"], 1)
        self.assertEqual(out["N=2"]["mean_tokens"], 20)

    def test_n1_matches_raw_success_rate(self):
        records = [
            {"model": "m", "seam_type": "arithmetic", "task_id": "t1", "status": PASS, "result": "a", "total_tokens": 10},
            {"model": "m", "seam_type": "arithmetic", "task_id": "t1", "status": FAIL, "result": "b", "total_tokens": 10},
        ]
        out = best_of_n(records)
        self.assertEqual(out["N=1"]["n_combinations"], 2)
        self.assertEqual(out["N=1"]["oracle_success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
