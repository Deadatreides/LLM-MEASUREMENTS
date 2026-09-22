"""Контрольные случаи харнесса эксперимента 12 (не моделей -- GPU не
нужен). Дисциплина -- та же, что в Эксперименте 11: неоднозначное
извлечение -> INAPPLICABLE, не угаданный FAIL; общий стартовый контекст
для O0/O1 должен быть буквально идентичен; антиутечная проверка должна
реально ловить подложенное нарушение, не только молчать на чистых данных.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import context_builder as cb  # noqa: E402
import context_manifest as cm  # noqa: E402
import harness  # noqa: E402
import outcome_classification as oc  # noqa: E402
from configs.model_registry import MODEL_IDS  # noqa: E402
from harness import ArmConfig, CollisionState, LeakDetected, build_arm_configs, run_arm  # noqa: E402
from seams import (  # noqa: E402
    ERROR, FAIL, INAPPLICABLE, PASS, UNKNOWN,
    classify_arithmetic_subtype, classify_code_subtype, collision_subtype,
    enriched_code_seam, multistep_arithmetic_seam,
)
from tasks.arithmetic_multistep_tasks import TASKS as ATASKS  # noqa: E402
from tasks.code_tasks import CODE_TASKS  # noqa: E402


# -- швы ----------------------------------------------------------------------


class MultistepArithmeticSeamTests(unittest.TestCase):
    def setUp(self):
        self.task = ATASKS["MSARITH_01"]
        self.steps = self.task["steps"]

    def _text(self, overrides=None):
        overrides = overrides or {}
        lines = []
        for s in self.steps:
            val = overrides.get(s["name"], s["value"])
            lines.append(f"{s['name']} = {val}")
        return "\n".join(lines)

    def test_all_correct_is_pass(self):
        r = multistep_arithmetic_seam(self._text(), self.steps)
        self.assertEqual(r["status"], PASS)
        self.assertTrue(all(x["status"] == PASS for x in r["step_results"]))

    def test_wrong_final_step_is_fail_numeric(self):
        final_name = self.steps[-1]["name"]
        r = multistep_arithmetic_seam(self._text({final_name: 999999}), self.steps)
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(r["collision_type"], "numeric")
        self.assertTrue(all(x["status"] == PASS for x in r["step_results"][:-1]))
        self.assertEqual(r["step_results"][-1]["status"], FAIL)

    def test_missing_final_step_is_inapplicable_not_fail(self):
        lines = [f"{s['name']} = {s['value']}" for s in self.steps[:-1]]
        r = multistep_arithmetic_seam("\n".join(lines), self.steps)
        self.assertEqual(r["status"], INAPPLICABLE)
        self.assertNotEqual(r["status"], FAIL)

    def test_ambiguous_line_shown_work_still_extracts_single_result(self):
        # "name = 3*18 = 54" -- ровно один equation-результат после
        # первого "=" -> извлекается однозначно, тот же принцип, что в
        # seams.py Эксперимента 11.
        first = self.steps[0]
        line = f"{first['name']} = {first['formula']} = {first['value']}"
        rest = "\n".join(f"{s['name']} = {s['value']}" for s in self.steps[1:])
        r = multistep_arithmetic_seam(line + "\n" + rest, self.steps)
        self.assertEqual(r["step_results"][0]["status"], PASS)

    def test_duplicate_step_name_is_inapplicable(self):
        first = self.steps[0]
        lines = [f"{first['name']} = {first['value']}", f"{first['name']} = {first['value']}"]
        lines += [f"{s['name']} = {s['value']}" for s in self.steps[1:]]
        r = multistep_arithmetic_seam("\n".join(lines), self.steps)
        self.assertEqual(r["step_results"][0]["status"], INAPPLICABLE)

    def test_never_returns_unknown(self):
        cases = [
            multistep_arithmetic_seam("nonsense output", self.steps),
            multistep_arithmetic_seam(self._text(), self.steps),
            multistep_arithmetic_seam(self._text({self.steps[-1]["name"]: 0}), self.steps),
        ]
        for r in cases:
            self.assertNotEqual(r["status"], UNKNOWN)


class EnrichedCodeSeamTests(unittest.TestCase):
    def setUp(self):
        self.task = CODE_TASKS["MSCODE_01"]

    def test_correct_code_is_pass_all_requirements(self):
        code = "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
        r = enriched_code_seam(code, self.task["function_name"], self.task["requirements"])
        self.assertEqual(r["status"], PASS)
        self.assertTrue(all(x["status"] == PASS for x in r["requirement_results"]))

    def test_wrong_logic_marks_only_failing_requirements(self):
        code = "```python\ndef is_even(n):\n    return True\n```"
        r = enriched_code_seam(code, self.task["function_name"], self.task["requirements"])
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(r["collision_type"], "logical")
        statuses = {x["name"]: x["status"] for x in r["requirement_results"]}
        self.assertEqual(statuses["typical_even"], PASS)
        self.assertEqual(statuses["typical_odd"], FAIL)

    def test_no_code_all_requirements_inapplicable(self):
        r = enriched_code_seam("I would use modulo.", self.task["function_name"], self.task["requirements"])
        self.assertEqual(r["status"], INAPPLICABLE)
        self.assertTrue(all(x["status"] == INAPPLICABLE for x in r["requirement_results"]))

    def test_syntax_error_all_requirements_inapplicable(self):
        code = "```python\ndef is_even(n)\n    return n % 2 == 0\n```"
        r = enriched_code_seam(code, self.task["function_name"], self.task["requirements"])
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(r["collision_type"], "syntactic")
        self.assertTrue(all(x["status"] == INAPPLICABLE for x in r["requirement_results"]))

    def test_infinite_loop_in_candidate_times_out_instead_of_hanging(self):
        # Регрессия: найдено на реальном пилоте эксперимента 13 --
        # eval(req["check"], ...) без таймаута блокировал процесс
        # НАВСЕГДА на функции с достижимым бесконечным циклом. Здесь
        # ожидается FAIL/logical с TIMEOUT в причине, а не зависание --
        # если фикс сломан, этот тест сам зависнет (что и обнаружит регресс).
        code = "```python\ndef is_even(n):\n    while True:\n        pass\n    return n % 2 == 0\n```"
        r = enriched_code_seam(code, self.task["function_name"], self.task["requirements"])
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(r["collision_type"], "logical")
        self.assertTrue(any("TIMEOUT" in f for f in r["details"]["failing_requirements"]))

    def test_wrong_function_name_is_structural(self):
        code = "```python\ndef check_even(n):\n    return n % 2 == 0\n```"
        r = enriched_code_seam(code, self.task["function_name"], self.task["requirements"])
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(r["collision_type"], "structural")


# -- построитель контекста и антиутечная проверка ----------------------------


class ContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.task = ATASKS["MSARITH_01"]
        self.steps = self.task["steps"]
        final = self.steps[-1]
        lines = [f"{s['name']} = {s['value']}" for s in self.steps[:-1]] + [f"{final['name']} = 999999"]
        self.seam = multistep_arithmetic_seam("\n".join(lines), self.steps)

    def test_k1_never_contains_any_step_value(self):
        text = cb.structure_text("K1", "arithmetic", self.task, self.seam)
        for step in self.steps:
            self.assertNotIn(str(step["value"]), text)
            if float(step["value"]).is_integer():
                self.assertNotIn(str(int(step["value"])), text)

    def test_k2_reveals_only_confirmed_steps(self):
        text = cb.structure_text("K2", "arithmetic", self.task, self.seam)
        for step in self.steps[:-1]:  # все, кроме финального, подтверждены PASS
            self.assertIn(str(step["value"]), text)
        final = self.steps[-1]
        self.assertNotIn(str(final["value"]), text)  # финальный шаг никогда не PASS для коллизии

    def test_check_no_leak_passes_on_legitimate_k1(self):
        text = cb.structure_text("K1", "arithmetic", self.task, self.seam)
        forbidden = cb.forbidden_values_for_level("K1", "arithmetic", self.task, self.seam)
        self.assertTrue(cb.check_no_leak(text, forbidden))

    def test_check_no_leak_passes_on_legitimate_k2(self):
        text = cb.structure_text("K2", "arithmetic", self.task, self.seam)
        forbidden = cb.forbidden_values_for_level("K2", "arithmetic", self.task, self.seam)
        self.assertTrue(cb.check_no_leak(text, forbidden))

    def test_check_no_leak_catches_deliberately_injected_answer(self):
        text = cb.structure_text("K1", "arithmetic", self.task, self.seam)
        forbidden = cb.forbidden_values_for_level("K1", "arithmetic", self.task, self.seam)
        broken = text + f"\n(hint: the final answer is {self.task['answer']})"
        self.assertFalse(cb.check_no_leak(broken, forbidden))

    def test_check_no_leak_catches_unconfirmed_step_value_in_k2(self):
        text = cb.structure_text("K2", "arithmetic", self.task, self.seam)
        forbidden = cb.forbidden_values_for_level("K2", "arithmetic", self.task, self.seam)
        final = self.steps[-1]
        broken = text + f"\n(by the way {final['name']} = {final['value']})"
        self.assertFalse(cb.check_no_leak(broken, forbidden))

    def test_describe_step_never_emits_a_raw_literal(self):
        for step in self.steps:
            desc = cb.describe_step(step["formula"])
            self.assertNotIn(str(step["value"]), desc)

    def test_check_no_leak_does_not_false_positive_on_number_substring(self):
        # regression: forbidden bare-integer variant "3" (from oracle
        # value 3.0) must NOT match inside an unrelated confirmed value
        # like "320.0" (naive `in` substring search did exactly this on
        # the pilot run -- caught and fixed via number-boundary regex).
        text = "combined_weight = 320.0  (already confirmed correct in your previous attempt)"
        self.assertTrue(cb.check_no_leak(text, [3.0]))
        # but a genuine standalone "3" must still be caught
        self.assertFalse(cb.check_no_leak("containers_needed = 3", [3.0]))
        self.assertFalse(cb.check_no_leak("containers_needed = 3.0", [3.0]))


class CodeStructureTextTests(unittest.TestCase):
    def test_k1_lists_requirement_names_not_check_expressions(self):
        task = CODE_TASKS["MSCODE_01"]
        seam = enriched_code_seam("no code", task["function_name"], task["requirements"])
        text = cb.structure_text("K1", "code", task, seam)
        for req in task["requirements"]:
            self.assertIn(req["name"], text)
            self.assertNotIn(req["check"], text)


# -- 7 плеч, общий стартовый контекст -----------------------------------------


class ArmConfigTests(unittest.TestCase):
    def test_seven_distinct_signatures(self):
        rng = random.Random(1)
        configs = build_arm_configs(MODEL_IDS[0], list(MODEL_IDS), rng)
        self.assertEqual(len(configs), 7)
        sigs = {c.signature() for c in configs.values()}
        self.assertEqual(len(sigs), 7)

    def test_retry_seed_never_matches_initial_seeds(self):
        rng = random.Random(1)
        configs = build_arm_configs(MODEL_IDS[0], list(MODEL_IDS), rng)
        for c in configs.values():
            self.assertNotIn(c.seed, harness.INITIAL_SEEDS)

    def test_o1_arms_share_the_same_alt_model(self):
        rng = random.Random(1)
        configs = build_arm_configs(MODEL_IDS[0], list(MODEL_IDS), rng)
        alt_models = {configs["K0O1"].model_id, configs["K1O1"].model_id, configs["K2O1"].model_id}
        self.assertEqual(len(alt_models), 1, "K0O1/K1O1/K2O1 must use the SAME alt model for the interaction test to be meaningful")
        self.assertNotEqual(configs["K0O1"].model_id, MODEL_IDS[0])

    def test_h_syn_arm_uses_base_model_and_k0_level(self):
        rng = random.Random(1)
        configs = build_arm_configs(MODEL_IDS[0], list(MODEL_IDS), rng)
        arm = configs["K0O0_promptB"]
        self.assertEqual(arm.model_id, MODEL_IDS[0])
        self.assertEqual(arm.level, "K0")
        self.assertEqual(arm.prompt_variant, "promptB")


class CommonStartingPointTests(unittest.TestCase):
    """п.3 задания: промпт для O0/O1 внутри одного уровня контекста
    обязан быть побайтово идентичен -- проверяется на РЕАЛЬНОМ пути
    run_arm(), не только на билдере промпта отдельно."""

    def _state(self):
        task = ATASKS["MSARITH_01"]
        steps = task["steps"]
        lines = [f"{s['name']} = {s['value']}" for s in steps[:-1]] + [f"{steps[-1]['name']} = 999999"]
        seam = multistep_arithmetic_seam("\n".join(lines), steps)
        return CollisionState(
            collision_id="col:x", task_family="arithmetic", task_id="MSARITH_01", model_id=MODEL_IDS[0],
            seed=1, initial_artifact="\n".join(lines), initial_seam_result=seam, initial_record={},
        )

    def _capture_prompt(self, state, config):
        captured = {}

        def fake_generate(llm, model_id, prompt, **kwargs):
            captured["prompt"] = prompt
            return {"raw_text": "x = 1", "generation_failed": False, "generation_error": None,
                    "input_tokens": 10, "output_tokens": 5, "generation_time_sec": 0.1}

        with patch.object(harness, "generate", side_effect=fake_generate):
            run_arm(state, config, llm=object())
        return captured["prompt"]

    def test_k0_prompt_identical_across_operators(self):
        state = self._state()
        p_o0 = self._capture_prompt(state, ArmConfig("K0O0", "K0", "standard", MODEL_IDS[0], 1000, 0.5))
        p_o1 = self._capture_prompt(state, ArmConfig("K0O1", "K0", "standard", MODEL_IDS[1], 1000, 0.5))
        self.assertEqual(p_o0, p_o1)

    def test_k2_prompt_identical_across_operators(self):
        state = self._state()
        p_o0 = self._capture_prompt(state, ArmConfig("K2O0", "K2", "standard", MODEL_IDS[0], 1000, 0.5))
        p_o1 = self._capture_prompt(state, ArmConfig("K2O1", "K2", "standard", MODEL_IDS[1], 1000, 0.5))
        self.assertEqual(p_o0, p_o1)

    def test_k0_and_k1_prompts_differ(self):
        state = self._state()
        p_k0 = self._capture_prompt(state, ArmConfig("K0O0", "K0", "standard", MODEL_IDS[0], 1000, 0.5))
        p_k1 = self._capture_prompt(state, ArmConfig("K1O0", "K1", "standard", MODEL_IDS[0], 1000, 0.5))
        self.assertNotEqual(p_k0, p_k1)


class RunArmLeakWiringTests(unittest.TestCase):
    """Проверяет ПРОВОДКУ, не только check_no_leak() саму по себе:
    если антиутечная проверка возвращает False, run_arm обязан бросить
    LeakDetected И не должен успеть вызвать generate()."""

    def _state(self):
        task = ATASKS["MSARITH_01"]
        steps = task["steps"]
        lines = [f"{s['name']} = {s['value']}" for s in steps[:-1]] + [f"{steps[-1]['name']} = 999999"]
        seam = multistep_arithmetic_seam("\n".join(lines), steps)
        return CollisionState(
            collision_id="col:leak", task_family="arithmetic", task_id="MSARITH_01", model_id=MODEL_IDS[0],
            seed=1, initial_artifact="\n".join(lines), initial_seam_result=seam, initial_record={},
        )

    def test_forced_leak_raises_and_blocks_generation(self):
        state = self._state()
        config = ArmConfig("K1O0", "K1", "standard", MODEL_IDS[0], 1000, 0.5)
        generate_called = []

        def fake_generate(*a, **kw):
            generate_called.append(True)
            return {"raw_text": "x", "generation_failed": False, "generation_error": None,
                    "input_tokens": 1, "output_tokens": 1, "generation_time_sec": 0.0}

        with patch.object(harness, "validate_prompt_against_manifest", return_value={"ok": False, "violations": ["forced"]}), \
             patch.object(harness, "generate", side_effect=fake_generate):
            with self.assertRaises(LeakDetected):
                run_arm(state, config, llm=object())
        self.assertEqual(generate_called, [], "generate() must not be called once a leak is detected")


# -- статистика ---------------------------------------------------------------


class StatisticsTests(unittest.TestCase):
    def test_paired_sign_test_detects_one_sided_win(self):
        a = [True, True, True, False]
        b = [False, False, False, False]
        r = harness.paired_sign_test(a, b)
        self.assertEqual(r["n_a_wins"], 3)
        self.assertLess(r["p_value"], 0.5)

    def test_bootstrap_ratio_diff_favors_cheaper_arm(self):
        cheap = [{"solved": True, "tokens": 50} for _ in range(20)]
        expensive = [{"solved": True, "tokens": 200} for _ in range(20)]
        r = harness.bootstrap_ratio_diff(cheap, expensive, n_resamples=500)
        self.assertLess(r["point_diff"], 0)

    def test_bootstrap_interaction_detects_true_interaction(self):
        # K0: O0 и O1 одинаковы (0.2 успеха); Ki: O1 намного лучше O0 ->
        # положительное взаимодействие.
        n = 60
        k0o0 = [i % 5 == 0 for i in range(n)]
        k0o1 = [i % 5 == 0 for i in range(n)]
        kio0 = [i % 5 == 0 for i in range(n)]
        kio1 = [i % 2 == 0 for i in range(n)]
        r = harness.bootstrap_interaction(k0o0, k0o1, kio0, kio1, n_resamples=500)
        self.assertGreater(r["point_estimate"], 0)

    def test_blind_resample_baseline_uses_only_sibling_seeds(self):
        state = CollisionState(
            collision_id="c1", task_family="arithmetic", task_id="T1", model_id="m", seed=1,
            initial_artifact="", initial_seam_result={"step_results": [{"status": PASS}]}, initial_record={},
        )
        records = [
            {"model": "m", "seam_type": "arithmetic", "task_id": "T1", "seed": 1, "status": FAIL, "total_tokens": 10},
            {"model": "m", "seam_type": "arithmetic", "task_id": "T1", "seed": 2, "status": PASS, "total_tokens": 20},
            {"model": "m", "seam_type": "arithmetic", "task_id": "T1", "seed": 3, "status": FAIL, "total_tokens": 30},
        ]
        r = harness.blind_resample_baseline([state], records)
        self.assertEqual(r["n_attempts"], 2)  # seed 2 и 3, НЕ seed 1 (это сам anchor)
        self.assertEqual(r["n_solved"], 1)


# -- v2: пятисоставный статус (outcome_classification) -----------------------


class OutcomeClassificationTests(unittest.TestCase):
    def test_pass(self):
        init = {"status": FAIL, "collision_type": "numeric", "details": {"final": {"name": "x", "extracted": 1.0}}}
        retry = {"status": PASS, "collision_type": None, "details": {}}
        self.assertEqual(oc.classify_outcome(init, retry), oc.PASS_STATUS)

    def test_same_failure_numeric(self):
        init = {"status": FAIL, "collision_type": "numeric", "details": {"final": {"name": "x", "extracted": 5.0}}}
        retry = {"status": FAIL, "collision_type": "numeric", "details": {"final": {"name": "x", "extracted": 5.0}}}
        self.assertEqual(oc.classify_outcome(init, retry), oc.SAME_FAILURE)

    def test_different_failure_numeric_value(self):
        init = {"status": FAIL, "collision_type": "numeric", "details": {"final": {"name": "x", "extracted": 5.0}}}
        retry = {"status": FAIL, "collision_type": "numeric", "details": {"final": {"name": "x", "extracted": 9.0}}}
        self.assertEqual(oc.classify_outcome(init, retry), oc.DIFFERENT_FAILURE)

    def test_different_failure_type_change(self):
        init = {"status": FAIL, "collision_type": "numeric", "details": {"final": {"name": "x", "extracted": 5.0}}}
        retry = {"status": FAIL, "collision_type": "logical", "details": {"failing_requirements": ["a: f() == 1 -> False"]}}
        self.assertEqual(oc.classify_outcome(init, retry), oc.DIFFERENT_FAILURE)

    def test_inapplicable_passthrough(self):
        init = {"status": FAIL, "collision_type": "numeric", "details": {"final": {"name": "x", "extracted": 5.0}}}
        retry = {"status": INAPPLICABLE, "collision_type": "format", "details": {}}
        self.assertEqual(oc.classify_outcome(init, retry), oc.INAPPLICABLE_STATUS)

    def test_error_passthrough(self):
        init = {"status": FAIL, "collision_type": "numeric", "details": {"final": {"name": "x", "extracted": 5.0}}}
        retry = {"status": ERROR, "collision_type": None, "details": {}}
        self.assertEqual(oc.classify_outcome(init, retry), oc.ERROR_STATUS)

    def test_unverified_artificial_case(self):
        # Ни один реальный шов этого стенда не производит статус,
        # отличный от PASS/FAIL/INAPPLICABLE/ERROR/UNKNOWN -- UNVERIFIED
        # architecturally недостижим в реальных данных (см. докстринг
        # модуля). Здесь подкладывается ИСКУССТВЕННЫЙ статус, чтобы
        # доказать, что классификатор УМЕЕТ произвести UNVERIFIED, если
        # когда-нибудь появится путь, дающий его.
        init = {"status": FAIL, "collision_type": "numeric", "details": {"final": {"name": "x", "extracted": 5.0}}}
        retry = {"status": "SOME_FUTURE_AMBIGUOUS_STATUS", "collision_type": None, "details": {}}
        self.assertEqual(oc.classify_outcome(init, retry), oc.UNVERIFIED)

    def test_regressions_and_improvements_symmetry(self):
        init = {"step_results": [{"name": "a", "status": PASS}, {"name": "b", "status": FAIL}]}
        retry_regressed = {"step_results": [{"name": "a", "status": FAIL}, {"name": "b", "status": FAIL}]}
        retry_improved = {"step_results": [{"name": "a", "status": PASS}, {"name": "b", "status": PASS}]}
        self.assertEqual(oc.detect_regressions(init, retry_regressed), ["a"])
        self.assertEqual(oc.detect_improvements(init, retry_regressed), [])
        self.assertEqual(oc.detect_regressions(init, retry_improved), [])
        self.assertEqual(oc.detect_improvements(init, retry_improved), ["b"])

    def test_run_arm_wires_same_and_different_failure_end_to_end(self):
        task = ATASKS["MSARITH_01"]
        steps = task["steps"]
        wrong_lines = [f"{s['name']} = {s['value']}" for s in steps[:-1]] + [f"{steps[-1]['name']} = 111"]
        seam = multistep_arithmetic_seam("\n".join(wrong_lines), steps)
        state = CollisionState(
            collision_id="col:same-diff", task_family="arithmetic", task_id="MSARITH_01", model_id=MODEL_IDS[0],
            seed=1, initial_artifact="\n".join(wrong_lines), initial_seam_result=seam, initial_record={},
        )
        config = ArmConfig("K0O0", "K0", "standard", MODEL_IDS[0], 1000, 0.5)

        def make_fake_generate(final_value):
            retry_lines = [f"{s['name']} = {s['value']}" for s in steps[:-1]] + [f"{steps[-1]['name']} = {final_value}"]

            def fake_generate(*a, **kw):
                return {"raw_text": "\n".join(retry_lines), "generation_failed": False, "generation_error": None,
                        "input_tokens": 10, "output_tokens": 5, "generation_time_sec": 0.1}
            return fake_generate

        with patch.object(harness, "generate", side_effect=make_fake_generate(111)):
            same = run_arm(state, config, llm=object())
        self.assertEqual(same["final_status"], oc.SAME_FAILURE)

        with patch.object(harness, "generate", side_effect=make_fake_generate(222)):
            different = run_arm(state, config, llm=object())
        self.assertEqual(different["final_status"], oc.DIFFERENT_FAILURE)


# -- v2: dependency vs value-calculation subtype ------------------------------


class ArithmeticSubtypeTests(unittest.TestCase):
    def setUp(self):
        self.task = ATASKS["MSARITH_03"]  # room1_area, room2_area, total_area, paint_needed, paint_cost
        self.steps = self.task["steps"]

    def test_fresh_value_error_on_correct_inputs(self):
        lines = [f"{s['name']} = {s['value']}" for s in self.steps[:-1]] + [f"{self.steps[-1]['name']} = 148.75"]
        r = multistep_arithmetic_seam("\n".join(lines), self.steps)
        self.assertEqual(classify_arithmetic_subtype(r["step_results"], self.steps), "value")

    def test_correctly_propagated_upstream_error_is_dependency(self):
        wrong_room1 = 999.0
        room2 = self.steps[1]["value"]
        total = wrong_room1 + room2
        paint_needed = total / 10
        paint_cost = paint_needed * 25
        lines = [
            f"room1_area = {wrong_room1}", f"room2_area = {room2}", f"total_area = {total}",
            f"paint_needed = {paint_needed}", f"paint_cost = {paint_cost}",
        ]
        r = multistep_arithmetic_seam("\n".join(lines), self.steps)
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(classify_arithmetic_subtype(r["step_results"], self.steps), "dependency")

    def test_other_when_not_fail(self):
        lines = [f"{s['name']} = {s['value']}" for s in self.steps]
        r = multistep_arithmetic_seam("\n".join(lines), self.steps)
        self.assertEqual(classify_arithmetic_subtype(r["step_results"], self.steps), "other")

    def test_code_subtype_mapping(self):
        self.assertEqual(classify_code_subtype("syntactic"), "syntax")
        self.assertEqual(classify_code_subtype("structural"), "contract_interface")
        self.assertEqual(classify_code_subtype("logical"), "semantic")
        self.assertEqual(classify_code_subtype("format"), "other")
        self.assertEqual(classify_code_subtype(None), "other")

    def test_collision_subtype_none_when_not_fail(self):
        code_task = CODE_TASKS["MSCODE_01"]
        good_code = "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
        r = enriched_code_seam(good_code, code_task["function_name"], code_task["requirements"])
        self.assertIsNone(collision_subtype("code", r))


# -- v2: context manifest -----------------------------------------------------


class ContextManifestTests(unittest.TestCase):
    def setUp(self):
        self.task = ATASKS["MSARITH_01"]
        self.steps = self.task["steps"]
        lines = [f"{s['name']} = {s['value']}" for s in self.steps[:-1]] + [f"{self.steps[-1]['name']} = 999999"]
        self.artifact = "\n".join(lines)
        self.seam = multistep_arithmetic_seam(self.artifact, self.steps)

    def test_all_three_levels_pass_on_legitimate_prompts(self):
        for level in ("K0", "K1", "K2"):
            prompt = cb.build_prompt(level, "arithmetic", self.task, self.artifact, self.seam)
            result = cm.validate_prompt_against_manifest(level, "arithmetic", self.task, self.artifact, self.seam, prompt)
            self.assertTrue(result["ok"], f"{level}: {result['violations']}")

    def test_k0_prompt_injected_with_structure_marker_is_caught(self):
        prompt = cb.build_prompt("K0", "arithmetic", self.task, self.artifact, self.seam)
        broken = prompt + "\n\nStructure of this problem (step name = how it is computed): x = y"
        result = cm.validate_prompt_against_manifest("K0", "arithmetic", self.task, self.artifact, self.seam, broken)
        self.assertFalse(result["ok"])

    def test_k1_prompt_missing_structure_marker_is_caught(self):
        result = cm.validate_prompt_against_manifest("K1", "arithmetic", self.task, self.artifact, self.seam, "just a plain prompt with no structure")
        self.assertFalse(result["ok"])

    def test_substituted_artifact_is_caught(self):
        prompt = cb.build_prompt("K0", "arithmetic", self.task, self.artifact, self.seam)
        substituted = prompt.replace(self.artifact, "totally different fabricated artifact")
        result = cm.validate_prompt_against_manifest("K0", "arithmetic", self.task, self.artifact, self.seam, substituted)
        self.assertFalse(result["ok"])

    def test_manifest_declares_all_three_levels(self):
        self.assertEqual(set(cm.MANIFEST.keys()) - {"forbidden_everywhere"}, {"K0", "K1", "K2"})


if __name__ == "__main__":
    unittest.main()
