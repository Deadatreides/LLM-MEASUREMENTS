"""Контрольные случаи харнесса эксперимента 13 (GPU не нужен). Дисциплина
-- та же, что в предыдущих экспериментах проекта: неоднозначное
извлечение -> INAPPLICABLE; общий evidence-контекст идентичен между
режимами; классификатор новизны не должен путать перефразирование с
настоящим изменением структуры (задание §10 -- псевдоновизна).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import context_builder as cb  # noqa: E402
import harness  # noqa: E402
import representation as rep  # noqa: E402
from harness import CollisionState13, PromptViolation, run_mode  # noqa: E402
from seams import ERROR, FAIL, INAPPLICABLE, PASS, final_answer_seam  # noqa: E402
from tasks import ARITH_TASKS, CODE_TASKS  # noqa: E402


# -- final_answer_seam ---------------------------------------------------------


class FinalAnswerSeamTests(unittest.TestCase):
    def test_exact_match_is_pass(self):
        r = final_answer_seam("some reasoning\nFINAL ANSWER = 129.8", 129.8)
        self.assertEqual(r["status"], PASS)

    def test_wrong_value_is_fail_numeric(self):
        r = final_answer_seam("FINAL ANSWER = 100", 129.8)
        self.assertEqual(r["status"], FAIL)
        self.assertEqual(r["collision_type"], "numeric")
        self.assertEqual(r["details"]["final"]["name"], "FINAL")

    def test_missing_marker_is_inapplicable_not_fail(self):
        r = final_answer_seam("I computed something but forgot to state it clearly.", 129.8)
        self.assertEqual(r["status"], INAPPLICABLE)
        self.assertNotEqual(r["status"], FAIL)

    def test_duplicate_marker_is_inapplicable(self):
        r = final_answer_seam("FINAL ANSWER = 1\nFINAL ANSWER = 2", 129.8)
        self.assertEqual(r["status"], INAPPLICABLE)

    def test_case_insensitive_marker(self):
        r = final_answer_seam("final answer = 129.8", 129.8)
        self.assertEqual(r["status"], PASS)


# -- representation.py: парсинг и fingerprint -----------------------------------


class SectionParsingTests(unittest.TestCase):
    def test_r1_sections_parsed(self):
        text = "NEW REPRESENTATION: use unit pricing\nSOLUTION:\nx = 1\ny = 2\nFINAL ANSWER = 3"
        sections = rep.parse_sections(text)
        self.assertEqual(sections["NEW REPRESENTATION"], "use unit pricing")
        self.assertIn("x = 1", sections["SOLUTION"])
        self.assertEqual(rep.representation_text("R1", sections), "use unit pricing")

    def test_r2_sections_parsed(self):
        text = "DECOMPOSITION DECISION: no change needed\nNEW DECOMPOSITION: unchanged\nSOLUTION:\nx = 1\nFINAL ANSWER = 1"
        sections = rep.parse_sections(text)
        combined = rep.representation_text("R2", sections)
        self.assertIn("no change needed", combined)
        self.assertIn("unchanged", combined)

    def test_missing_markers_gives_empty_sections(self):
        sections = rep.parse_sections("just some free text with no markers at all")
        self.assertEqual(sections, {})


class FingerprintTests(unittest.TestCase):
    def test_arithmetic_fingerprint_from_task(self):
        task = ARITH_TASKS["MSARITH_01"]
        fp = rep.arithmetic_fingerprint_from_task(task)
        self.assertTrue(fp["parsed"])
        self.assertEqual(fp["n_steps"], len(task["steps"]))

    def test_arithmetic_fingerprint_from_text_ignores_prose(self):
        fp = rep.arithmetic_fingerprint_from_text("This is just an explanation with no computation lines.")
        self.assertFalse(fp["parsed"])
        self.assertEqual(fp["n_steps"], 0)

    def test_arithmetic_fingerprint_parses_arbitrary_names(self):
        fp = rep.arithmetic_fingerprint_from_text("alpha = 1\nbeta = alpha + 2\ngamma = beta * 3")
        self.assertEqual(fp["n_steps"], 3)
        self.assertEqual(fp["step_name_set"], frozenset({"alpha", "beta", "gamma"}))
        self.assertGreaterEqual(fp["n_graph_edges"], 2)  # beta references alpha, gamma references beta

    def test_arithmetic_fingerprint_fallback_parses_labeled_prose_lines(self):
        # Регрессия: найдено на реальном пилоте -- модели в R1/R2 часто
        # пишут прозой с инлайн-вычислением, а не чистым `name = value`;
        # строгий парсер давал INVALID_STRUCTURE даже на PASS-случаях с
        # настоящей новой структурой. Запасной разбор должен ловить хотя
        # бы "label: ... = value" построчно.
        text = (
            "- Calculate the cost of sugar for one cake: $2.5kg * $2/kg = $5\n"
            "- Calculate the cost of flour for one cake: $4kg * $2/kg = $8\n"
            "- Calculate the total cost of ingredients for one cake: $5 + $8 = $13"
        )
        fp = rep.arithmetic_fingerprint_from_text(text)
        self.assertTrue(fp["parsed"])
        self.assertEqual(fp["n_steps"], 3)

    def test_code_fingerprint_no_code_not_parsed(self):
        fp = rep.code_fingerprint("I would use a loop but here is no code.")
        self.assertFalse(fp["parsed"])

    def test_code_fingerprint_detects_recursion(self):
        src = "```python\ndef f(n):\n    if n <= 1:\n        return 1\n    return n * f(n - 1)\n```"
        fp = rep.code_fingerprint(src)
        self.assertTrue(fp["parsed"])
        self.assertTrue(fp["is_recursive"])

    def test_code_fingerprint_detects_iteration(self):
        src = "```python\ndef f(n):\n    total = 1\n    for i in range(1, n + 1):\n        total *= i\n    return total\n```"
        fp = rep.code_fingerprint(src)
        self.assertTrue(fp["parsed"])
        self.assertFalse(fp["is_recursive"])
        self.assertEqual(fp["n_for"], 1)


class NoveltyClassificationTests(unittest.TestCase):
    """Задание §10 -- псевдоновизна: модель может ЗАЯВИТЬ о новом
    подходе, но фактически повторить прежнюю структуру -- это НЕ должно
    классифицироваться как NOVEL_STRUCTURE."""

    def test_prefix_padded_renaming_is_not_novel_structure(self):
        # Регрессия: найдено на реальном пилоте -- модель дописала
        # 'calculate_the_'/'add_the_' к прежним именам шагов, сохранив ту
        # же вычислительную структуру. Точное сравнение множеств имён
        # засчитывало это как NOVEL_STRUCTURE -- нечёткое сравнение
        # (подстрока/пересечение слов) должно поймать это как минимум
        # MINOR_REFORMULATION, не NOVEL_STRUCTURE.
        before_fp = {
            "parsed": True, "n_steps": 5,
            "step_names": ("discount_amount", "price_after_discount", "tax", "price_with_tax", "total"),
            "step_name_set": frozenset({"discount_amount", "price_after_discount", "tax", "price_with_tax", "total"}),
        }
        after_fp = {
            "parsed": True, "n_steps": 5,
            "step_names": ("calculate_the_discount_amount", "calculate_the_discounted_price", "calculate_the_tax", "calculate_the_price_with_tax", "add_the_shipping_fee"),
            "step_name_set": frozenset({"calculate_the_discount_amount", "calculate_the_discounted_price", "calculate_the_tax", "calculate_the_price_with_tax", "add_the_shipping_fee"}),
        }
        self.assertNotEqual(rep.classify_novelty("arithmetic", before_fp, after_fp), rep.NOVEL_STRUCTURE)

    def setUp(self):
        self.task = ARITH_TASKS["MSARITH_01"]
        self.before_fp = rep.arithmetic_fingerprint_from_task(self.task)

    def test_pseudo_novelty_same_steps_same_order_is_same_structure(self):
        steps = self.task["steps"]
        text = "\n".join(f"{s['name']} = {s['value']}" for s in steps)
        after_fp = rep.arithmetic_fingerprint_from_text(text)
        self.assertEqual(rep.classify_novelty("arithmetic", self.before_fp, after_fp), rep.SAME_STRUCTURE)
        self.assertEqual(rep.classify_pseudo_novelty("arithmetic", self.before_fp, after_fp), "rephrasing")

    def test_reordered_same_names_is_minor_reformulation(self):
        steps = list(reversed(self.task["steps"]))
        text = "\n".join(f"{s['name']} = {s['value']}" for s in steps)
        after_fp = rep.arithmetic_fingerprint_from_text(text)
        self.assertEqual(rep.classify_novelty("arithmetic", self.before_fp, after_fp), rep.MINOR_REFORMULATION)
        self.assertEqual(rep.classify_pseudo_novelty("arithmetic", self.before_fp, after_fp), "order_change")

    def test_different_names_and_count_is_novel_structure(self):
        text = "combined = 1\nblended_rate = 2\nresult = combined * blended_rate"
        after_fp = rep.arithmetic_fingerprint_from_text(text)
        self.assertEqual(rep.classify_novelty("arithmetic", self.before_fp, after_fp), rep.NOVEL_STRUCTURE)
        self.assertEqual(rep.classify_pseudo_novelty("arithmetic", self.before_fp, after_fp), "genuine_new_method")

    def test_unparseable_response_is_invalid_structure(self):
        after_fp = rep.arithmetic_fingerprint_from_text("I am not sure how to solve this.")
        self.assertEqual(rep.classify_novelty("arithmetic", self.before_fp, after_fp), rep.INVALID_STRUCTURE)
        self.assertEqual(rep.classify_pseudo_novelty("arithmetic", self.before_fp, after_fp), "invalid_attempt")

    def test_code_recursion_vs_iteration_is_novel(self):
        before_fp = rep.code_fingerprint("```python\ndef f(n):\n    total = 1\n    for i in range(1, n+1):\n        total *= i\n    return total\n```")
        after_fp = rep.code_fingerprint("```python\ndef f(n):\n    if n <= 1:\n        return 1\n    return n * f(n - 1)\n```")
        self.assertEqual(rep.classify_novelty("code", before_fp, after_fp), rep.NOVEL_STRUCTURE)

    def test_code_identical_is_same_structure(self):
        src = "```python\ndef f(n):\n    return n * 2\n```"
        fp = rep.code_fingerprint(src)
        self.assertEqual(rep.classify_novelty("code", fp, fp), rep.SAME_STRUCTURE)


# -- context_builder.py: leak/method-hint safety --------------------------------


class ContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.task = ARITH_TASKS["MSARITH_01"]
        steps = self.task["steps"]
        lines = [f"{s['name']} = {s['value']}" for s in steps[:-1]] + [f"{steps[-1]['name']} = 999999"]
        self.artifact = "\n".join(lines)
        import seams
        self.seam = seams.multistep_arithmetic_seam(self.artifact, steps)

    def test_all_modes_validate_cleanly(self):
        for mode in ("R0", "R1", "R2", "R0B"):
            prompt = cb.build_prompt(mode, "arithmetic", self.task, self.artifact, self.seam)
            result = cb.validate_prompt(mode, self.artifact, prompt)
            self.assertTrue(result["ok"], f"{mode}: {result['violations']}")

    def test_r0_has_no_representation_marker(self):
        prompt = cb.build_prompt("R0", "arithmetic", self.task, self.artifact, self.seam)
        self.assertNotIn("NEW REPRESENTATION", prompt)
        self.assertNotIn("DECOMPOSITION DECISION", prompt)

    def test_r1_and_r2_have_distinct_markers(self):
        r1 = cb.build_prompt("R1", "arithmetic", self.task, self.artifact, self.seam)
        r2 = cb.build_prompt("R2", "arithmetic", self.task, self.artifact, self.seam)
        self.assertIn("NEW REPRESENTATION:", r1)
        self.assertNotIn("DECOMPOSITION DECISION", r1)
        self.assertIn("DECOMPOSITION DECISION:", r2)
        self.assertNotIn("NEW REPRESENTATION:", r2)

    def test_forbidden_method_hint_is_caught(self):
        prompt = cb.build_prompt("R1", "arithmetic", self.task, self.artifact, self.seam)
        broken = prompt + "\n\nHint: try to break into steps."
        result = cb.validate_prompt("R1", self.artifact, broken)
        self.assertFalse(result["ok"])

    def test_cross_mode_marker_leak_is_caught(self):
        prompt = cb.build_prompt("R0", "arithmetic", self.task, self.artifact, self.seam)
        broken = prompt + "\n\nNEW REPRESENTATION: some hint"
        result = cb.validate_prompt("R0", self.artifact, broken)
        self.assertFalse(result["ok"])

    def test_substituted_artifact_is_caught(self):
        prompt = cb.build_prompt("R0", "arithmetic", self.task, self.artifact, self.seam)
        broken = prompt.replace(self.artifact, "a fabricated different artifact")
        result = cb.validate_prompt("R0", self.artifact, broken)
        self.assertFalse(result["ok"])

    def test_evidence_core_identical_across_modes(self):
        # общий evidence-контекст (задача+артефакт+evidence) должен быть
        # идентичен между режимами -- различаться должна только инструкция.
        prompts = {m: cb.build_prompt(m, "arithmetic", self.task, self.artifact, self.seam) for m in ("R0", "R1", "R2", "R0B")}
        core = cb._evidence_core("arithmetic", self.task, self.artifact, self.seam)
        for mode, p in prompts.items():
            self.assertTrue(p.startswith(core), f"{mode} prompt does not start with the shared evidence core")


# -- harness.py: run_mode wiring ------------------------------------------------


class RunModeTests(unittest.TestCase):
    def setUp(self):
        self.states = harness.load_exp12_states()
        self.arith_state = next(s for s in self.states if s.task_family == "arithmetic")
        self.task = ARITH_TASKS[self.arith_state.task_id]

    def test_r0_uses_strict_step_format_and_wires_pass(self):
        lines = "\n".join(f"{s['name']} = {s['value']}" for s in self.task["steps"])

        def fake_generate(*a, **kw):
            return {"raw_text": lines, "generation_failed": False, "generation_error": None,
                    "input_tokens": 5, "output_tokens": 5, "generation_time_sec": 0.1}

        with patch.object(harness, "generate", side_effect=fake_generate):
            r = run_mode(self.arith_state, "R0", llm=object())
        self.assertEqual(r["final_status"], "PASS")
        self.assertIsNone(r["representation_novelty"])  # R0 не измеряет новизну

    def test_r1_wires_novelty_and_final_answer_seam(self):
        text = f"NEW REPRESENTATION: alt framing\nSOLUTION:\nfoo = 1\nbar = 2\nFINAL ANSWER = {self.task['answer']}"

        def fake_generate(*a, **kw):
            return {"raw_text": text, "generation_failed": False, "generation_error": None,
                    "input_tokens": 5, "output_tokens": 5, "generation_time_sec": 0.1}

        with patch.object(harness, "generate", side_effect=fake_generate):
            r = run_mode(self.arith_state, "R1", llm=object())
        self.assertEqual(r["final_status"], "PASS")
        self.assertEqual(r["representation_novelty"], "NOVEL_STRUCTURE")
        self.assertEqual(r["record"]["new_representation"], "alt framing")

    def test_same_wrong_value_across_representations_is_same_failure(self):
        wrong = self.arith_state.initial_seam_result["details"]["final"]["extracted"]
        text = f"NEW REPRESENTATION: alt framing\nSOLUTION:\nfoo = 1\nFINAL ANSWER = {wrong}"

        def fake_generate(*a, **kw):
            return {"raw_text": text, "generation_failed": False, "generation_error": None,
                    "input_tokens": 5, "output_tokens": 5, "generation_time_sec": 0.1}

        with patch.object(harness, "generate", side_effect=fake_generate):
            r = run_mode(self.arith_state, "R1", llm=object())
        self.assertEqual(r["final_status"], "SAME_FAILURE")

    def test_generation_failure_is_error_not_fail(self):
        def fake_generate(*a, **kw):
            return {"raw_text": "", "generation_failed": True, "generation_error": "boom",
                    "input_tokens": None, "output_tokens": None, "generation_time_sec": 0.0}

        with patch.object(harness, "generate", side_effect=fake_generate):
            r = run_mode(self.arith_state, "R0", llm=object())
        self.assertEqual(r["final_status"], "ERROR")

    def test_prompt_violation_blocks_generation(self):
        generate_called = []

        def fake_generate(*a, **kw):
            generate_called.append(True)
            return {"raw_text": "x", "generation_failed": False, "generation_error": None,
                    "input_tokens": 1, "output_tokens": 1, "generation_time_sec": 0.0}

        with patch.object(harness, "validate_prompt", return_value={"ok": False, "violations": ["forced"]}), \
             patch.object(harness, "generate", side_effect=fake_generate):
            with self.assertRaises(PromptViolation):
                run_mode(self.arith_state, "R1", llm=object())
        self.assertEqual(generate_called, [])


# -- статистика ------------------------------------------------------------------


class StatisticsTests(unittest.TestCase):
    def test_paired_sign_test(self):
        r = harness.paired_sign_test([True, True, False, False], [False, False, False, False])
        self.assertEqual(r["n_a_wins"], 2)

    def test_bootstrap_rate_diff(self):
        r = harness.bootstrap_rate_diff([True] * 10, [False] * 10, n_resamples=200)
        self.assertAlmostEqual(r["point_diff"], 1.0)

    def test_unpaired_rate_diff_different_group_sizes(self):
        r = harness.unpaired_rate_diff([True, True, False], [False, False, False, False, False], n_resamples=200)
        self.assertEqual(r["n_a"], 3)
        self.assertEqual(r["n_b"], 5)
        self.assertGreater(r["point_diff"], 0)


if __name__ == "__main__":
    unittest.main()
