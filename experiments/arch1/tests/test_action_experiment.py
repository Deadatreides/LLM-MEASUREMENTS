"""Контрольные случаи для metrics/action_experiment.py.

Прямое применение урока этапа 8: оба найденных там дефекта лежали в
коде, который выносил суждение, но не считался проверкой и потому не имел
контрольных случаев. Измерительный аппарат подчиняется той же
доказательной дисциплине, что и измеряемая система, — иначе он выдаёт
уверенные числа, которые потом приходится отзывать.
"""

import random
import unittest

from metrics.action_experiment import (
    ARMS,
    BASE_TEMPERATURE,
    build_configs,
    defective_claims,
    plan_repair,
    solved,
)
from tasks.branch_tasks import TASKS

MODELS = ["model-a", "model-b", "model-c"]
TASK_ID = "BRANCH_01"
TASK = TASKS[TASK_ID]


def _correct_contents(task=TASK):
    return {
        "C1_ROOT": "two parts",
        "C2_METHOD_A": f"{task['qtyA']} and {task['rateA']}",
        "C3_METHOD_B": f"{task['qtyB']} and {task['rateB']}",
        "C4_SUBTOTAL_A": f"{task['qtyA']} * {task['rateA']} = {task['subtotalA']}",
        "C5_SUBTOTAL_B": f"{task['qtyB']} * {task['rateB']} = {task['subtotalB']}",
        "C6_FINAL_TOTAL": f"{task['subtotalA']} + {task['subtotalB']} = {task['final_total']}",
    }


class ConfigSignatureTests(unittest.TestCase):
    """Пункт 4: «запрещается повторять ту же конфигурацию» — значит все
    плечи обязаны быть РАЗЛИЧИМЫ по сигнатуре, иначе запрет пустой."""

    def setUp(self):
        self.configs = build_configs("model-a", MODELS, random.Random(0), seed=1000)

    def test_all_arms_present(self):
        self.assertEqual(set(self.configs), set(ARMS))

    def test_all_arm_signatures_are_distinct(self):
        signatures = [c.signature() for c in self.configs.values()]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_change_model_actually_changes_model(self):
        self.assertNotEqual(self.configs["CHANGE_MODEL"].model_id, "model-a")
        self.assertIn(self.configs["CHANGE_MODEL"].model_id, MODELS)

    def test_temperature_arms_differ_in_direction(self):
        low = self.configs["CHANGE_TEMPERATURE_LOW"].temperature
        high = self.configs["CHANGE_TEMPERATURE_HIGH"].temperature
        self.assertLess(low, BASE_TEMPERATURE)
        self.assertGreater(high, BASE_TEMPERATURE)

    def test_change_prompt_uses_other_profile(self):
        self.assertEqual(self.configs["CHANGE_PROMPT"].prompt_profile, "v2")
        self.assertEqual(self.configs["REGENERATE_SAME"].prompt_profile, "v1")

    def test_control_arm_differs_from_generation_seed(self):
        """Контроль обязан отличаться НОВЫМ seed: с тем же seed и тем же
        промптом повтор детерминированно обречён (89.8% побайтово
        идентичных повторов в кампании 3)."""
        self.assertNotIn(self.configs["REGENERATE_SAME"].seed, (1, 2, 3, 4))

    def test_alternative_model_choice_is_reproducible(self):
        again = build_configs("model-a", MODELS, random.Random(0), seed=1000)
        self.assertEqual(
            self.configs["CHANGE_MODEL"].model_id, again["CHANGE_MODEL"].model_id
        )


class SuccessCriterionTests(unittest.TestCase):
    """Критерий успеха обязан быть ОДИН для всех стратегий, иначе
    сравнение ремонта с best-of-N бессмысленно."""

    def test_all_correct_is_solved(self):
        self.assertTrue(solved(_correct_contents(), TASK))

    def test_one_wrong_value_is_not_solved(self):
        contents = _correct_contents()
        contents["C4_SUBTOTAL_A"] = f"{TASK['qtyA']} * {TASK['rateA']} = 999"
        self.assertFalse(solved(contents, TASK))

    def test_unparseable_claim_is_not_solved(self):
        """Не удалось извлечь значение -> INAPPLICABLE -> НЕ успех.
        Иначе нечитаемый ответ засчитывался бы как решение."""
        contents = _correct_contents()
        contents["C6_FINAL_TOTAL"] = "I cannot compute this."
        self.assertFalse(solved(contents, TASK))

    def test_defective_set_matches_failing_claims(self):
        contents = _correct_contents()
        contents["C5_SUBTOTAL_B"] = f"{TASK['qtyB']} * {TASK['rateB']} = 7"
        self.assertEqual(defective_claims(contents, TASK), {"C5_SUBTOTAL_B"})

    def test_inapplicable_is_not_counted_as_defective(self):
        """INAPPLICABLE != FAIL (SEAM_MODEL §4): непроверяемое не является
        опровергнутым и не может порождать origin."""
        contents = _correct_contents()
        contents["C5_SUBTOTAL_B"] = "no numbers here"
        self.assertEqual(defective_claims(contents, TASK), set())


class PlanRepairTests(unittest.TestCase):
    def test_plan_targets_the_defective_claim(self):
        contents = _correct_contents()
        contents["C4_SUBTOTAL_A"] = f"{TASK['qtyA']} * {TASK['rateA']} = 999"
        to_regenerate = plan_repair(contents, TASK_ID, ["C4_SUBTOTAL_A"])
        self.assertIn("C4_SUBTOTAL_A", to_regenerate)

    def test_correct_downstream_is_not_regenerated(self):
        """AMD-1: C6 верен сам по себе -> MAY_REUSE, а не MUST_REGENERATE.
        Это ровно та экономия, которую измеряет A-1."""
        contents = _correct_contents()
        contents["C4_SUBTOTAL_A"] = f"{TASK['qtyA']} * {TASK['rateA']} = 999"
        to_regenerate = plan_repair(contents, TASK_ID, ["C4_SUBTOTAL_A"])
        self.assertNotIn("C6_FINAL_TOTAL", to_regenerate)


if __name__ == "__main__":
    unittest.main()
