"""Контрольные случаи для src/repair_planner.py и src/collision.py.

На инъекционном стенде (tests/injection_harness.py), обязательный
минимум (WORK_PLAN.md, Этап 4):
1. инъекция в терминальный узел -> область = {он сам}
2. инъекция в корень -> область = весь граф
3. инъекция в ветвь A -> ветвь B НЕ затронута
4. merge-узел затрагивается только со стороны повреждённой ветви
5. narrow() переводит узел в unaffected ТОЛЬКО по evidence
6. план всегда содержит FULL_RETRY в alternatives_considered
7. повтор плана -> REPAIR_OSCILLATION -> STOP
8. хеш состояния повторился -> REPEATED_STATE -> STOP

Плюс сверх минимума: сверка предсказания с независимым ground truth на
каждом инъекционном случае, запрет «все предки чисты => потомок чист»
(решение №1 плана), недоверенный шов не сужает область (решение №2),
post_repair_check материализует скрытую зависимость как ребро графа.
"""

import re
import unittest

from src.collision import Collision, localize
from src.repair_planner import (
    DO_NOT_TOUCH,
    FULL_RETRY,
    MAY_REUSE,
    MUST_REGENERATE,
    MUST_VERIFY,
    REPAIR_OSCILLATION,
    REPEATED_STATE,
    RepairPlanner,
)
from src.seam_engine import ControlCase, SeamDefinition, SeamEngine, default_seam_engine
from src.state_engine import STRENGTH_HARD, SUPPORTS, Evidence, StateEngine
from src.storage import new_id
from tests.injection_harness import CLAIMS, TASK, build_correct_graph, inject, true_affected_set

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _seam_provider(graph, task=TASK):
    """Абсолютная проверка (решение Р1 плана): actual берётся из ТЕКУЩЕГО
    содержимого claim'а, expected -- из механического oracle задачи, не
    из состояния предка."""
    expected = {
        "C4_SUBTOTAL_A": task["subtotalA"],
        "C5_SUBTOTAL_B": task["subtotalB"],
        "C6_FINAL_TOTAL": task["final_total"],
    }

    def provider(claim_id, seam):
        if seam.seam_id != "seam:numeric" or claim_id not in expected:
            return None
        numbers = _NUMBER_RE.findall(graph.get_claim(claim_id).content)
        if not numbers:
            return None
        return {"actual": float(numbers[-1]), "expected": float(expected[claim_id])}

    return provider


def _setup(injected_claim_id):
    storage, graph = build_correct_graph()
    inject(graph, injected_claim_id)

    seams = default_seam_engine(storage)
    for seam_id in ("seam:exec", "seam:format", "seam:numeric"):
        result = seams.self_test(seam_id)
        assert result["passed"], (seam_id, result["failed_cases"])

    state = StateEngine(storage, graph)
    planner = RepairPlanner(graph, state, seams)

    collision = Collision(
        collision_id=new_id("col"),
        collision_type="MECHANICAL_FAILURE",
        participants=(injected_claim_id,),
    )
    localize(collision, graph, defective_claims={injected_claim_id})

    return storage, graph, seams, state, planner, collision


class CandidateAffectedRegionTests(unittest.TestCase):
    """Случаи 1-4: §2 candidate_affected_set, сверенное с независимым
    ground truth (не только с ожиданиями теста)."""

    def test_injection_in_terminal_node_region_is_itself(self):
        _, _, _, _, planner, collision = _setup("C6_FINAL_TOTAL")
        region = planner.affected_region(collision)
        self.assertEqual(region.candidate_affected_set, {"C6_FINAL_TOTAL"})
        self.assertEqual(region.candidate_affected_set, true_affected_set({"C6_FINAL_TOTAL"}))

    def test_injection_in_root_region_is_whole_graph(self):
        _, _, _, _, planner, collision = _setup("C1_ROOT")
        region = planner.affected_region(collision)
        self.assertEqual(region.candidate_affected_set, set(CLAIMS))
        self.assertEqual(region.candidate_affected_set, true_affected_set({"C1_ROOT"}))

    def test_injection_in_branch_a_does_not_affect_branch_b(self):
        _, _, _, _, planner, collision = _setup("C2_METHOD_A")
        region = planner.affected_region(collision)
        self.assertEqual(region.candidate_affected_set, true_affected_set({"C2_METHOD_A"}))
        self.assertNotIn("C3_METHOD_B", region.candidate_affected_set)
        self.assertNotIn("C5_SUBTOTAL_B", region.candidate_affected_set)

    def test_merge_node_affected_only_from_corrupted_branch_side(self):
        _, _, _, _, planner, collision = _setup("C2_METHOD_A")
        region = planner.affected_region(collision)
        self.assertIn("C6_FINAL_TOTAL", region.candidate_affected_set)
        self.assertEqual(region.candidate_affected_set, true_affected_set({"C2_METHOD_A"}))


class NarrowGuaranteesTests(unittest.TestCase):
    """Случай 5 + решения №1-2 плана."""

    def test_narrow_marks_unchanged_descendant_unaffected_by_evidence(self):
        _, graph, _, _, planner, collision = _setup("C2_METHOD_A")
        region = planner.affected_region(collision)
        narrowed = planner.narrow(region, _seam_provider(graph))

        # C4 -- ребёнок C2 (DERIVATION), но собственное содержимое C4 не
        # менялось и арифметически верно само по себе -- решение Р1:
        # абсолютная проверка находит его unaffected, а не verified.
        self.assertIn("C4_SUBTOTAL_A", narrowed.unaffected_confirmed_set)
        self.assertNotIn("C4_SUBTOTAL_A", narrowed.verified_affected_set)

    def test_narrow_never_infers_unaffected_from_clean_ancestors_alone(self):
        _, graph, _, _, planner, collision = _setup("C2_METHOD_A")
        region = planner.affected_region(collision)

        base_provider = _seam_provider(graph)

        def limited_provider(claim_id, seam):
            if claim_id == "C6_FINAL_TOTAL":
                return None  # намеренно: evidence для C6 недоступно
            return base_provider(claim_id, seam)

        narrowed = planner.narrow(region, limited_provider)

        self.assertIn("C4_SUBTOTAL_A", narrowed.unaffected_confirmed_set)
        # C6 -- потомок C2 (verified) и C4 (unaffected), но БЕЗ СОБСТВЕННОГО
        # evidence обязан остаться unknown -- API_BOUNDARIES §8: narrow
        # никогда не переводит узел в «не затронуто» без evidence (F38).
        self.assertIn("C6_FINAL_TOTAL", narrowed.unknown_set)
        self.assertNotIn("C6_FINAL_TOTAL", narrowed.unaffected_confirmed_set)
        self.assertNotIn("C6_FINAL_TOTAL", narrowed.verified_affected_set)

    def test_narrow_ignores_untrusted_seam_even_on_pass(self):
        storage, graph, _, state, _, collision = _setup("C4_SUBTOTAL_A")

        def always_pass(inputs):
            return {"status": "PASS", "details": {}}

        untrusted = SeamEngine(storage)
        untrusted.register(
            SeamDefinition(
                seam_id="seam:untrusted",
                seam_type="SEAM_EXECUTION",
                check=always_pass,
                control_cases=(ControlCase(name="c1", inputs={}, expected_status="FAIL"),),
                applicable_claim_types=frozenset({"VALUE"}),
            )
        )
        # намеренно НЕ вызываем self_test -- шов остаётся непроверенным

        planner = RepairPlanner(graph, state, untrusted)
        region = planner.affected_region(collision)

        def provider(claim_id, seam):
            return {} if seam.seam_id == "seam:untrusted" else None

        narrowed = planner.narrow(region, provider)

        self.assertNotIn("C6_FINAL_TOTAL", narrowed.unaffected_confirmed_set)
        self.assertIn("C6_FINAL_TOTAL", narrowed.unknown_set)


class BuildPlanTests(unittest.TestCase):
    def test_plan_always_includes_full_retry_alternative(self):
        _, graph, _, _, planner, collision = _setup("C6_FINAL_TOTAL")
        region = planner.affected_region(collision)
        narrowed = planner.narrow(region, _seam_provider(graph))
        plan = planner.build_plan(narrowed, collision, task={"risk_level": "MEDIUM"})

        strategies = {alt["strategy"] for alt in plan.alternatives_considered}
        self.assertIn(FULL_RETRY, strategies)

    def test_necessity_assignment_matches_region(self):
        _, graph, _, _, planner, collision = _setup("C1_ROOT")
        region = planner.affected_region(collision)
        narrowed = planner.narrow(region, _seam_provider(graph))
        plan = planner.build_plan(narrowed, collision, task={"risk_level": "MEDIUM"})

        by_claim = {e.claim_id: e for e in plan.entries}
        self.assertEqual(by_claim["C1_ROOT"].necessity, MUST_REGENERATE)  # origin
        self.assertEqual(by_claim["C4_SUBTOTAL_A"].necessity, MAY_REUSE)  # unaffected
        self.assertEqual(by_claim["C2_METHOD_A"].necessity, MUST_VERIFY)  # unknown, риск MEDIUM

    def test_claims_outside_candidate_set_are_do_not_touch(self):
        # build_plan (REPAIR_MODEL §4.3) строит entries только по
        # region.candidate_affected_set -- claims вне региона в план не
        # попадают вовсе. assign_necessity, вызванный напрямую на таком
        # claim'е, обязан вернуть DO_NOT_TOUCH (F35: независимые ветви).
        _, graph, _, _, planner, collision = _setup("C2_METHOD_A")
        region = planner.affected_region(collision)
        narrowed = planner.narrow(region, _seam_provider(graph))

        self.assertNotIn("C3_METHOD_B", narrowed.candidate_affected_set)
        self.assertEqual(planner.assign_necessity("C3_METHOD_B", narrowed, "MEDIUM"), DO_NOT_TOUCH)
        self.assertEqual(planner.assign_necessity("C5_SUBTOTAL_B", narrowed, "MEDIUM"), DO_NOT_TOUCH)

        plan = planner.build_plan(narrowed, collision, task={"risk_level": "MEDIUM"})
        plan_claim_ids = {e.claim_id for e in plan.entries}
        self.assertNotIn("C3_METHOD_B", plan_claim_ids)
        self.assertNotIn("C5_SUBTOTAL_B", plan_claim_ids)


class StopConditionTests(unittest.TestCase):
    def test_repeated_plan_two_iterations_back_triggers_oscillation(self):
        _, graph, _, _, planner, collision = _setup("C1_ROOT")
        region = planner.affected_region(collision)
        narrowed = planner.narrow(region, _seam_provider(graph))
        self.assertTrue(narrowed.unknown_set)  # сценарий действительно даёт unknown-узлы

        plan_a = planner.build_plan(narrowed, collision, task={"risk_level": "MEDIUM"})
        planner.record_iteration(collision, plan_a)

        plan_b = planner.build_plan(narrowed, collision, task={"risk_level": "HIGH"})
        self.assertNotEqual(plan_a.plan_hash, plan_b.plan_hash)
        planner.record_iteration(collision, plan_b)

        plan_c = planner.build_plan(narrowed, collision, task={"risk_level": "MEDIUM"})
        self.assertEqual(plan_a.plan_hash, plan_c.plan_hash)
        self.assertEqual(planner.check_stop_conditions(plan_c, collision), REPAIR_OSCILLATION)

    def test_same_graph_and_plan_seen_before_triggers_repeated_state(self):
        _, graph, _, _, planner, collision = _setup("C6_FINAL_TOTAL")
        region = planner.affected_region(collision)
        narrowed = planner.narrow(region, _seam_provider(graph))

        plan_a = planner.build_plan(narrowed, collision, task={"risk_level": "MEDIUM"})
        planner.record_iteration(collision, plan_a)

        plan_b = planner.build_plan(narrowed, collision, task={"risk_level": "MEDIUM"})
        self.assertEqual(plan_a.plan_hash, plan_b.plan_hash)
        self.assertEqual(planner.check_stop_conditions(plan_b, collision), REPEATED_STATE)


class PostRepairCheckTests(unittest.TestCase):
    def test_detects_hidden_dependency_and_materializes_edge(self):
        storage, graph, _, state, planner, _ = _setup("C2_METHOD_A")
        repair_set = {"C2_METHOD_A", "C4_SUBTOTAL_A"}

        # C6 раньше был CORRECT (прошёл проверку до ремонта)
        state.apply_evidence(
            "C6_FINAL_TOTAL",
            Evidence(
                evidence_id=new_id("ev"),
                subject="C6_FINAL_TOTAL",
                assertion=SUPPORTS,
                evidence_type="MECHANICAL",
                strength=STRENGTH_HARD,
                source={"kind": "test"},
                method="test",
            ),
        )

        # "ремонт" C4 сменил число -- C6 (не тронут) больше не согласован
        # с НОВЫМ значением C4. Абсолютная проверка C6 против oracle here
        # не годится (её собственный текст не менялся, "145" по-прежнему
        # формально верно) -- нужна проверка согласованности C6 с текущими
        # C4/C5, именно её и обязан обнаружить post_repair_check.
        graph.add_claim_version(
            artifact_id=graph.get_claim("C4_SUBTOTAL_A").artifact_id,
            content="2 * 20 = 999",
            claim_type="VALUE",
            claim_id="C4_SUBTOTAL_A",
        )

        def consistency_provider(claim_id, seam):
            if seam.seam_id != "seam:numeric" or claim_id != "C6_FINAL_TOTAL":
                return None
            c4_numbers = _NUMBER_RE.findall(graph.get_claim("C4_SUBTOTAL_A").content)
            c5_numbers = _NUMBER_RE.findall(graph.get_claim("C5_SUBTOTAL_B").content)
            expected = float(c4_numbers[-1]) + float(c5_numbers[-1])
            actual_numbers = _NUMBER_RE.findall(graph.get_claim("C6_FINAL_TOTAL").content)
            return {"actual": float(actual_numbers[-1]), "expected": expected}

        hidden = planner.post_repair_check(repair_set, consistency_provider)

        self.assertEqual(len(hidden), 1)
        source, target = hidden[0]
        self.assertEqual(target, "C6_FINAL_TOTAL")
        self.assertIn(source, repair_set)

        deps = [
            r.payload
            for r in storage.query(lambda rec: rec.kind == "dependency")
            if r.payload.target_claim == "C6_FINAL_TOTAL"
            and r.payload.dependency_type == "UNKNOWN_DEPENDENCY"
        ]
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0].status, "HYPOTHESIS")
        self.assertEqual(deps[0].created_by, "SEAM_OBSERVATION")

    def test_no_hidden_dependency_when_everything_still_consistent(self):
        storage, graph, _, state, planner, _ = _setup("C6_FINAL_TOTAL")
        repair_set = {"C6_FINAL_TOTAL"}

        hidden = planner.post_repair_check(repair_set, _seam_provider(graph))
        self.assertEqual(hidden, [])


class AmendmentAMD1Tests(unittest.TestCase):
    """Поправка AMD-1 (REPAIR_MODEL.md §3.2): проверка имеет приоритет над
    структурной эвристикой; эвристика — умолчание для непроверяемого.

    Случаи написаны по ФОРМЕ дефекта, найденного измерением ARCH-1: на
    240 прогонах эвристика перехватывала 87.2% не-origin узлов, и в части
    случаев осуждала ВЕРНЫЙ узел, потому что ошибочное значение предка
    случайно совпадало с верным значением потомка."""

    def _graph_with_value_coincidence(self):
        """origin C4 неверен и содержит 145; C6 ВЕРЕН и тоже содержит 145
        (145 -- правильный итог задачи). Значение предка буквально
        присутствует у потомка, но потомок не затронут."""
        storage, graph = build_correct_graph()
        artifact_id = graph.get_claim("C4_SUBTOTAL_A").artifact_id
        graph.add_claim_version(
            artifact_id=artifact_id,
            content="$145",  # неверно: ожидается 40; совпадает с верным итогом C6
            claim_type="VALUE",
            claim_id="C4_SUBTOTAL_A",
        )
        seams = default_seam_engine(storage)
        for seam_id in ("seam:exec", "seam:format", "seam:numeric"):
            seams.self_test(seam_id)
        planner = RepairPlanner(graph, StateEngine(storage, graph), seams)
        collision = Collision(
            collision_id=new_id("col"),
            collision_type="MECHANICAL_FAILURE",
            participants=("C4_SUBTOTAL_A",),
        )
        localize(collision, graph, defective_claims={"C4_SUBTOTAL_A"})
        return graph, planner, collision

    def test_verification_wins_over_structural_heuristic(self):
        graph, planner, collision = self._graph_with_value_coincidence()
        region = planner.affected_region(collision)

        # предусловие случая: эвристика ЗДЕСЬ срабатывает (значение предка
        # присутствует у потомка) -- то есть до AMD-1 C6 был бы осуждён
        self.assertTrue(
            planner._parent_value_literally_present("C6_FINAL_TOTAL", {"C4_SUBTOTAL_A"})
        )

        narrowed = planner.narrow(region, _seam_provider(graph))

        # C6 арифметически верен -> проверка обязана взять верх
        self.assertIn("C6_FINAL_TOTAL", narrowed.unaffected_confirmed_set)
        self.assertNotIn("C6_FINAL_TOTAL", narrowed.verified_affected_set)

    def test_structural_heuristic_still_applies_when_unverifiable(self):
        """Консервативность сохранена: без доступной проверки узел, у
        которого присутствует значение затронутого предка, по-прежнему
        уходит в verified, а НЕ в unknown."""
        graph, planner, collision = self._graph_with_value_coincidence()
        region = planner.affected_region(collision)

        narrowed = planner.narrow(region, lambda claim_id, seam: None)  # проверок нет

        self.assertIn("C6_FINAL_TOTAL", narrowed.verified_affected_set)
        self.assertNotIn("C6_FINAL_TOTAL", narrowed.unknown_set)


if __name__ == "__main__":
    unittest.main()
