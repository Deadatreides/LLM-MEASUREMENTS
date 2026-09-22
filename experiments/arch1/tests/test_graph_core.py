"""Контрольные случаи для src/graph_core.py и src/storage.py.

Минимум (WORK_PLAN.md, Этап 1):
1. downstream_closure на ветвящемся DAG из 6 узлов — точный ответ известен
2. независимая ветвь НЕ попадает в замыкание
3. root_origins различает корень и симптом (DEPENDENCY_MODEL.md §6.2)
4. добавление ребра, замыкающего цикл -> CYCLE_DETECTED, граф не изменён
5. collapse_scc сохраняет provenance всех слагаемых
6. storage не имеет update/delete (проверка интерфейса)

Плюс сверх минимума: обработка CONFIRMED-цикла (§5.2b), topological_order,
защита storage от дублирующего record_id.

Фикстура — ветвящийся DAG вручную (6 узлов + независимая ветвь), не из
отчётов экспериментов (arch1/WORK_PLAN.md запрещает читать
EXPERIMENTAL_BASIS.md на этом этапе):

    A -> B -> D -\
    A -> C -> E -> F      (F -- merge-узел)
    G -> H                (независимая ветвь)
"""

import unittest

from src.graph_core import (
    CYCLE_DETECTED,
    STATUS_CONFIRMED,
    STATUS_HYPOTHESIS,
    GraphCore,
)
from src.storage import Storage


class GraphCoreFixtureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = Storage()
        self.graph = GraphCore(self.storage)

        artifact_version_id = self.graph.add_artifact_version(
            artifact_type="DERIVATION",
            task_id="task:fixture",
            content="fixture artifact",
        )
        self.artifact_id = self.storage.get(artifact_version_id).payload.artifact_id

        for name in "ABCDEFGH":
            self.graph.add_claim_version(
                artifact_id=self.artifact_id,
                content=f"claim {name}",
                claim_type="OTHER",
                claim_id=self.c(name),
            )

        edges = [
            ("A", "B"), ("A", "C"),
            ("B", "D"), ("C", "E"),
            ("D", "F"), ("E", "F"),
            ("G", "H"),
        ]
        for source, target in edges:
            result = self.graph.add_dependency(
                source_claim=self.c(source),
                target_claim=self.c(target),
                dependency_type="DERIVATION",
                status=STATUS_CONFIRMED,
                created_by="HUMAN",
            )
            self.assertNotEqual(result, CYCLE_DETECTED, f"fixture edge {source}->{target}")

    @staticmethod
    def c(name: str) -> str:
        return f"claim:{name}"


class DownstreamClosureTests(GraphCoreFixtureTestCase):
    def test_downstream_closure_branching_dag(self) -> None:
        result = self.graph.downstream_closure(self.c("A"))
        self.assertEqual(result, {self.c(x) for x in "BCDEF"})

    def test_independent_branch_not_in_closure(self) -> None:
        result = self.graph.downstream_closure(self.c("A"))
        self.assertNotIn(self.c("G"), result)
        self.assertNotIn(self.c("H"), result)

        result_g = self.graph.downstream_closure(self.c("G"))
        self.assertEqual(result_g, {self.c("H")})


class RootOriginsTests(GraphCoreFixtureTestCase):
    def test_distinguishes_root_from_cascaded_symptom(self) -> None:
        # A -> B -> D -> F: если дефектны все четыре, корень -- только A.
        # Наивная реализация ("свой чек провалился => сам дефект") вернула
        # бы все четыре -- это и есть ошибка F31, которую тест ловит.
        cascade_defective = {self.c(x) for x in "ABDF"}
        roots = self.graph.root_origins(cascade_defective)
        self.assertEqual(roots, [self.c("A")])

    def test_multiple_independent_roots(self) -> None:
        independent_defective = {self.c("A"), self.c("G")}
        roots = self.graph.root_origins(independent_defective)
        self.assertEqual(set(roots), independent_defective)


class CycleDetectionTests(GraphCoreFixtureTestCase):
    def test_hypothesis_edge_closing_cycle_is_rejected_graph_unchanged(self) -> None:
        self.assertEqual(self.graph.downstream_closure(self.c("F")), set())
        self.assertEqual(self.graph.upstream_closure(self.c("A")), set())

        result = self.graph.add_dependency(
            source_claim=self.c("F"),
            target_claim=self.c("A"),
            dependency_type="LOGICAL",
            status=STATUS_HYPOTHESIS,
            created_by="HUMAN",
        )

        self.assertEqual(result, CYCLE_DETECTED)
        self.assertEqual(self.graph.downstream_closure(self.c("F")), set())
        self.assertEqual(self.graph.upstream_closure(self.c("A")), set())

    def test_confirmed_edge_closing_cycle_is_rejected_and_observed(self) -> None:
        before_seq = self.storage.current_seq()

        result = self.graph.add_dependency(
            source_claim=self.c("F"),
            target_claim=self.c("A"),
            dependency_type="LOGICAL",
            status=STATUS_CONFIRMED,
            created_by="HUMAN",
        )

        self.assertEqual(result, CYCLE_DETECTED)
        self.assertEqual(self.graph.downstream_closure(self.c("F")), set())

        observations = self.storage.query(lambda r: r.kind == "cycle_observation")
        self.assertEqual(len(observations), 1)
        observation = observations[0].payload
        self.assertEqual(observation.source_claim, self.c("F"))
        self.assertEqual(observation.target_claim, self.c("A"))
        self.assertEqual(observation.dependency_status, STATUS_CONFIRMED)

        # CONFIRMED-цикл не схлопывается автоматически (решение #7 плана) --
        # только CYCLE_OBSERVATION добавлена, никакого dependency-объекта.
        self.assertEqual(self.storage.current_seq(), before_seq + 1)


class CollapseSccTests(GraphCoreFixtureTestCase):
    def test_collapse_scc_preserves_provenance_and_reroutes_external_edges(self) -> None:
        members = {self.c("B"), self.c("D")}
        composite_id = self.graph.collapse_scc(members)

        composite = self.graph.get_claim(composite_id)
        self.assertEqual(composite.atomicity, "INDIVISIBLE_BLOCK")
        self.assertEqual(set(composite.provenance.input_refs), members)

        for member in members:
            original = self.graph.get_claim(member)  # не удалён, читается как был
            self.assertIn(member.split(":")[1], original.content)

        self.assertIn(composite_id, self.graph.direct_children(self.c("A")))
        self.assertIn(composite_id, self.graph.direct_parents(self.c("F")))


class TopologicalOrderTests(GraphCoreFixtureTestCase):
    def test_topological_order_respects_all_edges(self) -> None:
        order = self.graph.topological_order({self.c(x) for x in "ABCDEF"})
        position = {node: i for i, node in enumerate(order)}
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "F")]
        for source, target in edges:
            self.assertLess(position[self.c(source)], position[self.c(target)])


class StorageInterfaceTests(unittest.TestCase):
    def test_storage_has_no_update_or_delete(self) -> None:
        for forbidden in ("update", "delete", "set", "remove"):
            self.assertFalse(
                hasattr(Storage, forbidden), f"Storage must not define {forbidden!r}"
            )

    def test_storage_append_rejects_duplicate_record_id(self) -> None:
        storage = Storage()
        storage.append("rec:1", "kind", {"a": 1})
        with self.assertRaises(ValueError):
            storage.append("rec:1", "kind", {"a": 2})


if __name__ == "__main__":
    unittest.main()
