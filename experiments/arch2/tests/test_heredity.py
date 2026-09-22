"""Контрольные случаи наследственности и автокатализа (SPEC.md §13-19).

Каждый случай имеет заранее известный ответ по контракту спеки. Отдельно проверяется
регресс-тест на найденный латентный дефект: до §13 композиты попадали в
`generator_ids()`, но вставлялись с параметрами, а сетка у них пуста — такие мутанты
молча отбраковывались валидатором, и вставка композита не работала вообще.
"""

from __future__ import annotations

import random
import unittest

import genotype as G
import heredity as H
import library
import mutate as M
import registry as REG
import replay
import runner as R


def _mk(root=None, gen=0, origin="test"):
    return G.genotype(root or library.attempt("K0"), gen=gen, origin=origin)


def _res(mapping):
    """{task_id: bool} -> формат трасс, который читает fitness/heredity."""
    return {t: {"outcome": "RESOLVED" if ok else "UNRESOLVED", "cost": 100, "n_calls": 1}
            for t, ok in mapping.items()}


class EdgeLogTests(unittest.TestCase):
    def test_edge_types_are_closed(self):
        log = H.EdgeLog()
        with self.assertRaises(ValueError):
            log.add("a", "b", "teleported_from", 0)

    def test_append_only_and_counts(self):
        log = H.EdgeLog()
        log.add("b", "a", "mutated_from", 1)
        log.add("c", "a", "mutated_from", 1)
        log.add("c", "b", "crossed_from", 1)
        self.assertEqual(len(log), 3)
        self.assertEqual(log.counts(), {"mutated_from": 2, "crossed_from": 1})

    def test_ancestors_and_descendants_are_transitive(self):
        log = H.EdgeLog()
        log.add("b", "a", "mutated_from", 1)
        log.add("c", "b", "mutated_from", 2)
        self.assertEqual(log.ancestors("c"), {"b", "a"})
        self.assertEqual(log.descendants("a"), {"b", "c"})

    def test_similarity_edges_do_not_count_as_lineage(self):
        """reinforces/echo — НЕ родство: иначе «похожи» превращалось бы в «родня»."""
        log = H.EdgeLog()
        log.add("x", "y", "reinforces", 3)
        self.assertEqual(log.ancestors("x"), set())
        self.assertFalse(log.same_lineage("x", "y"))

    def test_same_lineage_via_common_ancestor(self):
        log = H.EdgeLog()
        log.add("b", "a", "mutated_from", 1)
        log.add("c", "a", "mutated_from", 1)
        self.assertTrue(log.same_lineage("b", "c"))

    def test_roundtrip_jsonl(self):
        import tempfile
        from pathlib import Path

        log = H.EdgeLog()
        log.add("b", "a", "mutated_from", 1, {"operator": "M1"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heredity.jsonl"
            log.to_jsonl(path)
            back = H.EdgeLog.from_jsonl(path)
        self.assertEqual(back.all(), log.all())


class PlacementTests(unittest.TestCase):
    def test_sem_bounds_and_extremes(self):
        a = _res({"t1": True, "t2": True, "t3": False})
        b = _res({"t1": True, "t2": True, "t3": False})
        c = _res({"t1": False, "t2": False, "t3": True})
        self.assertEqual(H.sem_similarity(a, b), 1.0)
        self.assertEqual(H.sem_similarity(a, c), 0.0)
        self.assertTrue(0.0 <= H.sem_similarity(a, _res({"t1": True, "t2": False, "t3": False})) <= 1.0)

    def test_sem_uses_intersection_only(self):
        """Урок E3: считать по объединению — значит давать очки за невиданные задачи."""
        a = _res({"t1": True, "t2": True})
        b = _res({"t1": True})                       # t2 конкурент не видел
        self.assertEqual(H.sem_similarity(a, b), 1.0)

    def test_sem_of_disjoint_evaluations_is_zero(self):
        self.assertEqual(H.sem_similarity(_res({"t1": True}), _res({"t9": True})), 0.0)

    def test_temporal_affinity_halves_at_half_life(self):
        self.assertAlmostEqual(H.temporal_affinity(0, 0), 1.0)
        self.assertAlmostEqual(H.temporal_affinity(0, int(H.TEMP_HALF_LIFE_GENS)), 0.5, places=6)
        self.assertLess(H.temporal_affinity(0, 9), H.temporal_affinity(0, 3))

    def test_density_falls_when_neighbours_are_close(self):
        res = {"a": _res({"t1": True}), "b": _res({"t1": True}), "c": _res({"t1": False})}
        dense = H.density_term("a", res, ["b"], ["t1"])
        sparse = H.density_term("a", res, ["c"], ["t1"])
        self.assertEqual(dense, 0.0)
        self.assertEqual(sparse, 1.0)

    def test_skel_identical_is_one(self):
        g = _mk()
        s = H.skeleton(g)
        self.assertAlmostEqual(H.skel_similarity(s, s), 1.0, places=6)

    def test_skel_differs_for_different_shapes(self):
        flat = H.skeleton(_mk(library.attempt("K0")))
        wide = H.skeleton(_mk(G.PAR(library.attempt("K0"), library.blind(1), library.blind(2))))
        self.assertLess(H.skel_similarity(flat, wide), 1.0)

    def test_place_is_bounded(self):
        self.assertAlmostEqual(H.place(1, 1, 1, 1), 1.0, places=6)
        self.assertAlmostEqual(H.place(0, 0, 0, 0), 0.0, places=6)


class RelationCaseTests(unittest.TestCase):
    """Решение принимают ЯВНЫЕ CASE-условия, не argmax скаляра (SPEC.md §15)."""

    def test_low_similarity_gives_no_edge(self):
        self.assertIsNone(H.classify_relation(0.5, 5, False))
        self.assertIsNone(H.classify_relation(0.5, 5, True))

    def test_different_lineage_gives_reinforces(self):
        self.assertEqual(H.classify_relation(0.95, 0, False), "reinforces")
        self.assertEqual(H.classify_relation(0.95, 9, False), "reinforces")

    def test_same_lineage_needs_generation_gap_for_echo(self):
        self.assertIsNone(H.classify_relation(0.95, H.ECHO_MIN_GEN - 1, True))
        self.assertEqual(H.classify_relation(0.95, H.ECHO_MIN_GEN, True), "echo")

    def test_donor_score_prefers_complementary_phenotype(self):
        same_skel_same_pheno = H.donor_score(skel=1.0, sem=1.0)
        same_skel_diff_pheno = H.donor_score(skel=1.0, sem=0.0)
        self.assertGreater(same_skel_diff_pheno, same_skel_same_pheno)


class DecayTests(unittest.TestCase):
    def test_weight_decays_with_idle_generations(self):
        self.assertGreater(H.weight(0, 0), H.weight(3, 0))
        self.assertGreater(H.weight(3, 0), H.weight(9, 0))

    def test_uses_raise_weight(self):
        self.assertGreater(H.weight(5, 3), H.weight(5, 0))

    def test_shadow_threshold(self):
        self.assertFalse(H.is_shadow(H.W_FLOOR + 1e-9))
        self.assertTrue(H.is_shadow(H.W_FLOOR - 1e-9))

    def test_long_idle_without_uses_reaches_shadow(self):
        idle = 1
        while not H.is_shadow(H.weight(idle, 0)) and idle < 100:
            idle += 1
        self.assertLess(idle, 100, "затухание не доводит до shadow ни при каком простое")


class NoMergeTests(unittest.TestCase):
    """SPEC.md §14.1: два complex_id никогда не сливаются в один."""

    def test_identical_phenotype_keeps_two_ids(self):
        a = _mk(library.attempt("K0"))
        b = _mk(library.attempt("K1"))
        self.assertNotEqual(a["complex_id"], b["complex_id"])
        res = {a["complex_id"]: _res({"t1": True}), b["complex_id"]: _res({"t1": True})}
        self.assertEqual(H.sem_similarity(res[a["complex_id"]], res[b["complex_id"]]), 1.0)
        # одинаковый фенотип -> максимум ребро, но никак не общий идентификатор
        log = H.EdgeLog()
        rel = H.classify_relation(1.0, 0, log.same_lineage(a["complex_id"], b["complex_id"]))
        self.assertEqual(rel, "reinforces")
        self.assertNotEqual(a["complex_id"], b["complex_id"])

    def test_structurally_identical_is_the_same_id_not_a_merge(self):
        self.assertEqual(_mk(library.attempt("K0"))["complex_id"],
                         _mk(library.attempt("K0"), gen=7)["complex_id"])


class CompositeInsertionTests(unittest.TestCase):
    """Регресс-тест найденного латентного дефекта (SPEC.md §13.3).

    До §13: `generator_ids()` уже возвращал композиты, а M1/`random_genotype` строили
    для них CALL с параметрами при пустой сетке — валидатор отбрасывал такие мутанты
    молча, и ни один композит в новые генотипы не попадал.
    """

    def setUp(self):
        self.ds = replay.default_dataset()
        self.reg = REG.Registry()
        self.be = replay.ReplayBackend(self.ds)
        inner = _mk(library.attempt("K0"))
        self.mid = self.reg.register_composite(inner["complex_id"], inner,
                                               {"median_tokens": 240, "median_calls": 1})
        self.ctx = {
            "generators": list(self.reg.generator_ids()),
            "observables": list(G.OBSERVABLES),
            "feasible_params": self.be.feasible_params("arithmetic"),
            "donors": [], "composites": list(self.reg.composite_ids()),
        }

    def test_composite_attempt_is_valid(self):
        g = _mk(library.composite_attempt(self.mid))
        self.assertEqual(G.validate(g, self.reg), [])

    def test_m1_can_insert_composite_and_stays_valid(self):
        rng = random.Random(3)
        parent = _mk(library.attempt("K0"))
        inserted = 0
        for _ in range(200):
            child, report = M.mutate(parent, rng, self.ctx, self.reg, gen=1)
            if child is None:
                continue
            self.assertEqual(G.validate(child, self.reg), [])
            if H.uses_composite(child):
                inserted += 1
        self.assertGreater(inserted, 0, "композит не попал ни в один генотип за 200 мутаций")

    def test_random_genotype_with_composites_is_valid(self):
        rng = random.Random(11)
        seen_composite = 0
        for _ in range(60):
            g, report = M.random_genotype(rng, self.ctx, self.reg, gen=0)
            if g is None:
                continue
            self.assertEqual(G.validate(g, self.reg), [])
            if H.uses_composite(g):
                seen_composite += 1
        self.assertGreater(seen_composite, 0, "лист-композит ни разу не появился")

    def test_m6_weight_rises_only_with_composites(self):
        self.assertEqual(M.operator_weights({"composites": []})["M6"],
                         M.OPERATOR_WEIGHTS["M6"])
        self.assertEqual(M.operator_weights({"composites": [self.mid]})["M6"],
                         H.M6_WEIGHT_WITH_COMPOSITES)

    def test_composite_call_really_executes(self):
        """A4: мёртвая ссылка (узел есть, исполнения нет) — провал, а не деталь."""
        outer = _mk(library.composite_attempt(self.mid))
        state = self.ds.collision_state(self.ds.collision_ids()[0])
        res = R.run(outer, state, self.reg, self.be, budget_tokens=10 ** 6)
        ops = [e["op"] for e in res.trace]
        self.assertIn("CALL_COMPOSITE", ops)
        inner = [e for e in res.trace if "/composite/" in e.get("node_path", "")]
        self.assertTrue(inner, "внутри композита не исполнилось ничего — мёртвая ссылка")


class RegistrationGateTests(unittest.TestCase):
    def setUp(self):
        self.reg = REG.default_registry()
        self.g = _mk(library.attempt("K0"))

    def _gate(self, metrics, gate):
        return H.registration_gate("cx-x", metrics, gate, self.g, self.reg, n_min=100)

    def test_rejects_too_few_evaluations(self):
        v = self._gate({"n_evaluated": 40, "r": 0.9, "c": 100.0}, {"r": 0.5, "c": 500.0})
        self.assertFalse(v["ok"])
        self.assertTrue(any("n_evaluated" in r for r in v["reasons"]))

    def test_rejects_when_dominated_by_gate(self):
        v = self._gate({"n_evaluated": 200, "r": 0.4, "c": 900.0}, {"r": 0.6, "c": 500.0})
        self.assertFalse(v["ok"])
        self.assertTrue(any("dominated" in r for r in v["reasons"]))

    def test_accepts_cheaper_even_if_slightly_worse(self):
        """Дешёвый край Парето обязан проходить: он и переносится надёжнее всего."""
        v = self._gate({"n_evaluated": 200, "r": 0.55, "c": 250.0}, {"r": 0.60, "c": 500.0})
        self.assertTrue(v["ok"], v["reasons"])

    def test_cost_profile_is_measured_not_estimated(self):
        prof = H.cost_profile_from_traces({
            "t1": {"cost": 100, "n_calls": 1}, "t2": {"cost": 300, "n_calls": 3},
            "t3": {"cost": 200, "n_calls": 2}})
        self.assertEqual(prof["median_tokens"], 200)
        self.assertEqual(prof["median_calls"], 2)
        self.assertEqual(prof["n_observations"], 3)


if __name__ == "__main__":
    unittest.main()
