"""Контрольные случаи усиления отбора (SPEC.md §21-28).

Три механизма проверяются на синтетических данных с заранее известным ответом:
ниши по каркасу (§23), парсимония регистрации (§24), dual-slice D3 (§25) — плюс
критерий регрессии V1 (§26). Ни один из тестов не требует GPU.
"""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import evolve
import fitness as F
import genotype as G
import heredity as H
import library
import registry as REG


class _StubRegistry:
    """Реестр-заглушка для тестов select_elite/_root_family: только family_of,
    больше ничего не требуется, если генотипы не содержат CALL на композит
    (composite_nesting_depth короткозамыкается ДО обращения к registry)."""

    def family_of(self, mid):
        return mid

    def is_composite(self, mid):
        return False

    def get(self, mid):
        return {}


def _mk_evolution(registry=None) -> evolve.Evolution:
    return evolve.Evolution(
        registry=registry, backend=None, dataset=None, experiment_id="test",
        runs_dir=Path(tempfile.mkdtemp()), rng=random.Random(0),
        screen_ids=[], control_slices=[[]], holdout_ids=[],
    )


class NicheKeyTests(unittest.TestCase):
    def test_identical_skeleton_same_key_different_molecule(self):
        a = G.genotype(G.SEQ(G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150),
                             G.CHECK("seam.s1")))
        b = G.genotype(G.SEQ(G.CALL("gen.m2", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150),
                             G.CHECK("seam.s1")))
        self.assertEqual(H.niche_key(a), H.niche_key(b))
        self.assertNotEqual(a["complex_id"], b["complex_id"])   # разные комплексы, одна ниша

    def test_different_shape_different_key(self):
        flat = G.genotype(G.SEQ(G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150),
                                G.CHECK("seam.s1")))
        wide = G.genotype(G.PAR(G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150),
                                G.CALL("gen.m2", prompt="K0", temperature=0.5, seed_slot=1, max_tokens=150)))
        self.assertNotEqual(H.niche_key(flat), H.niche_key(wide))


class ComplexityKeyTests(unittest.TestCase):
    def test_fewer_nodes_sorts_first(self):
        small = G.genotype(G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150))
        big = G.genotype(G.PAR(
            G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150),
            G.CALL("gen.m2", prompt="K0", temperature=0.5, seed_slot=1, max_tokens=150),
            G.CALL("gen.m3", prompt="K0", temperature=0.5, seed_slot=2, max_tokens=150),
        ))
        ks, kb = H.complexity_key(small, None), H.complexity_key(big, None)
        self.assertLess(ks, kb)     # кортеж (n_nodes, n_raw_calls, nesting) -- меньше значит проще

    def test_composite_nesting_depth_is_zero_without_registry_lookup(self):
        plain = G.genotype(G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150))
        self.assertEqual(H.composite_nesting_depth(plain, None), 0)


class DominatedByRegisteredTests(unittest.TestCase):
    """SPEC.md §24: дешёвый край Парето обязан проходить, доминируемая более
    сложная обёртка -- нет."""

    def test_worse_and_more_complex_is_rejected(self):
        registered = [{"molecule_id": "cx.simple", "r": 0.6, "c": 250.0, "n_nodes": 4}]
        dom = H.dominated_by_registered(0.5, 300.0, 9, registered)
        self.assertIsNotNone(dom)
        self.assertEqual(dom["molecule_id"], "cx.simple")

    def test_cheaper_despite_more_complex_is_not_rejected(self):
        """Дешевле, хоть и сложнее -- доминирования по (r,-c) нет ни в одну сторону."""
        registered = [{"molecule_id": "cx.simple", "r": 0.6, "c": 250.0, "n_nodes": 4}]
        dom = H.dominated_by_registered(0.25, 40.0, 9, registered)
        self.assertIsNone(dom)

    def test_simpler_candidate_never_rejected_regardless_of_metrics(self):
        registered = [{"molecule_id": "cx.simple", "r": 0.9, "c": 10.0, "n_nodes": 20}]
        dom = H.dominated_by_registered(0.1, 900.0, 3, registered)
        self.assertIsNone(dom)


class FailGateTests(unittest.TestCase):
    def test_negative_upper_bound_fails(self):
        self.assertTrue(H.fail_gate({"ci_95": [-0.5, -0.1]}))

    def test_ci_touching_zero_does_not_fail(self):
        self.assertFalse(H.fail_gate({"ci_95": [-0.5, 0.0]}))
        self.assertFalse(H.fail_gate({"ci_95": [0.1, 0.3]}))


class NicheReservationTests(unittest.TestCase):
    """SPEC.md §23: своп только избыточного представителя ВНЕ фронта; своя ниша
    резервируется только квалифицирующемуся (beats_gate) кандидату."""

    def setUp(self):
        self._old_elite_size = evolve.ELITE_SIZE
        evolve.ELITE_SIZE = 3          # достаточно тесно, чтобы стандартный отбор исключил Z

        skel_x = lambda mid, slot: G.SEQ(                                            # noqa: E731
            G.CALL(mid, prompt="K0", temperature=0.5, seed_slot=slot, max_tokens=150),
            G.CHECK("seam.s1"))
        Y = G.genotype(G.PAR(G.CALL("gen.my1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150),
                             G.CALL("gen.my2", prompt="K0", temperature=0.5, seed_slot=1, max_tokens=150)))
        X1 = G.genotype(skel_x("gen.mx1", 0))
        X2 = G.genotype(skel_x("gen.mx2", 0))
        Z = G.genotype(G.SWITCH("task_family", {"code": G.STOP("a")}, G.STOP("b")))

        self.Y, self.X1, self.X2, self.Z = Y["complex_id"], X1["complex_id"], X2["complex_id"], Z["complex_id"]
        self.ev = _mk_evolution(_StubRegistry())
        self.ev.genotypes = {self.Y: Y, self.X1: X1, self.X2: X2, self.Z: Z}

        # Y доминирует всех; X1/X2 взаимно недоминируемы (одна ниша, оба входят в
        # стандартную элиту); Z доминируется X1/X2, но НИША Z уникальна и Z бьёт планку.
        self.scored = {
            self.Y: {"fitness_vector": (0.9, -50.0, 0.0), "r": 0.9},
            self.X1: {"fitness_vector": (0.70, -100.0, 0.0), "r": 0.70},
            self.X2: {"fitness_vector": (0.65, -90.0, 0.0), "r": 0.65},
            self.Z: {"fitness_vector": (0.50, -150.0, 0.0), "r": 0.50},
        }
        self.ids = [self.Y, self.X1, self.X2, self.Z]

    def tearDown(self):
        evolve.ELITE_SIZE = self._old_elite_size

    def test_standard_selection_excludes_unique_niche_Z(self):
        elite = self.ev.select_elite(self.scored, self.ids, {}, [], win_details={})
        self.assertEqual(set(elite), {self.Y, self.X1, self.X2})
        self.assertNotIn(self.Z, elite)

    def test_qualifying_unique_niche_swaps_into_elite(self):
        win_details = {self.Z: {"beats_gate": True}}
        elite = self.ev.select_elite(self.scored, self.ids, {}, [], win_details=win_details)
        self.assertIn(self.Z, elite)
        self.assertIn(self.Y, elite)                          # фронт не тронут
        self.assertNotIn(self.X2, elite)                       # худший избыточной ниши свопнут
        self.assertIn(self.X1, elite)                          # лучший избыточной ниши остался
        self.assertEqual(len(elite), 3)                        # ELITE_SIZE не превышен

    def test_non_qualifying_niche_does_not_swap(self):
        elite = self.ev.select_elite(self.scored, self.ids, {}, [], win_details={self.Z: {"beats_gate": False}})
        self.assertNotIn(self.Z, elite)


class DualSliceD3Tests(unittest.TestCase):
    """SPEC.md §25: изгнание из элиты только после ДВУХ ПОДРЯД провальных срезов."""

    def setUp(self):
        self.ev = _mk_evolution(None)
        self.ev.accounts = {"X": evolve.Account(born_gen=0), "gate": evolve.Account(born_gen=0)}
        self.ev.references = ["gate"]
        self.task_ids = [f"t{i}" for i in range(20)]

    def _results(self, x_pass: int, gate_pass: int) -> dict:
        def build(n_pass):
            return {t: {"outcome": "RESOLVED" if i < n_pass else "UNRESOLVED", "cost": 10}
                    for i, t in enumerate(self.task_ids)}
        return {"X": build(x_pass), "gate": build(gate_pass)}

    def test_single_failure_does_not_drop(self):
        bad = self._results(x_pass=0, gate_pass=20)
        dropped = self.ev._d3_elite_dropouts(bad, ["X"], "gate", self.task_ids)
        self.assertEqual(dropped, [])
        self.assertEqual(self.ev.accounts["X"].gate_fail_streak, 1)

    def test_two_consecutive_failures_drop(self):
        bad = self._results(x_pass=0, gate_pass=20)
        self.ev._d3_elite_dropouts(bad, ["X"], "gate", self.task_ids)
        dropped = self.ev._d3_elite_dropouts(bad, ["X"], "gate", self.task_ids)
        self.assertEqual([c for c, _ in dropped], ["X"])
        self.assertEqual(self.ev.accounts["X"].gate_fail_streak, 2)

    def test_recovery_resets_streak(self):
        bad = self._results(x_pass=0, gate_pass=20)
        good = self._results(x_pass=20, gate_pass=0)
        self.ev._d3_elite_dropouts(bad, ["X"], "gate", self.task_ids)
        self.assertEqual(self.ev.accounts["X"].gate_fail_streak, 1)
        dropped = self.ev._d3_elite_dropouts(good, ["X"], "gate", self.task_ids)
        self.assertEqual(dropped, [])
        self.assertEqual(self.ev.accounts["X"].gate_fail_streak, 0)

    def test_reference_is_never_dropped(self):
        bad = self._results(x_pass=0, gate_pass=20)
        for _ in range(3):
            dropped = self.ev._d3_elite_dropouts(bad, ["gate"], "gate", self.task_ids)
        self.assertEqual(dropped, [])

    def test_complex_dead_since_d1_is_skipped_not_crashed(self):
        bad = self._results(x_pass=0, gate_pass=20)
        dropped = self.ev._d3_elite_dropouts(bad, ["X", "not_evaluated_this_gen"], "gate", self.task_ids)
        self.assertEqual(dropped, [])   # первый провал одного среза, второй -- не в results


class ParsimonyRegistrationTests(unittest.TestCase):
    """SPEC.md §24, пятый критерий регистрации: интеграционная проверка на реальном
    реестре (нужен для G.validate внутри registration_gate)."""

    def setUp(self):
        self.reg = REG.default_registry()
        self.ev = _mk_evolution(self.reg)
        self.ev.references = []
        self.control_ids = [f"t{i}" for i in range(120)]

    def _per_task(self, n_resolved: int, cost_each: int) -> dict:
        return {t: {"outcome": "RESOLVED" if i < n_resolved else "UNRESOLVED", "cost": cost_each}
                for i, t in enumerate(self.control_ids)}

    def test_dominated_more_complex_wrapper_is_rejected(self):
        wrapper = G.genotype(library.attempt("K0"))
        cid = wrapper["complex_id"]
        n_nodes = G.n_nodes(wrapper["root"])
        self.ev.genotypes = {cid: wrapper}
        self.ev.elite = [cid]
        self.ev.registered = [{"molecule_id": "cx.stub", "complex_id": "cx-stub", "gen": 0,
                               "r": 0.6, "c": 250.0, "n_nodes": max(1, n_nodes - 1), "cost_profile": {}}]
        results = {"gate": self._per_task(0, 10), cid: self._per_task(60, 300)}   # r=0.5, c=600

        rejections = self.ev._try_register_molecule(1, results, self.control_ids, {}, "gate")

        self.assertEqual(len(self.ev.registered), 1)          # ничего нового не добавлено
        self.assertEqual(len(rejections), 1)
        self.assertEqual(rejections[0]["complex_id"], cid)
        self.assertEqual(rejections[0]["dominated_by"], "cx.stub")

    def test_cheaper_but_weaker_candidate_is_not_rejected(self):
        cheap = G.genotype(library.attempt("K2"))
        cid = cheap["complex_id"]
        self.ev.genotypes = {cid: cheap}
        self.ev.elite = [cid]
        self.ev.registered = [{"molecule_id": "cx.stub", "complex_id": "cx-stub", "gen": 0,
                               "r": 0.6, "c": 250.0, "n_nodes": 4, "cost_profile": {}}]
        results = {"gate": self._per_task(0, 10), cid: self._per_task(30, 10)}   # r=0.25, c=40

        rejections = self.ev._try_register_molecule(1, results, self.control_ids, {}, "gate")

        self.assertEqual(rejections, [])
        self.assertEqual(len(self.ev.registered), 2)
        self.assertEqual(self.ev.registered[-1]["complex_id"], cid)
        self.assertIn("n_nodes", self.ev.registered[-1])


class PairedRatioCiTests(unittest.TestCase):
    """SPEC.md §26 (V1): провал <=> point(ratio) < 1 И верхняя граница CI < 1."""

    def _pair(self, a_resolved: int, a_cost: int, b_resolved: int, b_cost: int, n: int = 60):
        task_ids = [f"t{i}" for i in range(n)]

        def build(resolved, cost):
            return {t: {"outcome": "RESOLVED" if i < resolved else "UNRESOLVED", "cost": cost}
                    for i, t in enumerate(task_ids)}

        return task_ids, build(a_resolved, a_cost), build(b_resolved, b_cost)

    def test_equal_campaigns_ratio_is_one(self):
        task_ids, res, gate = self._pair(30, 100, 30, 100)
        out = F.paired_ratio_ci(res, gate, res, gate, task_ids, n_boot=500)
        self.assertAlmostEqual(out["point"], 1.0, places=6)
        self.assertFalse(out["regression"])

    def test_worse_b_gives_confident_regression(self):
        """A стоит РОВНО как её планка (E_A=1.0); B вдвое дешевле СВОЕЙ планки
        (E_B=2.0) -- A хуже B по относительному преимуществу => ratio<1 => регрессия."""
        task_ids = [f"t{i}" for i in range(80)]

        def build(n_pass, cost):
            return {t: {"outcome": "RESOLVED" if i < n_pass else "UNRESOLVED", "cost": cost}
                    for i, t in enumerate(task_ids)}

        gate = build(40, 100)                    # общая планка обеих кампаний, c=200
        a = build(40, 100)                        # c_a=200 -> E_A = 200/200 = 1.0
        b = build(40, 50)                         # c_b=100 -> E_B = 200/100 = 2.0
        out = F.paired_ratio_ci(a, gate, b, gate, task_ids, n_boot=2000)
        self.assertAlmostEqual(out["point"], 0.5, places=6)
        self.assertLess(out["ci_95"][1], 1.0)
        self.assertTrue(out["regression"])


if __name__ == "__main__":
    unittest.main()
