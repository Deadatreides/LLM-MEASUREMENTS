"""Контрольные случаи genotype.py: канонизация, идентичность, валидатор.

Каждый случай имеет ЗАРАНЕЕ ИЗВЕСТНЫЙ ответ по контракту SPEC.md §1.1/§7.2.
"""

from __future__ import annotations

import unittest

import genotype as G
from tests.fakes import FakeRegistry


def _leaf():
    return G.SEQ(G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150),
                 G.CHECK("seam.s1"))


class CanonicalisationTests(unittest.TestCase):
    def test_same_structure_different_key_order_gives_same_id(self):
        a = G.genotype({"op": "CHECK", "seam": "seam.s1"})
        b = G.genotype({"seam": "seam.s1", "op": "CHECK"})
        self.assertEqual(a["complex_id"], b["complex_id"])

    def test_provenance_does_not_change_id(self):
        a = G.genotype(_leaf(), gen=0, parent_ids=[], origin="seed")
        b = G.genotype(_leaf(), gen=7, parent_ids=["cx-aaaaaa"], origin="mutate:M3")
        self.assertEqual(a["complex_id"], b["complex_id"])

    def test_different_structure_gives_different_id(self):
        a = G.genotype(G.STOP("a"))
        b = G.genotype(G.STOP("b"))
        self.assertNotEqual(a["complex_id"], b["complex_id"])

    def test_roundtrip_json(self):
        a = G.genotype(_leaf())
        self.assertEqual(G.loads(G.dumps(a)), a)


class WalkTests(unittest.TestCase):
    def test_depth_and_node_count(self):
        g = G.genotype(G.BUDGET(500, G.PAR(_leaf(), G.STOP("x"))))
        self.assertEqual(G.n_nodes(g["root"]), 1 + 1 + (1 + 2) + 1)
        self.assertEqual(G.depth(g["root"]), 4)

    def test_paths_are_unique(self):
        g = G.genotype(G.PAR(_leaf(), _leaf()))
        paths = [p for p, _ in G.walk(g["root"])]
        self.assertEqual(len(paths), len(set(paths)))


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.reg = FakeRegistry()

    def test_valid_genotype_passes(self):
        g = G.genotype(G.SWITCH("task_family",
                                {"code": _leaf()},
                                G.BUDGET(700, G.PAR(_leaf(), G.STOP("give_up")))))
        self.assertEqual(G.validate(g, self.reg), [])

    def test_unknown_op_rejected(self):
        g = G.genotype({"op": "MAGIC"})
        self.assertTrue(any("unknown_op" in v for v in G.validate(g, self.reg)))

    def test_call_requires_generator(self):
        g = G.genotype(G.CALL("seam.s1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150))
        self.assertTrue(any("CALL_requires_generator" in v for v in G.validate(g, self.reg)))

    def test_check_requires_seam(self):
        g = G.genotype(G.CHECK("gen.m1"))
        self.assertTrue(any("CHECK_requires_seam" in v for v in G.validate(g, self.reg)))

    def test_param_off_grid_rejected(self):
        g = G.genotype(G.CALL("gen.m1", prompt="K0", temperature=0.77, seed_slot=0, max_tokens=150))
        self.assertTrue(any("param_off_grid" in v for v in G.validate(g, self.reg)))

    def test_missing_param_rejected(self):
        g = G.genotype(G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0))
        self.assertTrue(any("missing_params" in v for v in G.validate(g, self.reg)))

    def test_unknown_observable_rejected(self):
        g = G.genotype(G.SWITCH("vibes", {"good": G.STOP("x")}, G.STOP("y")))
        self.assertTrue(any("unknown_obs" in v for v in G.validate(g, self.reg)))

    def test_unknown_case_value_rejected(self):
        g = G.genotype(G.SWITCH("task_family", {"poetry": G.STOP("x")}, G.STOP("y")))
        self.assertTrue(any("unknown_case_values" in v for v in G.validate(g, self.reg)))

    def test_switch_without_default_rejected(self):
        g = G.genotype({"op": "SWITCH", "obs": "task_family", "cases": {"code": G.STOP("x")}})
        self.assertTrue(any("missing_fields" in v for v in G.validate(g, self.reg)))

    def test_empty_children_rejected(self):
        self.assertTrue(any("nonempty" in v for v in G.validate(G.genotype(G.SEQ()), self.reg)))
        self.assertTrue(any("nonempty" in v for v in G.validate(G.genotype(G.PAR()), self.reg)))

    def test_nonpositive_budget_rejected(self):
        g = G.genotype(G.BUDGET(0, G.STOP("x")))
        self.assertTrue(any("limit_tokens_must_be_positive_int" in v for v in G.validate(g, self.reg)))

    def test_unreachable_after_stop_rejected(self):
        g = G.genotype(G.SEQ(G.STOP("early"), G.CHECK("seam.s1")))
        self.assertTrue(any("unreachable_after_stop" in v for v in G.validate(g, self.reg)))

    def test_depth_limit_enforced(self):
        node = G.STOP("deep")
        for _ in range(G.MAX_DEPTH + 2):
            node = G.BUDGET(100, node)
        self.assertTrue(any(v.startswith("depth:") for v in G.validate(G.genotype(node), self.reg)))

    def test_unknown_molecule_rejected(self):
        class Empty(FakeRegistry):
            def has_molecule(self, mid): return False
        g = G.genotype(_leaf())
        self.assertTrue(any("unknown_molecule" in v or "unknown_seam" in v
                            for v in G.validate(g, Empty())))

    def test_extra_field_rejected(self):
        g = G.genotype({"op": "STOP", "reason": "x", "note": "лишнее"})
        self.assertTrue(any("unexpected_fields" in v for v in G.validate(g, self.reg)))

    def test_tampered_complex_id_detected(self):
        g = G.genotype(_leaf())
        g["complex_id"] = "cx-000000"
        self.assertTrue(any("complex_id_mismatch" in v for v in G.validate(g, self.reg)))


class CostBoundTests(unittest.TestCase):
    def setUp(self):
        self.reg = FakeRegistry(max_input_tokens=100)

    def test_seq_sums_and_switch_takes_max(self):
        call150 = G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=150)
        call320 = G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=320)
        self.assertEqual(G.cost_upper_bound(G.SEQ(call150, call320), self.reg), 250 + 420)
        self.assertEqual(G.cost_upper_bound(G.SWITCH("task_family", {"code": call150}, call320), self.reg), 420)

    def test_budget_caps_the_bound(self):
        call320 = G.CALL("gen.m1", prompt="K0", temperature=0.5, seed_slot=0, max_tokens=320)
        self.assertEqual(G.cost_upper_bound(G.BUDGET(100, call320), self.reg), 100)


if __name__ == "__main__":
    unittest.main()
