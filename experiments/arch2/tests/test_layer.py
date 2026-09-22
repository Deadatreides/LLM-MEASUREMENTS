"""Контрольные случаи уровня слоя: реестр, replay, фитнес, автокатализ.

Главный из них — `FitnessIndependenceTests`: он проверяет МЕХАНИЧЕСКИ, что пересчёт
фитнеса не зависит от исполнителя. Оба крупных дефекта измерения в истории проекта
жили внутри исполнителя и были невидимы его собственным метрикам.
"""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import fitness as F
import genotype as G
import library
import registry as REG
import replay
import runner as R

ARCH2 = Path(__file__).resolve().parents[1]


class FitnessIndependenceTests(unittest.TestCase):
    def test_fitness_does_not_import_runner_or_registry(self):
        tree = ast.parse((ARCH2 / "fitness.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for forbidden in ("runner", "registry", "replay", "evolve", "live"):
            self.assertNotIn(forbidden, imported,
                             f"fitness.py импортирует {forbidden}: пересчёт перестал быть независимым")

    def test_recomputes_from_disk_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "exp"
            base.mkdir(parents=True)
            rows = [
                {"complex_id": "cx-a", "task_id": "T1", "outcome": "RESOLVED", "cost": 100},
                {"complex_id": "cx-a", "task_id": "T2", "outcome": "UNRESOLVED", "cost": 50},
                {"complex_id": "cx-b", "task_id": "T1", "outcome": "UNRESOLVED", "cost": 10},
                {"complex_id": "cx-b", "task_id": "T2", "outcome": "RESOLVED", "cost": 10},
            ]
            with open(base / "all.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            res = F.load_results(Path(tmp), "exp")
            a = F.metrics(res["cx-a"], ["T1", "T2"])
            self.assertEqual((a["r"], a["c"]), (0.5, 150.0))
            self.assertEqual(F.unique_solves(res["cx-a"], [res["cx-b"]], ["T1", "T2"]), ["T1"])


class ParetoTests(unittest.TestCase):
    def test_layers_are_ordered_and_complete(self):
        pts = {"a": (1.0, -1.0, 0.0), "b": (0.5, -0.5, 0.0), "c": (0.2, -2.0, 0.0)}
        layers = F.pareto_layers(pts)
        self.assertIn("a", layers[0])
        self.assertIn("b", layers[0])          # b дешевле a -> недоминируем
        self.assertEqual(sum(len(l) for l in layers), 3)

    def test_novelty_axis_protects_expensive_precursor(self):
        """Дорогой и слабый комплекс, но с уникальными решениями, остаётся на фронте."""
        pts = {"strong": (0.9, -100.0, 0.0), "precursor": (0.1, -900.0, 0.2)}
        self.assertIn("precursor", F.pareto_layers(pts)[0])


class SeamGateTests(unittest.TestCase):
    def test_all_registered_seams_pass_their_control_cases(self):
        reg = REG.default_registry()
        for sid, rep in reg.self_test_report().items():
            self.assertTrue(rep["passed"], f"{sid}: {rep['failures']}")
            self.assertTrue(reg.is_trusted(sid))


class ReplayFidelityTests(unittest.TestCase):
    """Замыкание: генотип -> Runner -> replay -> шов должен ВОСПРОИЗВОДИТЬ
    опубликованные агрегаты эксп. 12. Расхождение означало бы, что слой считает
    не то, что считала исходная кампания."""

    PUBLISHED = {"K0": (0.3208, 240.8), "K1": (0.3553, 317.8),
                 "K2": (0.3616, 319.0), "promptB": (0.3931, 266.2)}

    @classmethod
    def setUpClass(cls):
        cls.ds = replay.default_dataset()
        cls.reg = REG.default_registry()

    def test_arms_reproduce_published_aggregates(self):
        be = replay.ReplayBackend(self.ds)
        ids = self.ds.collision_ids()
        for prompt, (q_exp, c_exp) in self.PUBLISHED.items():
            g = G.genotype(library.attempt(prompt))
            self.assertTrue(G.is_valid(g, self.reg), G.validate(g, self.reg))
            solved = cost = 0
            for cid in ids:
                res = R.run(g, self.ds.collision_state(cid), self.reg, be, budget_tokens=10 ** 6)
                solved += res.outcome == R.RESOLVED
                cost += res.cost
            self.assertAlmostEqual(solved / len(ids), q_exp, places=4, msg=f"arm {prompt}")
            self.assertAlmostEqual(cost / len(ids), c_exp, places=1, msg=f"arm {prompt}")

    def test_blind_resample_never_replays_the_collision_itself(self):
        be = replay.ReplayBackend(self.ds)
        g = G.genotype(library.blind(0))
        for cid in self.ds.collision_ids()[:25]:
            state = self.ds.collision_state(cid)
            R.run(g, state, self.reg, be, budget_tokens=10 ** 6)
        self.assertGreater(be.n_served, 0)

    def test_repair_depth_two_is_unavailable(self):
        be = replay.ReplayBackend(self.ds)
        g = G.genotype(G.SEQ(library.attempt("K0"), library.attempt("K0")))
        state = self.ds.collision_state(self.ds.collision_ids()[0])
        R.run(g, state, self.reg, be, budget_tokens=10 ** 6)
        self.assertTrue(any("depth > 1" in r for r in be.unavailable_reasons))


class CompositeTests(unittest.TestCase):
    """Автокатализ: зарегистрированный комплекс вызывается как молекула, и его
    внутренняя трасса ВИДНА (чёрных ящиков нет, SPEC.md §10)."""

    def test_registered_complex_runs_as_a_molecule(self):
        ds = replay.default_dataset()
        reg = REG.Registry()
        be = replay.ReplayBackend(ds)

        inner = G.genotype(library.attempt("K0"))
        mid = reg.register_composite(inner["complex_id"], inner, {"mean_tokens": 240.8})
        self.assertTrue(reg.is_composite(mid))

        outer = G.genotype(G.PAR({"op": "CALL", "molecule": mid, "params": {}},
                                 library.blind(0)))
        self.assertTrue(G.is_valid(outer, reg), G.validate(outer, reg))

        state = ds.collision_state(ds.collision_ids()[0])
        res = R.run(outer, state, reg, be, budget_tokens=10 ** 6)
        ops = [e["op"] for e in res.trace]
        self.assertIn("CALL_COMPOSITE", ops)
        self.assertTrue(any(e["op"] == "CHECK" for e in res.trace),
                        "внутренняя трасса композита не видна — появился чёрный ящик")


class MutationTests(unittest.TestCase):
    def test_every_mutant_is_valid_or_none(self):
        import random

        import mutate as M

        ds = replay.default_dataset()
        reg = REG.default_registry()
        be = replay.ReplayBackend(ds)
        ctx = {"generators": list(reg.generator_ids()), "observables": list(G.OBSERVABLES),
               "feasible_params": be.feasible_params("arithmetic"), "donors": [], "composites": []}
        rng = random.Random(1)
        parent = G.genotype(library.attempt("K0"))
        n_ok = 0
        for i in range(60):
            child, report = M.mutate(parent, rng, ctx, reg, gen=1)
            if child is not None:
                self.assertIn(report["operator"], ("M1", "M2", "M3", "M4", "M5", "M6"))
                self.assertEqual(G.validate(child, reg), [])
                self.assertNotEqual(child["complex_id"], parent["complex_id"])
                n_ok += 1
        self.assertGreater(n_ok, 0, "ни одной валидной мутации за 60 попыток")

    def test_random_genotypes_are_valid(self):
        import random

        import mutate as M

        ds = replay.default_dataset()
        reg = REG.default_registry()
        be = replay.ReplayBackend(ds)
        ctx = {"generators": list(reg.generator_ids()), "observables": list(G.OBSERVABLES),
               "feasible_params": be.feasible_params("arithmetic"), "donors": [], "composites": []}
        rng = random.Random(7)
        for _ in range(25):
            g, _report = M.random_genotype(rng, ctx, reg, gen=0)
            if g is not None:
                self.assertEqual(G.validate(g, reg), [])


if __name__ == "__main__":
    unittest.main()
