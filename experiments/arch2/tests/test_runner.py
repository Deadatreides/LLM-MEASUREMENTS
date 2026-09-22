"""Контрольные случаи runner.py — пять гарантий SPEC.md §7.2.

Гарантия, не покрытая контрольным случаем, считается невыполненной (норма проекта:
непройденная ветка не проверена).
"""

from __future__ import annotations

import unittest

import genotype as G
import runner as R
from tests.fakes import FakeRegistry, ScriptedBackend, seam_says_pass


def _state(**kw):
    base = dict(task_id="T1", task_family="code", task={}, origin_model="gen-origin")
    base.update(kw)
    return R.State(**base)


def _call(max_tokens=150, molecule="gen.m1", seed_slot=0):
    return G.CALL(molecule, prompt="K0", temperature=0.5, seed_slot=seed_slot, max_tokens=max_tokens)


def _attempt(text_marker="OK", **kw):
    return G.SEQ(_call(**kw), G.CHECK("seam.s1"))


class BudgetGuaranteeTests(unittest.TestCase):
    """ГАРАНТИЯ 1: фактическая стоимость никогда не превышает бюджет."""

    def test_call_skipped_when_estimate_exceeds_global_budget(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["OK"])
        g = G.genotype(_attempt())
        res = R.run(g, _state(), reg, be, budget_tokens=200)   # оценка 100+150=250 > 200
        self.assertEqual(res.cost, 0)
        self.assertEqual(len(be.calls), 0)
        self.assertEqual(res.outcome, R.EXHAUSTED)

    def test_call_runs_when_estimate_fits(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["OK"])
        g = G.genotype(_attempt())
        res = R.run(g, _state(), reg, be, budget_tokens=250)
        self.assertEqual(res.outcome, R.RESOLVED)
        self.assertEqual(res.cost, 100)

    def test_inner_budget_frame_binds_before_global(self):
        """Рамка считается по КОНСЕРВАТИВНОЙ оценке (100 + 150 = 250 на вызов),
        а не по фактическому расходу: до вызова факт неизвестен, а гарантия
        должна держаться при любом исходе. Рамка 400: две ветви проходят
        (остаток 400, затем 300 >= оценки 250), третьей остаётся 200 < 250."""
        reg = FakeRegistry(max_input_tokens=100)
        be = ScriptedBackend(["no", "no", "no"])
        g = G.genotype(G.BUDGET(400, G.PAR(_attempt(), _attempt(), _attempt())))
        res = R.run(g, _state(), reg, be, budget_tokens=100_000)
        self.assertEqual(len(be.calls), 2)
        self.assertEqual(res.cost, 200)
        self.assertTrue(res.exhausted)

    def test_exact_estimate_backend_makes_frames_tight(self):
        """Backend, знающий точную стоимость (offline replay читает записанные
        usage-токены), делает рамку точной, не ослабляя гарантию: est == факт."""

        class ExactBackend(ScriptedBackend):
            def estimate(self, **kw):
                return sum(self.tokens)

        reg, be = FakeRegistry(max_input_tokens=100), ExactBackend(["no", "no", "no"])
        g = G.genotype(G.BUDGET(300, G.PAR(_attempt(), _attempt(), _attempt())))
        res = R.run(g, _state(), reg, be, budget_tokens=100_000)
        self.assertEqual(len(be.calls), 3)      # 3 x 100 = 300, ровно в рамку
        self.assertEqual(res.cost, 300)

        be2 = ExactBackend(["no", "no", "no"])
        g2 = G.genotype(G.BUDGET(250, G.PAR(_attempt(), _attempt(), _attempt())))
        res2 = R.run(g2, _state(), reg, be2, budget_tokens=100_000)
        self.assertEqual(len(be2.calls), 2)
        self.assertEqual(res2.cost, 200)
        self.assertTrue(res2.exhausted)

    def test_backend_overrun_raises_instead_of_silently_exceeding(self):
        reg = FakeRegistry(max_input_tokens=100)
        be = ScriptedBackend(["OK"], tokens=(5000, 5000))       # backend врёт про расход
        g = G.genotype(_attempt())
        with self.assertRaises(R.RunnerInvariantError):
            R.run(g, _state(), reg, be, budget_tokens=100_000)


class TrustGateTests(unittest.TestCase):
    """ГАРАНТИЯ 5 (I-15): недоверенный шов не может объявить RESOLVED."""

    def test_untrusted_pass_is_not_resolved(self):
        reg = FakeRegistry(trusted=False, max_input_tokens=100, seam_impl=seam_says_pass)
        res = R.run(G.genotype(_attempt()), _state(), reg, ScriptedBackend(["OK"]), 10_000)
        self.assertEqual(res.final_status, "PASS")
        self.assertNotEqual(res.outcome, R.RESOLVED)

    def test_trusted_pass_is_resolved(self):
        reg = FakeRegistry(trusted=True, max_input_tokens=100, seam_impl=seam_says_pass)
        res = R.run(G.genotype(_attempt()), _state(), reg, ScriptedBackend(["OK"]), 10_000)
        self.assertEqual(res.outcome, R.RESOLVED)

    def test_untrusted_branch_does_not_close_par(self):
        reg = FakeRegistry(trusted=False, max_input_tokens=100, seam_impl=seam_says_pass)
        be = ScriptedBackend(["OK", "OK"])
        R.run(G.genotype(G.PAR(_attempt(), _attempt())), _state(), reg, be, 10_000)
        self.assertEqual(len(be.calls), 2)      # первая ветвь не закрыла PAR


class ParSemanticsTests(unittest.TestCase):
    def test_lazy_stop_on_first_trusted_pass(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["OK", "OK", "OK"])
        res = R.run(G.genotype(G.PAR(_attempt(), _attempt(), _attempt())), _state(), reg, be, 10_000)
        self.assertEqual(len(be.calls), 1)
        self.assertEqual(res.cost, 100)
        self.assertEqual(res.outcome, R.RESOLVED)

    def test_all_branches_run_and_cost_sums_when_none_passes(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["no", "no", "no"])
        res = R.run(G.genotype(G.PAR(_attempt(), _attempt(), _attempt())), _state(), reg, be, 10_000)
        self.assertEqual(len(be.calls), 3)
        self.assertEqual(res.cost, 300)
        self.assertEqual(res.outcome, R.UNRESOLVED)

    def test_branches_do_not_share_artifact(self):
        """Вторая ветвь стартует от ИСХОДНОГО артефакта, а не от результата первой."""
        seen = []

        class Recorder(ScriptedBackend):
            def call(self, **kw):
                seen.append(kw["state"].artifact)
                return super().call(**kw)

        reg, be = FakeRegistry(max_input_tokens=100), Recorder(["first", "second"])
        R.run(G.genotype(G.PAR(_attempt(), _attempt())), _state(artifact="INITIAL"), reg, be, 10_000)
        self.assertEqual(seen, ["INITIAL", "INITIAL"])


class SeqAndStopTests(unittest.TestCase):
    def test_seq_flows_artifact_forward(self):
        seen = []

        class Recorder(ScriptedBackend):
            def call(self, **kw):
                seen.append(kw["state"].artifact)
                return super().call(**kw)

        reg, be = FakeRegistry(max_input_tokens=100), Recorder(["first", "second"])
        g = G.genotype(G.SEQ(_call(), G.CHECK("seam.s1"), _call(), G.CHECK("seam.s1")))
        R.run(g, _state(artifact="INITIAL"), reg, be, 10_000)
        self.assertEqual(seen, ["INITIAL", "first"])

    def test_stop_terminates_whole_run(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["no", "no"])
        g = G.genotype(G.PAR(G.STOP("barrier_too_high"), _attempt()))
        res = R.run(g, _state(), reg, be, 10_000)
        self.assertEqual(len(be.calls), 0)
        self.assertEqual(res.stop_reason, "barrier_too_high")
        self.assertEqual(res.cost, 0)


class SwitchTests(unittest.TestCase):
    def test_dispatch_on_matching_case(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["OK"])
        g = G.genotype(G.SWITCH("task_family", {"code": _attempt()}, G.STOP("not_code")))
        res = R.run(g, _state(task_family="code"), reg, be, 10_000)
        self.assertEqual(res.outcome, R.RESOLVED)

    def test_falls_back_to_default(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["OK"])
        g = G.genotype(G.SWITCH("task_family", {"code": _attempt()}, G.STOP("not_code")))
        res = R.run(g, _state(task_family="arithmetic"), reg, be, 10_000)
        self.assertEqual(res.stop_reason, "not_code")
        self.assertEqual(len(be.calls), 0)


class TraceTests(unittest.TestCase):
    """ГАРАНТИИ 2-3: полнота трассы и детерминизм структуры."""

    def _run_once(self):
        reg = FakeRegistry(max_input_tokens=100)
        be = ScriptedBackend(["no", "OK"])
        g = G.genotype(G.SWITCH("task_family", {"code": G.PAR(_attempt(), _attempt())}, G.STOP("x")))
        return R.run(g, _state(), reg, be, 10_000)

    def test_raw_output_is_persisted(self):
        res = self._run_once()
        calls = [e for e in res.trace if e["op"] == "CALL" and "raw_output" in e]
        self.assertEqual([c["raw_output"] for c in calls], ["no", "OK"])

    def test_every_call_records_tokens_and_model(self):
        res = self._run_once()
        for e in res.trace:
            if e["op"] == "CALL" and "raw_output" in e:
                self.assertIn("input_tokens", e)
                self.assertIn("output_tokens", e)
                self.assertIsNotNone(e["model_id"])

    def test_structure_is_deterministic(self):
        a, b = self._run_once(), self._run_once()
        self.assertEqual([(e["node_path"], e["op"]) for e in a.trace],
                         [(e["node_path"], e["op"]) for e in b.trace])
        self.assertEqual((a.outcome, a.cost), (b.outcome, b.cost))

    def test_role_molecules_resolve_to_models(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["no", "no"])
        g = G.genotype(G.PAR(_attempt(molecule="gen.same_operator"),
                             _attempt(molecule="gen.other_operator")))
        res = R.run(g, _state(origin_model="M-origin"), reg, be, 10_000)
        models = [e["model_id"] for e in res.trace if e["op"] == "CALL" and "raw_output" in e]
        self.assertEqual(models, ["M-origin", "other-model"])


class SeamErrorTests(unittest.TestCase):
    def test_seam_exception_becomes_error_not_pass(self):
        def boom(text, task):
            raise ValueError("сломанный шов")

        reg = FakeRegistry(max_input_tokens=100, seam_impl=boom)
        res = R.run(G.genotype(_attempt()), _state(), reg, ScriptedBackend(["OK"]), 10_000)
        self.assertEqual(res.outcome, R.ERROR_OUT)
        self.assertNotEqual(res.final_status, "PASS")


if __name__ == "__main__":
    unittest.main()
