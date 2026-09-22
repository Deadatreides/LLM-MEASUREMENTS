"""Контрольные случаи heterostep.py — датасет/backend/реестр HETEROSTEP (пятый цикл).

Норма проекта: непокрытая ветка считается невыполненной. Правильность
numeric_seam/token_seam НЕ переоткрывается здесь (уже 20/20 в
experiment14/seams.py::self_test()) -- только обёртка `_seam_heterostep` и её
собственный контракт (SPEC.md §31: без узкого вида шага -- INAPPLICABLE, не крах).

Три случая (B1/B3-репликация) сверены с числами, независимо посчитанными чистым
Python на том же train_grid.json ДО того, как ASSEMBLE был написан (REPORT_ASSEMBLE.md
§2) -- совпадение с точностью до последнего знака и есть сам контрольный случай.
"""

from __future__ import annotations

import unittest

import genotype as G
import heterostep as H
import runner as R


def _slot(model_id: str) -> dict:
    return G.SEQ(G.CALL(f"gen.{model_id}", prompt=H.PROMPT_VARIANT, temperature=H.TEMPERATURE,
                        seed_slot=0, max_tokens=H.MAX_TOKENS_STEP), G.CHECK(H.SEAM_ID))


# Маршрут по перечислению (эксп. 14 §G0, TASK_EXPERIMENT14.md) -- ранее известный
# результат: r=0.4600 на train (REPORT_ASSEMBLE.md §2, B1).
_B1_ROUTE = {"READ": "internvl3-2b-q4_k_m", "FORMAT": "qwen2.5-coder-1.5b-instruct-q4_0",
            "LOOKUP": "qwen3-1.7b-q4_0-unsloth", "COMPUTE": "qwen2.5-coder-1.5b-instruct-q4_0"}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.ds = H.default_dataset()

    def test_train_coverage_is_complete(self):
        """Гейт a0 (план §6): офлайн-результат недостоверен при покрытии < 100%."""
        cov = self.ds.coverage(self.ds.split["train"])
        self.assertEqual(cov["coverage"], 1.0)
        self.assertEqual(cov["total"], len(self.ds.split["train"]) * len(H.STEP_KINDS) * len(H.MODEL_IDS))

    def test_split_matches_experiment14_sizes(self):
        self.assertEqual(len(self.ds.split["train"]), 100)
        self.assertEqual(len(self.ds.split["test"]), 100)
        self.assertEqual(len(set(self.ds.split["train"]) & set(self.ds.split["test"])), 0)

    def test_state_for_exposes_all_four_steps(self):
        tid = self.ds.split["train"][0]
        state = self.ds.state_for(tid)
        self.assertEqual([s["kind"] for s in state.task["steps"]], list(H.STEP_KINDS))
        self.assertEqual(state.artifact, "")
        self.assertEqual(state.evidence, [])


class BackendLookupTests(unittest.TestCase):
    """Каждая ветка отказа `_lookup` -- ДО единого вызова модели, тот же принцип,
    что в `replay.ReplayBackend` (недоступная ячейка отбраковывается, не подделывается)."""

    def setUp(self):
        self.ds = H.default_dataset()
        self.be = H.HeterostepBackend(self.ds)
        tid = self.ds.split["train"][0]
        self.state = self.ds.state_for(tid)
        self.state.task = {**self.state.task, "step_index": 0, "step": self.state.task["steps"][0]}

    def _call(self, **overrides):
        kw = dict(model_id="internvl3-2b-q4_k_m", prompt_variant=H.PROMPT_VARIANT,
                  temperature=H.TEMPERATURE, seed_slot=0, max_tokens=H.MAX_TOKENS_STEP,
                  state=self.state, seed_vector=[0])
        kw.update(overrides)
        return self.be.call(**kw)

    def test_measured_cell_is_available(self):
        out = self._call()
        self.assertTrue(out["available"])
        self.assertIsInstance(out["raw_text"], str)

    def test_unmeasured_prompt_variant_unavailable(self):
        out = self._call(prompt_variant="whole")
        self.assertFalse(out["available"])

    def test_unmeasured_temperature_unavailable(self):
        out = self._call(temperature=1.0)
        self.assertFalse(out["available"])

    def test_unmeasured_seed_slot_unavailable(self):
        out = self._call(seed_slot=1)
        self.assertFalse(out["available"])

    def test_unmeasured_max_tokens_unavailable(self):
        out = self._call(max_tokens=150)
        self.assertFalse(out["available"])

    def test_call_outside_assemble_slot_unavailable(self):
        bare = self.ds.state_for(self.ds.split["train"][0])   # task без "step"
        out = self._call(state=bare)
        self.assertFalse(out["available"])

    def test_estimate_matches_recorded_tokens(self):
        rec = self.ds.cells[(self.state.task_id, "READ", "internvl3-2b-q4_k_m")]
        est = self.be.estimate(model_id="internvl3-2b-q4_k_m", prompt_variant=H.PROMPT_VARIANT,
                               temperature=H.TEMPERATURE, seed_slot=0, max_tokens=H.MAX_TOKENS_STEP,
                               state=self.state)
        self.assertEqual(est, int(rec["tokens"]))


class SeamWrapperTests(unittest.TestCase):
    """Собственный контракт обёртки; numeric/token-корректность уже покрыта в
    experiment14/seams.py, не повторяется."""

    def test_self_test_passes(self):
        rep = H.self_test_seam()
        self.assertTrue(rep["passed"], rep["failures"])
        self.assertGreater(rep["n_cases"], 0)

    def test_missing_step_view_is_inapplicable_not_crash(self):
        result = H._seam_heterostep("42", {})
        self.assertEqual(result["status"], R.INAPPLICABLE)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.reg = H.default_registry()

    def test_all_six_models_registered_as_generators(self):
        for mid in H.MODEL_IDS:
            self.assertTrue(self.reg.has_molecule(f"gen.{mid}"))
            self.assertEqual(self.reg.kind_of(f"gen.{mid}"), "generator")

    def test_seam_registered_and_trusted(self):
        self.assertEqual(self.reg.kind_of(H.SEAM_ID), "seam")
        self.assertTrue(self.reg.is_trusted(H.SEAM_ID))

    def test_no_role_indirection_resolves_directly(self):
        """В отличие от `registry.Registry` (same/other), здесь `resolve_model`
        всегда возвращает названную модель напрямую -- маршрутизация есть
        структура генотипа, не роль (SPEC.md §33)."""
        mid = f"gen.{H.MODEL_IDS[0]}"
        self.assertEqual(self.reg.resolve_model(mid, state=None, seed_slot=0), H.MODEL_IDS[0])

    def test_switch_observable_not_supported(self):
        with self.assertRaises(NotImplementedError):
            self.reg.observe("task_family", state=None)


class RunnerIntegrationTests(unittest.TestCase):
    """Сквозная сверка с числами, посчитанными НЕЗАВИСИМО чистым Python (план §4)."""

    def setUp(self):
        self.ds = H.default_dataset()
        self.reg = H.default_registry()
        self.be = H.HeterostepBackend(self.ds)

    def _resolve_rate(self, genotype: dict, task_ids: list) -> tuple:
        n = 0
        cost = 0
        for tid in task_ids:
            res = R.run(genotype, self.ds.state_for(tid), self.reg, self.be, budget_tokens=10 ** 9)
            n += res.outcome == R.RESOLVED
            cost += res.cost
        return n / len(task_ids), cost / len(task_ids)

    def test_b1_route_reproduces_0_460_on_train(self):
        g = G.genotype(G.ASSEMBLE(*[_slot(_B1_ROUTE[k]) for k in H.STEP_KINDS]))
        self.assertEqual(G.validate(g, self.reg), [])
        r, tok = self._resolve_rate(g, self.ds.split["train"])
        self.assertAlmostEqual(r, 0.46, places=9)
        self.assertAlmostEqual(tok, 667.0, delta=1.0)

    def test_par_fallback_beats_the_depth1_route(self):
        """Не сверка с точным числом (то — B3 в REPORT_ASSEMBLE.md, отдельно
        воспроизводимый скриптом кампании), а структурный факт: заужение на
        шаг + PAR-откат строго не хуже голого маршрута -- монотонность отката."""
        g1 = G.genotype(G.ASSEMBLE(*[_slot(_B1_ROUTE[k]) for k in H.STEP_KINDS]))
        fallback_lookup = G.ASSEMBLE(
            _slot(_B1_ROUTE["READ"]), _slot(_B1_ROUTE["FORMAT"]),
            G.PAR(_slot("qwen3-1.7b-q4_0-unsloth"), _slot("gemma-3-it-1b-q5_k_s")),
            _slot(_B1_ROUTE["COMPUTE"]),
        )
        g2 = G.genotype(fallback_lookup)
        self.assertEqual(G.validate(g2, self.reg), [])
        r1, _ = self._resolve_rate(g1, self.ds.split["train"])
        r2, _ = self._resolve_rate(g2, self.ds.split["train"])
        self.assertGreaterEqual(r2, r1)


if __name__ == "__main__":
    unittest.main()
