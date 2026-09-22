"""Контрольные случаи ASSEMBLE — восьмого узла языка (SPEC.md §30-34, пятый цикл).

ASSEMBLE выведен из измеренного: эксп. 14 показал, что маршрут «шаг i от модели X»
даёт +38 п.п. (G1, CI [+0.27,+0.49]), но подбирался перечислением, а не отбором —
язык из семи узлов не мог его выразить. Гарантия, не покрытая контрольным случаем,
считается невыполненной (та же норма, что в test_runner.py).

Форма поля ASSEMBLE намеренно идентична SEQ/PAR (список под `children`), поэтому
здесь НЕ дублируются случаи, уже покрытые для SEQ/PAR структурно (произвольная
вложенность, канонизация id, roundtrip JSON) — они наследуются автоматически через
общие ветки `children_of`/`walk`/`cost_upper_bound`/валидатора. Здесь проверяется то,
что у ASSEMBLE РАСХОДИТСЯ: агрегатная семантика PASS/FAIL по ВСЕМ слотам и заужение
состояния на шаг задачи.
"""

from __future__ import annotations

import unittest

import genotype as G
import mutate as M
import runner as R
from tests.fakes import FakeRegistry, ScriptedBackend


def _state(**kw):
    base = dict(task_id="T1", task_family="code",
                task={"steps": [{"kind": "a"}, {"kind": "b"}]}, origin_model="gen-origin")
    base.update(kw)
    return R.State(**base)


def _call(max_tokens=150, molecule="gen.m1", seed_slot=0):
    return G.CALL(molecule, prompt="K0", temperature=0.5, seed_slot=seed_slot, max_tokens=max_tokens)


def _attempt(**kw):
    return G.SEQ(_call(**kw), G.CHECK("seam.s1"))


class ValidatorTests(unittest.TestCase):
    """SPEC.md §31: форма поля идентична SEQ/PAR, поле — `children`, непустой список."""

    def setUp(self):
        self.reg = FakeRegistry(max_input_tokens=100)

    def test_valid_assemble_has_no_violations(self):
        g = G.genotype(G.ASSEMBLE(_attempt(), _attempt()))
        self.assertEqual(G.validate(g, self.reg), [])

    def test_empty_children_rejected(self):
        g = G.genotype({"op": "ASSEMBLE", "children": []})
        self.assertTrue(any("ASSEMBLE_children_must_be_nonempty_list" in v
                            for v in G.validate(g, self.reg)))

    def test_missing_children_field_rejected(self):
        g = G.genotype({"op": "ASSEMBLE"})
        self.assertTrue(any("missing_fields" in v for v in G.validate(g, self.reg)))

    def test_extra_field_rejected(self):
        g = G.genotype({"op": "ASSEMBLE", "children": [G.STOP("x")], "extra": 1})
        self.assertTrue(any("unexpected_fields" in v for v in G.validate(g, self.reg)))

    def test_stop_mid_children_is_not_flagged_unreachable(self):
        """В отличие от SEQ, у ASSEMBLE нет понятия «недостижимый код» — то же
        решение, что уже принято для PAR (unreachable-check в _validate_node
        применяется только при op == "SEQ")."""
        g = G.genotype(G.ASSEMBLE(G.STOP("x"), _attempt()))
        self.assertEqual(G.validate(g, self.reg), [])

    def test_unknown_op_still_rejected(self):
        g = G.genotype({"op": "ASSEMBLE_TYPO", "children": []})
        self.assertTrue(any("unknown_op" in v for v in G.validate(g, self.reg)))


class StructureTests(unittest.TestCase):
    def test_walk_paths_use_assemble_label_and_are_unique(self):
        g = G.genotype(G.ASSEMBLE(_attempt(), _attempt(), G.STOP("x")))
        paths = [p for p, _ in G.walk(g["root"])]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn("root/assemble[0]", paths)
        self.assertIn("root/assemble[2]", paths)

    def test_cost_upper_bound_sums_all_slots(self):
        """Сумма, не максимум: ВСЕ слоты всегда исполняются (нет ленивого выхода,
        в отличие от PAR, где сумма — лишь худший случай)."""
        node = G.ASSEMBLE(_call(max_tokens=150), _call(max_tokens=150), _call(max_tokens=150))
        reg = FakeRegistry(max_input_tokens=100)
        self.assertEqual(G.cost_upper_bound(node, reg), 3 * (100 + 150))

    def test_depth_and_n_nodes_count_assemble_like_seq(self):
        g = G.genotype(G.ASSEMBLE(_attempt(), _attempt()))
        self.assertEqual(G.n_nodes(g["root"]), 1 + 2 * 3)   # ASSEMBLE + 2x(SEQ+CALL+CHECK)
        self.assertEqual(G.depth(g["root"]), 3)


class ExecutionSemanticsTests(unittest.TestCase):
    """Расходится с PAR: RESOLVED только если подтверждены ВСЕ слоты."""

    def test_resolved_iff_all_slots_confirmed(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["OK", "OK"])
        g = G.genotype(G.ASSEMBLE(_attempt(), _attempt()))
        res = R.run(g, _state(), reg, be, 10_000)
        self.assertEqual(res.outcome, R.RESOLVED)
        self.assertEqual(len(be.calls), 2)

    def test_first_slot_only_does_not_resolve(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["OK", "no"])
        res = R.run(G.genotype(G.ASSEMBLE(_attempt(), _attempt())), _state(), reg, be, 10_000)
        self.assertNotEqual(res.outcome, R.RESOLVED)

    def test_last_slot_only_does_not_resolve(self):
        """Критический случай упорядочивания: если бы исход читался по статусу
        ПОСЛЕДНЕГО слота (а не по агрегату), это ложно дало бы RESOLVED."""
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["no", "OK"])
        res = R.run(G.genotype(G.ASSEMBLE(_attempt(), _attempt())), _state(), reg, be, 10_000)
        self.assertNotEqual(res.outcome, R.RESOLVED)

    def test_untrusted_slot_pass_does_not_count_as_confirmed(self):
        """ГАРАНТИЯ 5 переносится на агрегат: слот с недоверенным PASS не должен
        давать RESOLVED всему ASSEMBLE."""
        reg = FakeRegistry(trusted=False, max_input_tokens=100)
        be = ScriptedBackend(["OK", "OK"])
        res = R.run(G.genotype(G.ASSEMBLE(_attempt(), _attempt())), _state(), reg, be, 10_000)
        self.assertNotEqual(res.outcome, R.RESOLVED)

    def test_cost_sums_across_all_slots(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["no", "no", "no"])
        g = G.genotype(G.ASSEMBLE(_attempt(), _attempt(), _attempt()))
        res = R.run(g, _state(), reg, be, 10_000)
        self.assertEqual(res.cost, 300)
        self.assertEqual(len(be.calls), 3)

    def test_all_slots_run_even_after_an_early_failure(self):
        """В отличие от PAR (ленивый выход по первому PASS), ASSEMBLE не выходит
        рано на провале — каждый слот отвечает за СВОЙ шаг, не за альтернативу."""
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["no", "OK"])
        R.run(G.genotype(G.ASSEMBLE(_attempt(), _attempt())), _state(), reg, be, 10_000)
        self.assertEqual(len(be.calls), 2)


class StepNarrowingTests(unittest.TestCase):
    """SPEC.md §31: слот i видит `state.task` заужённым на шаг i — без новых полей
    State (решение зафиксировано: шаг подставляется в уже существующее поле `task`)."""

    def test_each_slot_sees_its_own_step(self):
        seen = []

        class Recorder(ScriptedBackend):
            def call(self, **kw):
                seen.append(dict(kw["state"].task))
                return super().call(**kw)

        reg, be = FakeRegistry(max_input_tokens=100), Recorder(["OK", "OK"])
        R.run(G.genotype(G.ASSEMBLE(_attempt(), _attempt())), _state(), reg, be, 10_000)
        self.assertEqual([s["step"] for s in seen], [{"kind": "a"}, {"kind": "b"}])
        self.assertEqual([s["step_index"] for s in seen], [0, 1])

    def test_slots_do_not_share_artifact_or_evidence(self):
        """Каждый слот стартует с ПУСТЫМ артефактом (шаг — независимый вопрос, а
        не продолжение предыдущего слота), в отличие от PAR, где ветви делят
        входной артефакт как альтернативные попытки той же цели."""
        seen = []

        class Recorder(ScriptedBackend):
            def call(self, **kw):
                seen.append(kw["state"].artifact)
                return super().call(**kw)

        reg, be = FakeRegistry(max_input_tokens=100), Recorder(["first", "second"])
        g = G.genotype(G.ASSEMBLE(_attempt(), _attempt()))
        R.run(g, _state(artifact="INITIAL"), reg, be, 10_000)
        self.assertEqual(seen, ["", ""], "слот не должен видеть артефакт извне ASSEMBLE")

    def test_extra_children_beyond_step_count_still_execute_unnarrowed(self):
        """Если слотов больше, чем шагов задачи, лишние слоты просто не получают
        заужения (`state.task` не подменяется) — не падают и не пропускаются."""
        seen = []

        class Recorder(ScriptedBackend):
            def call(self, **kw):
                seen.append(kw["state"].task.get("step_index"))
                return super().call(**kw)

        reg, be = FakeRegistry(max_input_tokens=100), Recorder(["OK", "OK", "OK"])
        g = G.genotype(G.ASSEMBLE(_attempt(), _attempt(), _attempt()))
        res = R.run(g, _state(), reg, be, 10_000)
        self.assertEqual(seen, [0, 1, None])
        self.assertEqual(len(be.calls), 3)


class StopAndBudgetTests(unittest.TestCase):
    """STOP наследует решение PAR (SPEC.md §1.2): завершает весь прогон, не только
    свой слот. BUDGET-рамка держится и внутри ASSEMBLE (ГАРАНТИЯ 1 не локальна)."""

    def test_stop_in_a_slot_ends_the_whole_run(self):
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["no"])
        g = G.genotype(G.ASSEMBLE(G.SEQ(_call(), G.CHECK("seam.s1"), G.STOP("give_up")),
                                  _attempt()))
        res = R.run(g, _state(), reg, be, 10_000)
        self.assertEqual(len(be.calls), 1, "второй слот не должен исполниться после STOP")
        self.assertEqual(res.stop_reason, "give_up")
        self.assertNotEqual(res.outcome, R.RESOLVED)

    def test_earlier_slot_pass_does_not_leak_through_a_later_stop(self):
        """РЕГРЕССИЯ (найдена на кампании a1, ДО публикации любого результата):
        первая редакция дописывала evidence каждого слота в `merged` СРАЗУ внутри
        цикла. Слот 0 честно PASS, слот 1 -- голый STOP (M3 `to_stop` строит именно
        такие генотипы): `merged.evidence` заканчивался PASS'ом слота 0, и
        `_last_seam`/`run()` объявляли RESOLVED всему ASSEMBLE, хотя вычисление
        оборвано ДО решения. Обнаружено на реальном прогоне (r=1.000 при cost=170 --
        ниже стоимости ЛЮБОГО валидного маршрута)."""
        reg, be = FakeRegistry(max_input_tokens=100), ScriptedBackend(["OK"])
        g = G.genotype(G.ASSEMBLE(_attempt(), G.STOP("barrier_too_high"), _attempt()))
        res = R.run(g, _state(), reg, be, 10_000)
        self.assertEqual(len(be.calls), 1, "слот 0 честно прошёл, слот 2 не должен исполниться")
        self.assertEqual(res.stop_reason, "barrier_too_high")
        self.assertNotEqual(res.outcome, R.RESOLVED,
                            "PASS слота 0 не должен просачиваться как исход всего ASSEMBLE")

    def test_budget_frame_binds_across_slots(self):
        """Рамка 250 (одна консервативная оценка 100+150) пропускает ровно один
        слот, второй отклоняется до вызова — тот же механизм, что для PAR."""
        reg = FakeRegistry(max_input_tokens=100)
        be = ScriptedBackend(["no", "no"])
        g = G.genotype(G.BUDGET(250, G.ASSEMBLE(_attempt(), _attempt())))
        res = R.run(g, _state(), reg, be, budget_tokens=100_000)
        self.assertEqual(len(be.calls), 1)
        self.assertEqual(res.cost, 100)
        self.assertTrue(res.exhausted)


class M3ReachesSlotsTests(unittest.TestCase):
    """Плановое основание: откат внутри слота (PAR с несколькими ветвями) должен
    быть достижим существующими M1-M6 БЕЗ нового оператора мутации -- потому что
    ASSEMBLE стоит в одной ветке с SEQ/PAR в `mutate.addresses()` (SPEC.md §31/§33).
    Проверяется механизм напрямую (addresses/set_at), а не через недетерминированный
    `mutate.mutate()`: тот на каждый вызов возвращает ПЕРВУЮ успешную мутацию из
    40 попыток, и высоковероятные операторы (M1) почти всегда опережают редкое
    попадание M3 именно на слот ASSEMBLE -- это шум выбора оператора, не свойство
    механизма адресации, которое здесь и проверяется."""

    def test_addresses_exposes_each_slot_directly_under_assemble(self):
        root = G.ASSEMBLE(_attempt(molecule="gen.a"), _attempt(molecule="gen.b"))
        addrs = M.addresses(root)
        for i in range(2):
            match = [(a, n) for a, n in addrs if a == [("children", i)]]
            self.assertEqual(len(match), 1)
            self.assertEqual(match[0][1]["op"], "SEQ")

    def test_wrap_par_then_add_par_branch_deepen_a_slot_in_place(self):
        """Ровно операторы M3 (`wrap_par`, `add_par_branch`), не тронутые этим
        циклом, применённые напрямую через `addresses`/`set_at` -- те же функции,
        что использует `m3_structural` изнутри."""
        slot0, slot1 = _attempt(molecule="gen.a"), _attempt(molecule="gen.b")
        root = G.ASSEMBLE(slot0, slot1)

        addr, node = [(a, n) for a, n in M.addresses(root) if a == [("children", 1)]][0]
        widened = M.set_at(root, addr, G.PAR(node, node))          # wrap_par
        self.assertEqual(widened["op"], "ASSEMBLE")
        self.assertEqual(widened["children"][0]["op"], "SEQ")      # слот 0 не тронут
        self.assertEqual(widened["children"][1]["op"], "PAR")

        par_addr, par_node = [(a, n) for a, n in M.addresses(widened)
                              if n.get("op") == "PAR"][0]
        deepened = M.set_at(widened, par_addr,
                            dict(par_node, children=list(par_node["children"]) + [node]))
        self.assertEqual(len(deepened["children"][1]["children"]), 3)  # add_par_branch
        self.assertEqual(G.validate(G.genotype(deepened), FakeRegistry(max_input_tokens=100)), [])


class RandomGenotypeGateTests(unittest.TestCase):
    """SPEC.md §33: `assemble_n_steps` в ctx -- единственный переключатель. Без него
    поведение `_random_node` побайтово прежнее (существующие 25 случаев
    `test_layer.py::test_random_genotypes_are_valid` гоняются на реестрах MSARITH/
    кода, которые этот флаг никогда не ставят)."""

    def _ctx(self, **extra):
        from tests.fakes import GRID
        ctx = {"generators": ["gen.m1", "gen.m2"], "observables": ["task_family"],
               "feasible_params": GRID, "donors": [], "composites": []}
        ctx.update(extra)
        return ctx

    def test_flag_absent_never_produces_assemble(self):
        import random
        reg = FakeRegistry(max_input_tokens=100)
        rng = random.Random(20260819)
        for _ in range(200):
            g, _ = M.random_genotype(rng, self._ctx(), reg, gen=0)
            if g is not None:
                self.assertNotIn("ASSEMBLE", {n.get("op") for _, n in G.walk(g["root"])})

    def test_flag_present_eventually_produces_a_valid_assemble(self):
        import random
        reg = FakeRegistry(max_input_tokens=100)
        rng = random.Random(20260819)
        found = False
        for _ in range(500):
            g, _ = M.random_genotype(rng, self._ctx(assemble_n_steps=4), reg, gen=0)
            if g is not None and g["root"].get("op") == "ASSEMBLE":
                self.assertEqual(len(g["root"]["children"]), 4)
                self.assertEqual(G.validate(g, reg), [])
                found = True
                break
        self.assertTrue(found, "ASSEMBLE ни разу не сгенерирован за 500 попыток с флагом")


if __name__ == "__main__":
    unittest.main()
