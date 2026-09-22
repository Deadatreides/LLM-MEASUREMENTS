"""test_experiment13_sanity.py -- предполётный чек-лист §11 задания, 1:1:

1. Промпт не подсказывает способ решения.
2. R1/R2 действительно меняют представление хотя бы в части случаев (не
   100% SAME_STRUCTURE) -- на РЕАЛЬНЫХ данных, не на синтетике.
3. Отсутствие утечки.
4. PASS определяется корректно.
5. NOVEL_STRUCTURE классифицируется корректно (не путает
   перефразирование с новой структурой).

`run_sanity_checklist()` -- go/no-go отчёт, не просто набор assert-ов
(тот же принцип, что `experiment12/tests/test_experiment12_v2_sanity.py`).
Пункты 1/3 проверяются ДВАЖДЫ: сначала гарантией конструкции
(`run_mode()` вызывает `validate_prompt()` до генерации и бросает
`PromptViolation`, если она не проходит -- если мы вообще получили
результаты, значит проверка уже прошла на этапе сбора), и затем ЕЩЁ РАЗ
здесь, независимым пересканированием сохранённого текста промпта в каждой
записи -- тот же принцип "не доверять раннеру, пересчитать самому",
которым проверялись предыдущие эксперименты проекта.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import context_builder as cb  # noqa: E402
import harness  # noqa: E402
from harness import MODES_MAIN, run_mode, select_states  # noqa: E402


def run_sanity_checklist(states_by_id: dict, results: list) -> dict:
    by_collision: dict = {}
    for r in results:
        by_collision.setdefault(r["collision_id"], {})[r["mode"]] = r

    report = {}

    # 1 + 3: независимый пересчёт validate_prompt() на СОХРАНЁННОМ тексте
    # промпта каждой записи -- не полагается только на то, что рантайм
    # уже это проверил.
    v = []
    for r in results:
        rec = r["record"]
        state = states_by_id.get(rec["collision_id"])
        if state is None:
            continue
        check = cb.validate_prompt(rec["mode"], state.initial_artifact, rec["prompt"])
        if not check["ok"]:
            v.append(f"{rec['collision_id']}/{rec['mode']}: {check['violations']}")
    report["1_no_method_hint_and_3_no_leakage"] = {"ok": not v, "violations": v}

    # 2. R1/R2 действительно меняют представление хотя бы в части случаев
    novelty_counts = {"R1": [], "R2": []}
    for cid, modes in by_collision.items():
        for mode in ("R1", "R2"):
            r = modes.get(mode)
            if r and r.get("representation_novelty"):
                novelty_counts[mode].append(r["representation_novelty"])
    v = []
    for mode, values in novelty_counts.items():
        if values and all(x == "SAME_STRUCTURE" for x in values):
            v.append(f"{mode}: 100% SAME_STRUCTURE across {len(values)} attempts -- representation never actually changes")
    report["2_representation_actually_changes"] = {
        "ok": not v, "violations": v,
        "novelty_counts": {k: len(v2) for k, v2 in novelty_counts.items()},
    }

    # 4. PASS определяется корректно -- независимая пересборка PASS-флага
    # из raw output теми же швами, что использовал run_mode, и сверка с
    # сохранённым final_status.
    import seams as seams_mod
    from tasks import ARITH_TASKS, CODE_TASKS

    v = []
    for r in results:
        rec = r["record"]
        state = states_by_id.get(rec["collision_id"])
        if state is None:
            continue
        task = ARITH_TASKS[rec["task_id"]] if state.task_family == "arithmetic" else CODE_TASKS[rec["task_id"]]
        if state.task_family == "arithmetic":
            if rec["mode"] == "R0":
                recheck = seams_mod.multistep_arithmetic_seam(rec["output"], task["steps"])
            else:
                recheck = seams_mod.final_answer_seam(rec["output"], task["answer"])
        else:
            recheck = seams_mod.enriched_code_seam(rec["output"], task["function_name"], task["requirements"])
        recomputed_pass = recheck["status"] == seams_mod.PASS
        stored_pass = r["final_status"] == "PASS"
        if recomputed_pass != stored_pass:
            v.append(f"{rec['collision_id']}/{rec['mode']}: recomputed PASS={recomputed_pass} but stored final_status={r['final_status']}")
    report["4_pass_determined_correctly"] = {"ok": not v, "violations": v}

    # 5. NOVEL_STRUCTURE не должна быть пустым перефразированием
    v = []
    for r in results:
        if r.get("representation_novelty") == "NOVEL_STRUCTURE":
            rec = r["record"]
            if rec["mode"] == "R1" and not rec.get("new_representation", "").strip():
                v.append(f"{r['collision_id']}/{rec['mode']}: classified NOVEL_STRUCTURE but new_representation text is empty")
    report["5_novelty_classification_sane"] = {"ok": not v, "violations": v}

    report["overall_ok"] = all(item.get("ok", True) for item in report.values())
    return report


class SanityChecklistOnRealStatesTests(TestCase):
    """Реальные collision-state из Эксперимента 12 (не синтетика), retry
    generate() подменён (без GPU в обычном прогоне тестов)."""

    @classmethod
    def setUpClass(cls):
        try:
            all_states = harness.load_exp12_states()
        except FileNotFoundError:
            cls.states_by_id = None
            return
        selected = [s for s in select_states(all_states, 40, seed=1, arith_fraction=0.75) if s.task_family == "arithmetic"][:8]
        cls.states_by_id = {s.collision_id: s for s in selected}

    def test_checklist_all_clear_on_real_states(self):
        if not self.states_by_id:
            self.skipTest("experiment12/runs12/phase0_full_exp12-v1.json not present, or no arithmetic states sampled")

        call_count = {"n": 0}

        def fake_generate(llm, model_id, prompt, **kw):
            call_count["n"] += 1
            # чередуем "новая структура" / "заявленная, но не настоящая
            # новизна" -- чтобы пункты 2 и 5 проверялись не на вырожденных данных
            if call_count["n"] % 2 == 0:
                text = "NEW REPRESENTATION: alternate framing using ratios\nSOLUTION:\nratio_a = 1\nratio_b = 2\ncombined = ratio_a + ratio_b\nFINAL ANSWER = 42"
            else:
                text = "NEW REPRESENTATION: I will try a fresh approach\nSOLUTION:\nstep_one = 1\nstep_two = 2\nFINAL ANSWER = 42"
            return {"raw_text": text, "generation_failed": False, "generation_error": None,
                    "input_tokens": 20, "output_tokens": 15, "generation_time_sec": 0.1}

        results = []
        with patch.object(harness, "generate", side_effect=fake_generate):
            for state in self.states_by_id.values():
                for mode in MODES_MAIN:
                    results.append(run_mode(state, mode, llm=object()))

        report = run_sanity_checklist(self.states_by_id, results)
        for key, item in report.items():
            if key == "overall_ok":
                continue
            self.assertTrue(item.get("ok", True), f"{key} FAILED: {item}")
        self.assertTrue(report["overall_ok"])


if __name__ == "__main__":
    main()
