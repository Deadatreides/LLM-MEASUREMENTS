"""test_experiment12_v2_sanity.py -- предполётный чек-лист v2, 1:1 с
"ОБЯЗАТЕЛЬНЫЙ SANITY CHECK" из задания:

1. K0 действительно не содержит структуры.
2. K1 действительно содержит структуру, но не подтверждённые значения.
3. K2 действительно содержит только механически подтверждённые значения.
4. O0 и O1 отличаются только моделью (промпт побайтово идентичен).
5. Ни один вариант не получает информацию из будущего (context manifest).
6. Один и тот же collision_id действительно присутствует во всех 7 плечах
   (целостность парности).

"Если хотя бы один пункт нарушается -- не запускать основной эксперимент."
-- `run_sanity_checklist()` реализует это как явный go/no-go отчёт, а не
просто россыпь assert-ов, чтобы им можно было пользоваться и вне
unittest (например, как шаг CLI перед фазой 1 на реальных данных пилота).

`SanityChecklistOnRealPilotStatesTests` прогоняет чек-лист на РЕАЛЬНЫХ
collision-state из пилота эксп.12 v1 (`runs12/phase0_pilot_exp12-v1.json`
-- сырые артефакты и механические проверки настоящие, сгенерированы
реальными моделями) через настоящий `run_arm()` по всем 7 плечам, с
подменённым `generate()` (чтобы не требовать GPU в обычном прогоне
тестов) -- это ближе к духу задания ("взять несколько collision-state"),
чем чисто синтетические данные.
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import context_manifest as cm  # noqa: E402
import context_builder as cb  # noqa: E402
import harness  # noqa: E402
from configs.model_registry import MODEL_IDS  # noqa: E402
from harness import ALL_ARMS, ArmConfig, CollisionState, build_arm_configs, run_arm  # noqa: E402
from tasks.arithmetic_multistep_tasks import TASKS as ATASKS  # noqa: E402
from tasks.code_tasks import CODE_TASKS  # noqa: E402

_STRUCTURE_MARKERS = ("Structure of this problem", "Behavioral requirements this function must satisfy")


def _get_task(task_family: str, task_id: str) -> dict:
    return ATASKS[task_id] if task_family == "arithmetic" else CODE_TASKS[task_id]


def run_sanity_checklist(states: list, arm_results: list) -> dict:
    """Возвращает {"1_k0_no_structure": {"ok":bool,"violations":[...]}, ...,
    "overall_ok": bool}. Не бросает исключений -- вызывающий код решает,
    останавливать ли кампанию (тот же принцип, что LeakDetected в
    harness.py, но здесь это предполётная, а не рантайм-проверка)."""
    by_collision: dict = {}
    for r in arm_results:
        by_collision.setdefault(r["collision_id"], {})[r["arm"]] = r
    state_by_id = {s.collision_id: s for s in states}

    report = {}

    # 1. K0 не содержит структуры
    v = []
    for cid, arms in by_collision.items():
        for arm_id in ("K0O0", "K0O1", "K0O0_promptB"):
            r = arms.get(arm_id)
            if r and any(m in r["record"]["prompt"] for m in _STRUCTURE_MARKERS):
                v.append(f"{cid}/{arm_id}: K0-level prompt contains a structural marker")
    report["1_k0_no_structure"] = {"ok": not v, "violations": v}

    # 2. K1 содержит структуру, но НЕ содержит числовых значений шагов/requirements
    v = []
    for cid, arms in by_collision.items():
        state = state_by_id[cid]
        task = _get_task(state.task_family, state.task_id)
        for arm_id in ("K1O0", "K1O1"):
            r = arms.get(arm_id)
            if not r:
                continue
            prompt = r["record"]["prompt"]
            if not any(m in prompt for m in _STRUCTURE_MARKERS):
                v.append(f"{cid}/{arm_id}: K1 prompt missing structural marker")
            s_text = cb.structure_text("K1", state.task_family, task, state.initial_seam_result)
            forbidden = cb.forbidden_values_for_level("K1", state.task_family, task, state.initial_seam_result)
            if not cb.check_no_leak(s_text, forbidden):
                v.append(f"{cid}/{arm_id}: K1 structure text contains a forbidden numeric value")
    report["2_k1_structure_no_values"] = {"ok": not v, "violations": v}

    # 3. K2 содержит ТОЛЬКО подтверждённые значения
    v = []
    for cid, arms in by_collision.items():
        state = state_by_id[cid]
        task = _get_task(state.task_family, state.task_id)
        for arm_id in ("K2O0", "K2O1"):
            r = arms.get(arm_id)
            if not r:
                continue
            s_text = cb.structure_text("K2", state.task_family, task, state.initial_seam_result)
            forbidden = cb.forbidden_values_for_level("K2", state.task_family, task, state.initial_seam_result)
            if not cb.check_no_leak(s_text, forbidden):
                v.append(f"{cid}/{arm_id}: K2 structure text leaks an unconfirmed value")
    report["3_k2_only_confirmed_values"] = {"ok": not v, "violations": v}

    # 4. O0/O1 промпт побайтово идентичен внутри уровня
    v = []
    for cid, arms in by_collision.items():
        for level in ("K0", "K1", "K2"):
            a, b = arms.get(f"{level}O0"), arms.get(f"{level}O1")
            if a and b and a["record"]["prompt"] != b["record"]["prompt"]:
                v.append(f"{cid}/{level}: O0 and O1 prompts differ")
    report["4_o0_o1_prompt_identical"] = {"ok": not v, "violations": v}

    # 5. Ни один вариант не получает информацию из будущего (context manifest)
    v = []
    for cid, arms in by_collision.items():
        state = state_by_id[cid]
        task = _get_task(state.task_family, state.task_id)
        for arm_id, r in arms.items():
            level = r["level"]
            result = cm.validate_prompt_against_manifest(level, state.task_family, task, state.initial_artifact, state.initial_seam_result, r["record"]["prompt"])
            if not result["ok"]:
                v.append(f"{cid}/{arm_id}: {result['violations']}")
    report["5_no_future_leakage"] = {"ok": not v, "violations": v}

    # 6. Один и тот же collision_id присутствует во всех 7 плечах
    v = []
    for cid, arms in by_collision.items():
        missing = set(ALL_ARMS) - set(arms.keys())
        if missing:
            v.append(f"{cid}: missing arms {sorted(missing)}")
    report["6_pairing_integrity"] = {"ok": not v, "violations": v}

    report["overall_ok"] = all(report[k]["ok"] for k in report)
    return report


class SanityChecklistOnRealPilotStatesTests(TestCase):
    """Использует НАСТОЯЩИЕ collision-state из runs12/phase0_pilot_exp12-v1.json
    (реальные модели, реальные первичные генерации) -- generate() на
    retry-шаге подменён, чтобы не требовать GPU в обычном тестовом
    прогоне; сама сборка промптов и вся проверочная логика -- настоящая."""

    @classmethod
    def setUpClass(cls):
        pilot_path = Path(__file__).resolve().parent.parent / "runs12" / "phase0_pilot_exp12-v1.json"
        if not pilot_path.exists():
            cls.states = None
            return
        with open(pilot_path, encoding="utf-8") as f:
            data = json.load(f)
        raw_states = data["states"]
        arith = [s for s in raw_states if s["task_family"] == "arithmetic"][:2]
        code = [s for s in raw_states if s["task_family"] == "code"][:2]
        cls.states = [CollisionState(**s) for s in (arith + code)]

    def test_checklist_all_clear_on_real_states_with_mocked_retry(self):
        if not self.states:
            self.skipTest("runs12/phase0_pilot_exp12-v1.json not present -- run phase 0 pilot first")

        def fake_generate(llm, model_id, prompt, **kw):
            # эхо исходного артефакта -- не важно ЧТО отвечает модель для
            # цели этого теста (проверяется КОНТЕКСТ, не корректность
            # ремонта), лишь бы generate() не требовал GPU.
            return {"raw_text": "unchanged", "generation_failed": False, "generation_error": None,
                    "input_tokens": 10, "output_tokens": 5, "generation_time_sec": 0.05}

        config_rng = random.Random(1)
        arm_results = []
        with patch.object(harness, "generate", side_effect=fake_generate):
            for state in self.states:
                configs = build_arm_configs(state.model_id, list(MODEL_IDS), config_rng)
                for arm_id, config in configs.items():
                    arm_results.append(run_arm(state, config, llm=object()))

        report = run_sanity_checklist(self.states, arm_results)
        for key, item in report.items():
            if key == "overall_ok":
                continue
            self.assertTrue(item["ok"], f"{key} FAILED: {item['violations']}")
        self.assertTrue(report["overall_ok"])


if __name__ == "__main__":
    main()
