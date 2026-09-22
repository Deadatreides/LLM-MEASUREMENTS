"""Контрольные случаи для src/orchestrator.py.

Не входит буквально в список ВЫХОД этапа 6 в WORK_PLAN.md (там только
orchestrator.py/tasks/branch_tasks.py/run_task.py) — добавлен по жёсткому
правилу 3 из arch1/CLAUDE.md: «модуль без прошедших контрольных случаев
считается неготовым», оно распространяется на все модули проекта.

Критерий готовности этапа 6 (WORK_PLAN.md): одна задача проходит цикл
целиком (генерация -> швы -> состояния -> локализация -> план -> ремонт
-> повторная проверка -> STOP с причиной), граф восстановим из
хранилища -- проверяется здесь автоматически, а не только вручную через
run_task.py.
"""

import re
import unittest
from pathlib import Path

from src.action_executor import ActionExecutor
from src.evidence_engine import EvidenceEngine
from src.graph_core import GraphCore
from src.model_adapter import Capabilities, RawResult
from src.orchestrator import run_task
from src.repair_planner import COLLISION_RESOLVED, EXTERNAL_EVIDENCE_REQUIRED, MAX_ITERATIONS, RepairPlanner
from src.seam_engine import default_seam_engine
from src.state_engine import CORRECT, GENERATED, StateEngine
from src.storage import Storage
from tasks.branch_tasks import CLAIM_TYPE, CLAIMS, DEPENDENCY_EDGES, TASKS, generation_prompt, oracle_values

_CLAIM_MARKER_RE = re.compile(r"\[(C\d_[A-Z_]+)\]")


class ScriptedModelAdapter:
    """Отвечает по claim_id, извлечённому из маркера [claim_id], который
    generation_prompt() всегда вставляет в текст. responses[claim_id] --
    очередь (text, failed, error), потребляется по одному за вызов: так
    можно задать РАЗНЫЙ ответ на первичный GENERATE и на REGENERATE того
    же claim'а."""

    def __init__(self, responses: dict, model_id: str = "fake-model"):
        self._responses = {k: list(v) for k, v in responses.items()}
        self._model_id = model_id

    def capabilities(self):
        return Capabilities(model_id=self._model_id, family="fake", context_capacity=4096)

    def generate(self, content, sampling):
        match = _CLAIM_MARKER_RE.search(content)
        claim_id = match.group(1) if match else None
        queue = self._responses.get(claim_id, [])
        text, failed, error = queue.pop(0) if queue else ("1", False, None)
        return RawResult(
            rendered_prompt=f"<rendered>{content}</rendered>",
            raw_output=text,
            input_tokens=10,
            output_tokens=5,
            wall_time_sec=0.001,
            failed=failed,
            error=error,
        )


def _correct_responses(task: dict) -> dict:
    return {
        "C1_ROOT": [(f"This covers {task['keywordA']} and {task['keywordB']} separately.", False, None)],
        "C2_METHOD_A": [(f"quantity {task['qtyA']}, rate {task['rateA']}", False, None)],
        "C3_METHOD_B": [(f"quantity {task['qtyB']}, rate {task['rateB']}", False, None)],
        "C4_SUBTOTAL_A": [(f"{task['qtyA']} * {task['rateA']} = {task['subtotalA']}", False, None)],
        "C5_SUBTOTAL_B": [(f"{task['qtyB']} * {task['rateB']} = {task['subtotalB']}", False, None)],
        "C6_FINAL_TOTAL": [(f"{task['subtotalA']} + {task['subtotalB']} = {task['final_total']}", False, None)],
    }


def _collision_responses(task: dict) -> dict:
    """Как _correct_responses, но C4 сначала отвечает заведомо неверно
    (первый GENERATE), затем верно (REGENERATE после ремонта)."""
    responses = _correct_responses(task)
    wrong_subtotal_a = task["subtotalA"] + 999
    responses["C4_SUBTOTAL_A"] = [
        (f"{task['qtyA']} * {task['rateA']} = {wrong_subtotal_a}", False, None),
        (f"{task['qtyA']} * {task['rateA']} = {task['subtotalA']}", False, None),
    ]
    return responses


def _wire(responses: dict):
    storage = Storage()
    graph = GraphCore(storage)
    state = StateEngine(storage, graph)
    seams = default_seam_engine(storage)
    for seam_id in ("seam:exec", "seam:format", "seam:numeric"):
        result = seams.self_test(seam_id)
        assert result["passed"], (seam_id, result["failed_cases"])
    evidence = EvidenceEngine(storage)
    planner = RepairPlanner(graph, state, seams)
    executor = ActionExecutor(graph, state, seams, evidence, storage, ScriptedModelAdapter(responses))
    return storage, graph, state, seams, evidence, planner, executor


def _run(task_id, responses):
    task = TASKS[task_id]
    storage, graph, state, seams, evidence, planner, executor = _wire(responses)
    result = run_task(
        task_id, task, CLAIMS, CLAIM_TYPE, DEPENDENCY_EDGES,
        generation_prompt, oracle_values,
        graph, state, seams, evidence, planner, executor, storage,
    )
    return storage, graph, state, result


class HappyPathTests(unittest.TestCase):
    def test_all_correct_resolves_without_collision(self):
        task = TASKS["BRANCH_01"]
        _storage, _graph, _state, result = _run("BRANCH_01", _correct_responses(task))

        self.assertEqual(result.stop_reason, COLLISION_RESOLVED)
        self.assertEqual(result.collisions_detected, 0)
        self.assertEqual(result.collisions_resolved, 0)

        for claim_id in ("C4_SUBTOTAL_A", "C5_SUBTOTAL_B", "C6_FINAL_TOTAL"):
            self.assertEqual(result.claim_states[claim_id], CORRECT)
        # C1-C3: нет применимого HARD-шва к формулировке метода -- честно
        # остаются на дефолтном GENERATED (переход в UNVERIFIED не
        # реализован, решение этапа 2 -- см. STATUS.md), не помечаются
        # как ошибочные.
        for claim_id in ("C1_ROOT", "C2_METHOD_A", "C3_METHOD_B"):
            self.assertEqual(result.claim_states[claim_id], GENERATED)


class ForcedCollisionTests(unittest.TestCase):
    def test_wrong_then_correct_regenerate_resolves_collision(self):
        task = TASKS["BRANCH_01"]
        storage, graph, _state, result = _run("BRANCH_01", _collision_responses(task))

        self.assertEqual(result.stop_reason, COLLISION_RESOLVED)
        self.assertEqual(result.collisions_detected, 1)
        self.assertEqual(result.collisions_resolved, 1)
        self.assertEqual(result.claim_states["C4_SUBTOTAL_A"], CORRECT)
        self.assertEqual(result.claim_states["C6_FINAL_TOTAL"], CORRECT)

        current = graph.get_claim("C4_SUBTOTAL_A")
        self.assertEqual(current.claim_version, 2)  # регенерация создала новую версию
        old = graph.get_claim("C4_SUBTOTAL_A", version=1)
        self.assertNotEqual(old.content, current.content)  # старая версия сохранена, не перезаписана

    def test_graph_restorable_from_storage_snapshot(self):
        task = TASKS["BRANCH_01"]
        storage, graph, _state, _result = _run("BRANCH_01", _collision_responses(task))

        first_c4_seq = min(
            r.seq
            for r in storage.query(
                lambda rec: rec.kind == "claim_version" and rec.payload.claim_id == "C4_SUBTOTAL_A"
            )
        )
        snapshot = graph.graph_snapshot(first_c4_seq)

        self.assertEqual(snapshot.get_claim("C4_SUBTOTAL_A").claim_version, 1)
        self.assertNotEqual(
            snapshot.get_claim("C4_SUBTOTAL_A").content,
            graph.get_claim("C4_SUBTOTAL_A").content,
        )


def _real_model_available() -> bool:
    model_path = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "Llama-3.2-1B-Instruct-Q4_0"
        / "Llama-3.2-1B-Instruct-Q4_0.gguf"
    )
    if not model_path.exists():
        return False
    try:
        import sys

        configs_dir = Path(__file__).resolve().parents[2] / "experiment9" / "configs"
        if str(configs_dir) not in sys.path:
            sys.path.insert(0, str(configs_dir))
        from env_fix import fix_cuda_dll_path

        fix_cuda_dll_path()
        import llama_cpp  # noqa: F401
    except Exception:
        return False
    return True


class RealModelSmokeTest(unittest.TestCase):
    """Реальный run_task на настоящей локальной GGUF -- мягкие проверки
    (валидный stop_reason, граф непуст), без требования, что маленькая
    модель посчитает арифметику верно."""

    @unittest.skipUnless(_real_model_available(), "llama_cpp / local GGUF model not available")
    def test_real_run_task_reaches_valid_stop(self):
        from src.model_adapter import ModelAdapter

        task_id = "BRANCH_01"
        task = TASKS[task_id]
        storage = Storage()
        graph = GraphCore(storage)
        state = StateEngine(storage, graph)
        seams = default_seam_engine(storage)
        for seam_id in ("seam:exec", "seam:format", "seam:numeric"):
            seams.self_test(seam_id)
        evidence = EvidenceEngine(storage)
        planner = RepairPlanner(graph, state, seams)
        model = ModelAdapter("llama-3.2-1b-instruct-q4_0")
        executor = ActionExecutor(graph, state, seams, evidence, storage, model_adapter=model)

        result = run_task(
            task_id, task, CLAIMS, CLAIM_TYPE, DEPENDENCY_EDGES,
            generation_prompt, oracle_values,
            graph, state, seams, evidence, planner, executor, storage,
        )

        valid_reasons = {COLLISION_RESOLVED, MAX_ITERATIONS, EXTERNAL_EVIDENCE_REQUIRED}
        self.assertTrue(
            result.stop_reason in valid_reasons or result.stop_reason.startswith("GENERATE_FAILED"),
            result.stop_reason,
        )
        self.assertGreater(result.graph_version, 0)
        self.assertEqual(len(result.claim_states), len(CLAIMS))


if __name__ == "__main__":
    unittest.main()
