"""Контрольные случаи для src/action_executor.py и src/model_adapter.py.

Критерий готовности (WORK_PLAN.md, Этап 5): одиночный GENERATE
отрабатывает на локальной модели, сырьё лежит на диске, actual_cost
заполнен -- проверяется RealModelSmokeTest (пропускается автоматически,
если llama_cpp/файл модели недоступны; на машине разработки -- не
пропускается).

Основная масса тестов -- на FakeModelAdapter (без GPU, мгновенные):
предусловия всех четырёх действий, сохранение RUN независимо от исхода,
actual_cost по факту, GENERATE/REGENERATE/VERIFY/STOP.
"""

import unittest
from pathlib import Path

from src.action_executor import (
    COMPLETED,
    EMPTY_OUTPUT,
    FAILED,
    GENERATE,
    GENERATION_FAILED,
    REGENERATE,
    STOP,
    VERIFY,
    ActionExecutor,
)
from src.context_builder import build_context
from src.evidence_engine import EvidenceEngine
from src.graph_core import GraphCore
from src.model_adapter import Capabilities, RawResult
from src.seam_engine import default_seam_engine
from src.state_engine import (
    CORRECT,
    INCORRECT,
    REFUTES,
    STRENGTH_HARD,
    Evidence,
    StateEngine,
)
from src.storage import Sampling, Storage, new_id


class FakeModelAdapter:
    """Не наследует ModelAdapter (тот лениво грузит настоящую модель в
    конструкторе) -- только тот же интерфейс: capabilities()/generate()."""

    def __init__(self, responses=None, model_id="fake-model"):
        self._model_id = model_id
        self._responses = list(responses) if responses is not None else None

    def capabilities(self):
        return Capabilities(model_id=self._model_id, family="fake", context_capacity=4096)

    def generate(self, content, sampling):
        if self._responses:
            text, failed, error = self._responses.pop(0)
        else:
            text, failed, error = "42", False, None
        return RawResult(
            rendered_prompt=f"<rendered>{content}</rendered>",
            raw_output=text,
            input_tokens=10,
            output_tokens=5,
            wall_time_sec=0.01,
            failed=failed,
            error=error,
        )


def _base_fixture():
    storage = Storage()
    graph = GraphCore(storage)
    state = StateEngine(storage, graph)
    seams = default_seam_engine(storage)
    for seam_id in ("seam:exec", "seam:format", "seam:numeric"):
        result = seams.self_test(seam_id)
        assert result["passed"], (seam_id, result["failed_cases"])
    evidence = EvidenceEngine(storage)

    art_version = graph.add_artifact_version(artifact_type="DERIVATION", task_id="task:t", content="x")
    artifact_id = storage.get(art_version).payload.artifact_id

    return storage, graph, state, seams, evidence, artifact_id


def _refute_evidence(claim_id):
    return Evidence(
        evidence_id=new_id("ev"),
        subject=claim_id,
        assertion=REFUTES,
        evidence_type="MECHANICAL",
        strength=STRENGTH_HARD,
        source={"kind": "test"},
        method="test",
    )


class PreconditionTests(unittest.TestCase):
    def setUp(self):
        self.storage, self.graph, self.state, self.seams, self.evidence, self.artifact_id = _base_fixture()
        self.executor = ActionExecutor(self.graph, self.state, self.seams, self.evidence, self.storage)

    def test_generate_ok_when_claim_absent(self):
        action = self.executor.build_action(GENERATE, "claim:new")
        ok, violations = self.executor.check_preconditions(action)
        self.assertTrue(ok)
        self.assertEqual(violations, [])

    def test_generate_fails_when_claim_already_exists(self):
        self.graph.add_claim_version(
            artifact_id=self.artifact_id, content="x", claim_type="VALUE", claim_id="claim:X"
        )
        action = self.executor.build_action(GENERATE, "claim:X")
        ok, violations = self.executor.check_preconditions(action)
        self.assertFalse(ok)

    def test_regenerate_requires_repairable_state(self):
        self.graph.add_claim_version(
            artifact_id=self.artifact_id, content="x", claim_type="VALUE", claim_id="claim:X"
        )
        action = self.executor.build_action(REGENERATE, "claim:X")

        ok, _ = self.executor.check_preconditions(action)
        self.assertFalse(ok)  # состояние по умолчанию GENERATED -- не входит в repairable set

        self.state.apply_evidence("claim:X", _refute_evidence("claim:X"))
        ok, _ = self.executor.check_preconditions(action)
        self.assertTrue(ok)  # теперь INCORRECT

    def test_verify_requires_trusted_applicable_seam(self):
        self.graph.add_claim_version(
            artifact_id=self.artifact_id, content="no numbers here", claim_type="OTHER", claim_id="claim:X"
        )
        action = self.executor.build_action(VERIFY, "claim:X")
        ok, violations = self.executor.check_preconditions(action)
        self.assertFalse(ok)  # claim_type OTHER -- ни один встроенный шов не применим

    def test_stop_always_ok(self):
        action = self.executor.build_action(STOP, "claim:whatever")
        ok, violations = self.executor.check_preconditions(action)
        self.assertTrue(ok)


class GenerationExecutionTests(unittest.TestCase):
    def setUp(self):
        self.storage, self.graph, self.state, self.seams, self.evidence, self.artifact_id = _base_fixture()

    def test_generate_happy_path_persists_run_and_creates_claim(self):
        model = FakeModelAdapter(responses=[("2 * 20 = 40", False, None)])
        executor = ActionExecutor(self.graph, self.state, self.seams, self.evidence, self.storage, model)

        action = executor.build_action(GENERATE, "claim:X")
        context = build_context(GENERATE, "claim:X", self.graph, task_statement="compute")
        result = executor.execute(action, context, artifact_id=self.artifact_id, claim_type="VALUE")

        self.assertEqual(result.status, COMPLETED)
        self.assertIsNone(result.failure_mode)
        self.assertEqual(result.actual_cost.tokens, 15)
        self.assertEqual(len(result.result_refs), 2)

        self.assertEqual(self.graph.get_claim("claim:X").content, "2 * 20 = 40")

        runs = [r.payload for r in self.storage.query(lambda rec: rec.kind == "run")]
        self.assertEqual(len(runs), 1)
        self.assertFalse(runs[0].failed)
        self.assertEqual(runs[0].rendered_prompt, "<rendered>[TASK_CONTEXT]\ncompute</rendered>")

    def test_run_persisted_even_when_generation_fails(self):
        model = FakeModelAdapter(responses=[("", True, "RuntimeError: boom")])
        executor = ActionExecutor(self.graph, self.state, self.seams, self.evidence, self.storage, model)

        action = executor.build_action(GENERATE, "claim:X")
        context = build_context(GENERATE, "claim:X", self.graph, task_statement="compute")
        result = executor.execute(action, context, artifact_id=self.artifact_id, claim_type="VALUE")

        self.assertEqual(result.status, FAILED)
        self.assertEqual(result.failure_mode, GENERATION_FAILED)

        runs = [r.payload for r in self.storage.query(lambda rec: rec.kind == "run")]
        self.assertEqual(len(runs), 1)  # сырьё сохранено, несмотря на провал
        self.assertTrue(runs[0].failed)

        with self.assertRaises(KeyError):
            self.graph.get_claim("claim:X")  # claim не создан

    def test_run_persisted_even_when_output_is_empty(self):
        model = FakeModelAdapter(responses=[("   ", False, None)])
        executor = ActionExecutor(self.graph, self.state, self.seams, self.evidence, self.storage, model)

        action = executor.build_action(GENERATE, "claim:X")
        context = build_context(GENERATE, "claim:X", self.graph, task_statement="compute")
        result = executor.execute(action, context, artifact_id=self.artifact_id, claim_type="VALUE")

        self.assertEqual(result.status, FAILED)
        self.assertEqual(result.failure_mode, EMPTY_OUTPUT)
        runs = [r.payload for r in self.storage.query(lambda rec: rec.kind == "run")]
        self.assertEqual(len(runs), 1)

    def test_regenerate_creates_new_version_preserving_old(self):
        self.graph.add_claim_version(
            artifact_id=self.artifact_id, content="2 * 20 = 999", claim_type="VALUE", claim_id="claim:X"
        )
        self.state.apply_evidence("claim:X", _refute_evidence("claim:X"))

        model = FakeModelAdapter(responses=[("2 * 20 = 40", False, None)])
        executor = ActionExecutor(self.graph, self.state, self.seams, self.evidence, self.storage, model)

        action = executor.build_action(REGENERATE, "claim:X")
        context = build_context(REGENERATE, "claim:X", self.graph, task_statement="compute")
        result = executor.execute(action, context)

        self.assertEqual(result.status, COMPLETED)
        current = self.graph.get_claim("claim:X")
        self.assertEqual(current.content, "2 * 20 = 40")
        self.assertEqual(current.claim_version, 2)

        old = self.graph.get_claim("claim:X", version=1)
        self.assertEqual(old.content, "2 * 20 = 999")  # старая версия не удалена


class VerifyExecutionTests(unittest.TestCase):
    def setUp(self):
        self.storage, self.graph, self.state, self.seams, self.evidence, self.artifact_id = _base_fixture()
        self.executor = ActionExecutor(self.graph, self.state, self.seams, self.evidence, self.storage)
        self.graph.add_claim_version(
            artifact_id=self.artifact_id, content="2 * 20 = 40", claim_type="VALUE", claim_id="claim:X"
        )

    def test_verify_pass_produces_hard_supporting_evidence_and_correct_state(self):
        action = self.executor.build_action(VERIFY, "claim:X")
        result = self.executor.execute(
            action, seam_id="seam:numeric", seam_inputs={"actual": 40, "expected": 40}
        )

        self.assertEqual(result.status, COMPLETED)
        self.assertEqual(self.state.current_state("claim:X"), CORRECT)
        recorded = self.evidence.evidence_for("claim:X")
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0].strength, STRENGTH_HARD)

    def test_verify_fail_produces_hard_refuting_evidence_and_incorrect_state(self):
        action = self.executor.build_action(VERIFY, "claim:X")
        result = self.executor.execute(
            action, seam_id="seam:numeric", seam_inputs={"actual": 40, "expected": 999}
        )

        self.assertEqual(result.status, COMPLETED)
        self.assertEqual(self.state.current_state("claim:X"), INCORRECT)


class StopExecutionTests(unittest.TestCase):
    def test_stop_has_zero_cost_and_completes(self):
        storage, graph, state, seams, evidence, _artifact_id = _base_fixture()
        executor = ActionExecutor(graph, state, seams, evidence, storage)

        action = executor.build_action(STOP, "claim:whatever")
        result = executor.execute(action)

        self.assertEqual(result.status, COMPLETED)
        self.assertEqual(result.actual_cost.tokens, 0)
        self.assertEqual(result.actual_cost.calls, 0)


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
        from env_fix import fix_cuda_dll_path  # той же фикс, что использует model_adapter.py

        fix_cuda_dll_path()
        import llama_cpp  # noqa: F401
    except Exception:
        return False
    return True


class RealModelSmokeTest(unittest.TestCase):
    """Критерий готовности этапа 5: настоящий GENERATE на локальной GGUF.

    Пропускается автоматически там, где llama_cpp/файл модели
    недоступны -- не на этой машине."""

    @unittest.skipUnless(_real_model_available(), "llama_cpp / local GGUF model not available")
    def test_single_generate_on_real_local_model(self):
        from src.model_adapter import ModelAdapter

        storage, graph, state, seams, evidence, artifact_id = _base_fixture()
        model = ModelAdapter("llama-3.2-1b-instruct-q4_0")
        executor = ActionExecutor(graph, state, seams, evidence, storage, model)

        action = executor.build_action(
            GENERATE,
            "claim:real",
            sampling=Sampling(temperature=0.0, top_p=1.0, top_k=40, seed=1, max_tokens=32),
        )
        context = build_context(
            GENERATE, "claim:real", graph, task_statement="Reply with exactly the digit 4 and nothing else."
        )

        result = executor.execute(action, context, artifact_id=artifact_id, claim_type="VALUE")

        self.assertEqual(result.status, COMPLETED)
        self.assertIsNotNone(result.actual_cost)
        self.assertGreater(result.actual_cost.time_sec, 0)

        runs = [r.payload for r in storage.query(lambda rec: rec.kind == "run")]
        self.assertEqual(len(runs), 1)
        self.assertFalse(runs[0].failed)
        self.assertIsNotNone(runs[0].rendered_prompt)
        self.assertTrue(runs[0].raw_output)


if __name__ == "__main__":
    unittest.main()
