"""run_task.py — точка входа: один таск через главный цикл (этап 6).

Использование:
    python run_task.py [task_id] [model_id]

По умолчанию: BRANCH_01, llama-3.2-1b-instruct-q4_0 (наименьшая из трёх
зарегистрированных моделей, experiment9/configs/models.json).
"""

import sys

from src.action_executor import ActionExecutor
from src.evidence_engine import EvidenceEngine
from src.graph_core import GraphCore
from src.model_adapter import ModelAdapter
from src.orchestrator import run_task
from src.repair_planner import RepairPlanner
from src.seam_engine import default_seam_engine
from src.state_engine import StateEngine
from src.storage import Storage
from tasks.branch_tasks import CLAIM_TYPE, CLAIMS, DEPENDENCY_EDGES, TASKS, generation_prompt, oracle_values


def main() -> int:
    task_id = sys.argv[1] if len(sys.argv) > 1 else "BRANCH_01"
    model_id = sys.argv[2] if len(sys.argv) > 2 else "llama-3.2-1b-instruct-q4_0"

    task = TASKS[task_id]

    storage = Storage()
    graph = GraphCore(storage)
    state = StateEngine(storage, graph)
    seams = default_seam_engine(storage)
    for seam_id in ("seam:exec", "seam:format", "seam:numeric"):
        result = seams.self_test(seam_id)
        if not result["passed"]:
            print(f"FATAL: {seam_id} failed self_test: {result['failed_cases']}")
            return 1
    evidence = EvidenceEngine(storage)
    planner = RepairPlanner(graph, state, seams)
    model = ModelAdapter(model_id)
    executor = ActionExecutor(graph, state, seams, evidence, storage, model_adapter=model)

    print(f"Task {task_id!r} on model {model_id!r}: {task['question']}\n")

    result = run_task(
        task_id, task, CLAIMS, CLAIM_TYPE, DEPENDENCY_EDGES,
        generation_prompt, oracle_values,
        graph, state, seams, evidence, planner, executor, storage,
    )

    print(f"stop_reason: {result.stop_reason}")
    print(f"iterations: {result.iterations} (collisions detected: {result.collisions_detected}, resolved: {result.collisions_resolved})")
    print(f"graph_version: {result.graph_version}")
    print("claim states:")
    for claim_id, claim_state in result.claim_states.items():
        content = graph.get_claim(claim_id).content
        print(f"  {claim_id:16s} {claim_state:12s} {content!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
