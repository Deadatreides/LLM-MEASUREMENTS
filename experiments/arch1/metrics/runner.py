"""runner.py — измерительная кампания и агрегация (этап 7).

"Одна генерация" здесь = один полный orchestrator.run_task() (6 GENERATE
+, при коллизии, REGENERATE/VERIFY в цикле ремонта), а НЕ один вызов
модели, как в experiment9 (D-mode, все 6 секций одним вызовом).
Переопределение зафиксировано в arch1/STATUS.md, этап 6: ARCH-1
сознательно не переносит D-mode (REPAIR_MODEL требует REGENERATE ровно
одного claim'а, PATCH исключён из ARCH-1). 20 задач × 3 модели × 4 seed
= 240 таких полных прогонов, не 240 вызовов модели.

Только ветка A (граф задан вручную) — решение пользователя, этап 7.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from src.action_executor import ActionExecutor
from src.evidence_engine import EvidenceEngine
from src.graph_core import GraphCore
from src.orchestrator import run_task
from src.repair_planner import FULL_RETRY, LOCAL_REPAIR, RepairPlanner
from src.seam_engine import default_seam_engine
from src.state_engine import StateEngine
from src.storage import Sampling, Storage
from tasks.branch_tasks import CLAIM_TYPE, CLAIMS, DEPENDENCY_EDGES, TASKS, generation_prompt, oracle_values

SEAM_IDS = ("seam:exec", "seam:format", "seam:numeric")

# T=0.5 -- как в P1/T=0.5 experiment9 (см. models.json _comment): при
# temperature=0.0 (дефолт ActionExecutor, этапы 5-6) декодирование жадное
# и seed не влияет на результат -- для измерения вариативности по seed
# нужна температура > 0. Параметр только этого харнесса, ActionExecutor/
# orchestrator.py не меняются (см. план этапа 7).
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 200


def _record_to_dict(record) -> dict:
    payload = record.payload
    payload_dict = dataclasses.asdict(payload) if dataclasses.is_dataclass(payload) else payload
    return {"record_id": record.record_id, "kind": record.kind, "seq": record.seq, "payload": payload_dict}


def _vram_snapshot() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        used, total = out.stdout.strip().split(",")
        return {"used_mib": int(used.strip()), "total_mib": int(total.strip())}
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def run_one(
    model_adapter,
    model_id: str,
    task_id: str,
    seed: int,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Один полный прогон run_task() -- raw-данные + всё нужное для
    агрегации метрик (сериализуемо в JSON)."""
    task = TASKS[task_id]

    storage = Storage()
    graph = GraphCore(storage)
    state = StateEngine(storage, graph)
    seams = default_seam_engine(storage)
    self_test = {seam_id: seams.self_test(seam_id) for seam_id in SEAM_IDS}
    evidence = EvidenceEngine(storage)
    planner = RepairPlanner(graph, state, seams)
    executor = ActionExecutor(graph, state, seams, evidence, storage, model_adapter=model_adapter)

    sampling = Sampling(temperature=temperature, top_p=1.0, top_k=40, seed=seed, max_tokens=max_tokens)

    plans: list = []
    started = time.time()
    result = run_task(
        task_id, task, CLAIMS, CLAIM_TYPE, DEPENDENCY_EDGES,
        generation_prompt, oracle_values,
        graph, state, seams, evidence, planner, executor, storage,
        plan_collector=plans, sampling=sampling,
    )
    wall_time_sec = time.time() - started

    return {
        "model_id": model_id,
        "task_id": task_id,
        "seed": seed,
        "temperature": temperature,
        "wall_time_sec": wall_time_sec,
        "self_test": {seam_id: r["passed"] for seam_id, r in self_test.items()},
        "stop_reason": result.stop_reason,
        "iterations": result.iterations,
        "collisions_detected": result.collisions_detected,
        "collisions_resolved": result.collisions_resolved,
        "graph_version": result.graph_version,
        "claim_states": result.claim_states,
        "seam_results": [_record_to_dict(r) for r in storage.query(lambda rec: rec.kind == "seam_result")],
        "runs": [_record_to_dict(r) for r in storage.query(lambda rec: rec.kind == "run")],
        "plans": [dataclasses.asdict(p) for p in plans],
    }


def run_campaign(
    model_ids: list,
    seeds: list,
    task_ids: list,
    runs_dir,
    log: Callable[[str], None] = print,
) -> None:
    """Дисциплина запусков (WORK_PLAN §3): модель грузится один раз на
    модель, строго последовательно; каждый прогон чекпоинтится на диск
    сразу (атомарная запись через .tmp + replace) -- безопасно
    прерываемо и возобновляемо, повторный вызов пропускает готовые
    result.json."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    for model_id in model_ids:
        log(f"=== {model_id}: loading === VRAM before: {_vram_snapshot()}")
        from src.model_adapter import ModelAdapter  # ленивый импорт, см. model_adapter.py

        model_adapter = ModelAdapter(model_id)
        log(f"=== {model_id}: loaded === VRAM after: {_vram_snapshot()}")

        for seed in seeds:
            for task_id in task_ids:
                out_path = runs_dir / model_id / task_id / f"seed_{seed}" / "result.json"
                if out_path.exists():
                    log(f"skip (already done): {model_id}/{task_id}/seed_{seed}")
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    result = run_one(model_adapter, model_id, task_id, seed)
                    outcome = result["stop_reason"]
                except Exception as exc:  # харнесс не должен уронить всю кампанию из-за одного прогона
                    result = {
                        "model_id": model_id,
                        "task_id": task_id,
                        "seed": seed,
                        "harness_error": f"{type(exc).__name__}: {exc}",
                    }
                    outcome = f"HARNESS_ERROR: {result['harness_error']}"

                tmp_path = out_path.with_suffix(".json.tmp")
                tmp_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
                )
                tmp_path.replace(out_path)
                log(f"done: {model_id}/{task_id}/seed_{seed} -> {outcome}")

        del model_adapter
        log(f"=== {model_id}: unloaded === VRAM after: {_vram_snapshot()}")


def aggregate_measurements(runs_dir) -> dict:
    """Отдельный, независимо перезапускаемый проход по уже накопленным
    runs/**/result.json -- пересчёт метрик без повторной генерации."""
    runs_dir = Path(runs_dir)
    result_files = sorted(runs_dir.glob("*/*/*/result.json"))

    total_runs = len(result_files)
    harness_errors = 0
    stop_reason_counts: dict = {}
    self_test_all_passed = True

    unaffected_count = 0
    verified_by_narrow_count = 0
    local_repair_costs: list = []
    full_retry_costs: list = []
    seam_status_counts: dict = {}

    for path in result_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "harness_error" in data:
            harness_errors += 1
            continue

        stop_reason_counts[data["stop_reason"]] = stop_reason_counts.get(data["stop_reason"], 0) + 1

        for passed in data.get("self_test", {}).values():
            if not passed:
                self_test_all_passed = False

        for plan in data.get("plans", []):
            for entry in plan.get("entries", []):
                if entry["rationale"] == "UNAFFECTED_CONFIRMED":
                    unaffected_count += 1
                elif entry["rationale"] == "VERIFIED_BY_NARROW":
                    verified_by_narrow_count += 1
            for alt in plan.get("alternatives_considered", []):
                if alt["strategy"] == LOCAL_REPAIR:
                    local_repair_costs.append(alt["cost"])
                elif alt["strategy"] == FULL_RETRY:
                    full_retry_costs.append(alt["cost"])

        for seam_result in data.get("seam_results", []):
            status = seam_result["payload"]["status"]
            seam_status_counts[status] = seam_status_counts.get(status, 0) + 1

    def _mean(costs: list, field: str) -> Optional[float]:
        values = [c[field] for c in costs if c.get(field) is not None]
        return sum(values) / len(values) if values else None

    denom = unaffected_count + verified_by_narrow_count
    local_repair_mean_tokens = _mean(local_repair_costs, "tokens")
    full_retry_mean_tokens = _mean(full_retry_costs, "tokens")
    cost_ratio = (
        local_repair_mean_tokens / full_retry_mean_tokens
        if local_repair_mean_tokens is not None and full_retry_mean_tokens
        else None
    )
    total_seam_evaluations = sum(seam_status_counts.values())

    return {
        "total_runs": total_runs,
        "harness_errors": harness_errors,
        "stop_reason_counts": stop_reason_counts,
        "stale_resolved_by_verify_ratio": {
            "value": (unaffected_count / denom) if denom else None,
            "unaffected_count": unaffected_count,
            "verified_by_narrow_count": verified_by_narrow_count,
            "note": (
                "доля non-origin узлов региона ремонта, найденных unaffected "
                "(дешёвый VERIFY подтвердил) среди всех классифицированных "
                "narrow() (unaffected + verified) по всем планам ремонта "
                "кампании -- высокое значение = сужение даёт экономию"
            ),
        },
        "local_repair_over_full_retry_cost_ratio": {
            "value": cost_ratio,
            "local_repair_mean_tokens": local_repair_mean_tokens,
            "full_retry_mean_tokens": full_retry_mean_tokens,
            "n_plans": len(local_repair_costs),
            "note": (
                "среднее по всем построенным планам ремонта кампании; "
                "< 1.0 означает, что граф-ремонт дешевле полной перегенерации "
                "задачи. Стоимости из _default_cost_estimator (repair_planner.py) "
                "-- качественный плейсхолдер, не откалиброванные числа "
                "(MODEL_PROFILE не реализован, см. STATUS.md)"
            ),
        },
        "seam_self_test_vs_real_data": {
            "self_test_all_seams_passed": self_test_all_passed,
            "seam_status_counts_real_data": seam_status_counts,
            "error_rate_real_data": (
                seam_status_counts.get("ERROR", 0) / total_seam_evaluations
                if total_seam_evaluations
                else None
            ),
            "note": (
                "self_test -- статический контрольный набор (3 шва, "
                "заранее известные случаи); error_rate_real_data -- доля "
                "ERROR (сбой самой процедуры проверки) среди реальных "
                "evaluate() за кампанию, отдельно от FAIL (содержательный "
                "негативный результат)"
            ),
        },
        "branch_b_automatic_graph_inference": {
            "status": "OUT_OF_SCOPE",
            "reason": (
                "ARCH-1 сознательно не реализует DECOMPOSE/вывод "
                "зависимостей (решение этапа 6). Решение пользователя на "
                "этапе 7: измеряется только ветка A (граф задан вручную)."
            ),
        },
    }
