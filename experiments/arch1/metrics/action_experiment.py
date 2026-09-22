"""action_experiment.py — эксперимент 10 (ARCH-1.1).

Измеряет `collision -> action -> outcome`: какое действие ремонта что даёт,
при какой цене. Проверяет предсказанный ACTION_MODEL.md §106 порядок

    CHANGE_MODEL > CHANGE_PROMPT > CHANGE_TEMPERATURE > голый REGENERATE

который спека утверждает, но никогда не измеряла end-to-end.

ЧТО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ. Он не выбирает лучшее действие и не пишет
результат обратно в policy.py: выбор в прогоне равновероятный или
исчерпывающий, а закрепление найденного — задача эволюционного слоя, вне
ARCH-1 (WORK_PLAN.md §8). Смешение измерения с эксплуатацией сместило бы
частоты и сделало бы плечи несравнимыми.

КОНСТРУКЦИЯ — ПАРНАЯ. Все плечи стартуют из ОДНОГО И ТОГО ЖЕ замороженного
состояния коллизии (фаза A), поэтому сравниваются на одинаковых входах, а
не на разных коллизиях.

КОНТРОЛЬ — «та же конфигурация, НОВЫЙ seed». Вырожденный контроль (тот же
seed) уже измерен кампанией 3, где 89.8% повторных REGENERATE дали
побайтово идентичный вывод: промпт для REGENERATE не включает прежнюю
неверную версию (CONTEXT_MODEL §4.1), а seed фиксировался на весь прогон,
поэтому повтор не мог дать иного результата в принципе.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.action_executor import COMPLETED, REGENERATE, ActionExecutor
from src.collision import Collision, localize
from src.context_builder import build_context
from src.evidence_engine import EvidenceEngine
from src.graph_core import GraphCore
from src.repair_planner import MUST_REGENERATE, RepairPlanner
from src.seam_engine import default_seam_engine
from src.seams.numeric_seam import check as numeric_check
from src.state_engine import StateEngine
from src.storage import Sampling, Storage, new_id
from tasks.branch_tasks import (
    CLAIM_TYPE,
    CLAIMS,
    DEPENDENCY_EDGES,
    PROMPT_PROFILES,
    TASKS,
    oracle_values,
)

SEAM_IDS = ("seam:exec", "seam:format", "seam:numeric")
BASE_TEMPERATURE = 0.5
MAX_TOKENS = 200

# Плечи. Все — REGENERATE с разной конфигурацией: ACTION_MODEL.md §2
# определяет CHANGE_* как МОДИФИКАТОРЫ, а DATA_MODEL.md §11 уже несёт
# поля model_id / prompt_profile_id / sampling_config на самом ACTION.
ARMS = (
    "REGENERATE_SAME",
    "CHANGE_MODEL",
    "CHANGE_TEMPERATURE_LOW",
    "CHANGE_TEMPERATURE_HIGH",
    "CHANGE_PROMPT",
)


@dataclass(frozen=True)
class RepairConfig:
    """Конфигурация одной попытки ремонта."""

    arm: str
    model_id: str
    temperature: float
    prompt_profile: str
    seed: int

    def signature(self) -> tuple:
        """Что именно считается «той же конфигурацией» (пункт 4 задания).

        seed ВХОДИТ в сигнатуру: повтор с новым seed — это другая
        конфигурация, дающая шанс на иной результат. Повтор с тем же
        seed — вырожденный случай, детерминированно обречённый.
        """
        return (self.model_id, self.temperature, self.prompt_profile, self.seed)


def build_configs(base_model_id: str, all_model_ids: list, rng: random.Random, seed: int) -> dict:
    """По одной конфигурации на плечо. Альтернативная модель выбирается
    равновероятно из оставшихся — не «лучшая», выбор не предрешён."""
    others = [m for m in all_model_ids if m != base_model_id]
    alt_model = rng.choice(others) if others else base_model_id
    return {
        "REGENERATE_SAME": RepairConfig("REGENERATE_SAME", base_model_id, BASE_TEMPERATURE, "v1", seed),
        "CHANGE_MODEL": RepairConfig("CHANGE_MODEL", alt_model, BASE_TEMPERATURE, "v1", seed),
        "CHANGE_TEMPERATURE_LOW": RepairConfig("CHANGE_TEMPERATURE_LOW", base_model_id, 0.0, "v1", seed),
        "CHANGE_TEMPERATURE_HIGH": RepairConfig("CHANGE_TEMPERATURE_HIGH", base_model_id, 1.0, "v1", seed),
        "CHANGE_PROMPT": RepairConfig("CHANGE_PROMPT", base_model_id, BASE_TEMPERATURE, "v2", seed),
    }


# -- единый критерий успеха ------------------------------------------------


def solved(contents: dict, task: dict) -> bool:
    """Задача решена <=> все claims с oracle прошли numeric_seam.

    Применяется ОДИНАКОВО к ремонту, FULL_RETRY и обоим best-of-N —
    иначе стратегии несравнимы."""
    oracle = oracle_values(task)
    return all(
        numeric_check({"text": contents.get(c, ""), "expected": float(exp)})["status"] == "PASS"
        for c, exp in oracle.items()
    )


def defective_claims(contents: dict, task: dict) -> set:
    oracle = oracle_values(task)
    return {
        c
        for c, exp in oracle.items()
        if numeric_check({"text": contents.get(c, ""), "expected": float(exp)})["status"] == "FAIL"
    }


# -- фаза A: восстановление замороженных состояний (без GPU) ---------------


def _first_generations(result: dict) -> Optional[dict]:
    """Первые 6 вызовов прогона = первичная генерация (без ремонта)."""
    contents: dict = {}
    for record in result.get("runs", []):
        payload = record["payload"]
        prompt = payload.get("rendered_prompt") or ""
        for claim_id in CLAIMS:
            if f"[{claim_id}]" in prompt and claim_id not in contents:
                contents[claim_id] = payload["raw_output"].strip()
    return contents if all(c in contents for c in CLAIMS) else None


def replay_collision_states(runs_dir) -> list:
    """Фаза A. Реплей первичных генераций кампании 3 -> состояния коллизий.

    Детерминированно, без GPU. Даёт общую стартовую точку для всех плеч.
    """
    states = []
    for path in sorted(Path(runs_dir).glob("*/*/*/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "harness_error" in data:
            continue
        contents = _first_generations(data)
        if contents is None:
            continue
        task = TASKS[data["task_id"]]
        defective = defective_claims(contents, task)
        if not defective:
            continue  # коллизии нет — в эксперимент по ремонту не входит
        states.append(
            {
                "state_id": f"{data['model_id']}|{data['task_id']}|{data['seed']}",
                "model_id": data["model_id"],
                "task_id": data["task_id"],
                "seed": data["seed"],
                "contents": contents,
                "defective": sorted(defective),
            }
        )
    return states


def _build_graph(contents: dict):
    storage = Storage()
    graph = GraphCore(storage)
    artifact_version_id = graph.add_artifact_version(
        artifact_type="DERIVATION", task_id="task:replay", content="replayed"
    )
    artifact_id = storage.get(artifact_version_id).payload.artifact_id
    for claim_id in CLAIMS:
        graph.add_claim_version(
            artifact_id=artifact_id,
            content=contents[claim_id],
            claim_type=CLAIM_TYPE[claim_id],
            claim_id=claim_id,
        )
    for source, target, dep_type in DEPENDENCY_EDGES:
        graph.add_dependency(
            source_claim=source, target_claim=target,
            dependency_type=dep_type, status="CONFIRMED", created_by="HUMAN",
        )
    return storage, graph, artifact_id


def _wire(storage, graph):
    state = StateEngine(storage, graph)
    seams = default_seam_engine(storage)
    for seam_id in SEAM_IDS:
        result = seams.self_test(seam_id)
        assert result["passed"], (seam_id, result["failed_cases"])
    evidence = EvidenceEngine(storage)
    planner = RepairPlanner(graph, state, seams)
    return state, seams, evidence, planner


def plan_repair(contents: dict, task_id: str, defective: list) -> list:
    """Локализация + сужение + план -> какие claims подлежат регенерации.

    Использует штатные движки (включая поправку AMD-1), не свою логику.
    """
    task = TASKS[task_id]
    storage, graph, _ = _build_graph(contents)
    _state, _seams, _evidence, planner = _wire(storage, graph)

    collision = Collision(
        collision_id=new_id("col"),
        collision_type="MECHANICAL_FAILURE",
        participants=tuple(defective),
    )
    localize(collision, graph, defective_claims=set(defective))
    if not collision.origin:
        return []

    region = planner.affected_region(collision)
    oracle = oracle_values(task)

    def provider(claim_id, seam, _o=oracle, _c=contents):
        if seam.seam_id != "seam:numeric" or claim_id not in _o:
            return None
        return {"text": _c[claim_id], "expected": float(_o[claim_id])}

    narrowed = planner.narrow(region, provider)
    plan = planner.build_plan(narrowed, collision, task={"risk_level": "MEDIUM"})
    return [e.claim_id for e in plan.entries if e.necessity == MUST_REGENERATE]


# -- фаза B: одно плечо на одном состоянии (GPU) ---------------------------


def run_arm(state: dict, config: RepairConfig, model_adapter) -> dict:
    """Одна попытка ремонта в заданной конфигурации из замороженного
    состояния. Возвращает исход и ФАКТИЧЕСКИЕ токены."""
    task = TASKS[state["task_id"]]
    contents = dict(state["contents"])
    to_regenerate = plan_repair(contents, state["task_id"], state["defective"])

    if not to_regenerate:
        return {
            "state_id": state["state_id"], "arm": config.arm,
            "config": config.__dict__, "regenerated": [],
            "tokens": 0, "solved": solved(contents, task),
            "outcome": "NO_REPAIR_PLANNED",
        }

    storage, graph, _ = _build_graph(contents)
    state_engine, seams, evidence, _planner = _wire(storage, graph)
    executor = ActionExecutor(graph, state_engine, seams, evidence, storage, model_adapter=model_adapter)

    prompt_fn = PROMPT_PROFILES[config.prompt_profile]
    sampling = Sampling(
        temperature=config.temperature, top_p=1.0, top_k=40,
        seed=config.seed, max_tokens=MAX_TOKENS,
    )
    parents = {t: [s for s, tt, _ in DEPENDENCY_EDGES if tt == t] for t in CLAIMS}

    for claim_id in to_regenerate:
        context = build_context(
            REGENERATE, claim_id, graph,
            task_statement=prompt_fn(claim_id, task),
            dependency_claim_ids=tuple(parents[claim_id]),
        )
        action = executor.build_action(REGENERATE, claim_id, sampling=sampling)
        # состояние claim'а должно допускать REGENERATE -- проставляем его
        # тем же официальным каналом (VERIFY), а не правкой напрямую
        _verify(executor, graph, claim_id, oracle_values(task).get(claim_id))
        result = executor.execute(action, context)
        if result.status == COMPLETED:
            contents[claim_id] = graph.get_claim(claim_id).content

    tokens = sum(
        (r.payload.input_tokens or 0) + (r.payload.output_tokens or 0)
        for r in storage.query(lambda rec: rec.kind == "run")
    )
    return {
        "state_id": state["state_id"], "arm": config.arm,
        "config": config.__dict__, "regenerated": to_regenerate,
        "tokens": tokens, "solved": solved(contents, task),
        "outcome": "SOLVED" if solved(contents, task) else "NOT_SOLVED",
        "contents_after": contents,
    }


def _verify(executor, graph, claim_id, expected):
    if expected is None:
        return
    from src.action_executor import VERIFY

    action = executor.build_action(VERIFY, claim_id)
    executor.execute(
        action, seam_id="seam:numeric",
        seam_inputs={"text": graph.get_claim(claim_id).content, "expected": float(expected)},
    )


# -- фаза C: траектория с запретом повтора конфигурации (GPU) --------------

NO_EXPECTED_GAIN = "NO_EXPECTED_GAIN"
SOLVED = "SOLVED"


def run_escalation(state: dict, configs: dict, adapters: dict, rng: random.Random) -> dict:
    """Пункты 4 и 8 как сквозная стратегия.

    На каждой итерации конфигурация выбирается РАВНОВЕРОЯТНО из ещё не
    испробованных; повтор уже испробованной сигнатуры запрещён
    механически; когда неиспробованных не осталось -> STOP(NO_EXPECTED_GAIN).

    Порядок случайный НАМЕРЕННО: он не предполагает, какое действие лучше.
    Информированный порядок — задача эволюционного слоя, вне эксперимента.
    """
    task = TASKS[state["task_id"]]
    current = dict(state["contents"])
    tried: set = set()
    total_tokens = 0
    attempts = 0
    trace = []

    remaining = list(configs.values())
    rng.shuffle(remaining)

    for config in remaining:
        if config.signature() in tried:
            continue  # пункт 4: та же конфигурация повторно запрещена
        tried.add(config.signature())

        working = dict(state)
        working["contents"] = current
        working["defective"] = sorted(defective_claims(current, task))
        if not working["defective"]:
            break

        result = run_arm(working, config, adapters[config.model_id])
        attempts += 1
        total_tokens += result["tokens"]
        trace.append({"arm": config.arm, "tokens": result["tokens"], "solved": result["solved"]})
        current = result.get("contents_after", current)
        if result["solved"]:
            return {
                "state_id": state["state_id"], "solved": True, "tokens": total_tokens,
                "attempts": attempts, "stop_reason": SOLVED, "trace": trace,
            }

    return {
        "state_id": state["state_id"], "solved": solved(current, task),
        "tokens": total_tokens, "attempts": attempts,
        "stop_reason": NO_EXPECTED_GAIN, "trace": trace,
    }


# -- best-of-N (анализ по уже имеющимся seed, без GPU) ---------------------


def best_of_n(runs_dir) -> dict:
    """Пункт 11. Оба правила выбора:

    oracle   — успех, если ХОТЯ БЫ ОДНА из N генераций полностью верна
               (верхняя граница плоской стратегии; симметрично архитектуре,
               которая тоже имеет доступ к oracle);
    majority — успех, если модальное значение среди N верно (то, что
               доступно БЕЗ oracle; F14 предсказывает ненадёжность).
    """
    by_key: dict = {}
    for path in sorted(Path(runs_dir).glob("*/*/*/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "harness_error" in data:
            continue
        contents = _first_generations(data)
        if contents is None:
            continue
        key = (data["model_id"], data["task_id"])
        tokens = sum(
            (r["payload"].get("input_tokens") or 0) + (r["payload"].get("output_tokens") or 0)
            for r in data.get("runs", [])[:6]
        )
        by_key.setdefault(key, []).append({"contents": contents, "tokens": tokens})

    out: dict = {}
    for n in (1, 2, 3, 4):
        oracle_ok = majority_ok = total = token_sum = 0
        for (_model, task_id), samples in by_key.items():
            if len(samples) < n:
                continue
            window = samples[:n]
            task = TASKS[task_id]
            total += 1
            token_sum += sum(s["tokens"] for s in window)
            if any(solved(s["contents"], task) for s in window):
                oracle_ok += 1
            # majority: модальный ответ по финальному claim'у
            finals = [s["contents"].get("C6_FINAL_TOTAL", "") for s in window]
            modal = max(set(finals), key=finals.count)
            picked = next(s for s in window if s["contents"].get("C6_FINAL_TOTAL", "") == modal)
            if solved(picked["contents"], task):
                majority_ok += 1
        if total:
            out[f"N={n}"] = {
                "n_task_model_pairs": total,
                "mean_tokens": token_sum / total,
                "oracle_selected_success_rate": oracle_ok / total,
                "majority_selected_success_rate": majority_ok / total,
                "oracle_tokens_per_success": (token_sum / oracle_ok) if oracle_ok else None,
                "majority_tokens_per_success": (token_sum / majority_ok) if majority_ok else None,
            }
    return out


# -- агрегация -------------------------------------------------------------


def aggregate(attempts: list, escalations: list, runs_dir) -> dict:
    by_arm: dict = {}
    for a in attempts:
        bucket = by_arm.setdefault(a["arm"], {"n": 0, "solved": 0, "tokens": 0})
        bucket["n"] += 1
        bucket["solved"] += 1 if a["solved"] else 0
        bucket["tokens"] += a["tokens"]

    arms_table = {}
    for arm, b in by_arm.items():
        arms_table[arm] = {
            "n": b["n"],
            "success_rate": b["solved"] / b["n"] if b["n"] else None,
            "mean_tokens": b["tokens"] / b["n"] if b["n"] else None,
            "tokens_per_success": (b["tokens"] / b["solved"]) if b["solved"] else None,
        }

    observed_order = sorted(
        (a for a in arms_table if arms_table[a]["success_rate"] is not None),
        key=lambda a: arms_table[a]["success_rate"],
        reverse=True,
    )

    esc = None
    if escalations:
        n = len(escalations)
        solved_n = sum(1 for e in escalations if e["solved"])
        tok = sum(e["tokens"] for e in escalations)
        esc = {
            "n": n,
            "success_rate": solved_n / n,
            "mean_tokens": tok / n,
            "tokens_per_success": (tok / solved_n) if solved_n else None,
            "mean_attempts": sum(e["attempts"] for e in escalations) / n,
            "stopped_no_expected_gain": sum(1 for e in escalations if e["stop_reason"] == "NO_EXPECTED_GAIN"),
        }

    return {
        "arms": arms_table,
        "predicted_order_action_model_106": ["CHANGE_MODEL", "CHANGE_PROMPT",
                                             "CHANGE_TEMPERATURE_LOW/HIGH", "REGENERATE_SAME"],
        "observed_order_by_success_rate": observed_order,
        "escalation_no_repeat": esc,
        "best_of_n": best_of_n(runs_dir),
        "note": (
            "tokens_per_success -- главная метрика: кампания 3 показала, что "
            "стоимость попытки без вероятности успеха вводит в заблуждение "
            "(плейсхолдерная метрика давала 0.283 при фактических 1.06). "
            "Таблица НЕ записывается обратно в policy.py: закрепление -- "
            "задача эволюционного слоя, вне ARCH-1 (WORK_PLAN §8)."
        ),
    }
