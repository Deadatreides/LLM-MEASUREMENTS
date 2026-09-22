"""harness.py — эксперимент 12: контекст против оператора.

7 плеч на замороженном состоянии коллизии: 3 уровня контекста (K0
evidence-only / K1 структура без значений / K2 структура + подтверждённые
значения) x 2 уровня оператора (O0 та же модель / O1 другая модель) = 6,
плюс отдельное K0O0_promptB для предзарегистрированной проверки H-син
(см. TASK_EXPERIMENT12.md §2 и план -- пробел в исходном задании,
закрытый 7-м плечом, не входящим в основную 3x2 сетку).

EXPERIMENT_ID фиксирует кампанию (тот же принцип, что в
experiment11/harness.py: при обнаруженном дефекте конвейера -- новая
кампания с новым experiment_id, не патч поверх старой).

`paired_sign_test`/`bootstrap_ratio_diff` -- та же реализация, что в
`experiment11/harness.py` (не импортируются оттуда напрямую -- у обоих
файлов одинаковое имя модуля "harness", что дало бы коллизию, аналогично
`seams.py`/`configs/model_registry.py`; вместо ещё одной importlib-
обёртки эти две маленькие чистые функции просто скопированы).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from configs.model_registry import MODEL_REGISTRY, generate, load_model
from context_builder import build_prompt, build_prompt_promptB, structure_text
from context_manifest import validate_prompt_against_manifest
from outcome_classification import classify_outcome, detect_improvements, detect_regressions
from seams import ERROR, FAIL, INAPPLICABLE, PASS, collision_subtype, enriched_code_seam, multistep_arithmetic_seam
from tasks import arithmetic_multistep_tasks, code_tasks

EXPERIMENT_ID = "exp12-v2"  # v2: строгий протокол (пятисоставный статус, регрессии, dependency/value subtype, context manifest) -- см. TASK_EXPERIMENT12_V2 в чате и REPORT_EXPERIMENT12_V2.md. v1 (exp12-v1) данные и отчёт не трогаются.

BASE_TEMPERATURE = 0.5
TOP_P = 1.0
INITIAL_SEEDS = (1, 2, 3, 4)
RETRY_SEED = 1000  # новый seed для ВСЕХ retry-плеч -- никогда не совпадает с 1-4 (урок arch1/эксп.10/11)

MAX_TOKENS_ARITH = 150   # 4-6 строк "name = value" -- запас щедрый
MAX_TOKENS_CODE = 320

ARMS_MAIN = ("K0O0", "K0O1", "K1O0", "K1O1", "K2O0", "K2O1")
ARM_H_SYN = "K0O0_promptB"
ALL_ARMS = ARMS_MAIN + (ARM_H_SYN,)

MIN_CATEGORY_N = 30    # порог для разбивки по типу коллизии (ужесточён относительно эксп.11's 8)
MIN_HSYN_N = 100        # порог специфично для предзарегистрированной H-син (§9 задания)


class LeakDetected(RuntimeError):
    """Антиутечная проверка (§5 задания) поймала нарушение -- кампания
    должна остановиться, а не тихо пропустить запись."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _characteristics(text: str) -> dict:
    text = text or ""
    return {"length_chars": len(text), "length_words": len(text.split())}


def _prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _get_task(task_family: str, task_id: str) -> dict:
    if task_family == "arithmetic":
        return arithmetic_multistep_tasks.TASKS[task_id]
    return code_tasks.CODE_TASKS[task_id]


def _prompt_and_tokens_initial(task_family: str, task_id: str):
    task = _get_task(task_family, task_id)
    if task_family == "arithmetic":
        return arithmetic_multistep_tasks.generation_prompt(task), MAX_TOKENS_ARITH
    return code_tasks.generation_prompt(task), MAX_TOKENS_CODE


def _checked(task_family: str, task_id: str, raw: dict) -> dict:
    if raw.get("generation_failed"):
        return {"status": ERROR, "collision_type": None, "details": {"reason": raw.get("generation_error")}}
    task = _get_task(task_family, task_id)
    try:
        if task_family == "arithmetic":
            return multistep_arithmetic_seam(raw.get("raw_text", ""), task["steps"])
        return enriched_code_seam(raw.get("raw_text", ""), task["function_name"], task["requirements"])
    except Exception as exc:
        return {"status": ERROR, "collision_type": None, "details": {"reason": f"{type(exc).__name__}: {exc}"}}


def _make_record(*, task_id, collision_id, arm_id, model_id, temperature, seed, prompt,
                  initial_artifact, raw, task_family, seam_output, task,
                  initial_seam_result=None, context_level=None, operator_mode=None) -> dict:
    """`initial_seam_result` -- ТОЛЬКО для retry-плеч (сравнение с
    первичной генерацией); для самой первичной генерации остаётся None
    -- final_status/regressions/improvements не имеют смысла без базы
    для сравнения (задание v2, пятисоставный статус)."""
    entry = MODEL_REGISTRY[model_id]
    result_text = raw.get("raw_text", "")
    input_tokens = raw.get("input_tokens") or 0
    output_tokens = raw.get("output_tokens") or 0

    steps = task.get("steps") if task_family == "arithmetic" else None
    failure_type = collision_subtype(task_family, seam_output, steps)

    if initial_seam_result is not None:
        final_status = classify_outcome(initial_seam_result, seam_output)
        regressions = detect_regressions(initial_seam_result, seam_output)
        improvements = detect_improvements(initial_seam_result, seam_output)
    else:
        final_status, regressions, improvements = None, [], []

    mechanical_checks = seam_output.get("step_results") or seam_output.get("requirement_results") or []

    return {
        "experiment_id": EXPERIMENT_ID,
        "task_id": task_id,
        "collision_id": collision_id,
        "artifact_id": f"{collision_id}:{arm_id}",
        "prompt_id": _prompt_id(prompt),
        "arm_id": arm_id,
        "context_level": context_level,
        "operator_mode": operator_mode,
        "model": model_id,
        "model_family": entry["family"],
        "model_size": entry["size_label"],
        "temperature": temperature,
        "top_p": TOP_P,
        "seed": seed,
        "prompt": prompt,
        "prompt_characteristics": _characteristics(prompt),
        "initial_artifact": initial_artifact,
        "result": result_text,
        "result_characteristics": _characteristics(result_text),
        "seam_type": task_family,
        "seam_output": seam_output,
        "mechanical_checks": mechanical_checks,
        "status": seam_output["status"],
        "collision_type": seam_output.get("collision_type"),
        "failure_type": failure_type,
        "final_status": final_status,
        "regressions": regressions,
        "improvements": improvements,
        "evidence": seam_output.get("details"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency": raw.get("generation_time_sec"),
        "timestamp": _now_iso(),
        "generation_failed": raw.get("generation_failed", False),
        "generation_error": raw.get("generation_error"),
    }


# -- состояние коллизии -------------------------------------------------------


@dataclass
class CollisionState:
    collision_id: str
    task_family: str
    task_id: str
    model_id: str
    seed: int
    initial_artifact: str
    initial_seam_result: dict
    initial_record: dict


def find_collision_states(model_ids, seeds, arith_task_ids, code_task_ids, log=print) -> tuple:
    states: list = []
    all_records: list = []

    for model_id in model_ids:
        log(f"=== {model_id}: loading ===")
        llm, load_t = load_model(model_id)
        log(f"=== {model_id}: loaded in {load_t:.1f}s ===")

        for seed in seeds:
            for task_family, task_ids in (("arithmetic", arith_task_ids), ("code", code_task_ids)):
                for task_id in task_ids:
                    prompt, max_tokens = _prompt_and_tokens_initial(task_family, task_id)
                    raw = generate(llm, model_id, prompt, temperature=BASE_TEMPERATURE, top_p=TOP_P, seed=seed, max_tokens=max_tokens)
                    seam_out = _checked(task_family, task_id, raw)
                    record = _make_record(
                        task_id=task_id, collision_id=None, arm_id="INITIAL", model_id=model_id,
                        temperature=BASE_TEMPERATURE, seed=seed, prompt=prompt, initial_artifact="",
                        raw=raw, task_family=task_family, seam_output=seam_out, task=_get_task(task_family, task_id),
                    )
                    all_records.append(record)

                    if seam_out["status"] == FAIL:
                        collision_id = f"col:{model_id}:{task_family}:{task_id}:seed{seed}"
                        record["collision_id"] = collision_id
                        states.append(CollisionState(
                            collision_id=collision_id, task_family=task_family, task_id=task_id,
                            model_id=model_id, seed=seed, initial_artifact=raw.get("raw_text", ""),
                            initial_seam_result=seam_out, initial_record=record,
                        ))
        del llm
        log(f"=== {model_id}: unloaded ===")

    return states, all_records


# -- 7 плеч -------------------------------------------------------------------


@dataclass(frozen=True)
class ArmConfig:
    arm_id: str
    level: str            # "K0" | "K1" | "K2"
    prompt_variant: str   # "standard" | "promptB"
    model_id: str
    seed: int
    temperature: float

    def signature(self) -> tuple:
        return (self.level, self.prompt_variant, self.model_id, self.seed, self.temperature)


def build_arm_configs(base_model_id: str, all_model_ids, rng: random.Random, retry_seed: int = RETRY_SEED) -> dict:
    """Один и тот же alt_model для ВСЕХ трёх O1-плеч этого состояния --
    иначе тест взаимодействия (K_iO1-K_iO0)-(K0O1-K0O0) сравнивал бы
    разные модели на разных уровнях контекста, что его обессмысливает."""
    others = [m for m in all_model_ids if m != base_model_id]
    alt_model = rng.choice(others) if others else base_model_id

    configs = {}
    for level in ("K0", "K1", "K2"):
        configs[f"{level}O0"] = ArmConfig(f"{level}O0", level, "standard", base_model_id, retry_seed, BASE_TEMPERATURE)
        configs[f"{level}O1"] = ArmConfig(f"{level}O1", level, "standard", alt_model, retry_seed, BASE_TEMPERATURE)
    configs[ARM_H_SYN] = ArmConfig(ARM_H_SYN, "K0", "promptB", base_model_id, retry_seed, BASE_TEMPERATURE)
    return configs


def run_arm(state: CollisionState, config: ArmConfig, llm) -> dict:
    task = _get_task(state.task_family, state.task_id)
    operator_mode = "O1" if config.model_id != state.model_id else "O0"

    if config.prompt_variant == "promptB":
        prompt = build_prompt_promptB(state.task_family, task, state.initial_artifact, state.initial_seam_result)
    else:
        prompt = build_prompt(config.level, state.task_family, task, state.initial_artifact, state.initial_seam_result)

    manifest_check = validate_prompt_against_manifest(
        config.level, state.task_family, task, state.initial_artifact, state.initial_seam_result, prompt
    )
    if not manifest_check["ok"]:
        raise LeakDetected(
            f"MANIFEST VIOLATION in {config.arm_id} for {state.collision_id}: {manifest_check['violations']}"
        )

    max_tokens = MAX_TOKENS_ARITH if state.task_family == "arithmetic" else MAX_TOKENS_CODE
    raw = generate(llm, config.model_id, prompt, temperature=config.temperature, top_p=TOP_P, seed=config.seed, max_tokens=max_tokens)
    seam_out = _checked(state.task_family, state.task_id, raw)

    record = _make_record(
        task_id=state.task_id, collision_id=state.collision_id, arm_id=config.arm_id, model_id=config.model_id,
        temperature=config.temperature, seed=config.seed, prompt=prompt, initial_artifact=state.initial_artifact,
        raw=raw, task_family=state.task_family, seam_output=seam_out, task=task,
        initial_seam_result=state.initial_seam_result, context_level=config.level, operator_mode=operator_mode,
    )

    return {
        "collision_id": state.collision_id,
        "arm": config.arm_id,
        "level": config.level,
        "operator": operator_mode,
        "config": asdict(config),
        "status": seam_out["status"],
        "final_status": record["final_status"],
        "failure_type": record["failure_type"],
        "regressions": record["regressions"],
        "improvements": record["improvements"],
        "solved": seam_out["status"] == PASS,
        "tokens": record["total_tokens"],
        "latency": record["latency"],
        "leak_check_passed": True,
        "record": record,
    }


def n_confirmed_steps(state: CollisionState) -> int:
    """§8 задания: сколько шагов/requirements было подтверждено PASS на
    ПЕРВИЧНОЙ генерации (материал для K2). Финальный шаг арифметики
    никогда не PASS для состояния коллизии по определению (final
    неверен -> это и есть коллизия) -- считается отдельно ниже, если
    понадобится, здесь считаем ВСЕ подтверждённые, включая
    неитоговые."""
    if state.task_family == "arithmetic":
        return sum(1 for r in state.initial_seam_result["step_results"] if r["status"] == PASS)
    return sum(1 for r in state.initial_seam_result["requirement_results"] if r["status"] == PASS)


# -- статистика ---------------------------------------------------------------


def paired_sign_test(a_solved: list, b_solved: list) -> dict:
    n10 = sum(1 for a, b in zip(a_solved, b_solved) if a and not b)
    n01 = sum(1 for a, b in zip(a_solved, b_solved) if b and not a)
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        p = min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2**n * 2)
    return {"n_a_wins": n10, "n_b_wins": n01, "n_ties": len(a_solved) - n, "p_value": p}


def bootstrap_ratio_diff(arm_attempts: list, control_attempts: list, n_resamples: int = 10000, seed: int = 12345) -> dict:
    rng = random.Random(seed)
    n = len(arm_attempts)
    pairs = list(zip(arm_attempts, control_attempts))

    def ratio(items):
        tok = sum(x["tokens"] for x in items)
        solved = sum(1 for x in items if x["solved"])
        return tok / solved if solved else None

    point_arm, point_control = ratio(arm_attempts), ratio(control_attempts)
    diffs = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        ra = ratio([p[0] for p in sample])
        rc = ratio([p[1] for p in sample])
        if ra is not None and rc is not None:
            diffs.append(ra - rc)
    diffs.sort()
    if not diffs:
        return {"point_arm": point_arm, "point_control": point_control, "ci_95_diff": None, "n_resamples_valid": 0}
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]
    return {
        "point_arm": point_arm, "point_control": point_control,
        "point_diff": (point_arm - point_control) if (point_arm is not None and point_control is not None) else None,
        "ci_95_diff": [lo, hi], "n_resamples_valid": len(diffs),
    }


def bootstrap_rate_diff(a_solved: list, b_solved: list, n_resamples: int = 10000, seed: int = 24680) -> dict:
    """Парный bootstrap 95% CI и effect size разницы Resolution Rate
    (a-b), a/b -- списки bool, выровненные по одним и тем же состояниям.
    Нужен для ΔK1/ΔK2/Δhetero (задание v2, "СТАТИСТИКА") -- отдельно от
    `paired_sign_test` (p-value) и `bootstrap_ratio_diff` (стоимость):
    здесь именно эффект на Resolution Rate с доверительным интервалом,
    не только знак."""
    rng = random.Random(seed)
    n = len(a_solved)
    pairs = list(zip(a_solved, b_solved))
    point = sum(a_solved) / n - sum(b_solved) / n if n else None
    diffs = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        a = sum(p[0] for p in sample) / n
        b = sum(p[1] for p in sample) / n
        diffs.append(a - b)
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)] if diffs else None
    hi = diffs[min(int(0.975 * n_resamples), n_resamples - 1)] if diffs else None
    return {"point_diff": point, "ci_95_diff": [lo, hi] if diffs else None, "n_pairs": n}


def blind_resample_baseline(states: list, initial_records: list) -> dict:
    """Честный 1-к-1 контроль (методика Эксперимента 11, §13 его отчёта):
    для каждого состояния -- вероятность решения ОДНИМ дополнительным
    слепым (без всякого контекста) повтором, взятым из уже собранных
    seed того же (модель, семья, задача). Прямая экономическая планка
    для всех 7 плеч."""
    by_key = defaultdict(list)
    for r in initial_records:
        by_key[(r["model"], r["seam_type"], r["task_id"])].append(r)

    n_attempts = 0
    n_solved = 0
    total_tokens = 0
    for s in states:
        key = (s.model_id, s.task_family, s.task_id)
        siblings = [r for r in by_key[key] if r["seed"] != s.seed]
        for r in siblings:
            n_attempts += 1
            total_tokens += r["total_tokens"]
            if r["status"] == PASS:
                n_solved += 1

    if n_attempts == 0 or n_solved == 0:
        return {"n_attempts": n_attempts, "n_solved": n_solved, "success_rate": None, "tokens_per_resolved": None}
    mean_tokens = total_tokens / n_attempts
    success_rate = n_solved / n_attempts
    return {
        "n_attempts": n_attempts, "n_solved": n_solved, "success_rate": success_rate,
        "mean_tokens_per_attempt": mean_tokens, "tokens_per_resolved": mean_tokens / success_rate,
    }


def bootstrap_interaction(k0o0: list, k0o1: list, kio0: list, kio1: list, n_resamples: int = 10000, seed: int = 54321) -> dict:
    """Bootstrap CI разности разностей по Resolution Rate:
    (KiO1-KiO0) - (K0O1-K0O0). Положительно -> смена оператора помогает
    БОЛЬШЕ при обогащённом контексте (взаимодействие, исход 3 из §16
    задания). Все четыре списка -- bool, выровнены по одним и тем же
    состояниям (парно)."""
    rng = random.Random(seed)
    n = len(k0o0)

    def dod(idxs):
        def rate(lst):
            return sum(lst[i] for i in idxs) / len(idxs)
        return (rate(kio1) - rate(kio0)) - (rate(k0o1) - rate(k0o0))

    point = dod(range(n))
    samples = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        samples.append(dod(idxs))
    samples.sort()
    lo = samples[int(0.025 * n_resamples)]
    hi = samples[min(int(0.975 * n_resamples), n_resamples - 1)]
    return {"point_estimate": point, "ci_95": [lo, hi], "n_resamples": n_resamples, "n_paired": n}
