"""harness.py — эксперимент 11: collision -> action -> outcome.

Плоский, не-Mycelium стенд: состояние коллизии — dict/dataclass без
версионирования, без графа зависимостей, без state-machine. Каждое
действие — независимый вызов, результат которого не влияет на то, что
видят другие действия того же состояния (единый замороженный старт,
п.1 задания).

EXPERIMENT_ID фиксирует кампанию (п.13: при обнаруженном дефекте
конвейера — новая кампания с НОВЫМ experiment_id, не патч поверх старой).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from configs.model_registry import MODEL_REGISTRY, MODEL_IDS, generate, load_model
from seams import ERROR, FAIL, INAPPLICABLE, PASS, arithmetic_seam, code_seam
from tasks import arithmetic_tasks, code_tasks

EXPERIMENT_ID = "exp11-v1"

BASE_TEMPERATURE = 0.5      # A/B/C — как в эксп. 10, содержательный (не жадный) sampling
ALT_TEMPERATURE = 1.0       # D (CHANGE_TEMPERATURE) — единственное альтернативное значение,
                             # зафиксировано ДО прогона (п.2D: "фиксируются до запуска")
TOP_P = 1.0
INITIAL_SEEDS = (1, 2, 3, 4)   # 4 seed нужны для best-of-4 (п.10)
RETRY_SEED = 1000              # ВСЕГДА новый seed для retry-плеч (не совпадает с 1-4:
                                # иначе REGENERATE_SAME был бы вырожден -- урок arch1/эксп.10)

MAX_TOKENS_ARITH = 60
MAX_TOKENS_CODE = 320

ARMS = ("REGENERATE_SAME", "CHANGE_MODEL", "CHANGE_PROMPT", "CHANGE_TEMPERATURE")

MIN_CATEGORY_N = 8  # ниже этого порога разбивка по типу коллизии не даёт вывода (п.9)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _characteristics(text: str) -> dict:
    text = text or ""
    return {"length_chars": len(text), "length_words": len(text.split())}


# -- состояние коллизии -----------------------------------------------------


@dataclass
class CollisionState:
    collision_id: str
    task_family: str          # "arithmetic" | "code"
    task_id: str
    model_id: str
    seed: int
    initial_artifact: str
    initial_seam_result: dict
    initial_record: dict


def _check(task_family: str, task_id: str, text: str) -> dict:
    """VISIBLE-проверка (используется исходной генерацией и всеми
    действиями A-D). Для SEEK_EVIDENCE — отдельно, holdout, см. ниже."""
    if task_family == "arithmetic":
        task = arithmetic_tasks.TASKS[task_id]
        return arithmetic_seam(text, task["answer"])
    task = code_tasks.CODE_TASKS[task_id]
    return code_seam(text, task["function_name"], task["visible_tests"])


def _prompt_and_tokens(task_family: str, task_id: str):
    if task_family == "arithmetic":
        task = arithmetic_tasks.TASKS[task_id]
        return arithmetic_tasks.generation_prompt(task), MAX_TOKENS_ARITH
    task = code_tasks.CODE_TASKS[task_id]
    return code_tasks.generation_prompt(task), MAX_TOKENS_CODE


def _make_record(
    *, experiment_id, task_id, collision_id, action_id, model_id, temperature, top_p, seed,
    prompt, initial_artifact, raw, seam_type, seam_input, seam_output,
) -> dict:
    entry = MODEL_REGISTRY[model_id]
    result_text = raw.get("raw_text", "")
    input_tokens = raw.get("input_tokens") or 0
    output_tokens = raw.get("output_tokens") or 0
    return {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "collision_id": collision_id,
        "action_id": action_id,
        "model": model_id,
        "model_family": entry["family"],
        "model_size": entry["size_label"],
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "prompt": prompt,
        "prompt_characteristics": _characteristics(prompt),
        "initial_artifact": initial_artifact,
        "result": result_text,
        "result_characteristics": _characteristics(result_text),
        "seam_type": seam_type,
        "seam_input": seam_input,
        "seam_output": seam_output,
        "status": seam_output["status"],
        "collision_type": seam_output.get("collision_type"),
        "evidence": seam_output.get("details"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency": raw.get("generation_time_sec"),
        "timestamp": _now_iso(),
        "generation_failed": raw.get("generation_failed", False),
        "generation_error": raw.get("generation_error"),
    }


def _checked(task_family: str, task_id: str, raw: dict) -> dict:
    """Оборачивает вызов _check(): исключение самого конвейера проверки
    (не кандидата -- те уже пойманы внутри seams.py и превращены в
    FAIL/INAPPLICABLE) -> ERROR, никогда не FAIL (п.3)."""
    if raw.get("generation_failed"):
        return {"status": ERROR, "collision_type": None, "details": {"reason": raw.get("generation_error")}}
    try:
        return _check(task_family, task_id, raw.get("raw_text", ""))
    except Exception as exc:  # сбой САМОГО конвейера проверки
        return {"status": ERROR, "collision_type": None, "details": {"reason": f"{type(exc).__name__}: {exc}"}}


# -- фаза 0: поиск состояний коллизии ---------------------------------------


def find_collision_states(
    model_ids, seeds, arithmetic_task_ids, code_task_ids, log=print
) -> tuple:
    """Возвращает (states: [CollisionState], all_records: [dict]).

    Все НАЧАЛЬНЫЕ попытки сохраняются (не только FAIL) -- нужны для
    best-of-N (п.10) и для честного знаменателя (сколько задач вообще
    было решено с первой попытки)."""
    states: list = []
    all_records: list = []

    for model_id in model_ids:
        log(f"=== {model_id}: loading ===")
        llm, load_t = load_model(model_id)
        log(f"=== {model_id}: loaded in {load_t:.1f}s ===")

        for seed in seeds:
            for task_family, task_ids in (("arithmetic", arithmetic_task_ids), ("code", code_task_ids)):
                for task_id in task_ids:
                    prompt, max_tokens = _prompt_and_tokens(task_family, task_id)
                    raw = generate(
                        llm, model_id, prompt, temperature=BASE_TEMPERATURE, top_p=TOP_P,
                        seed=seed, max_tokens=max_tokens,
                    )
                    seam_out = _checked(task_family, task_id, raw)
                    record = _make_record(
                        experiment_id=EXPERIMENT_ID, task_id=task_id, collision_id=None,
                        action_id="INITIAL", model_id=model_id, temperature=BASE_TEMPERATURE,
                        top_p=TOP_P, seed=seed, prompt=prompt, initial_artifact="", raw=raw,
                        seam_type=task_family, seam_input={"task_id": task_id}, seam_output=seam_out,
                    )
                    all_records.append(record)

                    if seam_out["status"] == FAIL:
                        collision_id = f"col:{model_id}:{task_family}:{task_id}:seed{seed}"
                        record["collision_id"] = collision_id
                        states.append(
                            CollisionState(
                                collision_id=collision_id, task_family=task_family, task_id=task_id,
                                model_id=model_id, seed=seed, initial_artifact=raw.get("raw_text", ""),
                                initial_seam_result=seam_out, initial_record=record,
                            )
                        )
        del llm
        log(f"=== {model_id}: unloaded ===")

    return states, all_records


# -- плечи A-D ----------------------------------------------------------------


@dataclass(frozen=True)
class ArmConfig:
    arm: str
    model_id: str
    temperature: float
    prompt_profile: str  # "v1" | "v2"
    seed: int

    def signature(self) -> tuple:
        return (self.model_id, self.temperature, self.prompt_profile, self.seed)


def build_arm_configs(base_model_id: str, all_model_ids, rng: random.Random, retry_seed: int = RETRY_SEED) -> dict:
    others = [m for m in all_model_ids if m != base_model_id]
    alt_model = rng.choice(others) if others else base_model_id
    return {
        "REGENERATE_SAME": ArmConfig("REGENERATE_SAME", base_model_id, BASE_TEMPERATURE, "v1", retry_seed),
        "CHANGE_MODEL": ArmConfig("CHANGE_MODEL", alt_model, BASE_TEMPERATURE, "v1", retry_seed),
        "CHANGE_PROMPT": ArmConfig("CHANGE_PROMPT", base_model_id, BASE_TEMPERATURE, "v2", retry_seed),
        "CHANGE_TEMPERATURE": ArmConfig("CHANGE_TEMPERATURE", base_model_id, ALT_TEMPERATURE, "v1", retry_seed),
    }


def _describe_evidence(seam_result: dict) -> str:
    ctype = seam_result.get("collision_type")
    details = seam_result.get("details", {})
    if ctype == "numeric":
        return f"the computed value {details.get('actual')} does not match the required result"
    if ctype == "syntactic":
        return f"the code has a syntax/execution error ({details.get('reason', '')})"
    if ctype == "structural":
        return f"the required function was not defined correctly ({details.get('reason', '')})"
    if ctype == "logical":
        failing = "; ".join(details.get("failing_tests", [])[:2])
        return f"the code runs but produces wrong results on test cases ({failing})"
    return f"status={seam_result.get('status')}"


def classify_repeat(before: dict, after: dict) -> str:
    """п.8: повторилась/изменилась/исчезла/новая ошибка."""
    if after["status"] == PASS:
        return "fixed"
    if after["status"] != FAIL:
        return f"became_{after['status'].lower()}"
    if before.get("collision_type") != after.get("collision_type"):
        return "different_failure_type"
    ctype = before.get("collision_type")
    bd, ad = before.get("details", {}), after.get("details", {})
    if ctype == "numeric":
        same = bd.get("actual") == ad.get("actual")
    elif ctype == "syntactic":
        same = bd.get("reason") == ad.get("reason")
    elif ctype == "structural":
        same = bd.get("found") == ad.get("found")
    elif ctype == "logical":
        same = set(bd.get("failing_tests", [])) == set(ad.get("failing_tests", []))
    else:
        same = bd == ad
    return "same_failure" if same else "different_failure_value"


def _retry_prompt(task_family: str, task_id: str, prompt_profile: str, prior_artifact: str, evidence: str) -> str:
    if task_family == "arithmetic":
        task = arithmetic_tasks.TASKS[task_id]
        fn = arithmetic_tasks.retry_prompt if prompt_profile == "v1" else arithmetic_tasks.retry_prompt_v2
    else:
        task = code_tasks.CODE_TASKS[task_id]
        fn = code_tasks.retry_prompt if prompt_profile == "v1" else code_tasks.retry_prompt_v2
    return fn(task, prior_artifact, evidence)


def run_arm(state: CollisionState, config: ArmConfig, llm) -> dict:
    _prompt_probe, max_tokens = _prompt_and_tokens(state.task_family, state.task_id)
    evidence = _describe_evidence(state.initial_seam_result)
    prompt = _retry_prompt(state.task_family, state.task_id, config.prompt_profile, state.initial_artifact, evidence)

    raw = generate(
        llm, config.model_id, prompt, temperature=config.temperature, top_p=TOP_P,
        seed=config.seed, max_tokens=max_tokens,
    )
    seam_out = _checked(state.task_family, state.task_id, raw)

    record = _make_record(
        experiment_id=EXPERIMENT_ID, task_id=state.task_id, collision_id=state.collision_id,
        action_id=config.arm, model_id=config.model_id, temperature=config.temperature,
        top_p=TOP_P, seed=config.seed, prompt=prompt, initial_artifact=state.initial_artifact,
        raw=raw, seam_type=state.task_family, seam_input={"task_id": state.task_id}, seam_output=seam_out,
    )

    return {
        "collision_id": state.collision_id,
        "arm": config.arm,
        "config": asdict(config),
        "status": seam_out["status"],
        "solved": seam_out["status"] == PASS,
        "same_failure": classify_repeat(state.initial_seam_result, seam_out) if seam_out["status"] != ERROR else None,
        "tokens": record["total_tokens"],
        "latency": record["latency"],
        "record": record,
    }


# -- действие E: SEEK_EVIDENCE (без генерации) -------------------------------


def run_seek_evidence(state: CollisionState) -> dict:
    """Не генерирует. Для code -- прогоняет ОТЛОЖЕННЫЕ (held-out) тесты
    против УЖЕ ИМЕЮЩЕГОСЯ артефакта. Для arithmetic -- исходная проверка
    уже была точным механическим сравнением (максимальное evidence с
    самого начала), добавить нечего -> честный INAPPLICABLE (п.2)."""
    if state.task_family != "code":
        return {
            "collision_id": state.collision_id, "arm": "SEEK_EVIDENCE",
            "status": INAPPLICABLE, "solved": False, "tokens": 0,
            "note": "arithmetic: initial check already used exact mechanical oracle, no further evidence obtainable",
        }
    task = code_tasks.CODE_TASKS[state.task_id]
    try:
        seam_out = code_seam(state.initial_artifact, task["function_name"], task["holdout_tests"])
    except Exception as exc:
        seam_out = {"status": ERROR, "collision_type": None, "details": {"reason": f"{type(exc).__name__}: {exc}"}}
    return {
        "collision_id": state.collision_id, "arm": "SEEK_EVIDENCE",
        "status": seam_out["status"],
        # holdout PASS на артефакте, УЖЕ провалившем visible -- не "решение",
        # а находка (узкий отказ, не тотальный) -- см. REPORT §5, отдельная метрика
        "holdout_passes_despite_visible_fail": seam_out["status"] == PASS,
        "solved": False,  # SEEK_EVIDENCE не регенерирует -> не может "решить" по определению
        "tokens": 0,
        "seam_output": seam_out,
    }


def run_stop(state: CollisionState) -> dict:
    return {
        "collision_id": state.collision_id, "arm": "STOP",
        "status": state.initial_seam_result["status"], "solved": False, "tokens": 0,
        "note": "no action taken",
    }


# -- best-of-N (п.10, из уже накопленных начальных попыток) -----------------


def best_of_n(all_initial_records: list) -> dict:
    by_key: dict = {}
    for r in all_initial_records:
        key = (r["model"], r["seam_type"], r["task_id"])
        by_key.setdefault(key, []).append(r)

    import itertools

    out = {}
    for n in (1, 2, 3, 4):
        oracle_ok = majority_ok = cnt = 0
        token_sum = 0
        for _key, samples in by_key.items():
            if len(samples) < n:
                continue
            for combo in itertools.combinations(samples, n):
                cnt += 1
                token_sum += sum(s["total_tokens"] for s in combo)
                if any(s["status"] == PASS for s in combo):
                    oracle_ok += 1
                results_text = [s["result"] for s in combo]
                modal = max(set(results_text), key=results_text.count)
                picked = next(s for s in combo if s["result"] == modal)
                if picked["status"] == PASS:
                    majority_ok += 1
        if cnt:
            out[f"N={n}"] = {
                "n_combinations": cnt,
                "mean_tokens": token_sum / cnt,
                "oracle_success_rate": oracle_ok / cnt,
                "majority_success_rate": majority_ok / cnt,
                "oracle_tokens_per_success": (token_sum / cnt) / (oracle_ok / cnt) if oracle_ok else None,
                "majority_tokens_per_success": (token_sum / cnt) / (majority_ok / cnt) if majority_ok else None,
            }
    return out


# -- статистика (метод и порог фиксированы ДО прогона, см. план) -----------


def paired_sign_test(a_solved: list, b_solved: list) -> dict:
    """Точный биномиальный (Макнемар) тест по совпадающим состояниям."""
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
    """Парный bootstrap 95% CI разницы TOKENS_PER_RESOLVED_TASK (arm -
    control) -- ресэмплирование ПАР состояний сохраняет парность."""
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
