"""harness.py — эксперимент 13: representation shift.

Три режима (+ R0B опционально) на ОДНОЙ и той же collision-state, ОДНОЙ
и той же моделью (той, что породила исходную ошибку -- без CHANGE_MODEL,
намеренно, чтобы не смешивать факторы с программой экспериментов 10-12).
Состояния переиспользуются из `experiment12/runs12/phase0_full_exp12-v1.json`
-- НЕ генерируются заново (задание: "не менять датасет без необходимости").

`classify_outcome`/`detect_regressions`/`detect_improvements` переиспользуются
от `experiment12/outcome_classification.py` (та же importlib-загрузка по
явному пути, что уже трижды применялась в проекте против коллизии имён
модулей). `_normalize_seam_for_comparison()` -- необходимая надстройка:
у R1/R2 (final_answer_seam) финальный шаг всегда называется "FINAL", а
у R0/исходной генерации (multistep_arithmetic_seam) -- настоящим именем
шага (напр. "total"); без нормализации имени сравнение сигнатур ошибки
всегда давало бы DIFFERENT_FAILURE даже при побайтово одинаковом неверном
числе -- инструмент искажал бы именно то, что измеряется.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from configs.model_registry import MODEL_REGISTRY, generate, load_model
from context_builder import build_prompt, validate_prompt
from representation import (
    INVALID_STRUCTURE,
    arithmetic_fingerprint_from_task,
    arithmetic_fingerprint_from_text,
    classify_novelty,
    classify_pseudo_novelty,
    code_fingerprint,
    parse_sections,
    representation_text,
)
from seams import ERROR, FAIL, INAPPLICABLE, PASS, enriched_code_seam, final_answer_seam, multistep_arithmetic_seam
from tasks import ARITH_TASKS, CODE_TASKS

_EXP12_ROOT = Path(__file__).resolve().parents[1] / "experiment12"

_spec_oc = importlib.util.spec_from_file_location("experiment12_outcome_classification_for_exp13", _EXP12_ROOT / "outcome_classification.py")
_oc = importlib.util.module_from_spec(_spec_oc)
_spec_oc.loader.exec_module(_oc)
classify_outcome = _oc.classify_outcome
detect_regressions = _oc.detect_regressions
detect_improvements = _oc.detect_improvements

_spec_seams12 = importlib.util.spec_from_file_location("experiment12_seams_for_exp13_subtype", _EXP12_ROOT / "seams.py")
_seams12 = importlib.util.module_from_spec(_spec_seams12)
_spec_seams12.loader.exec_module(_seams12)
collision_subtype = _seams12.collision_subtype

EXPERIMENT_ID = "exp13-v1"

BASE_TEMPERATURE = 0.5
TOP_P = 1.0
RETRY_SEED = 1000  # новый seed для всех retry-режимов -- тот же принцип, что во всех предыдущих экспериментах

MAX_TOKENS_ARITH_R0 = 150
MAX_TOKENS_ARITH_STRUCTURED = 400   # R1/R2/R0B: описание представления + решение -- нужен больший бюджет
MAX_TOKENS_CODE_R0 = 320
MAX_TOKENS_CODE_STRUCTURED = 500

MODES_MAIN = ("R0", "R1", "R2")
ALL_MODES = ("R0", "R1", "R2", "R0B")

_EXP12_PHASE0_PATH = _EXP12_ROOT / "runs12" / "phase0_full_exp12-v1.json"


class PromptViolation(RuntimeError):
    """Промпт не прошёл проверку (подсказка метода / утечка структуры
    другого режима / артефакт подменён) -- кампания должна остановиться."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _characteristics(text: str) -> dict:
    text = text or ""
    return {"length_chars": len(text), "length_words": len(text.split())}


def _prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _get_task(task_family: str, task_id: str) -> dict:
    return ARITH_TASKS[task_id] if task_family == "arithmetic" else CODE_TASKS[task_id]


# -- состояние коллизии (переиспользовано из Эксп.12) -------------------------


@dataclass
class CollisionState13:
    collision_id: str
    task_family: str
    task_id: str
    model_id: str
    seed: int
    initial_artifact: str
    initial_seam_result: dict


def load_exp12_states(path: Path = _EXP12_PHASE0_PATH) -> list:
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        CollisionState13(
            collision_id=s["collision_id"], task_family=s["task_family"], task_id=s["task_id"],
            model_id=s["model_id"], seed=s["seed"], initial_artifact=s["initial_artifact"],
            initial_seam_result=s["initial_seam_result"],
        )
        for s in data["states"]
    ]


def select_states(states: list, n: int, seed: int, arith_fraction: float = 0.7) -> list:
    arith = [s for s in states if s.task_family == "arithmetic"]
    code = [s for s in states if s.task_family == "code"]
    rng = random.Random(seed)
    n_arith = min(round(n * arith_fraction), len(arith))
    n_code = min(n - n_arith, len(code))
    selected = rng.sample(arith, n_arith) + rng.sample(code, n_code)
    rng.shuffle(selected)
    return selected


# -- проверка/классификация ----------------------------------------------------


def _max_tokens(task_family: str, mode: str) -> int:
    if task_family == "arithmetic":
        return MAX_TOKENS_ARITH_R0 if mode == "R0" else MAX_TOKENS_ARITH_STRUCTURED
    return MAX_TOKENS_CODE_R0 if mode == "R0" else MAX_TOKENS_CODE_STRUCTURED


def _check(mode: str, task_family: str, task_id: str, raw: dict) -> dict:
    if raw.get("generation_failed"):
        return {"status": ERROR, "collision_type": None, "details": {"reason": raw.get("generation_error")}}
    task = _get_task(task_family, task_id)
    text = raw.get("raw_text", "")
    try:
        if task_family == "arithmetic":
            if mode == "R0":
                return multistep_arithmetic_seam(text, task["steps"])
            return final_answer_seam(text, task["answer"])
        return enriched_code_seam(text, task["function_name"], task["requirements"])
    except Exception as exc:
        return {"status": ERROR, "collision_type": None, "details": {"reason": f"{type(exc).__name__}: {exc}"}}


def _normalize_seam_for_comparison(seam_result: dict) -> dict:
    """Финальный шаг R0/исходной генерации называется настоящим именем
    (напр. 'total'), а final_answer_seam (R1/R2) -- всегда 'FINAL'. Без
    нормализации сравнение сигнатур ошибки в classify_outcome всегда
    давало бы DIFFERENT_FAILURE даже при идентичном неверном числе."""
    if seam_result.get("collision_type") != "numeric":
        return seam_result
    details = dict(seam_result.get("details") or {})
    final = dict(details.get("final") or {})
    if "name" in final:
        final["name"] = "FINAL"
        details["final"] = final
    return {**seam_result, "details": details}


def _structure_fingerprints(mode: str, task_family: str, task: dict, state: "CollisionState13", solution_text: str, raw_text: str):
    if task_family == "arithmetic":
        before_fp = arithmetic_fingerprint_from_task(task)
        after_fp = arithmetic_fingerprint_from_text(solution_text)
    else:
        before_fp = code_fingerprint(state.initial_artifact)
        after_fp = code_fingerprint(solution_text or raw_text)
    return before_fp, after_fp


def _old_decomposition_text(task_family: str, task: dict) -> str:
    if task_family == "arithmetic":
        return "\n".join(f"{s['name']} = {s['formula']}" for s in task["steps"])
    return "\n".join(f"{r['name']}: {r['description']}" for r in task["requirements"])


def _failure_type(state: "CollisionState13") -> str:
    steps = ARITH_TASKS[state.task_id]["steps"] if state.task_family == "arithmetic" else None
    return collision_subtype(state.task_family, state.initial_seam_result, steps)


def make_stop_predicate(mode: str, task_family: str, task: dict):
    """Ранняя остановка потоковой генерации, см. обоснование в
    `configs/model_registry.py::generate`. Не подсказывает МЕТОД решения
    -- только механически распознаёт, что ответ уже структурно
    завершён, по тому же контракту, что задан промптом (§2 задания:
    строгий формат шагов для R0, маркер `FINAL ANSWER =` для остальных
    режимов арифметики, закрытый код-блок с нужной функцией для кода)."""
    if task_family == "arithmetic":
        if mode == "R0":
            last_name = re.escape(task["steps"][-1]["name"])

            def predicate(text: str) -> bool:
                m = re.search(rf"(?im)^\s*{last_name}\s*=.*$", text)
                if not m:
                    return False
                return len(text) - m.end() >= 2  # хотя бы пара символов после строки -- значение дописано, не обрывается

            return predicate

        def predicate(text: str) -> bool:
            m = re.search(r"(?i)FINAL ANSWER\s*=\s*-?\d[\d.]*", text)
            if not m:
                return False
            return len(text) - m.end() >= 2

        return predicate

    function_name = re.escape(task["function_name"])

    def predicate(text: str) -> bool:
        fences = [m.start() for m in re.finditer(r"```", text)]
        if len(fences) < 2:
            return False
        block = text[fences[-2]:fences[-1]]
        return bool(re.search(rf"\bdef\s+{function_name}\b", block))

    return predicate


# -- один вызов -----------------------------------------------------------------


def run_mode(state: CollisionState13, mode: str, llm) -> dict:
    task = _get_task(state.task_family, state.task_id)
    prompt = build_prompt(mode, state.task_family, task, state.initial_artifact, state.initial_seam_result)

    validation = validate_prompt(mode, state.initial_artifact, prompt)
    if not validation["ok"]:
        raise PromptViolation(f"{mode}/{state.collision_id}: {validation['violations']}")

    max_tokens = _max_tokens(state.task_family, mode)
    stop_predicate = make_stop_predicate(mode, state.task_family, task)
    raw = generate(
        llm, state.model_id, prompt, temperature=BASE_TEMPERATURE, top_p=TOP_P, seed=RETRY_SEED,
        max_tokens=max_tokens, stop_predicate=stop_predicate,
    )
    seam_out = _check(mode, state.task_family, state.task_id, raw)

    final_status = classify_outcome(_normalize_seam_for_comparison(state.initial_seam_result), _normalize_seam_for_comparison(seam_out))
    regressions = detect_regressions(state.initial_seam_result, seam_out)
    improvements = detect_improvements(state.initial_seam_result, seam_out)

    sections = parse_sections(raw.get("raw_text", "")) if mode != "R0" else {}
    new_representation = representation_text(mode, sections) if mode in ("R1",) else ""
    new_decomposition = ""
    if mode == "R2":
        parts = [sections.get("DECOMPOSITION DECISION", ""), sections.get("NEW DECOMPOSITION", "")]
        new_decomposition = "\n".join(p for p in parts if p)

    novelty = None
    pseudo_novelty = None
    structure_before = structure_after = None
    if mode in ("R1", "R2", "R0B"):
        solution_text = sections.get("SOLUTION", "")
        structure_before, structure_after = _structure_fingerprints(mode, state.task_family, task, state, solution_text, raw.get("raw_text", ""))
        novelty = classify_novelty(state.task_family, structure_before, structure_after)
        pseudo_novelty = classify_pseudo_novelty(state.task_family, structure_before, structure_after)

    input_tokens = raw.get("input_tokens") or 0
    output_tokens = raw.get("output_tokens") or 0
    entry = MODEL_REGISTRY[state.model_id]

    record = {
        "experiment_id": EXPERIMENT_ID,
        "task_id": state.task_id,
        "collision_id": state.collision_id,
        "model": state.model_id,
        "model_settings": {"family": entry["family"], "size_label": entry["size_label"]},
        "temperature": BASE_TEMPERATURE,
        "prompt_id": _prompt_id(prompt),
        "prompt": prompt,
        "prompt_characteristics": _characteristics(prompt),
        "mode": mode,
        "input_artifact": state.initial_artifact,
        "evidence": state.initial_seam_result.get("details"),
        "old_representation": _old_decomposition_text(state.task_family, task),
        "new_representation": new_representation,
        "old_decomposition": _old_decomposition_text(state.task_family, task),
        "new_decomposition": new_decomposition,
        "structure_before": structure_before,
        "structure_after": structure_after,
        "output": raw.get("raw_text", ""),
        "output_characteristics": _characteristics(raw.get("raw_text", "")),
        "classification": pseudo_novelty,
        "representation_novelty": novelty,
        "failure_type": _failure_type(state),
        "mechanical_checks": seam_out.get("step_results") or seam_out.get("requirement_results") or [],
        "tokens_in": input_tokens,
        "tokens_out": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency": raw.get("generation_time_sec"),
        "final_status": final_status,
        "regressions": regressions,
        "improvements": improvements,
        "time_to_first_pass": raw.get("generation_time_sec") if final_status == "PASS" else None,
        "timestamp": _now_iso(),
        "generation_failed": raw.get("generation_failed", False),
        "generation_error": raw.get("generation_error"),
    }

    return {
        "collision_id": state.collision_id, "mode": mode, "status": seam_out.get("status"),
        "final_status": final_status, "solved": final_status == "PASS",
        "representation_novelty": novelty, "pseudo_novelty": pseudo_novelty,
        "failure_type": record["failure_type"], "tokens": record["total_tokens"], "latency": record["latency"],
        "record": record,
    }


# -- статистика (та же реализация, что experiment12/harness.py -- не
# импортируется оттуда напрямую из-за коллизии имени модуля "harness",
# см. обоснование в experiment12/harness.py) --------------------------------


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


def bootstrap_rate_diff(a_solved: list, b_solved: list, n_resamples: int = 10000, seed: int = 24680) -> dict:
    rng = random.Random(seed)
    n = len(a_solved)
    if n == 0:
        return {"point_diff": None, "ci_95_diff": None, "n_pairs": 0}
    pairs = list(zip(a_solved, b_solved))
    point = sum(a_solved) / n - sum(b_solved) / n
    diffs = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        a = sum(p[0] for p in sample) / n
        b = sum(p[1] for p in sample) / n
        diffs.append(a - b)
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[min(int(0.975 * n_resamples), n_resamples - 1)]
    return {"point_diff": point, "ci_95_diff": [lo, hi], "n_pairs": n}


def unpaired_rate_diff(a_solved: list, b_solved: list, n_resamples: int = 10000, seed: int = 13579) -> dict:
    """Для DeltaNovel: NOVEL_STRUCTURE vs SAME_STRUCTURE -- РАЗНЫЕ
    подвыборки (по исходу классификации), НЕ парное сравнение одного и
    того же состояния. Ресэмплирование независимое по каждой группе."""
    rng = random.Random(seed)
    na, nb = len(a_solved), len(b_solved)
    if na == 0 or nb == 0:
        return {"point_diff": None, "ci_95_diff": None, "n_a": na, "n_b": nb}
    point = sum(a_solved) / na - sum(b_solved) / nb
    diffs = []
    for _ in range(n_resamples):
        ra = sum(a_solved[rng.randrange(na)] for _ in range(na)) / na
        rb = sum(b_solved[rng.randrange(nb)] for _ in range(nb)) / nb
        diffs.append(ra - rb)
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[min(int(0.975 * n_resamples), n_resamples - 1)]
    return {"point_diff": point, "ci_95_diff": [lo, hi], "n_a": na, "n_b": nb}
