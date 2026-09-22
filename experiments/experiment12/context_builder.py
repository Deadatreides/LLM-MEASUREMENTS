"""context_builder.py — три уровня retry-контекста (K0/K1/K2) и
антиутечная проверка (§3, §5 TASK_EXPERIMENT12.md).

K0 — evidence-only: задача + предыдущий артефакт + механическое evidence
о провале (тот же принцип, что REGENERATE_SAME в Эксперименте 11).
K1 — K0 + структурная декомпозиция БЕЗ единого числового значения. Для
арифметики строится из AST формулы шага (`describe_step`), где ЛЮБОЙ
числовой литерал заменяется плейсхолдером -- утечка структурно
невозможна, не только проверена постфактум. Для кода — список названий
и поведенческих описаний requirements (без check-выражений).
K2 — K1 + значения/статусы ТЕХ шагов/requirements, чей статус на
ПЕРВИЧНОЙ генерации уже PASS.

Промпт для операторов O0/O1 внутри одного уровня контекста строится
ОДНОЙ и той же функцией с одними и теми же аргументами -- побайтовая
идентичность гарантирована архитектурой модуля, а не соглашением
(проверяется тестом `test_experiment12.py::CommonStartingPointTests`).
"""

from __future__ import annotations

import ast
import re

from seams import PASS

_PLACEHOLDER = "<value from the problem statement>"
# ASCII-only символы операторов -- не эстетика: не-ASCII здесь (например
# "×"/"÷") ломает вывод на Windows-консоли с кодировкой по умолчанию не
# UTF-8 именно в момент печати диагностики LeakDetected, то есть на самом
# критичном пути (поймано на пилоте вместе с багом границ чисел ниже).
_OP_SYMBOLS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def _describe_node(node) -> str:
    if isinstance(node, ast.Expression):
        return _describe_node(node.body)
    if isinstance(node, ast.BinOp) and type(node.op) in _OP_SYMBOLS:
        op = _OP_SYMBOLS[type(node.op)]
        return f"({_describe_node(node.left)} {op} {_describe_node(node.right)})"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return _PLACEHOLDER
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(-{_describe_node(node.operand)})"
    return "<expression>"


def describe_step(formula: str) -> str:
    """Структурное описание формулы шага БЕЗ единого числового литерала
    -- заменяются плейсхолдером на уровне AST, не строковым поиском-
    заменой постфактум. Имена других шагов остаются (это и есть
    структура: "subtotal = shirts_cost + pants_cost")."""
    tree = ast.parse(formula, mode="eval")
    return _describe_node(tree)


# -- K0: evidence-only (общий для арифметики и кода принцип) ----------------


def _describe_evidence(task_family: str, seam_result: dict) -> str:
    ctype = seam_result.get("collision_type")
    details = seam_result.get("details", {})
    if task_family == "arithmetic":
        if ctype == "numeric":
            final = details["final"]
            return f"the final value ({final['name']} = {final['extracted']}) does not match the required result"
        final = details.get("final", {})
        return f"the final step ({final.get('name', '?')}) could not be read from your answer ({details.get('reason', '')})"
    if ctype == "syntactic":
        return f"the code has a syntax/execution error ({details.get('reason', '')})"
    if ctype == "structural":
        return f"the required function was not defined correctly ({details.get('reason', '')})"
    if ctype == "logical":
        failing = "; ".join(details.get("failing_requirements", [])[:3])
        return f"the code runs but fails these checks: {failing}"
    return f"status={seam_result.get('status')}"


def _arith_base_prompt(task: dict, initial_artifact: str, evidence: str) -> str:
    step_names = ", ".join(s["name"] for s in task["steps"])
    return (
        f"{task['question']}\n\n"
        f"A previous attempt answered:\n{initial_artifact}\n\n"
        f"This was mechanically verified INCORRECT: {evidence}\n\n"
        "Provide a corrected, complete answer. Report EVERY step as its own line in the "
        f"format `name = value`, using exactly these step names in this order: {step_names}.\n"
        "Output nothing except these lines -- no explanation, no extra text."
    )


def _code_base_prompt(task: dict, initial_artifact: str, evidence: str) -> str:
    return (
        f"{task['description']}\n\n"
        f"A previous attempt produced this code:\n{initial_artifact}\n\n"
        f"This was mechanically verified INCORRECT: {evidence}\n\n"
        f"Provide a corrected, complete implementation, starting with `{task['signature_hint']}`. "
        "Reply with ONLY the function definition, no explanation."
    )


# -- K1/K2 structure text ----------------------------------------------------


def _arith_structure_text(task: dict, step_results) -> str:
    confirmed = {}
    if step_results is not None:
        confirmed = {r["name"]: r for r in step_results if r["status"] == PASS}
    lines = ["Structure of this problem (step name = how it is computed):"]
    for step in task["steps"]:
        name = step["name"]
        if name in confirmed:
            lines.append(f"  {name} = {step['value']}  (already confirmed correct in your previous attempt)")
        else:
            lines.append(f"  {name} = {describe_step(step['formula'])}")
    return "\n".join(lines)


def _code_structure_text(task: dict, requirement_results) -> str:
    confirmed_names = set()
    if requirement_results is not None:
        confirmed_names = {r["name"] for r in requirement_results if r["status"] == PASS}
    lines = ["Behavioral requirements this function must satisfy:"]
    for req in task["requirements"]:
        marker = "  (already confirmed satisfied by your previous attempt)" if req["name"] in confirmed_names else ""
        lines.append(f"  - {req['name']}: {req['description']}{marker}")
    return "\n".join(lines)


def structure_text(level: str, task_family: str, task: dict, seam_result: dict) -> str:
    """level in {"K1","K2"}. K2 передаёт per-step/per-requirement статусы
    ПЕРВИЧНОЙ генерации (seam_result), K1 не передаёт статусов вовсе (None
    -- ни один шаг не считается подтверждённым)."""
    results = None
    if level == "K2":
        results = seam_result["step_results"] if task_family == "arithmetic" else seam_result["requirement_results"]
    if task_family == "arithmetic":
        return _arith_structure_text(task, results)
    return _code_structure_text(task, results)


# -- полная сборка промпта ---------------------------------------------------


def build_prompt(level: str, task_family: str, task: dict, initial_artifact: str, seam_result: dict) -> str:
    """level in {"K0","K1","K2"}. Не зависит от оператора (модели) --
    оператор влияет только на то, КАКАЯ модель это исполнит (harness.py),
    не на текст промпта."""
    evidence = _describe_evidence(task_family, seam_result)
    base = _arith_base_prompt(task, initial_artifact, evidence) if task_family == "arithmetic" else _code_base_prompt(task, initial_artifact, evidence)
    if level == "K0":
        return base
    return base + "\n\n" + structure_text(level, task_family, task, seam_result)


def build_prompt_promptB(task_family: str, task: dict, initial_artifact: str, seam_result: dict) -> str:
    """K0-уровень, альтернативная формулировка -- только для плеча
    K0O0_promptB (проверка H-син), не входит в основную 3x2 сетку.
    Зафиксирована в коде до прогона, тот же принцип, что
    retry_prompt_v2 в Эксперименте 11."""
    evidence = _describe_evidence(task_family, seam_result)
    if task_family == "arithmetic":
        step_names = ", ".join(s["name"] for s in task["steps"])
        return (
            f"Problem: {task['question']}\n\n"
            f"Someone already tried and got this:\n{initial_artifact}\n"
            f"That is wrong ({evidence}). Work through the problem step by step in your head, "
            f"then write out each step. Report EVERY step as its own line in the format "
            f"`name = value`, using exactly these step names in this order: {step_names}.\n"
            "Your entire reply must be only these lines."
        )
    return (
        f"Fix this Python function so it correctly does the following:\n{task['description']}\n\n"
        f"Broken version:\n{initial_artifact}\n\n"
        f"Problem found: {evidence}\n\n"
        f"Think about which input breaks it, then write a fixed version starting with "
        f"`{task['signature_hint']}`. Output only the corrected code."
    )


# -- антиутечная проверка (§5) -----------------------------------------------


def _num_variants(value: float) -> list:
    variants = {str(value)}
    if float(value).is_integer():
        variants.add(str(int(value)))
    variants.add(f"{value:.1f}")
    variants.add(f"{value:.2f}")
    return [v for v in variants if v]


def forbidden_values_for_level(level: str, task_family: str, task: dict, seam_result: dict) -> list:
    """Значения, которые НИ В КОЕМ СЛУЧАЕ не должны встречаться в
    structure_text() для данного уровня. K1: ВСЕ значения шагов/
    requirements (K1 в принципе не должен содержать чисел). K2: значения
    шагов, НЕ подтверждённых как PASS на первичной генерации (финальный
    шаг коллизии по определению не PASS -- всегда в этом списке)."""
    if task_family != "arithmetic":
        return []  # у кода нет числового "ответа" -- утечка невозможна на этом уровне (requirements это категории, не значения)
    if level == "K1":
        return [s["value"] for s in task["steps"]]
    if level == "K2":
        confirmed = {r["name"] for r in seam_result["step_results"] if r["status"] == PASS}
        return [s["value"] for s in task["steps"] if s["name"] not in confirmed]
    return []


def check_no_leak(text: str, forbidden_values: list) -> bool:
    """True -> чисто. False -> найдена утечка (значение, которое не
    должно было появиться, встретилось в тексте).

    Границы числа проверяются регулярным выражением (символ до/после
    совпадения не цифра и не точка), а не голым `in` -- иначе короткий
    запрещённый вариант вроде "3" (bare-integer форма для 3.0) ложно
    сработал бы на "320.0" (подстрока "3" есть, но это другое число).
    Поймано и исправлено на пилоте эксперимента 12 -- см. STATUS/отчёт."""
    for value in forbidden_values:
        for variant in _num_variants(value):
            pattern = r"(?<![\d.])" + re.escape(variant) + r"(?![\d.])"
            if re.search(pattern, text):
                return False
    return True
