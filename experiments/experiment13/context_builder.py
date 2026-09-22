"""context_builder.py — промпты R0/R1/R2 (+R0B) и проверка отсутствия
подсказки метода / утечки (задание §2 "не давать список способов
reframe", §3 "критическое ограничение").

Все режимы получают ОДНО и то же evidence-ядро (задача + существующий
артефакт + механическое evidence) -- строится ОДНОЙ функцией без
параметра "режим", так же как в Эксперименте 12 build_prompt() не
принимал модель как параметр. Различается только ИНСТРУКЦИЯ, добавляемая
поверх -- фиксированный текст, не новая фактическая информация.
"""

from __future__ import annotations

import re

MODES = ("R0", "R1", "R2", "R0B")

# Явно запрещённые задание фразы (§2) -- ниR1, ни R2 промпт не должен их
# содержать; это self-check авторского текста инструкций, не только
# проверка ввода пользователя.
_FORBIDDEN_METHOD_HINTS = (
    "break into steps", "break it into steps", "разбей на шаги",
    "use a different formula", "другую формулу",
    "reverse check", "обратную проверку", "работай в обратном порядке",
    "different algorithm", "другой алгоритм",
)

_STRUCTURE_MARKERS = {
    "R1": ("NEW REPRESENTATION:",),
    "R2": ("DECOMPOSITION DECISION:", "NEW DECOMPOSITION:"),
    "R0": (),
    "R0B": ("REASONING:",),
}


def _describe_evidence(task_family: str, seam_result: dict) -> str:
    ctype = seam_result.get("collision_type")
    details = seam_result.get("details") or {}
    if task_family == "arithmetic":
        if ctype == "numeric":
            final = details.get("final", {}) or {}
            return f"the final value ({final.get('name')} = {final.get('extracted')}) does not match the required result"
        final = details.get("final", {}) or {}
        return f"the final step ({final.get('name', '?')}) could not be read from the answer ({details.get('reason', '')})"
    if ctype == "syntactic":
        return f"the code has a syntax/execution error ({details.get('reason', '')})"
    if ctype == "structural":
        return f"the required function was not defined correctly ({details.get('reason', '')})"
    if ctype == "logical":
        failing = "; ".join((details.get("failing_requirements") or [])[:3])
        return f"the code runs but fails these checks: {failing}"
    return f"status={seam_result.get('status')}"


def _evidence_core(task_family: str, task: dict, initial_artifact: str, seam_result: dict) -> str:
    evidence = _describe_evidence(task_family, seam_result)
    problem = task["question"] if task_family == "arithmetic" else task["description"]
    return (
        f"{problem}\n\n"
        f"A previous attempt produced this artifact:\n{initial_artifact}\n\n"
        f"This was mechanically verified INCORRECT: {evidence}"
    )


def _r0_instruction(task_family: str, task: dict) -> str:
    if task_family == "arithmetic":
        step_names = ", ".join(s["name"] for s in task["steps"])
        return (
            "Fix the existing artifact by resolving the collision described above. Preserve the "
            "original structure of the solution as much as possible. Report EVERY step as its own "
            f"line in the format `name = value`, using exactly these step names in this order: {step_names}.\n"
            "Output nothing except these lines -- no explanation, no extra text."
        )
    return (
        "Fix the existing artifact by resolving the collision described above. Preserve the "
        "original structure of the solution as much as possible. Provide a corrected, complete "
        f"implementation, starting with `{task['signature_hint']}`. Reply with ONLY the function "
        "definition, no explanation."
    )


def _r1_instruction(task_family: str, task: dict) -> str:
    core = (
        "Do not simply edit or patch this artifact. First, identify what aspect of how the problem "
        "was represented or approached may have caused this error. Then propose a new representation "
        "of the same problem -- a different way of framing what needs to be computed and how the "
        "pieces relate to each other. Describe this new representation explicitly. Then, using this "
        "new representation, construct your solution."
    )
    if task_family == "arithmetic":
        return (
            f"{core}\n\nStructure your response exactly as:\n"
            "NEW REPRESENTATION: <describe your new representation>\n"
            "SOLUTION: <your work -- report every intermediate quantity you compute as its own line "
            "in the form `name = value`>\n"
            "FINAL ANSWER = <a single number>\n\n"
            "Output nothing else."
        )
    return (
        f"{core}\n\nStructure your response exactly as:\n"
        "NEW REPRESENTATION: <describe your new representation>\n"
        f"SOLUTION: <a single Python code block with the complete, corrected function, starting with `{task['signature_hint']}`>\n\n"
        "Output nothing else."
    )


def _r2_instruction(task_family: str, task: dict) -> str:
    core = (
        "Do not simply fix this artifact. Decide whether the boundaries of this problem -- how it is "
        "broken into parts -- could be changed so that this specific error would not occur. If you "
        "believe a different decomposition would help, describe it. If you believe no change in "
        "decomposition is needed, state this explicitly and explain why. Then solve the problem."
    )
    if task_family == "arithmetic":
        return (
            f"{core}\n\nStructure your response exactly as:\n"
            "DECOMPOSITION DECISION: <your reasoning and decision -- new decomposition needed, or not, and why>\n"
            "NEW DECOMPOSITION: <describe the new decomposition, or write 'unchanged' if none is needed>\n"
            "SOLUTION: <your work -- report every intermediate quantity you compute as its own line "
            "in the form `name = value`>\n"
            "FINAL ANSWER = <a single number>\n\n"
            "Output nothing else."
        )
    return (
        f"{core}\n\nStructure your response exactly as:\n"
        "DECOMPOSITION DECISION: <your reasoning and decision -- new decomposition needed, or not, and why>\n"
        "NEW DECOMPOSITION: <describe the new decomposition, or write 'unchanged' if none is needed>\n"
        f"SOLUTION: <a single Python code block with the complete, corrected function, starting with `{task['signature_hint']}`>\n\n"
        "Output nothing else."
    )


def _r0b_instruction(task_family: str, task: dict) -> str:
    """Доп. контроль (задание §13): тот же объём инструктивного текста,
    что R1/R2, но БЕЗ требования менять представление -- отделяет
    "промпт длиннее/подробнее" от "модель реально сменила представление"."""
    core = (
        "Fix the existing artifact by resolving the collision described above. You may use any method "
        "or approach you find appropriate. Feel free to explain your reasoning in as much detail as "
        "you like before giving your answer. There is no requirement to preserve or to change the "
        "existing structure -- use whatever approach seems best to you."
    )
    if task_family == "arithmetic":
        return (
            f"{core}\n\nStructure your response exactly as:\n"
            "REASONING: <explain your reasoning>\n"
            "SOLUTION: <your work -- report every intermediate quantity you compute as its own line "
            "in the form `name = value`>\n"
            "FINAL ANSWER = <a single number>\n\n"
            "Output nothing else."
        )
    return (
        f"{core}\n\nStructure your response exactly as:\n"
        "REASONING: <explain your reasoning>\n"
        f"SOLUTION: <a single Python code block with the complete, corrected function, starting with `{task['signature_hint']}`>\n\n"
        "Output nothing else."
    )


_INSTRUCTION_BUILDERS = {"R0": _r0_instruction, "R1": _r1_instruction, "R2": _r2_instruction, "R0B": _r0b_instruction}


def build_prompt(mode: str, task_family: str, task: dict, initial_artifact: str, seam_result: dict) -> str:
    core = _evidence_core(task_family, task, initial_artifact, seam_result)
    instruction = _INSTRUCTION_BUILDERS[mode](task_family, task)
    return f"{core}\n\n{instruction}"


def validate_prompt(mode: str, initial_artifact: str, prompt_text: str) -> dict:
    """Проверка отсутствия утечки/подсказки метода. Возвращает
    {"ok": bool, "violations": [...]}."""
    violations = []

    for phrase in _FORBIDDEN_METHOD_HINTS:
        if phrase.lower() in prompt_text.lower():
            violations.append(f"forbidden method-hint phrase present: {phrase!r}")

    required_markers = _STRUCTURE_MARKERS[mode]
    for marker in required_markers:
        if marker not in prompt_text:
            violations.append(f"{mode} prompt is missing required marker {marker!r}")

    for other_mode, markers in _STRUCTURE_MARKERS.items():
        if other_mode == mode:
            continue
        for marker in markers:
            if marker and marker not in required_markers and marker in prompt_text:
                violations.append(f"{mode} prompt contains marker {marker!r} belonging to {other_mode}")

    if initial_artifact and initial_artifact not in prompt_text:
        violations.append("initial_artifact not verbatim present in prompt -- possible substitution")

    return {"ok": len(violations) == 0, "violations": violations}
