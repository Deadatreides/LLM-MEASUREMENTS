"""context_manifest.py — декларативный список разрешённых полей для
K0/K1/K2 (задание v2, "КОНТРОЛЬ УТЕЧКИ") и программная проверка
построенного промпта против него.

MANIFEST -- источник истины, из него же генерируется
`context_manifest_v2.json` (артефакт, сохраняется один раз до прогона,
не меняется по ходу кампании).

Гарантии по КОНСТРУКЦИИ (не только по проверке постфактум):
  - `context_builder.build_prompt()` не принимает НИ retry-результат
    другого плеча, НИ информацию о том, какое плечо оказалось PASS --
    у функции физически нет такого параметра (сигнатура: level,
    task_family, task, initial_artifact, seam_result -- только
    ПЕРВИЧНЫЙ seam_result, никогда не результат ремонта). Утечка
    future-информации в промпт архитектурно невозможна, не только
    проверена.
  - Числовые значения в K1 не могут появиться: `describe_step()` заменяет
    ЛЮБОЙ числовой литерал плейсхолдером на уровне AST.
  - K2 показывает значения ТОЛЬКО тех шагов/requirements, чей статус на
    ПЕРВИЧНОЙ генерации == PASS -- обеспечивается `structure_text()`,
    не отдельным фильтром постфактум.

`validate_prompt_against_manifest()` -- дополнительная, "второй линии"
механическая проверка ГОТОВОГО текста промпта (не полагается только на
гарантии конструкции выше -- задание требует именно этого: "если
невозможно гарантировать отсутствие утечки, эксперимент остановить").
"""

from __future__ import annotations

import json
from pathlib import Path

from context_builder import check_no_leak, forbidden_values_for_level, structure_text

MANIFEST = {
    "K0": {
        "allowed": ["task_question_or_description", "initial_artifact", "evidence_from_mechanical_check"],
        "forbidden_here": ["structural_description", "confirmed_values"],
    },
    "K1": {
        "allowed": ["task_question_or_description", "initial_artifact", "evidence_from_mechanical_check", "structural_description_no_values"],
        "forbidden_here": ["confirmed_values", "any_numeric_step_or_requirement_value"],
    },
    "K2": {
        "allowed": [
            "task_question_or_description", "initial_artifact", "evidence_from_mechanical_check",
            "structural_description_no_values", "confirmed_values_where_status_pass_on_initial_generation",
        ],
        "forbidden_here": ["unconfirmed_step_or_requirement_values"],
    },
    "forbidden_everywhere": [
        "retry_result", "future_traceback", "which_arm_was_pass", "post_hoc_interpretation",
        "other_model_answer", "unconfirmed_explanation_of_failure_cause",
    ],
}

_STRUCTURE_MARKERS = ("Structure of this problem", "Behavioral requirements this function must satisfy")


def save_manifest(path: Path = None) -> Path:
    path = path or (Path(__file__).resolve().parent / "context_manifest_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(MANIFEST, f, ensure_ascii=False, indent=2)
    return path


def validate_prompt_against_manifest(level: str, task_family: str, task: dict, initial_artifact: str, seam_result: dict, prompt_text: str) -> dict:
    """Возвращает {"ok": bool, "violations": [str, ...]}."""
    violations = []
    has_structure_marker = any(m in prompt_text for m in _STRUCTURE_MARKERS)

    if level == "K0" and has_structure_marker:
        violations.append("K0 prompt contains a structural-description marker -- K1/K2 content leaked into K0")
    if level in ("K1", "K2") and not has_structure_marker:
        violations.append(f"{level} prompt is missing the structural-description marker -- structure was not actually included")

    if level in ("K1", "K2"):
        s_text = structure_text(level, task_family, task, seam_result)
        forbidden = forbidden_values_for_level(level, task_family, task, seam_result)
        if not check_no_leak(s_text, forbidden):
            violations.append(f"{level} structure text fails check_no_leak against forbidden_values_for_level")

    if initial_artifact and initial_artifact not in prompt_text:
        violations.append("initial_artifact (the ORIGINAL artifact) is not verbatim present in the prompt -- possible substitution with a different artifact")

    return {"ok": len(violations) == 0, "violations": violations}
