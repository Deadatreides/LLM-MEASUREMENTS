"""seams.py — механические проверки для эксперимента 13.

`enriched_code_seam` переиспользуется от `experiment12/seams.py` БЕЗ
изменений: он уже проверяет только внешнее поведение функции (по
фиксированному имени/сигнатуре, вызывая её через `requirements`), не
внутреннюю реализацию -- R1/R2 (смена внутреннего подхода) не требуют
для кода никакого нового шва.

Для арифметики -- НОВЫЙ `final_answer_seam`, ТОЛЬКО для R1/R2. Причина:
`multistep_arithmetic_seam` (Эксп.12) требует ТОЧНО заданных имён шагов
в заданном порядке -- это часть контракта R0 («сохрани структуру»), но
R1/R2 приглашаются ИЗМЕНИТЬ представление; проверять их тем же строгим
швом означало бы автоматически проваливать любое подлинное изменение
структуры не из-за неверного ответа, а из-за несовпадения имён шагов --
заражение измерением того самого эффекта, который измеряется. См.
план/REPORT_EXPERIMENT13.md, раздел "критическое архитектурное решение".

`final_answer_seam` не требует конкретных имён -- только ОДНУ
однозначную строку `FINAL ANSWER = <число>` (формат ОТЧЁТА о результате,
не подсказка метода решения). Переиспользует `extract_numeric_value` из
`experiment11/seams.py` -- тот же принцип «один результат -> используется,
несколько/ноль -> INAPPLICABLE, не угадывается».
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

_EXPERIMENT11_SEAMS_PATH = Path(__file__).resolve().parents[1] / "experiment11" / "seams.py"
_spec11 = importlib.util.spec_from_file_location("experiment11_seams_for_exp13", _EXPERIMENT11_SEAMS_PATH)
_experiment11_seams = importlib.util.module_from_spec(_spec11)
_spec11.loader.exec_module(_experiment11_seams)

_EXPERIMENT12_SEAMS_PATH = Path(__file__).resolve().parents[1] / "experiment12" / "seams.py"
_spec12 = importlib.util.spec_from_file_location("experiment12_seams_for_exp13", _EXPERIMENT12_SEAMS_PATH)
_experiment12_seams = importlib.util.module_from_spec(_spec12)
_spec12.loader.exec_module(_experiment12_seams)

ERROR = _experiment11_seams.ERROR
FAIL = _experiment11_seams.FAIL
INAPPLICABLE = _experiment11_seams.INAPPLICABLE
PASS = _experiment11_seams.PASS
UNKNOWN = _experiment11_seams.UNKNOWN
extract_numeric_value = _experiment11_seams.extract_numeric_value
extract_code = _experiment11_seams.extract_code

enriched_code_seam = _experiment12_seams.enriched_code_seam
multistep_arithmetic_seam = _experiment12_seams.multistep_arithmetic_seam  # только для R0

__all__ = [
    "ERROR", "FAIL", "INAPPLICABLE", "PASS", "UNKNOWN",
    "extract_code", "extract_numeric_value",
    "enriched_code_seam", "multistep_arithmetic_seam", "final_answer_seam",
]

_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER\s*=\s*(.+)", re.IGNORECASE)


def final_answer_seam(text: str, expected: float, tolerance: float = 1e-6) -> dict:
    """R1/R2-арифметика: не требует конкретных имён шагов -- только
    единственную однозначную строку `FINAL ANSWER = <число>`."""
    if not isinstance(text, str):
        return {"status": INAPPLICABLE, "collision_type": "format", "details": {"reason": "not a string"}}

    matches = _FINAL_ANSWER_RE.findall(text)
    if not matches:
        return {"status": INAPPLICABLE, "collision_type": "format", "details": {"reason": "no FINAL ANSWER line found"}}
    if len(matches) > 1:
        return {
            "status": INAPPLICABLE, "collision_type": "format",
            "details": {"reason": f"ambiguous: {len(matches)} FINAL ANSWER lines"},
        }

    value, reason = extract_numeric_value(matches[0])
    if value is None:
        return {"status": INAPPLICABLE, "collision_type": "format", "details": {"reason": reason}}

    diff = abs(value - expected)
    details = {"final": {"name": "FINAL", "extracted": value}, "actual": value, "expected": expected, "diff": diff}
    if diff <= tolerance:
        return {"status": PASS, "collision_type": None, "details": details}
    return {"status": FAIL, "collision_type": "numeric", "details": details}
