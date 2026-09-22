"""seams.py — механические проверки, фиксированные ДО прогона кампании.

Извлечение значения происходит ВНУТРИ проверки (не снаружи, в
вызывающем коде) — прямой урок arch1 (этап 8): извлечение вне границы
шва не подлежит контрольным случаям и дало 59.6% ложных FAIL в той
работе. Здесь оба шва сразу написаны с извлечением внутри и с
контрольными случаями (tests/test_experiment11.py) на форму того же
дефекта: несколько результатов в тексте -> INAPPLICABLE (нарушение
границы утверждения), не угаданный FAIL/PASS.

PASS/FAIL/UNKNOWN/INAPPLICABLE/ERROR — строго разделены (п.3 задания):
UNKNOWN и INAPPLICABLE никогда не превращаются в FAIL. ERROR (сбой
самого конвейера проверки, не кандидата) не возвращается этими
функциями — они либо PASS/FAIL/INAPPLICABLE по кандидату, либо бросают
исключение, которое ловит вызывающий код харнесса и помечает как ERROR
(см. harness.py) — то же разделение, что SEAM_MODEL.md делает в arch1
(ERROR не транслируется в FAIL).
"""

from __future__ import annotations

import ast
import re
from typing import Optional

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
INAPPLICABLE = "INAPPLICABLE"
ERROR = "ERROR"

# -- arithmetic_seam ---------------------------------------------------------

_EQUATION_RESULT_RE = re.compile(r"=\s*[^\d\-]{0,3}(-?\d+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numeric_value(text) -> tuple:
    """Правило извлечения (то же обоснование, что в arch1/numeric_seam):
    ровно одно равенство -> его правая часть; ровно одно голое число ->
    оно; несколько равенств/чисел -> НАРУШЕНИЕ ГРАНИЦЫ утверждения,
    None (не угадывается)."""
    if not isinstance(text, str):
        return None, "not a string"
    results = _EQUATION_RESULT_RE.findall(text)
    if len(results) == 1:
        return float(results[0]), "single equation result"
    if len(results) > 1:
        return None, f"ambiguous: {len(results)} equation results"
    numbers = _NUMBER_RE.findall(text)
    if len(numbers) == 1:
        return float(numbers[0]), "single bare number"
    if not numbers:
        return None, "no numbers found"
    return None, f"ambiguous: {len(numbers)} numbers, no equation"


def arithmetic_seam(text: str, expected: float, tolerance: float = 1e-6) -> dict:
    value, reason = extract_numeric_value(text)
    if value is None:
        return {
            "status": INAPPLICABLE,
            "collision_type": "format",
            "details": {"reason": reason, "text": text},
        }
    diff = abs(value - expected)
    details = {"actual": value, "expected": expected, "diff": diff}
    if diff <= tolerance:
        return {"status": PASS, "collision_type": None, "details": details}
    return {"status": FAIL, "collision_type": "numeric", "details": details}


# -- code_seam ----------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    fences = _CODE_FENCE_RE.findall(text)
    if fences:
        return fences[-1].strip()
    if "def " in text:
        return text[text.index("def "):].strip()
    return None


def code_seam(text: str, function_name: str, tests: list) -> dict:
    """tests -- python-булевы выражения вида "f(1,2) == 3" (без assert).

    Классификация collision_type (механическая, по СТАДИИ отказа, не по
    угадыванию причины):
      не нашли код в тексте            -> INAPPLICABLE / format
      SyntaxError / ошибка exec        -> FAIL / syntactic
      функция с ожидаемым именем не найдена -> FAIL / structural
      выполняется, тест не проходит    -> FAIL / logical
      все тесты проходят               -> PASS
    """
    source = extract_code(text)
    if source is None:
        return {"status": INAPPLICABLE, "collision_type": "format", "details": {"reason": "no code found in output"}}

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "status": FAIL,
            "collision_type": "syntactic",
            "details": {"reason": f"SyntaxError: {exc}", "source": source},
        }

    fn_names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if function_name not in fn_names:
        return {
            "status": FAIL,
            "collision_type": "structural",
            "details": {"reason": f"function {function_name!r} not defined", "found": fn_names, "source": source},
        }

    namespace: dict = {}
    try:
        exec(compile(source, "<candidate>", "exec"), namespace)  # noqa: S102 -- локальный sandboxed стенд, тот же паттерн, что experiment8/run_tests.py
    except Exception as exc:
        return {
            "status": FAIL,
            "collision_type": "syntactic",
            "details": {"reason": f"exec error: {type(exc).__name__}: {exc}", "source": source},
        }

    failures = []
    for test_expr in tests:
        try:
            ok = eval(test_expr, dict(namespace))  # noqa: S307 -- фиксированные тесты из tasks/code_tasks.py, не пользовательский ввод
        except Exception as exc:
            ok = False
            failures.append(f"{test_expr} -> {type(exc).__name__}: {exc}")
            continue
        if not ok:
            failures.append(f"{test_expr} -> False")

    if failures:
        return {
            "status": FAIL,
            "collision_type": "logical",
            "details": {"failing_tests": failures, "source": source},
        }
    return {
        "status": PASS,
        "collision_type": None,
        "details": {"n_tests_passed": len(tests), "source": source},
    }
