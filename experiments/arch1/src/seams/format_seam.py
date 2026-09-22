"""format_seam — HARD-шов структуры (SEAM_MODEL.md §3, SEAM_FORMAT).

Извлечение/разбор — адаптация experiment8/configs/ast_checks.py
(extract_code, inspect_function_source): те же процедуры, скопированы
внутрь arch1 вместо импорта через путь к соседнему каталогу (arch1
самодостаточен, arch1/CLAUDE.md). Сами функции чистые (ast.parse +
regex, без побочных эффектов, без exec) — переносятся как есть.
"""

import ast
import re
from typing import Optional

from ..seam_engine import ControlCase, FAIL, INAPPLICABLE, PASS, SeamDefinition

SEAM_ID = "seam:format"
SEAM_TYPE = "SEAM_FORMAT"

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str, function_name: Optional[str]) -> Optional[str]:
    fences = _CODE_FENCE_RE.findall(text)
    if fences:
        if function_name:
            for fence in fences:
                if f"def {function_name}" in fence:
                    return fence.strip()
        return fences[-1].strip()
    if function_name and f"def {function_name}" in text:
        return text[text.index(f"def {function_name}"):].strip()
    if "def " in text:
        return text[text.index("def "):].strip()
    return None


def _inspect(source: str, expected_function_name: Optional[str]) -> dict:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"syntax_valid": False, "syntax_error": str(exc), "matches_expected_name": False}

    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    matches = (expected_function_name in names) if expected_function_name else bool(names)
    return {"syntax_valid": True, "function_names_defined": names, "matches_expected_name": matches}


def check(inputs: dict) -> dict:
    text = inputs["text"]
    expected_function_name = inputs.get("expected_function_name")

    source = _extract_code(text, expected_function_name)
    if source is None:
        return {"status": INAPPLICABLE, "details": {"reason": "no code found in output"}}

    inspection = _inspect(source, expected_function_name)
    if not inspection["syntax_valid"]:
        return {"status": FAIL, "details": inspection}
    if expected_function_name and not inspection["matches_expected_name"]:
        return {"status": FAIL, "details": inspection}
    return {"status": PASS, "details": inspection}


CONTROL_CASES = (
    ControlCase(
        name="well_formed_function",
        inputs={
            "text": "```python\ndef add(a, b):\n    return a + b\n```",
            "expected_function_name": "add",
        },
        expected_status=PASS,
    ),
    ControlCase(
        name="wrong_function_name",
        inputs={
            "text": "```python\ndef subtract(a, b):\n    return a - b\n```",
            "expected_function_name": "add",
        },
        expected_status=FAIL,
    ),
    ControlCase(
        name="no_code_in_output",
        inputs={"text": "I think the answer is 42.", "expected_function_name": "add"},
        expected_status=INAPPLICABLE,
    ),
)


def build_definition() -> SeamDefinition:
    return SeamDefinition(
        seam_id=SEAM_ID,
        seam_type=SEAM_TYPE,
        check=check,
        control_cases=CONTROL_CASES,
        applicable_claim_types=frozenset({"FORMAT"}),
    )
