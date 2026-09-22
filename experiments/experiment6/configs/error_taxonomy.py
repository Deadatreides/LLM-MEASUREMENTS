"""
E0-E10 error taxonomy (spec section 3), built entirely on top of the
existing mechanical LEVEL 0 verification (reference_tests via
run_tests.py, unchanged since experiment 2). No LLM judge anywhere.

Status is one of CORRECT / INCORRECT / UNKNOWN. UNKNOWN is never silently
folded into either bucket -- assigned only when we truly cannot tell what
the model intended (no parseable code at all), matching spec section 3's
explicit requirement.
"""
import ast


def _is_trivially_empty(code, function_name):
    """True if the target function's body is just pass / ... / a
    docstring / a bare 'raise NotImplementedError' -- a mechanically
    detectable 'incomplete solution' (E6), distinct from a genuine wrong
    computation (E5)."""
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function_name), None)
    if fn is None:
        return False
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant) and isinstance(s.value.value, str))]
    if not body:
        return True
    if len(body) == 1:
        s = body[0]
        if isinstance(s, ast.Pass):
            return True
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis:
            return True
        if isinstance(s, ast.Raise):
            return True
    return False


def classify(verification, code_text, function_name):
    """verification: {"status": MECHANICAL_PASS/MECHANICAL_FAIL, "error_class":..., "error_signature":...}
    Returns (correctness_status, error_type)."""
    if verification["error_class"] == "FORMAT_ERROR" and verification.get("error_signature") == "no_extractable_code":
        return "UNKNOWN", "E10"

    if verification["status"] == "MECHANICAL_PASS":
        return "CORRECT", "E0"

    ec = verification["error_class"]
    sig = verification.get("error_signature") or ""

    if ec == "SYNTAX_ERROR":
        return "INCORRECT", "E1"
    if ec == "TIMEOUT":
        return "INCORRECT", "E1"
    if ec == "RUNTIME_ERROR":
        if "NameError" in sig or "not defined" in sig:
            return "INCORRECT", "E2"  # contract violation: function not defined under the expected name
        return "INCORRECT", "E1"
    if ec == "TEST_FAILURE":
        if code_text and _is_trivially_empty(code_text, function_name):
            return "INCORRECT", "E6"
        return "INCORRECT", "E5"
    # anything else mechanically observed but not covered above
    return "INCORRECT", "E4"


E_DESCRIPTIONS = {
    "E0": "no error (CORRECT)",
    "E1": "mechanical error (syntax/timeout/generic runtime)",
    "E2": "contract violation (function not defined under expected name)",
    "E3": "factual error (not applicable to this code domain)",
    "E4": "logical error (uncategorized mechanical failure)",
    "E5": "wrong computation (reference tests fail, code runs)",
    "E6": "incomplete solution (trivially empty function body)",
    "E7": "contradiction between parts of the answer (not applicable to single-function code)",
    "E8": "semantic drift (not applicable to this code domain)",
    "E9": "persistent systematic error (derived at the state/cluster level, not per-answer)",
    "E10": "cannot be mechanically confirmed or refuted (no extractable code) -> UNKNOWN",
}
