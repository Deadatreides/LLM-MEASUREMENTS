"""
Static (non-executing) structural checks on model-produced Python source,
via ast.parse only -- never exec. Used for the A1<->A3 mechanical seam and
for basic shape-sanity checks on A2/A4 outputs, without running anything.
"""
import ast
import re
import json

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text, function_name=None):
    fences = CODE_FENCE_RE.findall(text)
    if fences:
        if function_name:
            for f in fences:
                if f"def {function_name}" in f:
                    return f.strip(), "fenced"
        return fences[-1].strip(), "fenced"
    if function_name and f"def {function_name}" in text:
        idx = text.index(f"def {function_name}")
        return text[idx:].strip(), "unfenced"
    if "def " in text:
        idx = text.index("def ")
        return text[idx:].strip(), "unfenced_any"
    return None, None


def extract_json_object(text):
    """Best-effort extraction of a JSON object from model output. Tries a
    fenced ```json block first, then a fenced ```(plain) block, then the
    outermost {...} span via brace matching. Returns (dict_or_None, raw_str_or_None)."""
    candidates = []
    fences = JSON_FENCE_RE.findall(text)
    candidates.extend(f.strip() for f in fences)
    # brace-matched outermost span as a fallback
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in candidates:
        try:
            return json.loads(cand), cand
        except Exception:
            continue
    return None, None


def inspect_function_source(source, expected_function_name=None):
    """Static structural inspection. Never executes the code.
    Returns a dict describing what was found."""
    out = {
        "syntax_valid": None,
        "syntax_error": None,
        "function_names_defined": [],
        "matches_expected_name": None,
        "arg_count": None,
        "arg_names": None,
        "n_top_level_statements": 0,
        "n_top_level_non_def_import": 0,
        "has_test_def_leak": False,
        "has_class_def": False,
        "top_level_kinds": [],
    }
    try:
        tree = ast.parse(source)
        out["syntax_valid"] = True
    except SyntaxError as e:
        out["syntax_valid"] = False
        out["syntax_error"] = str(e)
        return out

    fn_defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    out["function_names_defined"] = [f.name for f in fn_defs]
    out["has_test_def_leak"] = any(n.startswith("test_") for n in out["function_names_defined"])
    out["has_class_def"] = any(isinstance(n, ast.ClassDef) for n in tree.body)

    target_fn = None
    if expected_function_name:
        target_fn = next((f for f in fn_defs if f.name == expected_function_name), None)
        out["matches_expected_name"] = target_fn is not None
    elif fn_defs:
        target_fn = fn_defs[0]

    if target_fn is not None:
        args = target_fn.args
        names = [a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
        out["arg_count"] = len(names)
        out["arg_names"] = names

    for stmt in tree.body:
        kind = type(stmt).__name__
        out["top_level_kinds"].append(kind)
        out["n_top_level_statements"] += 1
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef)):
            # a bare docstring Expr is harmless and common; don't count it
            if not (isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant)):
                out["n_top_level_non_def_import"] += 1

    return out
