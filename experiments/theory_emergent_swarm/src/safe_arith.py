"""safe_arith.py — AST-whitelisted arithmetic evaluator. A CALCULATOR,
not an answer table: it contains zero knowledge of GSM8K or of any task.
Same legitimacy as FORK-1's `aggregate_det` (det arithmetic beat LLM
arithmetic 1.000 vs 0.000 there).

Never uses eval()/exec() on model text -- parses to AST and walks a
whitelist, so a model emitting `__import__(...)` gets None, not execution.
"""

from __future__ import annotations

import ast
import re

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)
_MAX_POW = 8
_EXPR_CHARS = re.compile(r"^[0-9\.\s\+\-\*/%\(\)]+$")

# "16 - 3 - 4 = 9"  /  "$<<9*2=18>>18"  -- capture <expr> = <value>
_EQUATION_RE = re.compile(r"([0-9][0-9\.\s\+\-\*/%\(\)]{2,}?)\s*=\s*\$?\s*(-?[0-9][0-9,\.]*)")


def safe_eval(expr: str):
    """-> float | None. None means 'not a plain arithmetic expression'."""
    s = (expr or "").strip().replace(",", "").replace("$", "").replace("×", "*").replace("÷", "/")
    if not s or len(s) > 200 or not _EXPR_CHARS.match(s):
        return None
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            r = node.right
            if not (isinstance(r, ast.Constant) and isinstance(r.value, (int, float))
                    and abs(r.value) <= _MAX_POW):
                return None
    try:
        val = eval(compile(tree, "<safe>", "eval"), {"__builtins__": {}}, {})   # noqa: S307
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
        return None
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None


def find_equations(text: str) -> list:
    """-> [(expr_str, stated_value, true_value_or_None)] in order of appearance."""
    out = []
    for m in _EQUATION_RE.finditer(text or ""):
        expr, stated = m.group(1).strip(), m.group(2).replace(",", "")
        try:
            stated_v = float(stated)
        except ValueError:
            continue
        out.append((expr, stated_v, safe_eval(expr)))
    return out


def arithmetic_slips(text: str) -> int:
    """How many of the model's own stated equations it evaluated wrong."""
    return sum(1 for _e, s, t in find_equations(text)
               if t is not None and abs(t - s) > 0.01)
