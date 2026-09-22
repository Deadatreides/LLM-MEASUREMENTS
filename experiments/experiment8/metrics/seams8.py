"""
Section 7: mechanical seam checks. Every seam expresses a CONCRETE,
checkable relation (spec section 7's explicit warning: "not a seam just
because two texts sit next to each other"). Frozen alongside the schema.

SEAM_DEPENDENCY -- literally the DEPENDS_ON edges from schema8.py,
  status VIOLATED if the downstream claim's own LOCAL check says CORRECT
  but its EFFECTIVE status (after propagation) is INCORRECT/UNKNOWN --
  i.e. exactly the "locally consistent with a wrong premise" case section
  6 asks to catch, made visible as a first-class object instead of only a
  derived number.

SEAM_CONSISTENCY -- MATH: does A3's shown computation match the
  WRONG-shortcut's characteristic number instead of the correct one, while
  A1 claims the correct method (text says one thing, arithmetic does
  another)? CODE: does A3_IMPLEMENTATION's actual AST-inspected signature
  (function name, arg count) agree with what A1_CONTRACT declared?

SEAM_EXECUTION -- can the terminal artifact be mechanically
  executed/verified at all (code parses & runs; final number is
  extractable)? Independent of whether it's correct.

SEAM_CONFLICT -- within the SAME model's 8 D-mode generations for one
  task, does a claim disagree with itself (some generations CORRECT, some
  INCORRECT) -- reused from experiments 6/7's two-way-conflict signal, at
  the artifact-pool level rather than the single-generation level.

SEAM_UNKNOWN -- a dependency exists (claim has a defined DEPENDS_ON edge
  or upstream artifact) but correctness cannot be established with the
  means available (effective status UNKNOWN, not a confirmed error).

SEAM_FORMAT is not implemented as a separate mechanism in this experiment
-- for these 6 tasks it would duplicate what SEAM_CONSISTENCY already
checks (CODE's return_type claim, MATH's final-answer format); this
scoping choice is stated explicitly rather than silently skipped.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from ast_checks import inspect_function_source, extract_code
from dependency_engine import normalize_status


def seam_dependency_status(claim_id, local_status, effective_by_claim, depends_on):
    deps = depends_on.get(claim_id, [])
    if not deps:
        return None
    local_norm = normalize_status(local_status)
    eff_status, _ = effective_by_claim[claim_id]
    violated = local_norm == "CORRECT" and eff_status != "CORRECT"
    return {"seam_type": "SEAM_DEPENDENCY", "source": deps, "target": claim_id,
            "status": "VIOLATED" if violated else "INTACT",
            "evidence": f"local={local_norm} effective={eff_status}"}


def seam_consistency_math(task, sections_text, c1_status, c3_status, c3_evidence):
    """A1(method) vs A3(computation): does the shown arithmetic match the
    WRONG-shortcut's characteristic number while A1 claims the correct
    method?"""
    a3_text = sections_text.get("A3_COMPUTATION") or ""
    wrong_answer = task.get("wrong_shortcut_final_answer")
    if wrong_answer is None or not a3_text:
        return {"seam_type": "SEAM_CONSISTENCY", "status": "NOT_APPLICABLE", "evidence": None}
    shows_wrong_number = _text_has_number(a3_text, wrong_answer)
    c1_norm = normalize_status(c1_status)
    if c1_norm == "CORRECT" and shows_wrong_number:
        return {"seam_type": "SEAM_CONSISTENCY", "status": "VIOLATED",
                "evidence": f"A1 claims correct method but A3 shows the wrong-shortcut number {wrong_answer}"}
    return {"seam_type": "SEAM_CONSISTENCY", "status": "INTACT", "evidence": None}


def seam_consistency_code(task, a1_obj, a3_impl_text):
    if not a1_obj or not a3_impl_text:
        return {"seam_type": "SEAM_CONSISTENCY", "status": "NOT_APPLICABLE", "evidence": None}
    code, _ = extract_code(a3_impl_text, task.get("function_name"))
    if not code:
        return {"seam_type": "SEAM_CONSISTENCY", "status": "NOT_APPLICABLE", "evidence": "no code extracted"}
    inspected = inspect_function_source(code, task.get("function_name"))
    if not inspected.get("syntax_valid"):
        return {"seam_type": "SEAM_CONSISTENCY", "status": "VIOLATED", "evidence": "A3 does not parse"}
    declared_name = a1_obj.get("function_name")
    declared_args = a1_obj.get("arguments")
    mismatches = []
    if declared_name and inspected.get("function_names_defined") and declared_name not in inspected["function_names_defined"]:
        mismatches.append(f"declared name {declared_name!r} not defined in A3 ({inspected['function_names_defined']})")
    if isinstance(declared_args, list) and inspected.get("arg_count") is not None and len(declared_args) != inspected["arg_count"]:
        mismatches.append(f"declared argc {len(declared_args)} != A3 argc {inspected['arg_count']}")
    if mismatches:
        return {"seam_type": "SEAM_CONSISTENCY", "status": "VIOLATED", "evidence": "; ".join(mismatches)}
    return {"seam_type": "SEAM_CONSISTENCY", "status": "INTACT", "evidence": None}


def _text_has_number(text, target, tol=0.01):
    import re
    for m in re.finditer(r"-?\$?\d[\d,]*\.?\d*%?", text):
        s = m.group(0).replace("$", "").replace(",", "").replace("%", "")
        try:
            v = float(s)
        except ValueError:
            continue
        if abs(v - target) <= tol:
            return True
    return False


def seam_execution_status(terminal_claim_status):
    norm = normalize_status(terminal_claim_status)
    executable = norm in ("CORRECT", "INCORRECT")  # a definite mechanical verdict was reachable
    return {"seam_type": "SEAM_EXECUTION", "status": "EXECUTABLE" if executable else "NOT_EXECUTABLE",
            "evidence": f"terminal_local_status={terminal_claim_status}"}


def seam_conflict_in_pool(claim_id, statuses_across_pool):
    """statuses_across_pool: list of raw local statuses for ONE claim_id
    across the 8 D-mode generations of one (task, model). CONFLICT if both
    CORRECT and INCORRECT genuinely occur (not just noise around
    OMITTED/AMBIGUOUS)."""
    norm = [normalize_status(s) for s in statuses_across_pool]
    has_correct = "CORRECT" in norm
    has_incorrect = "INCORRECT" in norm
    if has_correct and has_incorrect:
        return {"seam_type": "SEAM_CONFLICT", "claim_id": claim_id, "status": "CONFLICT",
                "evidence": f"{norm.count('CORRECT')}/{len(norm)} CORRECT vs {norm.count('INCORRECT')}/{len(norm)} INCORRECT"}
    return {"seam_type": "SEAM_CONFLICT", "claim_id": claim_id, "status": "NO_CONFLICT", "evidence": None}


def seam_unknown_status(claim_id, effective_by_claim):
    status, origin = effective_by_claim.get(claim_id, ("UNKNOWN", "NONE"))
    if status == "UNKNOWN":
        return {"seam_type": "SEAM_UNKNOWN", "claim_id": claim_id, "status": "UNKNOWN_CONFIRMED", "evidence": f"origin={origin}"}
    return {"seam_type": "SEAM_UNKNOWN", "claim_id": claim_id, "status": "NOT_UNKNOWN", "evidence": None}
