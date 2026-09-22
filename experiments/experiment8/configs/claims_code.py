"""
Mechanical claim classification engine. Applies the fixed checklists in
tasks/claims_def.py to one A1's parsed JSON contract. No LLM judge
anywhere -- every classification is a regex match against specific
contract fields, or a structural comparison against the task's known
function_name/arg count.

Classifications: CORRECT / INCORRECT / OMITTED / AMBIGUOUS. Uncertainty is
never silently resolved into CORRECT or INCORRECT.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ast_checks import extract_json_object


def _get_text(contract, fields):
    parts = []
    for f in fields:
        v = contract.get(f)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.append(" ".join(str(x) for x in v))
        elif v is not None:
            parts.append(str(v))
    return " ".join(parts)


def _classify_regex(text, correct_re, incorrect_re):
    c = bool(correct_re.search(text)) if correct_re else False
    i = bool(incorrect_re.search(text)) if incorrect_re else False
    if c and i:
        return "AMBIGUOUS", "both correct- and incorrect-pattern matched"
    if c:
        m = correct_re.search(text)
        return "CORRECT", text[max(0, m.start() - 20): m.end() + 20]
    if i:
        m = incorrect_re.search(text)
        return "INCORRECT", text[max(0, m.start() - 20): m.end() + 20]
    return "OMITTED", None


def classify_a1(raw_text, task, claim_defs):
    """Returns (contract_parse_status, contract_obj_or_None, [claim results])."""
    obj, raw_json = extract_json_object(raw_text)
    results = []

    if obj is None or not isinstance(obj, dict):
        for cd in claim_defs:
            results.append({
                "claim_id": cd["claim_id"], "claim_text": cd["claim_text"], "claim_type": cd["claim_type"],
                "classification": "OMITTED", "evidence": None, "reason": "contract_not_parseable",
            })
        return "FAIL", None, results

    args_text = json.dumps(obj.get("arguments"), ensure_ascii=False) if obj.get("arguments") is not None else ""
    return_text = _get_text(obj, ["return_type", "output_format"])
    all_text = _get_text(obj, ["behavior", "edge_cases", "constraints", "input_format", "output_format"])

    for cd in claim_defs:
        kind = cd["kind"]
        if kind == "structural_name":
            fn = obj.get("function_name")
            if not isinstance(fn, str) or not fn.strip():
                cls, ev = "OMITTED", None
            elif fn.strip() == task["function_name"]:
                cls, ev = "CORRECT", fn
            else:
                cls, ev = "INCORRECT", fn
        elif kind == "structural_argc":
            args = obj.get("arguments")
            if not isinstance(args, list):
                cls, ev = "OMITTED", None
            elif len(args) == cd["expected_argc"]:
                cls, ev = "CORRECT", str(len(args))
            else:
                cls, ev = "INCORRECT", str(len(args))
        elif kind == "text_args":
            cls, ev = _classify_regex(args_text.lower(), cd.get("correct_re"), cd.get("incorrect_re"))
        elif kind == "text_return":
            cls, ev = _classify_regex(return_text.lower(), cd.get("correct_re"), cd.get("incorrect_re"))
        elif kind == "text_all":
            cls, ev = _classify_regex(all_text.lower(), cd.get("correct_re"), cd.get("incorrect_re"))
        elif kind == "text_all_custom":
            cls, ev = cd["custom_fn"](all_text.lower())
        else:
            cls, ev = "OMITTED", None

        results.append({
            "claim_id": cd["claim_id"], "claim_text": cd["claim_text"], "claim_type": cd["claim_type"],
            "classification": cls, "evidence": ev, "reason": None,
        })

    return "OK", obj, results


def pairwise_claim_relation(cls_a, cls_b):
    """AGREE / CONFLICT / UNKNOWN for one claim across two independent A1s,
    derived from each side's classification against the (unseen-by-the-model)
    reference. See module docstring in claims.py for the reasoning."""
    if cls_a in ("OMITTED", "AMBIGUOUS") or cls_b in ("OMITTED", "AMBIGUOUS"):
        return "UNKNOWN"
    if cls_a == cls_b:
        return "AGREE"  # both CORRECT, or both INCORRECT (same wrong claim)
    return "CONFLICT"  # one CORRECT, one INCORRECT
