"""
D-mode CODE tasks split what claims_def_code.py's classify_a1 expects as a
SINGLE JSON object (with a nested "edge_cases" field) across TWO separate
sections (A1_CONTRACT's JSON, A2_EDGE_CASES' own JSON list). This merges
them into the single object shape classify_a1 expects, then reuses it
unmodified -- rather than duplicating/forking its claim-checking logic.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from claims_code import classify_a1
from ast_checks import extract_json_object


def classify_code_claims(a1_text, a2_text, task, claim_defs):
    a1_obj, _ = extract_json_object(a1_text or "")
    if a1_obj is None:
        # can't merge -- fall back to classify_a1's own OMITTED-everything path
        return classify_a1(a1_text or "", task, claim_defs)

    a2_obj, _ = extract_json_object(a2_text or "")
    if a2_obj is None:
        # A2 might just be a bare JSON list, or plain prose -- try list literal
        try:
            a2_obj = json.loads(a2_text) if a2_text else None
        except Exception:
            a2_obj = None

    merged = dict(a1_obj)
    if isinstance(a2_obj, list):
        merged["edge_cases"] = a2_obj
    elif a2_text:
        merged["edge_cases"] = [a2_text]

    merged_text = json.dumps(merged, ensure_ascii=False)
    return classify_a1(merged_text, task, claim_defs)
