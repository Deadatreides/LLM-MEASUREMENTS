"""Per-domain objective verification. Returns a uniform dict shape."""
import json
import subprocess
import sys
import os

from extraction import (
    extract_final_answer_text,
    normalize_numeric,
    normalize_text_answer,
    extract_code,
)

VERIFY_CODE_SCRIPT = os.path.join(os.path.dirname(__file__), "verify_code.py")
CODE_EXEC_TIMEOUT_SEC = 5


def _base_result():
    return {
        "verification_status": "UNVERIFIED",
        "error_class": None,
        "error_signature": None,
        "extracted_final_answer": None,
        "answer_format": "unknown",
        "has_final_answer": False,
        "has_multiple_candidates": False,
        "has_code": False,
        "code_length": None,
        "syntax_valid": None,
        "execution_status": None,
        "tests_passed": None,
        "tests_failed": None,
        "exception_type": None,
    }


def verify_math(ground_truth, raw_text):
    r = _base_result()
    ans_text, n_candidates = extract_final_answer_text(raw_text)
    r["has_multiple_candidates"] = n_candidates > 1
    if ans_text is None:
        # model produced no usable text at all (not a harness/generation
        # crash -- that case is handled separately in run_batch.py). This
        # is a mechanically-observed model failure, not a technical one.
        r["verification_status"] = "MECHANICAL_FAIL"
        r["error_class"] = "FORMAT_ERROR"
        r["error_signature"] = "empty_output"
        return r
    r["has_final_answer"] = True
    r["extracted_final_answer"] = ans_text
    num = normalize_numeric(ans_text)
    if num is None:
        r["answer_format"] = "non_numeric"
        r["verification_status"] = "REFERENCE_FAIL"
        r["error_class"] = "FORMAT_ERROR"
        r["error_signature"] = f"non_numeric_answer:{normalize_text_answer(ans_text)[:60]}"
        return r
    r["answer_format"] = "numeric"
    gt = normalize_numeric(str(ground_truth))
    if num == gt:
        r["verification_status"] = "REFERENCE_PASS"
    else:
        r["verification_status"] = "REFERENCE_FAIL"
        r["error_class"] = "WRONG_NUMERIC_ANSWER"
        r["error_signature"] = f"wrong_numeric:{num}"
    return r


def verify_logic(ground_truth, raw_text):
    r = _base_result()
    ans_text, n_candidates = extract_final_answer_text(raw_text)
    r["has_multiple_candidates"] = n_candidates > 1
    if ans_text is None:
        r["verification_status"] = "MECHANICAL_FAIL"
        r["error_class"] = "FORMAT_ERROR"
        r["error_signature"] = "empty_output"
        return r
    r["has_final_answer"] = True
    r["extracted_final_answer"] = ans_text
    r["answer_format"] = "text"
    norm_ans = normalize_text_answer(ans_text)
    norm_gt = normalize_text_answer(str(ground_truth))
    # accept exact match, or gt as a whole word inside a short concluding
    # sentence (e.g. "Therefore, Carol is the shortest of the three."),
    # as long as it isn't directly negated (e.g. "not Carol").
    whole_word_hit = re.search(rf"\b{re.escape(norm_gt)}\b", norm_ans)
    negated = bool(whole_word_hit) and re.search(
        rf"\b(not|isnt|isn t|never)\s+{re.escape(norm_gt)}\b", norm_ans
    )
    is_match = norm_ans == norm_gt or (
        len(norm_ans) <= 150 and whole_word_hit and not negated
    )
    if is_match:
        r["verification_status"] = "REFERENCE_PASS"
    else:
        r["verification_status"] = "REFERENCE_FAIL"
        r["error_class"] = "WRONG_LOGICAL_ANSWER"
        r["error_signature"] = f"wrong_logical:{norm_ans[:60]}"
    return r


import re  # noqa: E402  (used by verify_logic above)


def verify_code(function_name, test_cases, raw_text):
    r = _base_result()
    code, fmt = extract_code(raw_text, function_name)
    if code is None:
        # mechanically-observed failure to produce the requested code
        # (no def <function_name> found anywhere in the output) -- not a
        # harness/generation crash.
        r["verification_status"] = "MECHANICAL_FAIL"
        r["error_class"] = "FORMAT_ERROR"
        r["error_signature"] = "no_extractable_code"
        return r
    r["has_code"] = True
    r["code_length"] = len(code)
    r["answer_format"] = fmt

    job = json.dumps({"code": code, "function_name": function_name, "test_cases": test_cases})
    try:
        proc = subprocess.run(
            [sys.executable, VERIFY_CODE_SCRIPT],
            input=job,
            capture_output=True,
            text=True,
            timeout=CODE_EXEC_TIMEOUT_SEC,
        )
        if proc.returncode != 0:
            r["verification_status"] = "TECHNICAL_FAILURE"
            r["error_class"] = "MODEL_FAILURE"
            r["error_signature"] = "verifier_subprocess_crash"
            r["execution_status"] = "VERIFIER_CRASH"
            return r
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        r["verification_status"] = "MECHANICAL_FAIL"
        r["error_class"] = "TIMEOUT"
        r["error_signature"] = "execution_timeout"
        r["execution_status"] = "TIMEOUT"
        r["syntax_valid"] = None
        return r

    r["syntax_valid"] = out["syntax_valid"]
    r["tests_passed"] = out["tests_passed"]
    r["tests_failed"] = out["tests_failed"]
    r["exception_type"] = out["exception_type"]
    r["execution_status"] = out["status"]

    if out["status"] == "SYNTAX_ERROR":
        r["verification_status"] = "MECHANICAL_FAIL"
        r["error_class"] = "SYNTAX_ERROR"
        r["error_signature"] = f"syntax_error:{(out['exception_message'] or '')[:80]}"
    elif out["status"] == "RUNTIME_ERROR":
        r["verification_status"] = "MECHANICAL_FAIL"
        r["error_class"] = "RUNTIME_ERROR"
        r["error_signature"] = f"runtime_error:{out['exception_type']}"
    elif out["status"] == "TEST_FAILURE":
        r["verification_status"] = "MECHANICAL_FAIL"
        r["error_class"] = "TEST_FAILURE"
        failed = ",".join(str(i) for i in out["failed_indices"])
        r["error_signature"] = f"test_failure:{failed}"
    elif out["status"] == "ALL_PASS":
        r["verification_status"] = "MECHANICAL_PASS"

    return r


def verify(domain, task, raw_text):
    if domain == "math":
        return verify_math(task["ground_truth"], raw_text)
    elif domain == "logic":
        return verify_logic(task["ground_truth"], raw_text)
    elif domain == "code":
        return verify_code(task["function_name"], task["test_cases"], raw_text)
    else:
        raise ValueError(f"unknown domain {domain}")
