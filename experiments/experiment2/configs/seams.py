"""
Mechanical seam checks that require actually running code. All execution
goes through configs/run_tests.py in an isolated subprocess with an
external timeout (same safety pattern as experiment 1). No LLM is
involved anywhere in this file.
"""
import json
import subprocess
import sys
import os

from ast_checks import inspect_function_source

RUN_TESTS_SCRIPT = os.path.join(os.path.dirname(__file__), "run_tests.py")
EXEC_TIMEOUT_SEC = 6


def run_test_suite(impl_code, test_code):
    """Runs test_code's test_* functions against impl_code in an isolated
    subprocess. Returns the parsed run_tests.py output, or a synthetic
    TIMEOUT/CRASH result if the subprocess itself misbehaves."""
    job = json.dumps({"impl_code": impl_code or "", "test_code": test_code or "", "function_name": ""})
    try:
        proc = subprocess.run(
            [sys.executable, RUN_TESTS_SCRIPT],
            input=job,
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SEC,
        )
        if proc.returncode != 0:
            return {"harness_status": "SUBPROCESS_CRASH", "stderr": proc.stderr[-2000:]}
        return {"harness_status": "OK", **json.loads(proc.stdout.strip().splitlines()[-1])}
    except subprocess.TimeoutExpired:
        return {"harness_status": "TIMEOUT"}
    except Exception as e:
        return {"harness_status": "SUBPROCESS_CRASH", "stderr": str(e)}


def classify_test_run(run_result):
    """Turns a run_test_suite() result into a uniform verdict for the
    artifact being tested (the implementation side)."""
    hs = run_result.get("harness_status")
    if hs == "TIMEOUT":
        return {"status": "MECHANICAL_FAIL", "error_class": "TIMEOUT", "error_signature": "execution_timeout"}
    if hs == "SUBPROCESS_CRASH":
        return {"status": "TECHNICAL_FAILURE", "error_class": "MODEL_FAILURE", "error_signature": "verifier_subprocess_crash"}

    if run_result.get("impl_syntax_valid") is False:
        return {"status": "MECHANICAL_FAIL", "error_class": "SYNTAX_ERROR", "error_signature": f"impl:{(run_result.get('impl_exec_error') or '')[:80]}"}
    if run_result.get("impl_exec_error"):
        return {"status": "MECHANICAL_FAIL", "error_class": "RUNTIME_ERROR", "error_signature": f"impl_import:{run_result['impl_exec_error'][:80]}"}
    if run_result.get("test_syntax_valid") is False:
        return {"status": "MECHANICAL_FAIL", "error_class": "TEST_SUITE_SYNTAX_ERROR", "error_signature": f"tests:{(run_result.get('test_exec_error') or '')[:80]}"}
    if run_result.get("test_exec_error"):
        return {"status": "MECHANICAL_FAIL", "error_class": "TEST_SUITE_EXEC_ERROR", "error_signature": f"tests_import:{run_result['test_exec_error'][:80]}"}
    if run_result.get("n_tests_found", 0) == 0:
        return {"status": "MECHANICAL_FAIL", "error_class": "TEST_SUITE_EMPTY", "error_signature": "no_test_functions_found"}
    if run_result.get("n_fail", 0) > 0:
        failed = sorted(t["name"] for t in run_result.get("tests", []) if t["status"] != "PASS")
        return {"status": "MECHANICAL_FAIL", "error_class": "TEST_FAILURE", "error_signature": f"failed:{','.join(failed)}"}
    return {"status": "MECHANICAL_PASS", "error_class": None, "error_signature": None}


def seam_a1_a3(contract_result, a3_source):
    """Structural (non-executing) check that A3 actually implements what
    A1 specified: right function name, right argument count, no stray
    top-level code, no test-function leakage."""
    expected_name = contract_result.get("contract_function_name")
    inspect = inspect_function_source(a3_source, expected_function_name=expected_name)

    out = {"inspect": inspect, "status": None, "error_class": None, "error_signature": None}
    if not inspect["syntax_valid"]:
        out.update(status="MECHANICAL_FAIL", error_class="SYNTAX_ERROR", error_signature=f"a3:{(inspect['syntax_error'] or '')[:80]}")
        return out
    if expected_name and not inspect["matches_expected_name"]:
        out.update(status="MECHANICAL_FAIL", error_class="CONTRACT_MISMATCH", error_signature=f"name_mismatch:{inspect['function_names_defined']}")
        return out
    expected_argc = contract_result.get("contract_argument_count")
    if expected_argc is not None and inspect["arg_count"] is not None and inspect["arg_count"] != expected_argc:
        out.update(status="MECHANICAL_FAIL", error_class="CONTRACT_MISMATCH", error_signature=f"argc_mismatch:got={inspect['arg_count']}_expected={expected_argc}")
        return out
    if inspect["has_test_def_leak"]:
        out.update(status="MECHANICAL_FAIL", error_class="ROLE_VIOLATION", error_signature="a3_contains_test_defs")
        return out
    if inspect["n_top_level_non_def_import"] > 0:
        out.update(status="MECHANICAL_FAIL", error_class="EXTRA_TOP_LEVEL_CODE", error_signature=f"n={inspect['n_top_level_non_def_import']}")
        return out
    out["status"] = "MECHANICAL_PASS"
    return out
