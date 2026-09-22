"""
C8_TESTS_PASS (CODE tasks, artifact A3_IMPLEMENTATION): actual LEVEL-0
mechanical execution of the model's implementation against the task's
hand-authored REFERENCE_TESTS, in an isolated subprocess (same pattern as
experiments 1/2/6's run_tests.py). Independent of what A1/A2 claimed --
see tasks/schema8.py module docstring for why this is not a DEPENDS_ON
edge but a separate SEAM check.
"""
import json
import subprocess
import sys
import os

RUN_TESTS_SCRIPT = os.path.join(os.path.dirname(__file__), "run_tests.py")
EXEC_TIMEOUT_SEC = 6


def run_test_suite(impl_code, test_code):
    job = json.dumps({"impl_code": impl_code or "", "test_code": test_code or "", "function_name": ""})
    try:
        proc = subprocess.run(
            [sys.executable, RUN_TESTS_SCRIPT],
            input=job, capture_output=True, text=True, timeout=EXEC_TIMEOUT_SEC,
        )
        if proc.returncode != 0:
            return {"harness_status": "SUBPROCESS_CRASH", "stderr": proc.stderr[-2000:]}
        return {"harness_status": "OK", **json.loads(proc.stdout.strip().splitlines()[-1])}
    except subprocess.TimeoutExpired:
        return {"harness_status": "TIMEOUT"}
    except Exception as e:
        return {"harness_status": "SUBPROCESS_CRASH", "stderr": str(e)}


def classify_c8_tests_pass(impl_code, reference_tests):
    if not impl_code or not impl_code.strip():
        return "OMITTED", None
    result = run_test_suite(impl_code, reference_tests)
    hs = result.get("harness_status")
    if hs == "TIMEOUT":
        return "INCORRECT", "execution_timeout"
    if hs == "SUBPROCESS_CRASH":
        return "UNKNOWN", "verifier_subprocess_crash"
    if not result.get("impl_syntax_valid"):
        return "INCORRECT", f"syntax_error: {result.get('impl_exec_error')}"
    if result.get("impl_exec_error"):
        return "INCORRECT", f"exec_error: {result.get('impl_exec_error')}"
    n_tests = result.get("n_tests_found", 0)
    n_pass = result.get("n_pass", 0)
    if n_tests == 0:
        return "UNKNOWN", "no_tests_found"
    if n_pass == n_tests:
        return "CORRECT", f"{n_pass}/{n_tests} tests passed"
    return "INCORRECT", f"{n_pass}/{n_tests} tests passed"
