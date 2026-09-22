"""
Isolated test-execution worker (subprocess, external timeout applied by the
caller). Runs a set of assert-based test_* functions against a given
implementation. Used for every mechanical seam in this experiment: A2 vs
A3, A2 vs REFERENCE_IMPLEMENTATION, A2 vs mutants, REFERENCE_TESTS vs A3,
REFERENCE_TESTS vs mutants, REFERENCE_TESTS vs A4-patched implementation.

The implementation's namespace is injected directly into the test
namespace (no import statement required from the model) -- this avoids
penalizing a model for not knowing exactly how the harness wires files
together, which is a harness-integration detail, not a solution-quality
signal.

Input JSON on stdin:
  {"impl_code": "...", "test_code": "...", "function_name": "..."}
Output JSON on stdout:
  {"impl_syntax_valid": bool, "impl_exec_error": str|null,
   "test_syntax_valid": bool, "test_exec_error": str|null,
   "tests": [{"name": str, "status": "PASS"/"FAIL"/"ERROR", "exception_type": str|null, "exception_message": str|null}],
   "n_pass": int, "n_fail": int, "n_tests_found": int}
"""
import json
import sys


def main():
    job = json.loads(sys.stdin.read())
    impl_code = job["impl_code"]
    test_code = job["test_code"]

    result = {
        "impl_syntax_valid": None,
        "impl_exec_error": None,
        "test_syntax_valid": None,
        "test_exec_error": None,
        "tests": [],
        "n_pass": 0,
        "n_fail": 0,
        "n_tests_found": 0,
    }

    try:
        impl_compiled = compile(impl_code, "<impl>", "exec")
        result["impl_syntax_valid"] = True
    except SyntaxError as e:
        result["impl_syntax_valid"] = False
        result["impl_exec_error"] = f"SyntaxError: {e}"
        print(json.dumps(result))
        return

    impl_ns = {}
    try:
        exec(impl_compiled, impl_ns)
    except Exception as e:
        result["impl_exec_error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(result))
        return

    try:
        test_compiled = compile(test_code, "<tests>", "exec")
        result["test_syntax_valid"] = True
    except SyntaxError as e:
        result["test_syntax_valid"] = False
        result["test_exec_error"] = f"SyntaxError: {e}"
        print(json.dumps(result))
        return

    test_ns = dict(impl_ns)  # implementation symbols available to tests, no import needed
    try:
        exec(test_compiled, test_ns)
    except Exception as e:
        result["test_exec_error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(result))
        return

    test_fns = [
        (name, obj)
        for name, obj in test_ns.items()
        if name.startswith("test_") and callable(obj)
    ]
    result["n_tests_found"] = len(test_fns)

    for name, fn in sorted(test_fns):
        try:
            fn()
            result["tests"].append({"name": name, "status": "PASS", "exception_type": None, "exception_message": None})
            result["n_pass"] += 1
        except AssertionError as e:
            result["tests"].append({"name": name, "status": "FAIL", "exception_type": "AssertionError", "exception_message": str(e)})
            result["n_fail"] += 1
        except Exception as e:
            result["tests"].append({"name": name, "status": "ERROR", "exception_type": type(e).__name__, "exception_message": str(e)})
            result["n_fail"] += 1

    print(json.dumps(result))


if __name__ == "__main__":
    main()
