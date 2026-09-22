"""
Isolated code-verification worker. Invoked as a subprocess (with an external
timeout) so a model-generated infinite loop or crash cannot hang or take
down the runner. Reads a JSON job on stdin, writes a JSON result to stdout.

Input:  {"code": "...", "function_name": "...", "test_cases": [{"args": [...], "expected": ...}, ...]}
Output: {"syntax_valid": bool, "tests_passed": int, "tests_failed": int,
         "failed_indices": [...], "exception_type": str|null,
         "exception_message": str|null, "status": "..."}
"""
import json
import sys


def main():
    job = json.loads(sys.stdin.read())
    code = job["code"]
    function_name = job["function_name"]
    test_cases = job["test_cases"]

    result = {
        "syntax_valid": None,
        "tests_passed": 0,
        "tests_failed": 0,
        "failed_indices": [],
        "exception_type": None,
        "exception_message": None,
        "status": None,
    }

    try:
        compiled = compile(code, "<generated>", "exec")
        result["syntax_valid"] = True
    except SyntaxError as e:
        result["syntax_valid"] = False
        result["exception_type"] = "SyntaxError"
        result["exception_message"] = str(e)
        result["status"] = "SYNTAX_ERROR"
        print(json.dumps(result))
        return

    namespace = {}
    try:
        exec(compiled, namespace)
    except Exception as e:
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        result["status"] = "RUNTIME_ERROR"
        print(json.dumps(result))
        return

    if function_name not in namespace or not callable(namespace[function_name]):
        result["exception_type"] = "NameError"
        result["exception_message"] = f"function '{function_name}' not defined"
        result["status"] = "RUNTIME_ERROR"
        print(json.dumps(result))
        return

    fn = namespace[function_name]
    for i, tc in enumerate(test_cases):
        try:
            actual = fn(*tc["args"])
            if actual == tc["expected"]:
                result["tests_passed"] += 1
            else:
                result["tests_failed"] += 1
                result["failed_indices"].append(i)
        except Exception as e:
            result["tests_failed"] += 1
            result["failed_indices"].append(i)
            if result["exception_type"] is None:
                result["exception_type"] = type(e).__name__
                result["exception_message"] = str(e)

    if result["tests_failed"] == 0:
        result["status"] = "ALL_PASS"
    elif result["exception_type"] is not None and result["tests_passed"] == 0:
        result["status"] = "RUNTIME_ERROR"
    else:
        result["status"] = "TEST_FAILURE"

    print(json.dumps(result))


if __name__ == "__main__":
    main()
