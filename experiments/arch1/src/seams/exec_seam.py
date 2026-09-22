"""exec_seam — HARD-шов исполнения (SEAM_MODEL.md §3, SEAM_EXECUTION).

Алгоритм — тот же, что experiment8/configs/run_tests.py (compile+exec
impl → compile+exec test-кода с доступными символами impl → прогон
test_*-функций → подсчёт PASS/FAIL), но выполняется в процессе, без
subprocess-обёртки: изоляция актуальна при исполнении недоверенного
вывода модели (забота ACTION EXECUTOR, этап 5), для контрольных
случаев этого этапа не нужна и на Windows добавляет ломкости без
пользы. arch1 самодостаточен (arch1/CLAUDE.md) — experiment8 не
устанавливаемый пакет, поэтому логика скопирована и адаптирована,
а не импортирована через путь к соседнему каталогу.
"""

from ..seam_engine import ControlCase, FAIL, INAPPLICABLE, PASS, SeamDefinition

SEAM_ID = "seam:exec"
SEAM_TYPE = "SEAM_EXECUTION"


def _run(impl_code: str, test_code: str) -> dict:
    impl_ns: dict = {}
    try:
        exec(compile(impl_code, "<impl>", "exec"), impl_ns)
    except SyntaxError as exc:
        return {"phase": "impl_syntax", "error": f"SyntaxError: {exc}"}
    except Exception as exc:
        return {"phase": "impl_exec", "error": f"{type(exc).__name__}: {exc}"}

    test_ns = dict(impl_ns)  # символы impl доступны тестам без import (как в run_tests.py)
    try:
        exec(compile(test_code, "<tests>", "exec"), test_ns)
    except SyntaxError as exc:
        return {"phase": "test_syntax", "error": f"SyntaxError: {exc}"}
    except Exception as exc:
        return {"phase": "test_exec", "error": f"{type(exc).__name__}: {exc}"}

    test_fns = [
        (name, obj)
        for name, obj in test_ns.items()
        if name.startswith("test_") and callable(obj)
    ]

    results = []
    n_pass = n_fail = 0
    for name, fn in sorted(test_fns):
        try:
            fn()
            results.append({"name": name, "status": "PASS"})
            n_pass += 1
        except AssertionError as exc:
            results.append({"name": name, "status": "FAIL", "message": str(exc)})
            n_fail += 1
        except Exception as exc:
            results.append({"name": name, "status": "ERROR", "message": f"{type(exc).__name__}: {exc}"})
            n_fail += 1

    return {
        "phase": "ran",
        "tests": results,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_found": len(test_fns),
    }


def check(inputs: dict) -> dict:
    outcome = _run(inputs["impl_code"], inputs["test_code"])

    if outcome["phase"] in ("impl_syntax", "impl_exec"):
        # implementation не выполняется вовсе -- определённо негативный, детерминированный
        # результат (не сбой самой проверки), поэтому FAIL, а не ERROR
        return {"status": FAIL, "details": outcome}
    if outcome["phase"] in ("test_syntax", "test_exec"):
        # тестовый артефакт нельзя исполнить -- сверять не с чем
        return {"status": INAPPLICABLE, "details": outcome}
    if outcome["n_found"] == 0:
        return {"status": INAPPLICABLE, "details": outcome}
    if outcome["n_fail"] > 0:
        return {"status": FAIL, "details": outcome}
    return {"status": PASS, "details": outcome}


CONTROL_CASES = (
    ControlCase(
        name="passing_implementation",
        inputs={
            "impl_code": "def add(a, b):\n    return a + b\n",
            "test_code": "def test_add():\n    assert add(2, 3) == 5\n",
        },
        expected_status=PASS,
    ),
    ControlCase(
        name="failing_implementation",
        inputs={
            "impl_code": "def add(a, b):\n    return a - b\n",
            "test_code": "def test_add():\n    assert add(2, 3) == 5\n",
        },
        expected_status=FAIL,
    ),
    ControlCase(
        name="no_tests_found",
        inputs={
            "impl_code": "def add(a, b):\n    return a + b\n",
            "test_code": "x = 1\n",
        },
        expected_status=INAPPLICABLE,
    ),
)


def build_definition() -> SeamDefinition:
    return SeamDefinition(
        seam_id=SEAM_ID,
        seam_type=SEAM_TYPE,
        check=check,
        control_cases=CONTROL_CASES,
        applicable_claim_types=frozenset({"METHOD", "VALUE", "EDGE_CASE"}),
    )
