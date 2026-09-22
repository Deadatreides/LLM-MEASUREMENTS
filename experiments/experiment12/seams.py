"""seams.py — механические проверки для эксперимента 12.

Переиспользует низкоуровневые экстракторы из `experiment11/seams.py`
(`extract_numeric_value` для чисел построчно, `extract_code`+`ast`+`exec`
для кода) -- этот код прошёл 2790 вызовов в Эксперименте 11 без единой
ошибки конвейера, переписывать его заново означало бы повторно вносить
уже исправленные там дефекты извлечения. Новое здесь -- ПОШАГОВАЯ
классификация: у каждого шага арифметической задачи / requirement
кодовой задачи есть собственный статус, а не только общий вердикт.
Именно эти пошаговые статусы дают материал для K2 в
`context_builder.py` (какие шаги подтверждены).

PASS/FAIL/UNKNOWN/INAPPLICABLE/ERROR разделены так же строго, как в
Эксперименте 11: UNKNOWN не встречается (роль «не могу определить»
берёт на себя INAPPLICABLE), ERROR -- сбой конвейера проверки, не
кандидата, возвращается только оборачивающим кодом харнесса.
"""

from __future__ import annotations

import ast
import importlib.util
import threading
from pathlib import Path

_EXEC_TIMEOUT_SEC = 5.0


def _run_with_timeout(func, timeout_sec: float = _EXEC_TIMEOUT_SEC):
    """Выполняет func() в отдельном daemon-потоке с ограничением по
    времени -- защита от кандидатского кода с ДОСТИЖИМЫМ бесконечным
    циклом (напр. `while True: pass` внутри проверяемой функции).
    Найдено на реальном пилоте эксперимента 13: `eval(req["check"], ...)`
    без таймаута блокировал процесс НАВСЕГДА -- подтверждено отдельным
    воспроизведением (см. STATUS.md/отчёт эксп.13). В Python нет
    портируемого способа принудительно убить поток -- превысивший лимит
    поток останется висеть в фоне (daemon, не блокирует выход процесса и
    не блокирует ДАЛЬНЕЙШУЮ работу вызывающего кода), но вызывающий код
    получает TimeoutError и продолжает, не дожидаясь его."""
    box: dict = {}

    def _target():
        try:
            box["value"] = func()
        except BaseException as exc:  # noqa: BLE001 -- нужно поймать буквально всё из кандидатского кода
            box["error"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_sec)
    if t.is_alive():
        return None, TimeoutError(f"execution exceeded {timeout_sec}s (possible infinite loop in candidate code)")
    if "error" in box:
        return None, box["error"]
    return box.get("value"), None

# Загрузка по явному пути, не через sys.path/import seams -- у этого
# файла то же имя модуля ("seams"), что у experiment11/seams.py, и
# обычный `import seams` после добавления experiment11 в sys.path нашёл
# бы в sys.modules этот же (ещё не до конца загруженный) модуль вместо
# соседнего -- явная загрузка по пути исключает эту коллизию имён.
_EXPERIMENT11_SEAMS_PATH = Path(__file__).resolve().parents[1] / "experiment11" / "seams.py"
_spec = importlib.util.spec_from_file_location("experiment11_seams", _EXPERIMENT11_SEAMS_PATH)
_experiment11_seams = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_experiment11_seams)

ERROR = _experiment11_seams.ERROR
FAIL = _experiment11_seams.FAIL
INAPPLICABLE = _experiment11_seams.INAPPLICABLE
PASS = _experiment11_seams.PASS
UNKNOWN = _experiment11_seams.UNKNOWN
extract_code = _experiment11_seams.extract_code
extract_numeric_value = _experiment11_seams.extract_numeric_value

__all__ = [
    "ERROR", "FAIL", "INAPPLICABLE", "PASS", "UNKNOWN",
    "multistep_arithmetic_seam", "enriched_code_seam",
    "classify_arithmetic_subtype", "classify_code_subtype", "collision_subtype",
]


def _parse_step_lines(text: str) -> dict:
    """line 'name = ...' -> {name.lower(): [значение_после_первого '=', ...]}
    (список, чтобы отследить повтор одного имени -- неоднозначность)."""
    by_key: dict = {}
    if not isinstance(text, str):
        return by_key
    for line in text.split("\n"):
        if "=" not in line:
            continue
        left, _, right = line.partition("=")
        key = left.strip().lower()
        if not key:
            continue
        by_key.setdefault(key, []).append(right)
    return by_key


def multistep_arithmetic_seam(text: str, steps: list, tolerance: float = 1e-6) -> dict:
    """steps -- task['steps']: [{"name","formula","value"}, ...] (formula
    используется ТОЛЬКО для получения оракульного 'value' при построении
    задачи -- сюда формулы не передаются и не проверяются, только имя+
    оракульное значение).

    Возвращает {"status","collision_type","details","step_results"}, где
    step_results -- статус КАЖДОГО шага (не только финального) --
    материал для K2 в context_builder."""
    by_key = _parse_step_lines(text)
    step_results = []
    for step in steps:
        name = step["name"]
        oracle = step["value"]
        matches = by_key.get(name.lower(), [])
        if not matches:
            step_results.append({"name": name, "status": INAPPLICABLE, "reason": "missing", "extracted": None, "oracle": oracle})
            continue
        if len(matches) > 1:
            step_results.append({"name": name, "status": INAPPLICABLE, "reason": "duplicate_step_name", "extracted": None, "oracle": oracle})
            continue
        value, reason = extract_numeric_value(matches[0])
        if value is None:
            step_results.append({"name": name, "status": INAPPLICABLE, "reason": reason, "extracted": None, "oracle": oracle})
            continue
        if abs(value - oracle) <= tolerance:
            step_results.append({"name": name, "status": PASS, "reason": "match", "extracted": value, "oracle": oracle})
        else:
            step_results.append({"name": name, "status": FAIL, "reason": "mismatch", "extracted": value, "oracle": oracle})

    final = step_results[-1]
    if final["status"] == PASS:
        return {"status": PASS, "collision_type": None, "details": {"final": final}, "step_results": step_results}
    if final["status"] == INAPPLICABLE:
        return {
            "status": INAPPLICABLE, "collision_type": "format",
            "details": {"final": final, "reason": final["reason"]}, "step_results": step_results,
        }
    return {
        "status": FAIL, "collision_type": "numeric",
        "details": {"final": final, "actual": final["extracted"], "expected": final["oracle"]},
        "step_results": step_results,
    }


def enriched_code_seam(text: str, function_name: str, requirements: list) -> dict:
    """requirements -- task['requirements']: [{"name","description","check"}, ...].

    Стадийная классификация -- та же, что `experiment11/seams.code_seam`:
    нет кода -> INAPPLICABLE/format; SyntaxError/exec-ошибка -> FAIL/syntactic;
    имя функции не найдено -> FAIL/structural; иначе -- каждый requirement
    вычисляется независимо, любой провал -> FAIL/logical, все проходят -> PASS.

    Если код не дошёл до стадии выполнения требований (format/syntactic/
    structural), ВСЕ requirement_results помечаются INAPPLICABLE -- ни один
    не мог быть проверен, K2 в этом случае честно вырождается в K1."""
    source = extract_code(text)
    if source is None:
        req_results = [{"name": r["name"], "status": INAPPLICABLE} for r in requirements]
        return {
            "status": INAPPLICABLE, "collision_type": "format",
            "details": {"reason": "no code found in output"}, "requirement_results": req_results,
        }

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        req_results = [{"name": r["name"], "status": INAPPLICABLE} for r in requirements]
        return {
            "status": FAIL, "collision_type": "syntactic",
            "details": {"reason": f"SyntaxError: {exc}", "source": source}, "requirement_results": req_results,
        }

    fn_names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if function_name not in fn_names:
        req_results = [{"name": r["name"], "status": INAPPLICABLE} for r in requirements]
        return {
            "status": FAIL, "collision_type": "structural",
            "details": {"reason": f"function {function_name!r} not defined", "found": fn_names, "source": source},
            "requirement_results": req_results,
        }

    namespace: dict = {}
    _, exec_err = _run_with_timeout(lambda: exec(compile(source, "<candidate>", "exec"), namespace))  # noqa: S102 -- локальный sandboxed стенд, тот же паттерн, что experiment11/seams.py; таймаут -- см. _run_with_timeout
    if exec_err is not None:
        req_results = [{"name": r["name"], "status": INAPPLICABLE} for r in requirements]
        reason = (
            f"execution timeout: {exec_err}" if isinstance(exec_err, TimeoutError)
            else f"exec error: {type(exec_err).__name__}: {exec_err}"
        )
        return {
            "status": FAIL, "collision_type": "syntactic",
            "details": {"reason": reason, "source": source},
            "requirement_results": req_results,
        }

    req_results = []
    failing = []
    for req in requirements:
        value, err = _run_with_timeout(lambda check=req["check"]: eval(check, dict(namespace)))  # noqa: S307 -- фиксированные проверки из tasks/code_tasks.py; таймаут -- см. _run_with_timeout
        if err is not None:
            ok = False
            reason = f"TIMEOUT: {err}" if isinstance(err, TimeoutError) else f"{type(err).__name__}: {err}"
            failing.append(f"{req['name']}: {req['check']} -> {reason}")
        else:
            ok = value
            if not ok:
                failing.append(f"{req['name']}: {req['check']} -> False")
        req_results.append({"name": req["name"], "status": PASS if ok else FAIL})

    if failing:
        return {
            "status": FAIL, "collision_type": "logical",
            "details": {"failing_requirements": failing, "source": source}, "requirement_results": req_results,
        }
    return {
        "status": PASS, "collision_type": None,
        "details": {"n_requirements_passed": len(requirements), "source": source}, "requirement_results": req_results,
    }


# -- dependency vs value-calculation subtype (задание v2, таксономия коллизий) --

_CODE_SUBTYPE_MAP = {"syntactic": "syntax", "structural": "contract_interface", "logical": "semantic", "format": "other"}


def classify_arithmetic_subtype(step_results: list, steps: list) -> str:
    """'dependency' | 'value' | 'other'. Применимо только когда общий
    статус состояния -- FAIL (финальный шаг: значение извлечено, но не
    совпало с оракулом).

    Пересчитывает формулу ФИНАЛЬНОГО шага, подставляя СОБСТВЕННЫЕ (не
    оракульные) извлечённые значения модели для шагов, от которых он
    зависит:
      - пересчёт совпадает с тем, что модель написала для финального
        шага -> 'dependency' (шаг корректно применил формулу к
        унаследованным неверным входам -- ошибка пришла СВЕРХУ, не из
        этого шага);
      - не совпадает -> 'value' (свежая ошибка вычисления В ЭТОМ шаге,
        независимо от того, были ли входы верны);
      - пересчёт невозможен (какая-то зависимость сама не читаема,
        INAPPLICABLE/MISSING) -> 'other'."""
    final_result = step_results[-1]
    final_step = steps[-1]
    if final_result["status"] != FAIL or final_result.get("extracted") is None:
        return "other"

    namespace: dict = {}
    for r in step_results[:-1]:
        if r.get("extracted") is None:
            return "other"
        namespace[r["name"]] = r["extracted"]

    try:
        recomputed = float(eval(final_step["formula"], {"__builtins__": {}}, namespace))  # noqa: S307 -- фиксированные формулы задач, не пользовательский ввод
    except Exception:
        return "other"

    if abs(recomputed - final_result["extracted"]) <= 1e-6:
        return "dependency"
    return "value"


def classify_code_subtype(collision_type) -> str:
    """Прямое отображение существующей collision_type-таксономии кода на
    словарь задания. 'dependency'/'value' неприменимы к кодовым задачам
    этого эксперимента -- requirements независимы, не образуют цепочку
    (в отличие от шагов арифметики) -- честно не подгоняется, см. отчёт."""
    return _CODE_SUBTYPE_MAP.get(collision_type, "other")


def collision_subtype(task_family: str, seam_result: dict, steps=None):
    """None, если состояние не FAIL (подтип определён только для FAIL --
    для INAPPLICABLE/PASS/UNVERIFIED категория 'типа коллизии' не имеет
    смысла)."""
    if seam_result.get("status") != FAIL:
        return None
    if task_family == "arithmetic":
        return classify_arithmetic_subtype(seam_result["step_results"], steps)
    return classify_code_subtype(seam_result.get("collision_type"))
