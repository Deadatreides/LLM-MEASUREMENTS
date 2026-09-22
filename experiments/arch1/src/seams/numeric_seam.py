"""numeric_seam — HARD-шов сверки чисел (SEAM_MODEL.md §3).

Сверяет числовой результат с механическим oracle задачи. Детерминированное
сравнение -- EXACT_MATCH-класс evidence (EVIDENCE_MODEL.md §2).

ИЗВЛЕЧЕНИЕ ЗНАЧЕНИЯ ВНУТРИ ШВА (исправление, найденное на этапе 8).
Первая версия принимала уже извлечённое `actual`, а само извлечение
(«последнее число в тексте») жило в вызывающем коде — orchestrator.py и
metrics/runner.py — и потому НЕ покрывалось ни self_test, ни гейтом
доверия. На реальной кампании это дало 59.6% ЛОЖНЫХ FAIL: модель писала
верный ответ claim'а и продолжала решать задачу дальше, а «последнее
число» относилось уже к другому утверждению. Шов при этом был исправен —
сломан был неподнадзорный участок ПЕРЕД ним.

Поэтому извлечение перенесено внутрь шва: теперь оно под контрольными
случаями и под правилом «недоверенный шов не даёт HARD-evidence».

Правило извлечения (и почему именно такое). «Умного» правила не
существует: если claim нарушил свою границу и содержит несколько
результатов, то для C4_SUBTOTAL_A верен ПЕРВЫЙ из них, а для
C6_FINAL_TOTAL — ПОСЛЕДНИЙ; любой выбор ломает противоположный случай.
Гадать, какое из чисел claim «имел в виду», — значит подменять проверку
эвристикой. Архитектурно честный ответ: несколько результатов в claim'е,
который обязан утверждать ровно один, — это НАРУШЕНИЕ ГРАНИЦЫ, то есть
INAPPLICABLE (проверка неприменима), а не FAIL (утверждение неверно) и не
угаданный PASS. Различение INAPPLICABLE/FAIL здесь принципиально —
SEAM_MODEL.md §4 и F44.
"""

import re
from numbers import Number

from ..seam_engine import ControlCase, FAIL, INAPPLICABLE, PASS, SeamDefinition

SEAM_ID = "seam:numeric"
SEAM_TYPE = "SEAM_EXECUTION"

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# результат арифметического утверждения -- правая часть равенства;
# валюта/пробелы допускаются: "= $40", "=40", "= 40.5"
_EQUATION_RESULT_RE = re.compile(r"=\s*[^\d\-]{0,3}(-?\d+(?:\.\d+)?)")


def extract_value(text: str):
    """Возвращает (value, reason). value=None => извлечь нельзя, reason
    объясняет почему (уходит в details, чтобы диагностика была видимой)."""
    if not isinstance(text, str):
        return None, "text is not a string"

    results = _EQUATION_RESULT_RE.findall(text)
    if len(results) == 1:
        return float(results[0]), "single equation result"
    if len(results) > 1:
        # claim обязан утверждать ровно один результат -- несколько
        # равенств означают, что он вышел за свою границу; какое из них
        # является утверждением claim'а, определить нечем
        return None, f"claim boundary violation: {len(results)} equation results"

    numbers = _NUMBER_RE.findall(text)
    if len(numbers) == 1:
        return float(numbers[0]), "single bare number"
    if not numbers:
        return None, "no numbers found"
    return None, f"ambiguous: {len(numbers)} numbers, no equation"


def check(inputs: dict) -> dict:
    """Принимает либо уже извлечённое `actual` (прямое сравнение), либо
    `text` — тогда извлечение делает сам шов (см. докстринг модуля)."""
    expected = inputs.get("expected")
    tolerance = inputs.get("tolerance", 0.0)

    if "text" in inputs and "actual" not in inputs:
        actual, reason = extract_value(inputs["text"])
        if actual is None:
            return {
                "status": INAPPLICABLE,
                "details": {"reason": reason, "expected": expected, "text": inputs["text"]},
            }
    else:
        actual = inputs.get("actual")

    if not isinstance(actual, Number) or not isinstance(expected, Number):
        return {
            "status": INAPPLICABLE,
            "details": {"reason": "actual/expected not both numeric", "actual": actual, "expected": expected},
        }

    diff = abs(actual - expected)
    details = {"actual": actual, "expected": expected, "diff": diff, "tolerance": tolerance}
    if diff <= tolerance:
        return {"status": PASS, "details": details}
    return {"status": FAIL, "details": details}


CONTROL_CASES = (
    ControlCase(name="exact_match", inputs={"actual": 42, "expected": 42}, expected_status=PASS),
    ControlCase(name="mismatch", inputs={"actual": 41, "expected": 42}, expected_status=FAIL),
    ControlCase(
        name="non_numeric_actual",
        inputs={"actual": None, "expected": 42},
        expected_status=INAPPLICABLE,
    ),
    # -- контрольные случаи извлечения. Написаны по ФОРМЕ дефекта,
    # найденного на этапе 8, а не подобраны так, чтобы метрика выглядела
    # лучше (SEAM_MODEL.md §5.3 -- запрет на подстройку под результат).
    ControlCase(
        name="extract_from_single_equation",
        inputs={"text": "2 * $20 = $40", "expected": 40},
        expected_status=PASS,
    ),
    ControlCase(
        name="extract_from_single_equation_wrong_value",
        inputs={"text": "2 * $20 = $50", "expected": 40},
        expected_status=FAIL,
    ),
    ControlCase(
        name="extract_from_bare_number",
        inputs={"text": "40", "expected": 40},
        expected_status=PASS,
    ),
    ControlCase(
        # ЭТОТ случай воспроизводит дефект, из-за которого 59.6% FAIL
        # кампании этапа 7 были ложными: claim про subtotal A содержит
        # ещё и subtotal B, и итог. Правильный ответ -- INAPPLICABLE
        # (нельзя определить, что утверждает claim), НЕ FAIL и НЕ PASS.
        name="claim_boundary_violation_is_inapplicable_not_fail",
        inputs={"text": "2 * $20 = $40\n3 * $35 = $105\n$40 + $105 = $145", "expected": 40},
        expected_status=INAPPLICABLE,
    ),
    ControlCase(
        name="no_numbers_is_inapplicable",
        inputs={"text": "I cannot compute this.", "expected": 40},
        expected_status=INAPPLICABLE,
    ),
    ControlCase(
        name="ambiguous_numbers_without_equation_is_inapplicable",
        inputs={"text": "quantity 2, rate 20", "expected": 40},
        expected_status=INAPPLICABLE,
    ),
)


def build_definition() -> SeamDefinition:
    return SeamDefinition(
        seam_id=SEAM_ID,
        seam_type=SEAM_TYPE,
        check=check,
        control_cases=CONTROL_CASES,
        applicable_claim_types=frozenset({"VALUE", "DERIVATION_STEP"}),
    )
