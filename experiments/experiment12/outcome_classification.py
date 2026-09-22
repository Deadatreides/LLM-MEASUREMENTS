"""outcome_classification.py — пятисоставный статус результата действия
и учёт регрессий/улучшений (задание v2, "ОСНОВНЫЕ МЕТРИКИ" +
"ОСОБЕННО ВАЖНО").

final_status одного плеча — ОДНО из:
  PASS              — retry прошёл механическую проверку целиком.
  SAME_FAILURE      — retry всё ещё FAIL, и сигнатура ошибки совпадает
                       с исходной (это НЕ исправление, исходная коллизия
                       не устранена).
  DIFFERENT_FAILURE — retry всё ещё FAIL, но сигнатура другая (новая
                       коллизия появилась, старая, возможно, исчезла --
                       это тоже НЕ успех, только другой отказ).
  INAPPLICABLE      — нарушение ФОРМАТА ответа (нет кода / неоднозначная
                       строка / пропущенный шаг) -- то же значение, что
                       уже возвращает seams.py.
  UNVERIFIED        — формат ответа корректен (шов дошёл до сравнения с
                       оракулом), но конкретную проверку механически
                       провести не удалось. В швах этого эксперимента
                       такого пути формально нет по конструкции (либо
                       есть однозначное значение для сравнения, либо это
                       INAPPLICABLE) -- статус существует в схеме и
                       проверяется тестом на искусственном случае;
                       фактическая частота (ожидаемо 0%) -- честный
                       результат, не дефект, см. REPORT_EXPERIMENT12_V2.md.

ERROR (сбой конвейера генерации/проверки, не кандидата) остаётся ШЕСТЫМ,
отдельным значением -- тот же принцип разделения, что во всех предыдущих
экспериментах проекта (ERROR никогда не транслируется в FAIL/INAPPLICABLE).
"""

from __future__ import annotations

from seams import ERROR, FAIL, INAPPLICABLE, PASS, UNKNOWN  # noqa: F401 -- UNKNOWN не используется здесь намеренно, см. докстринг

PASS_STATUS = "PASS"
SAME_FAILURE = "SAME_FAILURE"
DIFFERENT_FAILURE = "DIFFERENT_FAILURE"
INAPPLICABLE_STATUS = "INAPPLICABLE"
UNVERIFIED = "UNVERIFIED"
ERROR_STATUS = "ERROR"

ALL_FINAL_STATUSES = (PASS_STATUS, SAME_FAILURE, DIFFERENT_FAILURE, INAPPLICABLE_STATUS, UNVERIFIED, ERROR_STATUS)


def _failure_signature(seam_result: dict):
    """Сигнатура FAIL-исхода -- используется только для сравнения 'та же
    коллизия vs другая коллизия', не для UI. Стабильна к перефразировке
    evidence (использует только структурированные details, не текст)."""
    ctype = seam_result.get("collision_type")
    details = seam_result.get("details") or {}
    if ctype == "numeric":
        final = details.get("final") or {}
        return (ctype, final.get("name"), final.get("extracted"))
    if ctype == "syntactic":
        return (ctype, details.get("reason"))
    if ctype == "structural":
        return (ctype, tuple(details.get("found") or ()))
    if ctype == "logical":
        failing = details.get("failing_requirements") or []
        names = tuple(sorted((f.split(":", 1)[0].strip() for f in failing)))
        return (ctype, names)
    return (ctype, None)


def classify_outcome(initial_seam_result: dict, retry_seam_result: dict) -> str:
    """Пять(+1) исходов -- см. докстринг модуля."""
    status = retry_seam_result.get("status")
    if status == ERROR:
        return ERROR_STATUS
    if status == PASS:
        return PASS_STATUS
    if status == INAPPLICABLE:
        return INAPPLICABLE_STATUS
    if status == UNKNOWN:
        # архитектурно не должно происходить (см. seams.py) -- если
        # когда-нибудь произойдёт, честно помечаем как UNVERIFIED, а не
        # молча укладываем в FAIL/PASS.
        return UNVERIFIED
    if status != FAIL:
        return UNVERIFIED
    sig_initial = _failure_signature(initial_seam_result)
    sig_retry = _failure_signature(retry_seam_result)
    return SAME_FAILURE if sig_initial == sig_retry else DIFFERENT_FAILURE


def _results_dict(seam_result: dict) -> dict:
    key = "step_results" if "step_results" in seam_result else "requirement_results"
    return {r["name"]: r["status"] for r in seam_result.get(key, [])}


def detect_regressions(initial_seam_result: dict, retry_seam_result: dict) -> list:
    """Имена шагов/requirements, что были PASS на первичной генерации и
    перестали быть PASS на этом плече -- буквально 'сломанное ранее
    исправное свойство' (задание, "ОСНОВНЫЕ МЕТРИКИ")."""
    init_status = _results_dict(initial_seam_result)
    retry_status = _results_dict(retry_seam_result)
    return sorted(name for name, st in init_status.items() if st == PASS and retry_status.get(name) != PASS)


def detect_improvements(initial_seam_result: dict, retry_seam_result: dict) -> list:
    """Имена шагов/requirements, что НЕ были PASS на первичной генерации
    и стали PASS на этом плече -- 'результат стал лучше исходного', даже
    если итоговый вердикт всё ещё не PASS."""
    init_status = _results_dict(initial_seam_result)
    retry_status = _results_dict(retry_seam_result)
    return sorted(name for name, st in init_status.items() if st != PASS and retry_status.get(name) == PASS)
