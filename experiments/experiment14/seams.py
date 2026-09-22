"""seams.py — механические швы эксперимента 14. Оракул известен, LLM-судей нет.

Три шва под три рода шагов: `numeric_seam` (READ, COMPUTE), `token_seam` (FORMAT),
и `whole_seam` — разбор целостного отклика плеча A на три значения.

ПРАВИЛО ИЗВЛЕЧЕНИЯ, объявленное ДО прогона (два яруса, оба — проверка СОБЛЮДЕНИЯ
контракта, ни один не угадывает среди конкурирующих кандидатов):

  ярус 1: весь отклик целиком является значением       -> сравниваем;
  ярус 2: ровно ОДНА строка отклика состоит только из значения -> сравниваем;
  иначе                                                 -> INAPPLICABLE.

Почему именно так. Контракт промпта требует «выведи только одно значение», поэтому
строка, состоящая ровно из значения, — соблюдение контракта, а не догадка. Но если
таких строк несколько, выбирать между ними значило бы гадать, что модель «имела в
виду» — а это ровно тот дефект, который arch1 нашёл у себя на этапе 8 и запретил:
«несколько результатов там, где обязан быть один, — это НАРУШЕНИЕ ГРАНИЦЫ (INAPPLICABLE),
а не неверное утверждение (FAIL)».

Различение INAPPLICABLE и FAIL здесь принципиально: первое — «проверить нечем»,
второе — «утверждение неверно». Смешивать их нельзя (F43/F44).
"""

from __future__ import annotations

import re

PASS = "PASS"
FAIL = "FAIL"
INAPPLICABLE = "INAPPLICABLE"
ERROR = "ERROR"

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*[=:]\s*(.+?)\s*$")


def _lines(text: str) -> list:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _strip_value(s: str) -> str:
    """Снимает обрамление, не меняющее значения: кавычки, точку в конце, валютные знаки,
    пробелы-разделители тысяч. Это нормализация записи, а не выбор среди кандидатов."""
    s = (s or "").strip().strip('"\'`*')
    s = s.rstrip(".").strip()
    s = re.sub(r"^[€$£]\s*|\s*(?:EUR|USD|GBP)$", "", s).strip()
    return s


# -- числовой шов ---------------------------------------------------------------------------


def _as_number(s: str):
    s = _strip_value(s).replace(" ", "").replace(" ", "")
    m = _NUMBER_RE.fullmatch(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


def numeric_seam(text: str, oracle: float, tolerance: float = 0.011) -> dict:
    """Толеранс 0.011 покрывает округление до двух знаков в обе стороны и не более —
    он объявлен здесь до прогона и не подбирается по результату."""
    if not isinstance(text, str):
        return {"status": ERROR, "reason": "not a string", "extracted": None, "oracle": oracle}

    whole = _as_number(text)
    if whole is not None:
        cand, tier = whole, 1
    else:
        hits = [v for ln in _lines(text) if (v := _as_number(ln)) is not None]
        if not hits:
            return {"status": INAPPLICABLE, "reason": "no value-only line",
                    "extracted": None, "oracle": oracle}
        if len(hits) > 1:
            return {"status": INAPPLICABLE, "reason": f"ambiguous: {len(hits)} value-only lines",
                    "extracted": None, "oracle": oracle}
        cand, tier = hits[0], 2

    ok = abs(cand - oracle) <= tolerance
    return {"status": PASS if ok else FAIL, "reason": f"tier{tier}",
            "extracted": cand, "oracle": oracle}


# -- токенный шов (дата) ---------------------------------------------------------------------


def token_seam(text: str, oracle: str) -> dict:
    if not isinstance(text, str):
        return {"status": ERROR, "reason": "not a string", "extracted": None, "oracle": oracle}

    whole = _strip_value(text)
    if _ISO_DATE_RE.fullmatch(whole):
        cand, tier = whole, 1
    else:
        hits = [v for ln in _lines(text)
                if _ISO_DATE_RE.fullmatch(v := _strip_value(ln))]
        if not hits:
            return {"status": INAPPLICABLE, "reason": "no ISO-date-only line",
                    "extracted": None, "oracle": oracle}
        if len(hits) > 1:
            return {"status": INAPPLICABLE, "reason": f"ambiguous: {len(hits)} date lines",
                    "extracted": None, "oracle": oracle}
        cand, tier = hits[0], 2

    return {"status": PASS if cand == oracle else FAIL, "reason": f"tier{tier}",
            "extracted": cand, "oracle": oracle}


SEAMS = {"numeric": numeric_seam, "token": token_seam}


def check_step(step: dict, text: str) -> dict:
    out = SEAMS[step["seam"]](text, step["oracle"])
    return {"name": step["name"], "kind": step["kind"], **out}


# -- целостный шов (плечо A) -------------------------------------------------------------------


def whole_seam(text: str, steps: list) -> dict:
    """Разбирает отклик вида «имя = значение» и прогоняет каждое значение своим швом.

    Повтор имени -> INAPPLICABLE для этого шага (`duplicate_name`): выбирать, какое из
    двух присваиваний «настоящее», значило бы гадать. Ровно этот дефект arch1 нашёл у
    себя на этапе 8 и запретил.
    """
    if not isinstance(text, str):
        return {"status": ERROR, "step_results": [], "reason": "not a string"}

    seen: dict = {}
    dup = set()
    for ln in _lines(text):
        m = _ASSIGN_RE.match(ln)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        if name in seen:
            dup.add(name)
        seen[name] = val

    results = []
    for st in steps:
        if st["name"] in dup:
            results.append({"name": st["name"], "kind": st["kind"], "status": INAPPLICABLE,
                            "reason": "duplicate_name", "extracted": None, "oracle": st["oracle"]})
        elif st["name"] not in seen:
            results.append({"name": st["name"], "kind": st["kind"], "status": INAPPLICABLE,
                            "reason": "missing", "extracted": None, "oracle": st["oracle"]})
        else:
            results.append(check_step(st, seen[st["name"]]))

    n_pass = sum(r["status"] == PASS for r in results)
    if n_pass == len(steps):
        status = PASS
    elif any(r["status"] == FAIL for r in results):
        status = FAIL
    else:
        status = INAPPLICABLE
    return {"status": status, "step_results": results, "n_pass": n_pass}


# -- контрольные случаи (известный ответ ПО КОНТРАКТУ, не по прогону) ------------------------

_CONTROL_CASES = {
    "numeric_seam": [
        ("отклик целиком есть значение -> PASS", ("42", 42.0), PASS),
        ("отклик целиком есть значение, но неверное -> FAIL", ("41", 42.0), FAIL),
        ("значение на отдельной строке среди рассуждений -> PASS (ярус 2)",
         ("Считаем: 16 x 53.39 = 854.24\n1033.63", 1033.63), PASS),
        ("две строки-значения -> INAPPLICABLE, не выбор из двух",
         ("1033.63\n1033.64", 1033.63), INAPPLICABLE),
        ("ни одной строки-значения -> INAPPLICABLE",
         ("итого будет примерно 1033.63 рублей", 1033.63), INAPPLICABLE),
        ("округление до двух знаков в пределах толеранса -> PASS", ("1033.64", 1033.63), PASS),
        ("запятая как десятичный разделитель -> PASS", ("1033,63", 1033.63), PASS),
        ("валютный знак снимается -> PASS", ("€1033.63", 1033.63), PASS),
        ("пустой отклик -> INAPPLICABLE", ("", 42.0), INAPPLICABLE),
    ],
    "token_seam": [
        ("отклик целиком есть дата -> PASS", ("2026-03-07", "2026-03-07"), PASS),
        ("верный формат, неверная дата -> FAIL", ("2026-03-08", "2026-03-07"), FAIL),
        ("дата отдельной строкой -> PASS (ярус 2)",
         ("Дата отгрузки:\n2026-03-07", "2026-03-07"), PASS),
        ("две даты -> INAPPLICABLE", ("2026-03-07\n2026-03-08", "2026-03-07"), INAPPLICABLE),
        ("неверный формат -> INAPPLICABLE, а не FAIL",
         ("07 March 2026", "2026-03-07"), INAPPLICABLE),
        ("дата внутри фразы -> INAPPLICABLE",
         ("отгрузка была 2026-03-07 утром", "2026-03-07"), INAPPLICABLE),
    ],
}

_WHOLE_STEPS = [
    {"name": "units", "kind": "READ", "seam": "numeric", "oracle": 3.0},
    {"name": "ship_date", "kind": "FORMAT", "seam": "token", "oracle": "2026-03-07"},
    {"name": "total_with_vat", "kind": "COMPUTE", "seam": "numeric", "oracle": 45.38},
]

_WHOLE_CASES = [
    ("все три верны -> PASS",
     "units = 3\nship_date = 2026-03-07\ntotal_with_vat = 45.38", PASS, 3),
    ("одно неверно -> FAIL",
     "units = 3\nship_date = 2026-03-07\ntotal_with_vat = 45.00", FAIL, 2),
    ("одно отсутствует -> INAPPLICABLE у него, общий не PASS",
     "units = 3\nship_date = 2026-03-07", INAPPLICABLE, 2),
    ("повтор имени -> INAPPLICABLE у него, не выбор из двух",
     "units = 3\nunits = 4\nship_date = 2026-03-07\ntotal_with_vat = 45.38", INAPPLICABLE, 2),
    ("свободный текст без присваиваний -> INAPPLICABLE",
     "Заказ на три единицы, отгружен седьмого марта.", INAPPLICABLE, 0),
]


def self_test() -> dict:
    failures = []
    for name, cases in _CONTROL_CASES.items():
        fn = {"numeric_seam": numeric_seam, "token_seam": token_seam}[name]
        for label, args, expected in cases:
            try:
                got = fn(*args)["status"]
            except Exception as exc:                                   # noqa: BLE001
                failures.append(f"{name}: {label}: raised {type(exc).__name__}: {exc}")
                continue
            if got != expected:
                failures.append(f"{name}: {label}: expected {expected}, got {got}")

    for label, text, expected, n_pass in _WHOLE_CASES:
        try:
            res = whole_seam(text, _WHOLE_STEPS)
        except Exception as exc:                                       # noqa: BLE001
            failures.append(f"whole_seam: {label}: raised {type(exc).__name__}: {exc}")
            continue
        if res["status"] != expected:
            failures.append(f"whole_seam: {label}: expected {expected}, got {res['status']}")
        if res["n_pass"] != n_pass:
            failures.append(f"whole_seam: {label}: expected n_pass={n_pass}, got {res['n_pass']}")

    n = sum(len(v) for v in _CONTROL_CASES.values()) + len(_WHOLE_CASES)
    return {"passed": not failures, "n_cases": n, "failures": failures}


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rep = self_test()
    print(f"контрольные случаи швов: {rep['n_cases']}, "
          f"{'ВСЕ ПРОЙДЕНЫ' if rep['passed'] else 'ЕСТЬ ПРОВАЛЫ'}")
    for f in rep["failures"]:
        print("  -", f)
    raise SystemExit(0 if rep["passed"] else 1)
