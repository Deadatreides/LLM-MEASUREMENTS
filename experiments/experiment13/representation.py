"""representation.py — парсинг ответов R1/R2 на секции, структурные
fingerprint'ы и классификация "настоящей" новизны представления против
косметического перефразирования (задание §6, §10).

Явно эвристический модуль (задание §6: "если автоматическая
классификация ненадёжна, сохранить сырые структуры для последующей
ручной разметки") -- сырые тексты (`representation_after`,
`decomposition_after`, распарсенные fingerprint) ВСЕГДА сохраняются в
raw-записи независимо от уверенности классификации, см. harness.py.
"""

from __future__ import annotations

import ast
import re

from seams import extract_code

NOVEL_STRUCTURE = "NOVEL_STRUCTURE"
MINOR_REFORMULATION = "MINOR_REFORMULATION"
SAME_STRUCTURE = "SAME_STRUCTURE"
INVALID_STRUCTURE = "INVALID_STRUCTURE"

# -- парсинг секций ответа ----------------------------------------------------

_SECTION_MARKERS = ("NEW REPRESENTATION:", "DECOMPOSITION DECISION:", "NEW DECOMPOSITION:", "SOLUTION:", "FINAL ANSWER")


def parse_sections(text: str) -> dict:
    """{marker_without_colon: text_until_next_marker}. Маркеры ищутся по
    факту появления в тексте (не по фиксированной позиции) -- терпимо к
    небольшим отклонениям модели от идеального форматирования."""
    if not isinstance(text, str):
        return {}
    positions = []
    for marker in _SECTION_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            positions.append((idx, marker))
    positions.sort()
    sections = {}
    for i, (idx, marker) in enumerate(positions):
        start = idx + len(marker)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections[marker.rstrip(":")] = text[start:end].strip()
    return sections


def representation_text(mode: str, sections: dict) -> str:
    """Сырой текст 'нового представления' для сохранения и сравнения."""
    if mode == "R1":
        return sections.get("NEW REPRESENTATION", "")
    if mode == "R2":
        parts = [sections.get("DECOMPOSITION DECISION", ""), sections.get("NEW DECOMPOSITION", "")]
        return "\n".join(p for p in parts if p)
    return ""


# -- структурные fingerprint --------------------------------------------------

_STEP_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")
# запасной разбор строк-прозы вида "Calculate the cost of sugar: $2.5kg * $2/kg = $5"
# -- метка до двоеточия, последнее число после последнего "=" в строке.
_LABELED_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _/'-]{2,60}?):\s*.*=\s*\$?(-?\d[\d,\.]*)\s*\.?$")


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def arithmetic_fingerprint_from_text(solution_text: str) -> dict:
    """Построчный разбор ПРОИЗВОЛЬНЫХ (не заранее известных) шагов вида
    `name = ...` в SOLUTION-секции ответа R1/R2. То же извлекающее
    правило, что multistep_arithmetic_seam, но без фиксированного
    словаря ожидаемых имён -- имена здесь как раз и есть то, что
    измеряется.

    Промпт просит `name = value`, но модели (особенно в R1/R2, где формат
    решения не задан так же строго, как в R0) нередко пишут прозой с
    инлайн-вычислением ("Calculate the cost of sugar: $2.5kg * $2/kg = $5")
    -- найдено на реальном пилоте: строгий парсер давал INVALID_STRUCTURE
    даже на случаях с настоящей новой структурой (и даже на PASS-случаях).
    `_LABELED_LINE_RE` -- запасной разбор такой прозы: текст ДО двоеточия
    как метка шага, последнее число ПОСЛЕ последнего "=" в строке как
    значение."""
    names = []
    formulas: dict = {}
    for raw_line in (solution_text or "").split("\n"):
        line = raw_line.strip().lstrip("-*•").strip()
        m = _STEP_LINE_RE.match(line)
        if m:
            name = m.group(1).strip().lower()
            formula = m.group(2).strip()
        else:
            m2 = _LABELED_LINE_RE.match(line)
            if not m2:
                continue
            name = _normalize_label(m2.group(1))
            if not name:
                continue
            formula = line
        names.append(name)
        formulas[name] = formula
    name_set = frozenset(names)
    graph_edges = 0
    for name, formula in formulas.items():
        for other in name_set:
            if other != name and re.search(rf"\b{re.escape(other)}\b", formula, re.IGNORECASE):
                graph_edges += 1
    return {
        "parsed": len(names) > 0,
        "n_steps": len(names),
        "step_names": tuple(names),
        "step_name_set": name_set,
        "n_graph_edges": graph_edges,
    }


def arithmetic_fingerprint_from_task(task: dict) -> dict:
    """'До' -- из исходного определения задачи (task['steps']), не из
    текста -- точнее, чем эвристический разбор (формулы уже структурно
    известны)."""
    steps = task["steps"]
    names = [s["name"].lower() for s in steps]
    name_set = frozenset(names)
    graph_edges = 0
    for step in steps:
        formula = step["formula"]
        for other in name_set:
            if other != step["name"].lower() and re.search(rf"\b{re.escape(other)}\b", formula, re.IGNORECASE):
                graph_edges += 1
    return {"parsed": True, "n_steps": len(names), "step_names": tuple(names), "step_name_set": name_set, "n_graph_edges": graph_edges}


def code_fingerprint(source_text: str) -> dict:
    """AST-fingerprint кода (после первичного `extract_code` -- ищет
    Markdown-код-блок/`def` в тексте, тот же экстрактор, что все швы
    проекта)."""
    source = extract_code(source_text) if source_text else None
    if source is None:
        return {"parsed": False}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"parsed": False}

    n_functions = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    n_for = sum(1 for n in ast.walk(tree) if isinstance(n, ast.For))
    n_while = sum(1 for n in ast.walk(tree) if isinstance(n, ast.While))
    n_if = sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
    n_calls = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call))
    n_return = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Return))
    fn_names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    is_recursive = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in fn_names for n in ast.walk(tree)
    )
    return {
        "parsed": True, "n_functions": n_functions, "n_for": n_for, "n_while": n_while, "n_if": n_if,
        "n_calls": n_calls, "n_return": n_return, "is_recursive": is_recursive,
        "n_top_level_statements": len(tree.body),
    }


# -- классификация новизны -----------------------------------------------------


def _fuzzy_name_match(a: str, b: str) -> bool:
    """Скорее один и тот же концепт под другими словами, чем разный шаг:
    подстрока в любую сторону (напр. 'discount_amount' в
    'calculate_the_discount_amount'), или пересечение слов >= половины
    меньшего набора слов. Найдено на реальном пилоте: модели часто просто
    дописывают 'calculate_the_'/'add_the_' к прежним именам шагов --
    точное сравнение множеств засчитывало это как NOVEL_STRUCTURE,
    хотя вычисление осталось тем же (задание §10 -- псевдоновизна)."""
    if a == b:
        return True
    if a in b or b in a:
        return True
    wa, wb = set(a.split("_")), set(b.split("_"))
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.6


def _fuzzy_overlap_count(before_names: frozenset, after_names: frozenset) -> int:
    matched: set = set()
    count = 0
    for a_name in after_names:
        for b_name in before_names:
            if b_name in matched:
                continue
            if _fuzzy_name_match(a_name, b_name):
                matched.add(b_name)
                count += 1
                break
    return count


def _classify_arithmetic_novelty(before_fp: dict, after_fp: dict) -> str:
    if not after_fp.get("parsed"):
        return INVALID_STRUCTURE
    if before_fp["step_name_set"] == after_fp["step_name_set"] and before_fp["step_names"] == after_fp["step_names"]:
        return SAME_STRUCTURE
    fuzzy_overlap = _fuzzy_overlap_count(before_fp["step_name_set"], after_fp["step_name_set"])
    larger = max(len(before_fp["step_name_set"]), len(after_fp["step_name_set"]), 1)
    fuzzy_jaccard = fuzzy_overlap / larger
    n_diff = abs(before_fp["n_steps"] - after_fp["n_steps"])
    if fuzzy_jaccard >= 0.6 and n_diff <= 1:
        return MINOR_REFORMULATION
    return NOVEL_STRUCTURE


def _classify_code_novelty(before_fp: dict, after_fp: dict) -> str:
    if not after_fp.get("parsed"):
        return INVALID_STRUCTURE
    if not before_fp.get("parsed"):
        return NOVEL_STRUCTURE  # исходный артефакт сам был структурно невалиден (напр. синтаксическая коллизия) -- любой валидный ответ уже структурно другой
    key_fields = ("n_functions", "n_for", "n_while", "n_if", "is_recursive")
    delta = sum(1 for k in key_fields if before_fp.get(k) != after_fp.get(k))
    same_top_level = before_fp.get("n_top_level_statements") == after_fp.get("n_top_level_statements")
    if delta == 0 and same_top_level:
        return SAME_STRUCTURE
    if delta <= 1:
        return MINOR_REFORMULATION
    return NOVEL_STRUCTURE


def classify_novelty(task_family: str, before_fp: dict, after_fp: dict) -> str:
    if task_family == "arithmetic":
        return _classify_arithmetic_novelty(before_fp, after_fp)
    return _classify_code_novelty(before_fp, after_fp)


def classify_pseudo_novelty(task_family: str, before_fp: dict, after_fp: dict) -> str:
    """Пятизначная классификация §10 задания: genuine_new_method /
    order_change / local_modification / rephrasing / invalid_attempt.
    Более тонкая, чем classify_novelty -- различает "переставили те же
    шаги" от "просто переписали словами то же самое"."""
    if not after_fp.get("parsed"):
        return "invalid_attempt"

    if task_family == "arithmetic":
        if before_fp["step_name_set"] == after_fp["step_name_set"]:
            if before_fp["step_names"] == after_fp["step_names"]:
                return "rephrasing"
            return "order_change"
        fuzzy_overlap = _fuzzy_overlap_count(before_fp["step_name_set"], after_fp["step_name_set"])
        larger = max(len(before_fp["step_name_set"]), len(after_fp["step_name_set"]), 1)
        if fuzzy_overlap == larger:
            return "rephrasing"  # каждое имя нашло пару, только по-другому названо/переставлено
        fuzzy_jaccard = fuzzy_overlap / larger
        return "local_modification" if fuzzy_jaccard >= 0.5 else "genuine_new_method"

    if not before_fp.get("parsed"):
        return "genuine_new_method"
    key_fields = ("n_functions", "n_for", "n_while", "n_if", "is_recursive")
    delta = sum(1 for k in key_fields if before_fp.get(k) != after_fp.get(k))
    if delta == 0:
        return "rephrasing"
    return "local_modification" if delta <= 1 else "genuine_new_method"
