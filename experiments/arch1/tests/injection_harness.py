"""Инъекционный стенд для tests/test_repair.py.

Адаптация методики experiment9 (metrics/injection9.py, ground_truth9.py,
tasks/schema9.py): заморозить корректный граф -> инъецировать ровно одну
ошибку в известную точку -> сравнить предсказанную область
(RepairPlanner) с независимо построенной истиной.

МЕТОДОЛОГИЧЕСКИЙ ИНВАРИАНТ (обязателен к сохранению): true_affected_set()
обходит ЗАМОРОЖЕННЫЙ словарь FROZEN_DEPENDS_ON своим собственным кодом и
НИКОГДА не вызывает GraphCore.downstream_closure() -- иначе предсказание
сверялось бы само с собой. Это ровно дефект эксперимента 8, который
чинил эксперимент 9 (ground_truth9.py: "ground truth must NOT be computed
by the same code that produces the prediction").

Граф -- тот же 6-узловой ветвящийся DAG формы schema9.py:

    C1_ROOT
      -> C2_METHOD_A -> C4_SUBTOTAL_A -\
      -> C3_METHOD_B -> C5_SUBTOTAL_B -> C6_FINAL_TOTAL   (merge)
"""

from src.graph_core import GraphCore
from src.storage import Storage

TASK = {
    "keywordA": "shirt", "qtyA": 2, "rateA": 20, "subtotalA": 40,
    "keywordB": "pant", "qtyB": 3, "rateB": 35, "subtotalB": 105,
    "final_total": 145,
}

CLAIMS = ("C1_ROOT", "C2_METHOD_A", "C3_METHOD_B", "C4_SUBTOTAL_A", "C5_SUBTOTAL_B", "C6_FINAL_TOTAL")

CLAIM_TYPE = {
    "C1_ROOT": "OTHER",
    "C2_METHOD_A": "METHOD",
    "C3_METHOD_B": "METHOD",
    "C4_SUBTOTAL_A": "VALUE",
    "C5_SUBTOTAL_B": "VALUE",
    "C6_FINAL_TOTAL": "VALUE",
}

# FROZEN -- источник истины для true_affected_set(). Не переиспользовать
# для предсказания и не выводить из графа -- см. докстринг модуля.
FROZEN_DEPENDS_ON = {
    "C1_ROOT": [],
    "C2_METHOD_A": ["C1_ROOT"],
    "C3_METHOD_B": ["C1_ROOT"],
    "C4_SUBTOTAL_A": ["C2_METHOD_A"],
    "C5_SUBTOTAL_B": ["C3_METHOD_B"],
    "C6_FINAL_TOTAL": ["C4_SUBTOTAL_A", "C5_SUBTOTAL_B"],
}

_EDGES = (
    ("C1_ROOT", "C2_METHOD_A", "CAUSAL"),
    ("C1_ROOT", "C3_METHOD_B", "CAUSAL"),
    ("C2_METHOD_A", "C4_SUBTOTAL_A", "DERIVATION"),
    ("C3_METHOD_B", "C5_SUBTOTAL_B", "DERIVATION"),
    ("C4_SUBTOTAL_A", "C6_FINAL_TOTAL", "COMPUTATIONAL"),
    ("C5_SUBTOTAL_B", "C6_FINAL_TOTAL", "COMPUTATIONAL"),
)


def reference_correct_content(task: dict = TASK) -> dict:
    return {
        "C1_ROOT": f"This problem has two parts that must be handled separately: {task['keywordA']} and {task['keywordB']}, then combined into one total.",
        "C2_METHOD_A": f"For {task['keywordA']}: multiply the quantity ({task['qtyA']}) by the rate ({task['rateA']}).",
        "C3_METHOD_B": f"For {task['keywordB']}: multiply the quantity ({task['qtyB']}) by the rate ({task['rateB']}).",
        "C4_SUBTOTAL_A": f"{task['qtyA']} * {task['rateA']} = {task['subtotalA']}",
        "C5_SUBTOTAL_B": f"{task['qtyB']} * {task['rateB']} = {task['subtotalB']}",
        "C6_FINAL_TOTAL": f"{task['subtotalA']} + {task['subtotalB']} = {task['final_total']}. Final total: {task['final_total']}",
    }


def build_correct_graph(task: dict = TASK) -> tuple[Storage, GraphCore]:
    """Замораживает корректный граф ДО инъекции."""
    storage = Storage()
    graph = GraphCore(storage)

    artifact_version_id = graph.add_artifact_version(
        artifact_type="DERIVATION", task_id="task:injection", content="branching word problem"
    )
    artifact_id = storage.get(artifact_version_id).payload.artifact_id

    content = reference_correct_content(task)
    for claim_id in CLAIMS:
        graph.add_claim_version(
            artifact_id=artifact_id,
            content=content[claim_id],
            claim_type=CLAIM_TYPE[claim_id],
            claim_id=claim_id,
        )

    for source, target, dependency_type in _EDGES:
        graph.add_dependency(
            source_claim=source,
            target_claim=target,
            dependency_type=dependency_type,
            status="CONFIRMED",
            created_by="HUMAN",
        )

    return storage, graph


def _wrong_number(correct: float, bump: float = 1.37) -> float:
    """Детерминированное, воспроизводимое неверное значение (по образцу
    experiment9/metrics/injection9.py: всегда доказуемо отличается от
    correct, даже после округления)."""
    wrong = round(correct * bump + 3, 4)
    if abs(wrong - correct) < 0.01:
        wrong += 5
    return wrong


def inject(graph: GraphCore, claim_id: str, task: dict = TASK) -> None:
    """Инъецирует РОВНО ОДНУ ошибку: новая версия claim_id с испорченным
    содержимым; все остальные claims остаются байт-в-байт как были."""
    artifact_id = graph.get_claim(claim_id).artifact_id

    if claim_id == "C1_ROOT":
        corrupted = f"This problem is about {task['keywordA']} only; compute the total cost for that."
    elif claim_id == "C2_METHOD_A":
        wrong_rate = _wrong_number(task["rateA"])
        corrupted = f"For {task['keywordA']}: multiply the quantity ({task['qtyA']}) by the rate ({wrong_rate})."
    elif claim_id == "C3_METHOD_B":
        wrong_rate = _wrong_number(task["rateB"])
        corrupted = f"For {task['keywordB']}: multiply the quantity ({task['qtyB']}) by the rate ({wrong_rate})."
    elif claim_id == "C4_SUBTOTAL_A":
        wrong_sub = _wrong_number(task["subtotalA"])
        corrupted = f"{task['qtyA']} * {task['rateA']} = {wrong_sub}"
    elif claim_id == "C5_SUBTOTAL_B":
        wrong_sub = _wrong_number(task["subtotalB"])
        corrupted = f"{task['qtyB']} * {task['rateB']} = {wrong_sub}"
    elif claim_id == "C6_FINAL_TOTAL":
        wrong_total = _wrong_number(task["final_total"])
        corrupted = f"{task['subtotalA']} + {task['subtotalB']} = {wrong_total}. Final total: {wrong_total}"
    else:
        raise ValueError(claim_id)

    graph.add_claim_version(
        artifact_id=artifact_id,
        content=corrupted,
        claim_type=CLAIM_TYPE[claim_id],
        claim_id=claim_id,
    )


def _downstream_of(claim_id: str, depends_on: dict = FROZEN_DEPENDS_ON) -> set[str]:
    reverse: dict[str, list[str]] = {}
    for claim, deps in depends_on.items():
        for dep in deps:
            reverse.setdefault(dep, []).append(claim)
    seen: set[str] = set()
    frontier = [claim_id]
    while frontier:
        cur = frontier.pop()
        for nxt in reverse.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def true_affected_set(injected_claim_ids) -> set[str]:
    """Истинная область влияния — независимый обход по FROZEN_DEPENDS_ON.
    НЕ вызывает GraphCore.downstream_closure() (см. докстринг модуля)."""
    affected = set(injected_claim_ids)
    for claim_id in injected_claim_ids:
        affected |= _downstream_of(claim_id)
    return affected
