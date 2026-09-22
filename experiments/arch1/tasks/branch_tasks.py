"""branch_tasks.py — 20 задач + структура графа (см. tasks/__init__.py).

TASKS/CLAIMS/DEPENDENCY_EDGES — перенесены буквально из
experiment9/tasks/schema9.py (тот же 6-узловой ветвящийся DAG,
section 1's minimal example):

    C1_ROOT (общая рамка: обе категории)
      -> C2_METHOD_A -> C4_SUBTOTAL_A -\
      -> C3_METHOD_B -> C5_SUBTOTAL_B -> C6_FINAL_TOTAL   (merge)

generation_prompt()/oracle_values() — новое: связывают эти данные с
ContextBuilder/ActionExecutor (этап 5) и numeric_seam (этап 3).
"""

from __future__ import annotations

from typing import Optional

TASKS = {
    "BRANCH_01": {"keywordA": "shirt", "keywordB": "pant", "qtyA": 2, "rateA": 20, "subtotalA": 40, "qtyB": 3, "rateB": 35, "subtotalB": 105, "final_total": 145, "question": "A store sells shirts at $20 each and pants at $35 each. If you buy 2 shirts and 3 pants, what is the total amount paid?"},
    "BRANCH_02": {"keywordA": "worker a", "keywordB": "worker b", "qtyA": 5, "rateA": 18, "subtotalA": 90, "qtyB": 7, "rateB": 15, "subtotalB": 105, "final_total": 195, "question": "Worker A is paid $18 per hour and works 5 hours. Worker B is paid $15 per hour and works 7 hours. What is the combined pay for both workers?"},
    "BRANCH_03": {"keywordA": "account a", "keywordB": "account b", "qtyA": 300, "rateA": 0.05, "subtotalA": 15.0, "qtyB": 450, "rateB": 0.03, "subtotalB": 13.5, "final_total": 28.5, "question": "Account A holds $300 earning 5% simple annual interest. Account B holds $450 earning 3% simple annual interest. What is the combined interest earned by both accounts in one year?"},
    "BRANCH_04": {"keywordA": "leg a", "keywordB": "leg b", "qtyA": 40, "rateA": 0.175, "subtotalA": 7.0, "qtyB": 60, "rateB": 0.14, "subtotalB": 8.4, "final_total": 15.4, "question": "On leg A of a trip, a car uses fuel costing $0.175 per mile over 40 miles. On leg B, it uses fuel costing $0.14 per mile over 60 miles. What is the total fuel cost for the trip?"},
    "BRANCH_05": {"keywordA": "room a", "keywordB": "room b", "qtyA": 20, "rateA": 4, "subtotalA": 80, "qtyB": 18, "rateB": 4, "subtotalB": 72, "final_total": 152, "question": "Room A has an area of 20 square meters and room B has an area of 18 square meters. Flooring costs $4 per square meter for both rooms. What is the total flooring cost?"},
    "BRANCH_06": {"keywordA": "apple", "keywordB": "orange", "qtyA": 3, "rateA": 2.5, "subtotalA": 7.5, "qtyB": 2, "rateB": 3, "subtotalB": 6, "final_total": 13.5, "question": "A shopper buys 3 kg of apples at $2.5 per kg and 2 kg of oranges at $3 per kg. What is the total cost?"},
    "BRANCH_07": {"keywordA": "recipe a", "keywordB": "recipe b", "qtyA": 2, "rateA": 250, "subtotalA": 500, "qtyB": 3, "rateB": 180, "subtotalB": 540, "final_total": 1040, "question": "Recipe A needs 2 batches of 250 grams of flour each. Recipe B needs 3 batches of 180 grams of flour each. What is the total flour needed in grams?"},
    "BRANCH_08": {"keywordA": "tank a", "keywordB": "tank b", "qtyA": 5, "rateA": 12, "subtotalA": 60, "qtyB": 9, "rateB": 8, "subtotalB": 72, "final_total": 132, "question": "Tank A fills at 12 liters per minute for 5 minutes. Tank B fills at 8 liters per minute for 9 minutes. What is the total water collected from both tanks?"},
    "BRANCH_09": {"keywordA": "salesperson a", "keywordB": "salesperson b", "qtyA": 2000, "rateA": 0.08, "subtotalA": 160.0, "qtyB": 3500, "rateB": 0.06, "subtotalB": 210.0, "final_total": 370.0, "question": "Salesperson A sold $2000 worth of goods earning an 8% commission. Salesperson B sold $3500 worth of goods earning a 6% commission. What is the total commission paid to both salespeople?"},
    "BRANCH_10": {"keywordA": "bookstore a", "keywordB": "bookstore b", "qtyA": 15, "rateA": 12, "subtotalA": 180, "qtyB": 22, "rateB": 9, "subtotalB": 198, "final_total": 378, "question": "Bookstore A ordered 15 books at $12 each. Bookstore B ordered 22 books at $9 each. What is the total cost of both orders?"},
    "BRANCH_11": {"keywordA": "vip", "keywordB": "general", "qtyA": 4, "rateA": 85, "subtotalA": 340, "qtyB": 12, "rateB": 40, "subtotalB": 480, "final_total": 820, "question": "A concert sold 4 VIP tickets at $85 each and 12 general tickets at $40 each. What is the total revenue from ticket sales?"},
    "BRANCH_12": {"keywordA": "wall a", "keywordB": "wall b", "qtyA": 20, "rateA": 0.3, "subtotalA": 6.0, "qtyB": 35, "rateB": 0.25, "subtotalB": 8.75, "final_total": 14.75, "question": "Wall A is 20 square meters and needs 0.3 liters of paint per square meter. Wall B is 35 square meters and needs 0.25 liters of paint per square meter. What is the total paint needed in liters?"},
    "BRANCH_13": {"keywordA": "fund a", "keywordB": "fund b", "qtyA": 1000, "rateA": 0.07, "subtotalA": 70.0, "qtyB": 1500, "rateB": 0.04, "subtotalB": 60.0, "final_total": 130.0, "question": "Fund A holds $1000 and grew by 7% this year. Fund B holds $1500 and grew by 4% this year. What is the combined growth amount (not including the original principal) from both funds?"},
    "BRANCH_14": {"keywordA": "cookie", "keywordB": "brownie", "qtyA": 40, "rateA": 1.5, "subtotalA": 60.0, "qtyB": 25, "rateB": 2.25, "subtotalB": 56.25, "final_total": 116.25, "question": "A bake sale sold 40 cookies at $1.5 each and 25 brownies at $2.25 each. What is the total revenue?"},
    "BRANCH_15": {"keywordA": "bed a", "keywordB": "bed b", "qtyA": 24, "rateA": 4, "subtotalA": 96, "qtyB": 20, "rateB": 4, "subtotalB": 80, "final_total": 176, "question": "Garden bed A covers 24 square meters and garden bed B covers 20 square meters. Soil costs $4 per square meter for both beds. What is the total soil cost?"},
    "BRANCH_16": {"keywordA": "department a", "keywordB": "department b", "qtyA": 12, "rateA": 50000, "subtotalA": 600000, "qtyB": 8, "rateB": 65000, "subtotalB": 520000, "final_total": 1120000, "question": "Department A has 12 employees earning an average of $50000 each per year. Department B has 8 employees earning an average of $65000 each per year. What is the combined total payroll for both departments?"},
    "BRANCH_17": {"keywordA": "car a", "keywordB": "car b", "qtyA": 500, "rateA": 0.12, "subtotalA": 60.0, "qtyB": 350, "rateB": 0.15, "subtotalB": 52.5, "final_total": 112.5, "question": "Car A travels 500 miles at a fuel cost of $0.12 per mile. Car B travels 350 miles at a fuel cost of $0.15 per mile. What is the total fuel cost for both cars?"},
    "BRANCH_18": {"keywordA": "morning", "keywordB": "afternoon", "qtyA": 45, "rateA": 30, "subtotalA": 1350, "qtyB": 60, "rateB": 25, "subtotalB": 1500, "final_total": 2850, "question": "A conference sold 45 morning-session passes at $30 each and 60 afternoon-session passes at $25 each. What is the total revenue?"},
    "BRANCH_19": {"keywordA": "warehouse a", "keywordB": "warehouse b", "qtyA": 120, "rateA": 3.2, "subtotalA": 384.0, "qtyB": 95, "rateB": 4.1, "subtotalB": 389.5, "final_total": 773.5, "question": "Warehouse A ships 120 boxes at a shipping cost of $3.2 per box. Warehouse B ships 95 boxes at a shipping cost of $4.1 per box. What is the total shipping cost?"},
    "BRANCH_20": {"keywordA": "team a", "keywordB": "team b", "qtyA": 6, "rateA": 450, "subtotalA": 2700, "qtyB": 9, "rateB": 300, "subtotalB": 2700, "final_total": 5400, "question": "Team A has 6 members each receiving a $450 bonus. Team B has 9 members each receiving a $300 bonus. What is the total bonus paid across both teams?"},
}

TASK_IDS = list(TASKS.keys())

CLAIMS = ("C1_ROOT", "C2_METHOD_A", "C3_METHOD_B", "C4_SUBTOTAL_A", "C5_SUBTOTAL_B", "C6_FINAL_TOTAL")

# claim_type -- какие встроенные швы (этап 3) вообще применимы:
#   OTHER/METHOD -- ни один (framing/метод текстом не проверяется механически)
#   VALUE        -- numeric_seam (есть числовой oracle из TASKS)
CLAIM_TYPE = {
    "C1_ROOT": "OTHER",
    "C2_METHOD_A": "METHOD",
    "C3_METHOD_B": "METHOD",
    "C4_SUBTOTAL_A": "VALUE",
    "C5_SUBTOTAL_B": "VALUE",
    "C6_FINAL_TOTAL": "VALUE",
}

DEPENDENCY_EDGES = (
    ("C1_ROOT", "C2_METHOD_A", "CAUSAL"),
    ("C1_ROOT", "C3_METHOD_B", "CAUSAL"),
    ("C2_METHOD_A", "C4_SUBTOTAL_A", "DERIVATION"),
    ("C3_METHOD_B", "C5_SUBTOTAL_B", "DERIVATION"),
    ("C4_SUBTOTAL_A", "C6_FINAL_TOTAL", "COMPUTATIONAL"),
    ("C5_SUBTOTAL_B", "C6_FINAL_TOTAL", "COMPUTATIONAL"),
)


def generation_prompt(claim_id: str, task: dict) -> str:
    """Инструкция для GENERATE этого claim'а (адаптация формулировок
    experiment9/configs/prompts9.py D_TEMPLATE — по одной секции на
    отдельный вызов, не на общий структурированный ответ).

    claim_id явно упомянут в тексте -- нужно тестам для детерминированного
    FakeModelAdapter (см. tests/test_orchestrator.py).
    """
    question = task["question"]
    if claim_id == "C1_ROOT":
        return (
            f"{question}\n\n[{claim_id}] State that this problem has two parts that must be "
            f"handled separately, naming both: {task['keywordA']} and {task['keywordB']}. "
            "Do not compute anything yet."
        )
    if claim_id == "C2_METHOD_A":
        return (
            f"{question}\n\n[{claim_id}] State the quantity and rate that apply to "
            f"{task['keywordA']}. Do not compute the subtotal yet."
        )
    if claim_id == "C3_METHOD_B":
        return (
            f"{question}\n\n[{claim_id}] State the quantity and rate that apply to "
            f"{task['keywordB']}. Do not compute the subtotal yet."
        )
    # VALUE-claims: контракт вывода объявлен ЯВНО (ровно одно равенство).
    # До этапа 8 он был сформулирован расплывчато («покажи арифметику и
    # укажи результат»), и модели штатно выдавали решение всей задачи
    # внутри одного claim'а — не нарушая инструкцию, а не имея её.
    # DATA_MODEL §14: output_contract — часть prompt_profile, а не
    # пожелание. Изменено между кампанией 1 и 2 (см. STATUS.md этапа 8).
    single_equation_contract = (
        "Reply with EXACTLY ONE line containing EXACTLY ONE equals sign, "
        "and nothing else — no explanation, no other lines, no other calculations."
    )
    if claim_id == "C4_SUBTOTAL_A":
        return (
            f"{question}\n\n[{claim_id}] Compute ONLY {task['keywordA']}'s subtotal "
            f"(quantity {task['qtyA']} times rate {task['rateA']}). "
            f"Do NOT compute {task['keywordB']} and do NOT compute the final total.\n"
            f"{single_equation_contract}\n"
            f"Format: {task['qtyA']} * {task['rateA']} = <result>"
        )
    if claim_id == "C5_SUBTOTAL_B":
        return (
            f"{question}\n\n[{claim_id}] Compute ONLY {task['keywordB']}'s subtotal "
            f"(quantity {task['qtyB']} times rate {task['rateB']}). "
            f"Do NOT compute {task['keywordA']} and do NOT compute the final total.\n"
            f"{single_equation_contract}\n"
            f"Format: {task['qtyB']} * {task['rateB']} = <result>"
        )
    if claim_id == "C6_FINAL_TOTAL":
        return (
            f"{question}\n\n[{claim_id}] Add the two subtotals given above and state the final total. "
            "Do NOT recompute the subtotals.\n"
            f"{single_equation_contract}\n"
            "Format: <subtotal_a> + <subtotal_b> = <result>"
        )
    raise ValueError(claim_id)


def generation_prompt_v2(claim_id: str, task: dict) -> str:
    """prompt_profile B — альтернативная формулировка ТЕХ ЖЕ claim'ов
    (эксперимент 10, плечо CHANGE_PROMPT).

    Написана ДО прогона и не подстраивается по результатам (SEAM_MODEL.md
    §5.3 — запрет подстройки под результат распространяется и на
    конфигурацию генерации, раз она является предметом измерения).

    Отличие от v1 содержательное, а не косметическое: v1 задаёт результат
    напрямую («Format: 2 * 20 = <result>»), v2 просит сначала назвать
    множители, затем произведение — то есть меняет порядок изложения, а
    не только слова. Ожидаемая сила рычага по ACTION_MODEL §106 —
    промежуточная (F2 STRONG для повторяемости, F7 NEGATIVE для
    независимости claim-уровня; спека прямо говорит, что обобщать нельзя).
    Именно поэтому это плечо измеряется, а не предполагается.
    """
    question = task["question"]
    if claim_id == "C1_ROOT":
        return (
            f"{question}\n\n[{claim_id}] Before any arithmetic: how many separate parts does "
            f"this problem have, and what is each about? Answer in one sentence."
        )
    if claim_id == "C2_METHOD_A":
        return (
            f"{question}\n\n[{claim_id}] Which two numbers must be multiplied together for "
            f"{task['keywordA']}? Name them. Do not multiply them yet."
        )
    if claim_id == "C3_METHOD_B":
        return (
            f"{question}\n\n[{claim_id}] Which two numbers must be multiplied together for "
            f"{task['keywordB']}? Name them. Do not multiply them yet."
        )

    one_line = (
        "Answer with a single line and nothing else. The line must contain one '=' sign."
    )
    if claim_id == "C4_SUBTOTAL_A":
        return (
            f"{question}\n\n[{claim_id}] The two numbers for {task['keywordA']} are "
            f"{task['qtyA']} and {task['rateA']}. What is their product?\n"
            f"{one_line}\n"
            f"Write it as: {task['qtyA']} * {task['rateA']} = <product>"
        )
    if claim_id == "C5_SUBTOTAL_B":
        return (
            f"{question}\n\n[{claim_id}] The two numbers for {task['keywordB']} are "
            f"{task['qtyB']} and {task['rateB']}. What is their product?\n"
            f"{one_line}\n"
            f"Write it as: {task['qtyB']} * {task['rateB']} = <product>"
        )
    if claim_id == "C6_FINAL_TOTAL":
        return (
            f"{question}\n\n[{claim_id}] Take the two subtotals shown above and add them. "
            "Use their values exactly as given; do not recompute them.\n"
            f"{one_line}\n"
            "Write it as: <first> + <second> = <sum>"
        )
    raise ValueError(claim_id)


PROMPT_PROFILES = {"v1": generation_prompt, "v2": generation_prompt_v2}


def oracle_values(task: dict) -> dict:
    """Механический oracle для VERIFY (numeric_seam) -- только claims
    типа VALUE (C4/C5/C6). C1-C3 остаются честно UNVERIFIED (решение
    этапа 6, см. STATUS.md)."""
    return {
        "C4_SUBTOTAL_A": task["subtotalA"],
        "C5_SUBTOTAL_B": task["subtotalB"],
        "C6_FINAL_TOTAL": task["final_total"],
    }


def get_task(task_id: str, tasks: Optional[dict] = None) -> dict:
    return (tasks or TASKS)[task_id]
