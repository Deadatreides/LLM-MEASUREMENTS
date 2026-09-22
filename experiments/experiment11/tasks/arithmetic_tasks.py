"""arithmetic_tasks.py — свежие однократные (не декомпозированные)
арифметические задачи для эксперимента 11.

НЕ переиспользует arch1/tasks/branch_tasks.py (clean-room: без доступа к
Mycelium). Каждая задача — один вопрос, один числовой ответ, ответ
вычислен ПРОГРАММНО по собственной формуле (не вписан вручную), чтобы
исключить ручную арифметическую ошибку в самих контрольных данных.
"""

from __future__ import annotations

_RAW = [
    # (question, formula-as-python-expression)
    ("A store sells shirts at $18 each and pants at $32 each. If you buy 3 shirts and 2 pants, what is the total amount paid, in dollars?",
     "3 * 18 + 2 * 32"),
    ("Worker A is paid $16 per hour and works 6 hours. Worker B is paid $21 per hour and works 4 hours. What is the combined pay for both workers, in dollars?",
     "6 * 16 + 4 * 21"),
    ("A rectangular room is 5 meters wide and 7 meters long. What is its area in square meters?",
     "5 * 7"),
    ("A car travels 60 miles per hour for 3 hours, then 45 miles per hour for 2 hours. What is the total distance travelled, in miles?",
     "60 * 3 + 45 * 2"),
    ("A recipe needs 250 grams of flour per batch. If you make 4 batches, how many grams of flour are needed in total?",
     "250 * 4"),
    ("A shop had 340 apples. It sold 128 apples in the morning and 95 apples in the afternoon. How many apples are left?",
     "340 - 128 - 95"),
    ("A water tank fills at 12 liters per minute. How many liters are in the tank after 15 minutes?",
     "12 * 15"),
    ("A school has 8 classrooms with 24 students each and 3 classrooms with 18 students each. What is the total number of students?",
     "8 * 24 + 3 * 18"),
    ("An account holds $500 and earns 4% simple annual interest. How much interest, in dollars, is earned in one year?",
     "500 * 0.04"),
    ("A factory produces 145 units per day. How many units does it produce over 12 days?",
     "145 * 12"),
    ("A bus has 48 seats. If 3 buses are full and a 4th bus has only 27 passengers, how many passengers are there in total?",
     "3 * 48 + 27"),
    ("A gardener plants 6 rows of 14 tulips each, then removes 9 tulips that did not grow. How many tulips remain?",
     "6 * 14 - 9"),
    ("A book costs $13. A discount of $4 is applied per book. If 7 books are bought, what is the total cost, in dollars?",
     "7 * (13 - 4)"),
    ("A train travels 220 kilometers in 4 hours at a constant speed. How many kilometers does it travel in 7 hours at the same speed?",
     "220 / 4 * 7"),
    ("A warehouse ships 85 boxes at $3 shipping cost per box, and 60 boxes at $5 shipping cost per box. What is the total shipping cost, in dollars?",
     "85 * 3 + 60 * 5"),
    ("A concert sold 120 tickets at $45 each and 80 tickets at $30 each. What is the total revenue, in dollars?",
     "120 * 45 + 80 * 30"),
    ("A pool holds 8000 liters. A pump removes 250 liters per hour. How many liters remain after 9 hours?",
     "8000 - 250 * 9"),
    ("A bakery uses 2.5 kilograms of sugar per cake. How many kilograms of sugar are needed for 11 cakes?",
     "2.5 * 11"),
    ("Team A has 9 members earning a $60 bonus each. Team B has 6 members earning a $85 bonus each. What is the total bonus paid across both teams, in dollars?",
     "9 * 60 + 6 * 85"),
    ("A field is being fenced. Each of the 4 sides is 34 meters long, and fencing costs $6 per meter. What is the total fencing cost, in dollars?",
     "4 * 34 * 6"),
]


def _build() -> dict:
    tasks = {}
    for i, (question, formula) in enumerate(_RAW, start=1):
        task_id = f"ARITH_{i:02d}"
        answer = eval(formula, {"__builtins__": {}})  # noqa: S307 -- literal numeric formula, no user input
        tasks[task_id] = {"question": question, "answer": float(answer), "formula": formula}
    return tasks


TASKS = _build()
TASK_IDS = tuple(TASKS)


def generation_prompt(task: dict) -> str:
    return (
        f"{task['question']}\n\n"
        "Reply with EXACTLY ONE line containing EXACTLY ONE equals sign, and nothing else "
        "-- no explanation, no other lines.\n"
        "Format: <your answer> = <number>"
    )


def retry_prompt(task: dict, prior_artifact: str, evidence: str) -> str:
    """Общий шаблон retry-промпта для плеч A/B/D (одинаковый исходный
    контекст для всех -- п.1 задания: нельзя давать одному плечу более
    выгодный контекст, чем другому)."""
    return (
        f"{task['question']}\n\n"
        f"A previous attempt answered:\n{prior_artifact}\n\n"
        f"This was mechanically verified INCORRECT: {evidence}\n\n"
        "Provide a corrected, complete answer.\n"
        "Reply with EXACTLY ONE line containing EXACTLY ONE equals sign, and nothing else.\n"
        "Format: <your answer> = <number>"
    )


def retry_prompt_v2(task: dict, prior_artifact: str, evidence: str) -> str:
    """prompt_profile B (плечо CHANGE_PROMPT) -- та же информация,
    другая формулировка, зафиксирована ДО прогона."""
    return (
        f"Problem: {task['question']}\n\n"
        f"Someone already tried and got this: {prior_artifact}\n"
        f"That is wrong ({evidence}). Work through the problem step by step in your head, "
        "then give only the final numeric equation.\n"
        "Your entire reply must be one line, one equals sign.\n"
        "Format: <your answer> = <number>"
    )
