"""atoms_hard.py — FORK-1 Path B: T_hard's LLM FILTER prompt (B-LLM's
only LLM-touched atom, PROTOCOL.md §2.3 -- SUM is always deterministic,
never asked of an LLM anywhere in this package).
"""

from __future__ import annotations

MAX_TOKENS_FILTER = 220   # same allowance as DELTA-0's own FILTER atom (comparable K/response size)


def build_filter_prompt(task: dict) -> str:
    return (
        f"База записей:\n{task['db_text']}\n\n"
        f"Найди ID всех записей, у которых category = \"{task['target_category']}\" "
        f"И region = \"{task['target_region']}\".\n"
        "Выведи ТОЛЬКО список подходящих ID через запятую, без пояснений. "
        "Если подходящих нет, выведи слово НЕТ."
    )
