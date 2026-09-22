"""atoms_hard.py — COPY (not import) of `agent_fork1_three_paths/src/
atoms_hard.py`, byte-identical logic. T_hard's LLM FILTER prompt.
"""

from __future__ import annotations

MAX_TOKENS_FILTER = 220


def build_filter_prompt(task: dict) -> str:
    return (
        f"База записей:\n{task['db_text']}\n\n"
        f"Найди ID всех записей, у которых category = \"{task['target_category']}\" "
        f"И region = \"{task['target_region']}\".\n"
        "Выведи ТОЛЬКО список подходящих ID через запятую, без пояснений. "
        "Если подходящих нет, выведи слово НЕТ."
    )
