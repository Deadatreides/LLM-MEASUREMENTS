"""atoms.py — DELTA-0 C8: atom specs (name/input schema/output schema/
parser/max tokens) + prompt builders. Exactly 3 types (PROTOCOL.md §2):

FILTER    -- DB-sized input (the only atom this large; the union chain's
             cost driver, matching whole's own natural cost).
AGGREGATE -- compact id->amount slice ONLY for the ids actually passed in
             (never the full DB again) -- kept cheap by construction.
DERIVE    -- tiny scalar input.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator as TG   # noqa: E402
import oracles as OR          # noqa: E402

MAX_TOKENS_FILTER = 220     # allows listing up to ~5-6 ids comfortably
MAX_TOKENS_AGGREGATE = 40
MAX_TOKENS_DERIVE = 20

ATOM_SPECS = {
    "FILTER": {
        "name": "FILTER",
        "input_schema": {"db_text": "str (full DB)", "category": "str", "region": "str"},
        "output_schema": {"ids": "comma-separated TXN-#### tokens, or НЕТ"},
        "checker": OR.check_filter, "max_tokens": MAX_TOKENS_FILTER,
    },
    "AGGREGATE": {
        "name": "AGGREGATE",
        "input_schema": {"id_amount_slice": "dict[id -> amount], only matched ids", "op": "SUM|COUNT|MAX"},
        "output_schema": {"value": "bare number"},
        "checker": OR.check_aggregate, "max_tokens": MAX_TOKENS_AGGREGATE,
    },
    "DERIVE": {
        "name": "DERIVE",
        "input_schema": {"aggregate_value": "float", "threshold": "float", "comparator": ">=|<"},
        "output_schema": {"answer": "ДА|НЕТ"},
        "checker": OR.check_derive, "max_tokens": MAX_TOKENS_DERIVE,
    },
}


def build_filter_prompt(task: dict) -> str:
    return (
        f"База транзакций:\n{task['db_text']}\n\n"
        f"Найди ID всех транзакций, у которых category = \"{task['target_category']}\" "
        f"И region = \"{task['target_region']}\".\n"
        "Выведи ТОЛЬКО список подходящих ID через запятую, без пояснений. "
        "Если подходящих нет, выведи слово НЕТ."
    )


def build_aggregate_prompt(id_amount: dict, op: str) -> str:
    lines = [f"{i} = {v:g}" for i, v in sorted(id_amount.items())]
    op_name = TG.RUSSIAN_OP_NAME[op]
    return (
        "Значения:\n" + "\n".join(lines) + "\n\n"
        f"Посчитай {op_name} этих значений. Выведи ТОЛЬКО число, без пояснений."
    )


def build_derive_prompt(aggregate_value: float, threshold: float, comparator: str) -> str:
    cmp_word = TG.RUSSIAN_COMPARATOR[comparator]
    return (
        f"Значение: {aggregate_value:g}. Порог: {threshold:g}.\n"
        f"Верно ли, что значение {cmp_word} порога? Ответь ровно одним словом: ДА или НЕТ."
    )
