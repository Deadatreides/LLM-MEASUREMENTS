"""task_generator_f2.py — COPY (not import) of `agent_delta0_new_grid/
src/task_generator.py`, byte-identical logic, for FORK-1 Path A's reuse
of DELTA-0's F2 TEST set (PROTOCOL.md §1.1: isolation rule is
copy-code/read-data, never import). `build_tasks()` with the same
SEED_T=20260823 deterministically reproduces DELTA-0's exact 40 test
tasks -- confirmed against `agent_delta0_new_grid/metrics/tasks_
manifest.json`'s own task_id list at K1.
"""

from __future__ import annotations

import random

SEED_T = 20260823
N_TRAIN = 80
N_TEST = 40
K = 18
STEP_KINDS = ("FILTER", "AGGREGATE", "DERIVE")

_COMPANIES = (
    "Meridian Logistics", "Northgate Traders", "Silvercrest Partners",
    "Bluewave Holdings", "Oakfield Ventures", "Redstone Capital",
    "Ironpeak Supply", "Amberline Group", "Cobalt & Finch",
    "Harborview Corp", "Fernwood Industries", "Granite Ridge LLC",
    "Solstice Trading", "Windmere Associates", "Cinderpath Co",
    "Larkspur Freight", "Thornfield Retail", "Bayshore Mercantile",
)
_CATEGORIES = ("shipping-fees", "office-supplies", "consulting", "software-licensing",
              "equipment-lease", "travel-expense")
_REGIONS = ("EU-West", "EU-East", "NA-Central", "APAC", "LATAM")
_MONTHS = ("January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December")

RUSSIAN_OP_NAME = {"SUM": "сумму", "COUNT": "количество", "MAX": "максимум"}
RUSSIAN_COMPARATOR = {">=": "не меньше", "<": "меньше"}


def _make_record(idx: int, rng: random.Random, category: str, region: str) -> dict:
    return {
        "id": f"TXN-{idx:04d}",
        "account": rng.choice(_COMPANIES),
        "category": category,
        "amount": round(rng.uniform(15.0, 480.0), 2),
        "region": region,
        "date": f"2026-{rng.randrange(12) + 1:02d}-{rng.randint(1, 28):02d}",
    }


def _render_db_text(records: list) -> str:
    lines = []
    for r in records:
        lines.append(f"{r['id']} | account: {r['account']} | category: {r['category']} | "
                     f"amount: {r['amount']:.2f} | region: {r['region']} | date: {r['date']}")
    return "\n".join(lines)


def _compute_aggregate(records: list, op: str) -> float:
    amounts = [r["amount"] for r in records]
    if op == "SUM":
        return round(sum(amounts), 2)
    if op == "COUNT":
        return float(len(amounts))
    if op == "MAX":
        return round(max(amounts), 2) if amounts else 0.0
    raise ValueError(op)


def evaluate_comparator(aggregate_value: float, threshold: float, comparator: str) -> bool:
    if comparator == ">=":
        return aggregate_value >= threshold
    if comparator == "<":
        return aggregate_value < threshold
    raise ValueError(comparator)


def build_task(idx: int, rng: random.Random) -> dict:
    task_id = f"D0_{idx:04d}"
    target_category = rng.choice(_CATEGORIES)
    target_region = rng.choice(_REGIONS)
    n_true_match = rng.randint(2, 5)

    records = []
    ridx = 1
    for _ in range(n_true_match):
        records.append(_make_record(ridx, rng, target_category, target_region)); ridx += 1

    n_cat_only = rng.randint(2, 4)
    for _ in range(n_cat_only):
        other_region = rng.choice([r for r in _REGIONS if r != target_region])
        records.append(_make_record(ridx, rng, target_category, other_region)); ridx += 1

    n_region_only = rng.randint(2, 4)
    for _ in range(n_region_only):
        other_category = rng.choice([c for c in _CATEGORIES if c != target_category])
        records.append(_make_record(ridx, rng, other_category, target_region)); ridx += 1

    while len(records) < K:
        other_category = rng.choice([c for c in _CATEGORIES if c != target_category])
        other_region = rng.choice([r for r in _REGIONS if r != target_region])
        records.append(_make_record(ridx, rng, other_category, other_region)); ridx += 1
    records = records[:K]

    rng.shuffle(records)

    matched = [r for r in records if r["category"] == target_category and r["region"] == target_region]
    matched_ids = sorted(r["id"] for r in matched)
    assert 2 <= len(matched_ids) <= 5, f"{task_id}: n_true_match invariant broken ({len(matched_ids)})"

    op = rng.choice(("SUM", "COUNT", "MAX"))
    aggregate_value = _compute_aggregate(matched, op)

    want_yes = rng.random() < 0.5
    comparator = rng.choice((">=", "<"))
    if op == "COUNT":
        offset = rng.choice((1, 2))
    else:
        offset = round(rng.uniform(20.0, 150.0), 2)

    if comparator == ">=":
        threshold = aggregate_value - offset if want_yes else aggregate_value + offset
    else:
        threshold = aggregate_value + offset if want_yes else aggregate_value - offset
    threshold = round(threshold, 2) if op != "COUNT" else float(int(round(threshold)))

    golden_final = evaluate_comparator(aggregate_value, threshold, comparator)
    assert golden_final == want_yes, f"{task_id}: threshold construction did not hit want_yes"

    db_text = _render_db_text(records)
    query_text = (
        f"Найди все транзакции, у которых category = \"{target_category}\" И region = "
        f"\"{target_region}\". Посчитай {RUSSIAN_OP_NAME[op]} их поля amount. "
        f"Верно ли, что это значение {RUSSIAN_COMPARATOR[comparator]} {threshold:g}? "
        "Ответь ровно одним словом: ДА или НЕТ."
    )

    return {
        "task_id": task_id, "records": records, "db_text": db_text,
        "target_category": target_category, "target_region": target_region,
        "op": op, "threshold": threshold, "comparator": comparator,
        "query_text": query_text,
        "matched_ids": matched_ids, "aggregate_value": aggregate_value,
        "final_oracle": "ДА" if golden_final else "НЕТ",
        "id_to_amount": {r["id"]: r["amount"] for r in records},
    }


def build_tasks(n_train: int = N_TRAIN, n_test: int = N_TEST, seed: int = SEED_T) -> dict:
    rng = random.Random(seed)
    total = n_train + n_test
    tasks = {}
    for i in range(total):
        t = build_task(i, rng)
        tasks[t["task_id"]] = t
    ids = sorted(tasks)
    return {"train": ids[:n_train], "test": ids[n_train:total], "tasks": tasks}
