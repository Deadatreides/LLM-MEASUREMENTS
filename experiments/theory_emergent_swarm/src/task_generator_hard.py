"""task_generator_hard.py — COPY (not import) of `agent_fork1_three_paths/
src/task_generator_hard.py`, byte-identical logic. E2 reuses FORK-1's
T_hard TEST set (same SEED_B=20260824 -> deterministically identical 40
test tasks). T_hard = EXTRACT-SUM: final answer is the exact SUM of
matched `amount` values, a bare float, tolerance 0.01, never a threshold
comparison. K_HARD=14 records, id prefix REC-####.
"""

from __future__ import annotations

import random

SEED_B = 20260824
N_TRAIN = 40
N_TEST = 40
K_HARD = 14

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


def _make_record(idx: int, rng: random.Random, category: str, region: str) -> dict:
    return {
        "id": f"REC-{idx:04d}",
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


def build_task(idx: int, rng: random.Random) -> dict:
    task_id = f"H0_{idx:04d}"
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

    while len(records) < K_HARD:
        other_category = rng.choice([c for c in _CATEGORIES if c != target_category])
        other_region = rng.choice([r for r in _REGIONS if r != target_region])
        records.append(_make_record(ridx, rng, other_category, other_region)); ridx += 1
    records = records[:K_HARD]

    rng.shuffle(records)

    matched = [r for r in records if r["category"] == target_category and r["region"] == target_region]
    matched_ids = sorted(r["id"] for r in matched)
    assert 2 <= len(matched_ids) <= 5, f"{task_id}: n_true_match invariant broken ({len(matched_ids)})"

    sum_value = round(sum(r["amount"] for r in matched), 2)

    db_text = _render_db_text(records)
    query_text = (
        f"Найди все записи, у которых category = \"{target_category}\" И region = "
        f"\"{target_region}\". Выведи ТОЛЬКО точную сумму их поля amount. "
        "Выведи одно число, без пояснений и без имени поля."
    )

    return {
        "task_id": task_id, "records": records, "db_text": db_text,
        "target_category": target_category, "target_region": target_region,
        "query_text": query_text,
        "matched_ids": matched_ids, "sum_value": sum_value,
        "final_oracle": sum_value,
        "id_to_amount": {r["id"]: r["amount"] for r in records},
    }


def build_tasks(n_train: int = N_TRAIN, n_test: int = N_TEST, seed: int = SEED_B) -> dict:
    rng = random.Random(seed)
    total = n_train + n_test
    tasks = {}
    for i in range(total):
        t = build_task(i, rng)
        tasks[t["task_id"]] = t
    ids = sorted(tasks)
    return {"train": ids[:n_train], "test": ids[n_train:total], "tasks": tasks}
