"""task_generator_m2.py — M1 with its three measured design defects fixed.
See PROTOCOL_M2.md. Kept separate from task_generator_m1.py so M1 stays
reproducible.

FIX 1 -- seam 1 is now VIOLABLE. Only N_MGR of N_DEPT departments have a
        manager (the max-budget one always does). A wrong hop-1 answer now
        has a real chance of naming a manager-less department, which МДП
        can detect. In M1 every department had a manager, so referential
        integrity never discriminated and МДП was null at seam 1 by
        construction -- my defect, not a property of МДП.

FIX 2 -- the final answer is an exact SET of employee ids, never a sum.
        In M1 the whole-task arm had to emit an exact multi-operand sum,
        which FORK-1 had already measured at 0/240 for these models, so
        WHOLE was floored at 0 for arithmetic reasons unrelated to
        multi-hop reasoning.

FIX 3 -- per-hop accuracy is the PRIMARY metric, not a diagnostic. The
        3-hop conjunction leaves the end-to-end number without resolution.
"""

from __future__ import annotations

import random

SEED_M2 = 20260826
N_TRAIN = 40
N_TEST = 40
N_DEPT = 8
N_MGR = 5           # < N_DEPT  ->  seam 1 is violable
N_EMP = 14

_DEPT_NAMES = ("Logistics", "Procurement", "Analytics", "Facilities", "Compliance",
              "Marketing", "Engineering", "Support", "Legal", "Research")
_PERSON = ("Larkin", "Vance", "Ostrom", "Beaulieu", "Kessler", "Ramirez", "Thorne",
          "Nakamura", "Delacroix", "Whitfield", "Bergstrom", "Ivanov", "Achebe",
          "Lindqvist", "Moreau", "Okafor")
_REGIONS = ("EU-West", "EU-East", "NA-Central", "APAC", "LATAM")


def build_task(idx: int, rng: random.Random) -> dict:
    task_id = f"M2_{idx:04d}"
    names = rng.sample(_DEPT_NAMES, N_DEPT)
    budgets = rng.sample(range(120, 990, 7), N_DEPT)           # distinct -> single max
    depts = [{"id": f"DEP-{i+1:04d}", "name": names[i], "budget": budgets[i],
              "region": rng.choice(_REGIONS)} for i in range(N_DEPT)]
    top = max(depts, key=lambda d: d["budget"])

    # the max-budget dept always has a manager; the rest are sampled, so some
    # departments have NONE -> naming one of those is a detectable seam breach
    others = [d for d in depts if d["id"] != top["id"]]
    rng.shuffle(others)
    managed = [top] + others[:N_MGR - 1]

    people = rng.sample(_PERSON, min(len(_PERSON), N_MGR + N_EMP))
    mgr_names, emp_names = people[:N_MGR], people[N_MGR:]
    mgrs = [{"id": f"MGR-{i+1:04d}", "name": mgr_names[i], "dept": managed[i]["id"],
             "years": rng.randint(1, 19)} for i in range(N_MGR)]
    target_mgr = next(m for m in mgrs if m["dept"] == top["id"])

    n_target = rng.randint(2, 4)
    emps = []
    for i in range(N_EMP):
        mgr = target_mgr["id"] if i < n_target else rng.choice(
            [m["id"] for m in mgrs if m["id"] != target_mgr["id"]])
        emps.append({"id": f"EMP-{i+1:04d}",
                     "name": emp_names[i % len(emp_names)] if emp_names else f"P{i}",
                     "mgr": mgr, "salary": round(rng.uniform(2100.0, 9400.0), 2)})
    rng.shuffle(emps); rng.shuffle(mgrs); rng.shuffle(depts)

    matched = sorted(e["id"] for e in emps if e["mgr"] == target_mgr["id"])
    assert 2 <= len(matched) <= 4, f"{task_id}: employee-count invariant broken"
    unmanaged = [d["id"] for d in depts if not any(m["dept"] == d["id"] for m in mgrs)]
    assert unmanaged, f"{task_id}: seam 1 must stay violable"

    block_a = "\n".join(f"{d['id']} | name: {d['name']} | budget: {d['budget']} | "
                        f"region: {d['region']}" for d in depts)
    block_b = "\n".join(f"{m['id']} | name: {m['name']} | dept: {m['dept']} | "
                        f"years: {m['years']}" for m in mgrs)
    block_c = "\n".join(f"{e['id']} | name: {e['name']} | mgr: {e['mgr']} | "
                        f"salary: {e['salary']:.2f}" for e in emps)

    return {
        "task_id": task_id, "depts": depts, "mgrs": mgrs, "emps": emps,
        "block_a": block_a, "block_b": block_b, "block_c": block_c,
        "db_text": f"ОТДЕЛЫ:\n{block_a}\n\nМЕНЕДЖЕРЫ:\n{block_b}\n\nСОТРУДНИКИ:\n{block_c}",
        "hop1_oracle": top["id"], "hop2_oracle": target_mgr["id"],
        "final_oracle": matched,              # a SET of ids -- no arithmetic anywhere
        "unmanaged_depts": unmanaged,
        "id_to_salary": {e["id"]: e["salary"] for e in emps},
    }


def build_tasks(n_train: int = N_TRAIN, n_test: int = N_TEST, seed: int = SEED_M2) -> dict:
    rng = random.Random(seed)
    tasks = {}
    for i in range(n_train + n_test):
        t = build_task(i, rng)
        tasks[t["task_id"]] = t
    ids = sorted(tasks)
    return {"train": ids[:n_train], "test": ids[n_train:n_train + n_test], "tasks": tasks}


_ASK = ("Найди отдел с наибольшим бюджетом. Найди менеджера этого отдела. "
        "Выведи ТОЛЬКО ID сотрудников этого менеджера через запятую, без пояснений.")


def prompt_whole(task: dict) -> str:
    return f"База:\n{task['db_text']}\n\n{_ASK}"


def prompt_hop1(task: dict) -> str:
    return (f"Таблица отделов:\n{task['block_a']}\n\n"
            "У какого отдела наибольший бюджет?\n"
            "Выведи ТОЛЬКО его ID вида DEP-XXXX, без пояснений.")


def prompt_hop2(task: dict, dep_id: str) -> str:
    return (f"Таблица менеджеров:\n{task['block_b']}\n\n"
            f"Кто менеджер отдела {dep_id}?\n"
            "Выведи ТОЛЬКО его ID вида MGR-XXXX. Если такого менеджера нет, выведи слово НЕТ.")


def prompt_hop3(task: dict, mgr_id: str) -> str:
    return (f"Таблица сотрудников:\n{task['block_c']}\n\n"
            f"Найди ID всех сотрудников, у которых mgr = {mgr_id}.\n"
            "Выведи ТОЛЬКО список подходящих ID через запятую, без пояснений.")


# -------- МДП: interface checks only; never redoes the neighbour's work --------

def mdp_seam1(task: dict, dep_id) -> bool:
    """Now discriminating: manager-less departments exist."""
    return dep_id is not None and any(m["dept"] == dep_id for m in task["mgrs"])


def mdp_seam2(task: dict, mgr_id) -> bool:
    return mgr_id is not None and any(e["mgr"] == mgr_id for e in task["emps"])


def mdp_seam2_consistency(task: dict, dep_id, mgr_id) -> bool:
    if mgr_id is None or dep_id is None:
        return False
    m = next((x for x in task["mgrs"] if x["id"] == mgr_id), None)
    return m is not None and m["dept"] == dep_id
