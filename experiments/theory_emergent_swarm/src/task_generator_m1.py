"""task_generator_m1.py — M1: multi-hop over THREE tables with genuine
inter-block dependencies. Built specifically so that МДП (cross-discipline
check) is NOT degenerate: the output of one discipline is the INPUT of the
next, so a seam can actually be violated and detected.

  Block A  departments  DEP-#### | name | budget | region
  Block B  managers     MGR-#### | name | dept=DEP-#### | years
  Block C  employees    EMP-#### | name | mgr=MGR-#### | salary

  hop 1 (A): department with the largest budget          -> dep_id
  hop 2 (B): the manager whose dept == dep_id            -> mgr_id
  hop 3 (C): sum of salaries of employees with that mgr  -> answer

Seams: hop1 -> hop2 (dep_id must exist in B), hop2 -> hop3 (mgr_id must
exist in C). Both are referential-integrity constraints checkable
deterministically WITHOUT knowing the right answer -- exactly Kahn's МДП:
you never redo the neighbour's work, you check the interface.

Invariants enforced by construction:
  - exactly ONE department has the max budget (no ties)
  - exactly ONE manager per department (bijective on the target)
  - the target manager has 2-4 employees; other managers also have some
"""

from __future__ import annotations

import random

SEED_M = 20260825
N_TRAIN = 40
N_TEST = 40
N_DEPT = 8
N_MGR = 8
N_EMP = 14

_DEPT_NAMES = ("Logistics", "Procurement", "Analytics", "Facilities", "Compliance",
              "Marketing", "Engineering", "Support", "Legal", "Research")
_PERSON = ("Larkin", "Vance", "Ostrom", "Beaulieu", "Kessler", "Ramirez", "Thorne",
          "Nakamura", "Delacroix", "Whitfield", "Bergstrom", "Ivanov", "Achebe",
          "Lindqvist", "Moreau", "Okafor")
_REGIONS = ("EU-West", "EU-East", "NA-Central", "APAC", "LATAM")


def build_task(idx: int, rng: random.Random) -> dict:
    task_id = f"M1_{idx:04d}"

    dept_names = rng.sample(_DEPT_NAMES, N_DEPT)
    budgets = rng.sample(range(120, 990, 7), N_DEPT)          # distinct -> no tie
    depts = [{"id": f"DEP-{i+1:04d}", "name": dept_names[i], "budget": budgets[i],
              "region": rng.choice(_REGIONS)} for i in range(N_DEPT)]
    top = max(depts, key=lambda d: d["budget"])

    people = rng.sample(_PERSON, min(len(_PERSON), N_MGR + N_EMP))
    mgr_names = people[:N_MGR]
    emp_names = people[N_MGR:]
    order = list(range(N_DEPT))
    rng.shuffle(order)
    mgrs = [{"id": f"MGR-{i+1:04d}", "name": mgr_names[i], "dept": depts[order[i]]["id"],
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
    rng.shuffle(emps)
    rng.shuffle(mgrs)
    rng.shuffle(depts)

    matched = [e for e in emps if e["mgr"] == target_mgr["id"]]
    assert 2 <= len(matched) <= 4, f"{task_id}: employee-count invariant broken"
    answer = round(sum(e["salary"] for e in matched), 2)

    block_a = "\n".join(f"{d['id']} | name: {d['name']} | budget: {d['budget']} | "
                        f"region: {d['region']}" for d in depts)
    block_b = "\n".join(f"{m['id']} | name: {m['name']} | dept: {m['dept']} | "
                        f"years: {m['years']}" for m in mgrs)
    block_c = "\n".join(f"{e['id']} | name: {e['name']} | mgr: {e['mgr']} | "
                        f"salary: {e['salary']:.2f}" for e in emps)

    query = ("Найди отдел с наибольшим бюджетом. Найди менеджера этого отдела. "
             "Посчитай сумму зарплат сотрудников этого менеджера. "
             "Выведи одно число, без пояснений.")

    return {
        "task_id": task_id, "depts": depts, "mgrs": mgrs, "emps": emps,
        "block_a": block_a, "block_b": block_b, "block_c": block_c,
        "db_text": f"ОТДЕЛЫ:\n{block_a}\n\nМЕНЕДЖЕРЫ:\n{block_b}\n\nСОТРУДНИКИ:\n{block_c}",
        "query_text": query,
        "hop1_oracle": top["id"], "hop2_oracle": target_mgr["id"],
        "hop3_oracle": sorted(e["id"] for e in matched),
        "final_oracle": answer,
        "id_to_salary": {e["id"]: e["salary"] for e in emps},
    }


def build_tasks(n_train: int = N_TRAIN, n_test: int = N_TEST, seed: int = SEED_M) -> dict:
    rng = random.Random(seed)
    tasks = {}
    for i in range(n_train + n_test):
        t = build_task(i, rng)
        tasks[t["task_id"]] = t
    ids = sorted(tasks)
    return {"train": ids[:n_train], "test": ids[n_train:n_train + n_test], "tasks": tasks}


# ---------------- discipline prompts (each strictly easier than the whole) -------------

def prompt_hop1(task: dict) -> str:
    return (f"Таблица отделов:\n{task['block_a']}\n\n"
            "У какого отдела наибольший бюджет?\n"
            "Выведи ТОЛЬКО его ID вида DEP-XXXX, без пояснений.")


def prompt_hop2(task: dict, dep_id: str) -> str:
    return (f"Таблица менеджеров:\n{task['block_b']}\n\n"
            f"Кто менеджер отдела {dep_id}?\n"
            "Выведи ТОЛЬКО его ID вида MGR-XXXX, без пояснений.")


def prompt_hop3(task: dict, mgr_id: str) -> str:
    return (f"Таблица сотрудников:\n{task['block_c']}\n\n"
            f"Найди ID всех сотрудников, у которых mgr = {mgr_id}.\n"
            "Выведи ТОЛЬКО список подходящих ID через запятую, без пояснений.")


def prompt_whole(task: dict) -> str:
    return f"База:\n{task['db_text']}\n\n{task['query_text']}"


# ---------------- МДП: interface checks, deterministic, 0 LLM calls --------------------

def mdp_seam1(task: dict, dep_id) -> bool:
    """Does the id produced by discipline A exist as a dept in block B?"""
    return dep_id is not None and any(m["dept"] == dep_id for m in task["mgrs"])


def mdp_seam2(task: dict, mgr_id) -> bool:
    """Does the id produced by discipline B exist as a mgr in block C?"""
    return mgr_id is not None and any(e["mgr"] == mgr_id for e in task["emps"])


def mdp_seam2_consistency(task: dict, dep_id, mgr_id) -> bool:
    """Did discipline B actually use the dept discipline A named?"""
    if mgr_id is None or dep_id is None:
        return False
    m = next((x for x in task["mgrs"] if x["id"] == mgr_id), None)
    return m is not None and m["dept"] == dep_id
