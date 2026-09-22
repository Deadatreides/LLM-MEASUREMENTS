"""metrics_lib.py — U/P/Coverage/P(A,B) + archive-вердикты V1/V2/V3.

Формула оракул-пространства U, пар P, Coverage_atoms/Coverage_pairs и
P(A)/P(B)/P(A^B) — КОПИЯ (не импорт) `agent_a2_assemble_budget/src/
coverage.py`, дословно та же логика: изоляция пакетов агентов друг от друга
(PROTOCOL.md §1). Определение НЕ меняется в этой копии.

Archive-вердикты V1_archive/V2_archive/V3_archive — новая часть, специфичная
для A3 (PROTOCOL.md §5.1): переоценка ВСЕГО архива прогона на ПОЛНОЙ
train-сетке (100 задач, не срез), сравнение с ФИКСИРОВАННЫМИ каноническими
числами B1/B2/B3, а не с шумным per-slice эталоном.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
ARCH2 = ROOT / "arch2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ARCH2) not in sys.path:
    sys.path.insert(0, str(ARCH2))

import genotype as G       # noqa: E402
import heterostep as HSTEP  # noqa: E402
import heterostep_seeds as HSEED  # noqa: E402
import runner as R          # noqa: E402

# -- канонические базовые линии (PROTOCOL.md §2/§5.1, фиксированы, не пересчитываются по срезу) --
CANON_B1_R, CANON_B1_C = 0.4600, 667.4
CANON_B2_R = 0.5800
CANON_B3_R, CANON_B3_C = 0.7000, 917.4


# -- копия формулы U/P/Coverage из agent_a2_assemble_budget/src/coverage.py --------------------


def build_oracle_space() -> Tuple[List[dict], List[dict]]:
    """Построение оракул-пространства атомов U и пар P на HETEROSTEP train grid.

    Идентично agent_a2_assemble_budget/src/coverage.py::build_oracle_space --
    скопировано, не импортировано (изоляция пакетов)."""
    ds = HSTEP.default_dataset()
    train = ds.split["train"]
    b1_map = HSEED.b1_route(ds, train)

    atoms = []
    for s, kind in enumerate(HSTEP.STEP_KINDS):
        b1_m = b1_map[kind][0]
        b1_passes = {t for t in train if ds.cells[(t, kind, b1_m)]["status"] == "PASS"}
        for m in ds.models:
            if m == b1_m:
                continue
            m_passes = {t for t in train if ds.cells[(t, kind, m)]["status"] == "PASS"}
            marginal = m_passes - b1_passes
            if marginal:
                atoms.append({
                    "id": f"atom_d1_s{s}_{m}", "slot": s, "kind": kind, "model": m,
                    "models": [m], "type": "depth1", "marginal_count": len(marginal),
                })
        for m in ds.models:
            if m == b1_m:
                continue
            m_passes = {t for t in train if ds.cells[(t, kind, m)]["status"] == "PASS"}
            marginal = m_passes - b1_passes
            if marginal:
                atoms.append({
                    "id": f"atom_d2_s{s}_{b1_m}+{m}", "slot": s, "kind": kind,
                    "models": [b1_m, m], "type": "depth2", "marginal_count": len(marginal),
                })

    pairs = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            a1, a2 = atoms[i], atoms[j]
            if a1["slot"] != a2["slot"]:
                pairs.append({
                    "id": f"pair_{a1['id']}__x__{a2['id']}", "atom1_id": a1["id"],
                    "atom2_id": a2["id"], "slot1": a1["slot"], "slot2": a2["slot"],
                })

    return atoms, pairs


def extract_genotype_models_per_slot(genotype_dict: dict) -> List[Set[str]]:
    """Идентично agent_a2_assemble_budget/src/coverage.py -- скопировано."""
    root = genotype_dict.get("root", {})
    slots_models: List[Set[str]] = [set() for _ in range(4)]
    if root.get("op") == "ASSEMBLE":
        children = root.get("children", [])
        for s in range(min(4, len(children))):
            for _, node in G.walk(children[s]):
                if node.get("op") == "CALL":
                    mol = node.get("molecule", "")
                    if mol.startswith("gen."):
                        slots_models[s].add(mol[4:])
    return slots_models


def genotype_has_atom(slots_models: List[Set[str]], atom: dict) -> bool:
    s = atom["slot"]
    return set(atom["models"]).issubset(slots_models[s])


def analyze_archive_coverage(genotypes: List[dict], atoms: List[dict], pairs: List[dict]) -> dict:
    """Идентично agent_a2_assemble_budget/src/coverage.py -- скопировано."""
    if not genotypes:
        return {"coverage_atoms": 0.0, "coverage_pairs": 0.0, "P_A": 0.0, "P_B": 0.0,
               "P_A_and_B": 0.0, "n_genotypes": 0}

    parsed = [extract_genotype_models_per_slot(g) for g in genotypes]

    covered_atoms = {a["id"] for a in atoms if any(genotype_has_atom(sm, a) for sm in parsed)}
    coverage_atoms = len(covered_atoms) / len(atoms) if atoms else 0.0

    atom_by_id = {a["id"]: a for a in atoms}
    covered_pairs = set()
    for pair in pairs:
        a1, a2 = atom_by_id[pair["atom1_id"]], atom_by_id[pair["atom2_id"]]
        if any(genotype_has_atom(sm, a1) and genotype_has_atom(sm, a2) for sm in parsed):
            covered_pairs.add(pair["id"])
    coverage_pairs = len(covered_pairs) / len(pairs) if pairs else 0.0

    n = len(parsed)
    count_a = sum(1 for sm in parsed if len(sm[3]) >= 2)   # COMPUTE = slot 3
    count_b = sum(1 for sm in parsed if len(sm[2]) >= 2)   # LOOKUP = slot 2
    count_ab = sum(1 for sm in parsed if len(sm[3]) >= 2 and len(sm[2]) >= 2)

    return {
        "coverage_atoms": round(coverage_atoms, 4), "coverage_pairs": round(coverage_pairs, 4),
        "P_A": round(count_a / n, 4), "P_B": round(count_b / n, 4),
        "P_A_and_B": round(count_ab / n, 4), "n_genotypes": n,
    }


# -- archive-вердикты (новая часть A3, PROTOCOL.md §5.1) ---------------------------------------


def evaluate_full_train(genotype_dict: dict, ds, reg, be, train_ids: list, budget: int = 3000) -> Tuple[float, float]:
    """r, c генотипа на ПОЛНОЙ train-сетке (все 100 задач) -- не срез."""
    n_resolved = 0
    total_cost = 0
    for tid in train_ids:
        res = R.run(genotype_dict, ds.initial_state(tid), reg, be, budget_tokens=budget)
        if res.outcome == R.RESOLVED:
            n_resolved += 1
        total_cost += res.cost
    return n_resolved / len(train_ids), total_cost / len(train_ids)


def _compute_slot_is_wide_par(genotype_dict: dict) -> bool:
    root = genotype_dict.get("root", {})
    if root.get("op") != "ASSEMBLE":
        return False
    children = root.get("children", [])
    if len(children) < 4:
        return False
    compute_slot = children[3]
    return compute_slot.get("op") == "PAR" and len(compute_slot.get("children", [])) >= 2


def archive_verdicts(archive_genotypes: List[dict], ds, reg, be, train_ids: list) -> dict:
    """PROTOCOL.md §5.1: переоценка ВСЕГО архива на полной сетке, сравнение с
    ФИКСИРОВАННЫМИ каноническими B1/B2/B3 (не пересчитанными по срезу)."""
    evaluated = []
    for g in archive_genotypes:
        r_full, c_full = evaluate_full_train(g, ds, reg, be, train_ids)
        evaluated.append((g, r_full, c_full))

    v1 = any(r > CANON_B1_R and c <= CANON_B1_C for _, r, c in evaluated)
    v2 = any(_compute_slot_is_wide_par(g) and r >= CANON_B2_R for g, r, _ in evaluated)
    v3 = any(r >= CANON_B3_R and c < CANON_B3_C for _, r, c in evaluated)

    if evaluated:
        best_g, best_r, best_c = max(evaluated, key=lambda t: (t[1], -t[2]))
    else:
        best_r, best_c = 0.0, float("inf")

    return {
        "v1_archive": v1, "v2_archive": v2, "v3_archive": v3,
        "best_r_archive": best_r, "best_c_archive": best_c if best_c != float("inf") else -1.0,
        "n_evaluated": len(evaluated),
    }
