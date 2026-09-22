"""coverage.py — Расчёт оракул-пространства U, пар P и метрик покрытия (Coverage)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
ARCH2 = ROOT / "arch2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ARCH2) not in sys.path:
    sys.path.insert(0, str(ARCH2))

import heterostep as HSTEP
import heterostep_seeds as HSEED
import genotype as G

def build_oracle_space() -> Tuple[List[dict], List[dict]]:
    """Построение оракул-пространства атомов U и пар P на HETEROSTEP train grid."""
    ds = HSTEP.default_dataset()
    train = ds.split["train"]
    b1_map = HSEED.b1_route(ds, train)

    atoms = []
    # depth-1 атомы
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
                    "id": f"atom_d1_s{s}_{m}",
                    "slot": s,
                    "kind": kind,
                    "model": m,
                    "models": [m],
                    "type": "depth1",
                    "marginal_count": len(marginal),
                })
        # depth-2 атомы (PAR пара с b1_m)
        for m in ds.models:
            if m == b1_m:
                continue
            m_passes = {t for t in train if ds.cells[(t, kind, m)]["status"] == "PASS"}
            marginal = m_passes - b1_passes
            if marginal:
                atoms.append({
                    "id": f"atom_d2_s{s}_{b1_m}+{m}",
                    "slot": s,
                    "kind": kind,
                    "models": [b1_m, m],
                    "type": "depth2",
                    "marginal_count": len(marginal),
                })

    # Пары P: пары атомов на РАЗНЫХ слотах
    pairs = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            a1 = atoms[i]
            a2 = atoms[j]
            if a1["slot"] != a2["slot"]:
                pairs.append({
                    "id": f"pair_{a1['id']}__x__{a2['id']}",
                    "atom1_id": a1["id"],
                    "atom2_id": a2["id"],
                    "slot1": a1["slot"],
                    "slot2": a2["slot"],
                })

    return atoms, pairs

def extract_genotype_models_per_slot(genotype_dict: dict) -> List[Set[str]]:
    """Извлекает множество моделей по 4 слотам ASSEMBLE генотипа."""
    root = genotype_dict.get("root", {})
    slots_models: List[Set[str]] = [set() for _ in range(4)]
    
    if root.get("op") == "ASSEMBLE":
        children = root.get("children", [])
        for s in range(min(4, len(children))):
            slot_node = children[s]
            # Обходим поддерево слота и ищем CALL
            for _, node in G.walk(slot_node):
                if node.get("op") == "CALL":
                    mol = node.get("molecule", "")
                    if mol.startswith("gen."):
                        mid = mol[4:]
                        slots_models[s].add(mid)
    return slots_models

def genotype_has_atom(slots_models: List[Set[str]], atom: dict) -> bool:
    s = atom["slot"]
    target_models = set(atom["models"])
    return target_models.issubset(slots_models[s])

def analyze_archive_coverage(genotypes: List[dict], atoms: List[dict], pairs: List[dict]) -> dict:
    """Вычисляет Coverage_atoms, Coverage_pairs, P(A), P(B), P(A^B) по архиву генотипов."""
    if not genotypes:
        return {
            "coverage_atoms": 0.0,
            "coverage_pairs": 0.0,
            "P_A": 0.0,
            "P_B": 0.0,
            "P_A_and_B": 0.0,
            "n_genotypes": 0,
        }

    parsed_genotypes = [extract_genotype_models_per_slot(g) for g in genotypes]
    
    # 1. Coverage_atoms
    covered_atom_ids = set()
    for atom in atoms:
        for sm in parsed_genotypes:
            if genotype_has_atom(sm, atom):
                covered_atom_ids.add(atom["id"])
                break
    coverage_atoms = len(covered_atom_ids) / len(atoms) if atoms else 0.0

    # 2. Coverage_pairs
    covered_pair_ids = set()
    atom_by_id = {a["id"]: a for a in atoms}
    for pair in pairs:
        a1 = atom_by_id[pair["atom1_id"]]
        a2 = atom_by_id[pair["atom2_id"]]
        for sm in parsed_genotypes:
            if genotype_has_atom(sm, a1) and genotype_has_atom(sm, a2):
                covered_pair_ids.add(pair["id"])
                break
    coverage_pairs = len(covered_pair_ids) / len(pairs) if pairs else 0.0

    # 3. P(A), P(B), P(A & B)
    # A = COMPUTE (slot 3) expanded (len(models) >= 2)
    # B = LOOKUP (slot 2) expanded (len(models) >= 2)
    count_A = 0
    count_B = 0
    count_AB = 0
    N = len(parsed_genotypes)

    for sm in parsed_genotypes:
        has_A = len(sm[3]) >= 2
        has_B = len(sm[2]) >= 2
        if has_A:
            count_A += 1
        if has_B:
            count_B += 1
        if has_A and has_B:
            count_AB += 1

    return {
        "coverage_atoms": round(coverage_atoms, 4),
        "coverage_pairs": round(coverage_pairs, 4),
        "P_A": round(count_A / N, 4),
        "P_B": round(count_B / N, 4),
        "P_A_and_B": round(count_AB / N, 4),
        "covered_atoms_count": len(covered_atom_ids),
        "total_atoms_count": len(atoms),
        "covered_pairs_count": len(covered_pair_ids),
        "total_pairs_count": len(pairs),
        "n_genotypes": N,
    }

def main():
    atoms, pairs = build_oracle_space()
    data = {
        "n_atoms": len(atoms),
        "n_pairs": len(pairs),
        "atoms": atoms,
        "pairs": pairs,
    }
    metrics_dir = ROOT / "agent_a2_assemble_budget" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "oracle_atoms.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[coverage] Создан oracle_atoms.json: {len(atoms)} атомов, {len(pairs)} пар.")

if __name__ == "__main__":
    main()
