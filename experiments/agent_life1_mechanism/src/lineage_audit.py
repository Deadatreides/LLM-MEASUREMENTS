"""lineage_audit.py — PROTOCOL.md §1/§2: Imp/A/AB/P(B|A)/L/overlap на
инструментированных прогонах (Gate R2 подтвердил: побитово тот же, что
архивный A5). Все геноитпы — деревья из `archive.json` (не парсинг трасс:
теперь есть полные структуры, парсинг был нужен только forensically в A5,
когда деревьев не было)."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402

HSTEP = LD.HSTEP
H = LD.H

STEP_KINDS = HSTEP.STEP_KINDS   # ("READ","FORMAT","LOOKUP","COMPUTE") -- позиция = индекс слота


def slot_models(slot_node: dict) -> list:
    """CALL model_id'ы, достижимые внутри поддерева слота (SEQ(single) или
    PAR(SEQ,SEQ,...)). Не зависит от формы -- обходит дерево целиком."""
    out: list = []

    def walk(n):
        if not isinstance(n, dict):
            return
        op = n.get("op")
        if op == "CALL":
            mol = n.get("molecule", "")
            if mol.startswith("gen."):
                out.append(mol[4:])
        elif op in ("SEQ", "PAR", "ASSEMBLE"):
            for c in n.get("children") or []:
                walk(c)
        elif op == "SWITCH":
            for c in (n.get("cases") or {}).values():
                walk(c)
            if "default" in n:
                walk(n["default"])
        elif op == "BUDGET":
            if "child" in n:
                walk(n["child"])

    walk(slot_node)
    return out


def best_single_per_slot(ds, panel: list) -> dict:
    """PROTOCOL.md §1.2: best_single(s) по панели, один раз на seed
    (панель фиксирована на весь прогон)."""
    out = {}
    for s, kind in enumerate(STEP_KINDS):
        best = 0
        for m in ds.models:
            n = sum(1 for t in panel if ds.cells[(t, kind, m)]["status"] == "PASS")
            best = max(best, n)
        out[s] = best
    return out


def covered_set(ds, panel: list, s: int, models: list) -> set:
    kind = STEP_KINDS[s]
    out: set = set()
    for m in models:
        out |= {t for t in panel if ds.cells[(t, kind, m)]["status"] == "PASS"}
    return out


def imp_set(ds, panel: list, best_single: dict, genotype: dict) -> frozenset:
    """PROTOCOL.md §1.2. genotype -- полный dict из archive.json (g["root"]
    -- ASSEMBLE с 4 children, позиция = STEP_KINDS индекс)."""
    root = genotype.get("root", {})
    children = root.get("children") or []
    out = set()
    for s in range(min(len(children), len(STEP_KINDS))):
        models = slot_models(children[s])
        if not models:
            continue
        cov = covered_set(ds, panel, s, models)
        if len(cov) > best_single[s]:
            out.add(s)
    return frozenset(out)


def load_seed_artifacts(seed_dir: Path) -> dict:
    with open(seed_dir / "archive.json", encoding="utf-8") as f:
        archive = json.load(f)["genotypes"]
    genotypes = {g["complex_id"]: g for g in archive}

    with open(seed_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    births = []
    births_path = seed_dir / "births.jsonl"
    if births_path.exists():
        for line in births_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                births.append(json.loads(line))

    edges = H.EdgeLog.from_jsonl(seed_dir / "heredity.jsonl")

    return {"genotypes": genotypes, "summary": summary, "births": births, "edges": edges}


def death_gen_index(summary: dict) -> dict:
    """complex_id -> поколение, на котором он появился в `deaths` (D1/D2)
    любого generation-отчёта. Не покрывает D3 elite-dropout (это НЕ смерть
    из архива, только выход из элиты) -- по PROTOCOL.md §2 L(A) считает
    именно смерть/конец прогона, не колебания элиты."""
    out = {}
    for rep in summary.get("generations", []):
        gen = rep["generation"]
        for d in rep.get("deaths", []):
            cid = d["complex_id"]
            if cid not in out:
                out[cid] = gen
    return out


def audit_seed(seed: int, seed_dir: Path, ds, verbose: bool = True) -> dict:
    art = load_seed_artifacts(seed_dir)
    genotypes, summary, births, edges = (art["genotypes"], art["summary"],
                                         art["births"], art["edges"])
    panel = summary["panel"]
    best_single = best_single_per_slot(ds, panel)
    last_gen = summary["n_generations"] - 1
    deaths_at = death_gen_index(summary)

    imp_cache: dict = {}

    def imp(cid: str) -> frozenset:
        if cid not in imp_cache:
            g = genotypes.get(cid)
            imp_cache[cid] = imp_set(ds, panel, best_single, g) if g else frozenset()
        return imp_cache[cid]

    # -- затравка (gen=0): AB-статус публикуется отдельно, НЕ смешивается
    # с эволюционным знаменателем (PROTOCOL.md §1.4) --
    seed_cids = [cid for cid, g in genotypes.items() if g.get("gen", 0) == 0]
    seed_ab = [cid for cid in seed_cids if len(imp(cid)) >= 2]

    # -- эволюционные рождения: только births с parent_id (mutation/recomb,
    # исключая "fresh" случайные генотипы -- у тех нет родителя, значит нет
    # понятия A-события) --
    evolved_births = [b for b in births if b.get("parent_id")]

    a_events = []
    for b in evolved_births:
        child_id, parent_id = b["child_id"], b["parent_id"]
        if child_id not in genotypes or parent_id not in genotypes:
            continue
        imp_child, imp_parent = imp(child_id), imp(parent_id)
        if imp_child.issuperset(imp_parent) and len(imp_child) == len(imp_parent) + 1:
            a_events.append({**b, "imp_parent": sorted(imp_parent), "imp_child": sorted(imp_child)})

    n_a = len(a_events)
    b_flags = []
    lifespans = []
    l_a_to_b = []
    for ev_row in a_events:
        cid = ev_row["child_id"]
        lineage = {cid} | edges.descendants(cid)
        is_ab_now = len(imp(cid)) >= 2
        reaches_ab = is_ab_now or any(len(imp(d)) >= 2 for d in lineage if d in genotypes)
        b_flags.append(reaches_ab)

        birth_gen = genotypes[cid].get("gen", ev_row.get("gen", 0))
        death_gen = deaths_at.get(cid, last_gen)
        lifespans.append(max(0, death_gen - birth_gen))
        if reaches_ab and not is_ab_now:
            # L(A->B): первое поколение, на котором ПОТОМОК достиг AB
            ab_gens = [genotypes[d].get("gen") for d in lineage
                      if d in genotypes and len(imp(d)) >= 2 and d != cid]
            if ab_gens:
                l_a_to_b.append(max(0, min(ab_gens) - birth_gen))

    p_b_given_a = (sum(b_flags) / n_a) if n_a else None
    median_l = statistics.median(lifespans) if lifespans else None

    # -- overlap: доля поколений, где ≥2 живых линии несут РАЗНЫЕ непустые
    # ImpSet (независимые блоки сосуществуют в популяции) --
    gens_with_overlap = 0
    n_gens_checked = 0
    for rep in summary.get("generations", []):
        pop = rep.get("population", [])
        imp_sets = {imp(cid) for cid in pop if cid in genotypes and imp(cid)}
        n_gens_checked += 1
        distinct_nonempty = {s for s in imp_sets if s}
        if len(distinct_nonempty) >= 2:
            gens_with_overlap += 1
    overlap = (gens_with_overlap / n_gens_checked) if n_gens_checked else None

    # -- P(AB | A alive) vs P(AB | A dead, использован донором) --
    alive_ab, alive_n, dead_donor_ab, dead_donor_n = 0, 0, 0, 0
    donor_ids_used = {b.get("donor_id") for b in births if b.get("donor_id")}
    for ev_row in a_events:
        cid = ev_row["child_id"]
        died = cid in deaths_at
        used_as_donor = cid in donor_ids_used
        if not died:
            alive_n += 1
            alive_ab += int(len(imp(cid)) >= 2)
        elif used_as_donor:
            dead_donor_n += 1
            dead_donor_ab += int(len(imp(cid)) >= 2)

    # -- по оператору --
    by_operator: dict = {}
    for ev_row in a_events:
        op = ev_row.get("operator") or "?"
        by_operator.setdefault(op, {"n": 0, "n_ab": 0})
        by_operator[op]["n"] += 1
        by_operator[op]["n_ab"] += int(len(imp(ev_row["child_id"])) >= 2)

    # -- M5/SWAP_SLOT атрибуция --
    n_m5 = sum(1 for b in births if b.get("operator") == "SWAP_SLOT")
    n_m5_slot_match = 0
    n_m5_to_ab = 0
    for b in births:
        if b.get("operator") != "SWAP_SLOT":
            continue
        slot = b.get("slot")
        donor_id, parent_id, child_id = b.get("donor_id"), b.get("parent_id"), b["child_id"]
        if slot is None or donor_id not in genotypes or parent_id not in genotypes:
            continue
        donor_imp, parent_imp = imp(donor_id), imp(parent_id)
        if slot in donor_imp and slot not in parent_imp:
            n_m5_slot_match += 1
        if child_id in genotypes and len(imp(child_id)) >= 2:
            n_m5_to_ab += 1

    # -- AB-доля в элите (пул по всем поколениям, эволюционные + затравка,
    # т.к. элита -- наблюдаемый факт отбора, не только эволюционный подсчёт) --
    elite_ever = set()
    for rep in summary.get("generations", []):
        elite_ever |= set(rep.get("elite", []))
    elite_ab = sum(1 for cid in elite_ever if cid in genotypes and len(imp(cid)) >= 2)
    elite_ab_share = (elite_ab / len(elite_ever)) if elite_ever else None

    n_ab_evolved = sum(1 for cid, g in genotypes.items()
                       if g.get("gen", 0) >= 1 and len(imp(cid)) >= 2)

    result = {
        "seed": seed, "panel_size": len(panel), "n_generations": summary["n_generations"],
        "extinct": summary["extinct"],
        "best_single_per_slot": {STEP_KINDS[s]: v for s, v in best_single.items()},
        "n_seed_genotypes": len(seed_cids), "n_seed_ab": len(seed_ab),
        "seed_ab_ids": sorted(seed_ab),
        "N_A": n_a, "N_AB_evolved": n_ab_evolved,
        "P_B_given_A": p_b_given_a, "median_L_A": median_l,
        "L_A_to_B_values": l_a_to_b,
        "median_L_A_to_B": statistics.median(l_a_to_b) if l_a_to_b else None,
        "overlap": overlap, "n_generations_checked_for_overlap": n_gens_checked,
        "P_AB_given_A_alive": (alive_ab / alive_n) if alive_n else None,
        "n_A_alive": alive_n,
        "P_AB_given_A_dead_used_as_donor": (dead_donor_ab / dead_donor_n) if dead_donor_n else None,
        "n_A_dead_used_as_donor": dead_donor_n,
        "AB_share_by_operator": {op: v["n_ab"] / v["n"] if v["n"] else None
                                 for op, v in by_operator.items()},
        "N_by_operator": {op: v["n"] for op, v in by_operator.items()},
        "N_M5": n_m5, "N_M5_slot_match": n_m5_slot_match, "N_M5_to_AB": n_m5_to_ab,
        "elite_ever_size": len(elite_ever), "elite_AB_share": elite_ab_share,
        "_l_a_values": lifespans, "_b_flags": b_flags,
    }
    if verbose:
        print(f"[audit] seed {seed}: N_A={n_a} N_AB_evolved={n_ab_evolved} "
              f"P(B|A)={p_b_given_a} median_L(A)={median_l} overlap={overlap} "
              f"elite_AB_share={elite_ab_share}", flush=True)
    return result


def pool_seeds(per_seed: list) -> dict:
    n_a_total = sum(r["N_A"] for r in per_seed)
    n_ab_total = sum(r["N_AB_evolved"] for r in per_seed)
    all_l = [l for r in per_seed for l in ([r["median_L_A"]] if r["median_L_A"] is not None else [])]
    # честнее: собрать L(A) по каждому A-событию, не медиану медиан -- но L(A)
    # по отдельным событиям не хранится в per_seed сводке; пересчитывается ниже
    p_b_vals = [r["P_B_given_A"] for r in per_seed if r["P_B_given_A"] is not None]
    overlap_vals = [r["overlap"] for r in per_seed if r["overlap"] is not None]
    elite_ab_vals = [r["elite_AB_share"] for r in per_seed if r["elite_AB_share"] is not None]
    return {
        "N_A_total": n_a_total, "N_AB_evolved_total": n_ab_total,
        "mean_P_B_given_A": statistics.mean(p_b_vals) if p_b_vals else None,
        "mean_overlap": statistics.mean(overlap_vals) if overlap_vals else None,
        "mean_elite_AB_share": statistics.mean(elite_ab_vals) if elite_ab_vals else None,
        "median_of_seed_median_L_A": statistics.median(all_l) if all_l else None,
    }


def classify_basket(pooled: dict, per_seed: list) -> dict:
    """PROTOCOL.md §2, предрегистрированные пороги. Считает по ПУЛУ всех
    5 seed (N_A суммируется; P(B|A)/median L(A) -- по объединению всех
    A-событий всех seed, не по среднему медиан по seed -- честнее для
    порогового решения)."""
    all_l_a = []
    all_b_flags = []
    for r in per_seed:
        # L(A) по отдельным событиям пересчитывается заново в run_audit.py и
        # передаётся сюда через per_seed[i]["_l_a_values"]/["_b_flags"] --
        # см. вызов ниже.
        all_l_a.extend(r.get("_l_a_values", []))
        all_b_flags.extend(r.get("_b_flags", []))

    n_a = pooled["N_A_total"]
    n_ab = pooled["N_AB_evolved_total"]
    p_b_given_a = (sum(all_b_flags) / len(all_b_flags)) if all_b_flags else None
    median_l = statistics.median(all_l_a) if all_l_a else None
    elite_ab_share = pooled["mean_elite_AB_share"]

    if n_a < 10:
        return {"basket": "MECH_UNKNOWN", "reason": f"N_A={n_a} < 10 (мало локальных блоков)"}

    fires = []
    if p_b_given_a is not None and median_l is not None:
        if p_b_given_a < 0.05 and median_l >= 3:
            fires.append("MECH_RECOMB")
        if p_b_given_a >= 0.05 and median_l < 3:
            fires.append("MECH_LIFETIME")
    if elite_ab_share is not None and n_ab >= 5 and elite_ab_share < 0.2:
        fires.append("MECH_SELECT")

    if len(fires) == 1:
        return {"basket": fires[0], "N_A": n_a, "N_AB": n_ab,
               "P_B_given_A": p_b_given_a, "median_L_A": median_l,
               "elite_AB_share": elite_ab_share}
    if len(fires) >= 2:
        return {"basket": "MECH_MIXED", "fired": fires, "N_A": n_a, "N_AB": n_ab,
               "P_B_given_A": p_b_given_a, "median_L_A": median_l,
               "elite_AB_share": elite_ab_share}
    return {"basket": "MECH_UNKNOWN", "N_A": n_a, "N_AB": n_ab,
           "P_B_given_A": p_b_given_a, "median_L_A": median_l,
           "elite_AB_share": elite_ab_share,
           "reason": ("ни один порог не сработал -- "
                      f"P(B|A)={p_b_given_a}, median_L(A)={median_l}, "
                      f"N_AB={n_ab}, elite_AB_share={elite_ab_share}")}
