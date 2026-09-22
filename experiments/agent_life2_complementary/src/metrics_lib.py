"""metrics_lib.py — Imp/A/AB core, копия (не импорт) `agent_life1_mechanism/
src/lineage_audit.py`'s `slot_models`/`best_single_per_slot`/`covered_set`/
`imp_set`/`load_seed_artifacts`/`death_gen_index` -- формула НЕ менялась
(PROTOCOL.md §2, "как LIFE-1, не двигать"). Добавлено: `N_AB_first_assembly`
(§2.1 -- родитель НЕ AB, ребёнок AB -- разделяет СБОРКУ от НАСЛЕДОВАНИЯ) и
`audit_cell` для схемы births.jsonl этого пакета (логирует КАЖДУЮ попытку,
не только принятые рождения -- LIFE-1 логировал только принятые).
"""

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
HSEED = LD.HSEED
H = LD.H

STEP_KINDS = HSTEP.STEP_KINDS   # ("READ","FORMAT","LOOKUP","COMPUTE") -- позиция = индекс слота


def slot_models(slot_node: dict) -> list:
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


def load_cell_artifacts(cell_dir: Path) -> dict:
    with open(cell_dir / "archive.json", encoding="utf-8") as f:
        archive = json.load(f)["genotypes"]
    genotypes = {g["complex_id"]: g for g in archive}

    with open(cell_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    births = []
    births_path = cell_dir / "births.jsonl"
    if births_path.exists():
        for line in births_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                births.append(json.loads(line))

    edges = H.EdgeLog.from_jsonl(cell_dir / "heredity.jsonl")

    return {"genotypes": genotypes, "summary": summary, "births": births, "edges": edges}


def death_gen_index(summary: dict) -> dict:
    out = {}
    for rep in summary.get("generations", []):
        gen = rep["generation"]
        for d in rep.get("deaths", []):
            cid = d["complex_id"]
            if cid not in out:
                out[cid] = gen
    return out


def audit_cell(p: float, seed: int, cell_dir: Path, ds, panel: list, best_single: dict,
              verbose: bool = True) -> dict:
    """PROTOCOL.md §7: метрики одной клетки (p, seed). `births.jsonl` здесь
    логирует КАЖДУЮ попытку размножения (`admitted` True/False), не только
    принятые -- в отличие от LIFE-1."""
    art = load_cell_artifacts(cell_dir)
    genotypes, summary, births, edges = (art["genotypes"], art["summary"],
                                         art["births"], art["edges"])
    last_gen = summary["n_generations"] - 1
    deaths_at = death_gen_index(summary)

    imp_cache: dict = {}

    def imp(cid: str) -> frozenset:
        if cid not in imp_cache:
            g = genotypes.get(cid)
            imp_cache[cid] = imp_set(ds, panel, best_single, g) if g else frozenset()
        return imp_cache[cid]

    # -- sanity: Imp(FORMAT) обязан быть 0 везде (PROTOCOL.md §2.2) --
    fmt_violations = [cid for cid in genotypes if 1 in imp(cid)]

    # -- затравка (gen=0): отдельно, не в знаменателе (PROTOCOL.md §2) --
    seed_cids = [cid for cid, g in genotypes.items() if g.get("gen", 0) == 0]
    seed_ab = [cid for cid in seed_cids if len(imp(cid)) >= 2]

    admitted_births = [b for b in births if b.get("admitted") and b.get("parent_id")
                       and b.get("child_id") in genotypes]

    a_events = []
    first_assembly = []
    delta_hist: dict = {}
    for b in admitted_births:
        child_id, parent_id = b["child_id"], b["parent_id"]
        if parent_id not in genotypes:
            continue
        imp_child, imp_parent = imp(child_id), imp(parent_id)
        d = len(imp_child) - len(imp_parent)
        delta_hist[d] = delta_hist.get(d, 0) + 1
        if imp_child.issuperset(imp_parent) and d == 1:
            a_events.append({**b, "imp_parent": sorted(imp_parent), "imp_child": sorted(imp_child)})
        if len(imp_parent) < 2 and len(imp_child) >= 2:
            first_assembly.append(b)

    n_a = len(a_events)
    n_ab_evolved = sum(1 for cid, g in genotypes.items()
                       if g.get("gen", 0) >= 1 and len(imp(cid)) >= 2)
    n_ab_first_assembly = len(first_assembly)

    b_flags, lifespans = [], []
    for ev_row in a_events:
        cid = ev_row["child_id"]
        lineage = {cid} | edges.descendants(cid)
        is_ab_now = len(imp(cid)) >= 2
        reaches_ab = is_ab_now or any(len(imp(d)) >= 2 for d in lineage if d in genotypes)
        b_flags.append(reaches_ab)
        birth_gen = genotypes[cid].get("gen", ev_row.get("gen", 0))
        death_gen = deaths_at.get(cid, last_gen)
        lifespans.append(max(0, death_gen - birth_gen))

    p_b_given_a = (sum(b_flags) / n_a) if n_a else None
    median_l = statistics.median(lifespans) if lifespans else None

    # -- overlap: доля поколений, где ≥2 живых линии несут РАЗНЫЕ непустые ImpSet --
    gens_with_overlap, n_gens_checked = 0, 0
    for rep in summary.get("generations", []):
        pop = rep.get("population", [])
        distinct_nonempty = {imp(cid) for cid in pop if cid in genotypes and imp(cid)}
        n_gens_checked += 1
        if len(distinct_nonempty) >= 2:
            gens_with_overlap += 1
    overlap = (gens_with_overlap / n_gens_checked) if n_gens_checked else None

    # -- loss:gain, ПО ВСЕМ операторам (overall) и отдельно только TRANSFER_SLOT --
    def loss_gain(rows):
        gains = sum(1 for r in rows if r["_d"] > 0)
        losses = sum(1 for r in rows if r["_d"] < 0)
        return losses / max(1, gains), gains, losses

    all_rows = [{"_d": len(imp(b["child_id"])) - len(imp(b["parent_id"]))}
               for b in admitted_births]
    transfer_rows = [{"_d": len(imp(b["child_id"])) - len(imp(b["parent_id"]))}
                    for b in admitted_births if b.get("operator") == "TRANSFER_SLOT"]
    loss_gain_overall, gains_overall, losses_overall = loss_gain(all_rows)
    loss_gain_transfer, gains_transfer, losses_transfer = loss_gain(transfer_rows)

    # -- slot_match: доля УСПЕШНЫХ TRANSFER_SLOT с complementary=True (ожидание: 1.0,
    # это проверка корректности реализации, не находка -- PROTOCOL.md §1) --
    transfer_success = [b for b in births if b.get("operator") == "TRANSFER_SLOT"]
    slot_match_rate = (sum(1 for b in transfer_success if b.get("complementary")) /
                       len(transfer_success)) if transfer_success else None

    # -- доля попыток HGT, реально нашедших комплементарного донора --
    hgt_attempts = [b for b in births if b.get("hgt_attempted")]
    hgt_successes = [b for b in hgt_attempts if b.get("operator") == "TRANSFER_SLOT"]
    transfer_success_rate = (len(hgt_successes) / len(hgt_attempts)) if hgt_attempts else None

    # -- AB-доля в элите --
    elite_ever = set()
    for rep in summary.get("generations", []):
        elite_ever |= set(rep.get("elite", []))
    elite_ab = sum(1 for cid in elite_ever if cid in genotypes and len(imp(cid)) >= 2)
    elite_ab_share = (elite_ab / len(elite_ever)) if elite_ever else None

    result = {
        "p": p, "seed": seed, "n_generations": summary["n_generations"],
        "extinct": summary["extinct"],
        "format_imp_violations": len(fmt_violations),
        "n_seed_genotypes": len(seed_cids), "n_seed_ab": len(seed_ab),
        "n_admitted_births": len(admitted_births), "n_attempted_births": len(births),
        "N_A": n_a, "N_AB_evolved": n_ab_evolved, "N_AB_first_assembly": n_ab_first_assembly,
        "P_B_given_A": p_b_given_a, "median_L_A": median_l,
        "overlap": overlap, "n_generations_checked_for_overlap": n_gens_checked,
        "loss_gain_overall": loss_gain_overall, "gains_overall": gains_overall,
        "losses_overall": losses_overall,
        "loss_gain_transfer_only": loss_gain_transfer, "gains_transfer": gains_transfer,
        "losses_transfer": losses_transfer,
        "slot_match_rate": slot_match_rate, "n_transfer_success": len(transfer_success),
        "n_hgt_attempts": len(hgt_attempts), "n_hgt_successes": len(hgt_successes),
        "transfer_success_rate": transfer_success_rate,
        "elite_ever_size": len(elite_ever), "elite_AB_share": elite_ab_share,
        "delta_impset_histogram": delta_hist,
        "_l_a_values": lifespans, "_b_flags": b_flags,
    }
    if verbose:
        print(f"[audit] p={p} seed={seed}: N_A={n_a} N_AB_evolved={n_ab_evolved} "
              f"N_AB_first_assembly={n_ab_first_assembly} loss:gain(overall)={loss_gain_overall:.2f} "
              f"slot_match={slot_match_rate} transfer_success_rate={transfer_success_rate} "
              f"elite_AB_share={elite_ab_share}", flush=True)
    if fmt_violations:
        print(f"[audit] !!! FORMAT Imp violation at p={p} seed={seed}: {fmt_violations[:5]}",
              flush=True)
    return result
