"""metrics_lib.py — Imp/A/AB core, копия (не импорт) `agent_life2_
complementary/src/metrics_lib.py` (сама копия LIFE-1's `lineage_audit.py`
-- формула Imp/A/AB НЕ менялась, PROTOCOL.md §2). Добавлено (LIFE-3-
специфично): A1-A5 для БЛОКОВ (PROTOCOL.md §4.3), не для целых
комплексов, и чтение `blocks.jsonl`.
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

STEP_KINDS = HSTEP.STEP_KINDS


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

    def _read_jsonl(path):
        out = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append(json.loads(line))
        return out

    births = _read_jsonl(cell_dir / "births.jsonl")
    blocks = _read_jsonl(cell_dir / "blocks.jsonl")
    edges = H.EdgeLog.from_jsonl(cell_dir / "heredity.jsonl")

    return {"genotypes": genotypes, "summary": summary, "births": births,
           "blocks": blocks, "edges": edges}


def death_gen_index(summary: dict) -> dict:
    out = {}
    for rep in summary.get("generations", []):
        gen = rep["generation"]
        for d in rep.get("deaths", []):
            cid = d["complex_id"]
            if cid not in out:
                out[cid] = gen
    return out


def _traces_dir(cell_dir: Path, exp_id: str) -> Path:
    return cell_dir / "traces" / exp_id


def _load_trace_lines(cell_dir: Path, exp_id: str, cid: str) -> list:
    path = _traces_dir(cell_dir, exp_id) / f"{cid}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def compute_block_a1_a5(cell_dir: Path, genotypes: dict, summary: dict, births: list,
                        blocks: list) -> dict:
    """PROTOCOL.md §4.3. `blocks` -- строки `blocks.jsonl`
    (`{"molecule_id","slot","signature","registered_gen",...}`).

    ВАЖНО (найдено смоук-тестом одной клетки, см. BLOCKERS.md): `arch2.
    evolve.Evolution.run_generation` САМ регистрирует композиты из ЦЕЛЫХ
    элитных генотипов (`_try_register_molecule`, часть `_heredity_pass`,
    безусловно активна и в WITH, и в CTRL -- НЕ отключается флагом
    `enable_blocks`, это встроенный механизм arch2, не мой). Такие
    композиты ТОЖЕ имеют префикс `cx.` -- `H.uses_composite(g)` без
    фильтра ловит ИХ тоже, не только мои блоки. Поэтому A2/A4 здесь
    фильтруют СТРОГО по `block_ids` (моё множество, из `blocks.jsonl`),
    а не по любому composite-CALL."""
    exp_id = summary["experiment_id"]

    # A1: сколько блоков зарегистрировано (МОИХ, не arch2's собственных
    # whole-genotype композитов)
    n_blocks = len(blocks)
    block_ids = {b["molecule_id"] for b in blocks}

    def _uses_my_block(g: dict) -> bool:
        return any(c["molecule"] in block_ids for c in H.composite_calls(g))

    # A2: потомки (gen>=1) с CALL на ЗАРЕГИСТРИРОВАННЫЙ МНОЙ блок
    users = [cid for cid, g in genotypes.items()
            if g.get("gen", 0) >= 1 and _uses_my_block(g)]
    n_a2 = len(users)

    # A3: из них -- дошли до элиты ИЛИ побили планку хотя бы раз
    elite_ever = set()
    beat_gate_ever = set()
    for rep in summary.get("generations", []):
        elite_ever |= set(rep.get("elite", []))
        for cid, d in (rep.get("vs_gate") or {}).items():
            if d.get("beats_gate"):
                beat_gate_ever.add(cid)
    a3_hits = [cid for cid in users if cid in elite_ever or cid in beat_gate_ever]
    n_a3 = len(a3_hits)

    # A4: доля task-оценок block-несущих генотипов, где block-CALL реально
    # исполнился (CALL_COMPOSITE появился в трассе) -- бюджет/STOP МОГУТ
    # оборвать прогон ДО слота с блоком, это и измеряется.
    reached, total = 0, 0
    for cid in users:
        lines = _load_trace_lines(cell_dir, exp_id, cid)
        for rec in lines:
            total += 1
            trace = rec.get("trace") or []
            if any(el.get("op") == "CALL_COMPOSITE" and el.get("molecule") in block_ids
                  for el in trace):
                reached += 1
    a4_rate = (reached / total) if total else None

    # A5: merge_events=0 и архив монотонен (унаследованный инвариант arch2)
    merge_events = summary.get("merge_events", 0)
    sizes = summary.get("archive_sizes_by_gen", [])
    monotonic = all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1)) if sizes else True
    a5_ok = (merge_events == 0) and monotonic

    passes = {
        "A1": n_blocks >= 1, "A2": n_a2 >= 1, "A3": n_a3 >= 1,
        "A4": (a4_rate is not None and a4_rate == 1.0),
        "A5": a5_ok,
    }
    all_pass = all(passes.values())

    return {
        "N_blocks_registered": n_blocks, "block_ids": sorted(block_ids),
        "N_A2_users": n_a2, "N_A3_elite_or_beat_gate": n_a3,
        "A4_execution_rate": a4_rate, "A4_reached": reached, "A4_total": total,
        "merge_events": merge_events, "archive_monotonic": monotonic,
        "passes": passes, "all_A1_A5_pass": all_pass,
    }


def audit_cell(mode: str, seed: int, cell_dir: Path, ds, panel: list, best_single: dict,
              verbose: bool = True) -> dict:
    """PROTOCOL.md §7. `mode` ∈ {"WITH","CTRL"}. `births.jsonl` логирует
    КАЖДУЮ попытку размножения (`admitted` True/False)."""
    art = load_cell_artifacts(cell_dir)
    genotypes, summary, births, blocks, edges = (art["genotypes"], art["summary"],
                                                 art["births"], art["blocks"], art["edges"])
    last_gen = summary["n_generations"] - 1
    deaths_at = death_gen_index(summary)

    imp_cache: dict = {}

    def imp(cid: str) -> frozenset:
        if cid not in imp_cache:
            g = genotypes.get(cid)
            imp_cache[cid] = imp_set(ds, panel, best_single, g) if g else frozenset()
        return imp_cache[cid]

    fmt_violations = [cid for cid in genotypes if 1 in imp(cid)]

    seed_cids = [cid for cid, g in genotypes.items() if g.get("gen", 0) == 0]
    seed_ab = [cid for cid in seed_cids if len(imp(cid)) >= 2]

    admitted_births = [b for b in births if b.get("admitted") and b.get("parent_id")
                       and b.get("child_id") in genotypes]

    a_events, first_assembly = [], []
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

    gens_with_overlap, n_gens_checked = 0, 0
    for rep in summary.get("generations", []):
        pop = rep.get("population", [])
        distinct_nonempty = {imp(cid) for cid in pop if cid in genotypes and imp(cid)}
        n_gens_checked += 1
        if len(distinct_nonempty) >= 2:
            gens_with_overlap += 1
    overlap = (gens_with_overlap / n_gens_checked) if n_gens_checked else None

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

    transfer_success = [b for b in births if b.get("operator") == "TRANSFER_SLOT"]
    slot_match_rate = (sum(1 for b in transfer_success if b.get("complementary")) /
                       len(transfer_success)) if transfer_success else None

    hgt_attempts = [b for b in births if b.get("hgt_attempted")]
    hgt_successes = [b for b in hgt_attempts if b.get("operator") == "TRANSFER_SLOT"]
    transfer_success_rate = (len(hgt_successes) / len(hgt_attempts)) if hgt_attempts else None

    block_attempts = [b for b in births if b.get("block_insert_attempted")]
    block_successes = [b for b in block_attempts if b.get("operator") == "BLOCK_INSERT"]
    block_success_rate = (len(block_successes) / len(block_attempts)) if block_attempts else None

    elite_ever = set()
    for rep in summary.get("generations", []):
        elite_ever |= set(rep.get("elite", []))
    elite_ab = sum(1 for cid in elite_ever if cid in genotypes and len(imp(cid)) >= 2)
    elite_ab_share = (elite_ab / len(elite_ever)) if elite_ever else None

    a1_a5 = compute_block_a1_a5(cell_dir, genotypes, summary, births, blocks) if mode == "WITH" else None

    result = {
        "mode": mode, "seed": seed, "n_generations": summary["n_generations"],
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
        "n_block_insert_attempts": len(block_attempts), "n_block_insert_success": len(block_successes),
        "block_success_rate": block_success_rate,
        "elite_ever_size": len(elite_ever), "elite_AB_share": elite_ab_share,
        "delta_impset_histogram": delta_hist,
        "block_a1_a5": a1_a5,
        "_l_a_values": lifespans, "_b_flags": b_flags,
    }
    if verbose:
        extra = ""
        if a1_a5:
            extra = (f" A1-A5={a1_a5['passes']} N_blocks={a1_a5['N_blocks_registered']} "
                    f"A4_rate={a1_a5['A4_execution_rate']}")
        print(f"[audit] mode={mode} seed={seed}: N_A={n_a} N_AB_evolved={n_ab_evolved} "
              f"N_AB_first_assembly={n_ab_first_assembly} loss:gain(overall)={loss_gain_overall:.2f} "
              f"slot_match={slot_match_rate} elite_AB_share={elite_ab_share}{extra}", flush=True)
    if fmt_violations:
        print(f"[audit] !!! FORMAT Imp violation at mode={mode} seed={seed}: {fmt_violations[:5]}",
              flush=True)
    return result
