"""metrics_lib.py — Imp core (copy of `agent_life5_slot_map/src/
unfold_metrics.py`, unchanged: unfold-aware `slot_models`, `imp_set`,
`covered_set`, `best_single_per_slot`, `build_audit_registry`,
`load_cell_artifacts`, `death_gen_index`) + `agent_life5_slot_map/src/
slot_map.py`'s exact `process_run()`/`aggregate()` (per-genotype-per-slot
record building; `aggregate()` already accepts a LIST of run-results, so
calling it with a 1-element list gives per-cell aggregation for free --
no new pooling logic needed, see PROTOCOL.md) + ONE new function this
package needs, `per_cell_slot_metrics`, plus the same `N_AB_first_
assembly`/`loss:gain(overall)` formula every LIFE-1..5 package has used
(no A1-A5/block logic at all -- this package never touches block
registry, see PROTOCOL.md §0).
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
H = LD.H

STEP_KINDS = HSTEP.STEP_KINDS
MAX_UNFOLD_DEPTH = 6


def slot_models(slot_node: dict, registry, _depth: int = 0, _seen: frozenset = frozenset()) -> list:
    """`CALL(cx.*)` разворачивается в `registry.get(mid)["genotype"]
    ["root"]` и обходится рекурсивно (весь `root`, как исполнитель в
    `arch2/runner.py`). Не используется в LIFE-6 evolve (нет block
    registry вообще -- см. PROTOCOL.md §0), но нужна для Imp/сигнатур
    (в этом пуле `cx.`-CALL просто никогда не встретится, unfold --
    защитный код, не активный путь)."""
    out: list = []

    def walk(n, depth, seen):
        if not isinstance(n, dict):
            return
        op = n.get("op")
        if op == "CALL":
            mol = n.get("molecule", "")
            if mol.startswith("gen."):
                out.append(mol[4:])
            elif mol.startswith("cx.") and depth < MAX_UNFOLD_DEPTH and mol not in seen:
                entry = registry.get(mol) if registry is not None else None
                inner_root = (entry or {}).get("genotype", {}).get("root")
                if inner_root is not None:
                    walk(inner_root, depth + 1, seen | {mol})
        elif op in ("SEQ", "PAR", "ASSEMBLE"):
            for c in n.get("children") or []:
                walk(c, depth, seen)
        elif op == "SWITCH":
            for c in (n.get("cases") or {}).values():
                walk(c, depth, seen)
            if "default" in n:
                walk(n["default"], depth, seen)
        elif op == "BUDGET":
            if "child" in n:
                walk(n["child"], depth, seen)

    walk(slot_node, _depth, _seen)
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


def imp_set(ds, panel: list, best_single: dict, genotype: dict, registry) -> frozenset:
    root = genotype.get("root", {})
    children = root.get("children") or []
    out = set()
    for s in range(min(len(children), len(STEP_KINDS))):
        models = slot_models(children[s], registry)
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
    blocks = _read_jsonl(cell_dir / "blocks.jsonl")  # always [] -- no registry in LIFE-6
    edges = H.EdgeLog.from_jsonl(cell_dir / "heredity.jsonl")

    return {"genotypes": genotypes, "summary": summary, "births": births,
           "blocks": blocks, "edges": edges}


class _AuditRegistry:
    """Read-only резолвер поверх archive.json+blocks.jsonl ОДНОЙ клетки.
    Копия `agent_life5_slot_map/src/unfold_metrics.py::_AuditRegistry`,
    без изменений -- `blocks` всегда `[]` здесь (нет registry в LIFE-6),
    так что на практике этот класс просто разворачивает СВОИ ЖЕ
    genotypes как `cx.<complex_id>` (для полноты API, не активный путь)."""

    def __init__(self, genotypes: dict, blocks: list):
        self._molecules: dict = {}
        for cid, g in genotypes.items():
            self._molecules[f"cx.{cid}"] = {"genotype": g}
        for b in blocks:
            carrier = genotypes.get(b.get("carrier_cid"))
            if carrier is None:
                continue
            children = (carrier.get("root") or {}).get("children") or []
            slot = b.get("slot")
            if slot is None or slot >= len(children):
                continue
            self._molecules[b["molecule_id"]] = {"genotype": {"root": children[slot]}}

    def get(self, mid: str):
        return self._molecules.get(mid)


def build_audit_registry(genotypes: dict, blocks: list) -> _AuditRegistry:
    return _AuditRegistry(genotypes, blocks)


def death_gen_index(summary: dict) -> dict:
    out = {}
    for rep in summary.get("generations", []):
        gen = rep["generation"]
        for d in rep.get("deaths", []):
            cid = d["complex_id"]
            if cid not in out:
                out[cid] = gen
    return out


# ---------------------------------------------------------------------
# agent_life5_slot_map/src/slot_map.py's process_run/aggregate, copied
# unchanged (per-genotype-per-slot record building; aggregate() already
# takes a LIST of run-results, so aggregate([process_run(...)], min_gen=1)
# gives per-CELL aggregation -- exactly what LIFE-6 needs, no new pooling
# logic required).
# ---------------------------------------------------------------------

def process_run(cell_dir: Path, run_id: str, ds, panel: list, imp_cache: dict, best_single: dict) -> dict:
    art = load_cell_artifacts(cell_dir)
    genotypes, summary, births, blocks = art["genotypes"], art["summary"], art["births"], art["blocks"]
    deaths_at = death_gen_index(summary)
    last_gen = summary["n_generations"] - 1
    registry = build_audit_registry(genotypes, blocks)

    def imp_of(s: int, models_key: tuple) -> bool:
        key = (s, models_key)
        if key in imp_cache:
            return imp_cache[key]
        cov = covered_set(ds, panel, s, list(models_key))
        val = len(cov) > best_single[s]
        imp_cache[key] = val
        return val

    records = []
    for cid, g in genotypes.items():
        gen = g.get("gen", 0)
        children = (g.get("root") or {}).get("children") or []
        for s in range(min(len(children), len(STEP_KINDS))):
            models = slot_models(children[s], registry)
            sig_models = tuple(sorted(set(models)))
            op = children[s].get("op")
            imp = imp_of(s, sig_models) if sig_models else False
            death_gen = deaths_at.get(cid, last_gen)
            records.append({
                "run_id": run_id, "complex_id": cid, "gen": gen, "slot": s, "op": op,
                "signature": (op, sig_models), "imp": imp,
                "birth_gen": gen, "death_gen": death_gen,
            })

    imp_lookup = {(r["complex_id"], r["slot"]): r["imp"] for r in records}
    delta_tally = {s: {"gain": 0, "loss": 0, "same": 0} for s in range(len(STEP_KINDS))}
    for b in births:
        if not b.get("admitted"):
            continue
        child_id, parent_id = b.get("child_id"), b.get("parent_id")
        if child_id not in genotypes or parent_id not in genotypes:
            continue
        for s in range(len(STEP_KINDS)):
            imp_c = imp_lookup.get((child_id, s), False)
            imp_p = imp_lookup.get((parent_id, s), False)
            if imp_c and not imp_p:
                delta_tally[s]["gain"] += 1
            elif imp_p and not imp_c:
                delta_tally[s]["loss"] += 1
            else:
                delta_tally[s]["same"] += 1

    return {"run_id": run_id, "records": records, "births": births,
           "genotypes": genotypes, "delta_tally": delta_tally, "summary": summary}


def aggregate(all_run_results: list, min_gen: int) -> dict:
    """min_gen=0 -> every archive member; min_gen=1 -> evolved only.
    Called with a single-element list -> per-cell aggregation."""
    per_slot: dict = {s: {} for s in range(len(STEP_KINDS))}
    for run_res in all_run_results:
        for r in run_res["records"]:
            if r["gen"] < min_gen:
                continue
            s = r["slot"]
            sig = r["signature"]
            bucket = per_slot[s].setdefault(sig, {"imp": r["imp"], "carriers": set(),
                                                   "seeds": set(), "lifespans": []})
            assert bucket["imp"] == r["imp"], (
                f"Imp inconsistent for slot={s} signature={sig}: "
                f"{bucket['imp']} vs {r['imp']}")
            bucket["carriers"].add(r["complex_id"])
            bucket["seeds"].add(r["run_id"])
            bucket["lifespans"].append(max(0, r["death_gen"] - r["birth_gen"]))

    out = {}
    for s, sigs in per_slot.items():
        n_imp = sum(1 for b in sigs.values() if b["imp"])
        out[s] = {"n_distinct_signatures": len(sigs), "n_signatures_with_imp": n_imp,
                  "signatures": sigs}
    return out


# ---------------------------------------------------------------------
# NEW for LIFE-6 (PROTOCOL.md §2 primary metrics): per-cell, per-slot
# retention metrics -- pooled ACROSS all Imp=1 signatures on that slot
# (not grouped by signature like LIFE-5's summarize_imp_signatures),
# since the question here is "how long do Imp=1 carriers survive on
# this slot overall," not "which specific signature is most common."
# ---------------------------------------------------------------------

def per_cell_slot_metrics(run_result: dict, min_gen: int = 1) -> dict:
    out = {}
    for s in range(len(STEP_KINDS)):
        imp_records = [r for r in run_result["records"]
                      if r["slot"] == s and r["gen"] >= min_gen and r["imp"]]
        lifespans = [max(0, r["death_gen"] - r["birth_gen"]) for r in imp_records]
        n_imp_sig = len({r["signature"] for r in imp_records})
        out[s] = {
            "median_L_imp": statistics.median(lifespans) if lifespans else None,
            "frac_L_ge3": (sum(1 for l in lifespans if l >= 3) / len(lifespans)
                          if lifespans else None),
            "n_imp_sig": n_imp_sig,
            "n_imp_carriers": len(lifespans),
        }
    return out


def imp_set_from_records(run_result: dict) -> dict:
    """{complex_id: frozenset(slots where Imp=1)} -- built directly from
    `process_run`'s per-genotype-per-slot records (no re-computation)."""
    out: dict = {}
    for r in run_result["records"]:
        out.setdefault(r["complex_id"], set())
        if r["imp"]:
            out[r["complex_id"]].add(r["slot"])
    return {cid: frozenset(s) for cid, s in out.items()}


def first_assembly_and_loss_gain(run_result: dict) -> dict:
    """Same formula every LIFE-1..5 package has used: `N_AB_first_
    assembly` = admitted births where parent had |ImpSet|<2 and child
    has |ImpSet|>=2 (assembled, not inherited); `loss:gain(overall)` =
    losses/max(1,gains) over Δ|ImpSet| across ALL admitted births. No
    A1-A5/block logic -- this package has no block registry (PROTOCOL.md §0)."""
    imp_by_cid = imp_set_from_records(run_result)
    genotypes = run_result["genotypes"]
    first_assembly = 0
    gains = losses = 0
    for b in run_result["births"]:
        if not b.get("admitted"):
            continue
        child_id, parent_id = b.get("child_id"), b.get("parent_id")
        if child_id not in genotypes or parent_id not in genotypes:
            continue
        imp_child = imp_by_cid.get(child_id, frozenset())
        imp_parent = imp_by_cid.get(parent_id, frozenset())
        d = len(imp_child) - len(imp_parent)
        if d > 0:
            gains += 1
        elif d < 0:
            losses += 1
        if len(imp_parent) < 2 and len(imp_child) >= 2:
            first_assembly += 1
    return {"N_AB_first_assembly": first_assembly,
           "loss_gain_overall": losses / max(1, gains), "gains": gains, "losses": losses}
