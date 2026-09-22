"""metrics_lib.py — Imp core (copy of `agent_life6_retention/src/
metrics_lib.py`, unchanged: unfold-aware `slot_models`, `imp_set`,
`covered_set`, `best_single_per_slot`, `build_audit_registry`,
`load_cell_artifacts`, `death_gen_index`, `process_run`, `aggregate`,
`imp_set_from_records`, `first_assembly_and_loss_gain`) + LIFE-7
PROTOCOL.md §2 additions:

- `per_cell_slot_metrics` extended with `mean_L_imp`/`p90_L_imp`
  alongside the existing `median_L_imp` -- PROTOCOL.md is explicit that
  median is SECONDARY here, never the sole pass/fail signal (LIFE-6
  §5.1's lesson: a lever can raise mean/tail without moving the median).
- NEW `per_generation_occupancy`: `occ_imp`/`frac_gen_AB_alive` computed
  from `summary["generations"][i]["survivors"]` -- the POST-this-
  generation alive set (confirmed by direct read of `arch2/evolve.py`
  this session: `report["population"]` is captured BEFORE that
  generation's own death is applied, i.e. it's the INCOMING population;
  `report["survivors"]` is the twice-D1/D2-filtered set that actually
  becomes `self.population` for the next generation -- using
  `"population"` here would be one generation stale).
- NEW `extinct_gen`/`any_gate_win`: `extinct_gen` reads the TOP-LEVEL
  `summary["extinct"]`/`summary["n_generations"]` only -- NEVER
  `summary["generations"][i]["stall_count"/"extinct"]`, which (confirmed
  this session) keeps showing arch2's original gate-based verdict for
  that generation forever, even under LIFE-7's A2 stall-from-improvement
  override (the override only updates the LIVE `ev.stall_count`/
  `ev.extinct` attributes post-hoc, not the already-serialized per-
  generation report copy). `any_gate_win` is informational only (gate is
  not this package's selection criterion) -- whether ANY generation ever
  saw `any_complex_beats_gate=True`.
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
RETENTION_SLOTS = (0, 2, 3)   # READ, LOOKUP, COMPUTE -- FORMAT excluded (Imp always False)


def slot_models(slot_node: dict, registry, _depth: int = 0, _seen: frozenset = frozenset()) -> list:
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
    blocks = _read_jsonl(cell_dir / "blocks.jsonl")  # always [] -- no registry in LIFE-8 either (P5)
    edges = H.EdgeLog.from_jsonl(cell_dir / "heredity.jsonl")

    return {"genotypes": genotypes, "summary": summary, "births": births,
           "blocks": blocks, "edges": edges}


class _AuditRegistry:
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


def _percentile(values: list, p: float):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def per_cell_slot_metrics(run_result: dict, min_gen: int = 1) -> dict:
    """`median_L_imp` kept as SECONDARY (PROTOCOL.md §2) -- `mean_L_imp`/
    `p90_L_imp` are primary alongside `occ_imp` (see
    `per_generation_occupancy`), pooled across ALL Imp=1 signatures on a
    slot (not grouped by signature)."""
    out = {}
    for s in range(len(STEP_KINDS)):
        imp_records = [r for r in run_result["records"]
                      if r["slot"] == s and r["gen"] >= min_gen and r["imp"]]
        lifespans = [max(0, r["death_gen"] - r["birth_gen"]) for r in imp_records]
        n_imp_sig = len({r["signature"] for r in imp_records})
        out[s] = {
            "median_L_imp": statistics.median(lifespans) if lifespans else None,
            "mean_L_imp": statistics.mean(lifespans) if lifespans else None,
            "p90_L_imp": _percentile(lifespans, 0.9),
            "frac_L_ge3": (sum(1 for l in lifespans if l >= 3) / len(lifespans)
                          if lifespans else None),
            "n_imp_sig": n_imp_sig,
            "n_imp_carriers": len(lifespans),
        }
    return out


def imp_set_from_records(run_result: dict) -> dict:
    out: dict = {}
    for r in run_result["records"]:
        out.setdefault(r["complex_id"], set())
        if r["imp"]:
            out[r["complex_id"]].add(r["slot"])
    return {cid: frozenset(s) for cid, s in out.items()}


def first_assembly_and_loss_gain(run_result: dict) -> dict:
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


# ---------------------------------------------------------------------
# NEW for LIFE-7: per-generation occupancy, using `survivors` (the
# POST-this-generation alive set), never `population` (the INCOMING one)
# -- see module docstring for why this distinction matters.
# ---------------------------------------------------------------------

def per_generation_occupancy(cell_dir: Path, ds, panel: list, best_single: dict) -> dict:
    art = load_cell_artifacts(cell_dir)
    genotypes, summary, blocks = art["genotypes"], art["summary"], art["blocks"]
    registry = build_audit_registry(genotypes, blocks)
    # `summary()` (arch2/evolve.py:776-808) has NO top-level "references"
    # field at all (only "final_population"/"final_elite"/"final_shadow")
    # -- confirmed by direct read this session. Per-generation
    # `rep["references"]` (arch2/evolve.py:462-463) IS a dict keyed by
    # reference complex_id -> score info; its key set is stable across
    # every generation (references never change), so the first
    # generation's keys suffice. Getting this wrong would silently
    # over-count occ_imp (a permanent reference carrying Imp=1 forever
    # would make that slot's occupancy trivially 1.0 regardless of the
    # actual evolving population).
    generations_list = summary.get("generations") or []
    references = set(generations_list[0].get("references", {}).keys()) if generations_list else set()

    imp_cache: dict = {}

    def imp_of_genotype(g: dict) -> frozenset:
        cid = g.get("complex_id")
        if cid is not None and cid in imp_cache:
            return imp_cache[cid]
        root = g.get("root", {})
        children = root.get("children") or []
        out = set()
        for s in range(min(len(children), len(STEP_KINDS))):
            models = slot_models(children[s], registry)
            if not models:
                continue
            cov = covered_set(ds, panel, s, models)
            if len(cov) > best_single[s]:
                out.add(s)
        val = frozenset(out)
        if cid is not None:
            imp_cache[cid] = val
        return val

    generations = summary.get("generations", [])
    n_gens = len(generations)
    occ_hits = {s: 0 for s in RETENTION_SLOTS}
    ab_hits = 0
    any_gate_win = False

    for rep in generations:
        survivors = [cid for cid in (rep.get("survivors") or []) if cid not in references]
        slot_hits_this_gen = set()
        ab_this_gen = False
        for cid in survivors:
            g = genotypes.get(cid)
            if g is None:
                continue
            imp = imp_of_genotype(g)
            slot_hits_this_gen |= imp
            if len(imp) >= 2:
                ab_this_gen = True
        for s in RETENTION_SLOTS:
            if s in slot_hits_this_gen:
                occ_hits[s] += 1
        if ab_this_gen:
            ab_hits += 1
        if rep.get("any_complex_beats_gate"):
            any_gate_win = True

    occ_imp = {s: (occ_hits[s] / n_gens if n_gens else None) for s in RETENTION_SLOTS}
    frac_gen_ab_alive = (ab_hits / n_gens) if n_gens else None
    extinct_gen = summary["n_generations"] - 1   # top-level field only, see module docstring

    return {"occ_imp": occ_imp, "frac_gen_AB_alive": frac_gen_ab_alive,
           "extinct_gen": extinct_gen, "extinct": summary["extinct"],
           "any_gate_win": any_gate_win, "n_generations": n_gens}
