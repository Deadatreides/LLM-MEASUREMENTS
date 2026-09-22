"""slot_map.py — Phase B (mandatory, primary science): walks archives and
builds per-slot Imp-signature tables. Two independent pools, NEVER blended
(PROTOCOL.md/plan): the mandatory 24 existing `agent_life3_block_live` +
`agent_life4_block_fix` runs, and (if Phase B.2 ran) the 4 new
`life5_default_s*` runs, reported as a separate cross-check.

`Imp(slot, signature)` is a pure function of `(slot, models)` given a FIXED
panel/grid (confirmed byte-identical across all 24+4 runs this session) --
computed ONCE per distinct `(slot, signature)` via a shared cache, not
redundantly per genotype/run. A signature's Imp value must never differ
between two occurrences -- checked with an assertion (would indicate the
panel-identity assumption broke, or a cache bug).
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD        # noqa: E402
import unfold_metrics as ML      # noqa: E402

STEP_KINDS = LD.HSTEP.STEP_KINDS   # ("READ","FORMAT","LOOKUP","COMPUTE")
FORMAT_SLOT = STEP_KINDS.index("FORMAT")

ROOT = LD.ROOT
LIFE3_RUNS = ROOT / "agent_life3_block_live" / "runs"
LIFE4_RUNS = ROOT / "agent_life4_block_fix" / "runs"

RUN_DIR_RE = re.compile(r"^life[34]_(with|ctrl)_s(\d+)$")


def discover_mandatory_runs() -> list:
    """(package, mode, seed, path) for all existing life3/life4 runs --
    globbed + regex-filtered, not hardcoded seed lists (robust against
    stray dirs like life4's `smoke_throwaway`)."""
    out = []
    for base, pkg in ((LIFE3_RUNS, "life3"), (LIFE4_RUNS, "life4")):
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            m = RUN_DIR_RE.match(d.name)
            if m and d.is_dir():
                mode, seed = m.group(1), int(m.group(2))
                out.append({"package": pkg, "mode": mode, "seed": seed,
                           "run_id": f"{pkg}_{mode}_s{seed}", "path": d})
    return out


def discover_optional_runs(runs_dir: Path) -> list:
    """The 4 (or however many exist) `life5_default_s*` runs from Phase B.2."""
    out = []
    if not runs_dir.exists():
        return out
    for d in sorted(runs_dir.iterdir()):
        m = re.match(r"^life5_default_s(\d+)$", d.name)
        if m and d.is_dir() and (d / "archive.json").exists():
            seed = int(m.group(1))
            out.append({"package": "life5", "mode": "default", "seed": seed,
                       "run_id": f"life5_default_s{seed}", "path": d})
    return out


def process_run(run_info: dict, ds, panel: list, imp_cache: dict, best_single: dict) -> dict:
    path = run_info["path"]
    art = ML.load_cell_artifacts(path)
    genotypes, summary, births, blocks = art["genotypes"], art["summary"], art["births"], art["blocks"]
    deaths_at = ML.death_gen_index(summary)
    last_gen = summary["n_generations"] - 1
    registry = ML.build_audit_registry(genotypes, blocks)
    run_id = run_info["run_id"]

    def imp_of(s: int, models_key: tuple) -> bool:
        key = (s, models_key)
        if key in imp_cache:
            return imp_cache[key]
        cov = ML.covered_set(ds, panel, s, list(models_key))
        val = len(cov) > best_single[s]
        imp_cache[key] = val
        return val

    records = []
    for cid, g in genotypes.items():
        gen = g.get("gen", 0)
        children = (g.get("root") or {}).get("children") or []
        for s in range(min(len(children), len(STEP_KINDS))):
            models = ML.slot_models(children[s], registry)
            sig_models = tuple(sorted(set(models)))
            op = children[s].get("op")
            imp = imp_of(s, sig_models) if sig_models else False
            death_gen = deaths_at.get(cid, last_gen)
            records.append({
                "run_id": run_id, "package": run_info["package"], "complex_id": cid,
                "gen": gen, "slot": s, "op": op, "signature": (op, sig_models),
                "imp": imp, "birth_gen": gen, "death_gen": death_gen,
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

    return {"run_id": run_id, "records": records, "delta_tally": delta_tally}


def format_sanity_check(all_run_results: list) -> tuple:
    """Imp(FORMAT) must be False everywhere -- one model saturates FORMAT
    80/80, structurally impossible to beat (LIFE-1..4 invariant)."""
    violations = []
    for run_res in all_run_results:
        for r in run_res["records"]:
            if r["slot"] == FORMAT_SLOT and r["imp"]:
                violations.append((run_res["run_id"], r["complex_id"]))
    return (len(violations) == 0), violations


def aggregate(all_run_results: list, min_gen: int) -> dict:
    """min_gen=0 -> every archive member; min_gen=1 -> evolved only.
    Returns {slot_index: {"signatures": {sig: bucket}, "n_distinct": int,
    "n_imp": int}}. `bucket` = {"imp","carriers"(set complex_id, deduped
    globally -- same complex_id in >1 run is genuinely one genotype, see
    module docstring),"seeds"(set run_id),"lifespans"(list, one entry per
    occurrence, NOT deduped -- lifespan is per-run)}."""
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
                f"{bucket['imp']} vs {r['imp']} (panel-identity assumption broke?)")
            bucket["carriers"].add(r["complex_id"])
            bucket["seeds"].add(r["run_id"])
            bucket["lifespans"].append(max(0, r["death_gen"] - r["birth_gen"]))

    out = {}
    for s, sigs in per_slot.items():
        n_imp = sum(1 for b in sigs.values() if b["imp"])
        out[s] = {"n_distinct_signatures": len(sigs), "n_signatures_with_imp": n_imp,
                  "signatures": sigs}
    return out


def summarize_imp_signatures(agg_slot: dict) -> list:
    """-> sorted list (desc by n_carriers) of Imp=1 signature summaries,
    for the report's per-slot table / top-5."""
    out = []
    for sig, bucket in agg_slot["signatures"].items():
        if not bucket["imp"]:
            continue
        lifespans = bucket["lifespans"]
        out.append({
            "op": sig[0], "models": list(sig[1]),
            "n_carriers": len(bucket["carriers"]), "n_seed_appeared": len(bucket["seeds"]),
            "seeds": sorted(bucket["seeds"]),
            "median_L": statistics.median(lifespans) if lifespans else None,
            "max_L": max(lifespans) if lifespans else None,
        })
    out.sort(key=lambda d: (-d["n_carriers"], -d["n_seed_appeared"]))
    return out


def build_full_map(runs: list, ds, panel: list, best_single: dict) -> dict:
    imp_cache: dict = {}
    all_run_results = [process_run(r, ds, panel, imp_cache, best_single) for r in runs]
    fmt_ok, fmt_violations = format_sanity_check(all_run_results)
    agg_gen0 = aggregate(all_run_results, min_gen=0)
    agg_gen1 = aggregate(all_run_results, min_gen=1)
    delta_pooled = {s: {"gain": 0, "loss": 0, "same": 0} for s in range(len(STEP_KINDS))}
    for run_res in all_run_results:
        for s, d in run_res["delta_tally"].items():
            for k in ("gain", "loss", "same"):
                delta_pooled[s][k] += d[k]
    return {
        "n_runs": len(runs), "run_ids": [r["run_id"] for r in runs],
        "format_sanity_ok": fmt_ok, "format_sanity_violations": fmt_violations,
        "agg_gen0": agg_gen0, "agg_gen1": agg_gen1, "delta_tally_pooled": delta_pooled,
        "per_run": all_run_results,
    }
