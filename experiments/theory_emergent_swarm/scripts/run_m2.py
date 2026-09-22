"""run_m2.py — M2: M1 with its three design defects fixed (PROTOCOL_M2.md).
Seam 1 is violable, the final answer is a SET (no arithmetic), and per-hop
accuracy is the PRIMARY metric.
Thresholds pre-registered in PROTOCOL_M1.md SS3.

МДП repairs cost ZERO extra calls: candidates are already collected from
all six models, and the check merely re-ranks them by seam validity. So
МДП cannot win by spending more budget.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_m2 as M1     # noqa: E402
import oracles as OR               # noqa: E402
import model_registry_11 as MR     # noqa: E402
import call_log                    # noqa: E402

METRICS = Path(__file__).resolve().parents[1] / "metrics"
TEMP = 0.0
SEED = 5150
MAX_ID, MAX_LIST, MAX_WHOLE = 16, 120, 200
_DEP = re.compile(r"DEP-\d{4}")
_MGR = re.compile(r"MGR-\d{4}")

_cache: dict = {}
_calls = 0


def ask(llm, model_id, key, prompt, max_tokens):
    global _calls
    if key in _cache:
        return _cache[key]
    r = MR.generate(llm, model_id, prompt, temperature=TEMP, top_p=1.0, seed=SEED,
                    max_tokens=max_tokens)
    txt = r.get("raw_text", "")
    _cache[key] = txt
    _calls += 1
    call_log.log_filter_call(model=model_id, task_id=str(key[0]), family=f"M2-{key[1]}",
                            seed=SEED, n_in=r.get("input_tokens") or 0,
                            n_out=r.get("output_tokens") or 0, ms=0.0,
                            failed=bool(r.get("generation_failed")), extracted=[], raw_text=txt)
    return txt


def first(rx, text):
    m = rx.search(text or "")
    return m.group(0) if m else None


def tally(vals):
    """-> [(value, votes)] sorted by votes desc, deterministic tie-break by value."""
    c = defaultdict(int)
    for v in vals:
        if v is not None:
            c[v] = c[v] + 1
    return sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    models = list(MR.MODEL_IDS)
    d = M1.build_tasks()
    tb, ids = d["tasks"], d["test"]
    print(f"M1: {len(ids)} tasks x {len(models)} models, 4 arms\n", flush=True)

    hop1, hop2, hop3, whole = {}, {}, {}, {}
    t0 = time.time()
    for mi, model_id in enumerate(models):
        llm, _ = MR.load_model(model_id)
        for tid in ids:
            task = tb[tid]
            whole.setdefault(tid, {})[model_id] = OR.extract_ids(
                ask(llm, model_id, (tid, "whole", model_id), M1.prompt_whole(task), MAX_WHOLE))
            hop1.setdefault(tid, {})[model_id] = first(
                _DEP, ask(llm, model_id, (tid, "h1", model_id), M1.prompt_hop1(task), MAX_ID))
        # hop2/hop3 need inputs that depend on hop1/hop2 across arms -> second pass below
        del llm
        print(f"  [{mi+1}/{len(models)}] {model_id} pass-1 done ({time.time()-t0:.0f}s)", flush=True)

    # decide which hop-2 inputs are needed: each model's own + the swarm consensus
    need2 = {tid: set() for tid in ids}
    for tid in ids:
        for m in models:
            if hop1[tid][m]:
                need2[tid].add(hop1[tid][m])
        cons = tally([hop1[tid][m] for m in models])
        if cons:
            need2[tid].add(cons[0][0])
            # МДП-repaired hop1 consensus may differ -> include it too
            for v, _n in cons:
                if M1.mdp_seam1(tb[tid], v):
                    need2[tid].add(v)
                    break

    for mi, model_id in enumerate(models):
        llm, _ = MR.load_model(model_id)
        for tid in ids:
            task = tb[tid]
            for dep in need2[tid]:
                hop2.setdefault(tid, {}).setdefault(dep, {})[model_id] = first(
                    _MGR, ask(llm, model_id, (tid, "h2", dep, model_id),
                              M1.prompt_hop2(task, dep), MAX_ID))
        del llm
        print(f"  [{mi+1}/{len(models)}] {model_id} pass-2 done ({time.time()-t0:.0f}s)", flush=True)

    need3 = {tid: set() for tid in ids}
    for tid in ids:
        for dep in need2[tid]:
            for m in models:
                v = hop2[tid][dep][m]
                if v:
                    need3[tid].add(v)
    for mi, model_id in enumerate(models):
        llm, _ = MR.load_model(model_id)
        for tid in ids:
            task = tb[tid]
            for mgr in need3[tid]:
                hop3.setdefault(tid, {}).setdefault(mgr, {})[model_id] = OR.extract_ids(
                    ask(llm, model_id, (tid, "h3", mgr, model_id),
                        M1.prompt_hop3(task, mgr), MAX_LIST))
        del llm
        print(f"  [{mi+1}/{len(models)}] {model_id} pass-3 done ({time.time()-t0:.0f}s)", flush=True)

    # ------------------------------- scoring -------------------------------
    def summed(task, emp_ids):
        """ВДП only: drop ids that do not exist in block C. No mgr filtering."""
        return frozenset(i for i in (emp_ids or set()) if i in task["id_to_salary"])

    def ok(task, val):
        return bool(val) and set(val) == set(task["final_oracle"])

    res = {a: set() for a in ("WHOLE", "CHAIN_SINGLE", "CHAIN_SWARM", "CHAIN_SWARM_MDP")}
    per_model_whole = {m: set() for m in models}
    per_model_chain = {m: set() for m in models}
    mdp_fired = {"seam1": 0, "seam2": 0}
    detail = {}

    for tid in ids:
        task = tb[tid]
        for m in models:
            if ok(task, summed(task, whole[tid][m])):
                per_model_whole[m].add(tid)
            dep = hop1[tid][m]
            mgr = hop2[tid].get(dep, {}).get(m) if dep else None
            emps = hop3[tid].get(mgr, {}).get(m) if mgr else None
            if emps and ok(task, summed(task, emps)):
                per_model_chain[m].add(tid)

        # --- swarm, no МДП ---
        c1 = tally([hop1[tid][m] for m in models])
        dep_s = c1[0][0] if c1 else None
        c2 = tally([hop2[tid].get(dep_s, {}).get(m) for m in models]) if dep_s else []
        mgr_s = c2[0][0] if c2 else None
        emps_s = set()
        if mgr_s:
            votes = defaultdict(int)
            for m in models:
                for e in (hop3[tid].get(mgr_s, {}).get(m) or set()):
                    votes[e] += 1
            emps_s = {e for e, v in votes.items() if v * 2 > len(models)}
        if ok(task, summed(task, emps_s)):
            res["CHAIN_SWARM"].add(tid)

        # --- swarm + МДП (re-rank already-collected candidates, 0 new calls) ---
        dep_r = next((v for v, _n in c1 if M1.mdp_seam1(task, v)), None)
        if dep_r != dep_s:
            mdp_fired["seam1"] += 1
        c2r = tally([hop2[tid].get(dep_r, {}).get(m) for m in models]) if dep_r else []
        mgr_r = next((v for v, _n in c2r
                      if M1.mdp_seam2(task, v) and M1.mdp_seam2_consistency(task, dep_r, v)), None)
        if mgr_r != (c2r[0][0] if c2r else None):
            mdp_fired["seam2"] += 1
        emps_r = set()
        if mgr_r:
            votes = defaultdict(int)
            for m in models:
                for e in (hop3[tid].get(mgr_r, {}).get(m) or set()):
                    votes[e] += 1
            emps_r = {e for e, v in votes.items() if v * 2 > len(models)}
        if ok(task, summed(task, emps_r)):
            res["CHAIN_SWARM_MDP"].add(tid)

        detail[tid] = {"dep_swarm": dep_s, "dep_mdp": dep_r, "dep_gold": task["hop1_oracle"],
                      "mgr_swarm": mgr_s, "mgr_mdp": mgr_r, "mgr_gold": task["hop2_oracle"]}

    best_whole_m = max(models, key=lambda m: len(per_model_whole[m]))
    best_chain_m = max(models, key=lambda m: len(per_model_chain[m]))
    res["WHOLE"] = per_model_whole[best_whole_m]
    res["CHAIN_SINGLE"] = per_model_chain[best_chain_m]
    n = len(ids)
    r = {k: len(v) / n for k, v in res.items()}

    union_chain = set().union(*per_model_chain.values())
    emergent = sorted(res["CHAIN_SWARM_MDP"] - union_chain)

    d_mdp = r["CHAIN_SWARM_MDP"] - r["CHAIN_SWARM"]
    d_swarm = r["CHAIN_SWARM"] - r["CHAIN_SINGLE"]
    d_dec = r["CHAIN_SINGLE"] - r["WHOLE"]

    print(f"\n=== M1 (n={n}, {_calls} live calls, {time.time()-t0:.0f}s) ===")
    print("\n--- per-model ---")
    for m in models:
        print(f"  {m:35s} whole={len(per_model_whole[m])/n:.3f}  chain={len(per_model_chain[m])/n:.3f}")
    print("\n--- arms ---")
    for a in ("WHOLE", "CHAIN_SINGLE", "CHAIN_SWARM", "CHAIN_SWARM_MDP"):
        print(f"  {a:18s} {r[a]:.3f}")
    print(f"\n  M1-decomp  chain_single - whole      = {d_dec:+.3f}")
    print(f"  M1-swarm   chain_swarm - chain_single = {d_swarm:+.3f}")
    print(f"  M1-mdp     +MDP - swarm               = {d_mdp:+.3f}   "
         f"[{'CONFIRMED' if d_mdp >= 0.05 else 'FALSIFIED' if d_mdp <= 0 else 'WEAK'}]")
    print(f"  MDP fired: seam1 {mdp_fired['seam1']}/{n}, seam2 {mdp_fired['seam2']}/{n}")
    print(f"\n  STRICT EMERGENCE (mdp right, no single chain right): {len(emergent)}")
    print(f"  any-of-6 chain oracle = {len(union_chain)/n:.3f}")

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "m2_result.json").write_text(json.dumps({
        "n": n, "calls": _calls, "arms": r,
        "per_model_whole": {m: len(v) / n for m, v in per_model_whole.items()},
        "per_model_chain": {m: len(v) / n for m, v in per_model_chain.items()},
        "best_whole_model": best_whole_m, "best_chain_model": best_chain_m,
        "delta_decomp": d_dec, "delta_swarm": d_swarm, "delta_mdp": d_mdp,
        "mdp_fired": mdp_fired, "n_emergent": len(emergent), "emergent": emergent,
        "chain_oracle": len(union_chain) / n,
        "hop1_swarm": h1s, "hop1_mdp": h1m, "hop2_swarm": h2s, "hop2_mdp": h2m,
        "delta_mdp_hop1": h1m - h1s, "delta_mdp_hop2": h2m - h2s, "tasks_broken_by_mdp": broke,
        "verdicts": {
            "M2_mdp1": "CONFIRMED" if h1m - h1s >= 0.05 else ("FALSIFIED" if h1m - h1s <= 0 else "WEAK"),
            "M2_mdp2": "CONFIRMED" if h2m - h2s >= 0.05 else ("FALSIFIED" if h2m - h2s <= 0 else "WEAK"),
            "M2_mdp_safe": "SAFE" if broke == 0 else "UNSAFE",
            "M1_mdp": "CONFIRMED" if d_mdp >= 0.05 else ("FALSIFIED" if d_mdp <= 0 else "WEAK"),
            "M1_swarm": "CONFIRMED" if d_swarm > 0 else "FALSIFIED",
            "M1_decomp": "CONFIRMED" if d_dec > 0 else "FALSIFIED",
            "M1_emergence": "CONFIRMED" if len(emergent) >= 3 else (
                "NO_EMERGENCE" if not emergent else "WEAK"),
        },
        "detail": detail,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
