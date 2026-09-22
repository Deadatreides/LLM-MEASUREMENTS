"""run_g1.py — G1: swarm on GSM8K, no codebook anywhere.
Thresholds pre-registered in PROTOCOL_G1.md SS3.

Phase 1: 200 problems x 6 models solve independently -> numbers.
Phase 2: low-margin problems get a verification round on the top-2
         model-generated candidates.
Phase 1 results are written to disk before Phase 2 starts, so the main
comparison survives even if Phase 2 is interrupted.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gsm8k_data as GD          # noqa: E402
import safe_arith as SA          # noqa: E402
import model_registry_11 as MR   # noqa: E402
import call_log                  # noqa: E402

METRICS = Path(__file__).resolve().parents[1] / "metrics"
N_TASKS = 200
MAX_TOKENS = 320
MAX_TOKENS_VERIFY = 260
TEMPERATURE = 0.0
SEED = 4242
N_SPLITS = 40
SPLIT_SEED = 7
ADMIT_FRAC = 0.5      # admit if pi_m >= 0.5 * pi_best(train) -- pre-registered


def key(v):
    return None if v is None else round(float(v), 4)


def solve_all(tasks, model_ids) -> dict:
    """-> {task_id: {model: {'raw','ans','repaired','slips'}}}"""
    out = {t["task_id"]: {} for t in tasks}
    for model_id in model_ids:
        llm, load_t = MR.load_model(model_id)
        print(f"[G1/solve] {model_id} loaded {load_t:.1f}s", flush=True)
        t0 = time.time()
        for t in tasks:
            r = MR.generate(llm, model_id, GD.build_solve_prompt(t), temperature=TEMPERATURE,
                            top_p=1.0, seed=SEED, max_tokens=MAX_TOKENS)
            txt = r.get("raw_text", "")
            ans = GD.extract_answer(txt)
            eqs = SA.find_equations(txt)
            # det repair: if the model's LAST equation is miscomputed and its stated
            # value is what it reported, replace with the true evaluation
            rep = ans
            for expr, stated, true_v in reversed(eqs):
                if true_v is not None and abs(true_v - stated) > 0.01:
                    if ans is not None and abs(ans - stated) <= 0.01:
                        rep = true_v
                    break
            out[t["task_id"]][model_id] = {
                "ans": ans, "repaired": rep,
                "slips": sum(1 for _e, s, tv in eqs if tv is not None and abs(tv - s) > 0.01),
            }
            call_log.log_filter_call(model=model_id, task_id=t["task_id"], family="G1-solve",
                                     seed=SEED, n_in=r.get("input_tokens") or 0,
                                     n_out=r.get("output_tokens") or 0, ms=0.0,
                                     failed=bool(r.get("generation_failed")),
                                     extracted=[ans], raw_text=txt)
        print(f"[G1/solve] {model_id} done in {time.time()-t0:.0f}s", flush=True)
        del llm
    return out


def reliab(res, tasks_by_id, model_ids, ids) -> dict:
    return {m: sum(1 for t in ids if GD.same(res[t][m]["ans"], tasks_by_id[t]["gold"])) / len(ids)
            for m in model_ids}


def vote(res, t, pool, field, rel) -> float:
    tally = defaultdict(float)
    best = defaultdict(float)
    for m in pool:
        k = key(res[t][m][field])
        if k is None:
            continue
        tally[k] += 1.0
        best[k] = max(best[k], rel.get(m, 0.0))
    if not tally:
        return None
    return max(tally.items(), key=lambda kv: (kv[1], best[kv[0]]))[0]


def vote_weighted(res, t, pool, rel) -> float:
    import math
    tally = defaultdict(float)
    for m in pool:
        k = key(res[t][m]["ans"])
        if k is None:
            continue
        p = min(max(rel.get(m, 0.02), 0.02), 0.98)
        tally[k] += math.log(p / (1 - p)) + 2.0     # shift so weak-but-positive still counts
    return max(tally.items(), key=lambda kv: kv[1])[0] if tally else None


def margin_of(res, t, pool) -> float:
    tally = defaultdict(int)
    for m in pool:
        k = key(res[t][m]["ans"])
        if k is not None:
            tally[k] += 1
    if not tally:
        return 0.0
    c = sorted(tally.values(), reverse=True)
    return c[0] - (c[1] if len(c) > 1 else 0)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model_ids = list(MR.MODEL_IDS)
    tasks = GD.load(N_TASKS)
    tasks_by_id = {t["task_id"]: t for t in tasks}
    ids = [t["task_id"] for t in tasks]
    print(f"G1: GSM8K n={len(tasks)}, {len(model_ids)} models, "
         f"{len(tasks)*len(model_ids)} calls\n", flush=True)

    res = solve_all(tasks, model_ids)

    # ---------- per-model baseline ----------
    full = reliab(res, tasks_by_id, model_ids, ids)
    print("\n--- per-model accuracy (the 'all weak' baseline) ---")
    for m in model_ids:
        slips = statistics.mean(res[t][m]["slips"] for t in ids)
        print(f"  {m:35s} {full[m]:.3f}   arith-slips/problem={slips:.2f}")

    # ---------- 40 random 50/50 splits ----------
    acc = defaultdict(list)
    rng = random.Random(SPLIT_SEED)
    for _ in range(N_SPLITS):
        sh = ids[:]
        rng.shuffle(sh)
        tr, te = sh[:len(sh) // 2], sh[len(sh) // 2:]
        rel = reliab(res, tasks_by_id, model_ids, tr)
        bm = max(model_ids, key=lambda m: rel[m])
        pi_best = rel[bm]
        pool_f = [m for m in model_ids if rel[m] >= ADMIT_FRAC * pi_best] or [bm]

        acc["best_single"].append(
            sum(1 for t in te if GD.same(res[t][bm]["ans"], tasks_by_id[t]["gold"])) / len(te))
        for label, pool, field in (("plain", model_ids, "ans"), ("filtered", pool_f, "ans"),
                                   ("repaired", pool_f, "repaired")):
            acc[label].append(sum(1 for t in te
                                  if GD.same(vote(res, t, pool, field, rel), tasks_by_id[t]["gold"]))
                              / len(te))
        acc["weighted"].append(
            sum(1 for t in te if GD.same(vote_weighted(res, t, pool_f, rel), tasks_by_id[t]["gold"]))
            / len(te))
        acc["_pool_size"].append(len(pool_f))

    print(f"\n--- {N_SPLITS} random 50/50 splits, mean [min..max] ---")
    for k in ("best_single", "plain", "filtered", "weighted", "repaired"):
        v = acc[k]
        print(f"  {k:14s} {statistics.mean(v):.3f}  [{min(v):.2f}..{max(v):.2f}]")
    gain = statistics.mean(acc["filtered"]) - statistics.mean(acc["best_single"])
    wins = sum(1 for a, b in zip(acc["filtered"], acc["best_single"]) if a > b)
    print(f"\n  filtered - best_single = {gain:+.3f}   wins {wins}/{N_SPLITS}")
    print(f"  mean admitted pool size = {statistics.mean(acc['_pool_size']):.1f}")

    phase1 = {
        "n": len(tasks), "per_model": full,
        "means": {k: statistics.mean(v) for k, v in acc.items() if not k.startswith("_")},
        "swarm_gain": gain, "swarm_wins": wins, "n_splits": N_SPLITS,
        "mean_pool": statistics.mean(acc["_pool_size"]),
        "verdict_G1_swarm": "SWARM_WORKS" if gain > 0 else "SWARM_DEAD",
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "g1_phase1.json").write_text(json.dumps(phase1, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"\n  [phase 1 saved]  verdict = {phase1['verdict_G1_swarm']}", flush=True)

    # ---------- margin diagnostic ----------
    rel_full = full
    pool_full = [m for m in model_ids
                 if rel_full[m] >= ADMIT_FRAC * max(rel_full.values())] or model_ids
    rows = []
    for t in ids:
        v = vote(res, t, pool_full, "ans", rel_full)
        rows.append((margin_of(res, t, pool_full), GD.same(v, tasks_by_id[t]["gold"]), t))
    rows.sort(key=lambda r: -r[0])
    half = len(rows) // 2
    hi = statistics.mean(c for _, c, _ in rows[:half])
    lo = statistics.mean(c for _, c, _ in rows[half:])
    print(f"\n--- margin diagnostic --- overall={statistics.mean(c for _,c,_ in rows):.3f}  "
         f"hi-half={hi:.3f}  lo-half={lo:.3f}  spread={hi-lo:+.3f}")

    # ---------- PHASE 2: verification on low-margin ----------
    low = [t for _, _, t in rows[half:]]
    cands = {}
    for t in low:
        tally = defaultdict(int)
        for m in pool_full:
            k = key(res[t][m]["ans"])
            if k is not None:
                tally[k] += 1
        cands[t] = [k for k, _ in sorted(tally.items(), key=lambda kv: -kv[1])[:2]]
    n_calls = sum(len(c) for c in cands.values()) * len(model_ids)
    print(f"\n=== PHASE 2: verify {len(low)} low-margin problems, {n_calls} calls ===", flush=True)

    ver = {t: {c: {} for c in cands[t]} for t in low}
    for model_id in model_ids:
        llm, _ = MR.load_model(model_id)
        for t in low:
            for c in cands[t]:
                r = MR.generate(llm, model_id, GD.build_verify_prompt(tasks_by_id[t], c),
                                temperature=TEMPERATURE, top_p=1.0, seed=SEED,
                                max_tokens=MAX_TOKENS_VERIFY)
                ver[t][c][model_id] = GD.extract_verdict(r.get("raw_text", ""))
                call_log.log_filter_call(model=model_id, task_id=f"{t}:{c}", family="G1-verify",
                                        seed=SEED, n_in=r.get("input_tokens") or 0,
                                        n_out=r.get("output_tokens") or 0, ms=0.0,
                                        failed=bool(r.get("generation_failed")), extracted=[],
                                        raw_text=r.get("raw_text", ""))
        del llm

    n_before = n_after = 0
    for t in low:
        base = vote(res, t, pool_full, "ans", rel_full)
        n_before += int(GD.same(base, tasks_by_id[t]["gold"]))
        scored = sorted(cands[t],
                        key=lambda c: -sum(1 for m in model_ids if ver[t][c][m] == "YES"))
        pick = scored[0] if scored else base
        n_after += int(GD.same(pick, tasks_by_id[t]["gold"]))
    lo_b, lo_a = n_before / len(low), n_after / len(low)
    overall_after = (hi * half + lo_a * len(low)) / len(rows)

    out = dict(phase1)
    out.update({
        "margin_overall": statistics.mean(c for _, c, _ in rows), "margin_hi": hi,
        "margin_lo": lo, "margin_spread": hi - lo,
        "verify_low_before": lo_b, "verify_low_after": lo_a,
        "overall_after_verify": overall_after,
        "verify_calls": n_calls,
        "verdict_G1_verify": "VERIFY_HELPS" if lo_a > lo_b else "VERIFY_NULL",
    })
    (METRICS / "g1_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"\n  low-margin  {lo_b:.3f} -> {lo_a:.3f}   [{out['verdict_G1_verify']}]")
    print(f"  overall with verification: {overall_after:.3f}")
    print(f"  best single (full set): {max(full.values()):.3f}")


if __name__ == "__main__":
    main()
