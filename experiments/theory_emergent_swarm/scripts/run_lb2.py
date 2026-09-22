"""run_lb2.py — LongBench v2 (Multi-Doc QA / hard / short): the whole
stack at once. PROTOCOL_LB2.md holds the pre-registered thresholds.

Arms, at EQUAL chunk-reading budget:
  TRUNC : no decomposition -- each model sees only the first chunk
  SOLO  : one model reads ALL chunks itself, assembles, then all answer
  SWARM : chunks distributed round-robin across 6 models, then all answer

SOLO vs SWARM is the clean test: identical number of chunk reads, the only
difference is whether one model or six covered the context.

Consensus is EIGEN-weighted (label-free). МДП checks the evidence->answer
seam. Low margin -> INAPPLICABLE rather than a guess.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import longbench_data as LB        # noqa: E402
import eigen_consensus as EC       # noqa: E402
import model_registry_11 as MR     # noqa: E402
import call_log                    # noqa: E402

METRICS = Path(__file__).resolve().parents[1] / "metrics"
SEED = 8080
MAX_EV, MAX_ANS = 150, 12
SOLO_MODEL = "internvl3-2b-q4_k_m"     # strongest on M2 hop1 and on GSM8K; fixed in advance
MARGIN_MIN = 0.0                        # >0 required to answer; else INAPPLICABLE


def ask(llm, model_id, prompt, max_tokens, tag, tid):
    r = MR.generate(llm, model_id, prompt, temperature=0.0, top_p=1.0, seed=SEED,
                    max_tokens=max_tokens)
    txt = r.get("raw_text", "")
    call_log.log_filter_call(model=model_id, task_id=tid, family=f"LB2-{tag}", seed=SEED,
                             n_in=r.get("input_tokens") or 0, n_out=r.get("output_tokens") or 0,
                             ms=0.0, failed=bool(r.get("generation_failed")), extracted=[],
                             raw_text=txt)
    return txt


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    models = list(MR.MODEL_IDS)
    tasks = LB.load()
    chunks = {t["task_id"]: LB.chunks_of(t) for t in tasks}
    total_chunks = sum(len(c) for c in chunks.values())
    print(f"LB2: {len(tasks)} questions, {total_chunks} chunks "
         f"(median {statistics.median(len(c) for c in chunks.values()):.0f}/question)\n", flush=True)

    # chunk -> model assignment: round-robin, deterministic
    assign = {t["task_id"]: [models[i % len(models)] for i in range(len(chunks[t["task_id"]]))]
              for t in tasks}

    ev_swarm = defaultdict(list)
    ev_solo = defaultdict(list)
    t0 = time.time()

    # ---------- SWARM coverage: each model reads only its assigned chunks ----------
    for model_id in models:
        llm, _ = MR.load_model(model_id)
        for t in tasks:
            tid = t["task_id"]
            for ci, ch in enumerate(chunks[tid]):
                if assign[tid][ci] != model_id:
                    continue
                got = LB.extract_evidence(
                    ask(llm, model_id, LB.prompt_evidence(t, ch), MAX_EV, "ev-swarm",
                        f"{tid}#c{ci}"), ch)
                if got:
                    ev_swarm[tid].append(got)
        del llm
        print(f"  [swarm] {model_id} done ({time.time()-t0:.0f}s)", flush=True)

    # ---------- SOLO coverage: one model reads everything (same call count) ----------
    llm, _ = MR.load_model(SOLO_MODEL)
    for t in tasks:
        tid = t["task_id"]
        for ci, ch in enumerate(chunks[tid]):
            got = LB.extract_evidence(
                ask(llm, SOLO_MODEL, LB.prompt_evidence(t, ch), MAX_EV, "ev-solo",
                    f"{tid}#c{ci}"), ch)
            if got:
                ev_solo[tid].append(got)
    del llm
    print(f"  [solo] {SOLO_MODEL} done ({time.time()-t0:.0f}s)", flush=True)

    # ---------- answer rounds ----------
    def answer_round(evidence_map, tag):
        out = defaultdict(dict)
        for model_id in models:
            l, _ = MR.load_model(model_id)
            for t in tasks:
                tid = t["task_id"]
                ev = "\n".join(evidence_map.get(tid, []))[:6000]
                out[tid][model_id] = LB.extract_letter(
                    ask(l, model_id, LB.prompt_answer(t, ev), MAX_ANS, tag, tid)) if ev else None
            del l
            print(f"  [{tag}] {model_id} done ({time.time()-t0:.0f}s)", flush=True)
        return out

    v_swarm = answer_round(ev_swarm, "ans-swarm")
    v_solo = answer_round(ev_solo, "ans-solo")

    v_trunc = defaultdict(dict)
    for model_id in models:
        l, _ = MR.load_model(model_id)
        for t in tasks:
            tid = t["task_id"]
            v_trunc[tid][model_id] = LB.extract_letter(
                ask(l, model_id, LB.prompt_truncated(t, chunks[tid][0]), MAX_ANS, "trunc", tid))
        del l
    print(f"  [trunc] all models done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------------------- scoring ----------------------------
    ids = [t["task_id"] for t in tasks]
    gold = {t["task_id"]: t["gold"] for t in tasks}
    tmap = {t["task_id"]: t for t in tasks}

    def per_model(votes):
        return {m: sum(1 for i in ids if votes[i].get(m) == gold[i]) / len(ids) for m in models}

    def eigen_arm(votes, evmap, use_mdp):
        w = EC.eigen_weights(votes, models, ids)              # NO labels used
        n_ok = n_inapp = n_mdp = 0
        for i in ids:
            lab, margin = EC.weighted_vote(votes[i], w, LB.LETTERS)
            ev = "\n".join(evmap.get(i, [])) if evmap is not None else ""
            if use_mdp and lab and ev and not LB.evidence_supports(ev, tmap[i], lab):
                alts = sorted({v for v in votes[i].values() if v and v != lab})
                nxt = next((a for a in alts if LB.evidence_supports(ev, tmap[i], a)), None)
                if nxt:
                    lab, n_mdp = nxt, n_mdp + 1
            if lab is None or margin <= MARGIN_MIN:
                n_inapp += 1
                continue
            n_ok += int(lab == gold[i])
        return n_ok / len(ids), n_inapp / len(ids), n_mdp, w

    res = {}
    for name, votes, evmap in (("TRUNC", v_trunc, None), ("SOLO", v_solo, ev_solo),
                               ("SWARM", v_swarm, ev_swarm)):
        pm = per_model(votes)
        r_eig, inapp, n_mdp, w = eigen_arm(votes, evmap, use_mdp=(evmap is not None))
        r_plain = sum(1 for i in ids
                      if max(((l, sum(1 for m in models if votes[i].get(m) == l))
                              for l in LB.LETTERS), key=lambda kv: kv[1])[0] == gold[i]) / len(ids)
        true_rel = [pm[m] for m in models]
        rho = EC.spearman([w[m] for m in models], true_rel)
        res[name] = {"per_model": pm, "best_single": max(pm.values()),
                     "plain_vote": r_plain, "eigen": r_eig, "inapplicable": inapp,
                     "mdp_fired": n_mdp, "eigen_weights": w, "spearman_eigen_vs_true": rho,
                     "evidence_items": (sum(len(v) for v in evmap.values()) if evmap else 0)}

    print(f"\n=== LB2 (n={len(ids)}, {time.time()-t0:.0f}s) ===")
    print(f"  random baseline = 0.250   |   n=44 -> SE~0.065, only large effects resolvable\n")
    for name in ("TRUNC", "SOLO", "SWARM"):
        r = res[name]
        print(f"  {name:6s} best_single={r['best_single']:.3f}  plain={r['plain_vote']:.3f}  "
             f"EIGEN={r['eigen']:.3f}  inapp={r['inapplicable']:.3f}  "
             f"mdp={r['mdp_fired']}  evidence={r['evidence_items']}")
        print(f"         eigen-vs-true reliability Spearman = {r['spearman_eigen_vs_true']:+.3f}")
    print(f"\n  DECOMP  solo_eigen - trunc_eigen  = {res['SOLO']['eigen']-res['TRUNC']['eigen']:+.3f}")
    print(f"  SWARM   swarm_eigen - solo_eigen  = {res['SWARM']['eigen']-res['SOLO']['eigen']:+.3f}")
    print(f"  vs best single (swarm arm)        = {res['SWARM']['eigen']-res['SWARM']['best_single']:+.3f}")

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "lb2_result.json").write_text(json.dumps({
        "n": len(ids), "total_chunks": total_chunks, "solo_model": SOLO_MODEL,
        "arms": res,
        "delta_decomp": res["SOLO"]["eigen"] - res["TRUNC"]["eigen"],
        "delta_swarm": res["SWARM"]["eigen"] - res["SOLO"]["eigen"],
        "delta_vs_best_single": res["SWARM"]["eigen"] - res["SWARM"]["best_single"],
        "elapsed_sec": time.time() - t0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
