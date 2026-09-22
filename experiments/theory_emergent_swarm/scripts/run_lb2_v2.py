"""run_lb2_v2.py — re-run ONLY the answer rounds on evidence re-assembled
with the span-level ВДП (see longbench_data.extract_evidence_v2).

The v1 sentence-level filter rejected 362/870 outputs; a hand-checked
sample showed ~60% of those were FALSE rejections (correct quotes with a
lead-in or re-bulleted). Fixing it recovered swarm 129->222 and solo
197->344 grounded items at ZERO GPU cost. Only the answer rounds need
new calls; the TRUNC arm is unaffected and is read from the v1 run.

Both v1 and v2 numbers are reported side by side -- the defect stays
visible, per the project rule.
"""

from __future__ import annotations

import json
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
SEED = 8081
MAX_ANS = 12
MARGIN_MIN = 0.0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    models = list(MR.MODEL_IDS)
    tasks = LB.load()
    tmap = {t["task_id"]: t for t in tasks}
    ids = [t["task_id"] for t in tasks]
    gold = {t["task_id"]: t["gold"] for t in tasks}

    ev = json.loads((METRICS / "lb2_evidence_v2.json").read_text(encoding="utf-8"))
    ev_swarm = ev["LB2-ev-swarm"]
    ev_solo = ev["LB2-ev-solo"]
    print(f"v2 evidence: swarm {sum(len(v) for v in ev_swarm.values())} items, "
         f"solo {sum(len(v) for v in ev_solo.values())} items\n", flush=True)

    t0 = time.time()

    def answer_round(evmap, tag):
        out = defaultdict(dict)
        for model_id in models:
            llm, _ = MR.load_model(model_id)
            for t in tasks:
                tid = t["task_id"]
                text = "\n".join(evmap.get(tid, []))[:6000]
                if not text:
                    out[tid][model_id] = None
                    continue
                r = MR.generate(llm, model_id, LB.prompt_answer(t, text), temperature=0.0,
                                top_p=1.0, seed=SEED, max_tokens=MAX_ANS)
                txt = r.get("raw_text", "")
                out[tid][model_id] = LB.extract_letter(txt)
                call_log.log_filter_call(model=model_id, task_id=tid, family=f"LB2v2-{tag}",
                                        seed=SEED, n_in=r.get("input_tokens") or 0,
                                        n_out=r.get("output_tokens") or 0, ms=0.0,
                                        failed=bool(r.get("generation_failed")), extracted=[],
                                        raw_text=txt)
            del llm
            print(f"  [{tag}] {model_id} done ({time.time()-t0:.0f}s)", flush=True)
        return out

    v_swarm = answer_round(ev_swarm, "ans-swarm")
    v_solo = answer_round(ev_solo, "ans-solo")

    def score(votes, evmap, name):
        pm = {m: sum(1 for i in ids if votes[i].get(m) == gold[i]) / len(ids) for m in models}
        w = EC.eigen_weights(votes, models, ids)
        n_ok = n_inapp = n_mdp = 0
        for i in ids:
            lab, margin = EC.weighted_vote(votes[i], w, LB.LETTERS)
            text = "\n".join(evmap.get(i, []))
            if lab and text and not LB.evidence_supports(text, tmap[i], lab):
                alts = sorted({v for v in votes[i].values() if v and v != lab})
                nxt = next((a for a in alts if LB.evidence_supports(text, tmap[i], a)), None)
                if nxt:
                    lab, n_mdp = nxt, n_mdp + 1
            if lab is None or margin <= MARGIN_MIN:
                n_inapp += 1
                continue
            n_ok += int(lab == gold[i])
        plain = sum(1 for i in ids
                    if max(((L, sum(1 for m in models if votes[i].get(m) == L))
                            for L in LB.LETTERS), key=lambda kv: kv[1])[0] == gold[i]) / len(ids)
        rho = EC.spearman([w[m] for m in models], [pm[m] for m in models])
        return {"per_model": pm, "best_single": max(pm.values()), "plain_vote": plain,
                "eigen": n_ok / len(ids), "inapplicable": n_inapp / len(ids),
                "mdp_fired": n_mdp, "eigen_weights": w, "spearman": rho,
                "evidence_items": sum(len(v) for v in evmap.values())}

    res = {"SOLO": score(v_solo, ev_solo, "SOLO"), "SWARM": score(v_swarm, ev_swarm, "SWARM")}
    v1 = json.loads((METRICS / "lb2_result.json").read_text(encoding="utf-8"))

    print(f"\n=== LB2 v2 (span-level ВДП), n={len(ids)}, random=0.250 ===")
    print(f"  {'arm':6s} {'best_single':>11s} {'plain':>7s} {'EIGEN':>7s} {'inapp':>6s} "
         f"{'mdp':>4s} {'evid':>5s} {'spearman':>9s}")
    print(f"  {'TRUNC':6s} {v1['arms']['TRUNC']['best_single']:11.3f} "
         f"{v1['arms']['TRUNC']['plain_vote']:7.3f} {v1['arms']['TRUNC']['eigen']:7.3f} "
         f"{v1['arms']['TRUNC']['inapplicable']:6.3f} {0:4d} {0:5d} "
         f"{v1['arms']['TRUNC']['spearman_eigen_vs_true']:+9.3f}   (unchanged)")
    for name in ("SOLO", "SWARM"):
        r, o = res[name], v1["arms"][name]
        print(f"  {name:6s} {r['best_single']:11.3f} {r['plain_vote']:7.3f} {r['eigen']:7.3f} "
             f"{r['inapplicable']:6.3f} {r['mdp_fired']:4d} {r['evidence_items']:5d} "
             f"{r['spearman']:+9.3f}")
        print(f"  {'  (v1)':6s} {o['best_single']:11.3f} {o['plain_vote']:7.3f} {o['eigen']:7.3f} "
             f"{o['inapplicable']:6.3f} {o['mdp_fired']:4d} {o['evidence_items']:5d} "
             f"{o['spearman_eigen_vs_true']:+9.3f}")

    d_dec = res["SOLO"]["eigen"] - v1["arms"]["TRUNC"]["eigen"]
    d_sw = res["SWARM"]["eigen"] - res["SOLO"]["eigen"]
    d_bs = res["SWARM"]["eigen"] - res["SWARM"]["best_single"]
    print(f"\n  DECOMP solo - trunc        = {d_dec:+.3f}")
    print(f"  SWARM  swarm - solo        = {d_sw:+.3f}")
    print(f"  vs best single (swarm arm) = {d_bs:+.3f}")
    print(f"  NOTE n=44, SE~0.065 -> only |delta| >= 0.15 is resolvable")

    (METRICS / "lb2_v2_result.json").write_text(json.dumps({
        "n": len(ids), "vdp": "span-level v2", "arms": res, "trunc_from_v1": v1["arms"]["TRUNC"],
        "delta_decomp": d_dec, "delta_swarm": d_sw, "delta_vs_best_single": d_bs,
        "v1_comparison": {k: v1["arms"][k] for k in ("SOLO", "SWARM")},
        "elapsed_sec": time.time() - t0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
