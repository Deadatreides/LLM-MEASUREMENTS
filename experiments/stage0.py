"""Stage 0 — the cheap filter that decides whether stage 2 runs at all.

No inference.  For every (needle, position) the haystack is split into blocks,
each block gets a BlockGraph, PPR personalised BY THE QUESTION, and
J = 1 - H(pi)/ln(n).  What matters is the RANK of the block that holds the
needle.

    R = P(needle block lands in the top third by J)
    a random selector gives R = 1/3 by construction

    R >= 0.60   selection works, run stage 2
    0.40-0.60   weak, run stage 2 cut down to A/B/C
    R <= 0.40   selection is random -- DO NOT RUN STAGE 2

Lexical leakage is the trap this stage is built to avoid.  personalization_vector
scores nodes by keyword overlap with the question, so if the needle shares
words with the question the whole thing degenerates into keyword search and R
is high for an uninteresting reason.  _keywords() is `\\b[a-zA-Z_]\\w{2,}\\b`
lowercased -- no stopword list, no stemming -- so needle and question here are
written with disjoint vocabulary, and the only thing they share is a numeric
identifier, which that regex cannot match because it must start with a letter.
assert_disjoint() enforces this before anything is measured.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
MYCELIUM = Path(r"<PROJECT_ROOT>\done\mycelium")
sys.path.insert(0, str(MYCELIUM))

from core.block_graph import BlockGraph, _keywords          # noqa: E402
from core.ppr import run_ppr, shannon_entropy               # noqa: E402

import needle as nd_mod                                     # noqa: E402

POSITIONS = [5, 30, 50, 70, 95]
ALPHA = 0.85
Q_ENV = 0.5
RHO = 1.0

# (statement wording, question wording) — deliberately synonym-disjoint.
# The shared referent is a number, which _keywords cannot see.
TEMPLATES = [
    ("Strongroom {n} opens on the combination {c}.",
     "Which sequence unlocks vault {n}?"),
    ("Ledger {n} carries an imprint reading {c}.",
     "What marking appears in journal {n}?"),
    ("Cabinet {n} was fitted with tumbler {c}.",
     "Which cypher belongs to locker {n}?"),
    ("Dispatch {n} bore the franking {c}.",
     "What stamp travelled with consignment {n}?"),
    ("Turbine {n} answers to the setting {c}.",
     "Which value governs generator {n}?"),
    ("Manuscript {n} closes on the colophon {c}.",
     "What inscription ends codex {n}?"),
    ("Berth {n} is logged against the pennant {c}.",
     "Which flag identifies mooring {n}?"),
    ("Crate {n} travelled beneath the chalk mark {c}.",
     "What symbol was daubed on parcel {n}?"),
]


def make_needles(n=50, seed=7):
    import random
    rng = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    digits = "23456789"
    out, used = [], set()
    while len(out) < n:
        # first character is a digit so the code is invisible to _keywords too
        code = rng.choice(digits) + "".join(
            rng.choice(alphabet + digits) for _ in range(5))
        if code in used:
            continue
        used.add(code)
        stmt, ques = TEMPLATES[len(out) % len(TEMPLATES)]
        num = 100 + len(out) * 7
        out.append(dict(id=len(out), code=code, num=num,
                        sentence=stmt.format(n=num, c=code),
                        question=ques.format(n=num)))
    return out


def assert_disjoint(needles):
    bad = []
    for nd in needles:
        inter = _keywords(nd["sentence"]) & _keywords(nd["question"])
        if inter:
            bad.append((nd["id"], sorted(inter)))
    print(f"lexical-leak check over {len(needles)} needles: "
          f"{'CLEAN' if not bad else 'LEAKS'}")
    if bad:
        for i, w in bad[:10]:
            print(f"   needle {i}: shares {w}")
        raise SystemExit("needles share content words with their questions; "
                         "stage 0 would measure a keyword retriever")
    ex = needles[0]
    print(f"   example needle  : {ex['sentence']}")
    print(f"   example question: {ex['question']}")
    print(f"   keywords(needle)  = {sorted(_keywords(ex['sentence']))}")
    print(f"   keywords(question)= {sorted(_keywords(ex['question']))}")
    return True


def block_stats(text, question):
    """J and Phi for one block."""
    g = BlockGraph().build(text, "general")
    if g.n <= 1:
        return dict(J=0.0, n=1, H=0.0, edges=0, phi=0.0, phi_mean=0.0,
                    d_eS=0.0, d_iS=0.0)
    P = g.transition_matrix()
    v = g.personalization_vector(question, "general")
    pi = run_ppr(P, v, alpha=ALPHA).pi
    H, _ = shannon_entropy(pi)
    ln_n = math.log(g.n)
    J = 1.0 - H / ln_n if ln_n > 0 else 0.0
    edges = int((g.adj > 0).sum())
    # d_eS = -rho * dH * q_env with dH = H - ln n (entropy exported vs uniform)
    d_eS = -RHO * (H - ln_n) * Q_ENV
    # d_iS = L_r * (pi_r - mean)^2.  The task writes L_r as the block's edge
    # count, a scalar, while pi_r is per node -- so the aggregation over nodes
    # is not specified, and the SIGN of Phi depends on the choice.  Both
    # readings are carried so the conclusion does not rest on mine.
    dev = float(((pi - pi.mean()) ** 2).sum())
    d_iS_sum = edges * dev
    d_iS_mean = edges * dev / g.n
    return dict(J=float(J), n=int(g.n), H=float(H), edges=edges,
                phi=float(abs(d_eS) - d_iS_sum),
                phi_mean=float(abs(d_eS) - d_iS_mean),
                d_eS=float(abs(d_eS)), d_iS=float(d_iS_sum))


def point_biserial(values, flags):
    v = np.asarray(values, float)
    f = np.asarray(flags, bool)
    if f.all() or not f.any():
        return float("nan")
    m1, m0 = v[f].mean(), v[~f].mean()
    s = v.std()
    p = f.mean()
    return float((m1 - m0) / s * math.sqrt(p * (1 - p))) if s > 0 else float("nan")


def run(n_needles=50, out_json=None, seed=0):
    paras = nd_mod.make_haystack(seed)
    needles = make_needles(n_needles)
    assert_disjoint(needles)
    print(f"\nhaystack {len(paras)} blocks; {len(needles)} needles x "
          f"{len(POSITIONS)} positions = {len(needles)*len(POSITIONS)} trials",
          flush=True)

    rows = []
    for nd in needles:
        for pos in POSITIONS:
            blocks, at = nd_mod.insert(paras, nd["sentence"], pos)
            st = [block_stats(b, nd["question"]) for b in blocks]
            J = np.array([s["J"] for s in st])
            phi = np.array([s["phi"] for s in st])
            phi_m = np.array([s["phi_mean"] for s in st])
            order = np.argsort(-J)
            rank = int(np.where(order == at)[0][0])           # 0 = best
            nb = len(blocks)
            rows.append(dict(
                needle=nd["id"], position=pos, n_blocks=nb, needle_block=at,
                rank=rank, rank_frac=rank / max(nb - 1, 1),
                top_third=bool(rank < nb / 3),
                J_needle=float(J[at]), J_mean=float(J.mean()),
                J_std=float(J.std()), J_min=float(J.min()),
                J_max=float(J.max()),
                phi_needle=float(phi[at]), phi_mean=float(phi.mean()),
                phi_needle_positive=bool(phi[at] > 0),
                phi_positive_frac=float((phi > 0).mean()),
                phi_mean_needle_positive=bool(phi_m[at] > 0),
                phi_mean_positive_frac=float((phi_m > 0).mean()),
                r_pb_J=point_biserial(J, np.arange(nb) == at),
                r_pb_phi=point_biserial(phi, np.arange(nb) == at),
            ))
        print(f"  needle {nd['id']+1}/{len(needles)} done", flush=True)

    R = float(np.mean([r["top_third"] for r in rows]))
    res = dict(
        R=R, n_trials=len(rows), n_needles=n_needles, positions=POSITIONS,
        n_blocks=rows[0]["n_blocks"],
        rank_frac_mean=float(np.mean([r["rank_frac"] for r in rows])),
        J_mean=float(np.mean([r["J_mean"] for r in rows])),
        J_std=float(np.mean([r["J_std"] for r in rows])),
        J_min=float(np.min([r["J_min"] for r in rows])),
        J_max=float(np.max([r["J_max"] for r in rows])),
        J_needle_mean=float(np.mean([r["J_needle"] for r in rows])),
        r_pb_J=float(np.nanmean([r["r_pb_J"] for r in rows])),
        r_pb_phi=float(np.nanmean([r["r_pb_phi"] for r in rows])),
        phi_needle_kept=float(np.mean([r["phi_needle_positive"] for r in rows])),
        phi_positive_frac=float(np.mean([r["phi_positive_frac"] for r in rows])),
        phi_mean_needle_kept=float(
            np.mean([r["phi_mean_needle_positive"] for r in rows])),
        phi_mean_positive_frac=float(
            np.mean([r["phi_mean_positive_frac"] for r in rows])),
    )
    by_pos = {}
    for p in POSITIONS:
        sel = [r for r in rows if r["position"] == p]
        by_pos[p] = float(np.mean([r["top_third"] for r in sel]))
    res["R_by_position"] = by_pos

    print("\n" + "=" * 70)
    print(f"R = P(needle block in top third by J) = {R:.4f}   "
          f"(random selector gives 0.3333)")
    print(f"mean rank fraction of the needle block = {res['rank_frac_mean']:.4f} "
          f"(0.5 = indistinguishable from chance)")
    print(f"J over blocks: mean {res['J_mean']:.4f} std {res['J_std']:.4f} "
          f"range [{res['J_min']:.4f}, {res['J_max']:.4f}]")
    print(f"J of the needle block: mean {res['J_needle_mean']:.4f}")
    print(f"point-biserial r(J, contains needle)   = {res['r_pb_J']:+.4f}")
    print(f"point-biserial r(Phi, contains needle) = {res['r_pb_phi']:+.4f}")
    print(f"Phi>0 (d_iS summed over nodes) keeps the needle block in "
          f"{100*res['phi_needle_kept']:.1f} % of trials; positive for "
          f"{100*res['phi_positive_frac']:.1f} % of all blocks")
    print(f"Phi>0 (d_iS averaged over nodes) keeps it in "
          f"{100*res['phi_mean_needle_kept']:.1f} %; positive for "
          f"{100*res['phi_mean_positive_frac']:.1f} % of all blocks")
    print("R by needle position:", {k: round(v, 3) for k, v in by_pos.items()})
    if R <= 0.40:
        verdict = ("SELECTION IS RANDOM — stage 2 must NOT run")
    elif R < 0.60:
        verdict = "weak — run stage 2 cut down to conditions A/B/C"
    else:
        verdict = "selection works — run stage 2"
    res["verdict"] = verdict
    print(f"VERDICT: {verdict}")
    print("=" * 70)

    out_json = out_json or str(ROOT / "out" / "stage0.json")
    Path(out_json).write_text(json.dumps(dict(summary=res, rows=rows), indent=1),
                              encoding="utf-8")
    print(f"wrote {out_json}")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--needles", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.needles, a.out)
