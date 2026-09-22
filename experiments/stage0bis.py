"""Stage 0-BIS — the rebuilt filter.

Three faults in the previous version, all of them fatal to different degrees:

1.  personalization_vector returns a UNIFORM vector when the question shares no
    keyword with any node (block_graph.py:274-ish).  The disjoint-vocabulary
    requirement therefore removed the only channel through which the question
    entered the computation: pi depended on graph structure alone.  A positive
    result was unreachable by construction.
2.  The sign was inverted relative to Mycelium, which selects on HIGH entropy.
    Read that way the old data give R = 0.724, not 0.000.
3.  Blocks were too small: n ~ 10 nodes, ln n ~ 2.3, J spread 0.014-0.108.

So: keyword disjointness is dropped, four criteria are measured on the same
material, the personalisation vector is computed both lexically and from
embeddings, and block size is swept rather than fixed.

    J-low      1 - H(pi)/ln n        concentration of pi
    H-high     H(pi)                 Mycelium's own rule; novelty
    PPR-mass   sum of pi over the nodes the question seeded
    BM25       standard, the honest baseline

If BM25 wins, that is the answer and it must be said plainly.
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
MYCELIUM = Path(r"<PROJECT_ROOT>\done\mycelium")
sys.path.insert(0, str(MYCELIUM))

from core.block_graph import BlockGraph, _keywords            # noqa: E402
from core.ppr import run_ppr, shannon_entropy                 # noqa: E402

import needle as nd_mod                                       # noqa: E402

POSITIONS = [5, 30, 50, 70, 95]
BLOCK_SIZES = [3, 6, 12, 24, 48]
ALPHA = 0.85
TAU_V = 0.1
EMB_FRACTION = 2 / 3          # hidden layer at ~2/3 of depth
CRITERIA = ["J_low", "H_high", "PPR_mass", "BM25"]

# Needles are written in the HAYSTACK's own vocabulary.  With the previous
# wording ("strongroom", "colophon", "tumbler") the needle was a lexical
# outlier in a haystack about archives and harbours, so BM25 scored R = 1.000
# and the B - D comparison measured nothing.  Here every content word also
# occurs in the distractor text, which leaves the numeric identifier as the
# only discriminator -- the situation a real retriever actually faces.
#
# `question` overlaps lexically, as a real query does.  `question_disjoint` is
# the previous crippled form, kept so the size of fault 1 can be shown.
TEMPLATES = [
    ("The committee logged reconciliation entry {c} for department {n}.",
     "Which entry did the committee log for department {n}?",
     "Which sequence was minuted by the panel for section {n}?"),
    ("Staff recorded shelf measurement {c} in ledger {n}.",
     "Which measurement did staff record in ledger {n}?",
     "What figure was noted by personnel in journal {n}?"),
    ("The officer countersigned procedure {c} for berth {n}.",
     "Which procedure did the officer countersign for berth {n}?",
     "Which protocol was endorsed by the warden for mooring {n}?"),
    ("Temporary staff cleared backlog {c} in department {n}.",
     "Which backlog did temporary staff clear in department {n}?",
     "Which arrears were settled by casual personnel in section {n}?"),
    ("The review recommended equipment {c} for schedule {n}.",
     "Which equipment did the review recommend for schedule {n}?",
     "Which apparatus was advised by the audit for timetable {n}?"),
    ("Correspondence names conventions {c} for reconciliation {n}.",
     "Which conventions does correspondence name for reconciliation {n}?",
     "Which usages appear in letters about balancing {n}?"),
    ("The committee shelved proposal {c} for budget {n}.",
     "Which proposal did the committee shelve for budget {n}?",
     "Which submission was deferred by the panel for allocation {n}?"),
    ("Measurements taken at midday gave discrepancy {c} for record {n}.",
     "Which discrepancy did midday measurements give for record {n}?",
     "Which deviation arose from noon readings in entry {n}?"),
]


def make_needles(n=25, seed=7):
    import random
    rng = random.Random(seed)
    alph, dig = "ABCDEFGHJKLMNPQRSTUVWXYZ", "23456789"
    out, used = [], set()
    while len(out) < n:
        code = rng.choice(dig) + "".join(rng.choice(alph + dig) for _ in range(5))
        if code in used:
            continue
        used.add(code)
        stmt, q_lex, q_dis = TEMPLATES[len(out) % len(TEMPLATES)]
        num = 100 + len(out) * 7
        out.append(dict(id=len(out), code=code, num=num,
                        sentence=stmt.format(n=num, c=code),
                        question=q_lex.format(n=num),
                        question_disjoint=q_dis.format(n=num)))
    return out


# ------------------------------------------------------------- corpus -------

def sentence_pool(seed=0):
    """The haystack as a flat sentence list, so blocks can be re-cut freely."""
    paras = nd_mod.make_haystack(seed, n_paragraphs=49, sents_per_para=6)
    sents = []
    for p in paras:
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
    return sents


def natural_pool(n_sents=294, seed=0):
    """Real English prose, for the H-as-novelty check.

    The synthetic haystack is 12 topics over 6 frames, so the needle is the
    only anomaly in it and H has an easy job.  Natural text moves H for many
    reasons.  Nothing is downloaded: docstrings from the installed packages are
    genuine technical English, which §"Побочно" allows ("проза ИЛИ
    документация").
    """
    import ast
    import random
    roots = []
    base = Path(sys.executable).parent / "Lib" / "site-packages"
    for pkg in ("numpy", "scipy", "sklearn", "transformers"):
        p = base / pkg
        if p.exists():
            roots.append(p)
    texts = []
    for r in roots:
        for f in list(r.rglob("*.py"))[:400]:
            try:
                tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    d = ast.get_docstring(node)
                    if d and len(d) > 300:
                        texts.append(d)
            if len(texts) > 3000:
                break
    sents = []
    for d in texts:
        for s in re.split(r"(?<=[.!?])\s+", " ".join(d.split())):
            s = s.strip()
            # prose only: drop signatures, parameter tables, code fragments
            if (60 <= len(s) <= 400 and s[0].isupper() and s.endswith(".")
                    and not re.search(r"[>{}|=\[\]`_]{2,}|::|\bdef\b|\bself\b", s)
                    and sum(c.isalpha() or c.isspace() for c in s) / len(s) > 0.92):
                sents.append(s)
    uniq = list(dict.fromkeys(sents))
    random.Random(seed).shuffle(uniq)
    if len(uniq) < n_sents:
        raise SystemExit(f"only {len(uniq)} usable prose sentences found")
    print(f"natural pool: {len(uniq)} prose sentences available, "
          f"using {n_sents}")
    return uniq[:n_sents]


def cut(sents, size):
    return ["\n\n".join(sents[i:i + size]) for i in range(0, len(sents), size)]


def insert_sentence(sents, needle_sentence, pct):
    """Put the needle in the sentence stream, mid-block by construction."""
    at = max(0, min(len(sents), round(len(sents) * pct / 100)))
    return sents[:at] + [needle_sentence] + sents[at:], at


# ---------------------------------------------------------- criteria --------

def uniform_v_warning(g, question):
    """Does personalization_vector actually carry question information?"""
    v = g.personalization_vector(question, "general")
    return float(np.abs(v - 1.0 / g.n).max())


def graph_scores(text, question, emb_lookup=None, q_emb=None):
    """All graph-side criteria for one block.

    PPR_mass is the task's definition, sum of pi over question-seeded nodes.
    PPR_dot = sum pi_i v_i is added because the plain sum rewards diffuse
    matching: a block with five weakly matched nodes outscores one with a
    single strong match, which is backwards for retrieval.  Both are reported.

    H_unif is H(pi) with a UNIFORM v, i.e. carrying no question information at
    all.  The novelty-detector claim rests on exactly that quantity, so it is
    measured separately rather than inferred.
    """
    g = BlockGraph().build(text, "general")
    if g.n <= 1:
        return dict(n=g.n, edges=0)
    P = g.transition_matrix()
    ln_n = math.log(g.n)
    out = {"n": g.n, "edges": int((g.adj > 0).sum())}

    pi_u = run_ppr(P, np.ones(g.n) / g.n, alpha=ALPHA).pi
    H_u, _ = shannon_entropy(pi_u)
    out["H_unif"] = float(H_u)
    out["J_unif"] = 1.0 - H_u / ln_n

    for tag, v in (("", _v_lex(g, question)),
                   ("_emb", _v_emb(g, emb_lookup, q_emb))):
        if v is None:
            continue
        pi = run_ppr(P, v, alpha=ALPHA).pi
        H, _ = shannon_entropy(pi)
        seed = v > (1.0 / g.n) * (1 + 1e-9)
        mass = float(pi[seed].sum()) if seed.any() else 0.0
        out[f"J_low{tag}"] = 1.0 - H / ln_n
        out[f"H_high{tag}"] = float(H)
        out[f"PPR_mass{tag}"] = mass
        out[f"PPR_dot{tag}"] = float((pi * v).sum() * g.n)
        out[f"v_spread{tag}"] = float(np.abs(v - 1.0 / g.n).max())
    return out


def _v_lex(g, question):
    return g.personalization_vector(question, "general")


def _v_emb(g, emb_lookup, q_emb):
    """v_i ~ exp(cos(emb(node_i), emb(question)) / tau).

    block_graph declares this path and leaves it unimplemented
    (`task_emb = None  # нет task embedding — пропускаем`), so it is supplied
    here from the model that is already loaded.
    """
    if emb_lookup is None or q_emb is None:
        return None
    E = np.stack([emb_lookup(lbl) for lbl in g.node_labels])
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    q = q_emb / max(np.linalg.norm(q_emb), 1e-9)
    cos = E @ q
    w = np.exp((cos - cos.max()) / TAU_V)
    return w / w.sum()


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(_tok(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], float)
        self.avg = self.len.mean() if len(self.len) else 1.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for w, c in df.items()}

    def scores(self, query):
        q = _tok(query)
        out = np.zeros(len(self.tf))
        for i, t in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = t.get(w, 0)
                if not f:
                    continue
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            out[i] = s
        return out


def _tok(s):
    return [w for w in re.findall(r"[a-zA-Z_]\w{2,}", s.lower())]


# ------------------------------------------------------------- runner -------

def embedder(tag="B"):
    """Mean-pooled hidden state at ~2/3 depth, cached per string."""
    import torch
    import probe
    torch.set_grad_enabled(False)
    tok, model, cfg = probe.load(tag)
    dev = "cuda" if cfg["device"] == "cuda" else "cpu"
    layer = int(model.config.num_hidden_layers * EMB_FRACTION)
    cache = {}
    print(f"embedder: {cfg['repo']} hidden layer {layer} of "
          f"{model.config.num_hidden_layers}", flush=True)

    def embed_many(texts):
        todo = [t for t in texts if t not in cache]
        for i in range(0, len(todo), 16):
            chunk = todo[i:i + 16]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(dev)
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[layer].float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            for t, e in zip(chunk, pooled.cpu().numpy()):
                cache[t] = e
        return [cache[t] for t in texts]

    return embed_many, cache


def run(n_needles=25, out_json=None, use_emb=True, seed=0, natural=False):
    sents = natural_pool(seed=seed) if natural else sentence_pool(seed)
    needles = make_needles(n_needles)
    print(f"sentence pool {len(sents)}; {len(needles)} needles x "
          f"{len(POSITIONS)} positions x {len(BLOCK_SIZES)} block sizes = "
          f"{len(needles)*len(POSITIONS)*len(BLOCK_SIZES)} trials", flush=True)

    # Fault 1, shown on the block that actually holds the needle -- that is
    # where the question must be able to act, and where the disjoint wording
    # collapses v to uniform.
    print("\n=== fault 1: does the question reach the computation? ===")
    nd = needles[0]
    s2, at_sent = insert_sentence(sents, nd["sentence"], 50)
    blocks = cut(s2, 6)
    gb = BlockGraph().build(blocks[at_sent // 6], "general")
    for lbl, q in (("overlapping", nd["question"]),
                   ("disjoint", nd["question_disjoint"])):
        v = gb.personalization_vector(q, "general")
        spread = float(np.abs(v - 1.0 / gb.n).max())
        print(f"  {lbl:12s} max|v - 1/n| = {spread:.3e}"
              + ("   <- UNIFORM: the question is invisible to PPR"
                 if spread < 1e-9 else ""))
    print(f"  (needle block, n = {gb.n} nodes)")
    print(f"  needle  : {nd['sentence']}")
    print(f"  question: {nd['question']}")
    print(f"  shared keywords: "
          f"{sorted(_keywords(nd['sentence']) & _keywords(nd['question']))}")

    embed_many = q_emb_of = None
    if use_emb:
        embed_many, _ = embedder("B")
        need = list(dict.fromkeys(sents + [n["sentence"] for n in needles]
                                  + [n["question"] for n in needles]))
        print(f"embedding {len(need)} distinct strings ...", flush=True)
        embed_many(need)

        def emb_lookup(s, _cache={}):
            return embed_many([s])[0]

        def q_emb_of(q):
            return embed_many([q])[0]
    else:
        def emb_lookup(s):
            return None

    rows = []
    for bs in BLOCK_SIZES:
        for nd in needles:
            qe = q_emb_of(nd["question"]) if use_emb else None
            for pos in POSITIONS:
                s2, at_sent = insert_sentence(sents, nd["sentence"], pos)
                blocks = cut(s2, bs)
                at = at_sent // bs
                assert nd["code"] in blocks[at], "needle block index wrong"
                st = [graph_scores(b, nd["question"],
                                   emb_lookup if use_emb else None, qe)
                      for b in blocks]
                bm = BM25(blocks).scores(nd["question"])
                nb = len(blocks)
                rec = dict(needle=nd["id"], position=pos, block_size=bs,
                           n_blocks=nb, needle_block=at,
                           n_nodes_mean=float(np.mean([s["n"] for s in st])))
                series = {c: [s.get(c, 0.0) for s in st] for c in
                          ("J_low", "H_high", "PPR_mass", "PPR_dot", "H_unif")}
                series["BM25"] = list(bm)
                if use_emb:
                    for c in ("J_low_emb", "H_high_emb", "PPR_mass_emb",
                              "PPR_dot_emb"):
                        series[c] = [s.get(c, 0.0) for s in st]
                for crit, vals in series.items():
                    v = np.asarray(vals, float)
                    order = np.argsort(-v)              # every criterion: higher is better
                    rank = int(np.where(order == at)[0][0])
                    rec[f"rank_{crit}"] = rank
                    rec[f"frac_{crit}"] = rank / max(nb - 1, 1)
                    rec[f"top3_{crit}"] = bool(rank < nb / 3)
                    rec[f"spread_{crit}"] = float(v.std())
                rows.append(rec)
        print(f"  block size {bs} done ({len(rows)} rows)", flush=True)

    crits = ["J_low", "H_high", "H_unif", "PPR_mass", "PPR_dot", "BM25"]
    if use_emb:
        crits += ["J_low_emb", "H_high_emb", "PPR_mass_emb", "PPR_dot_emb"]
    summary = {"n_trials": len(rows), "n_needles": n_needles,
               "block_sizes": BLOCK_SIZES, "by_block": {}, "overall": {}}
    print("\n" + "=" * 78)
    hdr = f"{'block':>6s} {'blocks':>7s} {'nodes':>6s} " + \
          " ".join(f"{c:>13s}" for c in crits)
    print("R = P(needle block in top third)")
    print(hdr)
    for bs in BLOCK_SIZES:
        sel = [r for r in rows if r["block_size"] == bs]
        line = (f"{bs:6d} {sel[0]['n_blocks']:7d} "
                f"{np.mean([r['n_nodes_mean'] for r in sel]):6.1f} ")
        d = {}
        for c in crits:
            R = float(np.mean([r[f"top3_{c}"] for r in sel]))
            d[c] = dict(R=R,
                        frac=float(np.mean([r[f"frac_{c}"] for r in sel])),
                        spread=float(np.mean([r[f"spread_{c}"] for r in sel])))
            line += f" {R:13.3f}"
        summary["by_block"][bs] = d
        print(line)
    print("\nmean rank fraction (0.5 = chance, lower is better)")
    print(hdr)
    for bs in BLOCK_SIZES:
        sel = [r for r in rows if r["block_size"] == bs]
        line = (f"{bs:6d} {sel[0]['n_blocks']:7d} "
                f"{np.mean([r['n_nodes_mean'] for r in sel]):6.1f} ")
        for c in crits:
            line += f" {np.mean([r[f'frac_{c}'] for r in sel]):13.3f}"
        print(line)

    best = None
    for c in crits:
        R = float(np.mean([r[f"top3_{c}"] for r in rows]))
        summary["overall"][c] = R
        if best is None or R > best[1]:
            best = (c, R)
    bestR = max(max(v[c]["R"] for c in crits)
                for v in summary["by_block"].values())
    summary["best_overall"] = {"criterion": best[0], "R": best[1]}
    summary["best_R_any_block"] = bestR
    print(f"\nbest overall criterion: {best[0]} with R = {best[1]:.3f}")
    print(f"best R at any block size: {bestR:.3f}   "
          f"(random selector gives 0.333)")
    if bestR >= 0.60:
        verdict = f"filter PASSED — run stage 2 with the best criterion"
    elif bestR > 0.40:
        verdict = "weak — run stage 2 cut down (A/B/C/D, one budget)"
    else:
        verdict = "filter FAILED — stop and report"
    summary["verdict"] = verdict
    print(f"VERDICT: {verdict}")
    print("=" * 78)

    out_json = out_json or str(ROOT / "out" / "stage0bis.json")
    Path(out_json).write_text(json.dumps(dict(summary=summary, rows=rows),
                                        indent=1), encoding="utf-8")
    print(f"wrote {out_json}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--needles", type=int, default=25)
    ap.add_argument("--no-emb", action="store_true")
    ap.add_argument("--natural", action="store_true",
                    help="use real prose instead of the synthetic haystack")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.needles, a.out, use_emb=not a.no_emb, natural=a.natural)
