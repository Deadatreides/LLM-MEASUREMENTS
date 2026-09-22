"""Part C: does splitting the context defeat "lost in the middle"?

Four conditions over the same trials:
  A  full linear context (~6000 tokens)          the baseline U-curve
  B  blocks + negentropy selection, 2000 budget  the claim
  C  blocks + RANDOM selection, 2000 budget      the control that makes B mean
                                                 something -- compression alone
                                                 moves the U-curve
  D  blocks + cosine-to-question selection       separates "negentropy" from
                                                 "any sensible retrieval"

The headline is the B - C gap at the middle positions.

Block graphs, PPR and Shannon entropy come from mycelium/core (block_graph.py,
ppr.py).  The task text points at ptg_core.py in PTG+MCP1.3, but that file has
none of the named interfaces -- BlockGraph, run_ppr, shannon_entropy live in
mycelium/core, and that is what is used here.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
MYCELIUM = Path(r"<PROJECT_ROOT>\done\mycelium")
sys.path.insert(0, str(MYCELIUM))

from core.block_graph import BlockGraph                   # noqa: E402
from core.ppr import run_ppr, shannon_entropy             # noqa: E402

MIN_BLOCK_CHARS = 10          # named in §C.3; it lives in ptg_core, not here

POSITIONS = [5, 20, 35, 50, 65, 80, 95]
N_NEEDLES = 50
BUDGET_TOKENS = 2000
HAYSTACK_TOKENS = 6000
ALPHA = 0.85


# ------------------------------------------------------------ haystack ------

TOPICS = [
    ("the municipal archive", "cataloguing", "shelf humidity", "index cards"),
    ("the harbour authority", "berth allocation", "tidal charts", "mooring fees"),
    ("the botanical garden", "seed exchange", "glasshouse heating", "labelling"),
    ("the railway workshop", "axle inspection", "brake wear", "paint schedules"),
    ("the printing house", "plate alignment", "ink viscosity", "paper stock"),
    ("the weather station", "sensor drift", "balloon launches", "log books"),
    ("the ferry service", "timetable revisions", "fuel logs", "life jackets"),
    ("the observatory", "mirror cleaning", "dome rotation", "seeing conditions"),
    ("the textile mill", "loom tension", "dye batches", "humidity control"),
    ("the cheese cellar", "rind washing", "shelf rotation", "ambient moisture"),
    ("the bell foundry", "mould drying", "tuning cuts", "alloy ratios"),
    ("the seed bank", "germination trials", "freezer logs", "accession numbers"),
]
FRAMES = [
    "Staff at {0} spent the quarter reviewing {1}, since the previous method had "
    "produced inconsistent records. The revised procedure assigns responsibility "
    "to a named officer and requires a countersignature before any entry is closed.",
    "A recurring difficulty at {0} concerns {2}. Measurements taken in the morning "
    "rarely match those taken after midday, and the discrepancy has been attributed "
    "to the ventilation schedule rather than to the instruments themselves.",
    "The committee overseeing {0} agreed that {3} should be standardised. Until now "
    "each department had used its own conventions, which made the annual "
    "reconciliation slow and, on two occasions, produced duplicate entries.",
    "During the refurbishment of {0}, work on {1} was suspended for eleven weeks. "
    "The backlog was cleared by temporary staff, whose notes were later found to "
    "omit the customary cross-references.",
    "An internal review of {0} recommended that {2} be logged twice daily rather "
    "than weekly. The additional effort was judged acceptable given how often the "
    "earlier records had to be corrected retrospectively.",
    "Correspondence held by {0} shows that {3} was debated as early as the previous "
    "decade. The proposals were shelved because the necessary equipment was not "
    "available at a price the budget allowed.",
]


def make_haystack(seed=0, n_paragraphs=49, sents_per_para=6):
    """Paragraphs of several sentences each.

    Block size matters here and is not a free choice: BlockGraph's text mode
    makes one node per sentence, so a two-sentence paragraph yields n = 2 and
    H is bounded by ln 2 -- the negentropy J = 1 - H/ln n then has no room to
    discriminate.  Blocks must carry enough sentences for the measure to mean
    anything; this was found by the §C.3 check before the run, not after.
    """
    rng = random.Random(seed)
    sentences = []
    for i, t in enumerate(TOPICS):
        for j, f in enumerate(FRAMES):
            sentences.extend(s.strip() + "." for s in f.format(*t).split(". ")
                             if len(s.strip()) > 20)
    rng.shuffle(sentences)
    paras, cur = [], []
    i = 0
    while len(paras) < n_paragraphs:
        cur.append(sentences[i % len(sentences)])
        i += 1
        if len(cur) == sents_per_para:
            paras.append(" ".join(cur))
            cur = []
    return paras


def make_needles(n=N_NEEDLES, seed=1):
    """Each needle answer is one unambiguous string, scored by substring match."""
    rng = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    names = ["ALPHA", "BRAVO", "CORAL", "DELTA", "ECHO", "FLINT", "GRANITE",
             "HALCYON", "INDIGO", "JUNIPER"]
    out, used = [], set()
    while len(out) < n:
        code = "".join(rng.choice(alphabet) for _ in range(6))
        if code in used:
            continue
        used.add(code)
        server = f"{names[len(out) % len(names)]}-{100 + len(out)}"
        out.append({
            "id": len(out),
            "server": server,
            "code": code,
            "sentence": f"The access code for server {server} is {code}.",
            "question": f"What is the access code for server {server}?",
        })
    return out


def insert(paras, needle_sentence, pct):
    """Put the needle INSIDE a paragraph, not as a paragraph of its own.

    A lone sentence makes a one-node graph, whose negentropy is zero by
    definition -- the needle block would then be unselectable in condition B
    for a purely mechanical reason.  Burying it mid-paragraph is also the more
    realistic placement.
    """
    out = list(paras)
    at = max(0, min(len(out) - 1, round((len(out) - 1) * pct / 100)))
    sents = out[at].split(". ")
    mid = max(1, len(sents) // 2)
    out[at] = ". ".join(sents[:mid] + [needle_sentence.rstrip(".")] + sents[mid:])
    return out, at


# ----------------------------------------------------------- selection ------

def negentropy(block_text, question, task_type="general", alpha=ALPHA):
    """J = 1 - H(pi)/ln(n) with pi from PPR personalised BY THE QUESTION."""
    g = BlockGraph().build(block_text, task_type)
    if g.n <= 1:
        return 0.0, 1, 0.0
    P = g.transition_matrix()
    v = g.personalization_vector(question, task_type)
    pi = run_ppr(P, v, alpha=alpha).pi
    H, _ = shannon_entropy(pi)
    ln_n = math.log(g.n)
    J = 1.0 - H / ln_n if ln_n > 0 else 0.0
    return float(J), g.n, float(H)


def select(blocks, scores, budget, count_tokens):
    """Take blocks by descending score, then restore the original order."""
    order = sorted(range(len(blocks)), key=lambda i: -scores[i])
    chosen, used = [], 0
    for i in order:
        n = count_tokens(blocks[i])
        if used + n > budget:
            continue
        chosen.append(i)
        used += n
        if used >= budget:
            break
    chosen.sort()
    return chosen, used


def cosine_scores(blocks, question):
    """Plain lexical cosine, the ordinary-retriever control (condition D)."""
    def bag(s):
        w = [t for t in "".join(c.lower() if c.isalnum() else " "
                                for c in s).split() if len(t) > 2]
        d = {}
        for t in w:
            d[t] = d.get(t, 0) + 1
        return d
    q = bag(question)
    qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
    out = []
    for b in blocks:
        d = bag(b)
        dn = math.sqrt(sum(v * v for v in d.values())) or 1.0
        dot = sum(v * d.get(k, 0) for k, v in q.items())
        out.append(dot / (qn * dn))
    return out


# --------------------------------------------------------------- check ------

def check_personalisation(seed=0):
    """§C.3 insists this be verified explicitly before the full run.

    With a uniform v and a regular P the stationary pi is uniform too, H = ln n,
    J = 0, and the whole mechanism is silent.  The question-personalised v must
    give a visibly different pi.
    """
    paras = make_haystack(seed)
    needles = make_needles()
    nd = needles[0]
    block = nd["sentence"] + " " + paras[0]
    g = BlockGraph().build(block, "general")
    P = g.transition_matrix()
    v_q = g.personalization_vector(nd["question"], "general")
    v_u = np.ones(g.n) / g.n
    pi_q = run_ppr(P, v_q, alpha=ALPHA).pi
    pi_u = run_ppr(P, v_u, alpha=ALPHA).pi
    Hq, _ = shannon_entropy(pi_q)
    Hu, _ = shannon_entropy(pi_u)
    ln_n = math.log(g.n)
    print("=== §C.3 personalisation check ===")
    print(f"  block nodes n = {g.n},  ln n = {ln_n:.4f}")
    print(f"  uniform v : H = {Hu:.4f}  J = {1 - Hu/ln_n:.4f}  "
          f"max|v - 1/n| = {np.abs(v_u - 1/g.n).max():.2e}")
    print(f"  question v: H = {Hq:.4f}  J = {1 - Hq/ln_n:.4f}  "
          f"max|v - 1/n| = {np.abs(v_q - 1/g.n).max():.2e}")
    print(f"  pi differs between the two: "
          f"max|dpi| = {np.abs(pi_q - pi_u).max():.3e}")
    spread = np.abs(v_q - 1 / g.n).max()
    if spread < 1e-6:
        print("  FAIL: the personalisation vector is uniform — J would be the "
              "same for every block and the mechanism cannot discriminate")
        return False
    print("  OK: v is question-dependent, so J can discriminate between blocks")
    return True


# ------------------------------------------------------------- runner ------

PROMPT = ("Read the notes below and answer the question using only what they "
          "say.\n\n--- NOTES ---\n{ctx}\n--- END NOTES ---\n\n"
          "Question: {q}\nAnswer with the code only.\nAnswer:")


def build_trials(n_needles, positions, seed=0):
    paras = make_haystack(seed)
    needles = make_needles(n_needles)
    return paras, [(nd, pos) for nd in needles for pos in positions]


def run_experiment(out_json, n_needles=N_NEEDLES, budget=BUDGET_TOKENS,
                   conditions=("A", "B", "C", "D"), limit=None, seed=0):
    import torch
    import probe
    torch.set_grad_enabled(False)
    tok, model, cfg = probe.load("B")
    dev = "cuda" if cfg["device"] == "cuda" else "cpu"
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    paras, trials = build_trials(n_needles, POSITIONS, seed)
    if limit:
        trials = trials[:limit]
    rng = np.random.default_rng(seed)
    print(f"{len(paras)} paragraphs, haystack "
          f"{ntok(chr(10).join(paras))} tokens, {len(trials)} trials, "
          f"conditions {conditions}", flush=True)

    rows = []
    from tqdm.auto import tqdm
    bar = tqdm(total=len(trials) * len(conditions), unit="trial")
    for nd, pos in trials:
        blocks, at = insert(paras, nd["sentence"], pos)
        full = "\n\n".join(blocks)
        j_scores = None
        for cond in conditions:
            if cond == "A":
                ctx, picked = full, list(range(len(blocks)))
            else:
                if cond == "B":
                    if j_scores is None:
                        j_scores = [negentropy(b, nd["question"])[0]
                                    for b in blocks]
                    sc = j_scores
                elif cond == "C":
                    sc = list(rng.random(len(blocks)))
                else:
                    sc = cosine_scores(blocks, nd["question"])
                picked, used = select(blocks, sc, budget, ntok)
                ctx = "\n\n".join(blocks[i] for i in picked)
            text = PROMPT.format(ctx=ctx, q=nd["question"])
            enc = tok(text, return_tensors="pt").to(dev)
            out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            ans = tok.decode(out[0, enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            rows.append(dict(needle=nd["id"], server=nd["server"],
                             code=nd["code"], position=pos, condition=cond,
                             ctx_tokens=int(enc["input_ids"].shape[1]),
                             needle_kept=bool(at in picked),
                             correct=bool(nd["code"] in ans),
                             answer=ans.strip()[:60]))
            bar.update(1)
    bar.close()
    Path(out_json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {out_json} ({len(rows)} rows)")
    return rows


# ------------------------------------------------------------ analysis -----

MIDDLE = [35, 50, 65]


def analyse(raw_json, out_md):
    rows = json.loads(Path(raw_json).read_text(encoding="utf-8"))
    conds = sorted({r["condition"] for r in rows})
    poss = sorted({r["position"] for r in rows})

    def acc(cond, positions=None):
        sel = [r for r in rows if r["condition"] == cond
               and (positions is None or r["position"] in positions)]
        return (sum(r["correct"] for r in sel) / len(sel), len(sel)) if sel \
            else (float("nan"), 0)

    L = ["# Часть C — дробление контекста против потери внимания в середине", ""]
    L.append("| условие | " + " | ".join(f"{p} %" for p in poss) +
             " | все | середина (35/50/65) | токенов контекста |")
    L.append("|---" * (len(poss) + 4) + "|")
    names = {"A": "A. полный линейный контекст", "B": "B. дробление + негэнтропия",
             "C": "C. дробление + случайный отбор", "D": "D. дробление + косинус"}
    tok_avg = {}
    for c in conds:
        cells = " | ".join(f"{acc(c, [p])[0]:.3f}" for p in poss)
        sel = [r for r in rows if r["condition"] == c]
        tok_avg[c] = np.mean([r["ctx_tokens"] for r in sel])
        L.append(f"| {names.get(c, c)} | {cells} | {acc(c)[0]:.3f} | "
                 f"{acc(c, MIDDLE)[0]:.3f} | {tok_avg[c]:.0f} |")
    L.append("")

    res = {"n_trials_per_condition": acc(conds[0])[1]}
    for c in conds:
        a_all = [acc(c, [p])[0] for p in poss]
        res[f"U_depth_{c}"] = float(min(a_all) / max(max(a_all), 1e-9))
        res[f"acc_{c}"] = acc(c)[0]
        res[f"acc_mid_{c}"] = acc(c, MIDDLE)[0]
    L.append("| условие | глубина U (min/max по позициям) |")
    L.append("|---|---|")
    for c in conds:
        L.append(f"| {c} | {res[f'U_depth_{c}']:.3f} |")
    L.append("")

    if "B" in conds and "C" in conds:
        gap_mid = (res["acc_mid_B"] - res["acc_mid_C"]) * 100
        gap_all = (res["acc_B"] - res["acc_C"]) * 100
        res["gap_BC_mid_pp"] = gap_mid
        res["gap_BC_all_pp"] = gap_all
        L.append(f"**Разрыв B − C на средних позициях = {gap_mid:+.1f} п.п.** "
                 f"(по всем позициям {gap_all:+.1f} п.п.)")
        L.append("")
        if gap_mid >= 10:
            v = ("негэнтропийный отбор работает, механизм подтверждён")
        elif abs(gap_mid) < 10 and res["acc_B"] > res["acc_A"] and \
                res["acc_C"] > res["acc_A"]:
            v = ("помогает дробление и сжатие, негэнтропия ни при чём")
        elif gap_mid < -10:
            v = "отбор по негэнтропии активно вредит"
        else:
            v = "дробление не помогает вовсе"
        L.append(f"Вывод по таблице §C.4: **{v}**.")
        L.append("")
    if "A" in conds and "B" in conds:
        res["gap_BA_mid_pp"] = (res["acc_mid_B"] - res["acc_mid_A"]) * 100
        L.append(f"Разрыв B − A на средних позициях = "
                 f"{res['gap_BA_mid_pp']:+.1f} п.п.")
        L.append("")

    # how often the selector actually kept the needle -- separates "the
    # selector missed it" from "the model had it and failed to use it"
    L.append("| условие | иголка попала в контекст | точность при попавшей "
             "иголке | точность при не попавшей |")
    L.append("|---|---|---|---|")
    for c in conds:
        sel = [r for r in rows if r["condition"] == c]
        kept = [r for r in sel if r["needle_kept"]]
        lost = [r for r in sel if not r["needle_kept"]]
        res[f"kept_{c}"] = len(kept) / max(len(sel), 1)
        L.append(f"| {c} | {len(kept)}/{len(sel)} = {res[f'kept_{c}']:.3f} | "
                 + (f"{sum(r['correct'] for r in kept)/len(kept):.3f} | "
                    if kept else "— | ")
                 + (f"{sum(r['correct'] for r in lost)/len(lost):.3f} |"
                    if lost else "— |"))
    L.append("")
    Path(out_md).write_text("\n".join(L), encoding="utf-8")
    Path(str(out_md).replace(".md", ".json")).write_text(
        json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--needles", type=int, default=N_NEEDLES)
    ap.add_argument("--out", default=str(ROOT / "out" / "needle_raw.json"))
    a = ap.parse_args()
    if a.run:
        run_experiment(a.out, n_needles=a.needles, limit=a.limit)
        sys.exit(0)
    if a.analyse:
        analyse(a.out, str(ROOT / "out" / "needle_report.md"))
        sys.exit(0)
    if a.check:
        ok = check_personalisation()
        paras = make_haystack()
        nd = make_needles()[0]
        chars = sum(len(p) for p in paras)
        print(f"\nhaystack: {len(paras)} paragraphs, {chars} chars "
              f"(~{chars//4} tokens)")
        print(f"needle  : {nd['sentence']}")
        print(f"question: {nd['question']}")
        blocks, at = insert(paras, nd["sentence"], 50)
        res = [negentropy(b, nd["question"]) for b in blocks]
        arr = np.array([r[0] for r in res])
        ns = np.array([r[1] for r in res])
        rank = int((arr > arr[at]).sum())
        print(f"nodes per block: min={ns.min()} max={ns.max()} "
              f"mean={ns.mean():.1f}")
        print(f"J over {len(blocks)} blocks: min={arr.min():.4f} "
              f"max={arr.max():.4f} mean={arr.mean():.4f} std={arr.std():.4f}")
        print(f"J of the needle-bearing block: {arr[at]:.4f} "
              f"(n={ns[at]}), rank {rank+1} of {len(blocks)}")
        if arr.std() < 1e-3:
            print("  VERDICT: J does not discriminate between blocks — "
                  "selection by negentropy would be arbitrary and condition B "
                  "would equal condition C by construction, not by measurement")
            ok = False
        else:
            print("  VERDICT: J varies across blocks, selection can discriminate")
        sys.exit(0 if ok else 1)
