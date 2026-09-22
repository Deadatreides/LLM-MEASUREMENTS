"""Stage 1 — haystack calibration.  Condition A only.

"Lost in the middle" is documented on large models.  Whether Qwen3-1.7B shows
it at all is unknown, and if it does not there is nothing for stage 2 to
straighten.  This measures the U-curve depth at three haystack sizes before
five hours are spent.

    5 needles x 5 positions x 3 sizes {2000, 4000, 6000} = 75 trials
    U depth = min(acc) / max(acc) over positions

    <= 0.70       usable, take the smallest such size
    0.70-0.85     weak U, take the largest size
    > 0.85 all    no U-curve; stage 2 must be reformulated as a plain
                  compression-quality measurement
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import stage0bis                                   # noqa: E402

POSITIONS = [5, 30, 50, 70, 95]
SIZES = [2000, 4000, 6000]
PROMPT = ("Read the notes below and answer the question using only what they "
          "say.\n\n--- NOTES ---\n{ctx}\n--- END NOTES ---\n\n"
          "Question: {q}\nAnswer with the identifier only.\nAnswer:")


def pool_for(tok, target_tokens, seed=0):
    """Enough synthetic haystack sentences to hit a token budget."""
    base = stage0bis.sentence_pool(seed)
    out, n = [], 0
    i = 0
    while n < target_tokens:
        s = base[i % len(base)]
        out.append(s)
        n += len(tok(s, add_special_tokens=False)["input_ids"])
        i += 1
    return out


def run(n_needles=5, out_json=None, seed=0):
    import torch
    import probe
    torch.set_grad_enabled(False)
    tok, model, cfg = probe.load("B")
    dev = "cuda" if cfg["device"] == "cuda" else "cpu"
    needles = stage0bis.make_needles(n_needles)

    pools = {}
    for H in SIZES:
        p = pool_for(tok, H, seed)
        n = len(tok("\n\n".join(p), add_special_tokens=False)["input_ids"])
        pools[H] = p
        print(f"haystack {H}: {len(p)} sentences, {n} tokens", flush=True)

    total = len(SIZES) * n_needles * len(POSITIONS)
    print(f"{total} trials, condition A only", flush=True)

    rows = []
    t0 = time.time()
    from tqdm.auto import tqdm
    bar = tqdm(total=total, unit="trial")
    for H in SIZES:
        for nd in needles:
            for pos in POSITIONS:
                sents, _ = stage0bis.insert_sentence(pools[H], nd["sentence"],
                                                     pos)
                ctx = "\n\n".join(sents)
                text = PROMPT.format(ctx=ctx, q=nd["question"])
                enc = tok(text, return_tensors="pt").to(dev)
                try:
                    out = model.generate(
                        **enc, max_new_tokens=24, do_sample=False,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                    ans = tok.decode(out[0, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
                    ok = nd["code"] in ans
                    failed = False
                except Exception as e:                # never abort the sweep
                    ans, ok, failed = f"{type(e).__name__}: {e}", False, True
                rows.append(dict(size=H, needle=nd["id"], position=pos,
                                 ctx_tokens=int(enc["input_ids"].shape[1]),
                                 correct=bool(ok), failed=failed,
                                 answer=ans.strip()[:60]))
                bar.update(1)
                bar.set_postfix_str(f"{time.time()-t0:.0f}s")
    bar.close()

    res = {"positions": POSITIONS, "sizes": SIZES, "by_size": {}}
    print("\naccuracy by position, condition A")
    print(f"{'size':>6s} " + " ".join(f"{p:>7d}%" for p in POSITIONS) +
          f" {'all':>7s} {'U depth':>8s}")
    for H in SIZES:
        accs = []
        for p in POSITIONS:
            sel = [r for r in rows if r["size"] == H and r["position"] == p]
            accs.append(np.mean([r["correct"] for r in sel]) if sel else np.nan)
        allacc = np.mean([r["correct"] for r in rows if r["size"] == H])
        depth = float(min(accs) / max(max(accs), 1e-9))
        res["by_size"][H] = dict(acc=[float(a) for a in accs],
                                 acc_all=float(allacc), U_depth=depth)
        print(f"{H:6d} " + " ".join(f"{a:8.3f}" for a in accs) +
              f" {allacc:7.3f} {depth:8.3f}")

    usable = [H for H in SIZES if res["by_size"][H]["U_depth"] <= 0.70]
    weak = [H for H in SIZES if 0.70 < res["by_size"][H]["U_depth"] <= 0.85]
    if usable:
        chosen, why = min(usable), "U depth <= 0.70, smallest such size"
    elif weak:
        chosen, why = max(weak), "weak U, largest size"
    else:
        chosen, why = max(SIZES), ("NO U-CURVE at any size — stage 2 must be "
                                   "reformulated as a compression-quality "
                                   "measurement, without the 'middle' claim")
    res["chosen_H"] = chosen
    res["verdict"] = why
    print(f"\nchosen haystack H = {chosen}: {why}")
    if any(r["failed"] for r in rows):
        print(f"failed trials: {sum(r['failed'] for r in rows)}")

    out_json = out_json or str(ROOT / "out" / "stage1.json")
    Path(out_json).write_text(json.dumps(dict(summary=res, rows=rows), indent=1),
                              encoding="utf-8")
    print(f"wrote {out_json}  ({time.time()-t0:.0f} s)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--needles", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.needles, a.out)
