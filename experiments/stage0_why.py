"""Two checks on the stage-0 result before it is believed.

1. Is `at` really the block holding the needle?  If the index were wrong the
   rank would be meaningless.
2. R is not merely at chance, it is below it (rank fraction 0.77, r = -0.11).
   Something must be pushing the needle-bearing block DOWN the J ordering.
   Comparing each paragraph with and without the needle isolates it.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import needle as nd_mod        # noqa: E402
import stage0                  # noqa: E402

paras = nd_mod.make_haystack(0)
needles = stage0.make_needles(20)

print("=== check 1: does block `at` contain the needle? ===")
bad = 0
for nd in needles[:20]:
    for pos in stage0.POSITIONS:
        blocks, at = nd_mod.insert(paras, nd["sentence"], pos)
        if nd["code"] not in blocks[at]:
            bad += 1
        if sum(nd["code"] in b for b in blocks) != 1:
            bad += 1
print(f"  mismatches over {20*len(stage0.POSITIONS)} trials: {bad}")
assert bad == 0, "the needle index is wrong; stage 0 would be meaningless"

print("\n=== check 2: what does inserting the needle do to J? ===")
d_json, d_n, d_edges = [], [], []
for nd in needles[:20]:
    for pos in stage0.POSITIONS:
        blocks, at = nd_mod.insert(paras, nd["sentence"], pos)
        with_n = stage0.block_stats(blocks[at], nd["question"])
        without = stage0.block_stats(paras[at], nd["question"])
        d_json.append(with_n["J"] - without["J"])
        d_n.append(with_n["n"] - without["n"])
        d_edges.append(with_n["edges"] - without["edges"])
d = np.array(d_json)
print(f"  dJ from inserting the needle: mean {d.mean():+.5f} "
      f"std {d.std():.5f}  negative in {100*(d<0).mean():.1f} % of trials")
print(f"  dn (nodes) mean {np.mean(d_n):+.2f}, "
      f"dedges mean {np.mean(d_edges):+.2f}")

print("\n=== check 3: what DOES J rank highly? ===")
nd = needles[0]
blocks, at = nd_mod.insert(paras, nd["sentence"], 50)
st = [stage0.block_stats(b, nd["question"]) for b in blocks]
J = np.array([s["J"] for s in st])
n = np.array([s["n"] for s in st])
e = np.array([s["edges"] for s in st])
order = np.argsort(-J)
print(f"  corr(J, n_nodes)  = {np.corrcoef(J, n)[0,1]:+.3f}")
print(f"  corr(J, n_edges)  = {np.corrcoef(J, e)[0,1]:+.3f}")
print(f"  corr(J, chars)    = "
      f"{np.corrcoef(J, [len(b) for b in blocks])[0,1]:+.3f}")
kw = [len(stage0._keywords(b) & stage0._keywords(nd['question']))
      for b in blocks]
print(f"  corr(J, shared keywords with question) = "
      f"{np.corrcoef(J, kw)[0,1]:+.3f}   (question overlap ranges "
      f"{min(kw)}..{max(kw)})")
print(f"  top-3 blocks by J have n = {n[order[:3]].tolist()}, "
      f"bottom-3 have n = {n[order[-3:]].tolist()}")
print(f"  needle block: J={J[at]:.4f} rank {int(np.where(order==at)[0][0])+1} "
      f"of {len(blocks)}, n={n[at]}, edges={e[at]}")
