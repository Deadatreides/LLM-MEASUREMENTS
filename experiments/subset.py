"""Cut a trace down to part A or part B of the corpus (task 2, A.3).

Part A is 510 single-turn requests of maximally dissimilar prompts; part B is
20 coherent five-turn sessions.  A.3 asks whether the corpus's own diversity
was masking adaptivity: with domains interleaved an adaptive policy has nothing
to track and must look like a static one whatever the routing does.

Writes a new .npz with the same schema, so sim.py needs no changes.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def load_parts():
    corpus = json.loads((ROOT / "corpus.json").read_text(encoding="utf-8"))
    return {c["request_id"]: c["part"] for c in corpus}


def subset(src, dst, part):
    z = np.load(src, allow_pickle=False)
    parts = load_parts()
    rid = z["request_id"]
    keep = np.array([parts.get(int(r), "?") == part for r in rid], dtype=bool)
    if not keep.any():
        raise SystemExit(f"no tokens of part {part} in {src}")

    out = {}
    for k in z.files:
        a = z[k]
        if k.startswith("band_ids_"):
            continue                      # rebuilt below from the counts
        if a.ndim >= 1 and a.shape[0] == len(rid):
            out[k] = a[keep]
        else:
            out[k] = a                    # scalars and metadata

    if "band_cnt" in z.files:
        cnt = z["band_cnt"].astype(np.int64)
        L = cnt.shape[1]
        for li in range(L):
            ids = z[f"band_ids_{li}"]
            off = np.concatenate([[0], np.cumsum(cnt[:, li])])
            idx = np.nonzero(keep)[0]
            lens = cnt[idx, li]
            starts = off[idx]
            base = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]),
                             lens)
            out[f"band_ids_{li}"] = ids[base + np.arange(lens.sum())]

    np.savez_compressed(dst, **out)
    n_req = len(np.unique(rid[keep]))
    print(f"{Path(src).name} part {part}: {keep.sum()} of {len(rid)} tokens, "
          f"{n_req} requests -> {Path(dst).name} "
          f"({Path(dst).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    tag, part = sys.argv[1], sys.argv[2]
    subset(ROOT / "traces" / f"{tag}.npz",
           ROOT / "traces" / f"{tag}-{part}.npz", part)
