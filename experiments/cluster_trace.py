"""Re-express the channel trace over co-activation clusters — task 2, A.4.

A.2 found S notably above zero at small group sizes (0.63 at k=4, 0.47 at k=8),
which is the condition A.4 sets for running the cache simulator on cluster
units.  Nothing is re-traced: a cluster is active exactly when one of its
channels is, so the cluster bitmap follows from the channel bitmap.

The eigendecomposition is reused from out/B_sigma.npz, so only k-means runs.
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def build(k, sigma_path, chan_path, out_path, seed=0):
    import clusters as cl
    import sim

    z = np.load(sigma_path)
    sigma = z["sigma"]
    L = int(z["n_layers"][0])
    gates = [z[f"gate_{i}"] for i in range(L)]

    tr = sim.Trace(chan_path)
    assert tr.unit == "channel", "needs the channel-granularity trace"
    d_ffn, T = tr.d_ffn, tr.n_tok
    n_clusters = d_ffn // k
    print(f"k={k}: {n_clusters} clusters per layer, {L} layers, {T} tokens",
          flush=True)

    src = np.load(chan_path, allow_pickle=False)
    out = {kk: src[kk] for kk in src.files
           if not kk.startswith("band_ids_") and kk not in
           ("n_bands", "band_s", "unit", "chan_cnt", "band_cnt")}

    labels_all = np.zeros((L, d_ffn), np.int32)
    cnt = np.zeros((T, L), np.int16)
    top1 = np.zeros((T, L), np.int64)
    chan_top1 = src["chan_cnt"]                 # dominant channel per token/layer

    for li in range(L):
        Z, _, _ = cl.embeddings(sigma[li], gates[li])
        lab = cl.cluster_labels(cl.pca_reduce(Z, seed), n_clusters, seed)
        labels_all[li] = lab
        rows = tr._bitrows[li]
        packed = np.zeros((T, (n_clusters + 7) // 8), np.uint8)
        step = 4096
        for i in range(0, T, step):
            act = np.unpackbits(rows[i:i + step], axis=1)[:, :d_ffn].astype(bool)
            hit = np.zeros((act.shape[0], n_clusters), bool)
            for j in range(act.shape[0]):
                hit[j, lab[act[j]]] = True
            packed[i:i + step] = np.packbits(hit, axis=1)
            cnt[i:i + step, li] = hit.sum(1)
        out[f"band_ids_{li}"] = packed
        top1[:, li] = lab[chan_top1[:, li]]
        print(f"  layer {li}/{L}: mean active clusters "
              f"{cnt[:, li].mean():.1f}/{n_clusters}", flush=True)

    out["n_bands"] = np.array([n_clusters])
    out["band_s"] = np.array([k])
    out["unit"] = np.array(["cluster"])
    out["band_cnt"] = cnt
    out["chan_cnt"] = top1.astype(np.int32)
    np.savez_compressed(out_path, **out)
    np.save(str(out_path).replace(".npz", ".labels.npy"), labels_all)
    print(f"wrote {out_path} ({Path(out_path).stat().st_size/1e6:.1f} MB), "
          f"mean active {cnt.mean(1).mean():.1f} of {n_clusters} per layer")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sigma", default=str(ROOT / "out" / "B_sigma.npz"))
    ap.add_argument("--chan", default=str(ROOT / "traces" / "Bch.npz"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or str(ROOT / "traces" / f"Bcl{a.k}.npz")
    build(a.k, a.sigma, a.chan, out)
