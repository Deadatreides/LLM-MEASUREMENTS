"""Stage 3 — the remaining weight-side numbers.  CPU only, no inference.

3.1  Q(C) = sigma_1^2 / sum_i sigma_i^2 over each cluster's gate_proj rows.
     This is the accuracy of the rank-1 probe g_C = sigma_1 |<v_1, x>| that
     would replace |C| dot products with one.  Control: random clusters of the
     same size must give Q ~ 1/|C|.

3.2  Run lengths of consecutive active clusters after permuting channels into
     cluster order, waste(k) = a_clust(k)/p, and traffic per token relative to
     k = 48.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
KS = [2, 4, 8, 16, 32]


def q_for_labels(gate, labels, n_clusters):
    """Energy share of the leading singular value, per cluster."""
    out = []
    order = np.argsort(labels, kind="stable")
    lab_sorted = labels[order]
    bounds = np.searchsorted(lab_sorted, np.arange(n_clusters + 1))
    for c in range(n_clusters):
        idx = order[bounds[c]:bounds[c + 1]]
        if len(idx) < 2:
            continue
        s = np.linalg.svd(gate[idx], compute_uv=False)
        tot = float((s ** 2).sum())
        if tot > 0:
            out.append(float(s[0] ** 2 / tot))
    return out


def stage31(sigma_path, out_json, layers=None, seed=0):
    import clusters as cl
    z = np.load(sigma_path)
    sigma = z["sigma"]
    L = int(z["n_layers"][0])
    gates = [z[f"gate_{i}"] for i in range(L)]
    d_ffn = gates[0].shape[0]
    layers = layers or list(range(0, L, 4))
    print(f"3.1 Q(k): layers {layers}, d_ffn {d_ffn}", flush=True)

    res = {}
    rng = np.random.default_rng(seed)
    for k in KS:
        n_clusters = d_ffn // k
        q_real, q_rand = [], []
        for li in layers:
            Z, _, _ = cl.embeddings(sigma[li], gates[li])
            lab = cl.cluster_labels(cl.pca_reduce(Z, seed), n_clusters, seed)
            q_real += q_for_labels(gates[li], lab, n_clusters)
            rnd = np.empty(d_ffn, np.int32)
            perm = rng.permutation(d_ffn)
            rnd[perm] = np.arange(d_ffn) // k
            q_rand += q_for_labels(gates[li], rnd, n_clusters)
        qr = np.array(q_real)
        qn = np.array(q_rand)
        res[k] = dict(
            Q_mean=float(qr.mean()), Q_std=float(qr.std()),
            Q_p10=float(np.percentile(qr, 10)),
            Q_p90=float(np.percentile(qr, 90)),
            Q_random_mean=float(qn.mean()), baseline=1.0 / k,
            n_clusters_measured=int(len(qr)))
        print(f"  k={k:3d}  Q={qr.mean():.4f} ± {qr.std():.4f}  "
              f"[p10 {np.percentile(qr,10):.3f}, p90 {np.percentile(qr,90):.3f}]"
              f"   random clusters {qn.mean():.4f}   1/|C| = {1/k:.4f}",
              flush=True)
    Path(out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {out_json}")
    return res


def stage32(trace_path, sigma_path, out_json, ks=(4, 8, 16), n_sample=2000,
            seed=0):
    """Run lengths of active clusters, waste, and traffic per token."""
    import clusters as cl
    import sim
    z = np.load(sigma_path)
    sigma = z["sigma"]
    L = int(z["n_layers"][0])
    gates = [z[f"gate_{i}"] for i in range(L)]
    tr = sim.Trace(trace_path)
    d_ffn = tr.d_ffn
    rng = np.random.default_rng(seed)
    toks = rng.choice(tr.n_tok, size=min(n_sample, tr.n_tok), replace=False)
    p = float(tr.band_cnt[toks].mean() / d_ffn)
    print(f"\n3.2 runs and traffic: p={p:.4f}, {len(toks)} tokens sampled",
          flush=True)

    res = {"p": p}
    for k in ks:
        n_clusters = d_ffn // k
        runs_all, act_all = [], []
        for li in range(0, L, 4):
            Z, _, _ = cl.embeddings(sigma[li], gates[li])
            lab = cl.cluster_labels(cl.pca_reduce(Z, seed), n_clusters, seed)
            order = np.argsort(lab, kind="stable")
            rows = tr._bitrows[li][toks]
            act = np.unpackbits(rows, axis=1)[:, :d_ffn].astype(bool)
            for i in range(act.shape[0]):
                hit = np.zeros(n_clusters, bool)
                hit[lab[act[i]]] = True
                act_all.append(hit.mean())
                m = hit.astype(np.int8)
                pad = np.zeros(m.size + 2, np.int8)
                pad[1:-1] = m
                d = np.diff(pad)
                runs_all += (np.flatnonzero(d == -1)
                             - np.flatnonzero(d == 1)).tolist()
        a = float(np.mean(act_all))
        r = np.array(runs_all)
        res[k] = dict(
            a_clust=a, waste=a / p,
            run_mean=float(r.mean()), run_median=float(np.median(r)),
            run_predicted=1.0 / max(1 - a, 1e-9),
            txn_units_mean=float(r.mean() * k),
            traffic_rel_k48=float(a * k / (0.9618 * 48)))
        print(f"  k={k:3d}  a_clust={a:.4f} waste={a/p:5.2f} "
              f"run mean {r.mean():.2f} median {np.median(r):.0f} "
              f"(1/(1-a) = {1/max(1-a,1e-9):.2f})  "
              f"effective transaction {r.mean()*k:.0f} channels  "
              f"traffic vs k=48: {a*k/(0.9618*48):.3f}", flush=True)
    Path(out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {out_json}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="31", choices=["31", "32"])
    ap.add_argument("--sigma", default=str(ROOT / "out" / "B_sigma.npz"))
    ap.add_argument("--trace", default=str(ROOT / "traces" / "Bch.npz"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.part == "31":
        stage31(a.sigma, a.out or str(ROOT / "out" / "stage3_Q.json"))
    else:
        stage32(a.trace, a.sigma, a.out or str(ROOT / "out" / "stage3_runs.json"))
