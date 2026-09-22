"""Co-activation clusters and S(k) — task 2, part A.2.

The point of A.2: contiguous channels are unrelated, so grouping them averages
sparsity away (a_contig(k) ~ 1-(1-p)^k, which is ~1 already at k=48).  Members
of a co-activation cluster fire together, so grouping *them* should preserve
sparsity.  The difference is the measurable claim.

    S(k) = (a_contig(k) - a_clust(k)) / (a_contig(k) - p)

  S -> 1  clustering fully preserves sparsity: strong structure
  S ~ .5  partial structure
  S -> 0  clusters no better than contiguous slabs: no structure

Clusters are built from the WEIGHTS, with no activation trace:
    1. Sigma_x per layer from a prefill of the corpus (rank 256)
    2. Sigma_x ~ U L U^T
    3. Z = W_gate · U · L^(1/2)          channel embedding, R^(d_ffn x 256)
    4. PCA 256->64, k-means into d_ffn/k clusters

Also splits off the globally dominant directions, which L^(1/2) amplifies:
    Z_glob = W · U[:, :5]   · L^(1/2)
    Z_cond = W · U[:, 5:]   · L^(1/2)      clustered after normalisation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
KS = [4, 8, 16, 32, 48, 64]
RANK = 256
PCA_DIM = 64
N_GLOB = 5


# ------------------------------------------------------- covariance pass ----

@torch.no_grad()
def covariance(tag, n_tokens=5000, chunk=512):
    """Sigma_x per layer: the second moment of the FFN input."""
    import probe
    tok, model, cfg = probe.load(tag)
    dev = "cuda" if cfg["device"] == "cuda" else "cpu"
    _, layers = probe.find_layers(model)
    L = len(layers)
    H = model.config.hidden_size

    corpus = json.loads((ROOT / "corpus.json").read_text(encoding="utf-8"))
    text = "\n\n".join(probe.turn_prompt(c["turns"][0], i == 0)
                       for i, c in enumerate(corpus[:400]))
    ids = tok(text, return_tensors="pt")["input_ids"][0][:n_tokens]
    print(f"covariance pass: {len(ids)} tokens, {L} layers, hidden {H}")

    sig = torch.zeros(L, H, H, dtype=torch.float64)
    seen = 0
    store = {}
    handles = [lay.mlp.register_forward_pre_hook(
        lambda m, a, li=li: store.__setitem__(li, a[0].detach()))
        for li, lay in enumerate(layers)]
    for i in range(0, len(ids), chunk):
        piece = ids[i:i + chunk].unsqueeze(0).to(dev)
        model(input_ids=piece)
        for li in range(L):
            h = store[li].reshape(-1, H).double().cpu()
            sig[li] += h.T @ h
        seen += piece.shape[1]
    for h in handles:
        h.remove()
    sig /= max(seen, 1)

    gates = [lay.mlp.gate_proj.weight.detach().float().cpu().numpy()
             for lay in layers]                      # [d_ffn, hidden]
    del model
    torch.cuda.empty_cache()
    return sig.numpy(), gates, seen


def embeddings(sigma, gate, rank=RANK, n_glob=N_GLOB):
    """Z = W · U · L^(1/2), plus the split into global and conditional parts."""
    w, U = np.linalg.eigh(sigma)                    # ascending
    w = w[::-1][:rank].copy()
    U = U[:, ::-1][:, :rank].copy()
    w = np.clip(w, 0, None)
    root = np.sqrt(w)
    Z = gate @ (U * root[None, :])                  # [d_ffn, rank]
    Z_cond = gate @ (U[:, n_glob:] * root[None, n_glob:])
    nrm = np.linalg.norm(Z_cond, axis=1, keepdims=True)
    Z_cond = Z_cond / np.maximum(nrm, 1e-9)
    return Z, Z_cond, w


def pca_reduce(Z, seed=0, pca_dim=PCA_DIM):
    from sklearn.decomposition import PCA
    return PCA(n_components=min(pca_dim, Z.shape[1]),
               random_state=seed).fit_transform(Z)


def cluster_labels(X, n_clusters, seed=0):
    """MiniBatch k-means: n_clusters runs up to d_ffn/4 = 1536, where plain
    Lloyd with n_init=4 costs minutes per layer per k."""
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=seed,
                         batch_size=1024, max_iter=100).fit(X)
    return km.labels_.astype(np.int32)


# ------------------------------------------------------------- measure ------

def group_activity_batch(rows, labels, n_groups, d_ffn):
    """Mean fraction of groups holding at least one active channel.

    A segmented maximum over label-sorted channels would be faster but breaks
    when k-means leaves a cluster empty, which MiniBatchKMeans does at these
    cluster counts; scattering per token is correct regardless.
    """
    act = np.unpackbits(rows, axis=1)[:, :d_ffn].astype(bool)
    hit = np.zeros((act.shape[0], n_groups), dtype=bool)
    for i in range(act.shape[0]):
        hit[i, labels[act[i]]] = True
    return float(hit.mean())


def measure(trace, sigma, gates, out_json, n_sample=2000, seed=0):
    import sim
    tr = sim.Trace(trace)
    assert getattr(tr, "unit", "band") == "channel", \
        "A.2 needs the channel-granularity trace"
    d_ffn, L = tr.d_ffn, tr.L
    rng = np.random.default_rng(seed)
    toks = rng.choice(tr.n_tok, size=min(n_sample, tr.n_tok), replace=False)

    res = {"trace": str(Path(trace).name), "d_ffn": d_ffn, "layers": L,
           "tokens_sampled": len(toks), "k": {}}

    # p, the plain channel activity, is the floor every grouping is measured against
    p = float(tr.band_cnt[toks].mean() / d_ffn)
    res["p"] = p
    print(f"p (active channels, k=1) = {p:.4f}")

    # the eigendecomposition is per layer, not per k -- recomputing it for
    # every k was the whole cost of the first attempt
    print("embedding channels per layer ...", flush=True)
    X_full, X_cond = [], []
    for li in range(L):
        Z, Z_cond, _ = embeddings(sigma[li], gates[li])
        X_full.append(pca_reduce(Z, seed))
        X_cond.append(pca_reduce(Z_cond, seed))
        if li % 7 == 0:
            print(f"  layer {li}/{L}", flush=True)

    acc = {k: dict(c=0.0, k=0.0, kc=0.0, r=0.0) for k in KS}
    for li in range(L):
        rows = tr._bitrows[li][toks]
        for k in KS:
            n_groups = d_ffn // k
            lab_full = cluster_labels(X_full[li], n_groups, seed)
            lab_cond = cluster_labels(X_cond[li], n_groups, seed)
            contig = (np.arange(d_ffn) // k).astype(np.int32)
            perm = rng.permutation(d_ffn)
            rand = np.empty(d_ffn, np.int32)
            rand[perm] = np.arange(d_ffn) // k
            acc[k]["c"] += group_activity_batch(rows, contig, n_groups, d_ffn)
            acc[k]["k"] += group_activity_batch(rows, lab_full, n_groups, d_ffn)
            acc[k]["kc"] += group_activity_batch(rows, lab_cond, n_groups, d_ffn)
            acc[k]["r"] += group_activity_batch(rows, rand, n_groups, d_ffn)
        print(f"  clustered layer {li}/{L}", flush=True)

    for k in KS:
        n_groups = d_ffn // k
        a_c = acc[k]["c"] / L
        a_k = acc[k]["k"] / L
        a_kc = acc[k]["kc"] / L
        a_r = acc[k]["r"] / L
        denom = max(a_c - p, 1e-9)
        entry = dict(a_contig=a_c, a_clust=a_k, a_clust_cond=a_kc,
                     a_random=a_r, p=p,
                     S=(a_c - a_k) / denom, S_cond=(a_c - a_kc) / denom,
                     S_random=(a_c - a_r) / denom,
                     a_contig_predicted=1 - (1 - p) ** k)
        res["k"][k] = entry
        print(f"k={k:3d}  a_contig={a_c:.4f} (pred {entry['a_contig_predicted']:.4f})"
              f"  a_clust={a_k:.4f}  a_cond={a_kc:.4f}  a_random={a_r:.4f}"
              f"  ->  S={entry['S']:.3f}  S_cond={entry['S_cond']:.3f}"
              f"  [control S_random={entry['S_random']:.3f}]", flush=True)
        # A.2's own control: a random regrouping must behave like contiguous
        if abs(a_r - a_c) > 0.02:
            print(f"      WARNING: random grouping differs from contiguous by "
                  f"{abs(a_r-a_c):.3f}; §A.2 says these must agree")

    Path(out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {out_json}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="B")
    ap.add_argument("--trace", default=str(ROOT / "traces" / "Bch.npz"))
    ap.add_argument("--cov", default=str(ROOT / "out" / "B_sigma.npz"))
    ap.add_argument("--out", default=str(ROOT / "out" / "B_clusters.json"))
    ap.add_argument("--sample", type=int, default=2000)
    a = ap.parse_args()

    if Path(a.cov).exists():
        z = np.load(a.cov)
        sigma = z["sigma"]
        gates = [z[f"gate_{i}"] for i in range(int(z["n_layers"][0]))]
        print(f"loaded covariance from {a.cov}")
    else:
        sigma, gates, seen = covariance(a.tag)
        np.savez_compressed(a.cov, sigma=sigma, n_layers=np.array([len(gates)]),
                            tokens=np.array([seen]),
                            **{f"gate_{i}": g for i, g in enumerate(gates)})
        print(f"wrote {a.cov}")
    measure(a.trace, sigma, gates, a.out, n_sample=a.sample)
