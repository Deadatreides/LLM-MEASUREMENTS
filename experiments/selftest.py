"""Offline self-test of sim.py on synthetic traces with known structure.

Three regimes, so the simulator can be checked against answers we know:
  uniform  - every unit equally likely, no conditional structure at all
             -> STATIC ~ M, adaptive policies should NOT beat it much
  modal    - a hidden mode switches every ~40 tokens, each mode uses its own
             small set of units -> adaptive policies must beat STATIC clearly
  static   - one fixed hot set for the whole trace
             -> STATIC should already be near-perfect
"""

import numpy as np
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parent
TMP = ROOT / "out" / "selftest"
TMP.mkdir(parents=True, exist_ok=True)

L, E, K = 16, 64, 8
N_REQ, TOK = 60, 60
RNG = np.random.default_rng(0)


def make_moe(regime):
    T = N_REQ * TOK
    ids = np.zeros((T, L, K), dtype=np.uint8)
    rid = np.repeat(np.arange(N_REQ), TOK).astype(np.int32)
    n_modes = 6
    mode_sets = [[RNG.choice(E, K + 4, replace=False) for _ in range(L)]
                 for _ in range(n_modes)]
    hot = [RNG.choice(E, K + 2, replace=False) for _ in range(L)]
    t = 0
    for r in range(N_REQ):
        mode = RNG.integers(n_modes)
        for i in range(TOK):
            if regime == "modal" and i % 40 == 39:
                mode = RNG.integers(n_modes)
            for l in range(L):
                if regime == "uniform":
                    ids[t, l] = RNG.choice(E, K, replace=False)
                elif regime == "static":
                    ids[t, l] = RNG.choice(hot[l], K, replace=False)
                else:
                    ids[t, l] = RNG.choice(mode_sets[mode][l], K, replace=False)
            t += 1
    return dict(
        model_tag=np.array([f"SYN-{regime}"]), repo=np.array(["synthetic"]),
        quant=np.array(["none"]), kind=np.array(["moe"]),
        n_layers=np.array([L]), n_experts=np.array([E]), top_k=np.array([K]),
        modules=np.array(["synthetic"]),
        request_id=rid, turn=np.zeros(T, np.int16),
        is_gen=np.ones(T, bool), token_pos=np.arange(T, dtype=np.int32),
        token_id=np.zeros(T, np.int32),
        topk_ids=ids,
        topk_prob=np.full((T, L, K), 1 / K, np.float16),
        gap=RNG.random((T, L)).astype(np.float16),
    )


def make_dense(regime="modal", n_bands=128, d_ffn=6144, n_layers=28):
    T = N_REQ * TOK
    rid = np.repeat(np.arange(N_REQ), TOK).astype(np.int32)
    n_modes = 6
    mode_sets = [[RNG.choice(n_bands, 40, replace=False) for _ in range(n_layers)]
                 for _ in range(n_modes)]
    cnt = np.zeros((T, n_layers), np.int64)
    per_layer = [[] for _ in range(n_layers)]
    t = 0
    for r in range(N_REQ):
        mode = RNG.integers(n_modes)
        for i in range(TOK):
            if i % 40 == 39:
                mode = RNG.integers(n_modes)
            for l in range(n_layers):
                k = int(RNG.integers(25, 35))
                sel = RNG.choice(mode_sets[mode][l], k, replace=False)
                per_layer[l].append(sel.astype(np.int32))
                cnt[t, l] = k
            t += 1
    blob = dict(
        model_tag=np.array([f"SYNDENSE-{regime}"]), repo=np.array(["synthetic"]),
        quant=np.array(["none"]), kind=np.array(["dense"]),
        n_layers=np.array([n_layers]), n_bands=np.array([n_bands]),
        band_s=np.array([d_ffn // n_bands]), d_ffn=np.array([d_ffn]),
        modules=np.array(["synthetic"]),
        request_id=rid, turn=np.zeros(T, np.int16),
        is_gen=np.ones(T, bool), token_pos=np.arange(T, dtype=np.int32),
        token_id=np.zeros(T, np.int32),
        band_cnt=cnt.astype(np.int16),
        chan_cnt=np.full((T, n_layers), 900, np.int32),
    )
    for l in range(n_layers):
        blob[f"band_ids_{l}"] = np.concatenate(per_layer[l])
    return blob


if __name__ == "__main__":
    import sim

    print("### interleave round-trip check (dense unit ordering) ###")
    b = make_dense()
    p = TMP / "dense.npz"
    np.savez_compressed(p, **b)
    tr = sim.Trace(p)
    ok = True
    for t in (0, 1, 17, 500):
        want = []
        for l in range(tr.L):
            o = tr._layer_offs[l]
            want.append(np.sort(tr._layer_ids[l][o[t]:o[t + 1]] + l * tr.n_bands))
        want = np.concatenate(want)
        got = np.sort(tr.token_units(t))
        ok &= np.array_equal(want, got)
    print(f"  per-token unit list matches per-layer sources: {ok}")
    assert ok

    print("  bitset popcount matches set size:",
          bool((sim.POPCNT[tr.bitsets()].sum(1) == tr.counts).all()))
    # the cache policies rely on this: no unit is requested twice in one token
    uniq = all(len(np.unique(tr.token_units(t))) == tr.counts[t]
               for t in range(0, tr.n_tok, 37))
    print(f"  units within a token are unique (policies assume it): {uniq}")
    assert uniq

    print("\n### miss-run helper ###")
    seq = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0], bool)
    print("  runs of [T F F T F T T F F F] ->", sim.miss_runs(seq).tolist(),
          "(expected [2, 1, 3])")
    assert sim.miss_runs(seq).tolist() == [2, 1, 3]

    # the streaming tracker must agree with the batch version, including runs
    # that straddle token boundaries
    rng = np.random.default_rng(3)
    for trial in range(200):
        chunks = [rng.random(rng.integers(1, 12)) > rng.random()
                  for _ in range(rng.integers(1, 8))]
        rt = sim.RunTracker()
        for c in chunks:
            rt.add(c)
        rt.finish()
        want = sim.miss_runs(np.concatenate(chunks))
        got = np.repeat(np.arange(len(rt.hist)), rt.hist)
        assert sorted(want.tolist()) == sorted(got.tolist()), (want, got)
    print("  streaming RunTracker matches the batch version on 200 random "
          "chunkings: True")

    print("\n### MoE regimes: does the simulator separate structure from noise? ###")
    for regime in ("uniform", "static", "modal"):
        p = TMP / f"moe_{regime}.npz"
        np.savez_compressed(p, **make_moe(regime))
        tr = sim.Trace(p)
        cap = round(tr.U * 0.05)
        row = {pol: sim.simulate(tr, cap, pol)["q_steady"]
               for pol in sim.POLICIES}
        ok, lp = sim.persistence(tr, kmax=64)
        sig, _ = sim.signatures(tr)
        Hp = sum(sim.conditional_entropy(sig))
        cv = sim.conditional_entropy_cv(sig)
        ceil = min(1.0, cap / tr.mean_active)
        print(f"  {regime:8s} M=5% ceiling={ceil:.3f}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in row.items()) +
              f"   L_persist={lp}")
        print(f"           H plug-in={Hp:6.2f}   H held-out={sum(cv['H_cond']):6.2f}"
              f"   info gain={sum(cv['info_gain']):6.2f}")
    print("\nexpected: uniform -> no policy beats STATIC; plug-in H collapses to 0")
    print("                     while held-out H stays high and info gain ~ 0;")
    print("          static  -> high q for all; info gain ~ 0 (no layer coupling);")
    print("          modal   -> adaptive policies above STATIC; info gain > 0.")

    print("\n### full analyse() path, MoE ###")
    sim.analyse(TMP / "moe_modal.npz", TMP / "moe_modal.analysis.json")
    print("\n### full analyse() path, dense ###")
    sim.analyse(TMP / "dense.npz", TMP / "dense.analysis.json")
