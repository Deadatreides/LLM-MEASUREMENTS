"""How concentrated is dense FFN activation, really?

Step 1 on real weights showed the 90 %-mass criterion of §3.2 selecting ~86 %
of the bands, which would make the dense q(M) curve trivially linear.  Before
spending an hour on the full trace, this measures the concentration directly:
what fraction of channels / bands carries a given fraction of the mass, per
layer, over real corpus text.

If the answer is "most of them", that IS the result for the dense path -- it
says the dense FFN has no compact working set at this granularity, and no
choice of band size fixes it (bands can only be less concentrated than the
channels they aggregate).
"""

import json
from pathlib import Path

import numpy as np
import torch

import probe

ROOT = Path(__file__).parent
FRACS = [0.50, 0.80, 0.90, 0.95, 0.99]


@torch.no_grad()
def main(tag="B", n_prompts=12):
    tok, model, cfg = probe.load(tag)
    dev = "cuda" if cfg["device"] == "cuda" else "cpu"
    hf = model.config
    cap = probe.DenseCapture(model, hf)
    d_ffn, n_bands, s = cap.d_ffn, cap.n_bands, cap.s
    L = cap.n_layers

    corpus = json.loads((ROOT / "corpus.json").read_text(encoding="utf-8"))
    picks = corpus[::len(corpus) // n_prompts][:n_prompts]
    text = "\n\n".join(probe.turn_prompt(c["turns"][0], i == 0)
                       for i, c in enumerate(picks))
    ids = tok(text, return_tensors="pt")["input_ids"][:, :512].to(dev)
    T = ids.shape[1]
    print(f"{cfg['repo']}  d_ffn={d_ffn} s={s} n_bands={n_bands} layers={L} "
          f"tokens={T}")

    # capture the raw activations again, this time keeping the full curve
    store = {}
    hs = []
    for li, layer in enumerate(probe.find_layers(model)[1]):
        _, dp = probe.find_down_proj(layer)
        hs.append(dp.register_forward_pre_hook(
            lambda m, a, li=li: store.__setitem__(
                li, a[0].detach().float().reshape(-1, d_ffn).abs().cpu())))
    model(input_ids=ids)
    for h in hs:
        h.remove()

    def frac_needed(mass, fr):
        srt = np.sort(mass, axis=-1)[:, ::-1]
        cum = np.cumsum(srt, axis=-1)
        thr = fr * cum[:, -1:]
        return ((cum < thr).sum(-1) + 1) / mass.shape[-1]

    print("\nfraction of CHANNELS carrying a given fraction of the mass")
    print("layer | " + " | ".join(f"{int(f*100):>3d} %" for f in FRACS))
    chan_rows, band_rows = [], []
    for li in range(L):
        a = store[li].numpy()
        row = [frac_needed(a, f).mean() for f in FRACS]
        chan_rows.append(row)
        b = a.reshape(a.shape[0], n_bands, s).sum(-1)
        band_rows.append([frac_needed(b, f).mean() for f in FRACS])
        if li % 4 == 0 or li == L - 1:
            print(f"{li:5d} | " + " | ".join(f"{v:.3f}" for v in row))

    c = np.array(chan_rows)
    bd = np.array(band_rows)
    print("\nmean over layers")
    print("            " + " | ".join(f"{int(f*100):>3d} %" for f in FRACS))
    print("channels  : " + " | ".join(f"{v:.3f}" for v in c.mean(0)))
    print("bands     : " + " | ".join(f"{v:.3f}" for v in bd.mean(0)))
    print("\nbest (most concentrated) layer at 90 % mass: "
          f"channels {c[:,2].min():.3f} at layer {int(c[:,2].argmin())}, "
          f"bands {bd[:,2].min():.3f} at layer {int(bd[:,2].argmin())}")
    print(f"worst: channels {c[:,2].max():.3f} at layer {int(c[:,2].argmax())}, "
          f"bands {bd[:,2].max():.3f} at layer {int(bd[:,2].argmax())}")

    # What a cache would have to hold, in units, for the whole token
    print(f"\nAt 90 % mass the whole token needs {bd[:,2].mean()*n_bands*L:.0f} "
          f"of {n_bands*L} band-units ({bd[:,2].mean()*100:.1f} %).")
    print(f"At 50 % mass it would need {bd[:,0].mean()*n_bands*L:.0f} "
          f"({bd[:,0].mean()*100:.1f} %).")
    np.savez(ROOT / "out" / f"{tag}_concentration.npz",
             chan=c, band=bd, fracs=np.array(FRACS))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "B")
