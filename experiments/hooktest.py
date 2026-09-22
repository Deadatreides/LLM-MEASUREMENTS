"""Step 1 of §10, validated without downloading anything.

Builds tiny OLMoE and Qwen3 models from config with random weights, so the
hook machinery -- module discovery, full router logits, the band-mass axis --
is checked against the real transformers implementations before an hour is
spent downloading real weights.  Nothing here touches the network.

Run this first once torch is installed:   python hooktest.py
"""

import numpy as np
import torch

import probe

torch.manual_seed(0)


def tiny_olmoe():
    from transformers import OlmoeConfig, OlmoeForCausalLM
    cfg = OlmoeConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        num_experts=16, num_experts_per_tok=4, max_position_embeddings=128,
    )
    return OlmoeForCausalLM(cfg).eval(), cfg


def tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128,
    )
    return Qwen3ForCausalLM(cfg).eval(), cfg


@torch.no_grad()
def check_moe():
    print("=" * 70)
    print("MoE hook check (tiny OLMoE, random weights)")
    model, cfg = tiny_olmoe()
    name, layers = probe.find_layers(model)
    print(f"  find_layers -> {name}  (n={len(layers)}, expected "
          f"{cfg.num_hidden_layers})")
    assert len(layers) == cfg.num_hidden_layers

    rname, router = probe.find_router(layers[0], cfg.num_experts)
    print(f"  find_router -> {rname}  class={type(router).__name__}  "
          f"weight={tuple(router.weight.shape)}")
    assert router.weight.shape[0] == cfg.num_experts

    cap = probe.MoECapture(model, cfg, verify=True)
    ids = torch.randint(0, 256, (1, 12))
    cap.reset()
    model(input_ids=ids)
    raw = cap.buf[0]
    print(f"  hook capture layer0 shape {tuple(raw.shape)} "
          f"(expected [12, {cfg.num_experts}])")
    assert raw.shape == (12, cfg.num_experts), "hook did not capture full logits"

    # recompute the logits independently from the router's input and weight
    w = cap.routers[0].weight.detach().float()
    ref = torch.nn.functional.linear(cap.inp[0].reshape(-1, w.shape[1]), w)
    d = (ref - raw).abs().max().item()
    print(f"  recompute W_gate @ x: max|diff| = {d:.3e}")
    assert d < 1e-4, "hook is not on the router"

    # the decisive property: the captured vector must NOT be a masked top-k
    finite = torch.isfinite(raw).all().item()
    n_distinct = len(torch.unique(raw[0]))
    print(f"  all entries finite: {finite}; distinct values in row 0: "
          f"{n_distinct}/{cfg.num_experts} -> not a top-k mask: "
          f"{n_distinct == cfg.num_experts}")
    assert finite and n_distinct == cfg.num_experts

    a, b, c = cap.collect(12)
    # decisive: our top-k must equal the experts the model actually ran
    truth = cap.sel[0].reshape(12, -1).numpy()
    same = (np.sort(truth, 1) == np.sort(a[:, 0, :].astype(truth.dtype), 1))
    print(f"  vs router_indices actually used: {100*same.mean():.2f} % identical")
    assert same.all(), "captured top-k differs from what the model ran"
    srt = raw.sort(-1, descending=True).values
    k = cfg.num_experts_per_tok
    want = (srt[:, k - 1] - srt[:, k]).numpy().astype(np.float16)
    print(f"  gap matches logit[k]-logit[k+1]: "
          f"{np.allclose(c[:, 0], want, atol=1e-2)}")
    assert np.allclose(c[:, 0], want, atol=1e-2)
    cap.remove()
    print("  MoE path OK")


@torch.no_grad()
def check_dense():
    print("=" * 70)
    print("Dense hook check (tiny Qwen3, random weights)")
    model, cfg = tiny_qwen3()
    name, layers = probe.find_layers(model)
    print(f"  find_layers -> {name}  (n={len(layers)})")
    assert len(layers) == cfg.num_hidden_layers

    cap = probe.DenseCapture(model, cfg, band_size=cfg.intermediate_size // 32)
    print(f"  d_ffn={cap.d_ffn} s={cap.s} n_bands={cap.n_bands}")
    ids = torch.randint(0, 256, (1, 12))
    cap.reset()
    model(input_ids=ids)
    cnt = np.concatenate(cap.out_cnt[0])
    chan = np.concatenate(cap.out_chan[0])
    print(f"  rows captured: {len(cnt)} (expected 12)")
    assert len(cnt) == 12

    # the input of down_proj must be exactly silu(gate_proj(h)) * up_proj(h)
    lay = layers[0]
    h = {}
    h1 = lay.mlp.down_proj.register_forward_pre_hook(
        lambda m, a: h.__setitem__("x", a[0].detach().float()))
    h2 = lay.mlp.register_forward_pre_hook(
        lambda m, a: h.__setitem__("h", a[0].detach()))
    model(input_ids=ids)
    h1.remove(); h2.remove()
    ref = (torch.nn.functional.silu(lay.mlp.gate_proj(h["h"]))
           * lay.mlp.up_proj(h["h"])).float()
    d = (ref - h["x"]).abs().max().item()
    print(f"  down_proj input == silu(gate)*up : max|diff|={d:.3e}  "
          f"shape {tuple(h['x'].shape)}")
    assert d < 1e-4, "the hooked tensor is not the FFN activation"
    assert h["x"].shape[-1] == cap.d_ffn, "band mass would sum over the wrong axis"

    # min_cover must return the minimal set covering >= 90 % of the row mass
    m = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    idx, k = probe.min_cover(m, 0.90)
    print(f"  min_cover on [10,1,1,1,1,1] @90%: k={k.tolist()} ids={idx.tolist()} "
          f"(10/15=0.667 so 10+1+1+1=0.867 <0.9, need 5 units)")
    assert k[0] == 5
    a = torch.rand(7, 64)
    _, kk = probe.min_cover(a, 0.90)
    srt = a.sort(-1, descending=True).values.cumsum(-1)
    tot = a.sum(-1)
    ok = all(srt[i, kk[i] - 1] >= 0.90 * tot[i] and
             (kk[i] == 1 or srt[i, kk[i] - 2] < 0.90 * tot[i]) for i in range(7))
    print(f"  min_cover minimality on random rows: {ok}")
    assert ok

    print(f"  active bands/token layer0: {cnt.tolist()}  of {cap.n_bands}")
    print(f"  active channels/token layer0: {chan.tolist()}  of {cap.d_ffn}")
    cap.remove()
    print("  dense path OK")


@torch.no_grad()
def check_replay_equivalence():
    """A full-sequence forward must give the same routing as step-by-step decode.

    This is the assumption the whole collection scheme rests on (see probe.py):
    if it were false, the trace would not describe decoding at all.
    """
    print("=" * 70)
    print("Full-sequence forward vs incremental decode")
    model, cfg = tiny_olmoe()
    cap = probe.MoECapture(model, cfg)
    ids = torch.randint(0, 256, (1, 10))

    cap.reset()
    model(input_ids=ids)
    full = cap.collect(10)[0]

    step = np.zeros_like(full)
    past = None
    for t in range(10):
        cap.reset()
        out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        step[t] = cap.collect(1)[0][0]
    same = (full == step).mean()
    print(f"  expert selections identical: {same*100:.2f} % of (token, layer, k)")
    cap.remove()
    assert same > 0.999, "full-sequence capture does not match incremental decode"
    print("  replay equivalence OK")


if __name__ == "__main__":
    check_moe()
    check_dense()
    check_replay_equivalence()
    print("=" * 70)
    print("All step-1 machinery verified on tiny random-weight models.")
    print("Next: python probe.py B --introspect   (real weights)")
