"""Which MoE architectures this probe can actually trace, checked offline.

Builds each candidate from config with random weights -- no download -- and
verifies that the decoder stack is found, the router is found by shape, the
captured logit vector is the FULL pre-top-k one, and our top-k equals the
experts the model really ran.
"""

import numpy as np
import torch

import probe


def check(tag, model, n_experts):
    try:
        name, layers = probe.find_layers(model)
    except Exception as e:
        print(f"{tag:12s} LAYERS FAIL: {e}")
        return
    line = f"{tag:12s} layers={name}(n={len(layers)})"
    try:
        rn, r = probe.find_router(layers[0], n_experts)
    except Exception as e:
        print(line + f"  ROUTER FAIL: {e}")
        return
    cap = probe.MoECapture(model, model.config, verify=True)
    ids = torch.randint(0, 200, (1, 8))
    cap.reset()
    model(input_ids=ids)
    raw = cap.buf[0]
    a, _, _ = cap.collect(8)
    full = raw.shape[-1] == n_experts and len(torch.unique(raw[0])) == n_experts
    ours = np.zeros((8, n_experts), dtype=bool)
    np.put_along_axis(ours, a[:, 0, :].astype(np.int64), True, axis=1)
    if 0 in cap.truth:
        match = f"{100*(ours == cap.truth[0]).all(1).mean():.0f}%"
    elif 0 in cap.sel:
        truth = cap.sel[0].reshape(8, -1).numpy()
        m = (np.sort(truth, 1) == np.sort(a[:, 0, :].astype(truth.dtype), 1))
        match = f"{100*m.mean():.0f}%"
    else:
        match = "n/a"
    cap.remove()
    print(line + f"  router={rn}  logits{tuple(raw.shape)} "
                 f"full={full} topk_match={match}")


if __name__ == "__main__":
    from transformers import (GraniteMoeConfig, GraniteMoeForCausalLM,
                              OlmoeConfig, OlmoeForCausalLM,
                              Qwen3MoeConfig, Qwen3MoeForCausalLM,
                              Qwen3Config, Qwen3ForCausalLM)
    common = dict(vocab_size=256, hidden_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=128)

    check("granitemoe", GraniteMoeForCausalLM(GraniteMoeConfig(
        intermediate_size=128, num_local_experts=16, num_experts_per_tok=4,
        **common)).eval(), 16)

    check("olmoe", OlmoeForCausalLM(OlmoeConfig(
        intermediate_size=128, num_experts=16, num_experts_per_tok=4,
        **common)).eval(), 16)

    check("qwen3_moe", Qwen3MoeForCausalLM(Qwen3MoeConfig(
        intermediate_size=128, moe_intermediate_size=64, num_experts=16,
        num_experts_per_tok=4, head_dim=16, **common)).eval(), 16)

    m = Qwen3ForCausalLM(Qwen3Config(intermediate_size=256, head_dim=16,
                                     **common)).eval()
    n, ls = probe.find_layers(m)
    cap = probe.DenseCapture(m, m.config, band_size=8)
    cap.reset()
    m(input_ids=torch.randint(0, 200, (1, 8)))
    cnt = np.concatenate(cap.out_cnt[0])
    cap.remove()
    print(f"{'qwen3 dense':12s} layers={n}(n={len(ls)})  ffn={cap.module_names[0]}"
          f"  bands={cap.n_bands} active/token={cnt.tolist()}")
