# Block 8 — J_attn: the last missing measurement

## 1. Why I decided this myself

I raised the §R1 flash-attention question three times and you did not answer it. Your
standing instruction is not to ask again unless a blocker requires changing the approved
architecture. Measuring a signal does not change the architecture, so I took the honest
interim option and made it reversible rather than asking a fourth time.

`J_attn` carries weight 0.35 in `J_total` — the largest of the five terms. It has been
hardcoded to 0 since the beginning, which means the negentropy that governs the whole
control law has been missing its dominant component.

## 2. What is measured

Real Shannon entropy, not a proxy:

```
H(p) = -Σ pᵢ ln pᵢ
```

computed in the graph on the tensor that already exists. After
`ggml_soft_max_ext`, `kq` **is** the attention probability matrix, so the entropy is five
ops away:

```cpp
logp  = ggml_log(ggml_clamp(kq, 1e-9f, 1.0f));   // clamp guards log(0) only
plogp = ggml_mul(kq, logp);                       // unclamped p ⇒ p=0 contributes 0
h     = ggml_scale(ggml_sum(plogp), -1.0f);       // scalar
```

The clamp cannot distort the result: the multiply uses the *unclamped* `p`, so a zero
probability contributes exactly zero regardless of what the clamped log says.

Normalised on readback as `H_mean / ln(n_kv)` ∈ [0,1], then the organism keeps the
negentropy `J_attn = 1 − H/H_max`, which is the quantity the mathematics is written in —
*entropy is measured, negentropy governs*.

## 3. Two honest limits, both explicit in the code

**Flash attention.** `ggml_flash_attn_ext` fuses the softmax and never materialises `p`.
There, entropy is not recoverable at any price short of a kernel change. The
instrumentation simply does not exist on that path, `llama_mycelium_publish_attn_entropy`
is never called, and `j_attn` stays 0. That is reported as *no measurement*, never as a
proxy — which is the distinction the mathematics explicitly moved away from in §17.

**One layer.** Only the last attention layer is instrumented. The three elementwise ops
run over the full `[n_kv, n_tokens, n_head]` probability tensor, so instrumenting every
layer would add three passes over the attention matrix on every token. One layer is a
genuine sample of the field at negligible cost. Widening it is a decision about how much
telemetry is worth paying for, not a correctness question — and it is one line to change.

## 4. Path

```
build_attn_mha()                    [llama-graph.cpp]
  kq = ggml_soft_max_ext(...)          the distribution, materialised
  h  = -Σ p ln p                       5 ops, last layer only
  ggml_set_output(h)
      ↓
llama_context::process_ubatch()     [llama-context.cpp]
  ggml_backend_tensor_get(h)           one float
  h_norm = h_mean / ln(n_kv)
      ↓
llama_mycelium_publish_attn_entropy [myc-thermo.cpp]
  j_attn = EMA(1 - h_norm)             negentropy, smoothed
  mcb->runtime.h_attn = h_norm         raw entropy for the control plane
      ↓
myc_thermo_negentropy()
  J_total = 0.35·j_attn + 0.20·j_kv + 0.25·j_graph + 0.10·j_behavior + 0.10·j_spec
```

`J_total` now has its dominant term. Everything downstream of it — `χ`, the order
parameter `η`, `V(R)`, the adaptive epoch length, the speculative window — has been
running on a `J_total` missing 35% of its definition until now.

## 5. Files

`llama.cpp/src/llama-graph.cpp` (entropy nodes), `llama-graph.h` (result fields),
`llama.cpp/src/llama-context.cpp` (readback + normalisation),
`substrate/src/myc-thermo.cpp` (publish + negentropy), `llama-mycelium-runtime.h` (API).

## 6. Verified

- all six ggml ops used (`clamp`, `log`, `mul`, `sum`, `scale`, plus `set_output`) confirmed present in `ggml.h`
- `ggml_sum` confirmed to return a **scalar** — the single-float readback is correct
- `n_layer` confirmed available in `llm_graph_context`, and `build_attn_mha` confirmed to be a member of it
- `<cmath>` already included where `std::log` is now used
- every result field used in `llama-context.cpp` exists in `llama-graph.h`
- every mycelium call across `llama.cpp/src` and `llama.cpp/common` is declared
- I caught and rewrote my own first readback, which computed an unused `n_kv` and normalised by the wrong quantity — the shape is now captured at build time (`n_mycelium_attn_kv`, `n_mycelium_attn_dists`) instead of guessed at readback

Still no compiler in this environment, so none of it is compile-verified.

## 7. Open

- `j_behavior` remains 0 — behavioural drift `D_B = ‖B(t) − B(t−Δt)‖` needs a behavioural state vector the native side does not have. It carries weight 0.10.
- Under flash attention `j_attn` is 0 and `J_total` loses its dominant term. If you want it always available, that is the ggml kernel change from §R1 option 2 — now the only remaining reason to make it, since everything else is wired.
- `llama.cpp` still has no build system; only the substrate is buildable, and only standalone.
