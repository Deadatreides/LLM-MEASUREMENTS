# Milestone: working thermodynamic routing (corridor activity → physical expert ranking)

Scope respected per your instructions: `substrate/` and `llama.cpp/src/llama-graph.{h,cpp}` only. Python (`bridge/`, `mycelium/`) and the root-level `mycelium-{cavity,telemetry,guardian,epoch,residency,hilbert,corridor}.cpp` files were not touched.

## Chain, now complete end-to-end

```
llama_decode()
  → ctx->decode()
      → build_moe_ffn() [llama-graph.cpp]
          llama_mycelium_graph_ensure_topology(lctx, n_expert)   // builds edges once per n_expert
          selection_probs += bias tensor   // filled from expert_energy via set_input()
          selected_experts = argsort_top_k(selection_probs, n_expert_used)   // now really influenced
  → llama_mycelium_full_step_hook() [already wired, Iteration 1]
      graph_node_activity[corridor] = corridors[corridor].usage_ema   // already wired, Iter. 5
      llama_mycelium_graph_propagate(ext)   // NEW: expert_energy[e] = Σ activity[from]*weight over edges into e
      llama_mycelium_graph_feedback(ext)    // NEW: edge.weight += 0.05 * n_ema * activity[from], clamped [0,1]
```

One-step lag is inherent and expected: `expert_energy` used by step *N*'s routing was computed by step *N-1*'s `full_step_hook` (propagate runs after decode, `set_input` reads before it). First decode of a session sees zero energy (neutral bias); it becomes non-uniform from the second decode onward.

## Files changed

- `substrate/include/llama-mycelium-runtime.h`: `LLAMA_MYC_MAX_GRAPH_EDGES` 64→512, new `LLAMA_MYC_MAX_EXPERTS_RUNTIME`/`LLAMA_MYC_GRAPH_DEGREE`/`LLAMA_MYC_GRAPH_WEIGHT_MIN`/`MAX`/`LLAMA_MYC_GRAPH_PLASTICITY` constants, `graph_topology_n_expert` + `expert_energy[256]` fields, new `llama_mycelium_graph_ensure_topology()` declaration, rewritten doc comment on `llama_mycelium_get_routing_bias()`.
- `substrate/src/llama-mycelium-runtime.cpp`: new `llama_mycelium_graph_edge_target()`, `llama_mycelium_graph_ensure_topology()`, `llama_mycelium_graph_propagate()`, `llama_mycelium_graph_feedback()`. `llama_mycelium_get_routing_bias()` rewritten to read `expert_energy` instead of `corridors[].j_star`. `full_step_hook()`: the old dead "7b" (uniform j_star += n_ema, which Iteration 4 already noted had zero ranking effect) replaced with real `propagate()`/`feedback()` calls.
- `llama.cpp/src/llama-graph.h`/`.cpp`: no signature changes to the existing bias input class or `get_routing_bias` call site — only added the `ensure_topology` extern decl and one call to it in `build_moe_ffn()`, right before the existing bias-tensor construction.

## Design choices (invented, as authorized)

- **Topology**: deterministic, not learned — `to = (corridor*7 + k*13 + 1) mod n_expert`, degree 8 per corridor (32 corridors × 8 = up to 256 edges, well under the 512 cap). Many-to-many by construction (each expert typically receives edges from multiple corridors). Chosen over randomization because it's reproducible and auditable, not because it's claimed to be correct.
- **Propagation**: linear sum, `expert_energy[e] = Σ activity[from]·weight`, recomputed fresh each step (no cross-step accumulation, so magnitude stays bounded by construction).
- **Feedback**: bounded Hebbian-style update using existing `n_ema` (negentropy) as the only "quality" signal available without inventing a new one. Explicitly not a claim of correctness — first working version.
- **Scope simplification, stated plainly**: one shared graph across all MoE layers (`il` still unused) — Heat field / spin-glass are not modeled as separate stages; negentropy (`n_ema`, already existed) stands in for both.

## Update: real selection feedback closed (llama-context.cpp now touched too)

The gap noted above is closed. Full loop, now using actual `argsort_top_k` output instead of a negentropy proxy:

```
build_moe_ffn()                                  [llama-graph.cpp]
  selected_experts = argsort_top_k(...)             (unchanged existing call)
  ggml_set_output(selected_experts)                 NEW
  res->t_mycelium_selected.push_back(selected_experts)   NEW

llama_context::process_ubatch()                  [llama-context.cpp]
  graph_compute(...)                                (unchanged)
  for t in res->t_mycelium_selected:                NEW
    ggml_backend_tensor_get(t, ...)                    read back real selected expert ids
    counts[id]++
  llama_mycelium_graph_report_selection(this, counts, 256)   NEW

llama_mycelium_graph_report_selection()          [llama-mycelium-runtime.cpp]
  expert_selected_count[e] += counts[e]             accumulates across all ubatches of one decode()

llama_mycelium_full_step_hook() -> llama_mycelium_graph_feedback()   [llama-mycelium-runtime.cpp, rewritten]
  for each edge (from=corridor, to=expert):
    target = expert_selected_count[to] > 0 ? activity[from] : 0
    weight += plasticity * fb * (target - weight)     decay-toward-target Hebbian update, fb from n_ema
  expert_selected_count[] reset to 0
```

Additional files touched: `llama.cpp/src/llama-context.cpp` (extern decl + ~20 lines in `process_ubatch`, right after `graph_compute()` succeeds), `llama.cpp/src/llama-graph.h` (`t_mycelium_selected` vector on `llm_graph_result`, cleared in `reset()`).

Feedback rule changed from "activity × global negentropy" to "did this edge's target expert actually fire, gated by source-corridor activity, modulated by negentropy" — a real (if simple) credit-assignment rule using ground-truth routing outcomes, not a proxy.

## What is still simplified (stated plainly, not hidden)

- One shared interaction graph across all MoE layers in a model (`il` still unused for differentiation).
- Deterministic topology (hash-based), not learned from data.
- Selection counts are aggregated across all MoE layers within a decode() call, not tracked per-layer.
- `LLAMA_MYCELIUM_MAX_EXPERTS_LOCAL = 256` in `llama-context.cpp` is a hand-mirrored copy of substrate's `LLAMA_MYC_MAX_EXPERTS_RUNTIME` (llama-context.cpp doesn't include substrate headers, per the established minimal-coupling pattern) — if that constant changes in substrate, this copy needs updating too.
