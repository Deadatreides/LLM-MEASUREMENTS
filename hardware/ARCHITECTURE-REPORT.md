# Mycelium Architecture Report — current state

Prepared instead of a further blind rewrite of scheduler/buffer-ownership/decode-pipeline internals (per your choice: report current state). Everything below is verified against the actual files, not recalled from memory.

## 1. Files rewritten/modified

| File | Nature of change |
|---|---|
| `substrate/include/llama-mycelium-runtime.h` | Extended: interaction-graph types/constants, `expert_energy`, `expert_selected_count`, `graph_node_activity` fields on `llama_myc_runtime_ext`; 3 new function declarations |
| `substrate/src/llama-mycelium-runtime.cpp` | Extended: 5 new functions; `llama_mycelium_get_routing_bias()` body fully rewritten (was corridor `j_star` passthrough, now interaction-graph energy); `full_step_hook()` internals changed (dead uniform-transport block replaced with real propagate+feedback calls) |
| `llama.cpp/src/llama-graph.h` | Extended: `lctx`/`mycelium_veto` fields on `llm_graph_context`, `lctx` field on `llm_graph_params`, `t_mycelium_selected` output-tracking vector on `llm_graph_result`, new `llm_graph_input_mycelium_expert_bias` class |
| `llama.cpp/src/llama-graph.cpp` | Extended: constructor init list, `reset()`, `build_moe_ffn()` (bias injection + topology-ensure call + output tracking), new `set_input()` implementation |
| `llama.cpp/src/llama-context.cpp` | Extended: `llama_mycelium_init`/`runtime_ext_init` calls in `llama_init_from_model()`; `graph_params()` populates `lctx`; `process_ubatch()` reads back selected-expert tensors and reports them to Runtime |
| `llama.cpp/src/llama-kv-cache.h` | Extended: `lctx_cached` field |
| `llama.cpp/src/llama-kv-cache.cpp` | Extended: `init_update()` caches `lctx`; `find_slot()`'s `heat_strict` gate extended with a guardrail-pressure condition |

Not touched: `llama-context.h`, `llama-model.cpp`, `llama-model.h`, `llama-batch.cpp`, `llama-memory.cpp`, `llama-sampler.cpp`, `common/speculative.cpp`, the ggml backend/scheduler files, and (per your explicit exclusion) all Python files and the root-level `mycelium-*.cpp` files.

Every change is additive or in-place-rewritten within an existing function; no file was deleted or replaced wholesale.

## 2. Deleted code

One block: the original "7b" transport in `full_step_hook()` (`corridors[i].j_star += n_ema` for all `i`, which Iteration 4's own report already noted had zero effect on ranking) was removed and replaced by the propagate/feedback calls. Nothing else was deleted — no functions removed, no upstream logic stripped.

## 3. New functions

Substrate (`llama-mycelium-runtime.cpp`):
- `llama_mycelium_graph_ensure_topology(ctx, n_expert)` — builds the corridor→expert edge set (idempotent per `n_expert`)
- `llama_mycelium_graph_propagate(ext)` [internal] — `expert_energy[e] = Σ activity[from]·weight` over edges into `e`
- `llama_mycelium_graph_feedback(ext)` [internal] — decay-toward-target Hebbian edge-weight update using real selection outcomes
- `llama_mycelium_graph_report_selection(ctx, counts, n)` — accumulates real per-expert selection counts from the graph
- `llama_mycelium_graph_edge_target(corridor_id, k, n_expert)` [internal] — deterministic topology hash

`llama.cpp` side: no new named functions — `set_input()` is a new override on a new input class, and `process_ubatch()` gained an inline block; no new top-level API.

## 4. Changed algorithms

- **`llama_mycelium_get_routing_bias()`**: was direct positional passthrough of `corridors[i].j_star`; now returns `expert_energy`, computed by graph propagation.
- **`find_slot()`'s `heat_strict` gate** (`llama-kv-cache.cpp`): was `!cont && n_swa > 0` only; now also true when `lctx_cached` reports a guardrail veto — unchanged for SWA models, newly active for non-SWA models under Runtime-reported pressure.
- **MoE expert ranking** (`build_moe_ffn`): `selection_probs` (the tensor fed to `ggml_argsort_top_k`) now includes an additive Runtime-supplied bias term before ranking. Aggregation, weighting, `n_expert`, `n_expert_used` — untouched.

## 5. `decode()` call chain (current, real)

```
llama_decode(ctx, batch)
  → ctx->decode(batch)
      → memory_update(false) → memory->init_update(this, ...)      [lctx_cached set on KV cache]
      → memory->init_batch(...) → prepare() → find_slot()          [heat_strict may trigger on guardrail veto]
      → for each ubatch:
          process_ubatch(ubatch, ...)
            → graph_params(...)                                     [populates lctx on llm_graph_params]
            → model.build_graph(gparams) → llm_graph_context(params)  [lctx cached on the context]
                → build_moe_ffn() [per MoE layer]
                    llama_mycelium_graph_ensure_topology(lctx, n_expert)
                    selection_probs += bias tensor (input, filled post-alloc)
                    selected_experts = ggml_argsort_top_k(selection_probs, n_expert_used)
                    ggml_set_output(selected_experts); res->t_mycelium_selected.push_back(...)
            → ggml_backend_sched_alloc_graph(...)
            → res->set_inputs(&ubatch)
                → llm_graph_input_mycelium_expert_bias::set_input()
                    llama_mycelium_get_routing_bias(lctx, il, n_expert, values) → ggml_backend_tensor_set(bias, values)
            → graph_compute(...)
            → [NEW] for t in res->t_mycelium_selected: ggml_backend_tensor_get(t, ...) → aggregate counts
                     → llama_mycelium_graph_report_selection(this, counts, 256)
  → llama_mycelium_full_step_hook(ctx, logits, ...)                 [unchanged call site, Iteration 1]
      → graph_node_activity[c] = corridors[c].usage_ema
      → llama_mycelium_graph_propagate(ext)   → expert_energy refreshed
      → llama_mycelium_graph_feedback(ext)    → edge weights updated, expert_selected_count reset
  → return
```

## 6. Runtime lifecycle (current)

```
llama_init_from_model()
  new llama_context(...)
  llama_mycelium_init(ctx)              [P1 MCB, global map keyed by ctx*]
  llama_mycelium_runtime_ext_init(ctx)  [P2 ext, global map keyed by ctx*]

per decode() call:
  full_step_hook: entropy/curvature/drift ← logits (real, native, every step)
                  kv hit/miss, kv thermo
                  layer hotness decay
                  corridor usage EMA ← corridor state
                  graph_node_activity ← corridor usage EMA
                  graph_propagate: expert_energy ← activity × edge weight
                  graph_feedback: edge weight ← decay toward (selection ? activity : 0), scaled by negentropy
                  telemetry ring write

llama_free(ctx)
  llama_mycelium_teardown(ctx)          [frees both P1 and P2 map entries]
```

Ownership: still global maps keyed by `llama_context*` (substrate's own documented stopgap, unchanged — this was explicitly not touched, see `llama-mycelium.h:107-108`'s own comment acknowledging it).

## 7. MoE routing (current)

```
routing scores (logits → gating function → optional model bias → group masking)  [upstream, unchanged]
  ↓
+ Mycelium bias tensor (expert_energy, filled at set_input time)                 [new]
  ↓
selection_probs                                                                   [same tensor, now biased]
  ↓
ggml_argsort_top_k(selection_probs, n_expert_used)                               [upstream call, unchanged]
  ↓
selected_experts                                                                  [now also tracked as output]
  ↓
aggregation / weighting (uses UNBIASED probs for weight values, selected_experts for which ones)  [upstream, unchanged]
```

Runtime influences *which* experts get selected; it does not touch aggregation, weighting, or count.

## 8. KV Cache (current)

```
find_slot(ubatch, cont)
  heat_strict = (!cont && n_swa > 0)                                    [upstream-derived, unchanged for SWA]
             || (!cont && lctx_cached && guardrail_check(lctx_cached) != NONE)  [new, non-SWA models]
  if heat_strict: heat_score(...) > 0.45 rejects a candidate cell        [upstream mechanism, unchanged]
  (existing retry-on-exhaustion fallback, unchanged, consumes new heat_strict identically)
```

KV Cache does not read `expert_energy`, the interaction graph, or corridor state directly — only the pre-existing `guardrail_check()` boolean.

## 9. Interaction Graph (current)

```
Nodes:  graph_node_activity[32]      = corridors[i].usage_ema, refreshed every decode step
Edges:  graph_edges[≤512]            = deterministic (corridor*7 + k*13 + 1) mod n_expert, degree 8/corridor
                                        built once per distinct n_expert (llama_mycelium_graph_ensure_topology)
Weights: edge.weight ∈ [0,1], start uniform (1/degree), updated every step by graph_feedback()

Forward:  expert_energy[e] = Σ (activity[from] · weight) over edges into e         [graph_propagate]
Backward: weight += 0.05 · fb · (target − weight), target = selected(e) ? activity[from] : 0,
          fb = clamp(1 + n_ema, 0.1, 1.5)                                          [graph_feedback]
```

Known simplifications, stated plainly: one graph shared across all MoE layers (no per-layer differentiation, `il` accepted but unused); topology is hand-derived, not learned; `LLAMA_MYC_MAX_EXPERTS_RUNTIME` caps at 256.

## 10. Diff vs. upstream llama.cpp

- **Additive in all 7 touched files** — no upstream function signature changed in a way that breaks non-Mycelium builds; every change is behind `#ifdef LLAMA_MYCELIUM_ENABLE` except the always-present but harmless plumbing fields (`lctx` on `llm_graph_params`/`llm_graph_context`, `lctx_cached` on `llama_kv_cache`, `t_mycelium_selected` on `llm_graph_result`) which default to null/empty and cost nothing when unused.
- **Public API**: zero changes to `include/llama.h`. Three new substrate-internal C functions (not part of llama.cpp's own API surface).
- **Build**: still blocked — no `llama.cpp/CMakeLists.txt` exists in this checkout to ever define `LLAMA_MYCELIUM_ENABLE`, so none of this compiles into any binary this checkout can currently produce (unchanged fact since Iteration 1; not addressed by this milestone, since you scoped this round to substrate + graph + context + KV cache).
- **Known loose end**: `mycelium_veto` (`llm_graph_context` member, Iteration 7) is read from `guardrail_check()` at construction time but not consumed anywhere — dead value, not yet wired to anything. Flagging it rather than hiding it.
