# Iteration 2 — Graph: expert-ranking bias (complete)

Core: Graph. Goal: Runtime influences a real inference decision (which experts get selected), not just telemetry — without touching tensor shapes, expert count, or aggregation logic.

## 1. Files changed

| File | Kind |
|---|---|
| `substrate/include/llama-mycelium-runtime.h` | +1 new accessor declaration (`llama_mycelium_get_routing_bias`) |
| `substrate/src/llama-mycelium-runtime.cpp` | +1 new accessor implementation |
| `llama.cpp/src/llama-graph.h` | +1 forward decl, +1 field on `llm_graph_params`, +1 field on `llm_graph_context`, +1 new input class |
| `llama.cpp/src/llama-graph.cpp` | +1 constructor init entry, +1 `set_input()` implementation, +1 insertion in `build_moe_ffn()` |
| `llama.cpp/src/llama-context.cpp` | +1 line in `graph_params()` |

No file outside these five was touched. No existing line was deleted or altered — every change is a pure addition.

## 2. Functions changed

**Design correction made mid-iteration:** the first pass added a scalar `llama_mycelium_corridor_get_j_star(ctx, corridor_id)` and had Graph loop over it treating `corridor_id` as an expert index. That baked a corridor↔expert mapping into `llama-graph.cpp` that the architecture explicitly does not intend (corridors are Mycelium virtual pseudo-experts, unrelated to a model's physical MoE experts). It was replaced before finishing the iteration with the version described below, which keeps that assumption — such as it is — entirely inside substrate, behind a name and contract that don't mention corridors at all.

- **New:** `llama_mycelium_get_routing_bias(ctx, il, n, out)` (substrate) — fills `out[0..return)` with an explicitly opaque "routing bias signal." The contract (see the header comment) states outright that callers must not assume anything about how the signal is derived or any correspondence between an `out` index and a Runtime-internal identifier. Today's implementation reuses existing state — corridor `j_star` values, taken positionally — purely because it's the only existing per-slot state of roughly the right shape; that is documented as an internal placeholder, not part of the contract. `il` (layer index) is accepted and threaded through end-to-end but unused by today's source — reserved so a future, layer-differentiated Runtime doesn't need a signature change.
- **New:** `llm_graph_input_mycelium_expert_bias::set_input()` (llama-graph.cpp) — one bulk call to that accessor per ubatch (not a per-expert loop), filling a `[n_expert]` F32 tensor after graph allocation. Contains zero references to corridors, `j_star`, or any Runtime-internal concept — confirmed by grep.
- **Modified (additive only):** `llama_context::graph_params()` — now also populates `params.lctx = this`.
- **Modified (additive only):** `llm_graph_context::llm_graph_context(...)` — now also initializes `lctx(params.lctx)`.
- **Modified (additive only):** `llm_graph_context::build_moe_ffn()` — inserts the bias tensor and `ggml_add`s it into `selection_probs` immediately before `ggml_argsort_top_k()`. Everything after that line (aggregation, weighting, `n_expert_used`, `n_expert`) is untouched, byte-for-byte identical to before this iteration.

## 3. Data path through Runtime

```
llama_context::graph_params()            // this iteration adds: params.lctx = this
  → llm_graph_params{ ..., lctx }
    → llm_graph_context(params)          // this iteration adds: lctx(params.lctx)
      → build_moe_ffn(...)               // per MoE layer, per ubatch
          selection_probs = <existing pipeline: logits → gating → optional exp_probs_b →
                              architecture overrides → optional group masking>
          if (lctx):
            inp = llm_graph_input_mycelium_expert_bias(lctx, n_expert, il)
            inp->bias = new [n_expert] F32 graph input tensor
            selection_probs = ggml_add(selection_probs, inp->bias)   // build time: shape only, no values yet
            res->add_input(inp)
          selected_experts = ggml_argsort_top_k(selection_probs, n_expert_used)   // UNCHANGED call
          ... (aggregation, weighting: fully unchanged) ...

  — graph allocated by backend scheduler —

llm_graph_result::set_inputs(ubatch)
  → inp->set_input(ubatch)                              // this iteration's new code
      llama_mycelium_get_routing_bias(lctx, il, n_expert, values)   // ONE opaque substrate call, no per-expert loop
      ggml_backend_tensor_set(bias, values)              // real values now land in the graph
  → graph_compute()                                      // selection_probs already includes the bias by now
```

An abstract Runtime "routing bias" signal — whose current concrete source is corridor `j_star` values (already computed by the substrate's existing `corridor_update_step`/flow-state machinery, set externally via `update_flow_state`, currently by the Python orchestrator) but whose contract does not expose that — now flows into the native `ggml_argsort_top_k` call that decides which experts run, for every MoE layer, every ubatch. This is the first point in the whole codebase where a Runtime-computed value changes what the graph actually computes, rather than being recorded after the fact. Graph's own code has no knowledge of, or dependency on, how the signal is derived — that derivation can change freely as the Runtime evolves without touching `llama-graph.cpp`/`.h` again.

## 4. Stubs / gaps closed

- **The Iteration-0 gap "Corridor masks are stored/updated but do not constrain graph construction"** is closed for the one signal traced through this iteration (an abstract routing-bias signal, currently sourced from corridor `j_star`) — it now does constrain (biases) expert selection.
- Note the scope precisely: this closes the gap for **routing/ranking**, not for "Corridor" as a whole. Corridor `state` (HOT/WARM/COLD), `usage_ema`, residency/offload fields remain unread by Graph — untouched, out of scope for this iteration.

## 5. What remains open

- **Neutral by default.** Today's concrete source (`j_star`) defaults to `0.0f` and is only ever non-zero if something external (currently: Python via `update_flow_state`) sets it. Until that happens, this iteration's code runs (one extra `ggml_add` of an all-zero tensor per MoE layer) but has no observable effect on ranking. This is expected and correct for this iteration — closing the *native* Runtime→Graph wire was the goal; a self-sufficient native producer of routing-bias values (so this works without Python attached) is a Runtime-side gap, not a Graph-side one, and is explicitly out of scope here.
- **Graph reuse is disabled whenever this input is active.** `llm_graph_input_mycelium_expert_bias` doesn't override `can_reuse()`, so it inherits the base class's conservative default (`false`) — same as several other existing inputs (e.g. `llm_graph_input_pos_bucket`). This is a deliberate, safe choice (no risk of a stale bias silently persisting across steps with different Runtime state), traded for losing a reuse optimization on MoE layers specifically. Not attempted here, per "smallest diff, no perfect abstraction."
- **Iteration 1's build-wiring gap is unchanged and still blocks everything in this iteration too** — none of this executes in any build this checkout can currently produce, for the same reasons documented in `ITERATION-1-REPORT.md` §4/§6 (no `llama.cpp/CMakeLists.txt` exists to define `LLAMA_MYCELIUM_ENABLE` or link `llama_mycelium`).
- Activation Bias, KV Cache policy (Heat/Pressure/Residency), Sampler integration, and native speculative-window control remain exactly as documented in Iteration 0 — none touched this iteration.
- Substrate's own current placeholder derivation caps at `LLAMA_MYC_MAX_CORRIDORS` (32) tracked signals; for `n_expert` beyond that, the excess entries come back neutral (`0.0f`), handled safely inside `llama_mycelium_get_routing_bias`'s own bounds check. This is entirely a substrate-internal implementation detail today and not something Graph's code encodes or depends on — a future Runtime source is free to cover any `n` without any change on the Graph side.

## 6. Proposed Iteration 3

**KV Cache**, per the original master-plan order (unchanged, not being proposed as a reordering). It's the next subsystem with a real, existing, safe precedent to build on: `heat_strict`/`heat_score` in `llama-kv-cache.cpp` already proves that a boolean-gated, non-shape-affecting policy decision is achievable there (unlike Graph, where this iteration's first attempt hit a genuine tensor-shape landmine and had to be reworked). The concrete candidate gap, from Iteration 0: `heat_strict = !cont && n_swa > 0` — `cont` is always `false` at both call sites, so this Runtime-adjacent eviction heuristic is silently inert for every non-SWA model, i.e. most real usage. Worth confirming with fresh eyes (per-iteration re-verification) whether Runtime's own KV pressure/thermo state (already computed, already read by `guardrail_check`) can extend or replace that gate for the common case, using the same "value-only, no shape/topology change" discipline this iteration ended up requiring for Graph. Not started; awaiting direction.
