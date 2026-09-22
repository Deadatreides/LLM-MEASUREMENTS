# Iteration 4 — Runtime: negentropy → corridor j_star (native, no Python required)

## 1. What code changed

- `substrate/src/llama-mycelium-runtime.cpp`, inside `llama_mycelium_full_step_hook()`: +9 lines, one new loop, right after the existing `n_current`/`n_ema` (negentropy) computation. No existing line removed or altered.

## 2. Why this implementation

Iteration 2 wired `corridors[i].j_star` into MoE routing bias, but `j_star` is only ever set by `corridor_update_step()` syncing from `mcb->flow.j_star[i]` — which only an external caller (currently Python's `update_flow_state`) ever writes. On a native-only build with nothing external attached, the whole Iteration 2/3 chain was live but permanently produced a zero (neutral) bias. `n_current`/`n_ema` (negentropy), by contrast, are computed every decode step from purely native state (corridor coherence, spec collapse rate, KV fragmentation/pending-queue pressure) — already real, already running, nothing external required. Adding `n_ema` on top of each corridor's `j_star` (after the existing flow-state sync, not replacing it) gives the already-built Graph mechanism real, native, per-step-varying values immediately.

**Correction:** the first version of this change multiplied the transported value by an alternating `+1/-1` sign keyed on corridor index, to make the bias non-uniform enough to actually move `ggml_argsort_top_k`'s output. That was invented differentiation math, not transport — corrected. The value is now added to every corridor unchanged, with no per-index modification. Consequence, stated plainly: this step alone does not yet change MoE ranking, since a uniform addition doesn't affect `argsort_top_k`. It only moves negentropy one step further along the path Iteration 2 built. Per-corridor differentiation is a separate, later transport.

## 3. Temporary assumptions

- Negentropy is added directly, unscaled, uniformly to every corridor's `j_star`. No per-corridor differentiation — none exists yet to transport.
- Added on top of whatever `flow.j_star` already contributed (via `corridor_update_step()`, unchanged), not replacing it.
- Still confined to `LLAMA_MYC_MAX_CORRIDORS` (32) — same existing loop bound already used two lines above for `corridor_coherence`.

## 4. Open questions

None. Graph-side code (`llama-graph.h`/`.cpp`) was not touched this iteration — confirms the Iteration 2 boundary held: Runtime's internal derivation changed, Graph's contract (`llama_mycelium_get_routing_bias`) did not.
