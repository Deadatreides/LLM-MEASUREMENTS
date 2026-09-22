# Iteration 7 — Graph consumes guardrail veto state (transport only)

Runtime value used: `llama_mycelium_guardrail_check()` (already existed, already used by KV cache in Iteration 3, never called from Graph).

## 1. Code changed

- `llama.cpp/src/llama-graph.h`: +5 lines, new member `const int32_t mycelium_veto = 0;` on `llm_graph_context`.
- `llama.cpp/src/llama-graph.cpp`: +15 lines — local enum + extern decl (same pattern as the other three files), one constructor init-list entry: `mycelium_veto(lctx ? llama_mycelium_guardrail_check(lctx) : 0)`.

No substrate file touched.

## 2. Why

Smallest possible consumption: read the existing accessor once at graph-build time, store the result. No tensor, no shape, no topology, nothing applied yet.

## 3. Temporary/incomplete

`mycelium_veto` is read and stored but not used anywhere else in the file yet — consumption without action, same discipline as Iteration 5/6's Runtime-side transports, now on the Graph side.

## 4. Open

None.
