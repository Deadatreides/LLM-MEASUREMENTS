# Iteration 6 — Interaction Graph: edge storage (empty)

## 1. Code changed

- `substrate/include/llama-mycelium-runtime.h`: renamed `graph_node_energy` → `graph_node_activity` (terminology fix, no semantic change). Added `llama_myc_graph_edge {from, to, weight}`, `LLAMA_MYC_MAX_GRAPH_EDGES` (64), and `graph_edges[]` + `graph_edge_count` on `llama_myc_runtime_ext`.
- `substrate/src/llama-mycelium-runtime.cpp`: updated the one transport line to the new field name.

## 2. Why

Node storage existed (Iteration 5); edge storage did not. Added it as pure storage: `graph_edge_count` stays 0 (value-initialized), no edges are populated, no weights computed.

## 3. Temporary/incomplete

- `graph_edges[]` is unused/empty. Nothing writes an edge yet.
- No connection between `graph_node_activity` and `graph_edges` beyond both existing on the same struct.

## 4. Open

None.
