# Iteration 5 — Runtime: Pseudo Expert Activity → Interaction Graph (node energy transport)

## 1. Code changed

- `substrate/include/llama-mycelium-runtime.h`: +3 lines, new field `float graph_node_energy[LLAMA_MYC_MAX_CORRIDORS];` on `llama_myc_runtime_ext`.
- `substrate/src/llama-mycelium-runtime.cpp`, in `llama_mycelium_full_step_hook()`: +2 lines, direct copy `graph_node_energy[i] = corridors[i].usage_ema` inside the existing per-corridor loop.

## 2. Why this implementation

`usage_ema` already exists per corridor (pseudo-expert activity). This transports it unchanged into a new node-energy slot representing the interaction graph's per-node state. No edges, no weights, no expert mapping — those don't exist yet and were not invented.

## 3. Temporary/incomplete

- No edges between graph nodes and physical experts exist yet.
- `graph_node_energy` is zero by default (value-initialized) and equals `usage_ema` exactly — identity transport, nothing computed.
- Nothing reads `graph_node_energy` yet.

## 4. Open

None.
