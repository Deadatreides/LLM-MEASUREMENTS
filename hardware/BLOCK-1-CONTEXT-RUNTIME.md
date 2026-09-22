# Block 1 — Context + Runtime (complete)

Full architectural rework of one block, not a patch series. No partially-migrated state left behind.

## 1. What changed

### Ownership inverted: Runtime lives in Context

**Deleted outright** — both global registries and everything that served them:

| Deleted | Was in |
|---|---|
| `static std::mutex g_rt_map_mu` | `llama-mycelium-runtime.cpp` |
| `static std::unordered_map<llama_context*, llama_myc_runtime_impl*> g_rt_map` | `llama-mycelium-runtime.cpp` |
| `static std::mutex g_mcb_map_mu` | `llama-mycelium.cpp` |
| `static std::unordered_map<llama_context*, llama_mycelium_impl*> g_mcb_map` | `llama-mycelium.cpp` |
| map lookup + lock in `get_rt()` / `get_impl()` | both |
| allocation path in `llama_mycelium_init` / `runtime_ext_init` | both |
| map erase path in `llama_mycelium_teardown` / `runtime_teardown` | both |
| `#include <mutex>`, `<unordered_map>` | both |

Every Runtime access was previously a mutex acquisition plus a hash lookup, on the hot path of every decode step. It is now one pointer dereference.

**New ownership:**

```
llama_context::mycelium              // direct member, llama-context.h
  ├─ created  in llama_context::llama_context()   → llama_mycelium_runtime_create(this)
  └─ destroyed in llama_context::~llama_context() → llama_mycelium_runtime_destroy(mycelium)

llama_mycelium_runtime               // the organism, owns both phases
  ├─ p1  → llama_mycelium_impl  (MCB + pending + active)   owned
  └─ ext → llama_myc_runtime_ext (P2 subsystems)           by value
```

`llama_mycelium_runtime_of(ctx)` is implemented in `llama-context.cpp` — the only place that knows the layout — so the substrate resolves through it instead of keeping its own registry.

`llama_free()` no longer calls teardown; the destructor owns it. `llama_init_from_model()` no longer calls init; the constructor owns it. The init/teardown *ceremony* is gone.

### Runtime decomposed into named subsystems

`llama_myc_runtime_ext` was a flat bag of ~20 fields. It is now an organism of subsystems, each with one responsibility:

| Subsystem | Holds |
|---|---|
| `llama_myc_heat` | per-layer hotness EMA, residency class |
| `llama_myc_corridors` | pseudo-expert registry (`meta[]`, `n_active`) |
| **`llama_myc_interaction_graph`** | `J_ij` couplings (`edges[]`), `h_i` (`node_field[]`), `s_i` (`node_spin[]`), `pending_delta[]`, `hamiltonian`, `laplacian_cost`, `connectivity` |
| `llama_myc_expert_energy` | `energy[]`, `selected_count[]` — projection onto physical experts |
| `llama_myc_thermodynamics` | `j_attn/j_kv/j_graph/j_behavior/j_spec`, `j_total`, `phi_in/phi_diss`, `chi`, `ds_ext/ds_int`, `eta`, plus the legacy `n_current/n_ema` |
| `llama_myc_flow_relax` | `j[]`, `tau` for `τ·J̇ = −(J − J*)` |
| `llama_myc_kv` (was `kv_thermo`) | KV thermodynamics |
| `spec`, `pending`, `ring` | unchanged, now siblings rather than peers of loose floats |

The **Interaction Graph is a first-class object**, not a set of fields on the Runtime. Its header comment states what it is in the mathematics — the spin-glass coupling structure over corridors — and that there is deliberately no fixed corridor→expert mapping anywhere in the type.

86 field references migrated; zero stale references remain (verified by grep).

### Epoch consistency enforced

The mathematics requires `L_active(t) = L_snapshot ∀t ∈ epoch`. `graph_feedback()` previously mutated `edge.weight` on **every decode step**, which violated it directly.

Now: per-step deltas accumulate into `graph.pending_delta[]` with a `pending_steps` counter, and are applied only by the new `llama_mycelium_graph_commit_epoch()` — averaged over the steps they were gathered from, so epoch length does not by itself change update magnitude. The coupling structure no longer drifts mid-epoch.

### Fabricated Φ_diss replaced with a real measurement

`llama_decode()` passed `kv_hits = kv_total` — a hardcoded 100% hit rate — into the step hook. Passing zeros instead would not have helped: `kv_report_step()` maps `total == 0` to `hit_frac = 1.0`, the same fiction.

So the measurement now actually exists. `llama_kv_cache::find_slot()` counts, per decode, cells taken from free space (`slot_clean`) versus cells that had to displace an occupied cell (`slot_displaced`) — that ratio *is* the allocation-pressure signal. Exposed via `kv_pressure_total/clean/reset()` and consumed in `llama_decode()`.

Speculative counters stay zero at that call site **by design**, and this is now documented in the code: `common/speculative.cpp` reports its own real accept/draft numbers through `llama_mycelium_spec_report_verification()`, and double-reporting would corrupt the EMA.

### Inference no longer allocates in the eviction path

`std::vector<llama_myc_eviction_entry> eviction_queue` → fixed `llama_myc_eviction_entry[64]` + `eviction_count`, with an overflow guard.

## 2. Files changed

| File | Change |
|---|---|
| `substrate/include/llama-mycelium-runtime.h` | subsystem decomposition; Interaction Graph as first-class type; ownership contract; 4 new declarations |
| `substrate/src/llama-mycelium-runtime.cpp` | Runtime organism struct; registry deleted; create/destroy; P1 ownership; epoch-pending feedback + commit; fixed-capacity eviction queue; 86 field migrations |
| `substrate/src/llama-mycelium.cpp` | registry deleted; `p1_create`/`p1_destroy`; `init`/`teardown` become resolve/disable |
| `llama.cpp/src/llama-context.h` | `mycelium` member on `llama_context` |
| `llama.cpp/src/llama-context.cpp` | ctor creates / dtor destroys Runtime; `llama_mycelium_runtime_of()`; init & teardown call sites removed; real KV pressure into the step hook; kv-cache include |
| `llama.cpp/src/llama-kv-cache.h` | `slot_clean`/`slot_displaced` + `kv_pressure_*()` accessors |
| `llama.cpp/src/llama-kv-cache.cpp` | pressure counted in `find_slot()` |

Python bridge untouched, and unbroken: all 27 bound symbols keep their signatures. `llama_mycelium_init` / `runtime_ext_init` remain idempotent from the caller's view — they resolve instead of allocating. `llama_mycelium_mcb` layout is byte-identical, which the ctypes mirrors depend on.

## 3. Temporary assumptions

- `llama_mycelium_teardown` / `runtime_teardown` are retained as no-op-ish disable calls purely because the Python bridge invokes them on layout mismatch. They no longer free anything — Context does.
- The new `thermodynamics` fields (`j_attn`, `phi_in`, `chi`, `eta`, …) and the graph's `node_field`/`node_spin`/`hamiltonian`/`connectivity` are **declared and zero-initialized but not yet computed**. They are the target shape from the математика; populating them is Block 3 (Thermodynamics), and `J_attn` specifically is blocked on the flash-attention decision in `MYCELIUM-TARGET-ARCHITECTURE.md` §R1.
- `llama_mycelium_graph_commit_epoch()` exists and is correct but has **no caller yet** — the epoch boundary lives in `llama_mycelium_epoch_end()`, which is Block 2 (Epoch/KV). Wiring it there is the first task of that block.

## 4. Open

- `substrate/tests/*.cpp` construct Runtimes via fake `llama_context*` pointers (`0xDEADBEEF`) and call `llama_mycelium_init()` to allocate. Under the new ownership that returns `nullptr`, so those tests no longer exercise anything. They need rewriting against `llama_mycelium_runtime_create()` directly — which is better testing anyway (the organism, not the ctx plumbing). Flagging rather than silently leaving them broken.
- Unchanged and still true: no `CMakeLists.txt` exists for `llama.cpp` in this checkout, so none of this is compile-verified. Everything above was verified by reading and by grep, not by a compiler.
