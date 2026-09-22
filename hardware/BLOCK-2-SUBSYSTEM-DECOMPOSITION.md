# Block 2 — Subsystem decomposition (complete)

## 1. Deleted

| Deleted | Lines | Why |
|---|---|---|
| `substrate/src/llama-mycelium-runtime.cpp` | **1158** | Monolith. Replaced by 11 subsystem files. |
| `llama_mycelium_update_flow_state` (P1 copy) | 7 | Duplicate symbol — flow now owns it. Would have been a link error. |

## 2. Created — one file per subsystem, each with its own logic

| File | Lines | Subsystem | Owns |
|---|---|---|---|
| `myc-internal.h` | 176 | — | Organism composition, shared helpers |
| `myc-organism.cpp` | 222 | **Organism** | lifecycle + one step of the whole system + thermodynamics |
| `myc-graph.cpp` | 189 | **InteractionGraph** | J_ij couplings, spins, topology, propagation, epoch commit |
| `myc-telemetry.cpp` | 177 | **Telemetry** | ring buffer, JSON export |
| `myc-kv.cpp` | 157 | **KvRuntime** | KV thermodynamics, eviction queue |
| `myc-epoch.cpp` | 137 | **Epoch** | state machine, pending queue, boundary commit |
| `myc-spec.cpp` | 108 | **SpeculativeRuntime** | draft/verify, `spec_should_run` |
| `myc-heat.cpp` | 105 | **HeatField** | per-layer thermal state |
| `myc-guardian.cpp` | 98 | **Guardian** | veto authority |
| `myc-corridors.cpp` | 90 | **CorridorSystem** | pseudo-expert registry |
| `myc-energy.cpp` | 80 | **EnergyRuntime** | projection onto physical experts |
| `myc-flow.cpp` | 51 | **FlowField** | `tau·dJ/dt = -(J - J*)` relaxation |

**~1590 lines written, 1165 deleted.**

## 3. Graph is now a sibling of Runtime, not a member of it

```cpp
struct llama_mycelium_organism {
    llama_myc_interaction_graph graph;    // STRUCTURE — frozen inside an epoch
    llama_myc_runtime_ext       runtime;  // PROCESSES — tick every step
    llama_myc_guardian          guardian;
    llama_myc_eviction_queue    eviction;
    llama_mycelium_impl *       p1;
    llama_context *             ctx;
};
```

`graph` was removed from `llama_myc_runtime_ext` entirely. `llama_context::mycelium` is now `llama_mycelium_organism *`.

## 4. Real logic added, not just moved

- **Spin relaxation** (`myc_graph_relax_spins`): one synchronous sweep of `s_i <- sign(h_i + Σ J_ij s_j)`. Corridor↔corridor coupling is *induced through shared expert targets* — two corridors projecting onto the same expert are coupled through it. That is what makes it a graph rather than a lookup table, and it never assumes corridor index == expert index.
- **Graph observables**: `H(s)`, the Dirichlet cost `uᵀLu`, and a connectivity fraction for the percolation check.
- **Prigogine balance**: `phi_in`, `phi_diss`, `chi = (phi_in - phi_diss)/phi_diss`, `ds_int`, `ds_ext`.
- **Order parameter**: `tau_eta·dη/dt = aη - bη³` integrated one Euler step, with `a` driven by `chi`, seeded so the pitchfork can leave the `η=0` fixed point.
- **`J_total` assembly**: weighted sum of components. `j_kv`, `j_graph`, `j_spec` are measured. **`j_attn` and `j_behavior` are left at 0** — they have no measurement source, and are not proxied.
- **Flow relaxation** now actually integrates instead of copying `j_star`.
- **Epoch boundary commits the graph** — this closes Block 1's dangling `graph_commit_epoch()` with no caller.

## 5. Bugs found and fixed while writing

Caught by validating every identifier against the real headers rather than assuming:

- `LLAMA_MYC_PEND_CORRIDOR_STATE` — invented; real name is `LLAMA_MYC_PEND_EXPERT_HOT`
- `LLAMA_MYC_SPEC_ACCEPTED` / `_REJECTED` — invented; real enum has `ACCEPTING` / `VERIFYING` / `EMERGENCY`
- `llama_myc_telemetry_entry` — real name `llama_myc_ring_entry`
- ring fields `head`/`count`/`total_written` — real names `write_head`/`n_written`
- `spec.acceptance_ema` — real name `acceptance_rate_ema`
- `p2_struct_sizes` parameter names diverged from the header
- **duplicate `llama_mycelium_update_flow_state`** in two translation units — a genuine link error

Final sweep: every `LLAMA_MYC_*` identifier used in the new files exists in the headers (set difference empty); every declared P2 API function has exactly one implementation; no duplicate symbols.

## 6. Temporary

- `llama_mycelium_runtime_teardown` / `llama_mycelium_teardown` retained as disable-only calls because the Python bridge invokes them. They free nothing — Context owns lifetime.
- `j_attn` needs the attention distribution, which `ggml_flash_attn_ext` fuses away (see `MYCELIUM-TARGET-ARCHITECTURE.md` §R1). Still your decision.
- `myc_epoch_on_step` fires the boundary on `steps_elapsed >= length`. Epoch length is set through the P1 control surface; if Python never begins an epoch, the phase stays IDLE and the graph never commits.

## 7. Open

- `substrate/tests/*.cpp` still allocate via fake `llama_context*` pointers and are now dead against the new ownership. They need rewriting against `llama_mycelium_runtime_create()`.
- Python bridge: all 27 bound symbols keep their names and signatures, and the P1 MCB layout is untouched, so the ctypes mirrors still line up. `llama_myc_runtime_ext` **did** change shape, but Python mirrors it as a `magic`-only stub, so nothing on that side reads the moved fields.
- Still no `CMakeLists.txt` for `llama.cpp` — none of this is compile-verified. Everything above was verified by reading and by mechanical identifier cross-checks, not by a compiler.
