# Architectural decomposition: llama.cpp under full Mycelium integration

Pure analysis. No code, no patches, no diffs. Grounded in the actual codebase (not the abstract pipeline diagram) and in what Mycelium's real, existing mathematics (corridor activity, interaction graph, expert energy, guardrail/pressure, negentropy) actually need — not in a generic maximalist assumption that everything must move.

## Verdict table

| Subsystem | Verdict |
|---|---|
| `llama-context.cpp/.h` | Rewrite (ownership only) — becomes the direct owner of Runtime |
| `llama-model.cpp/.h` | Leave unchanged |
| `llama-memory.cpp` (interface) | Leave unchanged |
| `llama-batch.cpp` | Leave unchanged |
| `llama-graph.cpp/.h` | Rewrite (partial) — primary integration surface stays here, dead code removed |
| `llama-kv-cache.cpp/.h` | Rewrite (partial) — eviction *policy* merges into Runtime; cell bookkeeping stays local |
| Scheduler (`ggml_backend_sched_*`) | Leave unchanged |
| Graph builder (per-architecture `build_*` in `llama-model.cpp`) | Leave unchanged |
| Decode pipeline (`decode()`/`process_ubatch()`) | Rewrite (partial) — already the integration backbone, tighten it |
| ggml graph construction (op primitives) | Leave unchanged |
| MoE router (selection mechanism itself) | Leave unchanged — only its *input* is Mycelium's |
| Speculative decoding (`common/speculative.cpp`) | Rewrite (partial) — merge shadow state into Runtime, wire `spec_should_run` |
| Buffer ownership | Leave unchanged (one narrow future exception: layer offload) |
| Tensor lifetime pattern (no_alloc + set_input/set_output) | Leave unchanged — Mycelium already rides on this correctly |
| Backend scheduling | Leave unchanged |

No subsystem gets **deleted** wholesale. No subsystem gets **fully replaced by a Mycelium algorithm** — the one candidate for that (MoE selection mechanism) is explicitly rejected below, with reasons. Three subsystems get **merged into Runtime** for one specific sub-responsibility each (KV heat scoring, speculative branch state) — never wholesale.

---

## `llama-context.cpp` / `llama-context.h`

1. **Exists now**: owns the per-session inference state — KV memory handle, scheduler, compute buffers, adapters, sampling state, output buffers, `encode()`/`decode()` entry points, state save/load.
2. **Role after Mycelium**: the correct, permanent owner of Runtime state. A session's Runtime (corridor activity, interaction graph, expert energy) is exactly session-scoped data, same lifetime as everything else Context already owns.
3. **Should it exist**: yes, unambiguously.
4. **Functions becoming redundant**: `llama_mycelium_init`/`llama_mycelium_runtime_ext_init`/`llama_mycelium_teardown` as *separate callable entry points* — once Runtime is a real member, these collapse into the constructor/destructor body directly; no separate init call, no global-map lookup.
5. **Functions to rewrite**: the constructor/destructor (own Runtime as a member); `graph_params()` (thread a direct Runtime reference instead of a raw `ctx*` that downstream code re-resolves through a global map); the `process_ubatch()` selection-readback block (call the owned Runtime object directly).
6. **Structures to merge**: `llama_context` + substrate's `llama_mycelium_impl`/`llama_myc_runtime_impl` (today two separate heap objects behind two separate global maps) → one Runtime state member owned directly by `llama_context`.
7. **Compute loops disappearing**: the `get_rt(ctx)`/`get_mcb(ctx)` global-map lookup + mutex lock, paid on *every* Runtime access today — replaced by direct member access. This is a real correctness/contention win, not cosmetic: two `std::mutex`-guarded `unordered_map`s are on the hot path of every decode step right now.
8. **Compute loops appearing**: none — pure ownership restructuring.
9. **Dependencies removed**: `<mutex>` / `<unordered_map>` in the substrate files, used only for the ctx*-keyed maps.
10. **New dependencies**: `llama-context.h` needs a forward declaration of the Runtime struct (pointer member is enough — no full header pull-in), same minimal-coupling discipline already used throughout this integration.

## `llama-model.cpp` / `llama-model.h`

1. **Exists now**: owns model weights, hyperparameters, per-architecture graph-building callbacks, GGUF loading, device placement.
2. **Role after Mycelium**: none beyond what it already has (n_expert/n_expert_used, already read by Graph). Model is per-*checkpoint*; Runtime is per-*session*. llama.cpp already supports multiple contexts sharing one loaded model — conflating Model with Runtime would break that real, load-bearing capability.
3. **Should it exist**: yes, unchanged. This is the clearest "leave alone" case in the whole decomposition, for a structural reason, not a diff-minimization habit.
4-10. No redundant functions, no rewrites, no merges, no loop or dependency changes.

## `llama-memory.cpp` (the `llama_memory_i` interface)

1. **Exists now**: abstracts different memory backends (unified KV, recurrent, hybrid, iswa) behind one contract so Context doesn't need to know which concrete type is active.
2. **Role after Mycelium**: this abstraction already suits Mycelium's own philosophy — Runtime's KV-pressure signal already flows through one boolean (`guardrail_check()`) into a concrete implementation (`llama_kv_cache`), without the interface itself needing to know Mycelium exists.
3. **Should it exist**: yes.
4-10. Unchanged.

## `llama-batch.cpp`

1. **Exists now**: validates/organizes an incoming `llama_batch` into internal `llama_ubatch`s, handles `n_ubatch` splitting.
2. **Role after Mycelium**: batching is about tokens/sequences, orthogonal to routing/thermodynamic concerns.
3. **Should it exist**: yes, unchanged. One speculative future connection worth naming honestly: Runtime's guardrail/emergency state could someday want to shrink ubatch size under pressure — but nothing in Mycelium's current math computes or needs that; noting it as a possible hook, not a requirement.
4-10. Unchanged.

## `llama-graph.cpp` / `llama-graph.h`

1. **Exists now**: the shared graph-construction toolkit (`build_attn`, `build_ffn`, `build_moe_ffn`, input classes) used by every architecture's model-building code; owns `llm_graph_result`/`llm_graph_context`.
2. **Role after Mycelium**: already the primary, proven integration surface (routing bias, topology-ensure, selection-tracking). Should **not** be merged into Runtime or the Interaction Graph — it must keep its distinct responsibility: emitting valid, correctly-shaped tensor graphs for arbitrary architectures. Runtime decides; Graph executes. This separation is exactly what caught the one real bug this integration produced (the `n_expert_used` aggregation mismatch) — collapsing the two would remove the boundary that made that bug visible.
3. **Should it exist**: yes, as a distinct executor layer.
4. **Functions becoming redundant**: `llama_exec_corridors[16]` / `llama_exec_corridor_touch` / `llama_exec_corridor_id` — the original write-only shadow array (dead since before this integration started, flagged in Iteration 0, never connected to anything real). Genuinely redundant now that a real interaction graph exists; delete.
5. **Functions to rewrite**: none of the core builders (`build_attn`, `build_ffn`, `build_norm`, ...) — architecture-agnostic infrastructure with no Mycelium relevance. `build_moe_ffn` keeps its current shape unless per-layer differentiation (the currently-unused `il` parameter) becomes real.
6. **Structures to merge**: none yet. `llm_graph_input_mycelium_expert_bias` could generalize into a shared "Mycelium signal input" class if a *second* concrete signal (e.g. real activation bias) appears — premature until then.
7. **Compute loops disappearing**: the dead per-tensor corridor-touch call inside `cb()`, once the dead array above is deleted.
8. **Compute loops appearing**: none beyond what already exists.
9-10. No dependency changes.

## `llama-kv-cache.cpp` / `llama-kv-cache.h`

1. **Exists now**: concrete `llama_memory_i` implementation — KV ring-buffer cells, slot allocation/eviction, sequence bookkeeping, shift/defrag, state save/load.
2. **Role after Mycelium**: has **two parallel, non-communicating heat models** today — the native `llama_kv_heat` struct (hotness/reuse/semantic/corridor-affinity/migration-cost, per-cell) and substrate's `kv_thermo` (pressure/fragmentation, per-context, coarser). This duplication predates and is unrelated to this integration but is real and should be resolved, not left doubled. Split the subsystem's two sub-responsibilities: **eviction policy** should ultimately be driven entirely by Runtime's pressure/guardrail state (partially done already — the `heat_strict` extension); **cell bookkeeping** (cells, streams, positions, shift/defrag) has no Mycelium relevance and stays local — Runtime has no business owning raw memory-cell state.
3. **Should it exist**: yes, as the concrete slot-management implementation, with eviction-policy responsibility eventually absorbed by Runtime.
4. **Functions becoming redundant** *if* the full merge happens: `llama_kv_heat_value()`, `heat_score()`, `heat_touch()`, `heat_decay()`, and the `llama_kv_heat` struct itself.
5. **Functions to rewrite**: `find_slot()`'s eviction-candidate scoring, to read a substrate accessor directly rather than the local heat struct.
6. **Structures to merge**: `llama_kv_heat` (per-cell, native) into substrate's `llama_myc_kv_thermo` — **but** substrate's model today is per-*context*, not per-*cell*, and would need real (non-trivial) extension to support per-cell granularity before this merge is honestly possible. Naming this gap explicitly rather than waving it away.
7. **Compute loops disappearing**: per-cell `heat_touch`/`heat_decay` EMA updates, if replaced by a coarser Runtime-driven policy.
8. **Compute loops appearing**: none beyond Runtime's existing per-step `kv_thermo_update`.
9. **Dependencies removed**: none forced — but worth flagging a false-friend risk: KV cache's own `llama_kv_compact_locality_hash` (a real, working, unrelated ring-buffer locality hash) must **not** be confused with or merged into substrate's *stubbed, unimplemented* `kv_dissipate_hilbert` — same "Hilbert" name, unrelated mechanisms, different jobs. Keep them separate.
10. **New dependencies**: KV cache already has `lctx_cached`; the full merge would need substrate to expose a per-cell-capable interface, which doesn't exist yet.

## Scheduler (`ggml_backend_sched_*`)

1. **Exists now**: assigns graph nodes to backends/devices, allocates buffers, determines compute order, manages pipeline parallelism. Generic ggml infrastructure, not llama.cpp- or Mycelium-specific.
2. **Role after Mycelium**: none, currently. Every Mycelium influence so far (bias values, KV eviction gating) operates through ordinary tensor values and ordinary boolean decisions at existing call sites — none of it changes which backend a tensor runs on or how buffers get allocated.
3. **Should it exist**: yes, entirely unchanged. Stating plainly: a prior instruction to "rewrite the scheduler" doesn't correspond to any actual requirement of Mycelium's current math. The one place this *would* become relevant is layer offload (moving cold layers between host/device buffers by Runtime hotness classification) — a real, specifically-named, still-unimplemented feature (substrate's own `layer_offload` is a documented stub: `TODO: call backend offload API`). That touches one narrow slice of buffer management when it's actually built, not the scheduler's graph-partitioning algorithm.
4-10. Unchanged, except the layer-offload exception above.

## Graph builder (per-architecture `build_*` functions in `llama-model.cpp`)

1. **Exists now**: translates a model's hparams into a concrete sequence of `build_attn`/`build_ffn`/`build_moe_ffn` calls for that architecture.
2. **Role after Mycelium**: unchanged — Mycelium's hook lives inside `build_moe_ffn` itself (in `llama-graph.cpp`), not in these per-architecture callers.
3. **Should it exist**: yes, unchanged.
4-10. Unchanged.

## Decode pipeline (`llama_context::decode()`, `process_ubatch()`)

1. **Exists now**: top-level orchestration — batch validation, scheduler reservation, memory update, per-ubatch build+compute+extract loop.
2. **Role after Mycelium**: already the real integration backbone — `full_step_hook()`, the selection-readback block, and `lctx` threading all live here.
3. **Should it exist**: yes — this is the inference loop itself.
4. **Functions becoming redundant**: none structurally, but one concrete, pre-existing wart is worth naming here rather than under Runtime: `llama_decode()` (the free function) currently passes **hardcoded** `kv_hits=kv_total` and `spec_drafted=spec_accepted=0` into `full_step_hook()` — flagged since Iteration 0, never fixed. These should be replaced with real values decode() already computes/has access to.
5. **Functions to rewrite**: once Runtime is a direct Context member (see above), the hook call sites become direct method calls instead of `extern "C"` functions taking a raw pointer — a real simplification, not a behavior change.
6. **Merges**: none — decode() should *call* Runtime, same relationship as today, just less indirection. It should not *become* Runtime.
7-8. No loop changes beyond the hardcoded-values fix above (which corrects existing loops, doesn't add new ones).
9-10. None beyond direct-ownership access.

## ggml graph construction (op primitives: `ggml_add`, `ggml_argsort_top_k`, `ggml_mul_mat`, ...)

1. **Exists now**: ggml's tensor-expression IR — the math primitives everything assembles into a graph.
2. **Role after Mycelium**: purely a toolkit. Everything Mycelium needs today (weighted sum in `graph_propagate`, scalar arithmetic in `graph_feedback`) is done in plain substrate C++, not in the ggml graph at all — no new primitive ops are required.
3. **Should it exist**: yes, obviously — it's the computational foundation.
4-10. Unchanged.

## MoE router (routing-score computation + top-k selection inside `build_moe_ffn`)

1. **Exists now**: decides, per token, which experts' weights get multiplied — softmax/sigmoid gating into `ggml_argsort_top_k`.
2. **Role after Mycelium**: already has a bias hook on its input (done, working). The deeper question — should the *selection mechanism itself* be replaced by a Mycelium algorithm — is explicitly **rejected**, and here's the specific reason: hard top-k over a fixed `n_expert_used` is a load-bearing shape invariant for every downstream tensor (aggregation, weighting all assume that fixed width). A genuinely different paradigm (e.g. soft mixing over all experts, no hard top-k) would require rewriting the entire downstream aggregation math — exactly the code region where this integration's one real bug already happened — with no compiler available to catch a second one. Bias-the-ranking-not-the-mechanism is not a diff-minimization choice here; it's the boundary that keeps tensor shapes provably safe.
3. **Should it exist**: yes, unchanged as a mechanism.
4-10. No redundant functions, no structural merges.

## Speculative decoding (`common/speculative.cpp`)

1. **Exists now**: draft-then-verify generation for latency reduction, independent of MoE.
2. **Role after Mycelium**: currently broken in three specific, already-documented ways: (a) `n_draft` is hardcoded (`// TODO get from config?`), nothing adjusts it; (b) an unconditional "shadow" branch-thermodynamics computation (`branch_entropy/energy/affinity/stability`) is exposed via a stats getter but consumed by nothing; (c) the one real Mycelium call site (`llama_mycelium_spec_report_verification`) is post-hoc telemetry only, the same "inert" pattern the MoE path had before this session fixed it there.
3. **Should it exist**: yes, unchanged in its core draft/verify mechanism.
4. **Functions becoming redundant**: the shadow branch-thermodynamics block — it duplicates state Runtime's own `full_step_hook` already tracks (`spec.collapse_rate_ema` already feeds `migration_instability`/`n_current` today; confirmed in the actual code, not assumed).
5. **Functions to rewrite**: `common_speculative_gen_draft()`'s window-size selection, to call `llama_mycelium_spec_should_run(ctx)` — already fully implemented in substrate, currently called from *nowhere*, a ready accessor sitting idle.
6. **Structures to merge**: shadow branch state → Runtime's existing `spec.collapse_rate_ema`/`n_ema`. Two systems independently computing near-identical "is drafting going well" signals; real merge candidate, not a stretch.
7. **Compute loops disappearing**: the shadow branch computation (`2·accept·reject` etc., run on every acceptance check).
8. **Compute loops appearing**: none — `spec_should_run()` already exists and runs cheaply.
9-10. Same minimal-coupling extern-declaration pattern already used in four other files; nothing architecturally new.

## Buffer ownership

1. **Exists now**: manages actual device memory backing tensors — allocation, lifetime, host/device transfer.
2. **Role after Mycelium**: no general role. The only relevant future feature is the same layer-offload stub named under Scheduler — moving specific weight tensors between buffers based on Runtime hotness classification.
3. **Should it exist**: yes, unchanged, except that one narrow, specifically-named, currently-unimplemented feature.
4-10. Unchanged outside that exception.

## Tensor lifetime pattern (`no_alloc` graph build → post-allocation `set_input`/`set_output`)

1. **Exists now**: the mechanism separating graph *topology* construction (cheap, metadata-only) from *value* population (after backend allocation) and *value* extraction (after compute).
2. **Role after Mycelium**: this is not a subsystem to change — it is the pattern this entire integration is already correctly built on (the bias tensor is a `set_input`; the selected-experts readback is a `set_output`). Worth stating explicitly since it's easy to list "tensor lifetime" as if it needs decomposing: the honest answer is it's infrastructure Mycelium already rides on correctly.
3-10. Unchanged.

## Backend scheduling

Same subsystem as "Scheduler" above — see that entry.

---

## What this decomposition implies, stated once, plainly

Two real, concrete merge targets exist and are worth doing: **KV eviction policy → Runtime** (resolves a pre-existing duplication, not something this integration created) and **speculative shadow state → Runtime** (same pattern, same kind of duplication). Both require Runtime-side extension before the merge is complete (per-cell KV granularity; a real caller for `spec_should_run`) — named as gaps, not glossed over. Everything else — Model, batching, the scheduler, the per-architecture graph builders, ggml's own op primitives, buffer ownership generally, and the tensor-lifetime pattern itself — has no genuine Mycelium-driven reason to change, and forcing changes onto them would trade a real, working separation of concerns for no verifiable gain.
