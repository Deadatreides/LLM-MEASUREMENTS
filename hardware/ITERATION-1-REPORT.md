# Iteration 1 — Runtime

Core: Runtime lifecycle. Per your instruction, build wiring and initialization are treated as part of this iteration rather than a separate one.

---

## Pre-check: can Python remain the sole source of Runtime policy, C++ a thin executor?

You asked me to verify this before touching code, and prefer it if the current implementation already naturally supports it. It doesn't, and adopting it would require changes this iteration is not allowed to make. Evidence:

1. **The substrate's own native code already computes Runtime's core signals itself**, per-step, from data only available natively (decode logits, KV occupancy): `corridor_update_step`, `layer_update_hotness`, `kv_thermo_update`, the negentropy scalar (`N_current`/`N_ema`), `guardrail_check`. These are exactly the algorithms the master prompt names as untouchable ("Hilbert Runtime; Runtime Scoring; Activation Bias; Corridor Logic; Layer Selection"). Making C++ "only a thin execution layer" would mean removing or bypassing this already-built native logic and re-routing every decision through Python — that is a redesign of Runtime, which I was told not to do.
2. **The master prompt's central goal is native participation**: Mycelium must become part of the real `llama_decode()` path, which runs with no Python process attached in the normal case (server, CLI, embedding use). If C++ only executes policy Python supplies, Runtime provides zero behavior whenever Python isn't attached — that doesn't close the architectural gap, it re-enshrines the current broken state under a different name.
3. **What the code already does, correctly, is the opposite emphasis**: the substrate computes its own baseline signals natively every step, and separately exposes `submit_*`/`set_*`/`update_*` entry points for an external actor (today: the Python orchestrator; per the master prompt's stated future: MCP/PTG) to push *periodic, coarse-grained* policy overrides on top — confirmed the calls in `local_runtime.py` (`submit_residency_plan`, `spec_set_window`, `update_flow`, etc.) fire only at epoch boundaries, not per token (`local_runtime.py:631-645`). That is already "Runtime treats external signals uniformly regardless of origin" (master prompt, Part 7) — it's just not reachable from the native path yet, which is exactly what this iteration fixes.

Conclusion: native Runtime stays the primary, always-available decision engine; Python remains an optional periodic contributor, exactly as already designed. No architecture change made or needed for this question.

---

## 1. Goal

Make the existing Runtime lifecycle (`llama_mycelium_init` / `llama_mycelium_runtime_ext_init` / `llama_mycelium_teardown`) actually run as part of every `llama_context`'s life, so that `llama_mycelium_full_step_hook()` — already called from `llama_decode()` — stops being a guaranteed no-op.

## 2. Root cause (from Iteration 0)

- `llama_mycelium_full_step_hook` and `llama_mycelium_spec_report_verification` both begin with `if (!get_rt(ctx)) return;`. `get_rt` only finds entries that `llama_mycelium_init`/`llama_mycelium_runtime_ext_init` would have inserted into the substrate's global maps.
- Nothing in `llama.cpp/` or `common/` ever called either init function — only `llama_free()` called `llama_mycelium_teardown()`. Init and teardown were asymmetric.

## 3. Files modified

**`llama.cpp/src/llama-context.cpp`** — 2 hunks, 6 lines added, 0 removed.

### Hunk 1 — extern declarations (near the existing ones, lines 21-38)
Added forward declarations for the two existing substrate init functions, matching the exact signatures in `substrate/include/llama-mycelium.h:377` and `substrate/include/llama-mycelium-runtime.h:345` (verified `struct llama_mycelium_mcb` / `struct llama_myc_runtime_ext` against the real `typedef struct ... { }` tags in those headers — no ABI mismatch):

```cpp
extern "C" void llama_mycelium_teardown(struct llama_context * ctx);
// paired with llama_mycelium_teardown() below: without this call, full_step_hook()
// and spec_report_verification() find no runtime state for ctx and no-op every time.
extern "C" struct llama_mycelium_mcb  * llama_mycelium_init(struct llama_context * ctx);
extern "C" struct llama_myc_runtime_ext * llama_mycelium_runtime_ext_init(struct llama_context * ctx);
#endif
```

### Hunk 2 — call site, in `llama_init_from_model()` (the only `new llama_context(...)` site in the codebase — confirmed by repo-wide grep)

```cpp
try {
    auto * ctx = new llama_context(*model, params);
#ifdef LLAMA_MYCELIUM_ENABLE
    llama_mycelium_init(ctx);
    llama_mycelium_runtime_ext_init(ctx);
#endif
    return ctx;
} catch (const std::exception & err) {
```

Order (P1 then P2) matches the header's own documented contract (`llama-mycelium-runtime.h:341`, *"Must be called after llama_mycelium_init()"*) and mirrors teardown's reverse order (P2 then P1) already in `llama_free()`. Calling both explicitly rather than relying on P2's internal self-heal-into-P1 (`llama-mycelium-runtime.cpp:116`) makes the lifecycle symmetric and explicit rather than accidental — no new behavior invented, just made deterministic.

No other files needed a code change. `llama-context.h` was left untouched: nothing about this fix requires a new member, method, or declaration there — Runtime state continues to live in the substrate's existing global maps (unchanged design, per the pre-check above and your instruction not to redesign Runtime). `include/llama.h` was left untouched: no new public API was needed:calling two already-existing `extern "C"` functions from inside `llama-context.cpp` requires no new symbol exposure.

## 4. Build wiring — what I did and did not do, and why

I did **not** create `llama.cpp/CMakeLists.txt` or `llama.cpp/src/CMakeLists.txt`. Reasoning:

- Iteration 0 established this checkout is missing ~15 headers/sources (`llama-batch.h`, `llama-model-loader.h`, `llama-adapter.h`, etc.) that a real `add_library(llama ...)` target requires. Any CMakeLists.txt I write can only list the ~18 files actually present here — which would misrepresent the target as buildable when it structurally cannot be, and would silently diverge from (and conflict with) the real upstream `CMakeLists.txt`/`src/CMakeLists.txt` the moment this checkout is reunited with a complete tree. That fails correctness (priority #1) and upstream-compatibility (priority #3) simultaneously to satisfy the letter of "wire the build" — not a trade I should make without flagging it.
- What genuinely closes the gap on the *substrate's* side is already correct and needed no change: `substrate/CMakeLists.txt:22` does `target_compile_definitions(llama_mycelium PUBLIC LLAMA_MYCELIUM_ENABLE)` — `PUBLIC`, meaning any real target that simply does `add_subdirectory(substrate)` + `target_link_libraries(llama PRIVATE llama_mycelium)` gets `LLAMA_MYCELIUM_ENABLE` transitively, for free, no further substrate-side work required.
- So the actual remaining build-wiring work is exactly those two CMake lines, in a `llama.cpp/CMakeLists.txt` (or `src/CMakeLists.txt`) that doesn't yet exist in this checkout and that I'm not able to safely reconstruct in full. This is recorded as the first item in §6 below rather than papered over.

## 5. Inference path — before and after

**Before:**
```
llama_decode(ctx, batch) → ctx->decode(...) → ... → llama_mycelium_full_step_hook(...)
                                                        → get_rt(ctx) == nullptr → return  (always, unconditionally)
```
Mycelium contributed nothing observable to any inference run, even in a hypothetical build where `LLAMA_MYCELIUM_ENABLE` was defined.

**After (once a real build defines `LLAMA_MYCELIUM_ENABLE` and links `llama_mycelium` — still blocked by §4/§6):**
```
llama_init_from_model(...) → new llama_context(...) → llama_mycelium_init(ctx) → llama_mycelium_runtime_ext_init(ctx)
  [ctx now has live entries in g_mcb_map / g_rt_map]

llama_decode(ctx, batch) → ctx->decode(...) → ... → llama_mycelium_full_step_hook(...)
                                                        → get_rt(ctx) != nullptr
                                                        → post_decode_hook (entropy/curvature/drift EMA)
                                                        → kv_report_step, kv_thermo_update
                                                        → layer hotness decay pass
                                                        → corridor_update_step
                                                        → ring_write

llama_free(ctx) → llama_mycelium_teardown(ctx) → g_rt_map/g_mcb_map entries freed
```
The hook now does real work and its state persists correctly across decode calls for the lifetime of the context. **This does not yet mean any of that state influences the graph, KV eviction, or sampling** — that remains true by design; those are later iterations (Graph, KV, Sampler), and per your instruction I have not touched `llama-graph.cpp`, `llama-kv-cache.cpp`, `llama-sampler.cpp`, or `common/speculative.cpp` in this iteration.

## 6. Remaining architectural gaps (unchanged scope, not addressed here)

1. **No CMake target actually defines `LLAMA_MYCELIUM_ENABLE` or links `llama_mycelium`** in any buildable configuration in this checkout (§4). Two lines once a real `llama.cpp` build file exists; cannot be safely authored from scratch without risking incorrect/conflicting content.
2. **Even with #1 solved, this checkout still cannot compile** — the ~15 missing headers/sources are unrelated to Mycelium and pre-date this work (documented in Iteration 0 §0).
3. Runtime state still lives in global maps keyed by `llama_context*`, not inside `llama_context` itself. Per your instruction not to redesign Runtime, this was deliberately left as-is — the header comment in `llama-mycelium.h:107-108` already documents it as a known stopgap for whenever that redesign is separately approved.
4. Graph, KV eviction, and Sampler still do not consume any Runtime-computed signal (Corridor, Activation Bias, Heat, Residency) — unchanged, out of scope for this iteration by design (Graph/KV/Sampler are their own iterations).
5. Two dead-code items identified in Iteration 0 (`llama_exec_corridors` in `llama-graph.cpp`, the unconditional "branch thermodynamics" shadow heuristic in `common/speculative.cpp`) were **not removed** — they live outside this iteration's Runtime/Context scope, and removing them now would mix this iteration's core with Graph's and Speculative's, which the master prompt's "one core per iteration" rule and your "do not redesign the implementation plan" instruction both rule out. They remain candidates for their respective future iterations.
6. `layer_update_hotness` inside `full_step_hook` is still only ever called with `access_weight = 0.0` (decay-only) — no real per-layer access signal exists yet. Unaffected by this iteration; would need Graph-iteration work to produce that signal.

No blocker requiring a new architectural decision was hit. §4 is a factual constraint (files absent, cannot be safely fabricated), not a design choice — flagged per your reporting requirements, not as a question.
