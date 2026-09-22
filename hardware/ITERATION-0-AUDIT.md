# Iteration 0 — Full Architectural Audit
### Mycelium Runtime (llama.cpp fork)

Status: **read-only analysis, no code changed.** Per the master prompt this is mandatory before any Iteration 1 work.

Method: every claim below is grounded in a `file:line` citation, personally verified or cross-verified against the actual source in this checkout (`<PROJECT_ROOT>`), not inferred from Codex's prior transcript or summary. Where this audit corrects or refines Codex's prior conclusions, that is called out explicitly in §12.

---

## 0. Environment caveat (read this before trusting any "PARTIAL/READY" verdict below)

This checkout **cannot be compiled**:

- No `.git` anywhere (`git status` → "not a git repository").
- **No `CMakeLists.txt` (or any build file) exists anywhere under `llama.cpp/`.** The *only* build file in the entire repository is `substrate/CMakeLists.txt`, a standalone project for the substrate static library alone.
- `llama.cpp/src/llama-context.cpp`, `llama-kv-cache.h`, and `common/speculative.cpp` `#include` headers that do not exist anywhere in this tree: `llama-batch.h`, `llama-mmap.h`, `llama-io.h`, `llama-ext.h`, `llama-model.h`, `llama-sampler.h`, plus the corresponding `.cpp` files. `llama.cpp/src/` here holds 34 files total — a real llama.cpp `src/` normally has 50+.
- `code_dump.txt` (a full-tree dump made by `damp.py`, unrelated to this session) independently confirms this is the true on-disk state, not an artifact of how the project was shared in chat.

Per your decision, this is treated as ground truth and analysis proceeds anyway. **Every "READY/PARTIAL/BROKEN/STUB" verdict below is from static reading and cross-referencing calls, not from a compiler, linker, or test run.** Iteration exit criteria that require "code compiles / tests pass" cannot be satisfied in this environment until a real build system is restored — this itself is flagged as an open question in §13.

---

## 1. Project file map

174 source files total (per `code_dump.txt`, excludes binary/data/log files). Grouped by architectural category:

| Category | Files | Notes |
|---|---|---|
| **Core Context** | `llama.cpp/src/llama-context.cpp` (4045 lines), `llama-context.h` (377 lines) | The only upstream file with any Mycelium awareness (2 extern "C" decls + 2 call sites) |
| **Core Graph** | `llama-graph.cpp` (2960), `llama-graph.h` (1067) | Contains an undocumented, unrelated-to-substrate dead corridor tracker (§5) |
| **KV Cache** | `llama-kv-cache.cpp` (2626), `llama-kv-cache.h` (434), `llama-kv-cells.h` | Contains an undocumented, unrelated-to-substrate native heat/eviction heuristic (§6) |
| **Model** | `llama-model.cpp` (2551) | No `llama-model.h` present in this checkout. Zero Mycelium references. Vanilla. |
| **Sampling** | `llama-sampler.cpp` (3885) | No `llama-sampler.h` present. Zero Mycelium references. Vanilla. |
| **Vocab/Arch/Impl/Memory** | `llama-vocab.cpp/h`, `llama-arch.cpp/h`, `llama-impl.cpp/h`, `llama-memory.cpp/h` | Vendor code, untouched |
| **Public API** | `include/llama.h`, `include/llama-cpp.h` | Zero Mycelium/corridor/runtime/telemetry symbols added — confirmed no public API surface exists for this subsystem |
| **Common** | `common/common.cpp/h`, `common/sampling.cpp/h`, `common/reasoning-budget.cpp/h`, `common/speculative.cpp/h` | `speculative.cpp` has 1 real Mycelium call site + 1 unrelated dead shadow heuristic (§ call graph) |
| **GGML** | `ggml/include/*.h`, `ggml/src/ggml-backend*.{cpp,h}` | Vendor code, untouched, partial subset only |
| **Mycelium substrate (native, C++)** | `substrate/include/llama-mycelium.h`, `llama-mycelium-runtime.h`, `substrate/src/llama-mycelium.cpp`, `llama-mycelium-runtime.cpp`, `substrate/tests/*` (2 files), `substrate/CMakeLists.txt` | The **one** subsystem actually wired (in principle) to `llama.cpp/src`. Standalone-buildable, isolated from real llama.cpp internals by design (`llama-mycelium.cpp:19-25` admits this explicitly) |
| **Mycelium legacy/prototype C++ (root-level)** | `mycelium-cavity.cpp`, `mycelium-corridor.cpp`, `mycelium-epoch.cpp`, `mycelium-guardian.cpp`, `mycelium-hilbert.cpp`, `mycelium-residency.cpp`, `mycelium-telemetry.cpp` | **Not in any build.** Not even internally consistent — see §11 |
| **Mycelium Python orchestration** | `mycelium/core/*.py` (17 files), `mycelium/main.py`, `mycelium/orchestrator.py` (1448 lines) | Fully implemented, all reachable from `main.py`. The most mature layer of the whole project |
| **Mycelium legacy Python** | `mycelium/legacy/search.py` (orphan/quarantined), `mycelium/legacy/trainer.py` (reachable via `--nightly`) | |
| **Mycelium config** | `mycelium/config/settings.yaml`, `models.yaml` | `settings.yaml` has a duplicate-key bug (see §1a) |
| **Tests (Python)** | `mycelium/tests/*.py` (9 files, 2291 lines) | All import real `core/*` modules; no stubs |
| **Bridge** | `bridge/mycelium_bridge.py` (639 lines) | ctypes ABI bridge; all 27 bound symbols verified to exist in substrate headers with matching signatures |
| **Swarm** | `swarm/{context,evolution,pool,tasks}/` | **All four directories are completely empty.** `pool/` and `tasks/` are pure dead scaffolding (created by `os.makedirs`, never referenced again anywhere). `context/`/`evolution/` are empty because their only producers are gated/unused code paths |
| **Data/logs** | `data/`, `logs/` | Runtime output paths, not source |

**1a.** `mycelium/config/settings.yaml` defines top-level `pseudomoe:` twice (lines 193 and 208) and `mge:` twice (195 and 238). YAML keeps only the second occurrence — the first blocks are silently dead config. Minor, but worth a one-line fix whenever that file is next touched.

---

## 2. Full call graph — `llama_decode()` to token, as it actually exists in this code

### 2a. Pure-native path (no Python involved)

```
llama_decode(ctx, batch)                                    [llama-context.cpp:3936]
  t0 = ggml_time_us()
  ret = ctx->decode(batch)                                  [instance method — real work happens here]
    ├─ validate batch / memory / token inputs
    ├─ scheduler reservation, memory_update(false)
    ├─ memory->init_batch(...)
    └─ for each ubatch:
         process_ubatch(...)
           ├─ mctx->apply()
           ├─ model.build_graph(...)
           │    └─ llm_graph_context::cb(cur, name, il)      [llama-graph.cpp:989-996, called per named tensor]
           │         └─ llama_exec_corridor_touch(...)        [llama-graph.cpp:991 — WRITE-ONLY, see §5]
           ├─ (within KV slot allocation) llama_kv_cache::find_slot(...)
           │    └─ heat_score() / heat_touch() / heat_decay() [llama-kv-cache.cpp:863-909 — see §6, active only for n_swa>0 models]
           ├─ ggml_backend_sched_alloc_graph(...)
           ├─ set graph inputs
           └─ graph_compute(...) → ggml_backend_sched_graph_compute_async(...)
    └─ copy logits / embeddings / output rows
  decode_us = elapsed
  if (ret == 0 || 1) and batch.n_tokens > 0:
    #ifdef LLAMA_MYCELIUM_ENABLE                              [never defined by any build in this checkout, §0]
      n_vocab = model.vocab.n_tokens()
      logits  = last output row
      kv_total = kv_hits = batch.n_tokens        ← HARD-CODED, always 100% "hit rate"
      spec_drafted = spec_accepted = 0            ← HARD-CODED, always zero
      llama_mycelium_full_step_hook(ctx, logits, n_vocab, last_tok, decode_us,
                                     kv_total, kv_hits, spec_drafted, spec_accepted, ...)
        └─ rt = get_rt(ctx)                                   [llama-mycelium-runtime.cpp:803-804]
             if (!rt) return;   ← ALWAYS TRUE on this path, see §3 — hook is a no-op
    #endif
  return ret

── caller-side (outside llama_decode, in whatever CLI/server code drives inference) ──
llama_sampler_sample(smpl, ctx, idx)                          [llama-sampler.cpp:806]
  → builds token_data_array from ctx logits, applies registered llama_sampler_i chain
  → ZERO Mycelium coupling anywhere in this file

── if speculative decoding is used (common/speculative.cpp) ──
common_speculative_gen_draft(...)  uses n_draft = 8            [speculative.cpp:1151, "// TODO get from config?"]
common_speculative_accept(...)
  ├─ impl->branch.branch_entropy/energy/affinity/stability     [speculative.cpp:1553-1557 — unconditional, WRITE-then-getter-only, see below]
  └─ #ifdef LLAMA_MYCELIUM_ENABLE
       llama_mycelium_spec_report_verification(spec->ctx_tgt, ...)  [speculative.cpp:1561]
         └─ same get_rt(ctx)==nullptr guard → no-op on this path
     #endif
common_speculative_get_stats(...) → returns aggregated impl->branch.* to caller; nothing internal reads it back to adjust n_draft
```

**Net result of the pure-native path: this fork currently behaves identically to upstream llama.cpp.** Every Mycelium touchpoint either (a) doesn't compile in (macro never defined), or (b) no-ops via the `get_rt(ctx) == nullptr` guard even if it did.

### 2b. Python-orchestrator-driven path (the apparent intended production path)

```
mycelium/main.py → orchestrator.py → core/local_runtime.py (LlamaCppAdapter)
  _load():
    self._llm = llama_cpp.Llama(model_path=...)    ← third-party PyPI `llama-cpp-python` package,
                                                        NOT built from this checkout (no build system exists to produce it)
    _attach_mycelium_bridge():
      loads bridge/mycelium_bridge.py, MycBridge(lib_path)
      bridge.init(ctx_address)
        → ctypes call to llama_mycelium_init(ctx) + llama_mycelium_runtime_ext_init(ctx)   [bridge.py:374,379]
        → ONLY takes effect if `lib_path` points at a shared library that (a) exports these
          symbols AND (b) is the exact same loaded binary llama_cpp_python's Llama object
          is using internally (same process address space / same llama_context*).
          Producing such a fused binary requires building substrate/ together with a
          LLAMA_MYCELIUM_ENABLE llama.cpp AND linking that into llama-cpp-python's build —
          no tooling for this exists anywhere in this checkout.

  per generated token, _decode_loop():
    self._capi.llama_decode(ctx, batch)             [local_runtime.py:421 — raw ctypes call into whatever lib is loaded]
    logits_view = self._get_logits_view()
    self._inject_logit_compensation_inplace(logits_view)   ← Python-side bias injection, own numpy code
    logits = self._apply_pseudomoe_logit_mask(logits)      ← Python-side expert-routing mask, own numpy code
    → Python then runs ITS OWN sampling on this numpy array — llama-sampler.cpp is never invoked
      for token selection on this path at all.

  policy pushed INTO substrate via bridge (real calls, but see effect column):
    self._bridge.submit_residency_plan(...)   → stored in mcb.residency; cell-level enforcement is a stub (§3)
    self._bridge.kv_pin(...) / kv_prefetch(...) → real: read back by eviction-skip logic, but the eviction
                                                    action itself is a stub outside LLAMA_MYCELIUM_INTERNAL (never defined)
    self._bridge.kv_dissipate(sigma_hat, anchor) → llama_mycelium_kv_dissipate_hilbert() — literal no-op stub (§3, §4)
    self._bridge.spec_set_window(k_draft) / spec_force_off(...) → writes into substrate spec state;
                                                    common/speculative.cpp NEVER reads this back — no connection exists
    self._bridge.update_flow(...), set_emergency(...) → real writes into substrate state

  telemetry pulled FROM substrate (real, thin):
    self._bridge.get_runtime_state()   [local_runtime.py:583]
    self._bridge.export_json()         [local_runtime.py:595,697,735]
```

**Net result of the Python path:** there is a real, working, bidirectional ctypes bridge and the Python orchestrator does actively drive substrate state — but (1) it requires a fused build artifact this repository cannot currently produce, (2) several of the policy pushes land on stub functions, (3) actual token selection happens in a **separate Python-native sampler that bypasses `llama-sampler.cpp` entirely** rather than "integrating" with it, and (4) the Python-pushed speculative controls have no native consumer regardless of build state.

---

## 3. Runtime lifecycle (substrate: `llama-mycelium.cpp`/`.h` + `llama-mycelium-runtime.cpp`/`.h`)

**Ownership:** entirely global, keyed by `llama_context*` in two separate `unordered_map`s guarded by mutexes — **not** a member of `llama_context`.

```cpp
// substrate/src/llama-mycelium.cpp:113-114 (P1 / MCB)
static std::mutex g_mcb_map_mu;
static std::unordered_map<struct llama_context *, llama_mycelium_impl *> g_mcb_map;

// substrate/src/llama-mycelium-runtime.cpp:62-63 (P2 / runtime ext)
static std::mutex g_rt_map_mu;
static std::unordered_map<struct llama_context *, llama_myc_runtime_impl *> g_rt_map;
```

The header itself documents this as a known stopgap: *"Since we can't modify llama_context directly here, we use a global map. In a full integration the pointer would live inside llama_context directly."* (`llama-mycelium.h:107-108`).

**Creation:** `llama_mycelium_init(ctx)` / `llama_mycelium_runtime_ext_init(ctx)`. **Confirmed by repo-wide grep: called only from (a) their own definitions, (b) each other (P2 self-heals into P1 if missing, `llama-mycelium-runtime.cpp:116`), (c) substrate's own tests, (d) the Python bridge (`bridge.py:374,379`).** Never called from `llama.cpp/` or `common/`.

**Destruction:** `llama_free(ctx)` calls `llama_mycelium_teardown(ctx)` before `delete ctx` (`llama-context.cpp:3477-3482`) — this **is** a real native call site. `teardown` correctly frees both maps in reverse order (P2 then P1), idempotent, no leak/double-free risk found. But since nothing on the native path ever populated the maps, teardown on that path is a no-op-on-empty-map, not a bug.

**The consequence that matters most:** `llama_mycelium_full_step_hook` (the one native decode-path entry point) begins with:
```cpp
auto * rt = get_rt(ctx);
if (!rt) return;                    // llama-mycelium-runtime.cpp:803-804
```
Since `get_rt` only finds entries the never-called `init` functions would have created, **every invocation from `llama_decode()` on the pure-native path returns immediately**, before any of the telemetry/state math below ever executes.

**What the hook computes when it *is* live** (i.e., only reachable via the Python-bridge path, §2b):

| Concept | Actually computed? | Feedback loop across calls? | Notes |
|---|---|---|---|
| Corridor usage (`corridor_update_step`) | Yes | Yes — `usage_ema`, `j_star` sync persist and are read back next call | Real, working, isolated to substrate's own state |
| Layer Heat/Hotness (`layer_update_hotness`) | Yes, but **`full_step_hook` only ever calls it with `access_weight = 0.0`** | Technically yes (EMA persists) | In practice this means hotness only ever decays toward zero — no real per-layer access signal is ever fed in from anywhere in llama.cpp |
| KV Pressure/Thermo (`kv_thermo_update`) | Yes — occupancy fraction, fragmentation heuristic | Yes — read by `guardrail_check` | Real |
| Activation Bias (`apply_activation_bias`) | **No — literal stub, always `return false`, does nothing** (`llama-mycelium.cpp:568-582`) | n/a | Comment: *"Placeholder... TODO: wire into llama_context::graph_get_cb()"* |
| Hilbert-ordered KV dissipation (`kv_dissipate_hilbert`) | **No Hilbert ordering exists anywhere in the file.** Computes a scalar `pressure`; real range-removal is gated behind `LLAMA_MYCELIUM_INTERNAL`, never defined by any build here | n/a | `impl->mcb.runtime.kv_evicted` is hard-set to `0` on the always-compiled path |
| Layer mask / offload | Written and stored (`impl->active.layer_masks[]`, `layer_offloaded[]`) | Stored, readable | **Nothing in llama.cpp's actual graph-building code reads these arrays anywhere in the repo** — write-and-park |
| Residency plan | Stored via `submit_residency_plan` | `pinned_positions` read back to skip eviction (real) | Cell-level enforcement itself is the stub above |
| Spec window / force-off | Written into substrate spec state | n/a | **Never read by `common/speculative.cpp`** — no connection exists regardless of init state |
| H_diag (Hessian proxy) | **No — literal stub, zero-fills, returns `false`** (`llama-mycelium.cpp:477-491`) | n/a | |

**5 unresolved TODOs remain in the substrate's own "live" implementation** (not the abandoned root files): `llama-mycelium.cpp:346,488,546,579,608` — backend offload, `graph_get_cb` wiring for bias, Hilbert eviction. These mark real, acknowledged, unfinished integration points even in the code path that is closest to working.

**13 exported substrate API functions have zero callers anywhere in the repository** (not native code, not bridge.py, not root files, not even tests beyond their own): `corridor_register`, `corridor_set_state`, `corridor_get_meta`, `kv_enqueue_eviction`, `kv_flush_eviction_queue`, `kv_thermo_get`, `layer_get_meta`, `layer_set_state`, `pending_count`, `ring_ptr`, `ring_read`, `spec_begin_draft`, and `full_step_hook` itself is only called from the one dead native site. A stale docstring (`llama-mycelium-runtime.h:24`) even references a function name (`llama_mycelium_runtime_ring_ptr`) that doesn't exist — the real name is `llama_mycelium_ring_ptr`.

**Verdict: BROKEN.** The substrate is the most architecturally complete of the three "layers" (native shadow heuristics, substrate, Python), and it does have real internal feedback loops — but it is functionally disconnected from actual inference in the one build configuration this repository can produce.

---

## 4. Context lifecycle (`llama-context.cpp`/`.h`)

- `struct llama_context` (`llama-context.h:41`) has **zero** Mycelium-related members. Confirmed by direct read of the full public/private interface.
- Creation: `llama_init_from_model()` → `new llama_context(...)` (`llama-context.cpp:3461-3462`). **No Mycelium init call anywhere in this path.**
- Destruction: `llama_free()` → `llama_mycelium_teardown(ctx)` then `delete ctx` (`llama-context.cpp:3477-3482`). This is the only lifecycle touchpoint, and it's asymmetric — teardown with no matching init.
- Decode: `llama_decode()` (free function, not a `llama_context` method) is the one place the hook is called, strictly after `ctx->decode()` returns (§2a).

**Verdict: the master prompt's core Invariant 6 ("один владелец каждой сущности") is violated for Runtime state specifically** — Context should own it per the architecture's stated intent, and currently doesn't; this is acknowledged in the substrate's own source comments (§3).

---

## 5. Graph lifecycle (`llama-graph.cpp`/`.h`)

Graph is built per-ubatch inside `process_ubatch()` → `model.build_graph()`. The **only** per-layer/per-tensor touchpoint in the entire file is:

```cpp
void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {   // llama-graph.cpp:989-996
    if (il >= 0) {
        llama_exec_corridor_touch(llama_exec_corridor_id(name, il));
    }
    if (cb_func) { cb_func(ubatch, cur, name, il); }
}
```

This feeds a **file-local, always-compiled (not gated by any `#ifdef`), completely separate from the substrate** corridor tracker:

```cpp
struct llama_exec_corridor_state { float heat; uint32_t reuse; };   // llama-graph.cpp:22-27
static llama_exec_corridor_state llama_exec_corridors[16];
```

**Confirmed dead: `llama_exec_corridors[...]` is written on every single graph-build tensor callback and read by nothing, anywhere** — no logging, no export, no decision logic, no connection to the substrate's own (unrelated) corridor concept of the same name. This is the clearest example in the codebase of the master prompt's Invariant 10 violation ("мёртвые вычисления запрещены").

Separately: `build_moe_ffn()` (`llama-graph.h:834,854`) is the existing, vanilla, upstream expert-routing extension point (standard llama.cpp MoE gating) — untouched, and the natural place any future Layer Selection / pseudo-expert routing work described in the master prompt would need to attach, since nothing Mycelium-specific exists there today.

No Layer Mask, Activation Bias, or Corridor-from-substrate value is read anywhere during graph construction.

**Verdict: BROKEN** (dead write-only instrumentation) for the local corridor tracker; **UNUSED** for any substrate-originated signal — Graph currently builds identically to vanilla upstream.

---

## 6. KV cache lifecycle (`llama-kv-cache.cpp`/`.h`)

Two entirely separate, non-communicating "heat" systems exist:

### 6a. Native in-file heat heuristic (always compiled, not gated by any Mycelium flag)

```cpp
struct llama_kv_heat {                                    // llama-kv-cache.h:20-26
    float hotness, reuse_probability, semantic_affinity, corridor_affinity, migration_cost;
};
static float llama_kv_heat_value(const llama_kv_heat & h); // llama-kv-cache.cpp:44-50, weighted sum
```
`heat_score()` / `heat_touch()` / `heat_decay()` (`llama-kv-cache.cpp:863-909`) are real, class-member functions. `heat_touch`/`heat_decay` run unconditionally on every KV cell write/removal (`:1165,1182`). `heat_score`'s *effect on eviction* is gated:

```cpp
bool heat_strict = !cont && n_swa > 0;                     // llama-kv-cache.cpp:1041
if (can_use && heat_strict && heat_score(...) > 0.45f) { can_use = false; }   // :1090
```

**Both call sites of `find_slot()` pass `cont=false` literally** (`:731`, `:2237` — no call site anywhere passes `true`), so `!cont` is always true. `n_swa` comes from `hparams.n_swa`, which defaults to `0` for non-sliding-window-attention architectures. **Net effect: `heat_strict` is silently always-`false` (heat-based eviction fully inactive) for ordinary dense models — the common case — and always-`true` only for SWA-architecture models.** This is neither a documented feature flag nor an obvious bug; it's an undocumented architecture-dependent kill-switch on a heuristic that otherwise looks fully wired.

### 6b. Substrate's KV concepts (Pressure, Thermo, Residency, Hilbert dissipation)

Entirely separate storage (`rt->ext.kv_thermo` in the P2 global map), entirely separate math, **zero cross-references between this and `llama_kv_heat`** — confirmed by grep, no shared symbols, no shared field names. Per §3: pressure/occupancy computation is real; actual eviction/dissipation enforcement is a stub outside the never-defined `LLAMA_MYCELIUM_INTERNAL` build.

**Verdict: PARTIAL for 6a** (real but silently inert for most models), **BROKEN/STUB for 6b** (computed, partially stored, enforcement not implemented in any buildable configuration), and **the two systems that both claim the concept "KV heat" never talk to each other** — a second, more subtle instance of the master prompt's "один источник истины" (Invariant 8) violation.

---

## 7. Sampler lifecycle (`llama-sampler.cpp`)

`llama_sampler_sample(smpl, ctx, idx)` (`llama-sampler.cpp:806`) — confirmed **zero** references to mycelium/corridor/bias/hilbert/runtime anywhere in this 3885-line file. Fully vanilla chain-of-`llama_sampler_i` architecture, untouched.

The only place any Mycelium-adjacent signal touches token selection at all is **Python-side**, and it bypasses this file completely: `core/local_runtime.py`'s `_decode_loop` applies `_inject_logit_compensation_inplace` and `_apply_pseudomoe_logit_mask` directly to a numpy copy of the logits, then runs its own Python sampling — `llama-sampler.cpp` is never invoked for token selection on that path either.

**Verdict: READY (untouched, vanilla) for the native file. UNUSED for any Mycelium coupling at any layer of the system** — "sampler integration" doesn't currently exist as an integration; where sampling-time influence exists at all, it's a full bypass implemented independently in Python.

---

## 8. Telemetry lifecycle

Two independent telemetry systems exist:

**8a. Substrate ring buffer** (`llama_mycelium_ring_write`/`ring_read`/`ring_ptr`/`export_json`) — 128-entry ring, written once per `full_step_hook` call (§3). Real, but per §3 only reachable via the Python-bridge path.

**8b. Python `core/telemetry.py`** (`RuntimeTelemetry`, `RuntimeState`, `NegentropyState`) — fully implemented, computes `sigma_hat` (its primary decision currency) and negentropy from logits it captures directly via its own ctypes/numpy path, **independent of the substrate ring** — it does not read `ring_read`/`export_json` to source its own math.

**What actually connects them:** `local_runtime.py` calls `self._bridge.get_runtime_state()` / `export_json()` at a handful of points (`:583,595,697,735`) to pull substrate state into the Python decision loop, and pushes decisions back via the bridge calls cataloged in §2b/§3. So a **thin, real, bidirectional channel exists between Python telemetry/decisions and substrate state** — but per the master prompt's own required cycle (`Runtime → Graph → KV → Sampler → Telemetry → Runtime`, Invariant 16, "no additional loops"), this is a different topology entirely: it's `(Python telemetry) → (Python decision) → (substrate state, mostly-stubbed enforcement)`, with the substrate's own native telemetry (post-decode hook) never live in the first place on the path that actually runs inference through `llama_cpp_python`.

**Verdict: PARTIAL.** Telemetry is collected (twice, redundantly, in two unconnected systems) but the "Runtime gets feedback" loop the master prompt requires exists only in the Python layer, driving a mostly-inert substrate, never closing back into the native graph/KV/sampler path at all.

---

## 9. Subsystem status table

| Subsystem | Status | Evidence |
|---|---|---|
| Substrate Runtime (P1/P2 state, global maps) | **BROKEN** | Real internal logic + feedback loops, but `get_rt(ctx)==nullptr` on every native call site (§3) |
| Corridor (substrate) | **PARTIAL** | Computed + stored + read back within substrate; never reaches Graph; native path never initializes it |
| Corridor (native shadow, `llama-graph.cpp`) | **BROKEN (dead code)** | Write-only 16-slot array, zero readers anywhere (§5) |
| Activation Bias | **STUB** | Literal no-op in both native call point and substrate impl; never even attempted (§3) |
| Layer Heat/Hotness (substrate) | **PARTIAL** | Real EMA machinery, but only ever fed `access_weight=0` — decays to zero, no real signal source exists |
| KV Heat (native, `llama-kv-cache.cpp`) | **PARTIAL** | Real and wired, but silently inert (`heat_strict` always false) for all non-SWA models (§6a) |
| KV Pressure/Thermo (substrate) | **PARTIAL** | Computed and read by guardrail internally; never reaches actual eviction decisions in any buildable config |
| Hilbert-ordered KV dissipation | **STUB** | No Hilbert ordering implemented anywhere; eviction execution gated behind a macro no build defines (§3) |
| Residency plan | **PARTIAL** | Stored + pin-list read back (real); cell-level enforcement is the same stub as above |
| Speculative control (native) | **BROKEN** | Telemetry report call exists but inert (`get_rt`==nullptr guard); `n_draft` hardcoded, never adjusted by anything |
| Speculative shadow heuristic (`speculative.cpp` branch stats) | **BROKEN (dead code)** | Unconditionally computed, exposed via a stats getter, never read internally to change behavior (§2a) |
| Sampler integration (native) | **UNUSED** | `llama-sampler.cpp` has zero Mycelium references |
| Sampler-equivalent (Python bypass) | **READY, but architecturally out-of-loop** | Fully functional, but bypasses the native sampler entirely rather than integrating with it |
| Telemetry (substrate ring) | **PARTIAL** | Real, but only live via the Python-bridge path |
| Telemetry (Python `core/telemetry.py`) | **READY**, operationally isolated | Fully implemented; its only leverage over inference is via the mostly-stubbed bridge pushes and the Python-native sampler bypass |
| Mycelium Python orchestration (`core/*`, 17 modules) | **READY** | All fully implemented, all reachable, well-tested — the most mature layer in the project |
| Root-level `mycelium-*.cpp` (7 files) | **UNUSED / non-compilable** | Not in any build; contains genuine compile-blocking bugs independent of the "not linked" problem (§11) |
| `legacy/search.py` | **UNUSED (deliberately quarantined)** | Fully built, but `orchestrator.py` swaps in an inline no-op `QuarantinedSearchModule` instead |
| `swarm/pool/`, `swarm/tasks/` | **UNUSED (dead scaffolding)** | Created, never referenced again anywhere |

---

## 10. Architectural gap table

| Subsystem | Where computed | Where it should be used | Why it isn't |
|---|---|---|---|
| Substrate Runtime state (all of it) | `substrate/src/llama-mycelium-runtime.cpp` | Read at the top of every `llama_decode()`/graph-build/KV/sampler step | Never initialized on the native path (`get_rt`→nullptr); no build produces a binary where `LLAMA_MYCELIUM_ENABLE` is even defined |
| Corridor mask / Layer mask / Layer offload | `substrate` (stored), or the dead `llama-graph.cpp` array | `model.build_graph()` node selection | No graph-building code anywhere reads either representation |
| Activation Bias | Nowhere (stub) | Sampler logits or graph pre-activation | Never implemented past a TODO comment, on either the native or substrate side |
| Heat/Hotness (per-layer) | `substrate` (decay-only) | Layer offload/residency decisions | `access_weight` is always fed as `0` — no real per-layer usage signal is ever produced by llama.cpp and passed in |
| KV Pressure/Residency/Hilbert dissipation | `substrate` (computed) / nowhere (Hilbert) | `llama_kv_cache::find_slot()` eviction/eviction ordering | Substrate and native KV cache are two disconnected systems (§6); enforcement gated behind an always-off macro |
| Speculative draft window | `common_speculative_get_stats()` (getter only) / substrate spec state (Python-writable) | `common_speculative_gen_draft()`'s `n_draft` | `n_draft` is a hardcoded `8`; nothing reads either signal to adjust it |
| Sampler-time bias/mask | Python (`local_runtime.py`) | `llama-sampler.cpp` chain | Implemented as a full bypass instead of an integration — native sampler never sees any Mycelium signal |
| Telemetry (substrate ring) | `substrate` | Feed back into Runtime's next-step decisions | Only ever populated via the Python path; the native decode-only path never writes to it either |

---

## 11. Original llama.cpp core files — status table

| File | Status | Explanation |
|---|---|---|
| `llama.cpp/src/llama.cpp` | *(not present as a separate top-level file in this checkout — `llama-context.cpp` contains the free-function API surface including `llama_decode`)* | n/a |
| `llama-context.cpp` | **PARTIAL** | 2 real call sites (teardown, post-decode hook), both structurally inert on the native path; no init call anywhere |
| `llama-context.h` | **PARTIAL/BROKEN** | Zero Mycelium-owned state; violates the architecture's own stated ownership intent (acknowledged in substrate source comments) |
| `llama-graph.cpp`/`.h` | **BROKEN** (for the local corridor code) / **READY, vanilla** (for everything else) | Dead write-only instrumentation; no consumption of any Mycelium signal in graph construction |
| `llama-kv-cache.cpp`/`.h` | **PARTIAL** | Real but silently inert heat heuristic for the common (non-SWA) case; substrate KV policy fully disconnected |
| `llama-model.cpp` (no `.h` present) | **READY, vanilla** | Zero Mycelium references found |
| `llama-sampler.cpp` (no `.h` present) | **READY, vanilla** | Zero Mycelium references found |
| `include/llama.h` | **READY, vanilla** | No public API added for this subsystem at all |
| `common/speculative.cpp`/`.h` | **BROKEN** | Report call exists but inert; parallel dead shadow heuristic; draft window hardcoded |

---

## 12. Corrections and additions to Codex's prior analysis

Where this audit **confirms** Codex's summary (`итоговое резюме кодекса.txt`): the post-hoc hook timing, the global-map ownership problem, "sampler integration absent," "speculative telemetry reported but not used to drive draft behavior," and the general assessment that the substrate is closer to real than the root-level files.

Where this audit **refines or corrects** it:

1. **"If the Python bridge has not initialized Mycelium state for that context, the hook can return without meaningful work"** — this undersells it. It is not conditional in practice: **nothing in `llama.cpp/` or `common/` ever calls the init functions.** The hook *always* no-ops on any pure-native build; only a specific, currently-unbuildable, fused Python+native deployment would ever populate it.
2. **"Native decode still uses normal KV allocation and eviction behavior"** — partially wrong. A real, non-substrate, fork-local heat-based eviction heuristic exists and is wired into `find_slot()` (§6a) — it's just silently disabled for every non-SWA model via `heat_strict = !cont && n_swa>0` with `cont` always `false`. For the common case the *outcome* (vanilla eviction) is correct, but the mechanism claim was imprecise, and this dead-for-most-models heuristic was not mentioned at all.
3. **Not surfaced by Codex at all:**
   - `llama.cpp/` has **no build system whatsoever** in this checkout — not just "build integration incomplete," but literally zero CMakeLists.txt anywhere under it.
   - The dead write-only corridor array in `llama-graph.cpp` (§5).
   - The dead write-only "branch thermodynamics" shadow heuristic in `common/speculative.cpp` (§2a), separate from the Mycelium-gated report call.
   - Activation Bias and Hilbert-ordered KV dissipation are not merely "unused" — they are **literal no-op stubs** with explicit `TODO: wire into llama_context::graph_get_cb()` comments; nothing to "connect," they'd need to be written first.
   - 13 substrate API functions with zero callers anywhere in the repository, and one stale docstring referencing a function name that doesn't exist.
   - The 7 root-level `mycelium-*.cpp` files don't just risk "duplicate symbol" problems in the abstract — they have **genuine compile-blocking bugs independent of any linking concern**: `mycelium-residency.cpp` includes a non-existent header (`mycelium-hilbert.h`); `llama_myc_hilbert_item` is used but only exists as a *commented-out* struct; `mycelium-epoch.cpp` forward-declares three `extern "C"` functions that are never defined anywhere. These files cannot compile standalone, let alone link against the substrate.
   - `mycelium/`'s Python layer (17 `core/` modules + orchestrator) is **fully implemented and fully reachable**, not merely "more behaviorally rich" — it is the single most complete, tested, and mature part of the entire codebase. The gap is not maturity, it's connection: its only real leverage over actual token generation is (a) directly bypassing `llama-sampler.cpp` with its own Python sampler, and (b) a handful of bridge calls that mostly land on native stubs.

---

## 13. Open architectural question — needs your decision before Iteration 1

Per the master prompt's own rule ("если обнаружится архитектурная проблема — остановись, опиши, предложи варианты, дождись решения"), one finding blocks the prescribed iteration order and needs a decision:

**The problem:** The master prompt's Iteration 1 ("Runtime") targets `llama.cpp/src/llama.cpp`, `llama-context.cpp/.h`, `include/llama.h` with the goal "Runtime должен стать полноценной частью жизненного цикла модели." But two more foundational blockers sit underneath that goal, and neither is really "Runtime decision logic" work:

1. **No build wiring.** `LLAMA_MYCELIUM_ENABLE` is never defined by any buildable target for `llama.cpp` itself, because `llama.cpp/` has no CMakeLists.txt at all in this checkout. Nothing you write in `llama-context.cpp` can be verified to compile without this.
2. **No init call.** Even with the macro defined, `llama_mycelium_init()`/`llama_mycelium_runtime_ext_init()` need a real call site — most naturally in `llama_init_from_model()`, paired with the existing `llama_free()` teardown call — before the hook stops being a guaranteed no-op.

**Options:**

- **A. Treat both as part of Iteration 1 "Runtime."** They touch exactly the files the master prompt already scoped for Iteration 1 (`llama-context.cpp/.h`, plus minimal CMake changes), and "Runtime becomes part of the context lifecycle" is arguably incomplete without them. Risk: adding a `CMakeLists.txt` for `llama.cpp/` is itself a nontrivial, somewhat separate concern (build-system work, not runtime-logic work), and this checkout is missing ~15 headers `llama-context.cpp` needs regardless (§0) — so a real compile still wouldn't succeed even after this fix, only the diff would become "correct in principle."
- **B. Split it into its own micro-iteration before Runtime** ("Iteration 0.5: make the hook reachable") — smallest possible diff: add the init call pairing to `llama_init_from_model`/`llama_free`, and a minimal CMake target that at least defines the macro, without attempting a full build restoration. Keeps Iteration 1 focused purely on what Runtime *does* once alive.
- **C. Defer build-system work entirely** and treat Iteration 1 as a logic-only exercise (add the init call, reason about correctness by inspection) while explicitly flagging that "code compiles" as an exit criterion cannot be satisfied until you decide how to handle the missing-headers problem (§0) separately.

I have not started any of these. Awaiting your decision on which option (or an alternative) to proceed with as the next iteration.
