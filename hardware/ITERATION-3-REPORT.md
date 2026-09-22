# Iteration 3 — KV Cache: extend heat_strict to non-SWA models under Runtime pressure

## 1. What code changed

- `llama.cpp/src/llama-kv-cache.h`: +1 member, `struct llama_context * lctx_cached = nullptr;`
- `llama.cpp/src/llama-kv-cache.cpp`:
  - `init_update(llama_context * lctx, bool optimize)`: +1 line, `lctx_cached = lctx;`
  - `find_slot(...)`: `heat_strict` (previously `!cont && n_swa > 0`, always false for non-SWA models) now also becomes `true` when `lctx_cached` is set and `llama_mycelium_guardrail_check(lctx_cached) != LLAMA_MYC_VETO_NONE`.
  - +1 local `#ifdef LLAMA_MYCELIUM_ENABLE` block (enum + extern decl for `llama_mycelium_guardrail_check`), same minimal-coupling pattern as the other three files touched so far.

All additive. No existing line removed. `heat_score`/`heat_touch`/`heat_decay`/the retry-on-exhaustion fallback are unchanged and consume the new `heat_strict` value exactly as they already consumed the SWA-only one.

## 2. Why this implementation

`init_update(llama_context*, bool)` already receives the context pointer every decode cycle, before `find_slot()` runs later in the same cycle (confirmed: `llama_context::memory_update()` → `memory->init_update(this, ...)` runs before `memory->init_batch()` → `prepare()` → `find_slot()`). Caching it avoids threading a new parameter through `find_slot()`/`prepare()`'s signatures (2+ call sites, one of them in a state-deserialization path with no context available at all). Smallest available option.

`llama_mycelium_guardrail_check()` already exists, already covers KV pressure (`LLAMA_MYC_VETO_KV_CONFLICT`) among its checks, already returns 0/`NONE` safely when no runtime state is attached. No new Runtime accessor was added — reused as-is.

## 3. Temporary assumptions

- Using the full guardrail result (any veto, not just `KV_CONFLICT`) as the trigger, rather than adding a second call to isolate just the KV-pressure check. Coarser than ideal; cheap to narrow later if wanted.
- `lctx_cached` is a raw pointer with no lifetime enforcement beyond "set every decode cycle before use." Matches how `llama_kv_cache_context` already stores the same pointer transiently elsewhere in this file — not a new pattern.

## 4. Open questions

None requiring a decision. Proposed next: Speculative decoding (`common/speculative.cpp`) — `n_draft` is hardcoded to 8 and nothing reads the substrate's spec-window state Python already writes; smallest fix is likely reading `llama_mycelium_spec_should_run(ctx)` (already implemented, currently uncalled) where `n_draft` is chosen.
