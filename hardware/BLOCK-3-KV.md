# Block 3 — KV (complete)

## 1. The gap this block closed

The control plane submitted residency plans; the KV cache never read them.

```
Python: submit_residency_plan(pinned=[0,1,2,...])
   -> mcb.residency.pinned_positions      stored
   -> epoch_end -> impl->active.pinned_positions   committed
   -> ...                                  read by NOTHING
```

Verified before touching anything: `grep -rn "pinned_positions|residency" llama.cpp/src/` returned **nothing**. Attention sinks and anchor tokens were declared protected and then displaced like any other cell. Likewise `llama_mycelium_kv_flush_eviction_queue()` had no caller anywhere — evictions were queued and never executed.

Both now work.

## 2. Direction of control inverted

The cache used to decide policy for itself: strictness derived from `n_swa > 0`, threshold a hardcoded `0.45f`, and (from the previous block) an inline guardrail poke. That has been replaced by a real contract.

**New:** `llama_myc_kv_policy` — resolved once per `find_slot()` by the Runtime, executed by the cache:

| Field | Decided by |
|---|---|
| `strict` | KV pressure vs guardian threshold, guardian veto, or emergency phase |
| `heat_threshold` | pressure-driven band `[0.25, 0.75]` (the old `0.45f` is now its midpoint) |
| `pinned` / `n_pinned` | P1 **committed** active set, not the pending plan — epoch consistency |
| `n_sink_protect` | measured sink count |

`llama_kv_cache::find_slot()` now:
- refuses to displace pinned positions and sink-range cells,
- uses the Runtime's threshold instead of a constant,
- reports back what it saw.

**New:** `llama_mycelium_kv_report_cells(n_examined, n_protected, mean_heat)` — the per-cell measurement the Runtime's KV term never had. Before this the substrate only ever saw per-context aggregates and inferred cell behaviour from occupancy.

## 3. Resolved: two writers for one field

`kv.pressure` was written by `myc_kv_thermo_update()` from occupancy *and* would now be written from real contention. Occupancy was only ever a proxy for pressure, and two writers on one field is exactly how the two heat models drifted apart in the first place. Ownership is now explicit: **`report_cells` owns `pressure`** (real contention), `thermo_update` owns fragmentation/dissipation and only falls back to occupancy when no cache ever reported.

## 4. Eviction queue now executes

`myc_epoch_commit()` gained a third phase. The boundary is the only point where flushing is safe (the queue must not run mid-decode), and it is now used:

```
epoch boundary:
  1. myc_graph_commit(...)                 structure
  2. myc_epoch_pending_flush(...)          queued discrete changes
  3. llama_mycelium_kv_flush_eviction_queue(...)   NEW - residency actually carried out
```

## 5. Deleted

| Deleted | Where | Why |
|---|---|---|
| hand-mirrored `llama_mycelium_veto_reason` enum + externs | `llama-kv-cache.cpp` | replaced by the real header |
| hand-mirrored enum + 2 externs | `llama-graph.cpp` | same |
| 5 hand-mirrored externs | `llama-context.cpp` | same |
| inline guardrail poke in `find_slot` | `llama-kv-cache.cpp` | replaced by the policy contract |
| `mcb->kv_stats.n_sink_cells` write | `myc-kv.cpp` | field does not exist; state belongs on the thermo struct |

Mirroring a C ABI by hand across four translation units was a standing correctness hazard, not a decoupling win — one header, one declaration site now. Cost: `llama.cpp` needs `substrate/include` on its include path, which linking against `llama_mycelium` (PUBLIC include dir) already provides.

## 6. Your math untouched

`llama_kv_heat`, `llama_kv_heat_value`, `heat_score/touch/decay` and `llama_kv_compact_locality_hash` are **unchanged**. The block changes *who decides the policy* those scores feed, not the scoring itself. `heat_score()` is still the cache's own function computing the same weighted sum over hotness / reuse / semantic affinity / corridor affinity / migration cost.

## 7. Bugs caught during the block

Found by validating every identifier and every `mcb->x.y` field access against the real headers:

- `mcb->kv_stats.n_sink_cells` — field doesn't exist (it's on `llama_myc_kv_thermo`)
- `mcb->kv_stats.n_kv_cells` — real name `kv_size`
- `mcb->epoch.steps_elapsed` / `.length` — real names `step_in_epoch` / `epoch_length`

Final sweep: every `LLAMA_MYC_*` identifier and every `mcb->` field access in the substrate resolves against the headers; every mycelium function called from `llama.cpp/src` is declared.

## 8. Open

- `myc_epoch_on_step` only fires when the phase is ACTIVE/EMERGENCY and `epoch_length > 0`. If the control plane never calls `epoch_begin`, the phase stays IDLE and neither the graph nor the eviction queue ever commits. **Native epoch initiation does not exist** — that is the next block's first task.
- The `LLAMA_MYCELIUM_INTERNAL` path in `kv_flush_eviction_queue` still guards the real `llama_memory_seq_rm` calls; without that macro the flush counts entries without freeing cells.
- `substrate/tests/*.cpp` remain dead against the new ownership (unchanged from Block 2).
- Still no `CMakeLists.txt` for `llama.cpp`; nothing compile-verified.
