# Block 6 — Slice: the model cut into pseudo-experts

## 1. The gap this closes

Corridors were abstract. They carried heat, sat on the Hilbert curve, condensed into
dissipative structures — and were bound to **nothing in the actual model**. Verified by
grep: no corridor↔layer binding existed anywhere in the substrate.

The consequence was severe and had gone unstated: **the only projection out of the swarm
went to MoE experts, so on a dense model the entire organism decided nothing at all.**
Most models are dense. The physics ran and then had nowhere to land.

Worse, the HeatField gave every corridor the *same* thermal background (a global layer
average), because no better referent existed. So every corridor was equally warm, and the
swarm had nothing to differentiate on — assembly weights were near-uniform by
construction, and dissipative structures could not form for any reason grounded in the
model.

## 2. The slicing

`myc-slice.cpp` — each pseudo-expert owns a contiguous band of layers:

```
layers  0..3  -> corridor 0
layers  4..7  -> corridor 1
...
```

Contiguous because transformer layers have strong sequential locality: adjacent layers
share working set and activation statistics, so a contiguous band is the cheapest cut
that keeps a pseudo-expert internally coherent. Band sizes differ by at most 1
(remainder spread over the first corridors). Never more corridors than layers — a
corridor owning nothing is exactly the state this file exists to eliminate.

## 3. The loop that now closes for dense models

```
layer access (cb, Block 5)
    -> layer heat
        -> corridor heat        [band mean, NOT the global average]
            -> swarm assembly   [heat x neighbourhood coherence]
                -> dissipative structures
                    -> layer residency class    <-- the decision
```

`myc_slice_apply_residency()` classifies each band HOT/WARM/COLD from its pseudo-expert's
assembly weight, **relative to the ensemble mean** rather than an absolute number — when
everything is hot, being hot is not distinguishing.

Two protections that matter:
- a corridor bound into a dissipative structure is never let go COLD — the structure is the thing doing the organising work
- explicit `lock_weights` / `pinned` from the control plane is never overridden
- residency is structural, so it only changes when `boundary.frozen` is false — epoch consistency, without flooding the pending queue every step

No tensor shape, no graph topology, no expert count is touched.

## 4. Guardian latch bug fixed

I flagged this myself at the end of Block 5: emergency detection used
`guardian.total_vetos > 0`, a **cumulative** counter, so the system latched into
permanent emergency after the first veto ever recorded. Now it reads `boundary.veto`,
which the guardian refreshes every step.

## 5. Files

New: `substrate/src/myc-slice.cpp` (~190).
Changed: `myc-fields.h` (`llama_myc_slice`), `myc-internal.h` (field + decls),
`myc-heat.cpp` (band heat replaces global average — the substantive change),
`myc-organism.cpp` (slice init + step), `myc-epoch.cpp` (latch fix),
`llama-context.cpp` (`slice_ensure` at construction),
`llama-mycelium-runtime.h` (2 new API functions), `CMakeLists.txt`.

Substrate is now 3772 lines across 18 files.

## 6. Verified

- every `LLAMA_MYC_*` / `MYC_*` identifier resolves (`MYC_EPOCH_MIN/MAX_LEN` are local to `myc-epoch.cpp`)
- `lock_weights`, `pinned`, `hotness_ema` confirmed present on `llama_myc_layer_meta` before relying on them
- every mycelium call from `llama.cpp/src` and `llama.cpp/common` is declared
- no duplicate symbols; every `org.*` access resolves against the organism

## 7. Open

- **`j_attn` and `j_behavior` remain 0.** `ggml_flash_attn_ext` fuses the attention distribution away. This is the last signal the mathematics specifies that the fork cannot produce, and it is the highest-weight term in `J_total`. §R1 decision is still yours — kernel change, explicit-softmax-only, or accept the gap.
- The slicing is uniform contiguous bands. The mathematics has MGE proposing branch/merge/prune on topology; that is your slow loop and your math, so the slicing here is deliberately fixed rather than adaptive.
- `substrate/tests/*.cpp` still dead against the Block 1 ownership model.
- Still no `CMakeLists.txt` for `llama.cpp`; nothing compile-verified.
