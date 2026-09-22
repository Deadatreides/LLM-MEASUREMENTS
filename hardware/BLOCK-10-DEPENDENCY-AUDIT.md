# Block 10 — dependency audit of the organism step

Nine blocks of changes had accumulated, each verified only against itself. This block
audited the assembled system: for every field, who writes it, who reads it, and in what
order within one step.

It found three real defects. All three were mine, and all three were silent — nothing
crashes, the numbers just quietly mean the wrong thing.

## 1. `runtime.kv.pressure` was frozen after the first step

```cpp
if (org.runtime.kv.n_evict_pending == 0 && th.pressure == 0.f && used > 0.f) {
    th.pressure = used;
}
```

The guard `th.pressure == 0.f` is true exactly once. The field was set on step one and
then **never updated again for the lifetime of the context**.

The guardian's `KV_CONFLICT` veto reads that field. So the KV pressure veto has been
firing — or not firing — on a value measured once, at startup, when the cache was empty.

I introduced this in Block 4 while removing a double-writer: I stopped `kv_thermo_update`
from clobbering pressure, added a fallback "so pressure is not silently stuck at zero",
and the fallback was itself the thing that stuck it.

**Fixed:** written unconditionally every step as `0.6·occupancy + 0.4·contention`,
smoothed. Deliberately *not* the organism's composite `thermo.pressure` — a `KV_CONFLICT`
veto should fire on a KV problem, not on system-wide load.

## 2. Pressure read `migration_instability` from the previous step

Inside `myc_thermo_step`, `th.pressure = myc_thermo_pressure(org)` ran **before** the line
that assigns `th.migration_instability` — which `myc_thermo_pressure` reads. Every
pressure value was computed with one input a step stale.

**Fixed** by ordering: `collapse_rate` and `migration_instability` are now computed first,
then pressure.

## 3. `heat.layers[].state` had two writers fighting

- `myc_heat_touch_layer` auto-classified HOT/WARM/COLD from raw `hotness_ema`
- `myc_slice_apply_residency` classified from swarm assembly weight

Slice writes at the end of a step (position 17); heat overwrites at the start of the next
(position 8). **The swarm's residency verdict survived less than one step and never
reached anything.** Block 6 shipped believing that loop was closed. It was not.

**Fixed by ownership, not by ordering:** heat no longer classifies at all. Heat is a
measurement; residency is a decision, and it belongs to the swarm via the band the model
was sliced into — a pseudo-expert's assembly weight, not one layer's raw temperature. If
the model was never sliced, state stays UNKNOWN, which is the honest answer.

**And the fix exposed a fourth defect,** which is why removing code is dangerous without
tracing readers: `myc_heat_step` decayed only layers whose `state != UNKNOWN`. With heat
no longer classifying, nothing would ever have been UNKNOWN-free before the swarm ran —
so **nothing would ever have been decayed**, and every layer the graph touched would have
climbed to 1.0 and stayed pinned there. Which layers exist is a property of the *slice*,
not of their residency class, so the loop now keys off `slice.n_layer`.

## 4. One genuine cycle, now documented instead of hidden

```
thermo.pressure → boundary.rigidity → boundary.veto → guardian → V(R) → thermo.pressure
```

This is a real feedback loop, and in a system with feedback that is expected — but one
edge must carry a delay or the step cannot be ordered at all. `boundary.rigidity` is that
edge and is read one step old.

It is the right edge to delay: boundary conditions are *by definition* what was fixed at
the last commit, so reading the previous value is what the physics means, not a
compromise. Now stated explicitly at the read site rather than being an accident of
ordering that happened to work.

## 5. Systematic check, not just spot checks

Wrote a static sweep over all 20 subsystem files that extracts every field assignment and
groups by field:

```
no multi-writer fields detected
```

Every field in the organism now has exactly one writing file — the rule stated in
`myc-internal.h` since Block 2, and violated three times since without detection because
nothing was checking it. This sweep is cheap to re-run and should be, after any block
that adds a writer.

## 6. Files

`myc-kv.cpp` (unconditional pressure), `myc-thermo.cpp` (ordering, cycle documented),
`myc-heat.cpp` (classification removed, decay keyed off the slice).

Substrate: 4595 lines.

## 7. Verified

- multi-writer sweep clean across all subsystem files
- every identifier resolves; every `org.*` access resolves; every llama.cpp call declared
- still no compiler in this environment — nothing compile-verified

## 8. Open

- Under flash attention `j_attn` is 0 and `J_total` loses its dominant term. §R1 kernel change is the only fix and remains your call.
- `llama.cpp` still has no build system; only the substrate is buildable, standalone.
- The step has a second implicit lag I did *not* change: `j_behavior` lands in the next step's `J_total`, documented in Block 9 as correct (drift describes a completed transition). Flagging it here so both lag edges are recorded in one place.
