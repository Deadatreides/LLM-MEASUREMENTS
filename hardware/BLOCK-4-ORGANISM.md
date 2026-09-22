# Block 4 — Organism as a physical system

~2100 lines written this pass. Substrate is now 2645 lines across 17 files.

## 1. What was wrong, and what changed

Every one of your seven objections was a real defect. Each is fixed structurally, not by renaming.

| Objection | Fix |
|---|---|
| Runtime was made the boss again | Runtime is now one organ among six global fields. It owns no policy and commands nothing. |
| KVPolicy was procedural | Deleted. Replaced by `llama_myc_observable` — a **state** carrying no instruction. |
| Thinking in RPC calls | The cache now *observes* fields and *publishes* samples into them. No subsystem calls another to make it act. |
| Heat belonged to KV | HeatField is global. The weights that were hardcoded inside `llama_kv_heat_value()` are now field parameters, and the cache samples the field instead of owning a private model. |
| pressure = contention only | `pressure = f(occupancy, fragmentation, contention, migration, hilbert transport, rigidity)` — six terms, weights summing to 1. |
| pinned was a list | It's a **boundary condition** now, in `myc-boundary.cpp`, next to Epoch and Guardian, with `rigidity` as its scalar summary feeding pressure. |
| Runtime knew too much (God Object) | Field ownership is declared in `myc-internal.h`: exactly one writer per field. |

## 2. The physics is now actually implemented

Your description — *модель нарезается на псевдоэкспертов; псевдоэксперты собираются весами по теплоте; отбор близости по многомерной кривой Гильберта; собираются в термодинамический рой, образуя диссипативные структуры* — had no representation in the code. It does now.

**`myc-hilbert.cpp` (181 lines)** — a real multidimensional Hilbert curve, not "Hilbert-inspired". Full Skilling transform, 4 phase-space axes (heat / usage / KV locality / flow), 8 bits per axis, exact and allocation-free. Proximity is a k-nearest query along the curve, because the curve preserves locality. `C_hilbert = Σ w_ij |π(i) − π(j)|` is computed as the transport cost.

**`myc-swarm.cpp` (223 lines)** — the assembly step, which is the heart of the system:
- `assembly_weight[i] = heat[i] × (0.5 + 0.5 × neighbourhood_support)` — a hot pseudo-expert inside a coherent neighbourhood outweighs an equally hot isolated one. That is what makes it an *ensemble* rather than a ranking.
- local entropy production per pseudo-expert (divergence from its neighbourhood — irreversible work)
- **dissipative structures**: flood fill over the Hilbert neighbourhood graph, admitting only pseudo-experts above the condensation threshold. Connected hot+coherent regions of the curve condense into structures. Explicit stack, no recursion, no allocation.
- `structure_order`: 0 = dust, 1 = one rigid block — the band the mathematics wants held open.

**`myc-thermo.cpp` (233 lines)** — free energy `F = −β⁻¹ ln Z` over the spin ensemble (mean-field, per-site `Z_i = 2cosh(βh_i)`, exact for non-interacting and controlled otherwise), entropy, composite pressure, Prigogine balance, order parameter, `V(R)`.

**Energy projection now consumes swarm assembly weights**, not raw corridor activity. A pseudo-expert's pull on physical execution is its weight in the swarm.

## 3. Your math untouched

`llama_kv_heat`, `heat_score/touch/decay`, `llama_kv_compact_locality_hash` in `llama-kv-cache.cpp` — unchanged. The five heat weights are initialised to exactly `0.35 / 0.30 / 0.20 / 0.15 / −0.25`, so arithmetic at startup is identical; they moved into the field so there is one copy instead of two. The old hardcoded `0.45f` threshold is now `heat.threshold_base`, same value.

## 4. Files

New: `myc-fields.h` (254), `myc-hilbert.cpp` (181), `myc-swarm.cpp` (223), `myc-thermo.cpp` (233), `myc-boundary.cpp` (67).
Rewritten: `myc-internal.h`, `myc-heat.cpp`, `myc-organism.cpp`, `myc-kv.cpp`, `myc-energy.cpp`.
Deleted: `llama_myc_kv_policy` and its two API functions; `myc_kv_resolve_policy`; `myc_kv_report_cells`.
Changed in llama.cpp: `llama-kv-cache.cpp` — observes state and interprets it locally.

## 5. Ownership is now enforceable

`myc-internal.h` declares one writer per field. Two writers on one field is how the two heat models drifted apart originally, so it's stated as a rule rather than a convention.

## 6. Verified

- no duplicate symbols across 17 files
- every `LLAMA_MYC_*` / `MYC_*` identifier used resolves against a header
- every mycelium function called from `llama.cpp/src` is declared
- every `org.*` field access resolves against the organism definition
- caught during writing: `pressure_occupancy` and `thermo_migration_hint` (invented), `llama_myc_observable` unreachable from llama.cpp behind a forward declaration (moved to the public header)

## 7. Open

- **`myc_epoch_on_step` still requires the control plane to call `epoch_begin`.** Phase stays IDLE otherwise, and neither graph couplings nor evictions ever commit. Native epoch initiation does not exist — first task of the next block.
- `j_attn` and `j_behavior` remain 0: no measurement source. `ggml_flash_attn_ext` fuses the attention distribution away (§R1 decision still yours).
- `substrate/tests/*.cpp` remain dead against the new ownership.
- No `CMakeLists.txt` for `llama.cpp`; nothing compile-verified — checked by reading and mechanical cross-checks only.
