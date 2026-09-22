# Block 9 — three ownership bugs I introduced, and the last negentropy term

## 1. Bugs found in my own work

I set out to add `j_behavior` and first checked whether Block 8 had created a second
writer on `mcb->runtime.h_attn`. It had — and looking properly turned up **three**
collisions, all introduced by me in Block 5 when I added a native epoch cycle without
checking that P1 already had one.

| Field | Writer A | Writer B | Consequence |
|---|---|---|---|
| `mcb->runtime.h_attn` | P1 `update_runtime_state` — entropy of the **output logits** | Block 8 `publish_attn_entropy` — entropy of the **attention distribution** | Two different quantities in one field. Whichever ran last won, and the guardian read the survivor. |
| `mcb->epoch.step_in_epoch` | P1 `update_runtime_state` | `myc_epoch_on_step` | **Double-incremented every step: epochs ran at 2× speed and committed early.** |
| epoch end / emergency | P1 (`epoch_end`, `set_emergency` at hardcoded `v_lyapunov < 0.3`) | `myc_epoch_on_step` (commit + adaptive length + V(R)) | Two competing state machines on one epoch. |

This is exactly the "two writers on one field is a bug" rule I wrote into
`myc-internal.h` — and I violated it four blocks later without noticing, because I added
the new machinery and never re-read the old.

**Resolved by ownership, not by patching:** P1's `update_runtime_state` now owns exactly
two fields — `drift` and `curvature`, both genuinely logit-derived and both needed by the
guardian's variance proxy. It no longer touches `h_attn`, no longer advances the epoch,
no longer ends it, no longer triggers emergency. Verified: `step_in_epoch++` now appears
in exactly one place.

The logit-entropy value is still computed — it feeds drift — but into P1's own local
history (`h_attn_prev`) rather than the shared field. The name is now the only misleading
thing left about it, and renaming a Python-visible struct member is not worth the ABI
churn.

## 2. j_behavior — the last zero term

`J_total`'s five components are now all measured. The mathematics defines behaviour as:

```
B(t) = [b₁ … bₙ]
D_B  = ‖B(t) − B(t−Δt)‖        drift
I_s  = ∫|dB/dt| dt              semantic inertia
"If D_B grows, Guardian reduces aggressiveness."
```

`myc-behavior.cpp` assembles B from **eight quantities the organism already measures** —
attention entropy, logit curvature, output drift, KV pressure, structure order, order
parameter η, speculative acceptance, graph connectivity. Nothing is invented; this file
only assembles the named vector and differentiates it.

Two details that matter:
- **Normalisation without a magic constant.** The largest possible step is √dims (every component swinging 0→1 at once), so dividing by it puts D_B in [0,1] by construction.
- **First sample reports nothing.** With no previous B, drift is *undefined*, not zero. The vector is seeded and the step returns — rather than publishing a 0 that was never measured. Same discipline as `j_attn` under flash attention.

`j_behavior = 1 − D_B`: steady behaviour is organised behaviour.

## 3. The guardian now acts on drift

"If D_B grows, Guardian reduces aggressiveness" is implemented as a *tightening threshold*
rather than a new check — at D_B = 1 the catastrophe threshold halves. That is what
reducing aggressiveness means here, and it avoids adding a fifth veto path for a signal
that modulates an existing one.

## 4. Files

New: `substrate/src/myc-behavior.cpp` (~95).
Changed: `llama-mycelium.cpp` (ownership stripped — the substantive fix), `myc-fields.h`
(`llama_myc_behavior`), `myc-internal.h` (field, decls, ownership table),
`myc-organism.cpp` (init + step placement), `myc-guardian.cpp` (drift tightens the
threshold), `myc-thermo.cpp` (observable), `llama-mycelium-runtime.h`, `CMakeLists.txt`.

## 5. Step ordering, and why

`myc_behavior_step` runs **after** thermodynamics (B samples η) and **before** the
guardian (which must see this step's drift). Its `j_behavior` therefore lands in the
*next* step's `J_total` — which is correct, not a lag bug: drift is a property of the
transition just completed, not of the state currently being assembled.

## 6. Verified

- `step_in_epoch++` now has exactly one writer
- no duplicate function symbols (the two reported are type names)
- every `org.*` access resolves; every identifier resolves against a header
- still no compiler in this environment — nothing compile-verified

## 7. Open

- Under flash attention `j_attn` is 0, so `J_total` still loses its dominant term there. The §R1 kernel change remains the only fix.
- `llama.cpp` still has no build system; only the substrate is buildable, standalone.
- `mcb->runtime.h_attn` is now attention entropy while the P1 field it sits next to (`drift`) is logit-derived. Consistent in ownership, mildly confusing in naming. Left alone deliberately: the struct is mirrored by the Python bridge and renaming it is ABI churn for cosmetics.
