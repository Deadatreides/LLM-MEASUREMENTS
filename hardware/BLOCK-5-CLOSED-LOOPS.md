# Block 5 — closing the open control loops

Four loops were open after Block 4. All four are closed. The organism now runs without
a control plane attached.

## 1. Epoch: native autonomy

**Was:** phase stayed `IDLE` forever unless Python called `epoch_begin`. Graph couplings
never committed, the eviction queue never flushed, the entire feedback path was inert on
a native-only build.

**Now** (`myc-epoch.cpp`): the organism runs its own epoch cycle.

- **Adaptive length.** `stability = 0.5·J_total + 0.3·structure_order − 0.3·pressure − 0.2·collapse_rate`, mapped onto `[8, 256]` steps. A stable, well-ordered system holds its structure longer; one losing coherence commits corrections sooner.
- **Self-initiation.** From `IDLE`, the organism begins its own epoch. External control still wins — `llama_mycelium_epoch_begin()` sets the phase and this code carries it — but the absence of a control plane no longer means the machinery never runs.
- **Emergency.** When `V(R)` is critical with vetoes in force, the current epoch commits immediately and drops to `EMERGENCY`, which runs at minimum length: recover, then re-plan.

## 2. Guardian: actually runs

**Was:** `myc_guardian_check` was called from exactly one place — `llm_graph_context`'s
constructor, incidentally, per graph build. `boundary.rigidity` reads `mcb->veto.reason`,
so the boundary never saw a veto on any step where no graph was built.

**Now:** the Guardian runs in the organism step, positioned *before* `myc_boundary_step`
because the boundary's rigidity depends on its verdict. Ordering is a dependency, not a
preference.

## 3. Layer heat: a real access signal

**Was:** `myc_heat_touch_layer` was only ever called with `access_weight = 0`. Layer
hotness could only decay toward zero — it carried no information.

**Now:** `llm_graph_context::cb()` reports layer access into the HeatField. That callback
fires for every named tensor of every layer, so it is the one place that knows which
layers actually execute. Filtered to `"kqv_out"` — emitted exactly once per attention
block — so a layer counts once per build, not once per op.

I checked the tensor name against the codebase rather than assuming: my first attempt
used `"attn_norm"`, which appears **zero** times here, and would have silently never
fired.

## 4. Speculative window: connected

**Was:** `llama_mycelium_spec_should_run()` fully implemented, called from nowhere.
`n_draft` hardcoded to `8` with a `// TODO get from config?`.

**Now:** `common_speculative_draft()` consults the organism through the **existing**
`n_max` constraint. Only ever tightens — the organism may shorten or veto a draft, never
lengthen one past what the caller allowed. `k == 0` drops that sequence to the classic
path for the step.

## 5. Deleted

| Deleted | Why |
|---|---|
| `llama_exec_corridors[16]` + `llama_exec_corridor_id` + `llama_exec_corridor_touch` | Written on every named tensor of every graph build, read by nothing, anywhere. A hash of a tensor name into 16 buckets with no consumer and no relation to the organism's corridors. Dead since before this project began; flagged in Iteration 0; now genuinely replaced by the HeatField feed at the same call site. |
| `llm_graph_context::mycelium_veto` | Computed once per graph build, consumed by nothing. Flagged in `MYCELIUM-TARGET-ARCHITECTURE.md` §Part 4. The Guardian now runs where it belongs, so the transported copy has no reason to exist. |
| hand-mirrored externs in `common/speculative.cpp` | Replaced by the single `llama-mycelium-runtime.h` include, matching the other three call sites. |

## 6. Files changed

`substrate/src/myc-epoch.cpp` (native epoch cycle, adaptive length, emergency),
`substrate/src/myc-organism.cpp` (guardian in the step),
`llama.cpp/src/llama-graph.cpp` (heat feed, dead array deleted, dead veto deleted),
`llama.cpp/src/llama-graph.h` (dead veto member deleted),
`llama.cpp/common/speculative.cpp` (draft window, include cleanup).

## 7. Verified

- every `LLAMA_MYC_*` / `MYC_*` identifier resolves (the two flagged are `MYC_EPOCH_MIN/MAX_LEN`, defined locally in `myc-epoch.cpp` — correct)
- every mycelium function called from `llama.cpp/src` **and** `llama.cpp/common` is declared
- `<cstring>` already included where `strcmp` is now used
- `spec->ctx_tgt` confirmed present (`speculative.cpp:1123`) before relying on it
- `llama_exec_corridor` and `mycelium_veto` have zero remaining references

## 8. Open

- `j_attn` and `j_behavior` still 0 — no measurement source. `ggml_flash_attn_ext` fuses the attention distribution away. §R1 decision remains yours; it is the last major signal the mathematics specifies that the fork cannot currently produce.
- `substrate/tests/*.cpp` still dead against the ownership model from Block 1.
- Still no `CMakeLists.txt` for `llama.cpp`; nothing compile-verified. Checked by reading and mechanical cross-checks only.
- Emergency detection uses `guardian.total_vetos > 0` as the "vetoes in force" condition, which is cumulative rather than current. It will latch after the first veto ever recorded. Flagging rather than leaving it silent — a per-step veto flag would be the honest fix, and it belongs with whatever block next touches the Guardian.
