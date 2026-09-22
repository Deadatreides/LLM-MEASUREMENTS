# llama.cpp under Mycelium: target architecture

No code. No patches. This is the map, derived from the mathematics — not from llama.cpp's existing shape.

---

## Part 0 — What this system is

Restating the математика in my own words, so the design below is verifiable against it rather than against convention.

This is **not** an inference engine with a controller bolted on. It is an **open non-equilibrium dissipative system** whose trajectory `x(t) = [R,E,K,F,G,B,S,M,C]` lives in execution-state space. The model is not the system; the model is the medium the system dissipates through. What is being managed is the *trajectory*, not the answer.

The governing quantities, and what each one demands:

- **Entropy is measured, negentropy governs.** `J_total = w_a·J_attn + w_k·J_kv + w_g·J_graph + w_b·J_behavior + w_s·J_spec`. Five *independently measured* negentropies. Not one scalar proxy.
- **Prigogine balance.** `dS/dt = d_eS/dt + d_iS/dt`, `χ = (Φ_in − Φ_diss)/Φ_diss`. Self-organization only when `χ > χ_c`. So the runtime must actually *account* for incoming flux (tokens, memory retrieval, feedback) versus dissipation (routing, flow control, KV regulation, verification). These are two distinct ledgers, both of which the compute layer must report honestly.
- **Order parameter.** `τ_η·η̇ = aη − bη³ + ξ`. A pitchfork bifurcation. `η` is emergent corridor coherence, and the target regime is `0 < η < 1` — edge of chaos. `η→1` is rigidity, `η→0` is noise. This means the system is *supposed* to sit in a metastable band, not converge.
- **Lyapunov.** `V(R) = αB_pressure + βQ_stall + γM_migration + δE_thrashing − ρ·J_total`, require `V̇ < 0`, emergency at `V > V_crit`.
- **Epoch consistency.** `L_active(t) = L_snapshot ∀t ∈ epoch`. Topology, routing, and corridor masks are **frozen inside an epoch**. Only boundary commits change structure. This is the single most architecturally consequential statement in the whole specification, and I will return to it repeatedly.
- **Flow relaxation.** `τ·J̇_i = −(J_i − J*_i) + ξ_i`. `j_star` is not a value to copy — it is the *fixed point of an ODE the runtime integrates*.
- **Graph control.** `L = D − A`, `C_graph = uᵀLu`, percolation `p_conn > p_c` or fragmentation → collapse.
- **Spin glass.** `H(s) = −Σ_{i<j}J_ij·s_i·s_j − Σ_i h_i·s_i`. Corridor states are **spins**. `J_ij` are corridor↔corridor couplings. `h_i` are local fields from measured execution state. Metastable reasoning states are the attractors of this Hamiltonian.
- **Pseudo-MoE over a dense base.** `y(x) = Σp_i(x)E_i(x)`, where "`E_i` — не отдельные модели, а virtual corridors над одной dense-основой." A dense anchor corridor must *always* exist.
- **Emergence criterion.** `J_total↑ ⟹ η↑`, held in `0 < η < 1`.

### The consequence that reorganizes everything

Pseudo-experts are corridors over a **dense** base. They are therefore **not** physical FFN experts, and the influence path cannot primarily be "corridor → MoE expert." Most models are dense and have no MoE experts at all. A corridor is a region of execution phase-space; its influence must express itself through whatever execution knobs exist:

- which layers execute, and how (corridor width / layer masks) — *the primary channel, available in every model*
- KV residency and cell ordering
- speculative depth `k_draft`
- sampling modulation
- **and, only in MoE models, expert routing bias** — a secondary specialization, not the mechanism

This is the correction that matters most against what is currently built. I will state it plainly in Part 4.

---

## Part 1 — Canonical data flow

Thinking in flow, not files:

```
Token / batch arrives
    ↓  [Φ_in accounting]
Context (owner of Runtime state)
    ↓
Runtime: solve state
    ├─ spin-glass relaxation over corridors:  s ← relax(H; J_ij, h_i, β)
    ├─ flow relaxation:                       τ·J̇_i = −(J_i − J*_i) + ξ_i
    ├─ order parameter:                       τ_η·η̇ = aη − bη³ + ξ
    ├─ negentropy assembly:                   J_total = Σ w_x·J_x
    ├─ Lyapunov:                              V(R), emergency if V > V_crit
    └─ epoch gate:                            inside epoch → structure FROZEN
    ↓
Interaction Graph  (A, L = D−A, C_graph = uᵀLu, percolation p_conn)
    ↓
Projection onto execution knobs
    ├─ corridor width / layer masks      ← primary, dense-model channel
    ├─ expert energy                      ← MoE models only
    ├─ KV residency plan + locality order (C_Hilbert)
    ├─ k_draft
    └─ sampling modulation
    ↓
Graph Builder  (pure executor: builds a graph for THIS policy)
    ↓
Router / Experts / Attention / KV   [Φ_diss — the actual dissipation]
    ↓
Measurement readback
    ├─ H(attn) per layer          → J_attn
    ├─ KV occupancy/fragmentation → J_kv
    ├─ graph coherence            → J_graph
    ├─ behavioral drift D_B       → J_behavior
    ├─ acceptance rate r_acc      → J_spec
    └─ selected experts / activation stats
    ↓
Feedback  (couplings J_ij updated — PENDING, committed only at epoch boundary)
    ↓
Runtime  → next step
```

Runtime is the source of state. Graph Builder is the executor. Router merely consumes already-computed energies. This matches the correction in your message — and it is what I argued in the previous document ("Runtime decides; Graph executes"), so we are aligned on direction; the disagreement was never here.

---

## Part 2 — What the mathematics demands from below

This is the part that actually answers "how must llama.cpp be rewritten." Each item is a capability the математика requires and the current compute layer cannot provide.

### R1. Attention entropy must be measurable — the hardest requirement

`J_attn` is the highest-weight term in `J_total`, and it is **not obtainable today**. Verified in this codebase:

- `llama-graph.cpp:2087` — `ggml_flash_attn_ext(...)`: softmax is **fused inside the kernel**. The attention probability matrix is never materialized. `H(attn)` is unrecoverable from this path at any price short of changing the kernel.
- `llama-graph.cpp:2147` — `ggml_soft_max_ext(ctx0, kq, ...)`: here `kq` **is** the materialized probability tensor. `H(attn)` is computable by appending a reduction over it.

So there is a real, unavoidable engineering fork, and it belongs to you, not to me:

| Option | Cost | Consequence |
|---|---|---|
| Add an entropy-reduction node on the explicit-softmax path only | ~3 ops/layer, negligible | `J_attn` available only when flash-attention is off — a real perf sacrifice |
| Extend `ggml_flash_attn_ext` to emit a per-head entropy accumulator | ggml kernel work, per backend (CPU/CUDA/Metal) | `J_attn` available always, at ~no runtime cost; touches ggml itself |
| Proxy `J_attn` from output statistics instead of true `H(p)` | cheap | violates "entropy is measured" — a proxy, which the математика explicitly moved away from |

My assessment: option 2 is the only one consistent with the mathematics as written, and it is the deepest genuine change this fork requires. Option 1 is the honest interim. Option 3 reintroduces exactly the entropy-proxy design the specification was written to replace.

### R2. Epoch semantics must become the graph lifecycle

`L_active(t) = L_snapshot ∀t ∈ epoch` is not a policy nicety — it is a hard constraint on when structure may change. And llama.cpp already has a mechanism whose semantics are *identical in shape*: `llm_graph_result::can_reuse()` / `graph_reuse_disable`.

These two concepts must be unified:

- **inside an epoch** → topology frozen → the computation graph is reusable, by construction
- **epoch boundary commit** → topology changes → graph invalidated and rebuilt

This is the one place where llama.cpp's existing design is *already* the right shape for the mathematics. It should be adopted deliberately, not worked around. Epoch boundary becomes the single authoritative graph-invalidation point.

### R3. Layer-level execution modulation must exist

Corridor width over a dense base is the primary influence channel, and it does not exist in any form today. The per-architecture builders in `llama-model.cpp` loop over `n_layer` and unconditionally emit every layer. What is required:

- a per-layer policy vector, frozen for the epoch, supplied by Runtime before graph build
- the builders consume it: a layer may be executed fully, executed with reduced width, or skipped with residual pass-through
- because it is epoch-frozen, this changes graph *topology* only at epoch boundaries — consistent with R2, and cheap

The dense anchor corridor requirement maps directly: there must always be a valid all-layers-active policy, and it must be the fallback whenever `V > V_crit`.

### R4. KV placement must be locality-ordered, not FIFO

`C_Hilbert = Σ w_ij·|π(i) − π(j)|` asks for a locality-preserving ordering over KV cells. Today `llama_kv_cache::find_slot` is a ring-buffer scan with a hash-based bucket hint. The rewrite: cell placement chosen to minimize the ordering cost under the Riemannian execution metric `g_ab` (locality, bandwidth, cache cost), with eviction driven by the Runtime residency plan rather than by a local heat model.

Note carefully: the existing `llama_kv_compact_locality_hash` is a crude, *unrelated* mechanism that happens to share vocabulary. It is not an implementation of `C_Hilbert` and should not be mistaken for one.

### R5. Both thermodynamic ledgers must be reported honestly

`χ = (Φ_in − Φ_diss)/Φ_diss` requires the compute layer to report real dissipation: actual KV hits/misses, real eviction counts, real acceptance rates, real timings. Today `llama_decode()` passes **fabricated** values into the hook — `kv_hits = kv_total` (a hardcoded 100% hit rate) and `spec_drafted = spec_accepted = 0`. Every downstream quantity computed from those — `χ`, `J_kv`, `J_spec`, and therefore `J_total`, `η`, and `V(R)` — is currently built on fiction. This is not a cosmetic defect; it silently corrupts the entire control law.

### R6. Cavity compensation needs activation statistics

`δw_rest = −H⁻¹_{rest,rest}·H_{rest,S}·δw_S` needs at minimum a Hessian-diagonal proxy — a per-layer EMA of squared activations/gradients. That requires the graph to expose per-layer activation statistics as small readable outputs. The `cb_eval` hook exists as an entry point; the statistics themselves do not. (`mycelium-cavity.cpp` implements the compensation math and must not be touched — this is purely about feeding it.)

---

## Part 3 — Target architecture, subsystem by subsystem

Derived from the flow, then mapped onto files.

### Context — owner of Runtime, orchestrator of the step

Becomes the single owner of Runtime state as a direct member. Every `extern "C"` accessor, global `ctx*`-keyed map, mutex, and `get_rt(ctx)` lookup is **deleted** — not deprecated. `decode()` is rewritten from "run the model, then notify Mycelium" into the Part 1 flow: begin step (Φ_in) → solve state → project policy → invoke compute → measure → feedback. Context owns Runtime and nothing else about it: no lookup, no registration, no initialization ceremony.

### Graph Builder — pure executor, plus instrumentation

Becomes a pure function of `(model, ubatch, RuntimePolicy)`. It makes no decisions. Two capabilities are added: consumption of the epoch-frozen layer policy (R3), and emission of measurement nodes (R1, R6). The dead `llama_exec_corridors[16]` array and its per-tensor touch call are deleted outright. The per-architecture `build_*` functions keep their structure but take policy as a parameter.

### Interaction Graph — corridor↔corridor couplings, not corridor→expert

This is the structural correction. The graph the mathematics describes is the **spin-glass coupling matrix `J_ij` between corridors**, with local fields `h_i` fed from measured execution state, plus a Laplacian `L = D − A` for the connectivity penalty `C_graph` and a percolation check `p_conn > p_c`. Expert energy is not the graph — it is one *projection* of the relaxed spin state onto one particular execution knob, and only in MoE models.

### KV Cache — Runtime-owned policy, locally-owned bookkeeping

Split by responsibility. Eviction/residency policy moves into Runtime entirely (it is `K(t)` in the state vector and contributes `J_kv`). Cell bookkeeping — cells, streams, positions, shift, defrag, state serialization — stays local; Runtime has no business owning raw memory cells. Placement is rewritten per R4. The local `llama_kv_heat` model is deleted, its role absorbed by Runtime's `K(t)`.

### Router — unchanged mechanism, Runtime-supplied input

`ggml_argsort_top_k` and the aggregation math stay exactly as they are. Only the input changes. This is deliberate and I will defend it: hard top-k over a fixed `n_expert_used` is a load-bearing tensor-shape invariant, and replacing the mechanism means rewriting the downstream aggregation — the precise code region where this integration's one real bug already occurred, with no compiler available to catch a second. The mathematics does not require replacing it: `y(x) = Σp_i(x)E_i(x)` describes the pseudo-MoE corridor layer, which lives above and is dense-based.

### Speculative — `k_draft` becomes a controlled variable

`k_draft` is explicitly a control variable in the mathematics (§12: `J_total↓ or drift↑ ⟹ k_draft↓`). Today it is hardcoded to `8` with a `// TODO get from config?`, and `llama_mycelium_spec_should_run()` — fully implemented — is called from nowhere. The shadow branch-thermodynamics block duplicates state Runtime already tracks and is deleted. `r_acc` becomes the real `J_spec` source.

### Model, batching, memory interface, scheduler, ggml ops, buffer ownership

Unchanged, and each for a specific structural reason, not from caution:

- **Model** is per-checkpoint; Runtime is per-session. llama.cpp supports several contexts sharing one loaded model — merging Model into Runtime destroys that. The mathematics gives no reason to.
- **Batching** is token/sequence organization, orthogonal to `x(t)`.
- **Memory interface** already abstracts backends behind one contract, which is exactly what Runtime-supplied policy needs.
- **Scheduler / backend scheduling / ggml ops** are the dissipative substrate itself — `Φ_diss` flows *through* them. Nothing in the mathematics addresses node-to-device assignment. The one genuine future exception is layer offload driven by corridor hotness, which touches buffer placement narrowly, not the partitioning algorithm.

If any of these must change later, it will be because a specific equation demands it — and that demand does not exist in the document you supplied.

---

## Part 4 — Where the current code contradicts the mathematics

Stated plainly, including where what I built is wrong.

1. **Epoch consistency is violated by my own feedback loop.** `graph_feedback()` updates edge weights on *every decode step*. The mathematics requires structure frozen within an epoch, with changes committed only at boundaries. The update must accumulate as pending and commit at the boundary. This is a direct contradiction, and it is mine.

2. **The interaction graph I built is the wrong graph.** I built corridor→physical-expert edges. The mathematics specifies corridor↔corridor couplings `J_ij` under a spin-glass Hamiltonian, with expert influence as a downstream projection. You told me "corridor ≠ expert" repeatedly; the mathematics says *why* — pseudo-experts are corridors over a dense base, so an expert-indexed graph cannot be the primary structure, and does not exist at all for dense models.

3. **`n_current` is not `J_total`.** Substrate computes `corridor_coherence − collapse_rate − migration_instability`. The specification requires five separately measured negentropies with weights. The current scalar is a proxy of exactly the kind §17 rejects.

4. **`J_attn` is unmeasurable today** (R1), yet carries the leading weight.

5. **`Φ_diss` is fabricated** (R5) — hardcoded `kv_hits`, zeroed speculative counters — so `χ`, `V(R)` and `η` are currently computed from fiction.

6. **No order parameter, no spin glass, no percolation, no flow ODE.** `η`, `H(s)`, `p_conn`, and `τ·J̇ = −(J − J*)` have no implementation. `j_star` is copied as a value where the mathematics defines it as the fixed point of a relaxation.

7. **Lyapunov sign convention conflicts.** The mathematics: emergency at `V > V_crit`. The substrate guardrail: `QUALITY_DROP` when `v_lyapunov < v_lyapunov_min`. These are opposite comparisons. One of them is wrong and I cannot tell which from the code alone — `V` is externally supplied today, so both conventions are internally consistent and silently incompatible. This needs your ruling.

8. **Dead value:** `mycelium_veto` on `llm_graph_context` is computed and never consumed.

---

## Part 5 — What must be settled before any rewrite

Four items, all of which are yours to decide and none of which I should decide for you:

1. **R1 — the flash-attention fork.** Kernel change (correct, invasive, per-backend), explicit-softmax-only (honest, costs performance), or proxy (contradicts §17). This gates `J_attn`, and therefore `J_total`.
2. **Lyapunov sign convention** (Part 4 §7).
3. **Epoch length and boundary policy** — since epoch boundaries become the graph-invalidation points (R2), epoch length now directly governs both structural adaptivity and compute cost. Currently `EpochManager` decides this Python-side; under the target architecture the native runtime must honor it.
4. **`η`'s target band.** `0 < η < 1` is the emergence condition, but the operating band, `a`, `b`, `τ_η` and the noise term are parameters of your mathematics, not mine to invent.

The honest constraint on all of it, unchanged: this checkout has no build system and is missing ~15 headers, so nothing here can be compile-verified. A rewrite of this depth — decode orchestration, KV placement, graph instrumentation — is where that stops being a nuisance and starts being the dominant risk. I would want a compiling tree before touching KV placement or the decode pipeline, and I will say so rather than deliver something I have only reasoned about.
