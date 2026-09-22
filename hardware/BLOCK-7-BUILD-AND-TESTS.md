# Block 7 — restoring buildability, and real tests

## 1. A regression I introduced, found and fixed

Before adding anything, I checked whether my own Block 1 change had broken the one thing
in this repository that could actually be built.

It had.

`llama_mycelium_runtime_of()` is implemented **only** in `llama.cpp/src/llama-context.cpp`
— correctly, since that is the sole translation unit that knows `llama_context`'s layout.
But `substrate/CMakeLists.txt` builds `llama_mycelium` as a standalone static library and
links the tests against it *alone*. So since Block 1 the standalone library has carried an
undefined symbol, and both test executables would have failed to link.

I moved ownership into `llama_context` and never checked what that did to the standalone
build. Reporting it rather than quietly patching over it.

**Fix:** `myc-standalone.cpp`, compiled only under `LLAMA_MYCELIUM_STANDALONE` (set by
CMake when `LLAMA_MYCELIUM_INTERNAL` is off). It supplies the resolver for that
configuration with a fixed 8-slot table — no mutex, no allocation. It is deliberately
**not** a fallback inside the production path: the production path has no registry, and
adding one "just in case" is exactly the temporary bridge this architecture removed.

## 2. Tests: 2087 lines deleted, replaced with real ones

The two existing test files allocated runtimes via `llama_mycelium_init()` with fake
context pointers. After Block 1 that function resolves rather than allocates, so every
one of those tests was silently exercising `nullptr`. They were dead, and I had flagged
them as dead for six blocks without fixing them.

Deleted: `test-llama-mycelium.cpp` (813), `test-llama-mycelium-runtime.cpp` (1274).
Added: `test-myc-organism.cpp` (~470).

The new suite tests **the physics**, not the plumbing:

| Test | What it actually proves |
|---|---|
| `slice` | every layer owned exactly once; band sizes differ by ≤1; a 4-layer model produces no empty pseudo-experts |
| `heat_field` | heating one band makes *that* pseudo-expert hot and leaves others cold — the thing that was impossible before Block 6, when every corridor shared a global background |
| `hilbert_locality` | nearest-neighbour on the curve is near in phase space (mean index gap must beat random pairing by 4×); distances sorted ascending; no self-loops |
| `swarm` | **a uniform field condenses nothing** (symmetry unbroken) while a concentrated field does; hot corridors outweigh cold in the ensemble; entropy production is non-negative |
| `epoch_consistency` | 10 feedback steps move **no** coupling weight, deltas accumulate, and only `graph_commit` moves them — `L_active(t) = L_snapshot` verified directly |
| `topology` | at least one expert is fed by more than one corridor — if every expert had exactly one source it would be a mapping, not a graph, which is the thing you told me repeatedly the architecture forbids |
| `guardian_not_latched` | the veto clears when its cause clears, and the cumulative counter still remembers — which is exactly why the epoch's emergency test must not read it (the bug I fixed in Block 6) |
| `energy_projection` | zero assembly gives zero energy regardless of `node_activity` — proving the projection follows swarm assembly, not raw activity |
| `boundary`, `lifecycle`, `isolation` | held positions are held; the organism is owned not registered; two contexts stay independent |

## 3. I could not run them

There is **no compiler in this environment**: no `cl.exe` (Visual Studio is installed but
carries no compiler), no `g++`, no `clang++`, no `cmake`. I checked before writing the
report rather than after.

So the honest status is: the tests are written and mechanically validated, and they have
never been executed. What I did verify:

- every `LLAMA_MYC_*` / `MYC_*` identifier used in the tests resolves against a header
- every function the tests call is defined somewhere in `src/` or `include/`
- every struct field the tests touch exists in the field definitions
- the arithmetic of each assertion traced by hand (e.g. uniform field → all assembly weights equal the mean → all below the `mean × 1.25` condensation threshold → zero structures, which is what the test asserts)

That is not the same as passing. If you have a compiler available, this is now the first
thing in the project that can actually be run:

```bash
cmake -S substrate -B substrate/build && cmake --build substrate/build && ctest --test-dir substrate/build --output-on-failure
```

## 4. Files

New: `substrate/src/myc-standalone.cpp`, `substrate/tests/test-myc-organism.cpp`.
Deleted: both old test files (2087 lines).
Changed: `substrate/CMakeLists.txt` (standalone flag, new source, new test target).

Substrate is now 4398 lines across 19 source files plus one test file.

## 5. Open

- **`j_attn` and `j_behavior` remain 0.** Still the §R1 decision, still yours, still the highest-weight term in `J_total`.
- The `llama.cpp` side still has no build system, so the *production* configuration remains uncompilable for reasons that predate this work. Only the substrate is buildable, and only standalone.
- `LLAMA_MYCELIUM_INTERNAL` builds skip the standalone resolver by design, but nothing in this repository produces such a build yet — so that configuration is untested in a stronger sense than the rest.
