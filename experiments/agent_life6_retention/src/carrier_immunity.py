"""carrier_immunity.py — LIFE-6 Branch B's lever (PROTOCOL.md §4, used
ONLY if Branch A/A3 fails to raise retention): exempt a small number of
young Imp=1 carriers from population death for one generation, without
editing `arch2/evolve.py`.

Confirmed this session by reading `arch2/evolve.py` in full: "death" that
actually removes a genotype from `self.population` (and is what
`death_gen_index`-measured lifespan tracks -- `summary["generations"][i]
["deaths"]` is built from exactly `d1 + d2`, `arch2/evolve.py:448`) is
produced by two plain, side-effect-free instance methods,
`_d1_screen_deaths`/`_d2_subsidy_deaths`, called from `run_generation()`
via ordinary `self.` dispatch -- no hidden global population cull exists
(`POP_SIZE` only sizes `reproduce()`'s next-gen output, never truncates
an existing population). D3 (`_d3_elite_dropouts`) does NOT remove
anything from `population` (SPEC.md: a dropped-elite genotype "remains
alive... on subsidy") -- there is nothing to protect there, so it is
deliberately NOT overridden here.

Reading "не D1-kill" as "exempt from population death broadly" (D1 ∪ D2),
not literally D1 alone (PROTOCOL.md §0/§4): D3 never kills, so exempting
only D1 while D2 still kills the same genotype the same generation would
be a no-op roughly half the time -- exempting both is the only reading
that actually moves `median_L`/`frac_L_ge3`, the primary metrics.

Architecturally safe (confirmed, not assumed): `self.archive` is
append-only regardless of death (written only in `_admit`/`seed`, never
deleted) -- so A5-style archive-monotonicity invariants prior packages
relied on stay true. `merge_events` is untouched by death. `stall_count`/
`extinct` can only be helped (more survivors -> more chances at
`any_win=True`), never hurt.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402

E = LD.E

M_PROTECTED_PER_SLOT = 2
PROTECTED_SLOTS = (0, 2, 3)   # READ, LOOKUP, COMPUTE -- FORMAT (1) excluded, Imp(FORMAT) always False


class ImmuneEvolution(E.Evolution):
    """Constructed exactly like `E.Evolution(...)` everywhere else in
    this lineage (same kwargs, same call site shape in `orchestrator.py`)
    -- `arch2.evolve.build()` hardcodes the concrete class, but every
    LIFE-3/4/5 orchestrator already constructs `Evolution(...)` directly,
    never through `build()`, so subclassing needs no monkey-patching."""

    def _d1_screen_deaths(self, scored: dict, ids: list) -> list:
        dead = super()._d1_screen_deaths(scored, ids)
        protected = getattr(self, "protected_ids", None) or set()
        return [(c, why) for c, why in dead if c not in protected]

    def _d2_subsidy_deaths(self, scored: dict, ids: list, front: set) -> list:
        dead = super()._d2_subsidy_deaths(scored, ids, front)
        protected = getattr(self, "protected_ids", None) or set()
        return [(c, why) for c, why in dead if c not in protected]


def compute_protected_ids(ev, gen: int, imp_fn, M: int = M_PROTECTED_PER_SLOT) -> set:
    """PROTOCOL.md §4: recomputed fresh every generation, from currently
    alive (`ev.population`), non-reference genotypes. Per slot s in
    {READ, LOOKUP, COMPUTE}: up to M with Imp(s)=1, smallest age
    (= gen - birth_gen, i.e. YOUNGEST) first, ties broken by `ev.rng`
    (deterministic given the run's own seed, not Python's unordered
    dict/set iteration)."""
    reference_cids = set(ev.references)
    protected: set = set()
    for s in PROTECTED_SLOTS:
        candidates = []
        for cid in ev.population:
            if cid in reference_cids:
                continue
            g = ev.genotypes.get(cid)
            if g is None:
                continue
            if s in imp_fn(g):
                birth_gen = g.get("gen", 0)
                age = gen - birth_gen
                candidates.append((age, cid))
        if not candidates:
            continue
        tiebreak = {cid: ev.rng.random() for _, cid in candidates}
        candidates.sort(key=lambda t: (t[0], tiebreak[t[1]]))   # smallest age first, random tie-break
        protected.update(cid for _, cid in candidates[:M])
    return protected
