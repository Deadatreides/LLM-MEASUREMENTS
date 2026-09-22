"""pareto_death.py — LIFE-7 Branch A's first lever (PROTOCOL.md §3, A1):
a genotype (non-reference) flagged dead by arch2's own economic D1/D2
survives anyway unless some OTHER alive non-reference genotype weakly
dominates it on an objective vector (r not less, extra-objectives not
less, and at least one strictly greater).

Confirmed this session by reading `arch2/evolve.py` + `arch2/fitness.py`
in full: `scored[cid]["r"]` (built by `F.evaluate_population` ->
`F.metrics`) is the plain absolute resolve-rate, already handed to
`_d1_screen_deaths`/`_d2_subsidy_deaths` as their own `scored` argument
-- no independent `F.metrics` call needed here. `_d1_screen_deaths`'s
`ids` parameter is `self.population` (full, unfiltered -- `self.
population` itself is never narrowed until AFTER both death calls, at
`arch2/evolve.py:469`); `_d2_subsidy_deaths`'s `ids` parameter is
`survivors`, the ALREADY-D1-filtered local variable. The Pareto-
domination pool below always uses the `ids` argument actually passed to
the override, never `self.population` -- using the latter inside the D2
override would incorrectly treat D1's casualties as still alive.

This `(r, extra)` notion is DELIBERATELY unrelated to arch2's own `front`
(the three-objective `(r, -c, u)` Pareto front already computed inside
`run_generation` and passed into `_d2_subsidy_deaths` as `front: set` --
see PROTOCOL.md for the full citation). Reusing that name here would be
misleading, since arch2's own front never sees `n_imp`/per-slot Imp at
all.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402

E = LD.E

SCALAR_NIMP = "scalar_nimp"     # Branch A: extra objective = (|ImpSet|,)
VECTOR_SLOTS = "vector_slots"   # Phase A3: extra objective = (Imp_READ, Imp_LOOKUP, Imp_COMPUTE)
RETENTION_SLOTS = (0, 2, 3)


def _extra_objective(genotype: dict, imp_fn, mode: str) -> tuple:
    imp = imp_fn(genotype)
    if mode == SCALAR_NIMP:
        return (len(imp),)
    if mode == VECTOR_SLOTS:
        return tuple(1 if s in imp else 0 for s in RETENTION_SLOTS)
    raise ValueError(f"unknown objective mode: {mode}")


class ParetoEvolution(E.Evolution):
    """Constructed exactly like `E.Evolution(...)` everywhere else in
    this lineage (same kwargs) -- no `arch2.evolve.build()` call site
    exists in any orchestrator here, so subclassing needs no
    monkeypatching (same precedent as LIFE-6's `ImmuneEvolution`).

    Two extra plain attributes, set once after construction (not a
    dataclass field -- `Evolution` has no `__slots__`, arbitrary
    attributes are safe, same pattern as LIFE-6's `protected_ids`):
      `self._imp_fn` -- the closure `orchestrator.py` already builds
      `self._objective_mode` -- SCALAR_NIMP (Branch A) or VECTOR_SLOTS (A3)
    Both default to `None`/`SCALAR_NIMP` so the class degrades to plain
    `Evolution` behavior if never configured (defensive, not relied on --
    `orchestrator.py` always sets `_imp_fn` before the first generation).
    """

    def _dominates(self, cid_b: str, cid_a: str, scored_map: dict) -> bool:
        """Does `cid_b` weakly-dominate `cid_a`? r not less, extra
        objectives not less (elementwise), at least one strictly
        greater."""
        imp_fn = getattr(self, "_imp_fn", None)
        if imp_fn is None:
            return False
        mode = getattr(self, "_objective_mode", SCALAR_NIMP)
        ra = scored_map.get(cid_a, {}).get("r")
        rb = scored_map.get(cid_b, {}).get("r")
        if ra is None or rb is None:
            return False
        ea = _extra_objective(self.genotypes[cid_a], imp_fn, mode)
        eb = _extra_objective(self.genotypes[cid_b], imp_fn, mode)
        if rb < ra or any(y < x for x, y in zip(ea, eb)):
            return False
        return (rb > ra) or any(y > x for x, y in zip(ea, eb))

    def _filter_non_dominated(self, dead_list: list, pool_ids: list, scored_map: dict) -> list:
        imp_fn = getattr(self, "_imp_fn", None)
        if imp_fn is None:
            return dead_list   # unconfigured -- degrade to plain D1/D2 behavior
        reference_cids = set(self.references)
        pool = [c for c in pool_ids if c not in reference_cids]
        kept_dead = []
        n_exempted = 0
        for cid, why in dead_list:
            dominated = any(self._dominates(other, cid, scored_map)
                           for other in pool if other != cid and other in scored_map)
            if dominated:
                kept_dead.append((cid, why))
            else:
                n_exempted += 1
        self._pareto_forensic = getattr(self, "_pareto_forensic", [])
        self._pareto_forensic.append({"gen": self.current_gen, "n_candidates": len(dead_list),
                                      "n_exempted": n_exempted})
        return kept_dead

    def _d1_screen_deaths(self, scored: dict, ids: list) -> list:
        dead = super()._d1_screen_deaths(scored, ids)
        return self._filter_non_dominated(dead, ids, scored)

    def _d2_subsidy_deaths(self, scored: dict, ids: list, front: set) -> list:
        dead = super()._d2_subsidy_deaths(scored, ids, front)
        return self._filter_non_dominated(dead, ids, scored)
