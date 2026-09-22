"""stall_from_improvement.py — LIFE-7 Branch A's second lever (PROTOCOL.md
§3, A2): redefine "stall" as "neither `best_r_alive` nor `best_n_imp_alive`
improved over this run's historical best," instead of arch2's own
gate-beating criterion, WITHOUT editing `arch2/evolve.py`.

Confirmed this session by reading `arch2/evolve.py` in full: the stall/
extinct block is inline in `run_generation` (`self.stall_count = 0 if
any_win else self.stall_count + 1`, `if self.stall_count >= g_stall:
self.extinct = True`, `arch2/evolve.py:427-443`) -- NOT its own
overridable method (only 4 code sites touch these fields at all: the
dataclass default, this block, one straight copy into that generation's
`report` dict, and reads by callers/`summary()`). Confirmed safe to
override via plain post-hoc attribute assignment on the `Evolution`
instance immediately after `ev.run_generation(gen)` returns: nothing else
inside that SAME call re-reads `self.stall_count`/`self.extinct` (the
report-copy happens BEFORE any external override could run), so
overwriting them here takes full effect for the caller's `if ev.extinct:
break` check and for the NEXT generation's internal `self.stall_count + 1`
arithmetic (which is thrown away and re-overwritten by this same tracker
every generation anyway, so what arch2 computes internally in between is
never actually read by anything of ours).

One thing this does NOT change (documented, not worked around):
`ev.generations[-1]["stall_count"/"extinct"]` (the already-serialized
per-generation history) keeps showing arch2's original gate-based verdict
for that generation forever -- only the LIVE attributes and the final
`summary()["extinct"]`/`summary()["n_generations"]` reflect this override.
`metrics_lib.py` is written to only ever read the top-level fields, never
the per-generation ones, for exactly this reason.
"""

from __future__ import annotations


class StallFromImprovement:
    """One instance per cell (per seed), NOT shared across cells --
    `best_r_so_far`/`best_n_imp_so_far` are this RUN's own historical
    bests, reset for every new `Evolution`."""

    def __init__(self, g_stall: int):
        self.g_stall = g_stall
        self.best_r_so_far = float("-inf")
        self.best_n_imp_so_far = -1
        self.stall_count = 0
        self.improved_log: list = []   # forensic: which generations actually reset stall

    def update(self, ev, imp_fn) -> None:
        """Call once, immediately after `ev.run_generation(gen)` returns
        and BEFORE the caller's `if ev.extinct: break` check. Overwrites
        `ev.stall_count`/`ev.extinct` in place."""
        reference_cids = set(ev.references)
        latest_report = ev.generations[-1]
        scores = latest_report.get("scores") or {}
        alive_non_ref = [cid for cid in ev.population if cid not in reference_cids]

        current_best_r = float("-inf")
        current_best_n_imp = -1
        for cid in alive_non_ref:
            r = scores.get(cid, {}).get("r")
            if r is not None and r > current_best_r:
                current_best_r = r
            g = ev.genotypes.get(cid)
            if g is not None:
                n_imp = len(imp_fn(g))
                if n_imp > current_best_n_imp:
                    current_best_n_imp = n_imp

        improved = (current_best_r > self.best_r_so_far) or (current_best_n_imp > self.best_n_imp_so_far)
        self.best_r_so_far = max(self.best_r_so_far, current_best_r)
        self.best_n_imp_so_far = max(self.best_n_imp_so_far, current_best_n_imp)
        self.stall_count = 0 if improved else self.stall_count + 1
        self.improved_log.append({"gen": latest_report["generation"], "improved": improved,
                                  "current_best_r": current_best_r,
                                  "current_best_n_imp": current_best_n_imp})

        ev.stall_count = self.stall_count
        ev.extinct = self.stall_count >= self.g_stall
