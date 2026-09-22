"""run_optional_evolve.py — Phase B.2 (user-confirmed this session): 4
fresh seeds through the default channel only (no registry code path at
all), on the LIFE-4 grid copied byte-identical into this package. Zero
new generate() calls (grid already built; B0/B1-B3 baselines not needed
here -- Phase B.2 only exists to produce archives for the cross-check
slot map, not to re-measure r vs baselines).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
AGENT_DIR = SCRIPTS.parent
SRC = AGENT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD    # noqa: E402
import orchestrator as O     # noqa: E402

RUNS_DIR = AGENT_DIR / "runs"


def run_all_seeds() -> dict:
    ds = LD.default_dataset()
    panel_info = O.build_panel(ds)
    panel, screen = panel_info["panel"], panel_info["screen"]

    results = {}
    for seed in O.SEEDS:
        out_dir = RUNS_DIR / f"life5_default_s{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        summary = O.run_cell(seed, ds, panel, screen, out_dir)
        dt = time.time() - t0
        print(f"[run_optional_evolve] seed={seed} done in {dt:.0f}s "
              f"n_generations={summary['n_generations']} extinct={summary['extinct']}", flush=True)
        results[seed] = summary
    return results


if __name__ == "__main__":
    t0 = time.time()
    run_all_seeds()
    print(f"[run_optional_evolve] всего {time.time() - t0:.0f}с для {len(O.SEEDS)} seed", flush=True)
