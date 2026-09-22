"""build_grid.py — тонкая точка входа: PROTOCOL.md §3.1 живая сетка.
Резюмируемо (см. `src/live_grid_builder.py`); безопасно перезапускать.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD          # noqa: E402
import live_grid_builder as GRID   # noqa: E402


def main() -> int:
    ds = LD.default_dataset()
    all_ids = sorted(ds.tasks)
    t_start = time.time()
    cells = GRID.build_live_grid(all_ids, ds.tasks, ds.split)
    n_rows = len(cells)
    target = len(all_ids) * len(GRID.STEP_KINDS) * len(GRID.MODEL_IDS)
    train_floor = int(0.8 * 6 * 4 * 100)
    n_train_rows = sum(1 for (t, k, m) in cells if t in set(ds.split["train"]))
    print(f"[build_grid] ИТОГО: {n_rows}/{target} ячеек за {time.time() - t_start:.0f}с "
          f"(train-строк {n_train_rows}, приёмочный минимум {train_floor})")
    if n_train_rows < train_floor:
        print("[build_grid] НИЖЕ ПРИЁМОЧНОГО МИНИМУМА -- писать в BLOCKERS.md", flush=True)
        return 1
    print("[build_grid] OK", flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
