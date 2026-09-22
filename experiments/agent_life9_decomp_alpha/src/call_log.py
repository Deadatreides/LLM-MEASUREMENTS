"""call_log.py — copy (not import) of `agent_life3_block_live/src/
call_log.py`'s pattern, adapted: writer of `metrics/decomp_calls.jsonl`
(Phase C's anti-fraud log, task's own naming) instead of
`live_call_log.jsonl` (Phase A/B spend zero new calls — nothing to log
there; this file is only ever written during Phase C).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = AGENT_DIR / "metrics" / "decomp_calls.jsonl"


def append_call_log(*, model: str, task_id: str, step: str, seed: int,
                    n_in, n_out, ms, path_to_raw: str = "") -> None:
    """`step` here carries Phase C's role label ("PLANNER"/"EXEC0".."EXEC3")."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.time(), "model": model, "task_id": task_id, "step": step,
        "seed": seed, "n_in": n_in, "n_out": n_out, "ms": ms, "path_to_raw": path_to_raw,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_rows() -> int:
    if not LOG_PATH.exists():
        return 0
    return sum(1 for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
