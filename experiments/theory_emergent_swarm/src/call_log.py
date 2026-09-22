"""call_log.py — this package's own call log writer, own file
(`metrics/filter_calls.jsonl`). Persists `raw_text` per call from the
FIRST run (DELTA-0's own postmortem lesson: aggregate-only persistence
loses exactly the detail a surprising result needs — do not retrofit
this later, do it from call zero).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = AGENT_DIR / "metrics"


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_filter_call(*, model: str, task_id: str, family: str, seed: int, n_in, n_out, ms,
                    failed: bool, extracted: list, raw_text: str) -> None:
    _append(METRICS_DIR / "filter_calls.jsonl", {
        "ts": time.time(), "model": model, "task_id": task_id, "family": family, "seed": seed,
        "n_in": n_in, "n_out": n_out, "ms": ms, "failed": failed, "extracted": extracted,
        "raw_text": raw_text,
    })


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
