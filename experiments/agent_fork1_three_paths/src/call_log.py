"""call_log.py — FORK-1's own call log writers, own files
(`metrics/whole_calls.jsonl`, `metrics/union_calls.jsonl`)."""

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


def log_whole_call(*, model: str, task_id: str, seed: int, n_in, n_out, ms, failed: bool) -> None:
    _append(METRICS_DIR / "whole_calls.jsonl", {
        "ts": time.time(), "model": model, "task_id": task_id, "seed": seed,
        "n_in": n_in, "n_out": n_out, "ms": ms, "failed": failed,
    })


def log_union_call(*, model: str, task_id: str, atom: str, seed: int, n_in, n_out, ms,
                   failed: bool) -> None:
    _append(METRICS_DIR / "union_calls.jsonl", {
        "ts": time.time(), "model": model, "task_id": task_id, "atom": atom, "seed": seed,
        "n_in": n_in, "n_out": n_out, "ms": ms, "failed": failed,
    })


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
