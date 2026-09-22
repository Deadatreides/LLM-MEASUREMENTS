"""call_log.py — единственный писатель metrics/live_call_log.jsonl (PROTOCOL.md §9).

live_grid_builder.py и baselines.py (B0) оба вызывают только эту функцию,
чтобы формат строки анти-фрод журнала не расходился между двумя местами,
где вообще вызывается generate() в этом пакете.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = AGENT_DIR / "metrics" / "live_call_log.jsonl"


def append_call_log(*, model: str, task_id: str, step: str, seed: int,
                    n_in, n_out, ms, path_to_raw: str = "") -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.time(), "model": model, "task_id": task_id, "step": step,
        "seed": seed, "n_in": n_in, "n_out": n_out, "ms": ms, "path_to_raw": path_to_raw,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_rows() -> int:
    if not LOG_PATH.exists():
        return 0
    return sum(1 for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip())


def read_rows() -> list:
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
