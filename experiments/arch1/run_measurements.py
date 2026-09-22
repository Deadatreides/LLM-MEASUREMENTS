"""run_measurements.py — измерительная кампания этапа 7.

Использование:
    python run_measurements.py --pilot          # 1 модель x 1 задача x 1 seed, синхронно
    python run_measurements.py                  # полная кампания: 3 модели x 4 seed x 20 задач
    python run_measurements.py --aggregate-only  # только пересчёт метрик по уже накопленным runs/
"""

import argparse
import json
import sys
from pathlib import Path

from metrics.runner import aggregate_measurements, run_campaign
from tasks.branch_tasks import TASK_IDS

MODEL_IDS = ("llama-3.2-1b-instruct-q4_0", "qwen2.5-coder-1.5b-instruct-q4_0", "qwen3-1.7b-q4_0-unsloth")
SEEDS = (1, 2, 3, 4)

RUNS_DIR = Path(__file__).resolve().parent / "runs"
MEASUREMENTS_PATH = Path(__file__).resolve().parent / "metrics" / "measurements.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="1 модель x 1 задача x 1 seed, синхронно")
    parser.add_argument("--aggregate-only", action="store_true", help="только пересчитать metrics/measurements.json")
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--tasks", nargs="+", default=list(TASK_IDS))
    args = parser.parse_args()

    if not args.aggregate_only:
        if args.pilot:
            models, seeds, tasks = args.models[:1], args.seeds[:1], args.tasks[:1]
        else:
            models, seeds, tasks = args.models, args.seeds, args.tasks

        print(f"campaign: {len(models)} models x {len(seeds)} seeds x {len(tasks)} tasks "
              f"= {len(models) * len(seeds) * len(tasks)} full run_task() runs")
        run_campaign(models, seeds, tasks, RUNS_DIR)

    measurements = aggregate_measurements(RUNS_DIR)
    MEASUREMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEASUREMENTS_PATH.write_text(json.dumps(measurements, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nmeasurements written to {MEASUREMENTS_PATH}")
    print(json.dumps(measurements, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
