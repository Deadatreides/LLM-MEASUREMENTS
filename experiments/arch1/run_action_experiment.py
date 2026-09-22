"""run_action_experiment.py — эксперимент 10 (ARCH-1.1).

Использование:
    python run_action_experiment.py --pilot            # 1 модель x 3 задачи, синхронно
    python run_action_experiment.py                    # полный прогон (фазы B и C)
    python run_action_experiment.py --phase analyze    # пересчёт без GPU
"""

import argparse
import json
import random
import sys
from pathlib import Path

from metrics.action_experiment import (
    ARMS,
    aggregate,
    build_configs,
    replay_collision_states,
    run_arm,
    run_escalation,
)

BASE = Path(__file__).resolve().parent
RUNS_DIR = BASE / "runs"
OUT_DIR = BASE / "runs_actions"
RESULT_PATH = BASE / "metrics" / "action_outcomes.json"

MODEL_IDS = ["llama-3.2-1b-instruct-q4_0", "qwen2.5-coder-1.5b-instruct-q4_0", "qwen3-1.7b-q4_0-unsloth"]
REPAIR_SEED = 1000  # новый seed, отличный от seed генерации (1-4)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _load_all(subdir: str) -> list:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((OUT_DIR / subdir).glob("*.json"))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--phase", choices=["b", "c", "analyze"], default=None)
    args = parser.parse_args()

    states = replay_collision_states(RUNS_DIR)
    print(f"фаза A: восстановлено состояний с коллизией: {len(states)}")

    if args.pilot:
        states = [s for s in states if s["model_id"] == MODEL_IDS[0]][:3]
        print(f"  пилот: {len(states)} состояний")

    if args.phase != "analyze":
        configs_by_state = {}
        for s in states:
            rng = random.Random(s["state_id"])
            configs_by_state[s["state_id"]] = build_configs(s["model_id"], MODEL_IDS, rng, REPAIR_SEED)

        # -- фаза B: группируем работу по требуемой модели, чтобы каждая
        # модель грузилась ОДИН раз, а не на каждую попытку
        if args.phase in (None, "b"):
            work = []
            for s in states:
                for arm in ARMS:
                    cfg = configs_by_state[s["state_id"]][arm]
                    out = OUT_DIR / "arms" / f"{s['state_id'].replace('|', '__')}__{arm}.json"
                    if not out.exists():
                        work.append((cfg.model_id, s, cfg, out))

            print(f"фаза B: попыток к исполнению: {len(work)}")
            for model_id in MODEL_IDS:
                items = [w for w in work if w[0] == model_id]
                if not items:
                    continue
                print(f"  === {model_id}: {len(items)} попыток ===")
                from src.model_adapter import ModelAdapter

                adapter = ModelAdapter(model_id)
                for _m, state, cfg, out in items:
                    try:
                        result = run_arm(state, cfg, adapter)
                    except Exception as exc:
                        result = {"state_id": state["state_id"], "arm": cfg.arm,
                                  "harness_error": f"{type(exc).__name__}: {exc}"}
                    _write(out, result)
                del adapter

        # -- фаза C: эскалация; нужны все модели сразу (плечо CHANGE_MODEL)
        if args.phase in (None, "c"):
            pending = [
                s for s in states
                if not (OUT_DIR / "escalation" / f"{s['state_id'].replace('|', '__')}.json").exists()
            ]
            print(f"фаза C: траекторий к исполнению: {len(pending)}")
            if pending:
                from src.model_adapter import ModelAdapter

                adapters = {m: ModelAdapter(m) for m in MODEL_IDS}
                for s in pending:
                    rng = random.Random("esc:" + s["state_id"])
                    try:
                        result = run_escalation(s, configs_by_state[s["state_id"]], adapters, rng)
                    except Exception as exc:
                        result = {"state_id": s["state_id"],
                                  "harness_error": f"{type(exc).__name__}: {exc}"}
                    _write(OUT_DIR / "escalation" / f"{s['state_id'].replace('|', '__')}.json", result)
                del adapters

    attempts = [a for a in _load_all("arms") if "harness_error" not in a]
    escalations = [e for e in _load_all("escalation") if "harness_error" not in e]
    errors = len(_load_all("arms")) - len(attempts) + len(_load_all("escalation")) - len(escalations)

    report = aggregate(attempts, escalations, RUNS_DIR)
    report["harness_errors"] = errors
    _write(RESULT_PATH, report)

    print(f"\nзаписано: {RESULT_PATH}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
