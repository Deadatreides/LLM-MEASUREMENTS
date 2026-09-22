"""
Sections 9, 18-20: evaluate 5 policies (A always-same, B fixed-temperature,
C diversify, D state-aware, RANDOM control) against the pre-built
action-outcome table (outcomes6.json), on a TRAIN/TEST split BY TASK
(spec section 23) -- no threshold in state_classifier.py was fit on this
data (all thresholds are fixed, chosen before looking at outcomes), so
train/test here exists specifically to check the STATE-AWARE POLICY's
generalization, not to fit anything.
"""
import json
import os
import random
import statistics
import sys

METRICS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))
from state_classifier import ACTION_BY_STATE  # noqa: E402

TRAIN_TASKS = {"CODE_01", "CODE_02", "CODE_03", "CODE_04"}
TEST_TASKS = {"CODE_05", "CODE_06"}
ACTIONS = ["STOP", "SAME_MODEL", "CHANGE_TEMPERATURE", "SWITCH_MODEL", "CHANGE_PROMPT"]
RNG_SEED = 20260817


def load_units():
    with open(os.path.join(METRICS_DIR, "outcomes6.json"), encoding="utf-8") as f:
        return json.load(f)


def policy_action(policy, unit, rng):
    if policy == "A_ALWAYS_SAME_MODEL":
        return "SAME_MODEL"
    if policy == "B_FIXED_TEMPERATURE":
        return "CHANGE_TEMPERATURE"
    if policy == "C_DIVERSIFY":
        return "SWITCH_MODEL" if unit["features"]["pass_at_k"] < 1.0 else "STOP"
    if policy == "D_STATE_AWARE":
        return ACTION_BY_STATE[unit["state"]]
    if policy == "RANDOM":
        return rng.choice(ACTIONS)
    raise ValueError(policy)


def evaluate(policy, units, rng):
    rows = []
    for u in units:
        action = policy_action(policy, u, rng)
        outcome = u["actions"].get(action)
        if outcome is None:
            continue
        rows.append({"unit": u["unit"], "state": u["state"], "action": action, **outcome})

    n = len(rows)
    if n == 0:
        return None
    total_tokens = sum(r["tokens"] for r in rows)
    total_time = sum(r["time_sec"] for r in rows)
    mean_delta_oracle = statistics.mean(r["delta_oracle"] for r in rows)
    mean_delta_pass = statistics.mean(r["delta_pass"] for r in rows)
    final_oracle = statistics.mean(r["oracle_after"] for r in rows)
    final_pass = statistics.mean(r["pass_after"] for r in rows)
    stop_rate = sum(1 for r in rows if r["action"] == "STOP") / n
    switch_rate = sum(1 for r in rows if r["action"] == "SWITCH_MODEL") / n
    same_rate = sum(1 for r in rows if r["action"] == "SAME_MODEL") / n

    return {
        "n_units": n, "final_mean_oracle": round(final_oracle, 4), "final_mean_pass": round(final_pass, 4),
        "mean_delta_oracle": round(mean_delta_oracle, 4), "mean_delta_pass": round(mean_delta_pass, 4),
        "total_tokens": total_tokens, "total_time_sec": round(total_time, 2),
        "overall_utility_per_token": round(sum(r["delta_oracle"] for r in rows) / total_tokens, 6) if total_tokens else None,
        "overall_utility_per_second": round(sum(r["delta_oracle"] for r in rows) / total_time, 5) if total_time else None,
        "stop_rate": round(stop_rate, 4), "switch_rate": round(switch_rate, 4), "same_model_rate": round(same_rate, 4),
        "rows": rows,
    }


def main():
    units = load_units()
    train_units = [u for u in units if u["task_id"] in TRAIN_TASKS]
    test_units = [u for u in units if u["task_id"] in TEST_TASKS]

    policies = ["A_ALWAYS_SAME_MODEL", "B_FIXED_TEMPERATURE", "C_DIVERSIFY", "D_STATE_AWARE", "RANDOM"]
    out = {"train": {}, "test": {}, "all": {}}
    for split_name, split_units in [("train", train_units), ("test", test_units), ("all", units)]:
        for p in policies:
            rng = random.Random(RNG_SEED)  # same seed sequence per policy call for reproducibility
            res = evaluate(p, split_units, rng)
            if res:
                res_no_rows = {k: v for k, v in res.items() if k != "rows"}
                out[split_name][p] = res_no_rows

    with open(os.path.join(METRICS_DIR, "policies6_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=== TEST split (held-out tasks CODE_05, CODE_06) ===")
    for p in policies:
        r = out["test"].get(p)
        if r:
            print(p, {k: v for k, v in r.items()})
    print()
    print("=== TRAIN split ===")
    for p in policies:
        r = out["train"].get(p)
        if r:
            print(p, {k: v for k, v in r.items()})


if __name__ == "__main__":
    main()
