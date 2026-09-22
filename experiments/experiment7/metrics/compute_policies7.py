"""
Sections 12, 13, 16-18: two separate, clearly-scoped comparisons over the
12-unit outcome table (outcomes7.json), both computed from state features
that never look past the first 8 generations of their unit.

Honest note on the state classifier before any of this runs (see
state7.py / build_outcomes7.py output): STATE_B_PRODUCTIVE_UNCERTAINTY was
never observed at either level in this dataset (0/12 task-level, 0/10
claim-level) -- Oracle saturates to 1.0 by k<=2 whenever it saturates at
all, so the "does Oracle grow somewhere across k=2,4,6,8" signal has no
room to fire. This state-aware rule was fixed BEFORE this script ran (not
adjusted afterward): STATE_B would route to SAME_MODEL, but the branch is
simply unused in this run -- documented, not deleted or hidden.

COMPARISON 1 (section 16-17, literal "SAME vs SWITCH" baselines):
  BASELINE_1_ALWAYS_SAME, BASELINE_2_RANDOM, BASELINE_3_ALWAYS_SWITCH,
  POLICY_STATE_AWARE_2ACTION (task_state: STATE_A->SWITCH, else->SAME).
  Metric: task-level delta_oracle_task/pass_task, tokens.

COMPARISON 2 (section 8-13/18-19, full 3-action routing):
  POLICY_STATE_AWARE_3ACTION (task_state: STATE_A->SWITCH_MODEL;
  STATE_C + target_claim available->LOCAL_RETRY; STATE_C + not
  applicable->SAME_MODEL) vs ALWAYS_SAME (claim-level outcome only,
  since LOCAL_RETRY has no task-level outcome by construction).
  Metric: claim-level delta_oracle_claim, tokens.
"""
import collections
import json
import os
import random

METRICS_DIR = os.path.dirname(__file__)
RNG_SEED = 20260815


def load_units():
    with open(os.path.join(METRICS_DIR, "outcomes7.json"), encoding="utf-8") as f:
        return json.load(f)


STATE_AWARE_2ACTION = {"STATE_A_STABLE_ERROR": "SWITCH_MODEL", "STATE_B_PRODUCTIVE_UNCERTAINTY": "SAME_MODEL", "STATE_C_NOISE": "SAME_MODEL"}


def comparison1(units, rng):
    policies = ["BASELINE_1_ALWAYS_SAME", "BASELINE_2_RANDOM", "BASELINE_3_ALWAYS_SWITCH", "POLICY_STATE_AWARE_2ACTION"]
    out = {}
    for policy in policies:
        rows = []
        for u in units:
            if policy == "BASELINE_1_ALWAYS_SAME":
                action = "SAME_MODEL"
            elif policy == "BASELINE_3_ALWAYS_SWITCH":
                action = "SWITCH_MODEL"
            elif policy == "BASELINE_2_RANDOM":
                action = rng.choice(["SAME_MODEL", "SWITCH_MODEL"])
            else:
                action = STATE_AWARE_2ACTION[u["task_state"]]
            entry = u["actions"].get(action)
            if entry is None or "tokens" not in entry:
                continue
            rows.append({"unit": u["unit"], "action": action, **entry})

        n = len(rows)
        if not n:
            continue
        total_tokens = sum(r["tokens"] for r in rows)
        total_time = sum(r["time_sec"] for r in rows)
        mean_delta_oracle_task = sum(r["delta_oracle_task"] for r in rows) / n
        mean_delta_pass_task = sum(r["delta_pass_task"] for r in rows) / n
        final_oracle_task = sum(r["oracle_task_after"] for r in rows) / n
        switch_rate = sum(1 for r in rows if r["action"] == "SWITCH_MODEL") / n
        out[policy] = {
            "n_units": n, "final_mean_oracle_task": round(final_oracle_task, 4),
            "mean_delta_oracle_task": round(mean_delta_oracle_task, 4), "mean_delta_pass_task": round(mean_delta_pass_task, 4),
            "total_tokens": total_tokens, "total_time_sec": round(total_time, 2),
            "utility_per_token": round(sum(r["delta_oracle_task"] for r in rows) / total_tokens, 6) if total_tokens else None,
            "switch_rate": round(switch_rate, 4),
        }
    return out


def comparison2(units, rng):
    policies = ["ALWAYS_SAME_MODEL", "STATE_AWARE_3ACTION"]
    out = {}
    for policy in policies:
        rows = []
        for u in units:
            if u["target_claim"] is None and policy == "STATE_AWARE_3ACTION":
                action = "SAME_MODEL"
            elif policy == "ALWAYS_SAME_MODEL":
                action = "SAME_MODEL"
            else:
                if u["task_state"] == "STATE_A_STABLE_ERROR":
                    action = "SWITCH_MODEL"
                else:
                    action = "LOCAL_RETRY"
            entry = u["actions"].get(action)
            if entry is None or entry.get("status") == "LOCAL_RETRY_NOT_APPLICABLE" or "delta_oracle_claim" not in entry:
                continue
            rows.append({"unit": u["unit"], "action": action, **entry})

        n = len(rows)
        if not n:
            continue
        total_tokens = sum(r["tokens"] for r in rows)
        total_time = sum(r["time_sec"] for r in rows)
        mean_delta_oracle_claim = sum(r["delta_oracle_claim"] for r in rows) / n
        mean_delta_pass_claim = sum(r["delta_pass_claim"] for r in rows) / n
        final_oracle_claim = sum(r["oracle_claim_after"] for r in rows) / n
        action_counts = collections.Counter(r["action"] for r in rows)
        out[policy] = {
            "n_units": n, "final_mean_oracle_claim": round(final_oracle_claim, 4),
            "mean_delta_oracle_claim": round(mean_delta_oracle_claim, 4), "mean_delta_pass_claim": round(mean_delta_pass_claim, 4),
            "total_tokens": total_tokens, "total_time_sec": round(total_time, 2),
            "utility_per_token": round(sum(r["delta_oracle_claim"] for r in rows) / total_tokens, 6) if total_tokens else None,
            "action_distribution": dict(action_counts),
        }
    return out


def main():
    units = load_units()
    rng = random.Random(RNG_SEED)
    out = {"comparison1_same_vs_switch": comparison1(units, rng), "comparison2_full_3action": comparison2(units, rng)}
    with open(os.path.join(METRICS_DIR, "policies7_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
