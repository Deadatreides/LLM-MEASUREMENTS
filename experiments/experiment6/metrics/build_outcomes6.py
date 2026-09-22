"""
Builds the per-unit state + full action-outcome table (spec sections 9-14).

Units = (task_id, model_id, base_temperature), 6 tasks x 2 models x 3
temperatures = 36. For each unit, the "observation" is seeds 1-8 of that
unit's own pool (reused unmodified from experiment 5 -- no new
generation). Each of the 5 possible actions' outcome is likewise built by
RE-SLICING already-existing data (experiment 5's other temperature/model
pools, or this experiment's new CHANGE_PROMPT arm) -- nothing here re-runs
a model for the "what happened" side of the table; only the CHANGE_PROMPT
arm required new generation (see runner6_prompt_arm.py), and the state
classifier is computed strictly from the first 8 (see state_classifier.py,
no leakage from any of the action pools).
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))

from state_classifier import cluster_of, first_k_features, classify_state, ACTION_BY_STATE

EXP5_RUNS = r"<PROJECT_ROOT>\trace-probe\experiment5\runs"
EXP6_RUNS = r"<PROJECT_ROOT>\trace-probe\experiment6\runs"
METRICS_DIR = os.path.dirname(__file__)

MODELS = ["qwen3-1.7b-q4_0-unsloth", "llama-3.2-1b-instruct-q4_0"]
OTHER_MODEL = {MODELS[0]: MODELS[1], MODELS[1]: MODELS[0]}
TASKS = ["CODE_01", "CODE_02", "CODE_03", "CODE_04", "CODE_05", "CODE_06"]
TEMPS = [0.0, 0.5, 1.0]
ALT_T = {0.0: 0.5, 0.5: 1.0, 1.0: 0.5}


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_exp5_row(row):
    """experiment 5 rows use 'correctness' (bool) + error_class/signature,
    not the E-taxonomy 'correctness_status' field used natively in
    experiment 6's own generations. Map onto the same 3-way status here,
    consistent with error_taxonomy.classify()'s UNKNOWN rule."""
    if row["correctness"]:
        status = "CORRECT"
    elif row["error_class"] == "FORMAT_ERROR" and row.get("error_signature") == "no_extractable_code":
        status = "UNKNOWN"
    else:
        status = "INCORRECT"
    return {**row, "correctness_status": status}


def load_exp5_pool(task_id, model_id, temperature):
    path = os.path.join(EXP5_RUNS, f"index.{model_id}.jsonl")
    rows = [r for r in load_jsonl(path) if r["task_id"] == task_id and r["temperature"] == temperature and not r["generation_failed"]]
    rows.sort(key=lambda r: r["seed"])
    return [normalize_exp5_row(r) for r in rows]


def load_changeprompt_pool(task_id, model_id, temperature):
    path = os.path.join(EXP6_RUNS, f"index.{model_id}.changeprompt.jsonl")
    rows = [r for r in load_jsonl(path) if r["task_id"] == task_id and r["temperature"] == temperature and not r["generation_failed"]]
    rows.sort(key=lambda r: r["seed"])
    return rows  # already has correctness_status natively


def action_outcome(first_k, new_pool, oracle_before, pass_before):
    if not new_pool:
        return None
    seen_clusters = set(cluster_of(r) for r in first_k)
    combined = first_k + new_pool
    combined_clusters = [cluster_of(r) for r in combined]
    oracle_after = 1.0 if "CORRECT" in combined_clusters else 0.0
    n_correct_combined = sum(1 for c in combined_clusters if c == "CORRECT")
    pass_after = n_correct_combined / len(combined)

    new_correct = new_error = repeated_error = repeated_correct = 0
    for r in new_pool:
        c = cluster_of(r)
        if c == "CORRECT":
            if "CORRECT" in seen_clusters:
                repeated_correct += 1
            else:
                new_correct += 1
        elif c != "UNKNOWN":
            if c in seen_clusters:
                repeated_error += 1
            else:
                new_error += 1

    n_new = len(new_pool)
    tokens = sum(r.get("output_tokens") or 0 for r in new_pool)
    time_sec = sum(r.get("generation_time_sec") or 0 for r in new_pool)
    delta_oracle = oracle_after - oracle_before
    delta_pass = pass_after - pass_before

    return {
        "n_new_generations": n_new, "oracle_after": oracle_after, "delta_oracle": round(delta_oracle, 4),
        "pass_after": round(pass_after, 4), "delta_pass": round(delta_pass, 4),
        "p_new_correct": round(new_correct / n_new, 4), "p_new_error": round(new_error / n_new, 4),
        "p_repeated_error": round(repeated_error / n_new, 4), "p_repeated_correct": round(repeated_correct / n_new, 4),
        "tokens": tokens, "time_sec": round(time_sec, 3), "calls": n_new,
        "utility_per_token": round(delta_oracle / tokens, 6) if tokens else None,
        "utility_per_second": round(delta_oracle / time_sec, 5) if time_sec else None,
    }


def build_unit(task_id, model_id, temperature):
    base_pool = load_exp5_pool(task_id, model_id, temperature)
    if len(base_pool) < 16:
        return None
    first8 = base_pool[:8]
    feats = first_k_features(base_pool, k=8)
    state, reason = classify_state(feats)
    oracle_before = feats["oracle_at_k"]
    pass_before = feats["pass_at_k"]

    actions = {"STOP": {"n_new_generations": 0, "oracle_after": oracle_before, "delta_oracle": 0.0,
                         "pass_after": pass_before, "delta_pass": 0.0, "p_new_correct": None, "p_new_error": None,
                         "p_repeated_error": None, "p_repeated_correct": None, "tokens": 0, "time_sec": 0.0,
                         "calls": 0, "utility_per_token": None, "utility_per_second": None}}

    same_pool = base_pool[8:16]
    actions["SAME_MODEL"] = action_outcome(first8, same_pool, oracle_before, pass_before)

    alt_t = ALT_T[temperature]
    alt_pool = load_exp5_pool(task_id, model_id, alt_t)[:8]
    actions["CHANGE_TEMPERATURE"] = action_outcome(first8, alt_pool, oracle_before, pass_before)

    switch_pool = load_exp5_pool(task_id, OTHER_MODEL[model_id], temperature)[:8]
    actions["SWITCH_MODEL"] = action_outcome(first8, switch_pool, oracle_before, pass_before)

    cp_pool = load_changeprompt_pool(task_id, model_id, temperature)[:8]
    actions["CHANGE_PROMPT"] = action_outcome(first8, cp_pool, oracle_before, pass_before)

    return {
        "unit": f"{task_id}/{model_id}/T{temperature}", "task_id": task_id, "model_id": model_id, "temperature": temperature,
        "state": state, "state_reason": reason, "features": feats,
        "recommended_action_state_aware": ACTION_BY_STATE[state],
        "actions": {k: v for k, v in actions.items() if v is not None},
    }


def main():
    units = []
    for task_id in TASKS:
        for model_id in MODELS:
            for t in TEMPS:
                u = build_unit(task_id, model_id, t)
                if u:
                    units.append(u)

    with open(os.path.join(METRICS_DIR, "outcomes6.json"), "w", encoding="utf-8") as f:
        json.dump(units, f, ensure_ascii=False, indent=2)

    state_counts = collections.Counter(u["state"] for u in units)
    print(f"n_units={len(units)}")
    print("state distribution:", dict(state_counts))
    incomplete = [u["unit"] for u in units if len(u["actions"]) < 5]
    if incomplete:
        print("units with missing action arms:", incomplete)


if __name__ == "__main__":
    main()
