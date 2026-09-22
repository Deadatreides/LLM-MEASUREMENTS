"""
Builds the per-unit (task_id, model_id) state + 3-action (SAME_MODEL,
SWITCH_MODEL, LOCAL_RETRY) outcome table for experiment 7.

Units = 4 tasks x 3 models = 12, all at prompt=P1, temperature=0.5 (spec
section 3/4). "First pass" = seeds 1-8 of each unit's own pool, reused
UNCHANGED from experiment 3's claims_index.jsonl for all 3 models (no
regeneration of data that already exists). SAME_MODEL's outcome = seeds
9-16 of the same (task, model): reused from experiment 3/4 for qwen3,
freshly generated this experiment for llama/coder (runs/claims_index7.jsonl).
SWITCH_MODEL's outcome = the OTHER model's own first-8 pool for the same
(task, prompt, temperature) -- also pre-existing, zero new generation.
LOCAL_RETRY's outcome = the 8 new short generations targeting this unit's
one selected problematic claim (runs/local_retry.<model>.jsonl), or
LOCAL_RETRY_NOT_APPLICABLE if metrics/target_claim.py found nothing to
target.

Two Oracle levels are tracked, kept explicitly separate (see module
docstring of state7.py):
  - TASK level (whole artifact defect-free) -- only meaningful for
    SAME_MODEL/SWITCH_MODEL, which produce full new artifacts.
  - TARGET-CLAIM level (just the one selected claim) -- meaningful and
    directly comparable across ALL THREE actions, since LOCAL_RETRY only
    ever touches this one claim. This is the basis for the core H1-H8-style
    action comparison (section 11-13).
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))

from target_claim import select_target_claim
from state7 import task_level_features, claim_level_features, classify_state, task_defect_signature

EXP3_RUNS = r"<PROJECT_ROOT>\trace-probe\experiment3\runs"
EXP7_RUNS = r"<PROJECT_ROOT>\trace-probe\experiment7\runs"
METRICS_DIR = os.path.dirname(__file__)

MODELS = ["qwen3-1.7b-q4_0-unsloth", "llama-3.2-1b-instruct-q4_0", "qwen2.5-coder-1.5b-instruct-q4_0"]
OTHER_MODELS = {m: [x for x in MODELS if x != m] for m in MODELS}
TASKS = ["CODE_01", "CODE_02", "CODE_03", "CODE_06"]
PROMPT = "P1"
TEMPERATURE = 0.5

CLAIM_ORDER = {
    "CODE_01": ["C1_NAME", "C2_ARGC", "C3_ARG_TYPE", "C4_RETURN_TYPE", "C5_SEMANTICS_CASE", "C6_EDGE_EMPTY", "C7_EDGE_SPACES"],
    "CODE_02": ["C1_NAME", "C2_ARGC", "C3_ARG_TYPE", "C4_RETURN_TYPE", "C5_SEMANTICS", "C6_EDGE_EMPTY", "C7_EDGE_NO_EVEN"],
    "CODE_03": ["C1_NAME", "C2_ARGC", "C3_ARG_TYPE", "C4_RETURN_TYPE", "C5_EDGE_ZERO", "C6_EDGE_ONE", "C7_SEMANTICS_RECURRENCE"],
    "CODE_06": ["C1_NAME", "C2_ARGC", "C3_RETURN_TYPE", "C4_SEMANTICS_INDICES", "C5_SEMANTICS_SMALLEST_I", "C6_EDGE_ONE_SOLUTION", "C7_EDGE_NOT_SAME"],
}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_claims():
    rows = load_jsonl(os.path.join(EXP3_RUNS, "claims_index.jsonl"))
    rows += load_jsonl(os.path.join(EXP7_RUNS, "claims_index7.jsonl"))
    return rows


def load_run_meta():
    """run_id -> (output_tokens, generation_time_sec). The original
    experiment 3 claims_index.jsonl does not carry these fields (only
    experiment 7's own claims_index7.jsonl does) -- pull them from the raw
    a1_index shards instead, which have them for every generation
    regardless of which experiment produced it."""
    meta = {}
    for shard in glob.glob(os.path.join(EXP3_RUNS, "a1_index.*.jsonl")):
        for r in load_jsonl(shard):
            meta[r["run_id"]] = (r.get("output_tokens") or 0, r.get("generation_time_sec") or 0)
    for shard in glob.glob(os.path.join(EXP7_RUNS, "a1_index.*.jsonl")):
        for r in load_jsonl(shard):
            meta[r["run_id"]] = (r.get("output_tokens") or 0, r.get("generation_time_sec") or 0)
    return meta


def claims_by_seed_for_unit(all_claims, task_id, model_id, prompt_id=PROMPT, temperature=TEMPERATURE):
    by_seed = collections.defaultdict(list)
    for r in all_claims:
        if (r["task_id"] == task_id and r["model_id"] == model_id
                and r["prompt_id"] == prompt_id and r["temperature"] == temperature):
            by_seed[r["seed"]].append(r)
    return by_seed


def run_ids_by_seed(all_claims, task_id, model_id):
    """run_id per seed (first row's run_id is enough, all 7 claim rows of
    one seed share the same run_id)."""
    by_seed = claims_by_seed_for_unit(all_claims, task_id, model_id)
    return {seed: rows[0]["run_id"] for seed, rows in by_seed.items()}


def token_time_for_runs(run_meta, run_ids):
    tokens = sum(run_meta[rid][0] for rid in run_ids if rid in run_meta)
    time_sec = sum(run_meta[rid][1] for rid in run_ids if rid in run_meta)
    return tokens, time_sec, len(run_ids)


def claim_oracle_pass(claim_rows):
    cls_seq = [r["classification"] for r in claim_rows]
    oracle = 1.0 if "CORRECT" in cls_seq else 0.0
    pass_rate = cls_seq.count("CORRECT") / len(cls_seq) if cls_seq else None
    return oracle, pass_rate


def build_unit(task_id, model_id, all_claims, run_meta):
    by_seed = claims_by_seed_for_unit(all_claims, task_id, model_id)
    seeds_sorted = sorted(by_seed.keys())
    if len(seeds_sorted) < 8:
        return None
    first8_seeds = seeds_sorted[:8]

    claims_by_id_first8 = collections.defaultdict(list)
    claims_by_run_first8 = {}
    for s in first8_seeds:
        rows = by_seed[s]
        run_id = rows[0]["run_id"]
        claims_by_run_first8[run_id] = rows
        for r in rows:
            claims_by_id_first8[r["claim_id"]].append(r)

    target_claim, tier, claim_feats_all = select_target_claim(claims_by_id_first8, CLAIM_ORDER[task_id])

    task_feats = task_level_features(list(claims_by_run_first8.keys())[:8] if False else
                                      [by_seed[s][0]["run_id"] for s in first8_seeds], claims_by_run_first8)
    task_state = classify_state(task_feats, dominant_is_error_fn=lambda f: not f["dominant_signature_is_clean"])

    claim_state = None
    claim_feats = None
    if target_claim:
        claim_rows_first8 = claims_by_id_first8[target_claim]
        claim_feats = claim_level_features(claim_rows_first8)
        claim_state = classify_state(claim_feats, dominant_is_error_fn=lambda f: f["dominant_cls"] not in ("CORRECT", "OMITTED"))

    oracle_task_before = task_feats["oracle_at_k"]
    pass_task_before = task_feats["pass_at_k"]
    oracle_claim_before, pass_claim_before = (None, None)
    if target_claim:
        oracle_claim_before, pass_claim_before = claim_oracle_pass(claims_by_id_first8[target_claim])

    actions = {}

    # ---- SAME_MODEL: seeds 9-16 of the SAME (task, model)
    same_seeds = [s for s in seeds_sorted if s > first8_seeds[-1]][:8]
    if len(same_seeds) == 8:
        same_run_ids = {by_seed[s][0]["run_id"] for s in same_seeds}
        tokens, time_sec, n_calls = token_time_for_runs(run_meta, same_run_ids)
        combined_runs = {**claims_by_run_first8, **{by_seed[s][0]["run_id"]: by_seed[s] for s in same_seeds}}
        sigs = [task_defect_signature(rows) for rows in combined_runs.values()]
        oracle_task_after = 1.0 if "CLEAN" in sigs else 0.0
        pass_task_after = sigs.count("CLEAN") / len(sigs)
        entry = {
            "n_calls": n_calls, "tokens": tokens, "time_sec": round(time_sec, 3),
            "oracle_task_after": oracle_task_after, "delta_oracle_task": round(oracle_task_after - oracle_task_before, 4),
            "pass_task_after": round(pass_task_after, 4), "delta_pass_task": round(pass_task_after - pass_task_before, 4),
        }
        if target_claim:
            combined_claim_rows = claims_by_id_first8[target_claim] + [
                r for s in same_seeds for r in by_seed[s] if r["claim_id"] == target_claim
            ]
            oracle_claim_after, pass_claim_after = claim_oracle_pass(combined_claim_rows)
            entry["oracle_claim_after"] = oracle_claim_after
            entry["delta_oracle_claim"] = round(oracle_claim_after - oracle_claim_before, 4)
            entry["pass_claim_after"] = round(pass_claim_after, 4)
            entry["delta_pass_claim"] = round(pass_claim_after - pass_claim_before, 4)
            entry["utility_claim_per_token"] = round(entry["delta_oracle_claim"] / tokens, 6) if tokens else None
            entry["utility_claim_per_second"] = round(entry["delta_oracle_claim"] / time_sec, 5) if time_sec else None
        actions["SAME_MODEL"] = entry

    # ---- SWITCH_MODEL: another model's own first-8 pool, same task/prompt/temp
    other_model = OTHER_MODELS[model_id][0]  # fixed pick: first alphabetinsensitive entry in the pre-registered MODELS list
    other_by_seed = claims_by_seed_for_unit(all_claims, task_id, other_model)
    other_seeds_sorted = sorted(other_by_seed.keys())[:8]
    if len(other_seeds_sorted) == 8:
        other_run_ids = {other_by_seed[s][0]["run_id"] for s in other_seeds_sorted}
        tokens, time_sec, n_calls = token_time_for_runs(run_meta, other_run_ids)
        other_runs = {other_by_seed[s][0]["run_id"]: other_by_seed[s] for s in other_seeds_sorted}
        combined_runs = {**claims_by_run_first8, **other_runs}
        sigs = [task_defect_signature(rows) for rows in combined_runs.values()]
        oracle_task_after = 1.0 if "CLEAN" in sigs else 0.0
        pass_task_after = sigs.count("CLEAN") / len(sigs)
        entry = {
            "switched_to": other_model, "n_calls": n_calls, "tokens": tokens, "time_sec": round(time_sec, 3),
            "oracle_task_after": oracle_task_after, "delta_oracle_task": round(oracle_task_after - oracle_task_before, 4),
            "pass_task_after": round(pass_task_after, 4), "delta_pass_task": round(pass_task_after - pass_task_before, 4),
        }
        if target_claim:
            other_claim_rows = [r for s in other_seeds_sorted for r in other_by_seed[s] if r["claim_id"] == target_claim]
            combined_claim_rows = claims_by_id_first8[target_claim] + other_claim_rows
            oracle_claim_after, pass_claim_after = claim_oracle_pass(combined_claim_rows)
            entry["oracle_claim_after"] = oracle_claim_after
            entry["delta_oracle_claim"] = round(oracle_claim_after - oracle_claim_before, 4)
            entry["pass_claim_after"] = round(pass_claim_after, 4)
            entry["delta_pass_claim"] = round(pass_claim_after - pass_claim_before, 4)
            entry["utility_claim_per_token"] = round(entry["delta_oracle_claim"] / tokens, 6) if tokens else None
            entry["utility_claim_per_second"] = round(entry["delta_oracle_claim"] / time_sec, 5) if time_sec else None
        actions["SWITCH_MODEL"] = entry

    # ---- LOCAL_RETRY: 8 short generations targeting the ONE selected claim
    if target_claim:
        lr_path = os.path.join(EXP7_RUNS, f"local_retry.{model_id}.jsonl")
        lr_rows = [r for r in load_jsonl(lr_path) if r["task_id"] == task_id and r["claim_id"] == target_claim]
        if len(lr_rows) >= 1:
            tokens = sum(r.get("output_tokens") or 0 for r in lr_rows)
            time_sec = sum(r.get("generation_time_sec") or 0 for r in lr_rows)
            combined_cls = [r["classification"] for r in claims_by_id_first8[target_claim]] + [r["classification"] for r in lr_rows]
            oracle_claim_after = 1.0 if "CORRECT" in combined_cls else 0.0
            pass_claim_after = combined_cls.count("CORRECT") / len(combined_cls)
            entry = {
                "n_calls": len(lr_rows), "tokens": tokens, "time_sec": round(time_sec, 3),
                "oracle_claim_after": oracle_claim_after, "delta_oracle_claim": round(oracle_claim_after - oracle_claim_before, 4),
                "pass_claim_after": round(pass_claim_after, 4), "delta_pass_claim": round(pass_claim_after - pass_claim_before, 4),
                "utility_claim_per_token": round((oracle_claim_after - oracle_claim_before) / tokens, 6) if tokens else None,
                "utility_claim_per_second": round((oracle_claim_after - oracle_claim_before) / time_sec, 5) if time_sec else None,
                "local_retry_classification_counts": dict(collections.Counter(r["classification"] for r in lr_rows)),
            }
            actions["LOCAL_RETRY"] = entry
    else:
        actions["LOCAL_RETRY"] = {"status": "LOCAL_RETRY_NOT_APPLICABLE"}

    return {
        "unit": f"{task_id}/{model_id}", "task_id": task_id, "model_id": model_id,
        "target_claim": target_claim, "target_claim_tier": tier,
        "task_state": task_state, "task_features": task_feats,
        "claim_state": claim_state, "claim_features": claim_feats,
        "oracle_task_before": oracle_task_before, "pass_task_before": pass_task_before,
        "oracle_claim_before": oracle_claim_before, "pass_claim_before": pass_claim_before,
        "actions": actions,
    }


def main():
    all_claims = load_all_claims()
    run_meta = load_run_meta()
    units = []
    for task_id in TASKS:
        for model_id in MODELS:
            u = build_unit(task_id, model_id, all_claims, run_meta)
            if u:
                units.append(u)

    with open(os.path.join(METRICS_DIR, "outcomes7.json"), "w", encoding="utf-8") as f:
        json.dump(units, f, ensure_ascii=False, indent=2)

    print(f"n_units={len(units)}")
    print("task_state dist:", collections.Counter(u["task_state"] for u in units))
    print("claim_state dist:", collections.Counter(u["claim_state"] for u in units if u["claim_state"]))
    not_applicable = [u["unit"] for u in units if u["target_claim"] is None]
    print("LOCAL_RETRY_NOT_APPLICABLE units:", not_applicable)
    missing_same = [u["unit"] for u in units if "SAME_MODEL" not in u["actions"]]
    missing_switch = [u["unit"] for u in units if "SWITCH_MODEL" not in u["actions"]]
    if missing_same:
        print("MISSING SAME_MODEL arm:", missing_same)
    if missing_switch:
        print("MISSING SWITCH_MODEL arm:", missing_switch)


if __name__ == "__main__":
    main()
