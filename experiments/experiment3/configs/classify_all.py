"""
Post-processing pass: applies configs/claims.py to every RAW A1 recorded in
runs/a1_index.<model_id>.jsonl, writing one row per claim to
runs/claims_index.jsonl. Pure function of stored RAW text -- costs no GPU
time and is safe to re-run any time the claim definitions are refined.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from claims import classify_a1
from claims_def import CLAIMS_BY_TASK
from tasks_pipeline import TASKS_BY_ID
from storage import EXP_DIR, load_jsonl

RUNS_DIR = os.path.join(EXP_DIR, "runs")


def main():
    out_path = os.path.join(RUNS_DIR, "claims_index.jsonl")
    out_rows = []
    n_runs = 0
    n_parse_fail = 0

    for shard in sorted(glob.glob(os.path.join(RUNS_DIR, "a1_index.*.jsonl"))):
        for run in load_jsonl(shard):
            if run["generation_failed"]:
                continue
            n_runs += 1
            raw_path = os.path.join(RUNS_DIR, run["raw_path"], "result.txt")
            with open(raw_path, encoding="utf-8") as f:
                raw_text = f.read()
            task = TASKS_BY_ID[run["task_id"]]
            status, obj, claim_results = classify_a1(raw_text, task, CLAIMS_BY_TASK[task["task_id"]])
            if status != "OK":
                n_parse_fail += 1
            for c in claim_results:
                out_rows.append({
                    "run_id": run["run_id"], "task_id": run["task_id"], "model_id": run["model_id"],
                    "prompt_id": run["prompt_id"], "temperature": run["temperature"], "seed": run["seed"],
                    "contract_parse_status": status,
                    "claim_id": c["claim_id"], "claim_text": c["claim_text"],
                    "error_type": c["claim_type"],  # SIGNATURE / TYPE / SEMANTICS / EDGE_CASE
                    "classification": c["classification"], "evidence": c["evidence"],
                    "reference_status": "HAS_REFERENCE",  # every claim here is checked against a hand-authored reference fact (claims_def.py); never AMBIGUOUS-because-no-reference
                    "error_cluster_id": f"{run['task_id']}/{c['claim_id']}" if c["classification"] == "INCORRECT" else None,
                })

    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"runs_processed={n_runs} parse_failures={n_parse_fail} claim_rows={len(out_rows)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
